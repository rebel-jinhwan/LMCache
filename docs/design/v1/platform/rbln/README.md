# RBLN Device Backend

Design notes for `lmcache/v1/platform/rbln/` -- the device-registry entry for
Rebellions NPUs. The engine is
[vllm-rbln](https://github.com/RBLN-SW/vllm-rbln).

## Scope

| path | supported | how it gets there |
|---|---|---|
| multiprocess, engine-driven | yes | `gather_paged_kv_to_cpu` / `scatter_cpu_to_paged_kv`, dispatching to `RblnDeviceOps.multi_layer_block_kv_transfer` |
| multiprocess, LMCache-driven | **no** | refused up front |
| in-process | not yet | `CreateGPUConnector` still raises for `rbln`; a follow-up adds the connector |

Multiprocess is the mode that matters first, and it is self-contained: the MP
path builds no GPU connector at all, resolving layouts through
`normalize_kv_and_discover_format` and moving KV through `RblnDeviceOps`.

`torch.rbln` comes from
[torch-rbln](https://github.com/RBLN-SW/torch-rbln) through a torch backend
entry point, so it is visible on a bare `import torch` -- LMCache never imports
it explicitly. It provides device discovery, `set_device()` and `synchronize()`,
but no `Stream` / `Event` types. The LMCache-driven path publishes KV buffers
across processes by exporting a device IPC handle and ordering the handoff with
a cross-process event, which cannot be expressed without an event type.
`RblnDeviceSpec` therefore overrides `is_handle_transfer_available()` to `False`
and leaves `ipc_wrapper_cls` / `event_ipc_backend` at their `None` defaults, so
`mp_transfer_mode=lmcache_driven` fails at its documented validation point
instead of crashing later on an attribute lookup. `mp_transfer_mode=auto`
already routes every non-CUDA device to the engine-driven context, so the
default needs no special casing.

## The 6-D KV cache

This is what makes RBLN unusual. vllm-rbln allocates each layer as

```
[2, num_blocks, num_kv_heads, 1, block_size, head_size]
```

-- HND with an **extra singleton axis between heads and block tokens**, which
the RBLN attention backend requires. Every other supported engine hands
LMCache a 5-D (or 4-D / 3-D) per-layer tensor.

Axis 3 is always 1, so the tensor is byte- and stride-identical to a 5-D
`[2, NB, NH, BS, HS]` layout. It is nonetheless **registered as its own
`EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS` (15)**, so detection reports what
vLLM-RBLN actually allocated instead of reshaping a device's KV cache to fit
another format's rank.

The alternative -- squeezing the axis during discovery so it classifies as the
existing `NL_X_TWO_NB_NH_BS_HS` (6) -- was rejected. Reshaping inside detection
requires a device hook, which makes discovery depend on process-global device
state rather than being a pure function of `(kv_caches, layout_hints)`, and it
forces a second device-specific rule (the HND override below) for a format that
is HND by definition. Registering the format removes both.

Consequences of the format being first-class:

- **Detection is device-independent.** The 6-D branch in `detectors/vllm.py`
  keys off the shape signature alone (`ndim == 6`, `shape[0] == 2`,
  `shape[3] == 1`), so the same input classifies identically on any host, and
  the detector needs no `rbln` entry in its device table.

  ```python
  if (list_depth == 1 and tensor_ndim == 6
          and first_tensor.shape[0] == 2 and first_tensor.shape[3] == 1):
      return lmc_ops.EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS, kv_caches
  ```

- **The reported layout is never consulted.** vllm-rbln does not set vLLM's KV
  cache layout, so `get_kv_cache_layout()` returns the NHD default, which under
  the shared 5-D format would have classified the cache as
  `NL_X_TWO_NB_BS_NH_HS` -- the wrong axis order for every transfer. Format 15
  is HND by definition, so the hint plays no part. `detectors/vllm.py` still
  forces HND for `cpu` (vLLM's CPU attention backend misreports its layout),
  but that table no longer has an `rbln` entry.

- **The squeeze happens where bytes move.** `RblnDeviceOps.multi_layer_block_kv_transfer`
  accepts only format 15 and the MLA layout (below) and applies
  `squeeze_singleton_axis` at entry on the HND side, so `kv_ops.py` keeps
  indexing a 5-D tensor. `kv_layout.py` therefore exports the strict squeeze
  plus the `is_rbln_kv_layout` predicate -- no tolerant pass-through variant,
  since the detected format has already established what the caller holds.

- **No transfer kernel handles format 15.** RBLN has no compiled
  block-transfer extension in tree, and the CUDA / SYCL kernels never see an
  RBLN cache, so their `default:` arm rejecting the format is correct rather
  than a gap. `csrc` therefore carries only the enum value, its
  `is_layer_list` classification, and the two pybind registrations.

The multiprocess path reaches the layout through `compute_kv_layout` / gather /
scatter, all of which resolve it via `normalize_kv_and_discover_format` and
never touch a connector -- which is why the format must be recognised by
detection rather than by a connector.

## The MLA layout

vllm-rbln's MLA attention backend
(`vllm_rbln/v1/attention/backends/mla`) allocates each layer as

```
[num_blocks, block_size, head_size]
```

-- a single latent plane with no K/V split and no head axis. Unlike the 6-D
HND cache this is not an RBLN-only shape: the vLLM detector already classifies
it as the existing `EngineKVFormat.NL_X_NB_BS_HS`, so no new format is
registered and detection needs no RBLN knowledge. Chunks stay in the canonical
single-plane wire layout `[L, T, HS]`, so -- as with HND -- a chunk stored
from an RBLN MLA cache is byte-compatible with every other device.

What is RBLN-specific is the **op sequence**, not the layout. The shared torch
MLA path stages its gather through `torch.empty(device=...)` +
`index_select(out=...)`, but on RBLN a raw `torch.empty` on the device is a
lazy SHM tensor: ops against it run on the CPU-fallback path, never on the
chip. `kv_ops.py` therefore builds the gather *functionally* --
`index_select` per layer, `stack` across layers, both device-native v2v
kernels -- and crosses the device boundary with one `copy_` per chunk. The
scatter mirrors it: one `.to(device)` DMA for the chunk window, then a
device-native `index_copy_` per layer.

How the two layouts relate inside the backend:

- **No transpose to hoist.** The HND sequence exists to move the head<->token
  transpose to the host; MLA has no head axis, so its sequence exists purely
  for the op ordering above. It uses no host staging buffers at all.
- **Nothing to squeeze, so the rank is pinned instead.**
  `validate_mla_layers` in `kv_layout.py` mirrors `squeeze_singleton_axis`'s
  strictness -- the detected format has already established what the caller
  holds, so any non-3-D tensor is a layout drift and fails loudly at the
  transfer boundary -- but returns the tensors unchanged.
- **Per-format addressing, shared bookkeeping.**
  `RblnDeviceOps.multi_layer_block_kv_transfer` dispatches with
  `lmcache_native.is_mla(engine_kv_format)`; the chunk/block bookkeeping
  (blocks-per-chunk split, global prefix skip translated to a per-chunk
  offset, direction handling) is shared between the layouts. The `kv_ops`
  names carry the split: `gather_blocks_to_chunk_hnd` / `_mla` and
  `scatter_chunk_to_blocks_hnd` / `_mla`.

Both layouts require the engine's KV caches to be real device tensors
(vLLM-RBLN: `VLLM_RBLN_USE_DEVICE_TENSOR=1`). With the default compile-mode
allocation the per-layer tensors are `meta`, and any transfer -- this
backend's or the shared path's -- dies at the first host copy with "Cannot
copy out of meta tensor".

## Token-major transfer

The ceiling is descriptors, not bandwidth: a host-DMA descriptor costs ~17 us on
CR13, a single contiguous D2H runs at 58 GB/s, and a block striped over 4
chiplets is 4+ descriptors per (block, layer, kv). The transfer lives in the
native extension `lmcache.rbln_ops` (`csrc/rbln/kv_transfer.cpp`, built by the
`rbln` profile when torch-rbln is installed):

1. D2D gather of whole blocks into `[S, 2, L, R, H, BS, D]` device staging
   (command-stream copies, ~240 GB/s);
2. the head<->token swap as a plain `copy_` from a permuted view -- torch-rbln
   runs a large permuted device copy as a compiled program (~0.4 ms per 117 MB
   against ~90 ms for the strided walk);
3. D2H of each chunk's bytes straight into the (pinned) chunk on a dedicated
   stream, one descriptor per whole chunk.

Two slots alternate; per-unit events order swap -> copy and copy -> slot reuse
(a blanket "wait for the copy stream" would make the swap wait for the *next*
unit's copy and serialise the pipeline); scatter is the mirror with the next
H2D prefetched. The overlap
comes from rebel-compiler's transfer context (a second UMD context for async
host copies -- the device runs one context's jobs in order) and its range-aware
sync-copy waits; see `rebel/src/runtime/vmemory/README.md`. `kv_ops.py` keeps
only the MLA sequence.

The extension uses ATen only -- it does not link torch-rbln, which supplies the
RBLN implementations of these copies at runtime. Two runtime facts shaped the
pipeline and cost a day to find, both about *binding*: a compiled program binds
its operands once, and rebinding a 117 MB buffer costs ~2 ms, so a program is
kept per source buffer (torch-rbln does this for the `copy_` fast path) rather
than one program fed alternating buffers.

In-worker e2e, verified (pinned chunks):

| gather / scatter | upstream | pipelined (native) |
|---|---|---|
| Qwen3-1.7B, 1 block (117 MB) | 21.7 / 20.0 ms | **4.51 / 3.14 ms** |
| Qwen3-1.7B, 8 blocks (940 MB) | 173 / 160 ms | **20.1 (46.8 GB/s) / 25.1 ms** |
| MiniMax-M2.5 geometry, 1 block (260 MB) | 37.5 / 36.0 ms | **7.02 / 8.57 ms** |
| MiniMax-M2.5 geometry, 8 blocks (2.08 GB) | 300 / 273 ms | **42.8 (48.6 GB/s) / 57.3 ms** |

What each mechanism is worth, A/B on the 940 MB gather / scatter: the transfer
context 20.1 -> 25.5 ms on gather and nothing on scatter; the copy stream and
its second staging slot 25.1 -> 42.5 ms on scatter (a correctly fenced single
slot is not slower, but the slot is what makes the overlap safe). Splitting a
single-batch transfer along the layer axis was removed: +0.71 ms gather,
-0.86 ms scatter.

`bench_kv_transfer_mp.py --trace-legs` synchronizes after every leg and hides
the overlap; use it for leg costs only.

## One chunk layout

HND chunks are always token-major (`[2, L, T, H*D]`), byte-compatible with
every other device. Earlier iterations carried a head-major
(`LMCACHE_RBLN_SAVE_HEAD_MAJOR`) and a chiplet-major (`LMCACHE_RBLN_CHUNK_LAYOUT`)
layout to avoid the head<->token transpose; once the transpose ran on the
device and the copies pipelined, token-major was as fast or faster
(chiplet-major: 43 ms per 940 MB against 20 ms now), so both layouts, the host
staging buffers and the per-chunk `_hnd` kernels were removed. MLA chunks have
no head axis and take the shared per-chunk path.

## Host memory: register through torch-rbln

`PinMemoryBackend` is written around `cudaHostRegister`: hand an existing host
address to the runtime and it records the region as DMA-able. RBLN has the
same thing since UMD 3.5 (`rblnRegisterHostMemory`), surfaced by rebel-compiler
as `rbln_host_register` and by torch-rbln >= 0.4.1 as
`torch.rbln.register_host_memory(address, nbytes)` / `unregister_host_memory`.
`RblnPinMemoryBackend` is a thin adapter over that pair.

What registration buys: the pages are pinned once, and every later host<->device
copy whose page-aligned operand lies inside the range is recorded against the
buffer's device VA, so the kernel reuses that pin instead of pinning the pages on
each command buffer. Measured on CR13, a D2H into a registered buffer runs at
36.6 GB/s against 13.8 GB/s into pageable memory. The mp SHM pool -- created by
the LMCache server, mapped by the worker -- is exactly the "memory another
process owns" case `cudaHostRegister` exists for, which is why the call sites
are the ones upstream already has: `EngineDrivenContextShm` on the worker's
mapping right after it opens the pool, and `torch_ops.alloc_shm_pinned_ptr` on
the server's when it creates it.

Bounds worth stating:

- **Unaligned operands do not take the device-VA path.** The pin itself does
  not need alignment, but the DMA engine wants 4 KiB-aligned operands; the
  runtime keeps an unaligned copy on its bounce path. The pool is page-aligned
  by construction (`mmap` base, 4 KiB `AddressManager` slots).
- **Unregister only after the copies are done**, as with `cudaHostUnregister`.
  The runtime drains the device's pending transfers before unpinning; a copy
  issued from another thread afterwards is the caller's to order.
- **No pretend pin.** The backend calls `torch.rbln.register_host_memory`
  directly and assumes it exists (torch-rbln >= 0.4.1); a refused registration
  (overlap, a UMD without `rblnRegisterHostMemory`) reports False and the caller
  keeps the per-copy pin path. A weaker fallback (populating the mapping with
  `madvise`) was tried and dropped: it hides the real answer and buys little
  once the copy is pinned per command buffer anyway.