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
  accepts format 15 (and the MLA layout, below) and applies
  `squeeze_singleton_axis` at entry on the HND side, so `lmcache.rbln_ops`
  receives a 5-D tensor. `kv_layout.py` therefore exports the strict squeeze
  plus the `is_rbln_kv_layout` predicate -- no tolerant pass-through variant,
  since the detected format has already established what the caller holds.

- **No shared transfer kernel handles format 15.** Only `lmcache.rbln_ops`
  (below) moves it, and the CUDA / SYCL kernels never see an RBLN cache, so
  their `default:` arm rejecting the format is correct rather than a gap.
  `csrc` outside `csrc/rbln` therefore carries only the enum value, its
  `is_layer_list` classification, and the two pybind registrations for
  format 15.

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

What is RBLN-specific is the **DMA shape**, not the layout. The shared torch
MLA path issues one `index_select` / `index_copy_` per layer. On torch-rbln
each of those is a separate v2v submission (the index is read back to the
host to build the copy descriptors), the gather result / `.to(device)` window
is a fresh device allocation per chunk, and every one of those ops carries a
whole-layer CPU fallback behind it (`submit_or_fallback`) should the runtime
reject a copy. Measured on a real vLLM-RBLN DeepSeek-V3 KV cache the fallback
never fires and the shared path is correct, so the RBLN sequence exists for
cost, not correctness: with two or more blocks per chunk, batching a chunk's
`L * B` whole-block copies into one device staging buffer and crossing the
boundary once is 3.5-5x faster (61-layer DeepSeek-V3, 137-549 MiB chunks:
10-41 ms vs 37-213 ms per chunk); with one block per chunk it is on par.

## `lmcache.rbln_ops`

Both sequences live in one compiled extension, `csrc/rbln/` -> `lmcache.rbln_ops`,
built by `setup_extensions/build_profiles/rbln.py` (`BUILD_WITH_RBLN=1`, or
auto-detected from an installed `torch_rbln`). It is plain ATen -- nothing
links against torch-rbln, which supplies the RBLN implementations of the
copies at runtime -- so it also runs on CPU tensors, which is how its tests
exercise the kernels without hardware.

- **Native only.** There is no torch fallback in `RblnDeviceOps`: without the
  extension the transfer raises `RuntimeError` naming `BUILD_WITH_RBLN`. One
  sequence per layout to keep correct and to measure.
- **HND: the swap moves to the device.** The earlier torch sequence crossed
  PCIe once per (block, layer, kv) and did the head<->token swap on the host.
  The extension instead runs one device sequence per block: a D2D gather of
  the whole block into device staging, the swap as a permuted `copy_` that
  torch-rbln runs as a compiled program, and a D2H of the chunk's bytes --
  one descriptor per whole chunk when `chunk_size == block_size`, one per
  (kv, layer) otherwise. Scatter is the mirror. Block after block, and
  nothing in the loop waits for the device: the D2D is dispatched
  non-blocking on the main stream, the swap programs chain behind it, and
  the host copy runs on a second stream fenced by events per block (see
  [Staging buffers](#staging-buffers)).
- **Staging slots, per thread, reused.** `staging()` in `kv_transfer.cpp`
  keeps one device buffer per `(thread, slot, shard, pipeline slot)`; a
  thread's gather and scatter share the HND staging (a thread runs one
  transfer at a time; the same bytes are viewed head-major by one and
  token-major by the other), and separate threads (the multiprocess server's
  pool) never share one. Buffers are reused across calls rather than freshly
  allocated: torch-rbln keys compiled device programs on the buffer's
  address, and a model's geometry is fixed after load, so a slot is only
  reallocated on the rare call whose shape doesn't match. HND owns two
  pipeline slots of `shards` buffers per thread, MLA two buffers.
- **MLA: one chunk at a time.** Gather: `_foreach_copy_` of the chunk's whole
  `[BS, HS]` blocks into their token windows of the `[L, bpc*BS, HS]` staging
  buffer (D2D, direct `memcpy_v2v`, no index tensor), then the chunk's bytes
  cross the host boundary -- one descriptor for a whole chunk, one per layer
  for a partial window (a trailing short chunk, or the chunk a prefix skip
  starts inside). Scatter is the mirror.
- **MLA geometry is pinned in the extension.** `geometry()` requires every
  layer to be a contiguous 3-D tensor: a permuted view would send each block
  copy down torch-rbln's strided path, which has a CPU fallback behind it.
  `RblnDeviceOps` only checks the format (`is_mla()` admits every MLA
  variant; only `NL_X_NB_BS_HS` is accepted) and hands the tensors through
  unsqueezed -- there is nothing to squeeze.

Both layouts require the engine's KV caches to be real device tensors
(vLLM-RBLN: `VLLM_RBLN_USE_DEVICE_TENSOR=1`). With the default compile-mode
allocation the per-layer tensors are `meta`, and any transfer -- this
backend's or the shared path's -- dies at the first host copy with "Cannot
copy out of meta tensor".

## Staging buffers

The native transfer (`csrc/rbln/kv_transfer.cpp`) moves each block through
device staging: the block is gathered head-major (`[2, L, H, BS, D]`) and the
host wants it token-major (`[2, L, BS, H, D]`), or the reverse on retrieve.

### Pipelined over two slots

The host copies run on their own stream, so a block's copy overlaps the next
block's gather and swap; events order the two streams per block -- not by
draining the copy stream, which would serialise exactly what the pipeline is
for. That needs **two staging sets**, taken in turn by block: in place, the
buffer a block's host copy is still reading is the buffer the next block wants
to gather into, so with one set the swap would have to wait. The second set is
sharded like the first, so it costs `block/shards` on a chiplet rather than a
whole block, and both directions of a thread share the two sets: a shard is
kept as `[2*nl, H*BS, D]`, from which the gather's head-major and the
scatter's token-major layout are each one reshape -- a view chain the compiled
copy replays from the buffer it is bound to, where a buffer kept in one
direction's shape could not be re-viewed for the other.

### Nothing waits on the host

The device-to-device gather of a block (224 pieces of 512 KB on Qwen3-1.7B,
1.9 ms) used to be a synchronous call: the host sat in it until the copy
landed, and only then dispatched the swaps, built the host copy's descriptors
and issued it -- about 1 ms of host work per block that ran *after* the copy
instead of *under* it. Measured with every call timed inside the pipeline
(sharded, pinned, 8 blocks): 2.95 ms per block, of which the D2D was 1.95 and
the host's own work the rest; the same pipeline with the host copy removed ran
at 2.45 ms per block, so the device was idle for the difference.

The D2D is now dispatched non-blocking (`_foreach_copy_(..., non_blocking=
true)`, which torch-rbln turns into a stream-ordered batch): the swap programs
chain behind it on the main stream, the host copy's event chains behind the
swaps, and the host issues a block and moves on to the next. The pace is then
the device's own -- the D2D plus the swaps -- with the host copy underneath.

### One buffer, swapped in place

The swap between the two layouts is a compiled device program, and torch-rbln's
`torch_rbln::copy_strided_view(src, out, inplace=True)` runs it with the output
aliasing the input, so a direction needs **one buffer, not a landing and a swap
buffer**:
on Qwen3-1.7B (117.44 MB per block) 235 MB per thread instead of 470. The
second buffer that is worth having is the pipeline's second slot above, not a
swap output.

`copy_` cannot be used for this -- ATen refuses partially overlapping pairs --
so the transfer calls the op directly. Correctness rests on the compiled
schedule finishing its reads of a region before writing it, which is the
compiler's tiling rather than anything this code controls; torch-rbln therefore
checks each geometry once against a host reference (measured bit-exact on ten
geometries: heads 2--16, tokens 16--256, head size 64/128, rows 2--72) and
raises if the check fails.

The three ops the transfer borrows from torch-rbln -- `copy_strided_view` with
its `inplace` flag, `bind_device_memory_at`, `chiplet_count` -- are resolved once
in `TorchRblnOps` and taken as given rather than probed for: this branch is built
against a torch-rbln that has them, and a build that is not says so on the first
transfer. They go through the dispatcher rather than as C++ calls because
`copy_strided_view` is implemented in Python (it drives the compile path), and
the other two follow it so this extension keeps linking only ATen.

The buffers are `thread_local`, so every thread that transfers holds its own
set: store runs on the engine-driven commit pool (4 workers by default),
retrieve on the caller's thread.

### Sharded over the chiplets

RBLN device DRAM is one pool per chiplet (32 GiB each on RBLN-CR13) and the
runtime pins an allocation to the chiplet it names, without spilling. Every
torch allocation names chiplet 0, so a block of staging per direction per
thread would all sit on chiplet 0's pool next to that chiplet's share of the
model. The KV cache itself is split over the chiplets **by head** (heads
`[2k, 2k+1]` of every layer and block live on chiplet `k` on this geometry).

The staging is therefore **sharded by layer**, one shard per chiplet: shard `s`
holds a contiguous run of layers, stages on chiplet `s`
(`torch.rbln.bind_device_memory(t, chiplet=s)` through the dispatcher op), is
swapped there in place, and its `(kv, layer)` rows are what the host chunk holds
contiguously -- so the host copies are the same 2 MiB rows as before.
Head-shaped shards would have matched the paged blocks, but the host wants every
head of a token together, so they would have needed a token-granular merge
(512 B pieces) on the device or a change of the host layout. Measured,
device-to-device and host DMA cost the same from any chiplet; the swap program
runs on chiplet 0 and reads the other chiplets over the die-to-die links, which
is the one cost of the sharding. `rbln_ops.staging_shard_count(t)` reports how
many shards are in use.

### What it costs a chiplet

A device runs out of memory on its heaviest chiplet, so that is the number to
watch: the high-water mark of one chiplet's allocated bytes over a transfer.
Measured on Qwen3-1.7B (117.44 MB per block) with the allocator's per-chiplet
peak counters, as the delta over the pre-transfer level:

| staging | 1 thread, gather | 1 thread, round trip | 4 threads, gather | 4 threads, round trip |
|---|---|---|---|---|
| two buffers, one chiplet (before) | 352.9 MB | 705.7 MB | 1411.5 MB | 2822.9 MB |
| in place, one chiplet | 352.9 | 588.3 | 941.7 | 1883.4 |
| **in place, sharded over 4 chiplets** | **146.9** | **293.8** | **352.6** | **470.0** |
| in place, sharded, pipelined over two slots shared by both directions | 117.9 | 236.3 | 117.9 | 236.3 |

Six times less on the busiest chiplet for the four-thread round trip, and the
sharding flattens the thread scaling: the staging is
`threads x shards x block/shards` either way, but only `1/shards` of it lands on
any one chiplet. With the pipeline's two slots shared by a thread's gather and
scatter, a round trip holds what a gather holds, and the four-thread round trip
is `4 x 2 x block/shards` = 235 MB a chiplet -- all of it staging: the compiled
swap programs allocate no output of their own (the runtime allocates an
executor output only for a run that does not supply one), and the twelve-fold
reduction against the two-buffer, one-chiplet starting point (2822.9 MB) holds
with the pipeline.

Correctness under threads is its own gate: the bench's `--verify` drives one
thread, so it cannot see two threads swapping each other's staging. The
multi-threaded check (4 threads, distinct per-block patterns, gather compared on
the host and scatter compared back on the device) is what caught the eager
out-tensor binding being process-wide rather than per thread, in the runtime.
