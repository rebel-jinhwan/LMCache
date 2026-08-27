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

- **The squeeze happens where bytes move on the torch fallback.**
  `RblnDeviceOps.multi_layer_block_kv_transfer` accepts only format 15 and
  applies `squeeze_singleton_axis` at entry, so `kv_ops.py` keeps indexing a
  5-D tensor. `kv_layout.py` therefore exports the strict squeeze plus the
  `is_rbln_kv_layout` predicate -- no tolerant pass-through variant, since the
  detected format has already established what the caller holds. The native
  kernel accepts the 6-D tensor (or the squeezed 5-D) directly.

- **Native transfer lives in `csrc/rbln`.** `lmcache.rbln_ops` exports
  `multi_layer_block_kv_transfer` with the DeviceOps argument list, and
  `RblnDeviceOps.ensure_native` binds it over the torch fallback the same way
  CUDA binds `cuda_ops`. CUDA / SYCL kernels still reject format 15 in their
  `default:` arm, because they never see an RBLN cache.

The multiprocess path reaches the layout through `compute_kv_layout` / gather /
scatter, all of which resolve it via `normalize_kv_and_discover_format` and
never touch a connector -- which is why the format must be recognised by
detection rather than by a connector.

## Native extension: `lmcache.rbln_ops`

`csrc/rbln/` builds `lmcache.rbln_ops`, which `RblnDeviceOps.ensure_native`
layers over the torch baseline through `DeviceOps.bind_native`. It exports

```
multi_layer_block_kv_transfer(paged_buffer_ptrs_tensor, lmcache_objects_ptrs,
                              block_ids, device, direction, shape_desc,
                              lmcache_chunk_size, engine_kv_format,
                              skip_prefix_n_blocks)
block_kv_transfer_mla(kv_caches, lmcache_chunks, block_ids,
                      direction, skip_prefix_n_blocks=0)
```

The first has the `DeviceOps` argument list, so after `ensure_native` the
multiprocess gather / scatter path (`gather_paged_kv_to_cpu` /
`scatter_cpu_to_paged_kv`) reaches the native kernel without knowing it exists.
It dispatches on `engine_kv_format`:

- **HND, format 15** (`[2, NB, NH, 1, BS, HS]` per layer, the layout vLLM-RBLN's
  attention backend allocates) <-> LMCache's canonical token-major chunk
  `[2, L, T, NH*HS]`. One paged block is a contiguous `NH*BS*HS` run on the
  device, so it moves in one DMA; the head <-> token permute into the wire
  layout is a host memcpy after (D2H) or before (H2D) that DMA. The chunk is
  byte-identical to what the torch fallback produces.
- **MLA, format 3** (`[NB, BS, HS]`) <-> `[L, T, HS]`. No head axis on either
  side, so one block of one layer is contiguous in both and there is no
  permute; `block_kv_transfer_mla` is also exported under its own name.

`direction` and `engine_kv_format` are the shared `lmcache_native` enums; the
extension registers none of its own, because pybind11 keys enum registrations
by C++ type and `lmcache_native` already owns them.

### How the kernel spends its time

Measured on RBLN-CR13 with `benchmarks/rbln/bench_kv_transfer_mp.py`
(Qwen3-Coder-30B-A3B, `block_size=1024`, 48 layers, one block = 100 MiB moved
as 96 slices of 1 MiB), inside a vLLM worker on the real DRAM KV:

| step | gather (D2H) | scatter (H2D) |
|---|---|---|
| one `rbln_memcpy_*_async` per slice + `rbln_device_synchronize` | 44.1 ms | 23.4 ms |
| one `rbln_memcpy_*_multi` per call | 32.0 ms | 15.8 ms |
| + 2 MiB-page (THP) staging | 10.1 ms | 10.2 ms |
| + permute on `at::get_num_threads()` threads | **8.0 ms** | **7.8 ms** |
| torch fallback, same run | 21.4 ms | 21.3 ms |

Three decisions follow from that table:

- **Batch the DMA.** `rbln_memcpy_{v2h,h2v}_multi` takes the whole descriptor
  list and is synchronous, so the kernel builds one `CopyBatch` per call and
  needs no async handles and no explicit device synchronize.
- **Stage through hugepages.** The runtime pins the host range of every copy
  on each command buffer; with 4 KiB pages that pin -- not the DMA -- was the
  dominant cost. The staging area (`host_staging`) is 2 MiB-aligned,
  `MADV_HUGEPAGE`d and pre-touched, which cuts the page count 512x. The hint
  is advisory; on plain pages the kernel still works, only slower. The
  staging area is `thread_local` and grows to the largest transfer seen.
- **Do not use `at::parallel_for` for the permute.** The mp worker calls this
  kernel off the main thread, where `at::parallel_for` degrades to a serial
  loop; `for_each_slice` fans the (block, K/V, layer) slices out on plain
  threads instead (0.9 ms per 100 MiB instead of 4.5 ms).

What is left is the DMA itself: ~7 ms per 100 MiB, i.e. six 16 MiB command
buffers at ~1 ms each, which is the runtime's device<->host rate inside a
compiled-model context on this hardware. Overlapping the permute with the DMA
was tried and removed: with the permute at 0.9 ms it recovers nothing until the
DMA gets faster.

### Build

`setup_extensions/build_profiles/rbln.py` registers the `rbln` build profile
(`BUILD_WITH_RBLN=1`, or auto-detected when the `torch_rbln` package is
installed). It compiles `csrc/rbln/` against the rebel runtime headers and
links `librbln.so`, both taken from the installed `rebel-compiler` wheel
(`rebel/include`, `tvm/librbln.so`) so header and library cannot drift apart;
`RBLN_RUNTIME_INCLUDE` / `RBLN_RUNTIME_LIB_DIR` override the lookup for a
runtime checkout. When the runtime is missing, an auto-detected build skips
the extension with a warning and an explicit `BUILD_WITH_RBLN=1` build fails.

The profile also selects `requirements/rbln_core.txt`, which adds `torch-rbln`
to `install_requires`. Two dependency facts shape that file:

- `torch-rbln` declares `torch==2.11.0+cpu`, a local version that only the
  PyTorch CPU wheel index serves, so `torch-rbln` has to be installed (or that
  index supplied) before `pip install lmcache` can resolve.
- `torch-rbln` imports `rebel` (the `rebel-compiler` distribution) at start-up
  but does **not** depend on it; `rebel-compiler` is a build-system requirement
  of torch-rbln only, and it is published solely on Rebellions' credentialed
  index (`pypi.rbln.ai`). It therefore cannot live in `install_requires` -- the
  same reason `rocm_core.txt` omits torch -- and is installed manually per the
  installation docs.
