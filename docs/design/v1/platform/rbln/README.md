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
  accepts only format 15 and applies `squeeze_singleton_axis` at entry, so
  `kv_ops.py` keeps indexing a 5-D tensor. `kv_layout.py` therefore exports the
  strict squeeze plus the `is_rbln_kv_layout` predicate -- no tolerant
  pass-through variant, since the detected format has already established what
  the caller holds.

- **No shared transfer kernel handles format 15.** The CUDA / SYCL kernels
  never see an RBLN cache, so their `default:` arm rejecting the format is
  correct rather than a gap. The shared `csrc` therefore carries only the enum
  value, its `is_layer_list` classification, and the two pybind registrations;
  the one RBLN kernel lives in `csrc/rbln/` and addresses the 6-D tensor
  directly (below).

The multiprocess path reaches the layout through `compute_kv_layout` / gather /
scatter, all of which resolve it via `normalize_kv_and_discover_format` and
never touch a connector -- which is why the format must be recognised by
detection rather than by a connector.

## Native extension: `lmcache.rbln_ops`

`csrc/rbln/` builds `lmcache.rbln_ops`, which `RblnDeviceOps.ensure_native`
layers over the torch baseline through `DeviceOps.bind_native`. It exports a
single **native-only** op:

```
block_kv_transfer_head_major(kv_caches, lmcache_chunks, block_ids,
                             direction, skip_prefix_n_blocks=0)
```

- `kv_caches` are the per-layer 6-D tensors as vllm-rbln allocated them (no
  squeeze: the kernel computes strides from the 6-D shape).
- `lmcache_chunks` are contiguous host tensors holding a **head-major**
  `[2, L, H, T, D]` view, `T = blocks_per_chunk * block_size`.
- `direction` is the shared `lmcache_native.TransferDirection`; the extension
  registers no enum of its own, because pybind11 keys enum registrations by
  C++ type and `lmcache_native` already owns this one.

Both sides being head-major is the whole point: every `(K|V, layer, block,
head)` slab is contiguous on the device and in the chunk, so it moves as one
`rbln_memcpy_{v2h,h2v}_async` DMA (one per `(K|V, layer)` when a chunk holds
exactly one block) and the head<->token permute that `kv_ops.py` must run on
the host for the token-major wire layout never happens. RBLN has no stream or
event objects, so the kernel drains the whole burst with one
`rbln_device_synchronize` before returning; the caller sees a fully
materialised chunk (D2H) or cache (H2D).

What it is **not**: a replacement for `multi_layer_block_kv_transfer`. That
method's contract is the token-major wire layout shared with every other
device, which this kernel deliberately does not produce. It is bound under its
own name and has no torch fallback; callers feature-detect it with `hasattr`
and fall back to the torch path when the extension is absent. Wiring it into
the head-major staging format is a separate change.

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
