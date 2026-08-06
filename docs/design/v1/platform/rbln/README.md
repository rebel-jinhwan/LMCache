# RBLN Device Backend

Design notes for `lmcache/v1/platform/rbln/` -- the device-registry entry for
Rebellions NPUs.

## Scope

**Engine-driven multiprocess (MP) transfer only.**

`torch.rbln` is contributed by the `torch_rbln` package through a torch backend
entry point, so it is visible on a bare `import torch`. It provides everything
the engine-driven path needs:

| Capability | `torch.rbln` | Needed by |
|---|---|---|
| `is_available()` | yes | device detection |
| `device_count()` | yes | device detection |
| `current_device()` / `set_device()` | yes | worker device binding |
| `synchronize()` | yes | gather / scatter ordering |
| `Stream` / `Event` | **no** | LMCache-driven path |

The missing `Stream` / `Event` types are what bound the scope. The
LMCache-driven path publishes KV buffers across processes by exporting a device
IPC handle and ordering the handoff with a cross-process event; with no event
type there is no way to express that ordering. The spec therefore:

- overrides `is_handle_transfer_available()` to `False`, and
- leaves `ipc_wrapper_cls` and `event_ipc_backend` at their `None` defaults.

`mp_transfer_mode=lmcache_driven` then fails at its documented validation point
with a clear error, instead of crashing later on an attribute lookup.
`mp_transfer_mode=auto` already routes every non-CUDA device to the
engine-driven context, so the default path needs no special casing.

## Availability probing

`RblnDeviceSpec.is_available()` swallows exceptions, and this is load-bearing
rather than defensive boilerplate. Unlike `torch.cuda.is_available()`,
`torch.rbln.is_available()` **raises** when the runtime cannot register a
physical NPU:

```
RuntimeError: rbln_register_device_id failed for rbln:4 on physical NPU(s) [4]
(rc=1); the device(s) may be in use by another process or hold stale
allocations. Free the device(s) or adjust RBLN_DEVICES.
```

This is the normal state on a shared host where another process holds the
NPUs. Device detection runs during `lmcache.v1.platform` import on **every**
LMCache start, so an escaping exception would abort import for every co-tenant
process on the box -- including CPU-only ones. The spec reports "unavailable"
instead.

## Ops

`RblnDeviceOps` inherits the torch baseline unchanged, following
`lmcache/v1/platform/hpu/device_ops.py`. The baseline is safe here:

- `lmcache_memcpy_async` takes its tensor-mode branch for non-CUDA devices.
- `record_completion_on_stream` / `record_event_on_stream` degrade to immediate,
  unordered publication. Ordering is supplied by the engine-driven transfer
  context, which brackets gather and scatter with `torch_dev.synchronize()`.

## What stays out of tree, and why

RBLN's accelerated path is **one coupled change**, and all of it lives
downstream. Recording the reasoning here so it is not re-litigated.

### The native kernels

RBLN's kernels need the Rebellions runtime headers and libraries
(`RBLN_RUNTIME_INCLUDE` / `RBLN_RUNTIME_LIB_DIR`), which upstream CI cannot
obtain. An in-tree extension -- the route `XpuDeviceOps.ensure_native()` takes
for `lmcache.xpu_ops`, built by `setup_extensions/build_profiles/sycl.py` --
would therefore be unbuildable and untestable here. No `BUILD_WITH_RBLN` build
profile is needed either; note that `setup_extensions/build_profiles/musa.py`
is itself a stub (`detect()` returns `False`, `build()` returns `([], {})`),
so registering a profile that builds nothing buys nothing.

### Why even a loader shim stays out

`lmcache/v1/platform/musa/native_kv_transfer.py` shows an out-of-tree loader
pattern -- env gate, `import_module("musa_aiter")` returning `None` when
absent, and a `check_native_abi()` version handshake -- and MUSA wires it into
`MusaDeviceOps.multi_layer_block_kv_transfer`.

**That wiring is exactly what RBLN cannot do.** The RBLN kernel's chunk
contract differs from upstream's:

| | staging chunk layout |
|---|---|
| upstream `multi_layer_block_kv_transfer` | token-major `[2, L, T, H*D]` |
| RBLN native kernel | head-major `[2, L, H, T, D]` |

The two are not interchangeable -- reinterpreting one as the other silently
corrupts KV bytes. So the native kernel cannot back
`DeviceOps.multi_layer_block_kv_transfer`, whose callers all assume token-major
staging. It is correct only when the *same* transfer context writes the chunk
on store and reads it on retrieve, which is what makes the round trip
self-consistent (the chunk is an opaque byte range to the cache server).

That makes the native kernel and the head-major engine-driven context a single
inseparable change. A loader shim landed upstream on its own would be called by
nothing.

## The KV layout

RBLN's per-layer KV cache is 6-D `[2, NB, NH, 1, BS, HS]`. The closest
registered format, `NL_X_TWO_NB_NH_BS_HS` (6), is 5-D `[2, NB, NH, BS, HS]` --
identical bytes and strides, one axis short.

**No new `EngineKVFormat` member is registered.** `VLLMPagedMemRBLNConnectorV2`
squeezes axis 3 at registration and discovers the format from the resulting 5-D
views, which resolve to `NL_X_TWO_NB_NH_BS_HS`. The squeeze is a view onto the
same storage, and it raises `ValueError` if axis 3 is not 1 rather than
transferring the wrong slots.

This follows `VLLMPagedMemHPUConnectorV2`, which already builds a proxy tensor
list so its `List[Tuple[Tensor, Tensor]]` caches survive format discovery. The
connector is the device-scoped extension point; normalising there keeps the
detector, the format enum, the spec registry and `torch_ops` untouched.

### Why not a new format

A `NL_X_TWO_NB_NH_ONE_BS_HS` member was prototyped and dropped. It is
reachable without touching C++ -- but only if its spec imports `EngineKVFormat`
from `lmcache.v1.platform.ops_types` directly, because `lmcache.c_ops` is a PEP
562 shim forwarding to `resolve_device_ops(torch_device_type)`, and
`DeviceOps.bind_native()` replaces the enum wholesale with the compiled
module's type:

| device | `ensure_native()` | `lmc_ops.EngineKVFormat` |
|---|---|---|
| RBLN (`RblnDeviceOps`) | no-op | Python enum -- member visible |
| CUDA with the built extension | binds `lmcache.c_ops` | C++ enum -- member absent |

Since `kv_format/specs/registry.py` imports **every** spec module on every
device, a spec written the conventional way (`import lmcache.c_ops as lmc_ops`)
raises `AttributeError` at import on a CUDA build.

Even done correctly it costs an enum member, a spec file, and per-format
branches in `torch_ops` (`is_layer_list`, `_is_hnd_format`,
`_per_layer_paged_shape`, the HND transfer kernel, and the `multi_layer_kv_transfer`
guard) -- to describe a layout indistinguishable in memory from one already
registered. The connector-side squeeze achieves the same with none of it.

### What the connector must handle itself

HND puts the head axis *between* blocks and block tokens, so tokens are not
contiguous within a layer. The flat `view(num_blocks * block_size, hidden_dim)`
reshape the NHD connectors use would address the wrong slots, and a
`permute(...).reshape(...)` would copy the whole KV cache. The connector
instead resolves each slot into `(block, offset)` and uses advanced indexing,
touching only the request's tokens.

It also passes `layout_hints={"kv_layout": "HND"}` explicitly. The vLLM
detector defaults to NHD and only forces HND when `torch_device_type == "cpu"`
-- which is what RBLN was accidentally relying on before `RblnDeviceSpec`
existed, since detection used to fall back to the CPU stub.

## MP mode is not covered by this

`EngineDrivenTransferContext.register()` calls `compute_kv_layout(kv_caches)`
directly; no connector participates. The 6-D tensors are still rejected there,
so the multiprocess path needs its own normalisation -- either at the adapter
that supplies `kv_caches`, or a detector change. That is deliberately out of
scope here.
