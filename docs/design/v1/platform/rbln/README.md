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

## The KV format

RBLN's per-layer KV cache is 6-D `[2, NB, NH, 1, BS, HS]`. The closest
registered format, `NL_X_TWO_NB_NH_BS_HS` (6), is 5-D `[2, NB, NH, BS, HS]` --
identical bytes, one axis short.

`NL_X_TWO_NB_NH_ONE_BS_HS` (15) is registered for it, **Python-side only**.
The engine's real rank stays visible everywhere it matters, and only the
transfer kernel collapses it:

| Surface | Rank |
|---|---|
| `EngineKVFormat` member, `describe_shape()` | 6-D |
| `NL_X_TWO_NB_NH_ONE_BS_HS_Spec` accessors | 6-D (`block_size` reads `shape[4]`) |
| `torch_ops._per_layer_paged_shape()` | 6-D `(2, nb, nh, 1, bs, hs)` |
| `torch_ops._transfer_per_layer_hnd()` | squeezed to 5-D on entry |

### Why the transfer path squeezes

`_squeeze_singleton_axis()` is about kernel reuse, not about the layout. The
existing HND kernel is written against a 4-D per-K/V tensor:

```python
scratch = torch.empty(n_valid, nh0, block_size, hs0, ...)   # 4-D
torch.index_select(k_t, 0, eff_idx, out=scratch)
```

With a 6-D layer, `k_t` is `[NB, NH, 1, BS, HS]` and `index_select` yields
`[n_valid, NH, 1, BS, HS]`, which does not fit `scratch` -- and the singleton
has to come off anyway before the `permute` into `[n_valid, BS, NH, HS]`. So
the axis is dropped either once at the function's entry or three or four times
inline. Once at entry is less code and less risk. The squeeze is a view over
identical bytes, returns a new list (callers' tensors are untouched), and
raises `ValueError` if axis 3 is not 1 rather than transferring the wrong
slots.

The equivalence is pinned by a round-trip test: the same data gathered through
format 15 and format 6 must produce byte-identical staging chunks, and must
restore identically.

### A Python-only member works, if the spec imports the enum directly

`EngineKVFormat` is dual-defined -- `lmcache/v1/platform/ops_types.py` and
`csrc/engine_kv_format.h` -- and `lmcache.c_ops` is a PEP 562 shim installed by
`lmcache/__init__.py` that forwards to `resolve_device_ops(torch_device_type)`,
the **detected** device's ops singleton. `DeviceOps.bind_native()` replaces
`EngineKVFormat` wholesale with the compiled module's type, and
`csrc/pybind.cpp` does export it.

So the enum a caller sees depends on the device:

| device | `ensure_native()` | `lmc_ops.EngineKVFormat` |
|---|---|---|
| RBLN (`RblnDeviceOps`) | no-op | Python enum -- new member visible |
| CUDA with the built extension | binds `lmcache.c_ops` | C++ enum -- new member absent |

The trap is that `kv_format/specs/registry.py::_discover_specs()` imports
**every** spec module unconditionally, on every device. A spec written the
conventional way -- `import lmcache.c_ops as lmc_ops`, then
`engine_kv_format = lmc_ops.EngineKVFormat.<NEW>` in the class body -- raises
`AttributeError` at import on a CUDA build, breaking CUDA users.

Importing the Python enum directly avoids this entirely:

```python
from lmcache.v1.platform.ops_types import EngineKVFormat


class NL_X_TWO_NB_NH_ONE_BS_HS_Spec(KVFormatSpec):
    engine_kv_format = EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS
```

Verified both ways: the registry then imports cleanly with and without a bound
native enum, and lookups for the existing formats still resolve through their
native-enum keys. Only `ops_types.py`, the spec file, and per-format shape
branches in `torch_ops._normalize_paged_layers` would be needed -- no C++.

### The alternative that was not taken

Because the two layouts are byte- and stride-identical, the RBLN integration
could instead `squeeze(3)` *before* handing tensors to LMCache and simply
declare format 6. That needs no upstream change at all.

It was rejected because it hides the engine's real tensor rank from every
surface that reports it. `describe_shape()` renders from the enum name, so a
6-D engine buffer would be described as 5-D in logs, spec geometry and error
messages, and the "axis 3 is always 1" assumption would live implicitly at an
integration boundary instead of explicitly in a registered format that
validates it.

Both designs rest on the same premise: axis 3 must always be 1. If a future
RBLN geometry makes it larger, format 15's squeeze guard raises -- whereas the
integration-side squeeze would silently mis-transfer. The existing RBLN kernels
share the premise, hardcoding index `0`.
