# RBLN Device Backend

Design notes for `lmcache/v1/platform/rbln/` -- the device-registry entry for
Rebellions NPUs. The engine is
[vllm-rbln](https://github.com/RBLN-SW/vllm-rbln).

## Scope

| path | supported | how it gets there |
|---|---|---|
| multiprocess, engine-driven | yes | `gather_paged_kv_to_cpu` / `scatter_cpu_to_paged_kv`, dispatching to `RblnDeviceOps.multi_layer_block_kv_transfer` |
| multiprocess, LMCache-driven | **no** | refused up front |
| in-process | yes | `VLLMPagedMemRBLNConnectorV2`, via `CreateGPUConnector` |

Multiprocess is self-contained: the MP path builds no GPU connector at all,
resolving layouts through `normalize_kv_and_discover_format` and moving KV
through `RblnDeviceOps`. The in-process path goes through the connector, but
resolves the layout through the same helper.

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
  correct rather than a gap. Format 15 is served by `csrc/rbln/` instead (see
  [The native extension](#the-native-extension) below); what the shared `csrc`
  carries for it is only the enum value, its `FORMAT_FACTS` row, and the two
  pybind registrations.

Both paths report the same format for the same cache, and both apply the
squeeze at the same depth -- where the paged tensors are indexed:

| path | reaches the layout through | squeezes in |
|---|---|---|
| multiprocess | `compute_kv_layout` / gather / scatter, no connector involved | `RblnDeviceOps.multi_layer_block_kv_transfer` |
| in-process | `VLLMPagedMemRBLNConnectorV2` | the connector's slot-indexing helper, which needs the 5-D views anyway |

That is why the format has to be recognised by detection rather than by a
connector: the MP path never builds one.

One consequence for the in-process path: **HND means tokens are not contiguous
within a layer** -- the head axis sits between blocks and block tokens. The flat
`view(num_blocks * block_size, hidden_dim)` reshape the NHD connectors use would
silently address the wrong slots, and `permute(...).reshape(...)` would copy the
whole cache. `VLLMPagedMemRBLNConnectorV2` resolves the slot mapping into
`(block, offset)` pairs and uses advanced indexing, touching only the tokens in
the request.

## The native extension

`lmcache/v1/platform/rbln/kv_ops.py` is the reference head-major transfer, and
the only one the unit tests exercise. `csrc/rbln/` is the same contract issued
as rebel runtime DMAs, built as `lmcache.rbln_ops` by
`setup_extensions/build_profiles/rbln.py`.

The reason it exists is the operand lists, not the copies. The torch kernels
have to materialise one tensor view per `(block, layer, kv)` triple to hand
`torch._foreach_copy_` its arguments; that count grows with blocks per chunk,
and above roughly a few hundred pairs building the views costs more than the
transfer they describe. The native kernel computes the same addresses from the
paged buffer's strides and submits them directly, so a multi-block chunk costs
the same per-block work as a single-block one. At one block per chunk -- the
shape serving actually uses -- the two are close, since that case coalesces to
one DMA per `(kv, layer)` on either path.

**How the two relate.** The same way CUDA and XPU relate to theirs:
`RblnDeviceOps.ensure_native()` imports the extension and hands it to
`DeviceOps.bind_native`, which `DeviceSpec.get_ops()` already calls once when it
builds the ops singleton. A missing extension is a logged soft-fail, not an
error -- it links the rebel runtime, so its absence is the ordinary case.

What binding adds is `head_major_block_kv_transfer`. There is no torch method of
that name, so `multi_layer_block_kv_transfer` tests for the attribute to decide
which implementation it holds, rather than carrying a flag of its own. There is
no env var and no adapter module: the extension is built only when `RblnProfile`
found a rebel runtime, so its presence *is* the opt-in.

`native_can_serve` then declines operands the kernel cannot address -- a paged
layer that is not a contiguous `rbln` tensor, a chunk that is not a contiguous
host tensor (the DMA's host end must be host memory), or a block list that does
not fill every chunk exactly, since the kernel derives a chunk index from the
flat block position and a ragged tail would land at another chunk's offsets.

**Build inputs.** Headers and library both come from the `rebel-compiler`
installation on the host: `<site-packages>/rebel/include` and
`<site-packages>/tvm/librbln.so`. Taking both from one installation is what
keeps them from drifting -- a vendored header copy that fell behind the loaded
`librbln.so` would be undefined behaviour rather than a build error.
`RBLN_RUNTIME_INCLUDE` and `RBLN_RUNTIME_LIB_DIR` override either for builds
against a runtime source tree. `RblnProfile.detect()` requires both to resolve,
so a host without a rebel runtime auto-detects some other profile and this
extension is simply absent.

## Dependencies

Nothing under `lmcache/` imports `torch_rbln`, `vllm`, or any Rebellions
package. `RblnDeviceSpec.is_available()` starts with `hasattr(torch, "rbln")`,
which is true only because torch-rbln registers the backend through a torch
entry point.

torch-rbln is therefore what makes the device detectable, and
`requirements/rbln_core.txt` declares it as a core requirement of an RBLN build:

```bash
BUILD_WITH_RBLN=1 pip install -e . \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

It pins `torch==2.11.0+cpu`, a local-version wheel that lives on
`download.pytorch.org` rather than PyPI, which is why that index is needed.
`setup.py` reads the file only when `RblnProfile` is the resolved profile, so a
CUDA or CPU wheel never carries the pin. torch itself is deliberately not listed
-- the pin arrives through torch-rbln, and repeating it would conflict with the
unpinned `torch` in `requirements/common.txt`. `requirements/rocm_core.txt`
omits torch for the same reason.

rebel-compiler, which supplies the headers and `librbln.so` the native
extension links, is declared nowhere: it ships from Rebellions' private index,
and its presence on the host is exactly what `RblnProfile.detect()` keys off.
