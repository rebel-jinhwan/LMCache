# RBLN Device Backend

Design notes for `lmcache/v1/platform/rbln/` -- the device-registry entry for
Rebellions NPUs.

## The 6-D KV cache

This is what makes RBLN unusual. vLLM-RBLN allocates each layer as

```
[2, num_blocks, num_kv_heads, 1, block_size, head_size]
```

-- HND with an **extra singleton axis between heads and block tokens**, which
the RBLN attention backend requires. Every other supported engine hands
LMCache a 5-D (or 4-D / 3-D) per-layer tensor.

Axis 3 is always 1, so the tensor is byte- and stride-identical to the already
registered `NL_X_TWO_NB_NH_BS_HS` (6) one axis short. **No new `EngineKVFormat`
is registered**: squeezing the axis is a free view that yields exactly format 6.
The rule lives in `kv_layout.py` (`is_rbln_kv_layout` / `squeeze_singleton_axis`).

It is applied in the **vLLM format detector**, not in the connector. That
placement is load-bearing: `compute_kv_layout`, `gather_paged_kv_to_cpu` and
`scatter_cpu_to_paged_kv` resolve layouts through
`normalize_kv_and_discover_format` and never touch a connector, so normalizing
in the connector alone left the multiprocess path raising
`ValueError: unsupported kv_caches structure` on the native tensors.

| path | reaches the layout through | normalized by |
|---|---|---|
| in-process | `VLLMPagedMemRBLNConnectorV2` | detector (plus its own 5-D views for slot indexing) |
| multiprocess | `compute_kv_layout` / gather / scatter | detector |

The detector branch is gated on `tensor_ndim == 6 and torch_device_type ==
"rbln"`, so no other accelerator's 6-D layout can be silently reinterpreted.

Two consequences for callers:

- **HND means tokens are not contiguous within a layer.** The flat
  `view(num_blocks * block_size, hidden_dim)` reshape the NHD connectors use
  addresses the wrong slots, and `permute(...).reshape(...)` copies the whole
  cache. `VLLMPagedMemRBLNConnectorV2` resolves slots into `(block, offset)`
  and uses advanced indexing instead.
- **The detector cannot infer HND from the shape.** `[2, NB, X, Y, HS]` is
  ambiguous between NH/BS and BS/NH, so the connector passes
  `layout_hints={"kv_layout": "HND"}` explicitly.

## Scope: engine-driven MP only

`torch.rbln` is contributed by the `torch_rbln` package through a torch backend
entry point, so it is visible on a bare `import torch`. It provides device
discovery, `set_device()` and `synchronize()` -- but **no `Stream` / `Event`
types**.

That is what bounds the scope. The LMCache-driven path publishes KV buffers
across processes by exporting a device IPC handle and ordering the handoff with
a cross-process event; with no event type there is no way to express that
ordering. `RblnDeviceSpec` therefore overrides `is_handle_transfer_available()`
to `False` and leaves `ipc_wrapper_cls` / `event_ipc_backend` at their `None`
defaults, so `mp_transfer_mode=lmcache_driven` fails at its documented
validation point instead of crashing later on an attribute lookup.
`mp_transfer_mode=auto` already routes every non-CUDA device to the
engine-driven context, so the default needs no special casing.

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

That is the normal state on a shared host where another process holds the NPUs.
Detection runs during `lmcache.v1.platform` import on **every** LMCache start,
so an escaping exception would abort import for every co-tenant process on the
box -- including CPU-only ones. The spec reports "unavailable" instead.

## Ops

`RblnDeviceOps` inherits the torch baseline for every op except
`multi_layer_block_kv_transfer`. The baseline is safe here:
`lmcache_memcpy_async` takes its tensor-mode branch for non-CUDA devices, and
the completion / event recorders degrade to immediate publication, with
ordering supplied by the engine-driven transfer context's
`torch_dev.synchronize()`.

Block transfer is overridden because RBLN stores heads before block tokens.
Upstream stages each chunk token-major (`[2, L, T, H*D]`), so the torch
baseline would issue an on-device head<->token permute per store and restore;
`kv_ops.py` fills the same buffer **head-major** (`[2, L, H, T, D]`) and never
permutes.

The head-major interpretation is scoped to one caller pair:
`gather_paged_kv_to_cpu` / `scatter_cpu_to_paged_kv` write and read the chunk
with the same code, and the cache server treats it as an opaque byte range, so
the round trip is self-consistent. The in-process connector addresses slots
directly and never calls the op; the LMCache-driven path is refused by
`is_handle_transfer_available()`. **A future caller that hands an RBLN chunk to
a token-major reader would break this invariant.**

Downstream, `lmcache-rbln` binds an RBLN-native C op over this same method when
its extension is built; it keeps the head-major contract byte for byte and only
changes how the bytes move. See that repo's
`docs/02_native_kv_transfer/README.md`.

## Running LMCache's own test suite on an RBLN host

`RblnDeviceSpec` changes what `torch_device_type` resolves to on a machine with
a free NPU, which changes behaviour for tests that assume a stream-capable
device. `tests/v1/gpu_connector/test_gds_context.py` monkeypatched
`torch_dev.current_stream`, which does not exist on `torch.rbln`; it now passes
`raising=False`, matching its own "no CUDA needed" intent. Expect similar
adjustments as more of the suite runs on RBLN CI.
