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

- **No transfer kernel handles format 15.** RBLN has no compiled
  block-transfer extension in tree, and the CUDA / SYCL kernels never see an
  RBLN cache, so their `default:` arm rejecting the format is correct rather
  than a gap. `csrc` therefore carries only the enum value, its
  `is_layer_list` classification, and the two pybind registrations.

The multiprocess path reaches the layout through `compute_kv_layout` / gather /
scatter, all of which resolve it via `normalize_kv_and_discover_format` and
never touch a connector -- which is why the format must be recognised by
detection rather than by a connector.

## Staging buffers: footprint and placement

The native transfer (`csrc/rbln/kv_transfer.cpp`) moves each block through
device staging: a landing buffer the paged block is gathered into, and a swap
buffer the head<->token permute writes. Both are a whole block
(`[2, L, H, BS, D]` / `[2, L, BS, H, D]`), and the pipelined path holds two
slots of each so a block's host copy can overlap the next block's swap.

### How many

`staging()` keys its buffers on `(shape, dtype, device, slot, role)` and keeps
them `thread_local`, so **every thread that transfers holds its own set**. Store
runs on the engine-driven commit pool (`DEFAULT_ENGINE_DRIVEN_COMMIT_WORKERS = 4`
threads), retrieve on the caller's thread. Gather and scatter ask for the same
two shapes (one's landing shape is the other's swap shape), and they cannot be
in flight together on one thread -- each call drains its copy stream before
returning -- so a thread that does both reuses the token-major buffer across
directions (the same role for both), and holds a second head-major one only
because the two directions want it on different chiplets (below). A serve thread
does one direction, so it holds:

| per thread | buffers | Qwen3-1.7B (117.44 MB / block) |
|---|---|---|
| two slots x (landing, swap) | 4 | 470 MB |
| x 4 store workers + 1 retrieve caller | 20 | 2.35 GB |

Measured (`rbln-stat`, one process): +322 MB after one thread's gather, +966 MB
after two, +1933 MB after four -- linear in the thread count, and a scatter on a
thread that already gathered adds nothing.

Tiling the staging over `k` layers would cut it by `L/k` but is not free:
halving it costs ~5% of scatter throughput, quartering it ~13%, and below that
the per-piece overhead dominates (measured 44.3 -> 42.2 -> 38.6 -> 32.4 GB/s at
k = 28, 14, 7, 4). It is not implemented; it becomes worth a knob when a block
approaches a gigabyte.

### Where

RBLN device DRAM is **one pool per chiplet** (32 GiB each on RBLN-CR13), and the
runtime pins an allocation to the chiplet it names -- it does not spill to
another when that one fills. Every torch allocation names chiplet 0, so with no
further action the whole staging set lands on chiplet 0's pool, next to that
chiplet's share of the model.

Which buffers can leave chiplet 0 without slowing the transfer was measured
(Qwen3-1.7B, 8 blocks, everything pinned to one chiplet):

| staging on chiplet | gather | scatter |
|---|---|---|
| 0 | 46.3 GB/s | 41.2 GB/s |
| 1 | 37.9 | 38.3 |
| 2 | 35.7 | 38.1 |
| 3 | 29.0 | 36.2 |

A host DMA endpoint on another chiplet is slower in both directions (gather's
D2H source, scatter's H2D landing), and so is the source of a D2D (scatter's
swap output feeding the paged block); the cost grows with the chiplet's distance
from 0. The one buffer that is only ever the *destination* of a D2D and then
read by the compiled permute -- gather's landing buffer -- moves at full rate
from any chiplet: gather held 46.6 GB/s with it on chiplets 1 and 2, while
floating either of scatter's buffers cost 8%.

`LMCACHE_RBLN_STAGING_CHIPLET` decides where staging goes, read once per process:

| value | placement |
|---|---|
| `spread` (default) | gather's landing buffers round-robin over chiplets 1.., with one counter for the process so the per-thread sets interleave; every other buffer on chiplet 0. No throughput cost; halves a store worker's chiplet-0 staging |
| `spread-all` | every buffer round-robin over every chiplet -- chiplet 0 keeps a quarter, at the cost above (measured 35.6 / 39.2 GB/s) |
| `main` | chiplet 0 for everything, the runtime's default |
| `<n>` | pin every staging buffer to chiplet `n` |

Placement is applied when a buffer is first allocated, through torch-rbln's
dispatcher op `torch_rbln::bind_device_memory_at` (the same thing
`torch.rbln.bind_device_memory(t, chiplet=n)` does), so this extension keeps
linking only ATen. A torch-rbln without the op leaves the buffers where the
runtime puts them and warns once. `lmcache.rbln_ops.staging_placement()`
reports the policy in effect.

Two things in the layers below make this hold:

- The runtime keeps a caller's placement when a compiled program later binds
  the tensor as an operand. The program's I/O configuration names the chiplet
  its compile assumed (0); without this the first permute would sync the buffer
  to the host, free it and reallocate it on chiplet 0 -- which is also why a
  rebind used to cost a host round trip.
- The DMA descriptors carry each area's chiplet, and I/O relocation is an
  address patch, so a block on chiplet 0 gathers into a landing buffer on
  chiplet 2 and the permute reads it there without any special casing.

`torch.rbln.memory_stats_per_chiplet()` shows the per-chiplet result.
