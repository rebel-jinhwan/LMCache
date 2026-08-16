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

## Direct storage (RDS)

`lmcache/v1/storage_backend/rds_backend.py` writes KV chunks from RBLN device
memory to NVMe without a host hop, using `rebel.rds`. It is the RBLN analogue of
`GdsBackend`, and it is enabled per engine:

```
extra_config:
  enable_rbln_rds: true
max_local_disk_size: 64   # GB, per rank
```

It registers in-process, in `CreateStorageBackends`, alongside the other
backends — the engine constructs it directly, so a stored chunk never leaves the
worker's address space and no cache server is involved.

**Why it needs its own address allocator.** GDS writes a file per key and lets
the filesystem answer "where does this object go". `rebel.rds` exposes a flat,
fixed-size region with no filesystem, so `NvmeOffsetAllocator` answers it
explicitly: an `AddressManager` over the chunk's byte-offset space, first-fit
and coalescing on release, the same structure `GDSL1MemoryManager` uses over its
slab file.

Ownership is split deliberately. The chunk's *address space* belongs to the
backend and the staging *areas* belong to the allocator, because their lifetimes
differ: a staging area is recycled as soon as its write completes, while the
NVMe range must stay readable until the key is evicted.

**Two constraints come from the runtime, not from LMCache:**

- `rds` transfers a whole vmem *area* and requires the transfer size to equal
  the area's size, so `RDSMemoryAllocator` hands out one full area per object
  rather than sub-ranges of a slab.
- A stream `Chunk.read` DMAs into device vmem without updating the vmem's sync
  state — only the synchronous read path does that internally. The backend calls
  `mark_device_updated` after each batched read; without it the restored KV is
  silently stale rather than wrong-looking.

**Foreign objects are staged, not refused.** This is the one place RDS differs
from GDS. `GdsBackend` is an `AllocatorBackendInterface` too, but
`StorageManager._get_allocator_backend` never selects it: cuFile writes whatever
buffer it is handed, so GDS allocates only its own read destinations and
tolerates a `LocalCPUBackend` object on the write path. `rebel.rds` cannot —
it transfers a bound vmem *area* and requires the transfer size to equal that
area's size, so a `MemoryObj` RDS did not allocate is not a writable source.

Rather than claim the allocator slot for the whole engine, the write path
copies such an object into a staging area first (`_as_write_source`) and
releases the area once the stream drains. `StorageManager` is left alone, so
RDS registers like any other tier and the host pool in front of it keeps
working. The cost is one full-chunk copy per store — the object was allocated
in host memory, so the KV reaches NVMe as
`device -> host -> device vmem -> NVMe`.

The copy-free path would be a deployment with no host pool, where the storage
manager would hand the gather an RDS area directly. That is not reachable
today for a reason outside this backend: `_get_allocator_backend` ends in an
unconditional `storage_backends["LocalCPUBackend"]`, so `max_local_cpu_size: 0`
raises `KeyError` there before RDS is consulted. Making it fall back to the
first backend that owns an allocator is a shared-code fix, and belongs in its
own change rather than here.

Reads never stage: `batched_get_blocking` allocates its own destination areas,
exactly as GDS does.
