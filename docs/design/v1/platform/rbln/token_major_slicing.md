# Slicing a token-major transfer

How a transfer is cut into the units the copy/compute pipeline alternates over,
and what that reduces to once `chunk_size == block_size` is the only
configuration worth optimizing. The three legs of the transfer, the ordering
contract and the measured end-to-end numbers live in
[README.md](README.md#token-major-transfer); this document covers only the
cutting.

## Vocabulary

| term | meaning |
|---|---|
| block | one paged KV block: `[H, BS, D]` per (layer, kv half) |
| chunk | one LMCache staging chunk, token-major `[2, L, T, H*D]` |
| `bpc` | blocks per chunk = `lmcache_chunk_size / block_size` (`device_ops.py`) |
| `block_bytes` | `2 * L * H * BS * D * itemsize` -- one block across every layer and both halves |
| slice / unit | one fill of a staging buffer; the pipeline alternates two of them |
| slot | `unit_index % 2`; each slot owns a staging buffer and a swap program |

## Before: `(chunk, offset, run)` segments

`chunk_segments(start, n, bpc, run_cap)` walks block positions `[start, n)` and
emits a segment per step:

```
chunk = pos / bpc;  offset = pos % bpc;
run   = min(run_cap, bpc - offset, n - pos);
```

The length is the minimum of three constraints:

1. `run_cap` -- how many blocks the staging budget allows;
2. `bpc - offset` -- a segment never crosses a chunk boundary;
3. `n - pos` -- a segment never runs past the end of the transfer.

Constraint 2 is the load-bearing one. The device-to-host leg writes a segment
straight into the destination chunk, and the number of DMA descriptors it costs
depends on whether the segment covers a whole chunk:

| segment | descriptors per unit |
|---|---|
| covers a whole chunk, unit holds every layer | 1 per chunk |
| covers a whole chunk, unit holds half the layers | 2 per chunk (one per kv half) |
| covers part of a chunk | `2 * L` per chunk (one per kv half and layer) |

At ~17 us per descriptor on CR13 that difference is the transfer's cost, so
segments exist to keep the destination writes chunk-aligned.

`geometry()` derives the budget:

```
block_bytes      = 2 * L * H * BS * D * itemsize
per_slice_blocks = max(1, max_staging_bytes / block_bytes)   // default cap 128 MiB
run_cap          = min(bpc, per_slice_blocks)
per_slice        = max(1, per_slice_blocks / run_cap)        // segments per slice
```

`pipeline_units()` then groups segments into batches of `per_slice`.

**Observation.** For the geometries that run today `block_bytes` already meets
or exceeds the budget -- 117.44 MB for Qwen3-1.7B (L=28, H=8, BS=1024, D=128,
bf16) and 260.05 MB for the MiniMax-M2.5 KV geometry (L=62) against a 128 MiB
cap -- so `per_slice_blocks == 1`, `run_cap == 1` and `per_slice == 1`. Every
segment is exactly one block, and the `run` axis carries no information.

## Decision: optimize `chunk_size == block_size` only

The production serve sets both to 1024 (`tests/e2e/serve_env.rc`,
`--block-size 1024`), giving `bpc == 1`. Other values are reachable -- LMCache's
`chunk_size` defaults to 256 and `device_ops.py` accepts any multiple of the
engine block size -- but `bpc > 1` is the shape that costs `2 * L` descriptors
per unit anyway. It stays **correct, not fast**, and needs no separate path.

## Result: units without segments

```cpp
struct Unit { int64_t first, count; };  // blocks [first, first + count)

blocks_per_slice = max(1, max_staging_bytes / block_bytes);  // the only derived value
```

`units()` splits `[start, n)` into batches of `blocks_per_slice` blocks. A
single-batch transfer was also cut along the layer axis so its halves could
pipeline; measured on a 117 MB block that bought gather 0.71 ms and cost
scatter 0.86 ms -- a split chunk no longer crosses as one descriptor -- so it
is gone and every unit holds every layer.

Staging buffers lose the `run` axis:

| direction | staging | swap input | after the swap |
|---|---|---|---|
| gather | `[count, 2, L, H, BS, D]` | `[count*2*L, H, BS, D]` | `[count, 2, L, BS, H*D]` |
| scatter | `[count, 2, L, BS, H, D]` | `[count*2*L, BS, H, D]` | `[count, 2, L, H, BS, D]` |

Pairing a staging slot with its paged block becomes a single loop:

```cpp
for (j in [0, count))
  block = block_ids[first + j];
  for (l in [0, L))
    for (half in {0, 1})
      staged[j][half][l]  <->  layers[l][half][block]
```

and the destination mapping is a two-way branch on `pos = first + j`,
`chunk = pos / bpc`, `offset = pos % bpc`:

| condition | chunk region | staging piece |
|---|---|---|
| `bpc == 1` | `chunks[pos]` | `token_major[j]` (already `[2, L, T, H*D]`) |
| `bpc > 1` | `chunk[half][l].slice(0, offset*BS, (offset+1)*BS)` | `token_major[j][half][l]` |

The first row is the optimized case: one descriptor for the whole chunk, and no
`flatten` on the staging side because a `bpc == 1` chunk's token axis *is* the
block's `BS`.

### Removed

`Segment`, `chunk_segments()`, `Geometry::run_cap`, the three-way `min`, the
`seg x j` nested loop in the block pairing, the `flatten(2, 3)` that merged
`run_cap` into the chunk's token axis, and the single-batch layer split.

### Kept

`blocks_per_slice` batching, the `first` position tracking, the chunk-capacity
check, and the two-slot / event ordering.

## Ordering contract (unchanged)

Two slots alternate, and the waits are **per unit**, never "everything queued on
the copy stream" -- the pipeline issues the next unit's host copy before the
current unit's swap, so a blanket wait would serialise them.

- gather: before the swap of unit `u`, wait for the D2H of unit `u - 2` (that
  copy read the output buffer this swap is about to overwrite); after the swap,
  the copy stream waits for it, then issues the D2H.
- scatter: the H2D of unit `u + 1` is issued before the swap of unit `u`, after
  waiting for the swap of unit `u - 1` (which read that landing slot); the swap
  of unit `u` waits only for its own H2D.

## Edge cases

- `skip_prefix_n_blocks` moves `start`; with `bpc == 1` there is no partial
  leading chunk to reason about.
- Chunks past the end of `block_ids` are left untouched; the capacity check only
  refuses a chunk list too small to hold the transfer.
- `bpc` is passed in and also derivable from `chunks[0].size(2) / block_size`;
  `check_chunks` refuses a mismatch at entry.

## If fast `bpc > 1` is ever needed

Reintroduce `run` and `run_cap`: the destination mapping's second row collapses
into the first whenever a segment covers a whole chunk. Nothing else in
the pipeline depends on the segment shape.

## Current gaps

None: `csrc/rbln/kv_transfer.cpp` implements the units above as of 2026-08-30.
The swap is a plain `out.copy_(in.permute({0, 2, 1, 3}))` that torch-rbln runs
as a compiled program, so the extension needs no torch-rbln header or library.
