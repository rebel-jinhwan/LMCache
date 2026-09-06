// SPDX-License-Identifier: Apache-2.0
#include "kv_transfer.h"

#include <ATen/core/dispatch/Dispatcher.h>

#include <c10/core/Event.h>
#include <c10/core/StreamGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>

#include <algorithm>
#include <map>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace lmcache::rbln {
namespace {

// Staging slots: one device buffer per thread per slot -- per layer group, see
// staging(). A thread runs one transfer at a time, so its gather and its scatter
// share the HND staging: the same bytes, viewed head-major by one and token-major
// by the other. Sharing halves what a thread holds, and halves the buffer addresses
// torch-rbln keeps a compiled program for. Buffers are reused across calls rather
// than freshly allocated: torch-rbln keys compiled device programs (the HND
// permute) on the buffer's address, and rebinding one to another buffer costs it
// a wait for every transfer in flight; a model's geometry is fixed after load, so
// a slot is only reallocated on the rare call whose shape doesn't match.
enum Slot : int {
  kHndStaging,
  kMlaGather,
  kMlaScatter,
  kSlotCount
};

// Two staging sets per direction, taken in turn by block, so a block's host copy
// overlaps the next block's gather and swap. One set would serialise them: in
// place, the buffer a block's host copy is still reading is the buffer the next
// block wants to gather into. This is what the second buffer per direction is
// worth, and it is sharded like the first, so it costs block/shards a chiplet.
constexpr int64_t kPipelineSlots = 2;

// The host copies run on their own stream, and events order the two streams.
// The waits are per block, never "everything queued on the copy stream": the
// pipeline issues the next block's host copy before the current block's swap,
// so a blanket wait would serialise them. CPU tensors -- the extension's own
// tests run the kernel on them -- have no events and one stream, on which the
// copies simply run in order, which is what the events establish anyway; a
// fence recorded there is empty and waiting on it is a no-op.
using Fence = std::optional<c10::Event>;

Fence record(const c10::Stream& stream) {
  if (stream.device_type() == c10::DeviceType::CPU) return std::nullopt;
  Fence fence(std::in_place, stream.device_type());
  fence->record(stream);
  return fence;
}

void wait(Fence& fence, const c10::Stream& stream) {
  if (fence.has_value()) fence->block(stream);
}

// The head<->token swap is the compiled view copy run in place (its output
// aliasing its input), so it permutes the staging buffer in the buffer's own
// storage (torch-rbln verifies each geometry once against a host reference).
// One staging buffer per direction rather than a landing and a swap buffer --
// on Qwen3-1.7B, 117.44 MB per block, 235 MB per thread instead of 470. A
// second buffer buys nothing here: a block is swapped and copied out before
// the next one starts. It would buy something to a pipeline that overlaps a
// block's host copy with the next block's swap, and that is where it belongs.
//
// What this transfer borrows from torch-rbln, resolved once. Not optional and not
// probed for: the branch is built against a torch-rbln that has them, and a build
// that is not says so on the first transfer. They come through the dispatcher rather
// than as C++ calls because copy_strided_view is implemented in Python -- it drives
// the compile path -- and the other two follow it so this extension keeps linking
// only ATen.
struct TorchRblnOps {
  // (src, out, inplace) -> ran: the compiled view copy, in place when asked.
  c10::TypedOperatorHandle<bool(const at::Tensor&, at::Tensor&, bool)> swap =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torch_rbln::copy_strided_view", "")
          .typed<bool(const at::Tensor&, at::Tensor&, bool)>();
  // Chiplets per NPU of the tensor's device: how many shards the staging is cut into.
  c10::TypedOperatorHandle<int64_t(const at::Tensor&)> chiplet_count =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torch_rbln::chiplet_count", "")
          .typed<int64_t(const at::Tensor&)>();
  // Place a device allocation on a chiplet.
  c10::TypedOperatorHandle<void(at::Tensor&, int64_t)> bind_at =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torch_rbln::bind_device_memory_at", "")
          .typed<void(at::Tensor&, int64_t)>();
};

const TorchRblnOps& ops() {
  static const TorchRblnOps resolved;
  return resolved;
}

// How the staging is sharded, and where the shards live. RBLN device DRAM is
// one pool per chiplet and an
// allocation is pinned to the chiplet it names, without spilling; every torch
// allocation names chiplet 0 unless told otherwise, so a whole block of staging
// per direction per thread would sit on chiplet 0 next to that chiplet's share
// of the model. So the staging is sharded, one shard per chiplet.
//
// The shards are cut by layer. The KV cache itself is split over the chiplets
// by head, but the host wants every head of a token together, so head-shaped
// shards would need a token-granular merge; layer-shaped ones need none: a
// shard's (kv, layer) rows are exactly what the host chunk holds contiguously. Device-to-device and host DMA cost
// the same from any chiplet (measured); the swap program runs on chiplet 0 and
// reads the other chiplets over the die-to-die links, the one cost of the
// split. Placement goes through torch-rbln's dispatcher ops, so this extension
// keeps linking only ATen.
int64_t staging_shards(const at::Tensor& any_device_tensor) {
  static const int64_t shards =
      std::max<int64_t>(ops().chiplet_count.call(any_device_tensor), 1);
  return shards;
}

// The layers [lo, hi) a shard covers: contiguous runs of ceil(L / shards), so
// shard s holds the s-th run.
void shard_layers(int64_t layers, int64_t shards, int64_t shard, int64_t& lo,
                  int64_t& hi) {
  const int64_t per = (layers + shards - 1) / shards;
  lo = std::min(shard * per, layers);
  hi = std::min(lo + per, layers);
}

// One buffer per thread, per slot, per shard -- and this is where a shard
// becomes a placement: shard s is pinned to chiplet s. A fresh allocation
// would otherwise land on chiplet 0 like every other torch tensor, which is
// the thing the sharding exists to avoid.
at::Tensor staging(at::IntArrayRef shape, at::ScalarType dtype,
                   const at::Device& device, Slot slot, int64_t shard,
                   int64_t shards, int64_t pipeline_slot = 0) {
  using Key = std::tuple<int, int64_t, int64_t>;
  thread_local std::map<Key, at::Tensor> buffers;
  at::Tensor& buf = buffers[Key{slot, shard, pipeline_slot}];
  if (!buf.defined() || buf.sizes() != shape || buf.scalar_type() != dtype ||
      buf.device() != device) {
    buf = at::empty(shape, at::TensorOptions().dtype(dtype).device(device));
    if (shards > 1) {
      ops().bind_at.call(buf, /*chiplet=*/shard);
    }  // one chiplet, nothing to place
  }
  return buf;
}

// Every chunk holds `bpc` blocks of `block_size` tokens along `token_dim`,
// and there are enough chunks for `n_blocks`.
void check_chunks(const std::vector<at::Tensor>& chunks, int64_t token_dim,
                  int64_t bpc, int64_t block_size, int64_t n_blocks) {
  TORCH_CHECK(!chunks.empty(), "no chunks");
  TORCH_CHECK(chunks[0].size(token_dim) == bpc * block_size, "chunk holds ",
              chunks[0].size(token_dim), " tokens, not ", bpc, " blocks of ",
              block_size);
  TORCH_CHECK(static_cast<int64_t>(chunks.size()) * bpc >= n_blocks,
              chunks.size(), " chunks x ", bpc, " blocks cannot hold ",
              n_blocks, " blocks");
}

// Per-layer paged geometry. HND layers are [2, NB, NH, BS, HS]; MLA layers
// are a single latent plane [NB, BS, HS], read as kv == 1, heads == 1. MLA
// must be contiguous so that `layer[block]` is one contiguous [BS, HS] run
// and takes the direct device copy rather than torch-rbln's strided path.
struct Geometry {
  int64_t kv, layers, heads, block_size, head_size;
  at::ScalarType dtype;
  at::Device device;
};

Geometry geometry(const std::vector<at::Tensor>& layers) {
  TORCH_CHECK(!layers.empty(), "no paged layers");
  const auto& l0 = layers[0];
  const int64_t n = static_cast<int64_t>(layers.size());
  if (l0.dim() == 5) {
    return {2,          n, l0.size(2), l0.size(3), l0.size(4), l0.scalar_type(),
            l0.device()};
  }
  for (const auto& layer : layers) {
    TORCH_CHECK(layer.dim() == 3 && layer.is_contiguous(),
                "paged layers must be [2, NB, NH, BS, HS] or contiguous "
                "[NB, BS, HS]; got ",
                layer.sizes(), " with strides ", layer.strides());
  }
  return {1, n, 1, l0.size(1), l0.size(2), l0.scalar_type(), l0.device()};
}

// ── HND ──────────────────────────────────────────────────────────────

// The HND staging of a shard: `nl` layers of the block, shared by the gather
// (which stages it head-major, [2, nl, H, BS, D]) and the scatter (token-major,
// [2, nl, BS, H, D]). It is kept as [2*nl, H*BS, D]: both layouts are one split
// of its middle dim, which is a view chain torch-rbln's compiled copy can
// replay from the buffer it is bound to. A buffer kept in one direction's 5-D
// shape cannot be re-viewed for the other -- that reshape is not one the copy
// classifies -- and a flat one cannot be viewed for either.
at::Tensor hnd_staging(const Geometry& g, int64_t nl, int64_t shard,
                       int64_t shards, int64_t pslot) {
  return staging({2 * nl, g.heads * g.block_size, g.head_size}, g.dtype,
                 g.device, kHndStaging, shard, shards, pslot);
}

// Pair every staging slot [half, layer - lo] with its whole paged block, for
// layers [lo, hi).
void hnd_block_copy_lists(const std::vector<at::Tensor>& layers, int64_t block,
                          const at::Tensor& staged, int64_t lo, int64_t hi,
                          std::vector<at::Tensor>& slots,
                          std::vector<at::Tensor>& blocks) {
  for (int64_t l = lo; l < hi; ++l) {
    for (int64_t half = 0; half < 2; ++half) {
      slots.push_back(staged[half][l - lo]);
      blocks.push_back(layers[static_cast<size_t>(l)][half][block]);
    }
  }
}

// Pair the chunk region holding `pos` with its token-major staging piece, for
// layers [lo, hi). A block that is a whole chunk (chunk_size == block_size, the
// configuration this path optimizes) crosses the host boundary as one
// descriptor per (kv, layer run) -- one when the group is the whole block; a
// chunk that holds several blocks costs one per (kv, layer).
void hnd_chunk_copy_lists(const std::vector<at::Tensor>& chunks,
                          const at::Tensor& token_major, int64_t pos,
                          int64_t bpc, int64_t block_size, int64_t lo,
                          int64_t hi, std::vector<at::Tensor>& regions,
                          std::vector<at::Tensor>& pieces) {
  const at::Tensor& chunk = chunks[pos / bpc];
  if (bpc == 1) {
    if (lo == 0 && hi == chunk.size(1)) {
      regions.push_back(chunk);
      pieces.push_back(token_major);
      return;
    }
    for (int64_t half = 0; half < 2; ++half) {
      regions.push_back(chunk[half].slice(0, lo, hi));
      pieces.push_back(token_major[half]);
    }
    return;
  }
  const int64_t tok = (pos % bpc) * block_size;
  for (int64_t half = 0; half < 2; ++half) {
    for (int64_t l = lo; l < hi; ++l) {
      regions.push_back(chunk[half][l].slice(0, tok, tok + block_size));
      pieces.push_back(token_major[half][l - lo]);
    }
  }
}

// ── MLA ──────────────────────────────────────────────────────────────

// Pair blocks [lo, hi) of `block_ids` -- the chunk starting at block `first`
// -- with their token windows in `staged` ([L, bpc*BS, HS]), per layer. Both
// sides are contiguous [BS, HS].
void mla_block_copy_lists(const std::vector<at::Tensor>& layers,
                          const std::vector<int64_t>& block_ids, int64_t first,
                          int64_t lo, int64_t hi, int64_t block_size,
                          const at::Tensor& staged,
                          std::vector<at::Tensor>& slots,
                          std::vector<at::Tensor>& blocks) {
  for (size_t l = 0; l < layers.size(); ++l) {
    const at::Tensor layer_staged = staged[static_cast<int64_t>(l)];
    for (int64_t u = lo; u < hi; ++u) {
      slots.push_back(
          layer_staged.narrow(0, (u - first) * block_size, block_size));
      blocks.push_back(layers[l][block_ids[u]]);
    }
  }
}

// Pair the chunk's block window [lo, hi) (relative to the chunk) with the
// same window of `staged`. A whole chunk crosses the host boundary as one
// descriptor; a partial one (trailing short chunk, prefix skip) costs one per
// layer.
void mla_chunk_copy_lists(const at::Tensor& chunk, const at::Tensor& staged,
                          int64_t lo, int64_t hi, int64_t bpc,
                          int64_t block_size, std::vector<at::Tensor>& regions,
                          std::vector<at::Tensor>& pieces) {
  if (lo == 0 && hi == bpc) {
    regions.push_back(chunk);
    pieces.push_back(staged);
    return;
  }
  for (int64_t l = 0; l < chunk.size(0); ++l) {
    regions.push_back(chunk[l].slice(0, lo * block_size, hi * block_size));
    pieces.push_back(staged[l].slice(0, lo * block_size, hi * block_size));
  }
}

}  // namespace

int64_t staging_shard_count(const at::Tensor& any_device_tensor) {
  return staging_shards(any_device_tensor);
}

void gather_blocks_to_chunks_hnd(const std::vector<at::Tensor>& layers,
                                 const std::vector<int64_t>& block_ids,
                                 const std::vector<at::Tensor>& chunks,
                                 int64_t bpc) {
  const int64_t n = static_cast<int64_t>(block_ids.size());
  if (n == 0) return;
  const Geometry g = geometry(layers);
  TORCH_CHECK(g.kv == 2, "HND transfer needs [2, NB, NH, BS, HS] layers");
  check_chunks(chunks, /*token_dim=*/2, bpc, g.block_size, n);
  // Sharded by layer: each shard of the block gathers, swaps and reads out on its
  // own staging buffer, which lives on its own chiplet. The copies are issued per
  // block, not per shard: one _foreach_copy_ carries every shard's pieces, because
  // a copy call costs the host whatever it carries plus a fixed amount, and four
  // calls of a quarter each measured 0.7 ms per block worse than one.
  //
  // Nothing in the loop waits for the device. The device-to-device gather is
  // dispatched non-blocking, so it is ordered on the main stream like the swap
  // programs after it rather than waited for on the host: a synchronous gather
  // held the host for the copy's 1.9 ms per block, and everything else the host
  // does for a block (the swap dispatches, the host copy's descriptors, these
  // lists) then ran after it instead of under it. Per block the host now issues
  // and moves on, and the device's own time is what sets the pace.
  const int64_t shards = std::min(staging_shards(layers[0]), g.layers);
  std::vector<int64_t> lo(shards), hi(shards);
  for (int64_t shard = 0; shard < shards; ++shard) {
    shard_layers(g.layers, shards, shard, lo[shard], hi[shard]);
  }
  auto staged_at = [&](int64_t shard, int64_t pslot) {
    const int64_t nl = hi[shard] - lo[shard];
    return hnd_staging(g, nl, shard, shards, pslot)
        .view({2, nl, g.heads, g.block_size, g.head_size});
  };
  c10::impl::VirtualGuardImpl guard_impl(g.device.type());
  const c10::Stream main = guard_impl.getStream(g.device);
  const c10::Stream copy = guard_impl.getNewStream(g.device);
  std::vector<Fence> d2h_done(n);
  for (int64_t u = 0; u < n; ++u) {
    const int64_t pslot = u % kPipelineSlots;
    std::vector<at::Tensor> staged(shards);
    for (int64_t shard = 0; shard < shards; ++shard) staged[shard] = staged_at(shard, pslot);
    // The block that last had this pipeline slot is still reading it out.
    if (u >= kPipelineSlots) wait(d2h_done[u - kPipelineSlots], main);

    std::vector<at::Tensor> slots, blocks;
    for (int64_t shard = 0; shard < shards; ++shard) {
      hnd_block_copy_lists(layers, block_ids[u], staged[shard], lo[shard],
                           hi[shard], slots, blocks);
    }
    at::_foreach_copy_(slots, blocks, /*non_blocking=*/true);

    std::vector<at::Tensor> regions, pieces;
    for (int64_t shard = 0; shard < shards; ++shard) {
      const int64_t nl = hi[shard] - lo[shard], rows = 2 * nl;
      // In place: the program reads the buffer head-major and writes it back
      // token-major. The tensor's shape has to follow the bytes.
      at::Tensor token_major_view =
          staged[shard].view({rows, g.block_size, g.heads, g.head_size});
      TORCH_CHECK(ops().swap.call(staged[shard]
                                      .view({rows, g.heads, g.block_size,
                                             g.head_size})
                                      .permute({0, 2, 1, 3}),
                                  token_major_view, /*inplace=*/true),
                  "copy_strided_view declined the gather swap");
      hnd_chunk_copy_lists(chunks,
                           staged[shard].view({2, nl, g.block_size,
                                               g.heads * g.head_size}),
                           u, bpc, g.block_size, lo[shard], hi[shard], regions,
                           pieces);
    }
    Fence swapped = record(main);
    wait(swapped, copy);
    {
      c10::StreamGuard guard(copy);
      at::_foreach_copy_(regions, pieces, /*non_blocking=*/true);
    }
    d2h_done[u] = record(copy);
  }
  guard_impl.synchronizeStream(copy);
}

void scatter_chunks_to_blocks_hnd(const std::vector<at::Tensor>& layers,
                                  const std::vector<int64_t>& block_ids,
                                  const std::vector<at::Tensor>& chunks,
                                  int64_t bpc, int64_t skip_prefix_n_blocks) {
  const int64_t n = static_cast<int64_t>(block_ids.size());
  const int64_t start = std::min(std::max<int64_t>(skip_prefix_n_blocks, 0), n);
  if (start >= n) return;
  const Geometry g = geometry(layers);
  TORCH_CHECK(g.kv == 2, "HND transfer needs [2, NB, NH, BS, HS] layers");
  check_chunks(chunks, /*token_dim=*/2, bpc, g.block_size, n);
  // Sharded by layer, and issued per block, as in the gather.
  const int64_t shards = std::min(staging_shards(layers[0]), g.layers);
  const int64_t m = n - start;  // blocks in this transfer
  std::vector<int64_t> lo(shards), hi(shards);
  for (int64_t shard = 0; shard < shards; ++shard) {
    shard_layers(g.layers, shards, shard, lo[shard], hi[shard]);
  }
  auto staged_at = [&](int64_t shard, int64_t pslot) {
    const int64_t nl = hi[shard] - lo[shard];
    return hnd_staging(g, nl, shard, shards, pslot)
        .view({2, nl, g.block_size, g.heads, g.head_size});
  };
  c10::impl::VirtualGuardImpl guard_impl(g.device.type());
  const c10::Stream main = guard_impl.getStream(g.device);
  const c10::Stream copy = guard_impl.getNewStream(g.device);
  std::vector<Fence> h2d_done(m), scattered(m);

  // Fill block u's landing buffers from the host, on the copy stream: one call
  // for the block, carrying every shard's pieces.
  auto issue_copy = [&](int64_t u) {
    std::vector<at::Tensor> regions, pieces;
    for (int64_t shard = 0; shard < shards; ++shard) {
      const int64_t nl = hi[shard] - lo[shard];
      hnd_chunk_copy_lists(chunks,
                           staged_at(shard, u % kPipelineSlots)
                               .view({2, nl, g.block_size,
                                      g.heads * g.head_size}),
                           start + u, bpc, g.block_size, lo[shard], hi[shard],
                           regions, pieces);
    }
    {
      c10::StreamGuard guard(copy);
      at::_foreach_copy_(pieces, regions, /*non_blocking=*/true);
    }
    h2d_done[u] = record(copy);
  };

  if (m > 0) issue_copy(0);
  for (int64_t u = 0; u < m; ++u) {
    if (u + 1 < m) {
      // That H2D refills the buffers the block a pipeline slot back swapped and
      // scattered out of -- in place, the same buffers.
      if (u >= kPipelineSlots - 1) wait(scattered[u - (kPipelineSlots - 1)], copy);
      issue_copy(u + 1);
    }
    // This block's H2D, not the ones issued after it.
    wait(h2d_done[u], main);

    std::vector<at::Tensor> slots, blocks;
    for (int64_t shard = 0; shard < shards; ++shard) {
      const int64_t nl = hi[shard] - lo[shard], rows = 2 * nl;
      at::Tensor staged = staged_at(shard, u % kPipelineSlots);
      // In place, the other way round: token-major in, head-major out.
      at::Tensor head_major_view =
          staged.view({rows, g.heads, g.block_size, g.head_size});
      TORCH_CHECK(ops().swap.call(staged
                                      .view({rows, g.block_size, g.heads,
                                             g.head_size})
                                      .permute({0, 2, 1, 3}),
                                  head_major_view, /*inplace=*/true),
                  "copy_strided_view declined the scatter swap");
      hnd_block_copy_lists(layers, block_ids[start + u],
                           staged.view({2, nl, g.heads, g.block_size,
                                        g.head_size}),
                           lo[shard], hi[shard], slots, blocks);
    }
    // Non-blocking, like the gather's: the paged blocks are complete for work
    // on the main stream, and the next H2D into these buffers waits for it
    // through `scattered`. A host that reads the blocks reads behind it too --
    // the runtime orders a synchronous copy behind the transfers touching it.
    at::_foreach_copy_(blocks, slots, /*non_blocking=*/true);
    scattered[u] = record(main);
  }
  guard_impl.synchronizeStream(copy);
}

void gather_blocks_to_chunks_mla(const std::vector<at::Tensor>& layers,
                                 const std::vector<int64_t>& block_ids,
                                 const std::vector<at::Tensor>& chunks,
                                 int64_t bpc) {
  const int64_t n = static_cast<int64_t>(block_ids.size());
  if (n == 0) return;
  const Geometry g = geometry(layers);
  TORCH_CHECK(g.kv == 1, "MLA transfer needs contiguous [NB, BS, HS] layers");
  check_chunks(chunks, /*token_dim=*/1, bpc, g.block_size, n);
  at::Tensor staged = staging({g.layers, bpc * g.block_size, g.head_size},
                              g.dtype, g.device, kMlaGather,
                              /*shard=*/0, /*shards=*/1);
  for (int64_t c = 0; c * bpc < n; ++c) {
    const int64_t first = c * bpc;
    const int64_t held = std::min(n, first + bpc) - first;

    std::vector<at::Tensor> slots, blocks;
    mla_block_copy_lists(layers, block_ids, first, first, first + held,
                         g.block_size, staged, slots, blocks);
    at::_foreach_copy_(slots, blocks);

    std::vector<at::Tensor> regions, pieces;
    mla_chunk_copy_lists(chunks[c], staged, 0, held, bpc, g.block_size, regions,
                         pieces);
    at::_foreach_copy_(regions, pieces);
  }
}

void scatter_chunks_to_blocks_mla(const std::vector<at::Tensor>& layers,
                                  const std::vector<int64_t>& block_ids,
                                  const std::vector<at::Tensor>& chunks,
                                  int64_t bpc, int64_t skip_prefix_n_blocks) {
  const int64_t n = static_cast<int64_t>(block_ids.size());
  const int64_t start = std::min(std::max<int64_t>(skip_prefix_n_blocks, 0), n);
  if (start >= n) return;
  const Geometry g = geometry(layers);
  TORCH_CHECK(g.kv == 1, "MLA transfer needs contiguous [NB, BS, HS] layers");
  check_chunks(chunks, /*token_dim=*/1, bpc, g.block_size, n);
  at::Tensor staged = staging({g.layers, bpc * g.block_size, g.head_size},
                              g.dtype, g.device, kMlaScatter,
                              /*shard=*/0, /*shards=*/1);
  for (int64_t c = start / bpc; c * bpc < n; ++c) {
    const int64_t first = c * bpc;
    const int64_t lo = std::max(start, first) - first;
    const int64_t hi = std::min(n, first + bpc) - first;

    std::vector<at::Tensor> regions, pieces;
    mla_chunk_copy_lists(chunks[c], staged, lo, hi, bpc, g.block_size, regions,
                         pieces);
    at::_foreach_copy_(pieces, regions);

    std::vector<at::Tensor> slots, blocks;
    mla_block_copy_lists(layers, block_ids, first, first + lo, first + hi,
                         g.block_size, staged, slots, blocks);
    at::_foreach_copy_(blocks, slots);
  }
}

}  // namespace lmcache::rbln
