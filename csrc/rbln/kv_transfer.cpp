// SPDX-License-Identifier: Apache-2.0
#include "kv_transfer.h"

#include <ATen/core/dispatch/Dispatcher.h>

#include <algorithm>
#include <map>
#include <utility>
#include <vector>

namespace lmcache::rbln {
namespace {

// Staging slots: one device buffer per thread per slot -- per layer group, see
// staging(). Gather and scatter never share a slot, so a round trip on one
// thread cannot fight over a buffer. Buffers are reused across calls rather
// than freshly allocated: torch-rbln keys compiled device programs (the HND
// permute) on the buffer's address and rebinding a new one costs milliseconds,
// and a model's geometry is fixed after load, so a slot is only reallocated on
// the rare call whose shape doesn't match.
enum Slot : int {
  kHndGather,
  kHndScatter,
  kMlaGather,
  kMlaScatter,
  kSlotCount
};

// The head<->token swap is a compiled device program, and torch-rbln's
// copy_strided_view_inplace runs it with the output aliasing the input: the
// program permutes the buffer in its own storage (torch-rbln verifies each
// geometry once against an out-of-place copy), so a direction holds one
// staging buffer rather than a landing and a swap buffer -- on Qwen3-1.7B,
// 117.44 MB per block, 235 MB per thread instead of 470. A second buffer buys
// nothing here: a block is swapped and copied out before the next one starts.
// It would buy something to a pipeline that overlaps a block's host copy with
// the next block's swap, and that is where it belongs.
//
// buf holds rows x [a, b, d]; afterwards it holds rows x [b, a, d].
void swap_in_place(at::Tensor& buf, int64_t rows, int64_t a, int64_t b,
                   int64_t d) {
  static const auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("torch_rbln::copy_strided_view_inplace", "")
          .typed<bool(const at::Tensor&, at::Tensor&)>();
  at::Tensor src = buf.view({rows, a, b, d}).permute({0, 2, 1, 3});
  at::Tensor out = buf.view({rows, b, a, d});
  TORCH_CHECK(op.call(src, out),
              "copy_strided_view_inplace declined the swap view");
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
  static const int64_t shards = [&] {
    const auto count = c10::Dispatcher::singleton().findSchema(
        {"torch_rbln::chiplet_count", ""});
    TORCH_CHECK(count.has_value() &&
                    c10::Dispatcher::singleton()
                        .findSchema({"torch_rbln::bind_device_memory_at", ""})
                        .has_value(),
                "this torch-rbln cannot place a tensor on a chiplet "
                "(torch_rbln::chiplet_count, torch_rbln::bind_device_memory_at); "
                "the RBLN KV transfer needs both");
    return std::max<int64_t>(
        count->typed<int64_t(const at::Tensor&)>().call(any_device_tensor), 1);
  }();
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
                   int64_t shards) {
  using Key = std::pair<int, int64_t>;
  thread_local std::map<Key, at::Tensor> buffers;
  at::Tensor& buf = buffers[Key{slot, shard}];
  if (!buf.defined() || buf.sizes() != shape || buf.scalar_type() != dtype ||
      buf.device() != device) {
    buf = at::empty(shape, at::TensorOptions().dtype(dtype).device(device));
    if (shards > 1) {
      static const auto bind =
          c10::Dispatcher::singleton()
              .findSchemaOrThrow("torch_rbln::bind_device_memory_at", "")
              .typed<void(at::Tensor&, int64_t)>();
      bind.call(buf, /*chiplet=*/shard);
    }
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
  // Sharded by layer: each shard of the block is gathered, swapped and read
  // out on its own staging buffer, which lives on its own chiplet.
  const int64_t shards = std::min(staging_shards(layers[0]), g.layers);
  for (int64_t u = 0; u < n; ++u) {
    for (int64_t shard = 0; shard < shards; ++shard) {
      int64_t lo, hi;
      shard_layers(g.layers, shards, shard, lo, hi);
      const int64_t nl = hi - lo;
      const std::vector<int64_t> shape{2, nl, g.heads, g.block_size,
                                       g.head_size};
      at::Tensor staged =
          staging(shape, g.dtype, g.device, kHndGather, shard, shards);
      std::vector<at::Tensor> slots, blocks;
      hnd_block_copy_lists(layers, block_ids[u], staged, lo, hi, slots, blocks);
      at::_foreach_copy_(slots, blocks);

      // The bytes are token-major after this; the tensor's shape has to say so.
      swap_in_place(staged, 2 * nl, g.heads, g.block_size, g.head_size);
      at::Tensor token_major =
          staged.view({2, nl, g.block_size, g.heads * g.head_size});

      std::vector<at::Tensor> regions, pieces;
      hnd_chunk_copy_lists(chunks, token_major, u, bpc, g.block_size, lo, hi,
                           regions, pieces);
      at::_foreach_copy_(regions, pieces);
    }
  }
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
  // Sharded by layer, as in the gather.
  const int64_t shards = std::min(staging_shards(layers[0]), g.layers);
  for (int64_t u = start; u < n; ++u) {
    for (int64_t shard = 0; shard < shards; ++shard) {
      int64_t lo, hi;
      shard_layers(g.layers, shards, shard, lo, hi);
      const int64_t nl = hi - lo;
      const std::vector<int64_t> shape{2, nl, g.block_size, g.heads,
                                       g.head_size};
      at::Tensor staged =
          staging(shape, g.dtype, g.device, kHndScatter, shard, shards);
      at::Tensor token_major =
          staged.view({2, nl, g.block_size, g.heads * g.head_size});
      std::vector<at::Tensor> regions, pieces;
      hnd_chunk_copy_lists(chunks, token_major, u, bpc, g.block_size, lo, hi,
                           regions, pieces);
      at::_foreach_copy_(pieces, regions);

      // The bytes are head-major now; the tensor's shape has to say so too.
      swap_in_place(staged, 2 * nl, g.block_size, g.heads, g.head_size);
      at::Tensor head_major =
          staged.view({2, nl, g.heads, g.block_size, g.head_size});
      std::vector<at::Tensor> slots, blocks;
      hnd_block_copy_lists(layers, block_ids[u], head_major, lo, hi, slots,
                           blocks);
      at::_foreach_copy_(blocks, slots);
    }
  }
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
