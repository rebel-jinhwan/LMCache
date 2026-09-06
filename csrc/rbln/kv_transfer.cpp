// SPDX-License-Identifier: Apache-2.0
#include "kv_transfer.h"

#include <ATen/core/dispatch/Dispatcher.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace lmcache::rbln {
namespace {

// Staging slots: one device buffer per thread per slot. Gather and scatter
// never share a slot, so a gather/scatter round trip on one thread cannot
// fight over a buffer -- even though HND's gather landing shape is its
// scatter's swap-output shape. Buffers are reused across calls rather than
// freshly allocated: torch-rbln keys compiled device programs (the HND
// permute) on the buffer's address and rebinding a new one costs
// milliseconds, and a model's geometry is fixed after load, so a slot is only
// reallocated on the rare call whose shape doesn't match.
enum Slot : int {
  kHndGatherIn,
  kHndGatherOut,
  kHndScatterIn,
  kHndScatterOut,
  kMlaGather,
  kMlaScatter,
  kSlotCount
};

// The head<->token swap is a compiled device program. torch-rbln's
// copy_strided_view_inplace runs it with the output aliasing the input -- the
// program permutes the buffer in its own storage, and torch-rbln checks each
// geometry once against an out-of-place copy -- so a direction holds one
// staging buffer instead of a landing and a swap buffer (the kHnd*Out slots
// stay empty): on Qwen3-1.7B, 117.44 MB per block, 235 MB per thread rather
// than 470. The transfer keeps the two-buffer path for
// LMCACHE_RBLN_STAGING_INPLACE=0 and for a torch-rbln without the op (warned
// once).
bool inplace_swap() {
  static const bool enabled = [] {
    const char* value = std::getenv("LMCACHE_RBLN_STAGING_INPLACE");
    if (value != nullptr && std::strcmp(value, "0") == 0) return false;
    const bool has_op = c10::Dispatcher::singleton()
                            .findSchema({"torch_rbln::copy_strided_view_inplace", ""})
                            .has_value();
    if (!has_op) {
      TORCH_WARN_ONCE(
          "this torch-rbln has no torch_rbln::copy_strided_view_inplace; the KV "
          "transfer keeps a separate swap buffer per staging slot");
    }
    return has_op;
  }();
  return enabled;
}

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

at::Tensor staging(at::IntArrayRef shape, at::ScalarType dtype,
                   const at::Device& device, Slot slot) {
  thread_local std::array<at::Tensor, kSlotCount> buffers;
  at::Tensor& buf = buffers[slot];
  if (!buf.defined() || buf.sizes() != shape || buf.scalar_type() != dtype ||
      buf.device() != device) {
    buf = at::empty(shape, at::TensorOptions().dtype(dtype).device(device));
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

// Pair every staging slot [half, layer] with its whole paged block.
void hnd_block_copy_lists(const std::vector<at::Tensor>& layers, int64_t block,
                          const at::Tensor& staged,
                          std::vector<at::Tensor>& slots,
                          std::vector<at::Tensor>& blocks) {
  for (size_t l = 0; l < layers.size(); ++l) {
    for (int64_t half = 0; half < 2; ++half) {
      slots.push_back(staged[half][static_cast<int64_t>(l)]);
      blocks.push_back(layers[l][half][block]);
    }
  }
}

// Pair the chunk region holding `pos` with its token-major staging piece. A
// block that is a whole chunk (chunk_size == block_size, the configuration
// this path optimizes) crosses the host boundary as one descriptor; a chunk
// that holds several blocks costs one per (kv, layer).
void hnd_chunk_copy_lists(const std::vector<at::Tensor>& chunks,
                          const at::Tensor& token_major, int64_t pos,
                          int64_t bpc, int64_t block_size,
                          std::vector<at::Tensor>& regions,
                          std::vector<at::Tensor>& pieces) {
  const at::Tensor& chunk = chunks[pos / bpc];
  if (bpc == 1) {
    regions.push_back(chunk);
    pieces.push_back(token_major);
    return;
  }
  const int64_t lo = (pos % bpc) * block_size;
  for (int64_t half = 0; half < 2; ++half) {
    for (int64_t l = 0; l < chunk.size(1); ++l) {
      regions.push_back(chunk[half][l].slice(0, lo, lo + block_size));
      pieces.push_back(token_major[half][l]);
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

std::string staging_swap_mode() {
  return inplace_swap() ? "inplace" : "two-buffer";
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
  const std::vector<int64_t> in_shape{2, g.layers, g.heads, g.block_size,
                                      g.head_size};
  const std::vector<int64_t> out_shape{2, g.layers, g.block_size, g.heads,
                                       g.head_size};
  const int64_t rows = 2 * g.layers;
  const bool inplace = inplace_swap();
  at::Tensor in = staging(in_shape, g.dtype, g.device, kHndGatherIn);
  for (int64_t u = 0; u < n; ++u) {
    std::vector<at::Tensor> slots, blocks;
    hnd_block_copy_lists(layers, block_ids[u], in, slots, blocks);
    at::_foreach_copy_(slots, blocks);

    at::Tensor token_major;
    if (inplace) {
      swap_in_place(in, rows, g.heads, g.block_size, g.head_size);
      token_major = in.view({2, g.layers, g.block_size, g.heads * g.head_size});
    } else {
      at::Tensor out = staging(out_shape, g.dtype, g.device, kHndGatherOut);
      // A permuted device copy: torch-rbln runs it as a compiled program.
      out.view({rows, g.block_size, g.heads, g.head_size})
          .copy_(in.view({rows, g.heads, g.block_size, g.head_size})
                     .permute({0, 2, 1, 3}));
      token_major = out.view({2, g.layers, g.block_size, g.heads * g.head_size});
    }

    std::vector<at::Tensor> regions, pieces;
    hnd_chunk_copy_lists(chunks, token_major, u, bpc, g.block_size, regions,
                         pieces);
    at::_foreach_copy_(regions, pieces);
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
  const std::vector<int64_t> in_shape{2, g.layers, g.block_size, g.heads,
                                      g.head_size};
  const std::vector<int64_t> out_shape{2, g.layers, g.heads, g.block_size,
                                       g.head_size};
  const int64_t rows = 2 * g.layers;
  const bool inplace = inplace_swap();
  at::Tensor in = staging(in_shape, g.dtype, g.device, kHndScatterIn);
  for (int64_t u = start; u < n; ++u) {
    at::Tensor token_major =
        in.view({2, g.layers, g.block_size, g.heads * g.head_size});
    std::vector<at::Tensor> regions, pieces;
    hnd_chunk_copy_lists(chunks, token_major, u, bpc, g.block_size, regions,
                         pieces);
    at::_foreach_copy_(pieces, regions);

    at::Tensor head_major;
    if (inplace) {
      // The bytes are head-major now; the tensor's shape has to say so too.
      swap_in_place(in, rows, g.block_size, g.heads, g.head_size);
      head_major = in.view(out_shape);
    } else {
      head_major = staging(out_shape, g.dtype, g.device, kHndScatterOut);
      head_major.view({rows, g.heads, g.block_size, g.head_size})
          .copy_(in.view({rows, g.block_size, g.heads, g.head_size})
                     .permute({0, 2, 1, 3}));
    }
    std::vector<at::Tensor> slots, blocks;
    hnd_block_copy_lists(layers, block_ids[u], head_major, slots, blocks);
    at::_foreach_copy_(blocks, slots);
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
                              g.dtype, g.device, kMlaGather);
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
                              g.dtype, g.device, kMlaScatter);
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
