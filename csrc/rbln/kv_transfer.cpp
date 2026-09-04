// SPDX-License-Identifier: Apache-2.0
#include "kv_transfer.h"

#include <algorithm>
#include <array>
#include <vector>

namespace lmcache::rbln {
namespace {

// Staging slots: one device buffer per thread per slot. Gather and scatter
// never share a slot, so a gather/scatter round trip on one thread cannot
// fight over a buffer. Buffers are reused across calls rather than freshly
// allocated: torch-rbln keys compiled device programs on the buffer's address
// and rebinding a new one costs milliseconds, and a model's geometry is fixed
// after load, so a slot is only reallocated on the rare call whose shape
// doesn't match. Each layout adds its own slots here.
enum Slot : int { kMlaGather, kMlaScatter, kSlotCount };

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

// Per-layer paged geometry. MLA layers are a single latent plane
// [NB, BS, HS], read as kv == 1, heads == 1; a K/V-split layout adds its own
// branch here. MLA must be contiguous so that `layer[block]` is one contiguous
// [BS, HS] run and takes the direct device copy rather than torch-rbln's
// strided path.
struct Geometry {
  int64_t kv, layers, heads, block_size, head_size;
  at::ScalarType dtype;
  at::Device device;
};

Geometry geometry(const std::vector<at::Tensor>& layers) {
  TORCH_CHECK(!layers.empty(), "no paged layers");
  const auto& l0 = layers[0];
  const int64_t n = static_cast<int64_t>(layers.size());
  for (const auto& layer : layers) {
    TORCH_CHECK(layer.dim() == 3 && layer.is_contiguous(),
                "paged layers must be contiguous [NB, BS, HS]; got ",
                layer.sizes(), " with strides ", layer.strides());
  }
  return {1, n, 1, l0.size(1), l0.size(2), l0.scalar_type(), l0.device()};
}

// Pair blocks [lo, hi) of `block_ids` -- the chunk starting at block `first`
// -- with their token windows in `staged` ([L, bpc*BS, HS]), per layer. Both
// sides are contiguous [BS, HS].
void block_copy_lists(const std::vector<at::Tensor>& layers,
                      const std::vector<int64_t>& block_ids, int64_t first,
                      int64_t lo, int64_t hi, int64_t block_size,
                      const at::Tensor& staged, std::vector<at::Tensor>& slots,
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
void chunk_copy_lists(const at::Tensor& chunk, const at::Tensor& staged,
                      int64_t lo, int64_t hi, int64_t bpc, int64_t block_size,
                      std::vector<at::Tensor>& regions,
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

void gather_blocks_to_chunks_mla(const std::vector<at::Tensor>& layers,
                                 const std::vector<int64_t>& block_ids,
                                 const std::vector<at::Tensor>& chunks,
                                 int64_t bpc) {
  const int64_t n = static_cast<int64_t>(block_ids.size());
  if (n == 0) return;
  const Geometry g = geometry(layers);
  check_chunks(chunks, /*token_dim=*/1, bpc, g.block_size, n);
  at::Tensor staged = staging({g.layers, bpc * g.block_size, g.head_size},
                              g.dtype, g.device, kMlaGather);
  for (int64_t c = 0; c * bpc < n; ++c) {
    const int64_t first = c * bpc;
    const int64_t held = std::min(n, first + bpc) - first;

    std::vector<at::Tensor> slots, blocks;
    block_copy_lists(layers, block_ids, first, first, first + held,
                     g.block_size, staged, slots, blocks);
    at::_foreach_copy_(slots, blocks);

    std::vector<at::Tensor> regions, pieces;
    chunk_copy_lists(chunks[c], staged, 0, held, bpc, g.block_size, regions,
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
  check_chunks(chunks, /*token_dim=*/1, bpc, g.block_size, n);
  at::Tensor staged = staging({g.layers, bpc * g.block_size, g.head_size},
                              g.dtype, g.device, kMlaScatter);
  for (int64_t c = start / bpc; c * bpc < n; ++c) {
    const int64_t first = c * bpc;
    const int64_t lo = std::max(start, first) - first;
    const int64_t hi = std::min(n, first + bpc) - first;

    std::vector<at::Tensor> regions, pieces;
    chunk_copy_lists(chunks[c], staged, lo, hi, bpc, g.block_size, regions,
                     pieces);
    at::_foreach_copy_(pieces, regions);

    std::vector<at::Tensor> slots, blocks;
    block_copy_lists(layers, block_ids, first, first + lo, first + hi,
                     g.block_size, staged, slots, blocks);
    at::_foreach_copy_(blocks, slots);
  }
}

}  // namespace lmcache::rbln
