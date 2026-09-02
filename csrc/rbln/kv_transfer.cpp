// SPDX-License-Identifier: Apache-2.0
#include "kv_transfer.h"

#include <algorithm>
#include <array>
#include <vector>

namespace lmcache::rbln {
namespace {

// Staging buffers: one per thread per kind. kind 0/1 are gather's landing
// buffer and swap output, kind 2/3 the same for scatter -- gather and
// scatter never share a slot, even though one's landing shape is the
// other's swap-output shape, so alternating directions (as a gather/scatter
// round trip does) can't fight over one slot. Reused across calls rather
// than freshly allocated: torch-rbln's compiled permute keys its device
// program on the buffer's address and rebinding a new one costs
// milliseconds, so a call that always got a fresh address would rebind
// every time instead of every geometry change. A model's geometry is fixed
// after load, so one buffer per kind is enough -- it is simply reallocated
// on the rare call whose shape doesn't match.
at::Tensor staging(at::IntArrayRef shape, at::ScalarType dtype,
                   const at::Device& device, int kind) {
  thread_local std::array<at::Tensor, 4> buffers;
  at::Tensor& buf = buffers[kind];
  if (!buf.defined() || buf.sizes() != shape || buf.scalar_type() != dtype ||
      buf.device() != device) {
    buf = at::empty(shape, at::TensorOptions().dtype(dtype).device(device));
  }
  return buf;
}

struct Geometry {
  int64_t layers, heads, block_size, head_size;
  at::ScalarType dtype;
  at::Device device;
};

Geometry geometry(const std::vector<at::Tensor>& layers) {
  TORCH_CHECK(!layers.empty(), "no paged layers");
  const auto& l0 = layers[0];
  TORCH_CHECK(l0.dim() == 5, "paged layers must be [2, NB, NH, BS, HS]");
  return {static_cast<int64_t>(layers.size()),
          l0.size(2),
          l0.size(3),
          l0.size(4),
          l0.scalar_type(),
          l0.device()};
}

// Pair every staging slot [half, layer] with its whole paged block.
void block_copy_lists(const std::vector<at::Tensor>& layers, int64_t block,
                      const at::Tensor& staged, std::vector<at::Tensor>& slots,
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
void chunk_copy_lists(const std::vector<at::Tensor>& chunks,
                      const at::Tensor& token_major, int64_t pos, int64_t bpc,
                      int64_t block_size, std::vector<at::Tensor>& regions,
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

void check_chunks(const std::vector<at::Tensor>& chunks, int64_t bpc,
                  int64_t block_size, int64_t n_blocks) {
  TORCH_CHECK(!chunks.empty(), "no chunks");
  TORCH_CHECK(chunks[0].size(2) == bpc * block_size, "chunk holds ",
              chunks[0].size(2), " tokens, not ", bpc, " blocks of ",
              block_size);
  TORCH_CHECK(static_cast<int64_t>(chunks.size()) * bpc >= n_blocks,
              chunks.size(), " chunks x ", bpc, " blocks cannot hold ",
              n_blocks, " blocks");
}

}  // namespace

void gather_blocks_to_chunks_hnd(const std::vector<at::Tensor>& layers,
                                 const std::vector<int64_t>& block_ids,
                                 const std::vector<at::Tensor>& chunks,
                                 int64_t bpc) {
  const int64_t n = static_cast<int64_t>(block_ids.size());
  if (n == 0) return;
  const Geometry g = geometry(layers);
  check_chunks(chunks, bpc, g.block_size, n);
  const std::vector<int64_t> in_shape{2, g.layers, g.heads, g.block_size,
                                      g.head_size};
  const std::vector<int64_t> out_shape{2, g.layers, g.block_size, g.heads,
                                       g.head_size};
  const int64_t rows = 2 * g.layers;
  at::Tensor in = staging(in_shape, g.dtype, g.device, /*kind=*/0);
  at::Tensor out = staging(out_shape, g.dtype, g.device, /*kind=*/1);
  for (int64_t u = 0; u < n; ++u) {
    std::vector<at::Tensor> slots, blocks;
    block_copy_lists(layers, block_ids[u], in, slots, blocks);
    at::_foreach_copy_(slots, blocks, false);

    // A permuted device copy: torch-rbln runs it as a compiled program.
    out.view({rows, g.block_size, g.heads, g.head_size})
        .copy_(in.view({rows, g.heads, g.block_size, g.head_size})
                   .permute({0, 2, 1, 3}));
    at::Tensor token_major =
        out.view({2, g.layers, g.block_size, g.heads * g.head_size});

    std::vector<at::Tensor> regions, pieces;
    chunk_copy_lists(chunks, token_major, u, bpc, g.block_size, regions,
                     pieces);
    // Blocking: a non-blocking dispatch can land on a different UMD context
    // than the next block's swap (the runtime may route async host copies
    // there once it can), and different contexts run concurrently -- nothing
    // here would then order that copy's read of `out` before the next
    // iteration's write to it.
    at::_foreach_copy_(regions, pieces, /*non_blocking=*/false);
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
  check_chunks(chunks, bpc, g.block_size, n);
  const std::vector<int64_t> in_shape{2, g.layers, g.block_size, g.heads,
                                      g.head_size};
  const std::vector<int64_t> out_shape{2, g.layers, g.heads, g.block_size,
                                       g.head_size};
  const int64_t rows = 2 * g.layers;
  at::Tensor in = staging(in_shape, g.dtype, g.device, /*kind=*/2);
  at::Tensor out = staging(out_shape, g.dtype, g.device, /*kind=*/3);
  for (int64_t u = start; u < n; ++u) {
    at::Tensor token_major =
        in.view({2, g.layers, g.block_size, g.heads * g.head_size});
    std::vector<at::Tensor> regions, pieces;
    chunk_copy_lists(chunks, token_major, u, bpc, g.block_size, regions,
                     pieces);
    // Blocking, for the same reason as the gather's D2H: the swap below
    // reads `in` right after, and a non-blocking dispatch gives no guarantee
    // this write to it has actually landed first if the two ran on
    // different UMD contexts.
    at::_foreach_copy_(pieces, regions, /*non_blocking=*/false);

    out.view({rows, g.heads, g.block_size, g.head_size})
        .copy_(in.view({rows, g.block_size, g.heads, g.head_size})
                   .permute({0, 2, 1, 3}));

    std::vector<at::Tensor> slots, blocks;
    block_copy_lists(layers, block_ids[u], out, slots, blocks);
    at::_foreach_copy_(blocks, slots, false);
  }
}

}  // namespace lmcache::rbln
