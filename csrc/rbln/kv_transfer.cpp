// SPDX-License-Identifier: Apache-2.0
#include "kv_transfer.h"

#include <c10/core/Event.h>
#include <c10/core/StreamGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>

#include <algorithm>
#include <map>
#include <optional>
#include <tuple>

namespace lmcache::rbln {
namespace {

// Staging buffers: per thread, per (shape, dtype, device, slot, kind). Two
// slots alternate per direction; kind 0 is the landing buffer, kind 1 is the
// buffer the swap writes into.
at::Tensor staging(at::IntArrayRef shape, at::ScalarType dtype,
                   const at::Device& device, int slot, int kind) {
  using Key =
      std::tuple<std::vector<int64_t>, at::ScalarType, std::string, int, int>;
  thread_local std::map<Key, at::Tensor> buffers;
  Key key{shape.vec(), dtype, device.str(), slot, kind};
  auto it = buffers.find(key);
  if (it == buffers.end()) {
    at::Tensor buf =
        at::empty(shape, at::TensorOptions().dtype(dtype).device(device));
    it = buffers.emplace(key, buf).first;
  }
  return it->second;
}

// The host copies run on their own stream so they overlap the next block's
// gather and swap, and events order the two streams.
//
// The waits are per block, never "everything queued on the copy stream": the
// pipeline issues the next block's host copy before the current block's swap,
// so a blanket wait would serialise them.
using Fence = std::optional<c10::Event>;

Fence record(const c10::Stream& stream) {
  Fence fence(std::in_place, stream.device_type());
  fence->record(stream);
  return fence;
}

void wait(Fence& fence, const c10::Stream& stream) {
  if (fence.has_value()) fence->block(stream);
}

void copy_pairs(const c10::Stream& stream, const std::vector<at::Tensor>& dsts,
                const std::vector<at::Tensor>& srcs) {
  c10::StreamGuard guard(stream);
  at::_foreach_copy_(dsts, srcs, /*non_blocking=*/true);
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

void gather_blocks_to_chunks_token_major(const std::vector<at::Tensor>& layers,
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
  c10::impl::VirtualGuardImpl guard_impl(g.device.type());
  const c10::Stream main = guard_impl.getStream(g.device);
  const c10::Stream copy = guard_impl.getNewStream(g.device);
  std::vector<Fence> d2h_done(n);
  for (int64_t u = 0; u < n; ++u) {
    const int slot = static_cast<int>(u % 2);
    at::Tensor in = staging(in_shape, g.dtype, g.device, slot, /*kind=*/0);
    std::vector<at::Tensor> slots, blocks;
    block_copy_lists(layers, block_ids[u], in, slots, blocks);
    at::_foreach_copy_(slots, blocks, false);

    at::Tensor out = staging(out_shape, g.dtype, g.device, slot, /*kind=*/1);
    // That block's D2H read the output slot this swap is about to overwrite.
    if (u >= 2) wait(d2h_done[u - 2], main);
    // A permuted device copy: torch-rbln runs it as a compiled program.
    out.view({rows, g.block_size, g.heads, g.head_size})
        .copy_(in.view({rows, g.heads, g.block_size, g.head_size})
                   .permute({0, 2, 1, 3}));
    at::Tensor token_major =
        out.view({2, g.layers, g.block_size, g.heads * g.head_size});

    std::vector<at::Tensor> regions, pieces;
    chunk_copy_lists(chunks, token_major, u, bpc, g.block_size, regions,
                     pieces);
    Fence swapped = record(main);
    wait(swapped, copy);
    copy_pairs(copy, regions, pieces);
    d2h_done[u] = record(copy);
  }
  guard_impl.synchronizeStream(copy);
}

void scatter_chunks_to_blocks_token_major(const std::vector<at::Tensor>& layers,
                                          const std::vector<int64_t>& block_ids,
                                          const std::vector<at::Tensor>& chunks,
                                          int64_t bpc,
                                          int64_t skip_prefix_n_blocks) {
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
  c10::impl::VirtualGuardImpl guard_impl(g.device.type());
  const c10::Stream main = guard_impl.getStream(g.device);
  const c10::Stream copy = guard_impl.getNewStream(g.device);
  const int64_t m = n - start;  // blocks in this transfer

  auto landing = [&](int64_t u) {
    return staging(in_shape, g.dtype, g.device, static_cast<int>(u % 2),
                   /*kind=*/0);
  };
  std::vector<Fence> h2d_done(m), swapped(m);
  auto issue_copy = [&](int64_t u) {
    at::Tensor in = landing(u);
    at::Tensor token_major =
        in.view({2, g.layers, g.block_size, g.heads * g.head_size});
    std::vector<at::Tensor> regions, pieces;
    chunk_copy_lists(chunks, token_major, start + u, bpc, g.block_size, regions,
                     pieces);
    copy_pairs(copy, pieces, regions);
    h2d_done[u] = record(copy);
  };

  if (m > 0) issue_copy(0);
  for (int64_t u = 0; u < m; ++u) {
    if (u + 1 < m) {
      // That H2D overwrites the landing slot the swap two blocks back read.
      if (u >= 1) wait(swapped[u - 1], copy);
      issue_copy(u + 1);
    }
    at::Tensor in = landing(u);
    const int slot = static_cast<int>(u % 2);
    at::Tensor out = staging(out_shape, g.dtype, g.device, slot, /*kind=*/1);
    // This block's H2D, not the ones issued after it.
    wait(h2d_done[u], main);
    out.view({rows, g.heads, g.block_size, g.head_size})
        .copy_(in.view({rows, g.block_size, g.heads, g.head_size})
                   .permute({0, 2, 1, 3}));
    swapped[u] = record(main);

    std::vector<at::Tensor> slots, blocks;
    block_copy_lists(layers, block_ids[start + u], out, slots, blocks);
    at::_foreach_copy_(blocks, slots, false);
  }
  guard_impl.synchronizeStream(copy);
}

}  // namespace lmcache::rbln
