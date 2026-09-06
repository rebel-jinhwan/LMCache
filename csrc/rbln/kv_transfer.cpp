// SPDX-License-Identifier: Apache-2.0
#include "kv_transfer.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/Event.h>
#include <c10/core/StreamGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <tuple>

namespace lmcache::rbln {
namespace {

// Staging buffers: per thread, per (shape, dtype, device, slot, role). Two
// slots alternate so a block's swap does not overwrite what the previous
// block's host copy is still reading.
//
// The two directions ask for the same two shapes: gather lands in
// [2, L, H, BS, D] and swaps into [2, L, BS, H, D], scatter lands in the second
// and swaps into the first. They cannot be in flight together on one thread --
// each call drains its copy stream before returning -- so a thread that does
// both reuses a buffer across directions when the role (below) agrees: the
// token-major buffer is a host DMA endpoint for both, so it is shared, while
// the head-major one is gather's device-filled landing but scatter's D2D
// source, and those want different chiplets, so a thread doing both holds two.
// In the serve a thread does one direction (store on the commit pool, retrieve
// on its caller), so it holds 4 buffers -- 470 MB on this geometry -- or 2 when
// the swap runs in place (below), and the buffers are thread_local, so that
// multiplies by the pool's thread count.
// Where a staging buffer lives on the device. RBLN DRAM is one pool per chiplet
// and an allocation is pinned to the chiplet it lands on -- the runtime does not
// spill -- and every torch allocation lands on chiplet 0 unless told otherwise.
// The staging set is large (a whole block per buffer, several buffers per thread,
// one set per transfer thread), so left alone it all stacks on chiplet 0's pool
// next to that chiplet's share of the model.
//
// Which buffers can leave chiplet 0 without slowing the transfer was measured
// (Qwen3-1.7B KV, 8 blocks): a host DMA endpoint on another chiplet is slower
// in both directions (gather 46 -> 36 GB/s with its D2H source there, scatter
// 41 -> 38 with its H2D landing there), and so is the source of a D2D (scatter
// 41 -> 38 with its swap output there); the cost grows with the chiplet's
// distance from 0 and chiplet 3 is the worst (gather 29 with everything on it).
// The one buffer that is only ever the *destination* of a D2D and then read by
// the compiled permute -- gather's landing buffer -- moves at full rate from any
// chiplet (gather held 46.6 with it on chiplets 1 and 2). LMCACHE_RBLN_STAGING_CHIPLET:
//   spread      gather's landing buffers round-robin over chiplets 1.., every
//               other buffer on chiplet 0 -- no throughput cost (default)
//   spread-all  every buffer round-robin over every chiplet
//   main        chiplet 0 for everything, the runtime's default placement
//   <n>         pin every staging buffer to chiplet n
// Placement goes through torch-rbln's dispatcher op so this extension keeps
// linking only ATen; a torch-rbln without the op leaves the buffers where the
// runtime puts them.
enum class Placement { kSpread, kSpreadAll, kMain, kPinned };

// How a staging buffer is reached: only as the destination of a D2D and by the
// compiled permute (may live on any chiplet), or as a host DMA endpoint or a
// D2D source (stays on chiplet 0).
enum class Role { kDeviceFilled, kDmaEndpoint };

struct StagingPlacement {
  Placement policy;
  int64_t chiplet;  // kPinned only
};

StagingPlacement parse_placement(const char* value) {
  if (value == nullptr || *value == '\0' || std::strcmp(value, "spread") == 0) {
    return {Placement::kSpread, 0};
  }
  if (std::strcmp(value, "spread-all") == 0) return {Placement::kSpreadAll, 0};
  if (std::strcmp(value, "main") == 0) return {Placement::kMain, 0};
  char* end = nullptr;
  const long n = std::strtol(value, &end, 10);
  TORCH_CHECK(end != value && *end == '\0' && n >= 0,
              "LMCACHE_RBLN_STAGING_CHIPLET must be 'spread', 'spread-all', 'main' "
              "or a non-negative chiplet index, got '",
              value, "'");
  return {Placement::kPinned, static_cast<int64_t>(n)};
}

const StagingPlacement& staging_placement() {
  static const StagingPlacement placement =
      parse_placement(std::getenv("LMCACHE_RBLN_STAGING_CHIPLET"));
  return placement;
}

// Places a freshly allocated staging buffer per the policy. A no-op under kMain
// and when torch-rbln predates the placement op.
void place_staging(at::Tensor& buf, Role role) {
  const StagingPlacement& placement = staging_placement();
  if (placement.policy == Placement::kMain) return;
  static const auto bind = c10::Dispatcher::singleton().findSchema(
      {"torch_rbln::bind_device_memory_at", ""});
  static const auto count =
      c10::Dispatcher::singleton().findSchema({"torch_rbln::chiplet_count", ""});
  if (!bind.has_value() || !count.has_value()) {
    TORCH_WARN_ONCE(
        "LMCACHE_RBLN_STAGING_CHIPLET is set but this torch-rbln has no "
        "torch_rbln::bind_device_memory_at; staging buffers stay on the runtime's "
        "default chiplet");
    return;
  }
  int64_t chiplet = placement.chiplet;
  if (placement.policy != Placement::kPinned) {
    static const int64_t n_chiplets = std::max<int64_t>(
        count->typed<int64_t(const at::Tensor&)>().call(buf), 1);
    // One counter for the process, so the per-thread staging sets interleave
    // across chiplets instead of each thread starting over at the same one.
    static std::atomic<int64_t> next{0};
    if (placement.policy == Placement::kSpreadAll) {
      chiplet = next.fetch_add(1) % n_chiplets;
    } else if (role == Role::kDmaEndpoint || n_chiplets == 1) {
      chiplet = 0;
    } else {
      chiplet = 1 + next.fetch_add(1) % (n_chiplets - 1);
    }
  }
  bind->typed<void(at::Tensor&, int64_t)>().call(buf, chiplet);
}

at::Tensor staging(at::IntArrayRef shape, at::ScalarType dtype,
                   const at::Device& device, int slot, Role role) {
  using Key =
      std::tuple<std::vector<int64_t>, at::ScalarType, std::string, int, Role>;
  thread_local std::map<Key, at::Tensor> buffers;
  Key key{shape.vec(), dtype, device.str(), slot, role};
  auto it = buffers.find(key);
  if (it == buffers.end()) {
    at::Tensor buf =
        at::empty(shape, at::TensorOptions().dtype(dtype).device(device));
    place_staging(buf, role);
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

// The head<->token swap is a compiled device program. torch-rbln's
// copy_strided_view_inplace runs it with the output aliasing the input -- the
// program permutes the buffer in its own storage, and torch-rbln checks each
// geometry once against an out-of-place copy -- so a slot holds one buffer
// instead of a landing and a swap buffer: 2 per thread rather than 4, 235 MB
// on Qwen3-1.7B. The price is placement: the one buffer is the block's DMA
// endpoint as well as the permute's operand, so it stays on chiplet 0, where
// the two-buffer path could float gather's landing buffer to another chiplet.
// Chiplet 0 holds the same bytes either way; the other chiplets hold none.
// LMCACHE_RBLN_STAGING_INPLACE=0 keeps the two-buffer path (a torch-rbln
// without the op does too, with a warning once).
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

std::string staging_swap_mode() {
  return inplace_swap() ? "inplace" : "two-buffer";
}

std::string staging_placement_name() {
  const StagingPlacement& placement = staging_placement();
  switch (placement.policy) {
    case Placement::kSpread:
      return "spread";
    case Placement::kSpreadAll:
      return "spread-all";
    case Placement::kMain:
      return "main";
    case Placement::kPinned:
      return "chiplet:" + std::to_string(placement.chiplet);
  }
  return "spread";
}

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
  c10::impl::VirtualGuardImpl guard_impl(g.device.type());
  const c10::Stream main = guard_impl.getStream(g.device);
  const c10::Stream copy = guard_impl.getNewStream(g.device);
  const bool inplace = inplace_swap();
  std::vector<Fence> d2h_done(n);
  for (int64_t u = 0; u < n; ++u) {
    const int slot = static_cast<int>(u % 2);
    // In place, the landing buffer is also the D2H source (a DMA endpoint), and
    // the D2H of the block two back read this very buffer.
    at::Tensor in = staging(in_shape, g.dtype, g.device, slot,
                            inplace ? Role::kDmaEndpoint : Role::kDeviceFilled);
    if (inplace && u >= 2) wait(d2h_done[u - 2], main);
    std::vector<at::Tensor> slots, blocks;
    block_copy_lists(layers, block_ids[u], in, slots, blocks);
    at::_foreach_copy_(slots, blocks, false);

    at::Tensor token_major;
    if (inplace) {
      swap_in_place(in, rows, g.heads, g.block_size, g.head_size);
      token_major = in.view({2, g.layers, g.block_size, g.heads * g.head_size});
    } else {
      at::Tensor out =
          staging(out_shape, g.dtype, g.device, slot, Role::kDmaEndpoint);
      // That block's D2H read the output slot this swap is about to overwrite.
      if (u >= 2) wait(d2h_done[u - 2], main);
      // A permuted device copy: torch-rbln runs it as a compiled program.
      out.view({rows, g.block_size, g.heads, g.head_size})
          .copy_(in.view({rows, g.heads, g.block_size, g.head_size})
                     .permute({0, 2, 1, 3}));
      token_major = out.view({2, g.layers, g.block_size, g.heads * g.head_size});
    }

    std::vector<at::Tensor> regions, pieces;
    chunk_copy_lists(chunks, token_major, u, bpc, g.block_size, regions,
                     pieces);
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
                   Role::kDmaEndpoint);
  };
  const bool inplace = inplace_swap();
  // Per block: its H2D; the swap that read its landing buffer; the D2D into
  // the paged blocks that read the swapped buffer (in place, the same buffer).
  std::vector<Fence> h2d_done(m), swapped(m), scattered(m);
  auto issue_copy = [&](int64_t u) {
    at::Tensor in = landing(u);
    at::Tensor token_major =
        in.view({2, g.layers, g.block_size, g.heads * g.head_size});
    std::vector<at::Tensor> regions, pieces;
    chunk_copy_lists(chunks, token_major, start + u, bpc, g.block_size, regions,
                     pieces);
    {
      c10::StreamGuard guard(copy);
      at::_foreach_copy_(pieces, regions, /*non_blocking=*/true);
    }
    h2d_done[u] = record(copy);
  };

  if (m > 0) issue_copy(0);
  for (int64_t u = 0; u < m; ++u) {
    if (u + 1 < m) {
      // That H2D overwrites the landing slot the block before last used: read
      // by its swap, and in place by its D2D into the paged blocks as well.
      if (u >= 1) wait(inplace ? scattered[u - 1] : swapped[u - 1], copy);
      issue_copy(u + 1);
    }
    at::Tensor in = landing(u);
    const int slot = static_cast<int>(u % 2);
    // This block's H2D, not the ones issued after it.
    wait(h2d_done[u], main);
    at::Tensor head_major;
    if (inplace) {
      // The bytes are head-major now; the tensor's shape has to say so too.
      swap_in_place(in, rows, g.block_size, g.heads, g.head_size);
      head_major = in.view(out_shape);
    } else {
      head_major =
          staging(out_shape, g.dtype, g.device, slot, Role::kDmaEndpoint);
      head_major.view({rows, g.heads, g.block_size, g.head_size})
          .copy_(in.view({rows, g.block_size, g.heads, g.head_size})
                     .permute({0, 2, 1, 3}));
    }
    swapped[u] = record(main);

    std::vector<at::Tensor> slots, blocks;
    block_copy_lists(layers, block_ids[start + u], head_major, slots, blocks);
    at::_foreach_copy_(blocks, slots, false);
    scattered[u] = record(main);
  }
  guard_impl.synchronizeStream(copy);
}

}  // namespace lmcache::rbln
