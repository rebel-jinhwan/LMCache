// SPDX-License-Identifier: Apache-2.0

#include "kv_transfer.h"

#include <ATen/Parallel.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <tuple>
#include <vector>

#include <rebel/runtime/api/rbln_runtime_api.h>
#include <sys/mman.h>

namespace lmcache {
namespace rbln {
namespace {

void check(RBLNRetCode rc, const char* what) {
  TORCH_CHECK(rc == RBLNRetCode_SUCCESS, what, " failed: rbln retcode ",
              static_cast<int>(rc));
}

// One direction's worth of device<->host copies, submitted in a single batched
// runtime call. rbln_memcpy_{v2h,h2v}_multi is synchronous: it drains prior
// work on the torch device, dispatches every descriptor (split internally at
// the command-buffer limit) and returns once the destination is complete, so
// no per-slice async handle and no explicit rbln_device_synchronize is needed.
// Host ranges within one batch must not overlap; every device range must live
// on the same torch device (both hold for a per-layer paged KV cache).
class CopyBatch {
 public:
  CopyBatch(TransferDirection dir, size_t capacity) : dir_(dir) {
    if (dir_ == TransferDirection::D2H) {
      v2h_.reserve(capacity);
    } else {
      h2v_.reserve(capacity);
    }
  }

  void add(uint64_t dev_vaddr, uintptr_t host_ptr, uint64_t nbytes) {
    if (dir_ == TransferDirection::D2H) {
      v2h_.emplace_back(dev_vaddr, host_ptr, nbytes);
    } else {
      h2v_.emplace_back(host_ptr, dev_vaddr, nbytes);
    }
  }

  void submit() {
    if (dir_ == TransferDirection::D2H) {
      if (!v2h_.empty()) {
        check(::rbln::rbln_memcpy_v2h_multi(v2h_), "rbln_memcpy_v2h_multi");
      }
    } else if (!h2v_.empty()) {
      check(::rbln::rbln_memcpy_h2v_multi(h2v_), "rbln_memcpy_h2v_multi");
    }
  }

 private:
  TransferDirection dir_;
  // (src_vaddr, dst_host_ptr, size)
  std::vector<std::tuple<uint64_t, uintptr_t, uint64_t>> v2h_;
  // (src_host_ptr, dst_vaddr, size)
  std::vector<std::tuple<uintptr_t, uint64_t, uint64_t>> h2v_;
};

// Per-thread host staging area for the DMA leg of the HND path. Backed by
// 2 MiB-aligned memory with MADV_HUGEPAGE and pre-touched: the runtime pins
// the host range on every transfer, and pinning 4 KiB pages -- not the DMA
// itself -- dominated; hugepages cut the page count 512x and lift measured
// device<->host throughput from ~14 to ~33 GB/s on RBLN-CR13. The hint is
// advisory: without THP the buffer still works on plain pages.
char* host_staging(size_t bytes) {
  constexpr size_t kHuge = size_t{2} << 20;
  struct Area {
    char* ptr = nullptr;
    size_t size = 0;
    ~Area() { std::free(ptr); }
  };
  thread_local Area area;
  if (area.size >= bytes) return area.ptr;
  const size_t rounded = (bytes + kHuge - 1) / kHuge * kHuge;
  void* p = nullptr;
  TORCH_CHECK(posix_memalign(&p, kHuge, rounded) == 0, "posix_memalign(",
              rounded, ") failed for RBLN staging");
  madvise(p, rounded, MADV_HUGEPAGE);
  std::memset(p, 0, rounded);  // fault in once, off the hot path
  std::free(area.ptr);
  area.ptr = static_cast<char*>(p);
  area.size = rounded;
  return area.ptr;
}

struct BlockList {
  int num_chunks = 0;
  int total_blocks = 0;
  int blocks_per_chunk = 0;
  int64_t chunk_tokens = 0;
};

BlockList inspect_blocks(const std::vector<at::Tensor>& lmcache_chunks,
                         const std::vector<int64_t>& block_ids,
                         int64_t block_size, int skip_prefix_n_blocks,
                         int64_t num_blocks) {
  TORCH_CHECK(!lmcache_chunks.empty(), "lmcache_chunks is empty");
  BlockList out;
  out.num_chunks = static_cast<int>(lmcache_chunks.size());
  out.total_blocks = static_cast<int>(block_ids.size());
  TORCH_CHECK(out.total_blocks >= 1, "block_ids is empty");
  TORCH_CHECK(out.total_blocks % out.num_chunks == 0, "block_ids length (",
              out.total_blocks, ") must be divisible by num_chunks (",
              out.num_chunks, ")");
  out.blocks_per_chunk = out.total_blocks / out.num_chunks;
  out.chunk_tokens = static_cast<int64_t>(out.blocks_per_chunk) * block_size;
  TORCH_CHECK(skip_prefix_n_blocks >= 0, "skip_prefix_n_blocks must be >= 0");
  for (int flat = skip_prefix_n_blocks; flat < out.total_blocks; ++flat) {
    TORCH_CHECK(block_ids[flat] >= 0 && block_ids[flat] < num_blocks,
                "block id ", block_ids[flat], " out of range [0, ", num_blocks,
                ")");
  }
  return out;
}

// HND paged block [NH, BS, HS] <-> one token-major window [BS, NH*HS].
void permute_hnd_block(char* token_major, char* hnd, int64_t num_heads,
                       int64_t block_size, int64_t head_size, int64_t elem,
                       bool hnd_to_token_major) {
  const int64_t hidden = num_heads * head_size;
  for (int64_t h = 0; h < num_heads; ++h) {
    for (int64_t bs = 0; bs < block_size; ++bs) {
      char* hnd_ptr = hnd + ((h * block_size + bs) * head_size) * elem;
      char* tm_ptr = token_major + ((bs * hidden + h * head_size) * elem);
      if (hnd_to_token_major) {
        std::memcpy(tm_ptr, hnd_ptr, static_cast<size_t>(head_size * elem));
      } else {
        std::memcpy(hnd_ptr, tm_ptr, static_cast<size_t>(head_size * elem));
      }
    }
  }
}

// Runs fn(slice) for every slice in [0, n) on up to at::get_num_threads()
// plain threads. at::parallel_for is avoided on purpose: it degrades to a
// serial loop when the caller is not the main thread, which is how the mp
// worker reaches this code.
template <typename Fn>
void for_each_slice(int64_t n, Fn fn) {
  const int nt = static_cast<int>(
      std::min<int64_t>(std::max(1, at::get_num_threads()), n));
  if (nt <= 1) {
    for (int64_t s = 0; s < n; ++s) fn(s);
    return;
  }
  std::vector<std::thread> pool;
  pool.reserve(nt);
  std::atomic<int64_t> next{0};
  for (int t = 0; t < nt; ++t) {
    pool.emplace_back([&] {
      for (int64_t s = next.fetch_add(1); s < n; s = next.fetch_add(1)) {
        fn(s);
      }
    });
  }
  for (auto& th : pool) th.join();
}

}  // namespace

void block_kv_transfer_hnd(std::vector<at::Tensor> kv_caches,
                           std::vector<at::Tensor> lmcache_chunks,
                           std::vector<int64_t> block_ids,
                           TransferDirection direction,
                           int skip_prefix_n_blocks) {
  TORCH_CHECK(direction == TransferDirection::D2H ||
                  direction == TransferDirection::H2D,
              "unsupported transfer direction ", static_cast<int>(direction));
  TORCH_CHECK(!kv_caches.empty(), "kv_caches is empty");
  const at::Tensor& first = kv_caches[0];
  TORCH_CHECK(!first.device().is_cpu(), "kv_caches must live on RBLN");
  TORCH_CHECK(first.dim() == 6 || first.dim() == 5,
              "HND kv_caches must be 6-D [2, NB, NH, 1, BS, HS] or squeezed "
              "5-D [2, NB, NH, BS, HS]; got dim ",
              first.dim());
  TORCH_CHECK(first.size(0) == 2, "HND kv_caches dim 0 must be 2 (K,V); got ",
              first.size(0));

  const int64_t num_blocks = first.size(1);
  const int64_t num_heads = first.size(2);
  const int64_t block_size = first.dim() == 6 ? first.size(4) : first.size(3);
  const int64_t head_size = first.dim() == 6 ? first.size(5) : first.size(4);
  if (first.dim() == 6) {
    TORCH_CHECK(first.size(3) == 1, "HND singleton axis must be 1; got ",
                first.size(3));
  }
  const int64_t elem = first.element_size();
  const int num_layers = static_cast<int>(kv_caches.size());
  const BlockList blocks = inspect_blocks(lmcache_chunks, block_ids, block_size,
                                          skip_prefix_n_blocks, num_blocks);

  const int64_t hidden = num_heads * head_size;
  const int64_t chunk_numel = 2 * num_layers * blocks.chunk_tokens * hidden;
  const int64_t head_stride = block_size * head_size;
  const int64_t block_stride = num_heads * head_stride;
  const int64_t kv_stride = num_blocks * block_stride;
  const uint64_t block_bytes =
      static_cast<uint64_t>(block_stride) * static_cast<uint64_t>(elem);
  const int64_t lm_layer_stride = blocks.chunk_tokens * hidden;
  const int64_t lm_kv_stride =
      static_cast<int64_t>(num_layers) * lm_layer_stride;

  for (const auto& kv : kv_caches) {
    TORCH_CHECK(kv.is_contiguous(), "kv_caches tensors must be contiguous");
    TORCH_CHECK(!kv.device().is_cpu(), "kv_caches must live on RBLN");
    TORCH_CHECK(kv.sizes() == first.sizes() && kv.dtype() == first.dtype(),
                "every kv_caches layer must share the first layer's shape and "
                "dtype");
  }
  for (const auto& chunk : lmcache_chunks) {
    TORCH_CHECK(chunk.is_contiguous(), "lmcache_chunks must be contiguous");
    TORCH_CHECK(chunk.device().is_cpu(),
                "lmcache_chunks must live on the host");
    TORCH_CHECK(chunk.dtype() == first.dtype(),
                "lmcache_chunks dtype must match kv_caches dtype");
    TORCH_CHECK(chunk.numel() >= chunk_numel, "lmcache chunk numel (",
                chunk.numel(), ") smaller than HND chunk size (", chunk_numel,
                ")");
  }

  // Staging holds one HND-layout block per active flat block:
  // [active, kv, layer] slots of block_bytes each.
  const int n_active = blocks.total_blocks - skip_prefix_n_blocks;
  const int64_t slot_stride =
      2 * static_cast<int64_t>(num_layers) * block_stride;
  // Resolve the thread-local staging area once, on the calling thread: the
  // permute workers below must address this thread's buffer, not their own.
  char* const staging_base = host_staging(static_cast<size_t>(n_active) *
                                          static_cast<size_t>(slot_stride) *
                                          static_cast<size_t>(elem));
  auto staged = [=](int active_idx, int kv, int layer) -> char* {
    const int64_t off = (static_cast<int64_t>(active_idx) * slot_stride +
                         kv * static_cast<int64_t>(num_layers) * block_stride +
                         static_cast<int64_t>(layer) * block_stride) *
                        elem;
    return staging_base + off;
  };

  // Every (block, K/V, layer) slice is one contiguous HND block on the device
  // and one [BS, NH*HS] window in its chunk; slice s enumerates them.
  const int64_t n_slices =
      static_cast<int64_t>(n_active) * 2 * static_cast<int64_t>(num_layers);
  auto slice_coords = [&](int64_t s, int& active_idx, int& kv, int& layer) {
    active_idx = static_cast<int>(s / (2 * num_layers));
    const int rem = static_cast<int>(s % (2 * num_layers));
    kv = rem / num_layers;
    layer = rem % num_layers;
  };
  auto permute_slice = [&](int64_t s, bool hnd_to_token_major) {
    int active_idx, kv, layer;
    slice_coords(s, active_idx, kv, layer);
    const int flat = active_idx + skip_prefix_n_blocks;
    const int chunk_idx = flat / blocks.blocks_per_chunk;
    const int block_in_chunk = flat % blocks.blocks_per_chunk;
    auto* chunk_base = static_cast<char*>(lmcache_chunks[chunk_idx].data_ptr());
    const int64_t tok_off = static_cast<int64_t>(block_in_chunk) * block_size;
    char* tm =
        chunk_base +
        (kv * lm_kv_stride + layer * lm_layer_stride + tok_off * hidden) * elem;
    permute_hnd_block(tm, staged(active_idx, kv, layer), num_heads, block_size,
                      head_size, elem, hnd_to_token_major);
  };

  const bool is_d2h = direction == TransferDirection::D2H;
  if (!is_d2h) {
    for_each_slice(n_slices, [&](int64_t s) {
      permute_slice(s, /*hnd_to_token_major=*/false);
    });
  }

  // One descriptor per slice; a single batched dispatch moves the whole burst
  // between DRAM and the staging area.
  CopyBatch batch(direction, static_cast<size_t>(n_slices));
  for (int64_t s = 0; s < n_slices; ++s) {
    int active_idx, kv, layer;
    slice_coords(s, active_idx, kv, layer);
    const int64_t b = block_ids[active_idx + skip_prefix_n_blocks];
    const auto eng_base =
        reinterpret_cast<uint64_t>(kv_caches[layer].data_ptr());
    const int64_t eng_off = kv * kv_stride + b * block_stride;
    const uint64_t dev =
        eng_base + static_cast<uint64_t>(eng_off) * static_cast<uint64_t>(elem);
    batch.add(dev, reinterpret_cast<uintptr_t>(staged(active_idx, kv, layer)),
              block_bytes);
  }
  batch.submit();

  if (is_d2h) {
    for_each_slice(n_slices, [&](int64_t s) {
      permute_slice(s, /*hnd_to_token_major=*/true);
    });
  }
}

void block_kv_transfer_mla(std::vector<at::Tensor> kv_caches,
                           std::vector<at::Tensor> lmcache_chunks,
                           std::vector<int64_t> block_ids,
                           TransferDirection direction,
                           int skip_prefix_n_blocks) {
  TORCH_CHECK(direction == TransferDirection::D2H ||
                  direction == TransferDirection::H2D,
              "unsupported transfer direction ", static_cast<int>(direction));
  TORCH_CHECK(!kv_caches.empty(), "kv_caches is empty");
  const at::Tensor& first = kv_caches[0];
  TORCH_CHECK(first.dim() == 3,
              "MLA kv_caches must be 3-D [NB, BS, HS]; got dim ", first.dim());
  const int64_t num_blocks = first.size(0);
  const int64_t block_size = first.size(1);
  const int64_t head_size = first.size(2);
  const int64_t elem = first.element_size();
  const int num_layers = static_cast<int>(kv_caches.size());
  const BlockList blocks = inspect_blocks(lmcache_chunks, block_ids, block_size,
                                          skip_prefix_n_blocks, num_blocks);

  const int64_t chunk_numel = num_layers * blocks.chunk_tokens * head_size;
  for (const auto& kv : kv_caches) {
    TORCH_CHECK(kv.is_contiguous(), "kv_caches tensors must be contiguous");
    TORCH_CHECK(kv.sizes() == first.sizes() && kv.dtype() == first.dtype(),
                "every kv_caches layer must share the first layer's shape and "
                "dtype");
  }
  for (const auto& chunk : lmcache_chunks) {
    TORCH_CHECK(chunk.is_contiguous(), "lmcache_chunks must be contiguous");
    TORCH_CHECK(chunk.device().is_cpu(),
                "lmcache_chunks must live on the host");
    TORCH_CHECK(chunk.dtype() == first.dtype(),
                "lmcache_chunks dtype must match kv_caches dtype");
    TORCH_CHECK(chunk.numel() >= chunk_numel, "lmcache chunk numel (",
                chunk.numel(), ") smaller than MLA chunk size (", chunk_numel,
                ")");
  }

  const int64_t block_numel = block_size * head_size;
  const uint64_t block_bytes =
      static_cast<uint64_t>(block_numel) * static_cast<uint64_t>(elem);
  const int64_t lm_layer_stride = blocks.chunk_tokens * head_size;

  // MLA blocks are already token-major, so each (block, layer) copies straight
  // between DRAM and its slot in the chunk: one descriptor each, one dispatch.
  const int n_active = blocks.total_blocks - skip_prefix_n_blocks;
  CopyBatch batch(direction, static_cast<size_t>(n_active) *
                                 static_cast<size_t>(num_layers));
  for (int flat = skip_prefix_n_blocks; flat < blocks.total_blocks; ++flat) {
    const int chunk_idx = flat / blocks.blocks_per_chunk;
    const int block_in_chunk = flat % blocks.blocks_per_chunk;
    const int64_t b = block_ids[flat];
    const auto host_base =
        reinterpret_cast<uintptr_t>(lmcache_chunks[chunk_idx].data_ptr());
    const int64_t eng_off = b * block_numel;
    for (int layer = 0; layer < num_layers; ++layer) {
      const auto eng_base =
          reinterpret_cast<uint64_t>(kv_caches[layer].data_ptr());
      const int64_t lm_off = layer * lm_layer_stride +
                             static_cast<int64_t>(block_in_chunk) * block_numel;
      const uint64_t dev = eng_base + static_cast<uint64_t>(eng_off) *
                                          static_cast<uint64_t>(elem);
      const uintptr_t host = host_base + static_cast<uintptr_t>(lm_off) *
                                             static_cast<uintptr_t>(elem);
      batch.add(dev, host, block_bytes);
    }
  }
  batch.submit();
}

void multi_layer_block_kv_transfer(
    std::vector<at::Tensor> kv_caches, std::vector<at::Tensor> lmcache_chunks,
    std::vector<int64_t> block_ids, const torch::Device& device,
    TransferDirection direction, PageBufferShapeDesc shape_desc,
    int lmcache_chunk_size, EngineKVFormat engine_kv_format,
    int skip_prefix_n_blocks) {
  (void)device;  // taken from the tensors
  TORCH_CHECK(
      shape_desc.bs > 0 && lmcache_chunk_size % shape_desc.bs == 0,
      "lmcache_chunk_size must be a positive multiple of shape_desc.bs");
  switch (engine_kv_format) {
    case EngineKVFormat::NL_X_TWO_NB_NH_ONE_BS_HS:
      block_kv_transfer_hnd(std::move(kv_caches), std::move(lmcache_chunks),
                            std::move(block_ids), direction,
                            skip_prefix_n_blocks);
      return;
    case EngineKVFormat::NL_X_NB_BS_HS:
      block_kv_transfer_mla(std::move(kv_caches), std::move(lmcache_chunks),
                            std::move(block_ids), direction,
                            skip_prefix_n_blocks);
      return;
    default:
      TORCH_CHECK(false,
                  "RBLN block transfer supports only NL_X_TWO_NB_NH_ONE_BS_HS "
                  "and NL_X_NB_BS_HS; got ",
                  static_cast<int>(engine_kv_format));
  }
}

}  // namespace rbln
}  // namespace lmcache
