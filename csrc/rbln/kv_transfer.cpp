// SPDX-License-Identifier: Apache-2.0

#include "kv_transfer.h"

#include <cstdint>
#include <vector>

#include <rebel/runtime/api/rbln_runtime_api.h>

namespace lmcache {
namespace rbln {
namespace {

void check(RBLNRetCode rc, const char* what) {
  TORCH_CHECK(rc == RBLNRetCode_SUCCESS, what, " failed: rbln retcode ",
              static_cast<int>(rc));
}

// One async device<->host copy in the requested direction. A device vaddr is a
// torch rbln tensor's data_ptr(); the host ptr is an lmcache chunk's
// data_ptr(). The returned rebel handle is drained by the caller's device sync.
void copy_async(TransferDirection dir, uint64_t dev_vaddr, uintptr_t host_ptr,
                uint64_t nbytes) {
  uint64_t handle = 0;
  if (dir == TransferDirection::D2H) {
    check(::rbln::rbln_memcpy_v2h_async(dev_vaddr, host_ptr, nbytes, &handle),
          "rbln_memcpy_v2h_async");
  } else {
    check(::rbln::rbln_memcpy_h2v_async(host_ptr, dev_vaddr, nbytes, &handle),
          "rbln_memcpy_h2v_async");
  }
}

}  // namespace

void block_kv_transfer_mla(std::vector<at::Tensor> kv_caches,
                           std::vector<at::Tensor> lmcache_chunks,
                           std::vector<int64_t> block_ids,
                           TransferDirection direction,
                           int skip_prefix_n_blocks) {
  TORCH_CHECK(direction == TransferDirection::D2H ||
                  direction == TransferDirection::H2D,
              "unsupported transfer direction ", static_cast<int>(direction));
  const int num_layers = static_cast<int>(kv_caches.size());
  const int num_chunks = static_cast<int>(lmcache_chunks.size());
  TORCH_CHECK(num_layers >= 1, "kv_caches is empty");
  TORCH_CHECK(num_chunks >= 1, "lmcache_chunks is empty");

  // --- Geometry from the first (representative) layer -----------------------
  const at::Tensor& first = kv_caches[0];
  TORCH_CHECK(first.dim() == 3,
              "MLA kv_caches must be 3-D [NB, BS, HS]; got dim ", first.dim());
  const int64_t num_blocks = first.size(0);
  const int64_t block_size = first.size(1);
  const int64_t head_size = first.size(2);
  const int64_t elem = first.element_size();

  const int total_blocks = static_cast<int>(block_ids.size());
  TORCH_CHECK(total_blocks >= 1, "block_ids is empty");
  TORCH_CHECK(total_blocks % num_chunks == 0, "block_ids length (",
              total_blocks, ") must be divisible by num_chunks (", num_chunks,
              ")");
  const int blocks_per_chunk = total_blocks / num_chunks;
  const int64_t chunk_tokens =
      static_cast<int64_t>(blocks_per_chunk) * block_size;

  // Each layer is contiguous with the representative geometry; each chunk is a
  // contiguous host buffer big enough for a token-major [L, T, HS] view.
  // Checked up front so a mismatch is a clean error, not a silent OOB DMA.
  const int64_t chunk_numel = num_layers * chunk_tokens * head_size;
  for (const auto& kv : kv_caches) {
    TORCH_CHECK(kv.is_contiguous(), "kv_caches tensors must be contiguous");
    TORCH_CHECK(kv.sizes() == first.sizes() && kv.dtype() == first.dtype(),
                "every kv_caches layer must share the first layer's shape and "
                "dtype");
  }
  for (const auto& c : lmcache_chunks) {
    TORCH_CHECK(c.is_contiguous(), "lmcache_chunks must be contiguous");
    TORCH_CHECK(c.device().is_cpu(), "lmcache_chunks must live on the host");
    TORCH_CHECK(c.dtype() == first.dtype(),
                "lmcache_chunks dtype must match kv_caches dtype");
    TORCH_CHECK(c.numel() >= chunk_numel, "lmcache chunk numel (", c.numel(),
                ") smaller than MLA chunk size (", chunk_numel, ")");
  }
  TORCH_CHECK(skip_prefix_n_blocks >= 0, "skip_prefix_n_blocks must be >= 0");
  for (int flat = skip_prefix_n_blocks; flat < total_blocks; ++flat) {
    TORCH_CHECK(block_ids[flat] >= 0 && block_ids[flat] < num_blocks,
                "block id ", block_ids[flat], " out of range [0, ", num_blocks,
                ")");
  }

  // One block of one layer: contiguous on both sides.
  const int64_t block_numel = block_size * head_size;
  const uint64_t block_bytes = static_cast<uint64_t>(block_numel) * elem;
  // Chunk [L, T, HS] element strides.
  const int64_t lm_layer_stride = chunk_tokens * head_size;

  int device_id =
      -1;  // resolved once from the first vaddr, for the final sync.

  for (int flat = skip_prefix_n_blocks; flat < total_blocks; ++flat) {
    const int chunk_idx = flat / blocks_per_chunk;
    const int block_in_chunk = flat % blocks_per_chunk;
    const int64_t b = block_ids[flat];
    const auto host_base =
        reinterpret_cast<uintptr_t>(lmcache_chunks[chunk_idx].data_ptr());
    const int64_t eng_off = b * block_numel;  // element offset of the block

    for (int layer = 0; layer < num_layers; ++layer) {
      const auto eng_base =
          reinterpret_cast<uint64_t>(kv_caches[layer].data_ptr());
      const int64_t lm_off =
          layer * lm_layer_stride + block_in_chunk * block_numel;
      const uint64_t dev = eng_base + static_cast<uint64_t>(eng_off) * elem;
      const uintptr_t host = host_base + static_cast<uintptr_t>(lm_off) * elem;
      copy_async(direction, dev, host, block_bytes);

      if (device_id < 0) {
        uint32_t did = 0;
        check(::rbln::rbln_get_torch_device_id_from_vaddr(eng_base, did),
              "rbln_get_torch_device_id_from_vaddr");
        device_id = static_cast<int>(did);
      }
    }
  }

  // Drain every issued transfer before returning: rebel does no auto-sync on
  // host read, and the caller must see a fully materialized chunk (D2H) or a
  // fully written KV cache (H2D). One device sync covers the whole burst.
  if (device_id >= 0) {
    check(::rbln::rbln_device_synchronize(static_cast<uint32_t>(device_id)),
          "rbln_device_synchronize");
  }
}

}  // namespace rbln
}  // namespace lmcache
