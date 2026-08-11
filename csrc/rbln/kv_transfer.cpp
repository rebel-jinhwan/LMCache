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

// One async device<->host copy. The rebel handle is drained by the caller's
// single device sync, so it is not kept.
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

void head_major_block_kv_transfer(const std::vector<at::Tensor>& paged_layers,
                                  const std::vector<at::Tensor>& chunks,
                                  const std::vector<int64_t>& block_ids,
                                  TransferDirection direction,
                                  int64_t skip_prefix_n_blocks) {
  const int64_t num_layers = static_cast<int64_t>(paged_layers.size());
  const int64_t num_chunks = static_cast<int64_t>(chunks.size());
  TORCH_CHECK(num_layers >= 1, "paged_layers is empty");
  TORCH_CHECK(num_chunks >= 1, "chunks is empty");
  TORCH_CHECK(skip_prefix_n_blocks >= 0, "skip_prefix_n_blocks must be >= 0");

  // Every layer must match the first, so one set of strides addresses all.
  const at::Tensor& first = paged_layers[0];
  TORCH_CHECK(first.dim() == 5,
              "paged_layers must be 5-D [2, NB, NH, BS, HS]; got dim ",
              first.dim());
  TORCH_CHECK(first.size(0) == 2, "paged_layers dim 0 must be 2 (K,V); got ",
              first.size(0));
  const int64_t num_blocks = first.size(1);
  const int64_t num_kv_heads = first.size(2);
  const int64_t block_size = first.size(3);
  const int64_t head_size = first.size(4);
  const int64_t elem = first.element_size();

  const int64_t total_blocks = static_cast<int64_t>(block_ids.size());
  TORCH_CHECK(total_blocks % num_chunks == 0, "block_ids length (",
              total_blocks, ") must be divisible by the chunk count (",
              num_chunks, ")");
  const int64_t blocks_per_chunk = total_blocks / num_chunks;
  const int64_t chunk_tokens = blocks_per_chunk * block_size;

  // Checked up front: a mismatch below would be a silent out-of-bounds DMA.
  const int64_t chunk_numel =
      2 * num_layers * num_kv_heads * chunk_tokens * head_size;
  for (const auto& layer : paged_layers) {
    TORCH_CHECK(layer.is_contiguous(),
                "paged_layers tensors must be contiguous");
    TORCH_CHECK(layer.sizes() == first.sizes(),
                "every paged layer must share the first layer's shape");
    TORCH_CHECK(layer.scalar_type() == first.scalar_type(),
                "every paged layer must share the first layer's dtype");
  }
  for (const auto& chunk : chunks) {
    TORCH_CHECK(chunk.is_contiguous(), "chunks must be contiguous");
    TORCH_CHECK(chunk.scalar_type() == first.scalar_type(),
                "chunks must share the paged buffer's dtype");
    TORCH_CHECK(chunk.numel() >= chunk_numel, "chunk numel (", chunk.numel(),
                ") is smaller than the head-major chunk size (", chunk_numel,
                ")");
  }
  for (const int64_t b : block_ids) {
    TORCH_CHECK(b >= 0 && b < num_blocks, "block id ", b,
                " is out of range for a buffer with ", num_blocks, " blocks");
  }

  // Element strides of the paged buffer (contiguous [2, NB, NH, BS, HS]).
  const int64_t head_stride = block_size * head_size;       // one head
  const int64_t block_stride = num_kv_heads * head_stride;  // one block
  const int64_t kv_stride = num_blocks * block_stride;      // K half vs V half

  // Element strides of the head-major chunk [2, L, H, T, D].
  const int64_t lm_head_stride = chunk_tokens * head_size;
  const int64_t lm_layer_stride = num_kv_heads * lm_head_stride;
  const int64_t lm_kv_stride = num_layers * lm_layer_stride;

  // A one-block chunk is contiguous on both sides, so a whole (K|V, layer)
  // block moves in one DMA instead of one per head.
  const bool coalesce_heads = (chunk_tokens == block_size);

  int64_t device_id = -1;  // resolved once from the first vaddr, for the sync

  for (int64_t flat = skip_prefix_n_blocks; flat < total_blocks; ++flat) {
    const int64_t chunk_idx = flat / blocks_per_chunk;
    const int64_t block_in_chunk = flat % blocks_per_chunk;
    const int64_t b = block_ids[flat];
    const auto host_base =
        reinterpret_cast<uintptr_t>(chunks[chunk_idx].data_ptr());

    for (int64_t kv = 0; kv < 2; ++kv) {
      for (int64_t layer = 0; layer < num_layers; ++layer) {
        const auto dev_base =
            reinterpret_cast<uint64_t>(paged_layers[layer].data_ptr());
        const int64_t dev_off = kv * kv_stride + b * block_stride;
        const int64_t lm_off = kv * lm_kv_stride + layer * lm_layer_stride +
                               block_in_chunk * block_size * head_size;

        if (coalesce_heads) {
          copy_async(direction,
                     dev_base + static_cast<uint64_t>(dev_off) * elem,
                     host_base + static_cast<uintptr_t>(lm_off) * elem,
                     static_cast<uint64_t>(block_stride) * elem);
        } else {
          for (int64_t h = 0; h < num_kv_heads; ++h) {
            copy_async(
                direction,
                dev_base +
                    static_cast<uint64_t>(dev_off + h * head_stride) * elem,
                host_base +
                    static_cast<uintptr_t>(lm_off + h * lm_head_stride) * elem,
                static_cast<uint64_t>(head_stride) * elem);
          }
        }
        if (device_id < 0) {
          uint32_t did = 0;
          check(::rbln::rbln_get_torch_device_id_from_vaddr(dev_base, did),
                "rbln_get_torch_device_id_from_vaddr");
          device_id = static_cast<int64_t>(did);
        }
      }
    }
  }

  // The runtime does no auto-sync on host read, so drain the whole burst
  // before returning.
  if (device_id >= 0) {
    check(::rbln::rbln_device_synchronize(static_cast<uint32_t>(device_id)),
          "rbln_device_synchronize");
  }
}

}  // namespace rbln
}  // namespace lmcache
