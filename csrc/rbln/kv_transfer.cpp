// SPDX-License-Identifier: Apache-2.0

#include "kv_transfer.h"

#include <ATen/Parallel.h>

#include <algorithm>

#include <cstdint>
#include <cstring>
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
// device sync, so it is not kept.
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

// Host landing buffer for the device<->host leg, per thread. Sized for one
// chunk's blocks and reused across chunks and calls: staging the whole
// transfer at once would double the peak host footprint of a batched store,
// which for a 62-layer chunk is already hundreds of MB.
char* host_staging(int64_t nbytes) {
  thread_local std::vector<char> buffer;
  if (static_cast<int64_t>(buffer.size()) < nbytes) {
    buffer.resize(static_cast<size_t>(nbytes));
  }
  return buffer.data();
}

// Move one chunk's blocks between the staging buffer, laid out like the device
// as [B, L, 2, H, BS, D], and the token-major chunk [2, L, T, H*D].
//
// The innermost run is one head's head_size elements: contiguous on both
// sides. Heads run innermost so the chunk side walks straight through a token
// row, and the parallel split is over (kv, layer) -- 2 * L tasks, each owning
// a disjoint slice of both buffers.
//
// ``first_block`` is the chunk's first live block: both sides are indexed by
// the same block number, so the skipped prefix is simply never visited. The
// chunk pointer stays at the chunk's start -- its token axis sits inside each
// (kv, layer) plane, so a prefix cannot be skipped by moving the base pointer.
void transpose_staging(char* staging, char* chunk, bool staging_to_chunk,
                       int64_t first_block, int64_t n_blocks,
                       int64_t num_layers, int64_t num_kv_heads,
                       int64_t block_size, int64_t head_size,
                       int64_t chunk_tokens, int64_t elem) {
  const int64_t run = head_size * elem;                          // one head
  const int64_t st_head = block_size * head_size;                // [BS, D]
  const int64_t st_kv = num_kv_heads * st_head;                  // [H, BS, D]
  const int64_t st_layer = 2 * st_kv;                            // [2, H, BS, D]
  const int64_t st_block = num_layers * st_layer;                // [L, 2, ...]
  const int64_t ch_token = num_kv_heads * head_size;             // [H*D]
  const int64_t ch_layer = chunk_tokens * ch_token;              // [T, H*D]
  const int64_t ch_kv = num_layers * ch_layer;                   // [L, T, H*D]

  at::parallel_for(0, 2 * num_layers, 1, [&](int64_t begin, int64_t end) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t kv = task / num_layers;
      const int64_t layer = task % num_layers;
      for (int64_t block = first_block; block < first_block + n_blocks;
           ++block) {
        for (int64_t s = 0; s < block_size; ++s) {
          const int64_t token = block * block_size + s;
          char* ch = chunk + (kv * ch_kv + layer * ch_layer + token * ch_token) * elem;
          char* st = staging +
                     (block * st_block + layer * st_layer + kv * st_kv + s * head_size) *
                         elem;
          for (int64_t h = 0; h < num_kv_heads; ++h) {
            char* src = staging_to_chunk ? st + h * st_head * elem : ch + h * run;
            char* dst = staging_to_chunk ? ch + h * run : st + h * st_head * elem;
            std::memcpy(dst, src, static_cast<size_t>(run));
          }
        }
      }
    }
  });
}

}  // namespace

void block_kv_transfer(const std::vector<at::Tensor>& paged_layers,
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
      2 * num_layers * chunk_tokens * num_kv_heads * head_size;
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
                ") is smaller than the token-major chunk size (", chunk_numel,
                ")");
  }
  for (const int64_t b : block_ids) {
    TORCH_CHECK(b >= 0 && b < num_blocks, "block id ", b,
                " is out of range for a buffer with ", num_blocks, " blocks");
  }

  // Element strides of the paged buffer (contiguous [2, NB, NH, BS, HS]).
  const int64_t block_stride = num_kv_heads * block_size * head_size;
  const int64_t kv_stride = num_blocks * block_stride;

  // Staging holds one chunk's blocks, device-shaped: [B, L, 2, H, BS, D].
  const int64_t staging_block_stride = num_layers * 2 * block_stride;
  char* const staging =
      host_staging(blocks_per_chunk * staging_block_stride * elem);

  const bool is_d2h = direction == TransferDirection::D2H;
  int64_t device_id = -1;  // resolved once from the first vaddr, for the sync

  for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    // Blocks of this chunk the caller did not skip. The skip is a flat prefix,
    // so an entire leading chunk can fall away.
    const int64_t chunk_first = chunk_idx * blocks_per_chunk;
    const int64_t first_live =
        std::max(chunk_first, skip_prefix_n_blocks) - chunk_first;
    if (first_live >= blocks_per_chunk) {
      continue;
    }
    const int64_t n_live = blocks_per_chunk - first_live;
    char* const chunk_base = static_cast<char*>(chunks[chunk_idx].data_ptr());

    // H2D reads the chunk, so the transpose has to run before the copies; D2H
    // writes it, so it runs after they are drained.
    if (!is_d2h) {
      transpose_staging(staging, chunk_base, /*staging_to_chunk=*/false,
                        first_live, n_live, num_layers, num_kv_heads,
                        block_size, head_size, chunk_tokens, elem);
    }

    for (int64_t local = first_live; local < blocks_per_chunk; ++local) {
      const int64_t b = block_ids[chunk_first + local];
      for (int64_t kv = 0; kv < 2; ++kv) {
        for (int64_t layer = 0; layer < num_layers; ++layer) {
          const auto dev_base =
              reinterpret_cast<uint64_t>(paged_layers[layer].data_ptr());
          const int64_t dev_off = kv * kv_stride + b * block_stride;
          const int64_t st_off = local * staging_block_stride +
                                 layer * 2 * block_stride + kv * block_stride;
          copy_async(direction, dev_base + static_cast<uint64_t>(dev_off) * elem,
                     reinterpret_cast<uintptr_t>(staging + st_off * elem),
                     static_cast<uint64_t>(block_stride) * elem);
          if (device_id < 0) {
            uint32_t did = 0;
            check(::rbln::rbln_get_torch_device_id_from_vaddr(dev_base, did),
                  "rbln_get_torch_device_id_from_vaddr");
            device_id = static_cast<int64_t>(did);
          }
        }
      }
    }

    // The runtime does no auto-sync on host read, and the staging buffer is
    // reused by the next chunk, so drain before touching it either way.
    if (device_id >= 0) {
      check(::rbln::rbln_device_synchronize(static_cast<uint32_t>(device_id)),
            "rbln_device_synchronize");
    }

    if (is_d2h) {
      transpose_staging(staging, chunk_base, /*staging_to_chunk=*/true,
                        first_live, n_live, num_layers, num_kv_heads,
                        block_size, head_size, chunk_tokens, elem);
    }
  }
}

}  // namespace rbln
}  // namespace lmcache
