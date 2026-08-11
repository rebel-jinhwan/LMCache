// SPDX-License-Identifier: Apache-2.0
//
// RBLN head-major block KV transfer -- the native form of
// ``lmcache.v1.platform.rbln.kv_ops``, whose docstring owns the head-major
// chunk layout and why it is sound.
//
// It moves whole paged blocks between per-layer HND KV tensors on an RBLN NPU
// and head-major LMCache chunks (``[2, L, H, T, D]``), issuing one rebel
// runtime DMA per contiguous run instead of building torch views. RBLN has no
// CUDA, so the copies run on the runtime's async transfer queue and are drained
// by a single device sync.

#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <vector>

#include "../kv_transfer_types.h"

namespace lmcache {
namespace rbln {

// Transfer whole paged blocks between per-layer RBLN KV caches and head-major
// LMCache chunks.
//
//   paged_layers    per-layer device tensors, contiguous and 5-D
//                   [2, num_blocks, num_kv_heads, block_size, head_size] --
//                   the 6-D vLLM-RBLN cache with its singleton axis squeezed
//   chunks          one contiguous host tensor per chunk, read head-major
//   block_ids       flat paged-block indices, num_chunks * blocks_per_chunk of
//                   them, in chunk-token order
//   direction       D2H (store) or H2D (retrieve)
//   skip_prefix_n_blocks  leading flat blocks neither read nor written
void head_major_block_kv_transfer(const std::vector<at::Tensor>& paged_layers,
                                  const std::vector<at::Tensor>& chunks,
                                  const std::vector<int64_t>& block_ids,
                                  TransferDirection direction,
                                  int64_t skip_prefix_n_blocks);

}  // namespace rbln
}  // namespace lmcache
