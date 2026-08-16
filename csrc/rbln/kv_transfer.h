// SPDX-License-Identifier: Apache-2.0
//
// RBLN block KV transfer -- the native form of
// ``lmcache.v1.platform.rbln.kv_ops``, whose docstring owns the chunk layout
// and why it is what it is.
//
// It moves whole paged blocks between per-layer HND KV tensors on an RBLN NPU
// and LMCache's canonical token-major chunks (``[2, L, T, H*D]``). RBLN stores
// heads before block tokens, so the two layouts differ by a head<->token
// transpose; a block is contiguous on the device but scattered across token
// rows in the chunk. The transfer therefore lands blocks in a host staging
// buffer shaped like the device -- one rebel runtime DMA per (block, kv,
// layer), no strided device access -- and transposes there, on host memory
// bandwidth. RBLN has no CUDA, so the copies run on the runtime's async
// transfer queue and are drained by a device sync.

#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <vector>

#include "../kv_transfer_types.h"

namespace lmcache {
namespace rbln {

// Transfer whole paged blocks between per-layer RBLN KV caches and token-major
// LMCache chunks.
//
//   paged_layers    per-layer device tensors, contiguous and 5-D
//                   [2, num_blocks, num_kv_heads, block_size, head_size] --
//                   the 6-D vLLM-RBLN cache with its singleton axis squeezed
//   chunks          one contiguous host tensor per chunk, read and written
//                   token-major as [2, num_layers, chunk_tokens, H*D]
//   block_ids       flat paged-block indices, num_chunks * blocks_per_chunk of
//                   them, in chunk-token order
//   direction       D2H (store) or H2D (retrieve)
//   skip_prefix_n_blocks  leading flat blocks neither read nor written
void block_kv_transfer(const std::vector<at::Tensor>& paged_layers,
                       const std::vector<at::Tensor>& chunks,
                       const std::vector<int64_t>& block_ids,
                       TransferDirection direction,
                       int64_t skip_prefix_n_blocks);

}  // namespace rbln
}  // namespace lmcache
