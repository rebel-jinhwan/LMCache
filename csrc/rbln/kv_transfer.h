// SPDX-License-Identifier: Apache-2.0
//
// RBLN-native head-major block KV transfer.
//
// Moves whole paged blocks between the native vLLM-RBLN 6-D KV cache
// (``[2, num_blocks, num_kv_heads, 1, block_size, head_dim]``, HND) and a
// head-major LMCache chunk (``[2, L, H, T, D]``). Because both sides are
// head-major, a (K|V, layer, block, head) slab is contiguous on both ends and
// moves as one DMA -- no head<->token permute is ever issued, which is the
// host-side cost that dominates the pure-torch token-major path in
// ``lmcache/v1/platform/rbln/kv_ops.py``.
//
// Copies run on the rebel runtime's async transfer queue
// (``rbln_memcpy_{v2h,h2v}_async``) and are drained with one device
// synchronize before returning; RBLN has no stream / event objects, so the
// completed transfer is the only ordering primitive available.

#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <vector>

#include "kv_transfer_types.h"  // TransferDirection, shared by every backend

namespace lmcache {
namespace rbln {

// Transfer whole paged blocks between per-layer 6-D RBLN KV caches and
// head-major LMCache chunks.
//
//   kv_caches       per-layer device tensors, each contiguous 6-D
//                   [2, num_blocks, num_kv_heads, 1, block_size, head_dim]
//   lmcache_chunks  one contiguous CPU tensor per chunk; numel must be at
//                   least 2 * num_layers * num_kv_heads * (blocks_per_chunk *
//                   block_size) * head_dim (head-major [2, L, H, T, D])
//   block_ids       flat paged-block indices, length == num_chunks *
//                   blocks_per_chunk, in chunk-token order
//   direction       D2H (store / gather) or H2D (retrieve / scatter)
//   skip_prefix_n_blocks  leading flat blocks neither read nor written
//
// Throws (c10::Error) on any geometry violation before issuing a DMA, and on
// any non-success rebel runtime return code.
void block_kv_transfer_head_major(std::vector<at::Tensor> kv_caches,
                                  std::vector<at::Tensor> lmcache_chunks,
                                  std::vector<int64_t> block_ids,
                                  TransferDirection direction,
                                  int skip_prefix_n_blocks);

}  // namespace rbln
}  // namespace lmcache
