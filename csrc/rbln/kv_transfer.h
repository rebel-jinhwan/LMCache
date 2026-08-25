// SPDX-License-Identifier: Apache-2.0
//
// RBLN-native MLA block KV transfer.
//
// vLLM-RBLN's MLA attention backend allocates each layer as a single latent
// plane ``[num_blocks, block_size, head_size]`` -- no K/V split and no head
// axis (``EngineKVFormat::NL_X_NB_BS_HS``). LMCache's canonical MLA chunk is
// ``[L, T, HS]``. With no head axis on either side, one whole block of one
// layer is a contiguous ``block_size * head_size`` run both in the paged cache
// and in the chunk, so the transfer is one DMA per (layer, block) and no
// permute or device-side staging is ever needed.
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

// Transfer whole paged blocks between per-layer 3-D RBLN MLA caches and
// canonical ``[L, T, HS]`` LMCache chunks.
//
//   kv_caches       per-layer device tensors, each contiguous 3-D
//                   [num_blocks, block_size, head_size]
//   lmcache_chunks  one contiguous CPU tensor per chunk; numel must be at
//                   least num_layers * (blocks_per_chunk * block_size) *
//                   head_size (token-major [L, T, HS])
//   block_ids       flat paged-block indices, length == num_chunks *
//                   blocks_per_chunk, in chunk-token order
//   direction       D2H (store / gather) or H2D (retrieve / scatter)
//   skip_prefix_n_blocks  leading flat blocks neither read nor written
//
// Throws (c10::Error) on any geometry violation before issuing a DMA, and on
// any non-success rebel runtime return code.
void block_kv_transfer_mla(std::vector<at::Tensor> kv_caches,
                           std::vector<at::Tensor> lmcache_chunks,
                           std::vector<int64_t> block_ids,
                           TransferDirection direction,
                           int skip_prefix_n_blocks);

}  // namespace rbln
}  // namespace lmcache
