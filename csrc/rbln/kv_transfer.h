// SPDX-License-Identifier: Apache-2.0
//
// RBLN-native block KV transfer. Copies are collected into one descriptor
// list per call and submitted through the rebel runtime's batched, synchronous
// host<->vmemory API (``rbln_memcpy_{v2h,h2v}_multi``), so a store or retrieve
// costs a single dispatch instead of one async request per KV slice.
//
// Two layouts:
//   - HND vLLM-RBLN (``EngineKVFormat::NL_X_TWO_NB_NH_ONE_BS_HS``): paged
//     ``[2, NB, NH, 1, BS, HS]`` <-> canonical token-major ``[2, L, T,
//     NH*HS]``. Each paged block is a contiguous ``NH*BS*HS`` run, so it moves
//     in one DMA; the head<->token permute into the token-major wire layout is
//     a host memcpy after (D2H) or before (H2D) that DMA.
//   - MLA (``EngineKVFormat::NL_X_NB_BS_HS``): paged ``[NB, BS, HS]`` <->
//     canonical ``[L, T, HS]``. No permute; one DMA per (layer, block).

#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <vector>

#include "engine_kv_format.h"
#include "kv_transfer_plan_types.h"
#include "kv_transfer_types.h"

namespace lmcache {
namespace rbln {

// DeviceOps entry, same argument list as the Python method / CUDA op so
// ``bind_native`` can shadow the torch fallback.
void multi_layer_block_kv_transfer(
    std::vector<at::Tensor> kv_caches, std::vector<at::Tensor> lmcache_chunks,
    std::vector<int64_t> block_ids, const torch::Device& device,
    TransferDirection direction, PageBufferShapeDesc shape_desc,
    int lmcache_chunk_size, EngineKVFormat engine_kv_format,
    int skip_prefix_n_blocks);

void block_kv_transfer_hnd(std::vector<at::Tensor> kv_caches,
                           std::vector<at::Tensor> lmcache_chunks,
                           std::vector<int64_t> block_ids,
                           TransferDirection direction,
                           int skip_prefix_n_blocks);

void block_kv_transfer_mla(std::vector<at::Tensor> kv_caches,
                           std::vector<at::Tensor> lmcache_chunks,
                           std::vector<int64_t> block_ids,
                           TransferDirection direction,
                           int skip_prefix_n_blocks);

}  // namespace rbln
}  // namespace lmcache
