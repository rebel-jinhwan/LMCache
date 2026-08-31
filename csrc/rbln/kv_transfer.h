// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <vector>

namespace lmcache::rbln {

// Largest transfer slice staged at once; bounds device memory (two slots per
// direction) and keeps a batched D2D copy under the runtime's descriptor limit.
constexpr int64_t kDefaultMaxStagingBytes = int64_t{128} << 20;

// Gather whole paged blocks into token-major chunks [2, L, T, H*D]: D2D gather
// into device staging, head<->token swap on the device, D2H of each chunk's
// bytes on a copy stream, pipelined over two staging slots. `layers` are
// per-layer HND tensors [2, NB, NH, BS, HS].
void gather_blocks_to_chunks_token_major(
    const std::vector<at::Tensor>& layers,
    const std::vector<int64_t>& block_ids,
    const std::vector<at::Tensor>& chunks, int64_t blocks_per_chunk,
    int64_t max_staging_bytes = kDefaultMaxStagingBytes);

// Mirror of the gather; `skip_prefix_n_blocks` leading blocks are left
// untouched.
void scatter_chunks_to_blocks_token_major(
    const std::vector<at::Tensor>& layers,
    const std::vector<int64_t>& block_ids,
    const std::vector<at::Tensor>& chunks, int64_t blocks_per_chunk,
    int64_t skip_prefix_n_blocks,
    int64_t max_staging_bytes = kDefaultMaxStagingBytes);

}  // namespace lmcache::rbln
