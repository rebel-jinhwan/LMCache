// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <vector>

namespace lmcache::rbln {

// Gather whole paged blocks into token-major chunks [2, L, T, H*D]: D2D gather
// into device staging, head<->token swap on the device, D2H of each chunk's
// bytes, one block at a time. `layers` are per-layer HND tensors
// [2, NB, NH, BS, HS].
void gather_blocks_to_chunks_hnd(const std::vector<at::Tensor>& layers,
                                 const std::vector<int64_t>& block_ids,
                                 const std::vector<at::Tensor>& chunks,
                                 int64_t blocks_per_chunk);

// Mirror of the gather; `skip_prefix_n_blocks` leading blocks are left
// untouched.
void scatter_chunks_to_blocks_hnd(const std::vector<at::Tensor>& layers,
                                  const std::vector<int64_t>& block_ids,
                                  const std::vector<at::Tensor>& chunks,
                                  int64_t blocks_per_chunk,
                                  int64_t skip_prefix_n_blocks);

// Gather whole MLA paged blocks into single-plane chunks [L, T, HS]: one D2D
// batch of the chunk's blocks into device staging, then one D2H of the
// chunk's bytes, one chunk at a time. `layers` are per-layer contiguous
// [NB, BS, HS] tensors.
void gather_blocks_to_chunks_mla(const std::vector<at::Tensor>& layers,
                                 const std::vector<int64_t>& block_ids,
                                 const std::vector<at::Tensor>& chunks,
                                 int64_t blocks_per_chunk);

// Mirror of the gather; `skip_prefix_n_blocks` leading blocks are left
// untouched.
void scatter_chunks_to_blocks_mla(const std::vector<at::Tensor>& layers,
                                  const std::vector<int64_t>& block_ids,
                                  const std::vector<at::Tensor>& chunks,
                                  int64_t blocks_per_chunk,
                                  int64_t skip_prefix_n_blocks);

}  // namespace lmcache::rbln
