// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

#include "kv_transfer.h"

PYBIND11_MODULE(rbln_ops, m) {
  m.doc() =
      "LMCache RBLN block transfer (token-major, pipelined device transpose)";
  m.def("gather_blocks_to_chunks_hnd",
        &lmcache::rbln::gather_blocks_to_chunks_hnd, py::arg("paged_layers"),
        py::arg("block_ids"), py::arg("chunks"), py::arg("blocks_per_chunk"));
  m.def("scatter_chunks_to_blocks_hnd",
        &lmcache::rbln::scatter_chunks_to_blocks_hnd, py::arg("paged_layers"),
        py::arg("block_ids"), py::arg("chunks"), py::arg("blocks_per_chunk"),
        py::arg("skip_prefix_n_blocks") = 0);
  m.def("staging_swap_mode", &lmcache::rbln::staging_swap_mode,
        "How the head<->token swap is staged: 'inplace' or 'two-buffer' "
        "(LMCACHE_RBLN_STAGING_INPLACE, and whether torch-rbln has the op)");
  m.def("staging_placement", &lmcache::rbln::staging_placement_name,
        "The staging-buffer placement in effect: 'spread', 'spread-all', 'main' "
        "or 'chiplet:<n>' (LMCACHE_RBLN_STAGING_CHIPLET, read once)");
}
