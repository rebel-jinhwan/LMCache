// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

#include "kv_transfer.h"

PYBIND11_MODULE(rbln_ops, m) {
  m.doc() = "LMCache RBLN block transfer (device-staged, torch-rbln copies)";
  m.def("gather_blocks_to_chunks_hnd",
        &lmcache::rbln::gather_blocks_to_chunks_hnd, py::arg("paged_layers"),
        py::arg("block_ids"), py::arg("chunks"), py::arg("blocks_per_chunk"));
  m.def("scatter_chunks_to_blocks_hnd",
        &lmcache::rbln::scatter_chunks_to_blocks_hnd, py::arg("paged_layers"),
        py::arg("block_ids"), py::arg("chunks"), py::arg("blocks_per_chunk"),
        py::arg("skip_prefix_n_blocks") = 0);
  m.def("gather_blocks_to_chunks_mla",
        &lmcache::rbln::gather_blocks_to_chunks_mla, py::arg("paged_layers"),
        py::arg("block_ids"), py::arg("chunks"), py::arg("blocks_per_chunk"));
  m.def("scatter_chunks_to_blocks_mla",
        &lmcache::rbln::scatter_chunks_to_blocks_mla, py::arg("paged_layers"),
        py::arg("block_ids"), py::arg("chunks"), py::arg("blocks_per_chunk"),
        py::arg("skip_prefix_n_blocks") = 0);
  m.def("staging_swap_mode", &lmcache::rbln::staging_swap_mode,
        "How the head<->token swap is staged: 'inplace' or 'two-buffer' "
        "(LMCACHE_RBLN_STAGING_INPLACE, and whether torch-rbln has the op)");
  m.def("staging_split_groups", &lmcache::rbln::staging_split_groups,
        py::arg("any_device_tensor"),
        "How many layer groups the HND staging is split into, one per chiplet "
        "(LMCACHE_RBLN_STAGING_SPLIT)");
}
