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
}
