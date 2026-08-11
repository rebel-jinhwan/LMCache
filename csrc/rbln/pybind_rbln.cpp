// SPDX-License-Identifier: Apache-2.0
//
// Python bindings for the RBLN native ops, exposed as ``lmcache.rbln_ops``
// (mirroring ``lmcache.c_ops`` / ``lmcache.xpu_ops``).
//
// ``direction`` crosses as a plain int: ``lmcache.c_ops`` already registers
// ``TransferDirection``, and pybind11 rejects a second registration of the same
// C++ type in one interpreter.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>

#include "kv_transfer.h"

namespace py = pybind11;

PYBIND11_MODULE(rbln_ops, m) {
  m.doc() = "RBLN native ops for LMCache";

  m.def(
      "head_major_block_kv_transfer",
      [](const std::vector<at::Tensor>& paged_layers,
         const std::vector<at::Tensor>& chunks,
         const std::vector<int64_t>& block_ids, int64_t direction,
         int64_t skip_prefix_n_blocks) {
        lmcache::rbln::head_major_block_kv_transfer(
            paged_layers, chunks, block_ids,
            static_cast<TransferDirection>(direction), skip_prefix_n_blocks);
      },
      py::arg("paged_layers"), py::arg("chunks"), py::arg("block_ids"),
      py::arg("direction"), py::arg("skip_prefix_n_blocks") = 0,
      py::call_guard<py::gil_scoped_release>());
}
