// SPDX-License-Identifier: Apache-2.0
//
// Python bindings for the RBLN-native extension, exposed as
// ``lmcache.rbln_ops`` and bound onto ``RblnDeviceOps`` through
// ``DeviceOps.bind_native``.
//
// ``TransferDirection`` is deliberately *not* registered here: pybind11 keys
// enum registrations by C++ type, and ``lmcache.lmcache_native`` already owns
// that one. Importing ``lmcache_native`` first (which ``RblnDeviceOps`` does)
// lets this module accept the shared ``lmcache_native.TransferDirection``.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kv_transfer.h"

namespace py = pybind11;

PYBIND11_MODULE(rbln_ops, m) {
  m.doc() = "RBLN-native block KV transfer kernels for LMCache";

  m.def(
      "block_kv_transfer_mla", &lmcache::rbln::block_kv_transfer_mla,
      py::arg("kv_caches"), py::arg("lmcache_chunks"), py::arg("block_ids"),
      py::arg("direction"), py::arg("skip_prefix_n_blocks") = 0,
      py::call_guard<py::gil_scoped_release>(),
      R"doc(Move whole paged blocks between RBLN MLA caches and [L, T, HS] chunks.

Args:
    kv_caches: Per-layer contiguous RBLN tensors ``[NB, BS, HS]``
        (``EngineKVFormat.NL_X_NB_BS_HS``).
    lmcache_chunks: Contiguous host tensors, one per chunk, each holding a
        token-major ``[L, T, HS]`` view (``T = blocks_per_chunk * BS``).
    block_ids: Flat paged-block ids, ``len == num_chunks * blocks_per_chunk``.
    direction: ``TransferDirection.D2H`` (store) or ``H2D`` (retrieve).
    skip_prefix_n_blocks: Leading flat blocks neither read nor written.

Raises:
    RuntimeError: On a geometry mismatch or a rebel runtime failure.
)doc");
}
