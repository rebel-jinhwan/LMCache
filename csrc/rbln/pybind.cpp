// SPDX-License-Identifier: Apache-2.0
//
// Python bindings for the RBLN-native extension, exposed as
// ``lmcache.rbln_ops`` and bound onto ``RblnDeviceOps`` through
// ``DeviceOps.bind_native``.
//
// ``TransferDirection`` / ``EngineKVFormat`` / ``PageBufferShapeDesc`` are
// not registered here: they live in ``lmcache.lmcache_native``. Functions
// that need enums accept their underlying ``int`` values and cast locally.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "kv_transfer.h"

namespace py = pybind11;

namespace {

std::vector<int64_t> as_block_ids(const py::object& block_ids) {
  if (py::isinstance<at::Tensor>(block_ids)) {
    const at::Tensor tensor =
        block_ids.cast<at::Tensor>().contiguous().to(at::kCPU).to(at::kLong);
    const int64_t* data = tensor.data_ptr<int64_t>();
    return {data, data + tensor.numel()};
  }
  return block_ids.cast<std::vector<int64_t>>();
}

}  // namespace

PYBIND11_MODULE(rbln_ops, m) {
  py::module_::import("lmcache.lmcache_native");
  m.doc() = "RBLN-native block KV transfer for LMCache";

  m.def(
      "multi_layer_block_kv_transfer",
      [](std::vector<at::Tensor> kv_caches,
         std::vector<at::Tensor> lmcache_chunks, const py::object& block_ids,
         const torch::Device& device, int direction,
         PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
         int engine_kv_format, int skip_prefix_n_blocks) {
        std::vector<int64_t> ids = as_block_ids(block_ids);
        py::gil_scoped_release release;
        lmcache::rbln::multi_layer_block_kv_transfer(
            std::move(kv_caches), std::move(lmcache_chunks), std::move(ids),
            device, static_cast<TransferDirection>(direction), shape_desc,
            lmcache_chunk_size, static_cast<EngineKVFormat>(engine_kv_format),
            skip_prefix_n_blocks);
      },
      py::arg("paged_buffer_ptrs_tensor"), py::arg("lmcache_objects_ptrs"),
      py::arg("block_ids"), py::arg("device"), py::arg("direction"),
      py::arg("shape_desc"), py::arg("lmcache_chunk_size"),
      py::arg("engine_kv_format"), py::arg("skip_prefix_n_blocks"),
      R"doc(DeviceOps entry: move whole paged blocks between RBLN KV and chunks.

Accepts the ``DeviceOps.multi_layer_block_kv_transfer`` argument list with
tensor operands and dispatches on ``engine_kv_format``: the HND vLLM-RBLN
layout (``[2, NB, NH, 1, BS, HS]``, token-major ``[2, L, T, NH*HS]`` chunks)
and the MLA layout (``[NB, BS, HS]``, ``[L, T, HS]`` chunks).
``RblnDeviceOps.ensure_native`` binds this over the torch fallback.
)doc");

  m.def(
      "block_kv_transfer_mla",
      [](std::vector<at::Tensor> kv_caches,
         std::vector<at::Tensor> lmcache_chunks, const py::object& block_ids,
         int direction, int skip_prefix_n_blocks) {
        std::vector<int64_t> ids = as_block_ids(block_ids);
        py::gil_scoped_release release;
        lmcache::rbln::block_kv_transfer_mla(
            std::move(kv_caches), std::move(lmcache_chunks), std::move(ids),
            static_cast<TransferDirection>(direction), skip_prefix_n_blocks);
      },
      py::arg("kv_caches"), py::arg("lmcache_chunks"), py::arg("block_ids"),
      py::arg("direction"), py::arg("skip_prefix_n_blocks") = 0,
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
