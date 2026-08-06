# SPDX-License-Identifier: Apache-2.0
"""Per-layer, HND, K/V-axis first, singleton axis: ``NL x [2, NB, NH, 1, BS, HS]``.

A ``list[NL]`` of a 6-D tensor shaped like
:class:`~lmcache.v1.gpu_connector.kv_format.specs.nl_x_two_nb_nh_bs_hs.NL_X_TWO_NB_NH_BS_HS_Spec`
with one extra singleton axis between heads and block tokens. Produced by RBLN
(Rebellions NPU) non-MLA attention.

Axis 3 is always 1, so this layout is byte- and stride-identical to
``NL_X_TWO_NB_NH_BS_HS``; the accessors below simply skip it. The torch
fallback squeezes it before transferring -- see
``torch_ops._squeeze_singleton_axis``.

Unlike its sibling specs, this module imports :class:`EngineKVFormat` from
:mod:`lmcache.v1.platform.ops_types` rather than through ``lmcache.c_ops``.
``NL_X_TWO_NB_NH_ONE_BS_HS`` exists only in the Python enum, while
``lmcache.c_ops`` resolves to the C++ enum on builds that bind the compiled
extension. Since ``specs/registry.py`` imports every module in this folder on
every device, going through ``lmcache.c_ops`` here would raise ``AttributeError``
at import time on a CUDA build.
"""

# Each spec indexes ``kv_caches`` (Tensor | nested list) per its format, so the
# ``.shape`` / ``[...]`` access is well-defined though mypy cannot prove it.
# mypy: disable-error-code="union-attr,call-overload"
# Standard
from typing import cast

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.specs.base import KVFormatSpec
from lmcache.v1.platform.ops_types import EngineKVFormat


class NL_X_TWO_NB_NH_ONE_BS_HS_Spec(KVFormatSpec):
    engine_kv_format = EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS
    attention_backends = ("RBLN non-MLA attention (HND layout)",)

    def num_layers(self) -> int:
        return len(self.kv_caches)

    def num_blocks(self) -> int:
        return self.kv_caches[0].shape[1]

    def block_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[4]

    def page_buffer_size(self) -> int:
        return self.kv_caches[0].shape[1] * self.kv_caches[0].shape[4]

    def kv_size(self) -> int:
        return 2

    def num_heads(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[2]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        t = self.kv_caches[layer_idx]
        return t.shape[2] * t.shape[5]

    def head_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[5]

    def tokens_per_layer(self) -> int:
        k = self.kv_caches[0][0].shape
        return k[0] * k[3]

    def elements_per_layer(self) -> int:
        return self.kv_caches[0][0].shape.numel() * 2

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches[layer_idx].dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        layers = cast(list[torch.Tensor], self.kv_caches)
        return [layers[i].data_ptr() for i in layer_indices]
