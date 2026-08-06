# SPDX-License-Identifier: Apache-2.0
"""vLLM KV cache discovery."""

# mypy: disable-error-code="union-attr"
# Standard
from typing import Optional, cast

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.detectors.base import (
    EngineDetector,
    measure_list_depth_until_tensor,
)
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache, LayoutHints
import lmcache.c_ops as lmc_ops


class VLLM_Detector(EngineDetector):
    engine_type = EngineType.VLLM

    def discover(
        self, kv_caches: DiscoverableKVCache, layout_hints: LayoutHints
    ) -> "tuple[Optional[lmc_ops.EngineKVFormat], DiscoverableKVCache]":
        # vLLM's CPU attention backend stores KV in HND but misreports it, so
        # force HND there; otherwise honor the hint, defaulting to NHD.
        kv_layout = layout_hints.get("kv_layout")
        if torch_device_type == "cpu":
            kv_layout = "HND"
        elif kv_layout is None:
            kv_layout = "NHD"
        is_hnd = kv_layout == "HND"

        # Blocks-first fused K/V is the only rank-4 vLLM layout, so its raw rank
        # identifies it unambiguously (a 5-D split would collide with
        # flash-infer when num_heads == 2). The two middle axes are NH/BS
        # (HND) or BS/NH (NHD) -- indistinguishable from the shape alone, so the
        # resolved kv_layout decides. The tensor is kept raw: the trailing axis
        # is the per-head content size (2 * head_size, K/V packed).
        if (
            isinstance(kv_caches, list)
            and kv_caches
            and isinstance(kv_caches[0], torch.Tensor)
            and kv_caches[0].dim() == 4
        ):
            if is_hnd:
                return lmc_ops.EngineKVFormat.NL_X_NB_NH_BS_CS, kv_caches
            return lmc_ops.EngineKVFormat.NL_X_NB_BS_NH_CS, kv_caches

        list_depth, tensor_ndim, first_tensor = measure_list_depth_until_tensor(
            kv_caches
        )

        if list_depth == 0:
            return lmc_ops.EngineKVFormat.NB_NL_TWO_BS_NH_HS, kv_caches
        if (
            list_depth == 1
            and tensor_ndim == 6
            and torch_device_type == "rbln"
            and isinstance(kv_caches, list)
        ):
            # vLLM-RBLN adds a singleton axis between heads and block tokens.
            # It is always 1, so squeezing it is a free view onto identical
            # bytes and yields exactly NL_X_TWO_NB_NH_BS_HS. Normalizing here
            # (rather than in the connector alone) is what lets the
            # multiprocess path work: its register / gather / scatter helpers
            # resolve layouts through this function, never through a connector.
            # First Party
            from lmcache.v1.platform.rbln.kv_layout import (
                is_rbln_kv_layout,
                squeeze_singleton_axis,
            )

            # ``list_depth == 1`` already established a flat list of tensors.
            layers = cast("list[torch.Tensor]", kv_caches)
            if is_rbln_kv_layout(cast("torch.Tensor", first_tensor)):
                return (
                    lmc_ops.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
                    cast(DiscoverableKVCache, squeeze_singleton_axis(layers)),
                )
        if list_depth == 1 and tensor_ndim == 5:
            if first_tensor.shape[0] == 2:  # K/V axis first
                if is_hnd:
                    return lmc_ops.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS, kv_caches
                return lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS, kv_caches
            if first_tensor.shape[1] == 2:  # num_blocks first
                if is_hnd:
                    return lmc_ops.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS, kv_caches
                return lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS, kv_caches
        if list_depth == 1 and tensor_ndim == 3:  # MLA (or DSA indexer cache)
            if first_tensor.dtype == torch.uint8 and int(first_tensor.shape[-1]) == 132:
                return lmc_ops.EngineKVFormat.NL_X_NB_BSV_BSS, kv_caches
            return lmc_ops.EngineKVFormat.NL_X_NB_BS_HS, kv_caches
        return None, kv_caches
