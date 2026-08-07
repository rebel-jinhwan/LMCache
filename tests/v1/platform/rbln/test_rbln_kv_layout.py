# SPDX-License-Identifier: Apache-2.0
"""Tests for resolving the native RBLN KV layout end to end.

The squeeze that removes vLLM-RBLN's degenerate axis is engine-agnostic and is
covered by ``tests/v1/gpu_connector/test_singleton_axis.py``. What is
RBLN-specific is the *layout* the squeezed cache resolves to: vLLM-RBLN stores
HND but never reports a KV layout, so the detector forces HND on this device
the same way it already does for vLLM's CPU attention backend. Without that,
discovery would fall back to the NHD default and classify the cache as
``NL_X_TWO_NB_BS_NH_HS`` -- the wrong axis order for every transfer.

The connector can squeeze the caches itself, but the multiprocess path never
goes through a connector: ``compute_kv_layout``, ``gather_paged_kv_to_cpu`` and
``scatter_cpu_to_paged_kv`` all resolve layouts through
``normalize_kv_and_discover_format``, so it is discovery that has to get this
right.

``torch_device_type`` is patched rather than requiring an NPU, so these run
anywhere.
"""

# Standard
from unittest.mock import patch

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format import detectors
from lmcache.v1.gpu_connector.kv_format.singleton_axis import squeeze_singleton_kv_axis
from lmcache.v1.gpu_connector.kv_format.types import LayoutHints
from lmcache.v1.gpu_connector.utils import normalize_kv_and_discover_format
from lmcache.v1.platform.ops_types import EngineKVFormat

NUM_LAYERS = 2
NUM_BLOCKS = 8
NUM_HEADS = 2
BLOCK_SIZE = 4
HEAD_SIZE = 16


def _native_kv() -> list[torch.Tensor]:
    """Per-layer tensors in the native RBLN 6-D layout."""
    torch.manual_seed(3)
    shape = (2, NUM_BLOCKS, NUM_HEADS, 1, BLOCK_SIZE, HEAD_SIZE)
    return [torch.randn(shape) for _ in range(NUM_LAYERS)]


def _discover(
    kv_caches: list[torch.Tensor],
    layout_hints: "LayoutHints | None" = None,
):
    """Run discovery with ``torch_device_type`` forced to ``rbln``."""
    with patch.object(detectors.vllm, "torch_device_type", "rbln"):
        return normalize_kv_and_discover_format(
            kv_caches, EngineType.VLLM, layout_hints=layout_hints
        )


def test_native_layout_resolves_to_the_registered_hnd_format() -> None:
    """6-D input resolves to the registered HND format, squeezed."""
    fmt, normalized = _discover(_native_kv())
    assert int(fmt) == int(EngineKVFormat.NL_X_TWO_NB_NH_BS_HS)
    assert [t.ndim for t in normalized] == [5] * NUM_LAYERS


@pytest.mark.parametrize("hints", [None, {"kv_layout": "NHD"}], ids=["absent", "nhd"])
def test_hnd_is_forced_regardless_of_the_reported_layout(
    hints: "LayoutHints | None",
) -> None:
    """vLLM-RBLN stores HND but does not report it, so the hint cannot be trusted.

    ``get_kv_cache_layout()`` is unset on vLLM-RBLN and defaults to NHD, so
    honouring it would silently pick the wrong axis order.
    """
    fmt, _ = _discover(_native_kv(), layout_hints=hints)
    assert int(fmt) == int(EngineKVFormat.NL_X_TWO_NB_NH_BS_HS)


def test_squeezed_input_still_resolves() -> None:
    """Pre-squeezed 5-D input takes the ordinary path, unchanged.

    This is the in-process connector's path: it squeezes the caches itself for
    its own slot indexing and hands the 5-D views to discovery.
    """
    views = squeeze_singleton_kv_axis(_native_kv())
    fmt, normalized = _discover(views)
    assert int(fmt) == int(EngineKVFormat.NL_X_TWO_NB_NH_BS_HS)
    assert [t.ndim for t in normalized] == [5] * NUM_LAYERS
