# SPDX-License-Identifier: Apache-2.0
"""Tests for RBLN native-layout normalization in the vLLM format detector.

The connector can squeeze the singleton axis itself, but the multiprocess path
never goes through a connector: ``compute_kv_layout``, ``gather_paged_kv_to_cpu``
and ``scatter_cpu_to_paged_kv`` all resolve layouts through
``normalize_kv_and_discover_format``. Normalizing in the detector is what makes
both paths work off the same rule.

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
from lmcache.v1.gpu_connector.utils import normalize_kv_and_discover_format
from lmcache.v1.platform.ops_types import EngineKVFormat
from lmcache.v1.platform.rbln.kv_layout import (
    is_rbln_kv_layout,
    squeeze_singleton_axis,
)

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
    device_type: str = "rbln",
    layout_hints: "dict[str, str] | None" = None,
):
    """Run discovery with ``torch_device_type`` forced to ``device_type``."""
    with patch.object(detectors.vllm, "torch_device_type", device_type):
        return normalize_kv_and_discover_format(
            kv_caches, EngineType.VLLM, layout_hints=layout_hints
        )


# ---------------------------------------------------------------------------
# Layout predicate
# ---------------------------------------------------------------------------


def test_recognizes_the_native_layout() -> None:
    """Only 6-D, K/V-first, singleton-at-3 qualifies."""
    assert is_rbln_kv_layout(_native_kv()[0]) is True


@pytest.mark.parametrize(
    "shape",
    [
        (2, NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_SIZE),
        (2, NUM_BLOCKS, NUM_HEADS, 2, BLOCK_SIZE, HEAD_SIZE),
        (NUM_BLOCKS, 2, NUM_HEADS, 1, BLOCK_SIZE, HEAD_SIZE),
    ],
    ids=["5d", "non-singleton-axis", "blocks-first"],
)
def test_rejects_other_layouts(shape: tuple[int, ...]) -> None:
    """Anything else is not the RBLN layout."""
    assert is_rbln_kv_layout(torch.zeros(shape)) is False


def test_squeeze_is_a_free_view() -> None:
    """Normalization must not copy the KV cache."""
    native = _native_kv()
    for view, tensor in zip(squeeze_singleton_axis(native), native, strict=True):
        assert view.data_ptr() == tensor.data_ptr()
        assert tuple(view.shape) == (
            2,
            NUM_BLOCKS,
            NUM_HEADS,
            BLOCK_SIZE,
            HEAD_SIZE,
        )


def test_squeeze_rejects_a_foreign_layout() -> None:
    """A mismatched rank fails loudly instead of mis-transferring."""
    with pytest.raises(ValueError, match=r"\[2, NB, NH, 1, BS, HS\]"):
        squeeze_singleton_axis([torch.zeros(2, NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE)])


# ---------------------------------------------------------------------------
# Detector integration
# ---------------------------------------------------------------------------


def test_detector_normalizes_the_native_layout_on_rbln() -> None:
    """6-D input resolves to the registered HND format, squeezed."""
    fmt, normalized = _discover(_native_kv())
    assert int(fmt) == int(EngineKVFormat.NL_X_TWO_NB_NH_BS_HS)
    assert [t.ndim for t in normalized] == [5] * NUM_LAYERS


def test_detector_leaves_other_devices_alone() -> None:
    """The branch is device-scoped: elsewhere a 6-D cache stays unsupported.

    Guards against the normalization firing for some future engine that ships a
    different 6-D layout on another accelerator.
    """
    with pytest.raises(ValueError, match="unsupported kv_caches structure"):
        _discover(_native_kv(), device_type="cuda")


def test_squeezed_input_still_resolves() -> None:
    """Pre-squeezed 5-D input takes the ordinary path, unchanged."""
    views = squeeze_singleton_axis(_native_kv())
    fmt, normalized = _discover(views, layout_hints={"kv_layout": "HND"})
    assert int(fmt) == int(EngineKVFormat.NL_X_TWO_NB_NH_BS_HS)
    assert [t.ndim for t in normalized] == [5] * NUM_LAYERS
