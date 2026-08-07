# SPDX-License-Identifier: Apache-2.0
"""Tests for degenerate-axis removal in the vLLM format detector.

Some attention backends allocate a per-layer KV tensor with an extra axis of
length 1 between the head and block-token axes (vLLM-RBLN does). Squeezing it
is metadata-only, and deliberately does *not* decide the format: the squeezed
cache falls through to the ordinary rank-5 classification, so HND vs NHD is
still resolved from ``layout_hints`` exactly as it is for a natively 5-D cache.

``torch_device_type`` is patched to a device with no layout override of its
own, so these cover the engine-agnostic behaviour rather than any one
accelerator's.
"""

# Standard
from unittest.mock import patch

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format import detectors
from lmcache.v1.gpu_connector.kv_format.singleton_axis import (
    has_singleton_kv_axis,
    squeeze_singleton_kv_axis,
)
from lmcache.v1.gpu_connector.kv_format.types import LayoutHints
from lmcache.v1.gpu_connector.utils import normalize_kv_and_discover_format
from lmcache.v1.platform.ops_types import EngineKVFormat

NUM_LAYERS = 2
NUM_BLOCKS = 8
NUM_HEADS = 2
BLOCK_SIZE = 4
HEAD_SIZE = 16


def _kv_with_singleton_axis() -> list[torch.Tensor]:
    """Per-layer tensors carrying the degenerate axis at position 3."""
    torch.manual_seed(3)
    shape = (2, NUM_BLOCKS, NUM_HEADS, 1, BLOCK_SIZE, HEAD_SIZE)
    return [torch.randn(shape) for _ in range(NUM_LAYERS)]


def _discover(
    kv_caches: list[torch.Tensor],
    layout_hints: "LayoutHints | None" = None,
):
    """Run discovery on a device that applies no layout override of its own."""
    with patch.object(detectors.vllm, "torch_device_type", "cuda"):
        return normalize_kv_and_discover_format(
            kv_caches, EngineType.VLLM, layout_hints=layout_hints
        )


# ---------------------------------------------------------------------------
# Predicate
# ---------------------------------------------------------------------------


def test_recognizes_a_squeezable_layout() -> None:
    """Only 6-D, K/V-first, singleton-at-3 qualifies."""
    assert has_singleton_kv_axis(_kv_with_singleton_axis()[0]) is True


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
    """Anything else is left for the ordinary branches to classify."""
    assert has_singleton_kv_axis(torch.zeros(shape)) is False


def test_squeeze_is_a_free_view() -> None:
    """Normalization must not copy the KV cache."""
    native = _kv_with_singleton_axis()
    for view, tensor in zip(squeeze_singleton_kv_axis(native), native, strict=True):
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
    with pytest.raises(ValueError, match=r"\[2, NB, X, 1, Y, HS\]"):
        squeeze_singleton_kv_axis([torch.zeros(2, NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE)])


# ---------------------------------------------------------------------------
# Detector integration
# ---------------------------------------------------------------------------


def test_squeezed_cache_is_classified_by_the_layout_hint() -> None:
    """The squeeze defers HND/NHD to the ordinary rank-5 branch.

    Squeezing a size-1 axis cannot reorder the head and block-token axes, so
    the resulting 5-D cache is exactly as ambiguous as a natively 5-D one and
    is resolved the same way.
    """
    hnd_fmt, hnd_normalized = _discover(
        _kv_with_singleton_axis(), layout_hints={"kv_layout": "HND"}
    )
    assert int(hnd_fmt) == int(EngineKVFormat.NL_X_TWO_NB_NH_BS_HS)
    assert [t.ndim for t in hnd_normalized] == [5] * NUM_LAYERS

    nhd_fmt, _ = _discover(_kv_with_singleton_axis(), layout_hints={"kv_layout": "NHD"})
    assert int(nhd_fmt) == int(EngineKVFormat.NL_X_TWO_NB_BS_NH_HS)


def test_unsqueezable_six_dim_cache_stays_unsupported() -> None:
    """A 6-D cache without the degenerate axis is not silently reinterpreted."""
    dense = [torch.zeros(2, NUM_BLOCKS, NUM_HEADS, 2, BLOCK_SIZE, HEAD_SIZE)]
    with pytest.raises(ValueError, match="unsupported kv_caches structure"):
        _discover(dense, layout_hints={"kv_layout": "HND"})
