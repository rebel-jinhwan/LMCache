# SPDX-License-Identifier: Apache-2.0
"""Tests for the RBLN KV layout ``NL_X_TWO_NB_NH_ONE_BS_HS``.

The format declares RBLN's 6-D per-layer buffer
``[2, NB, NH, 1, BS, HS]``. Axis 3 is always 1, so the layout is byte- and
stride-identical to the registered ``NL_X_TWO_NB_NH_BS_HS``; the torch fallback
squeezes it and reuses that transfer path.

The load-bearing test here is the equivalence round-trip: the same data pushed
through both formats must produce identical staging chunks and restore
identically. Everything else could pass while the squeeze wiring silently
transferred the wrong slots.

Runs on CPU -- no RBLN hardware needed.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.specs import registry
from lmcache.v1.gpu_connector.kv_format.specs.nl_x_two_nb_nh_one_bs_hs import (
    NL_X_TWO_NB_NH_ONE_BS_HS_Spec,
)
from lmcache.v1.platform import torch_ops
from lmcache.v1.platform.ops_types import EngineKVFormat, TransferDirection

NUM_LAYERS = 2
NUM_BLOCKS = 8
BLOCK_SIZE = 4
NUM_HEADS = 2
HEAD_SIZE = 8
BLOCKS_PER_CHUNK = 4
CHUNK_TOKENS = BLOCKS_PER_CHUNK * BLOCK_SIZE
HIDDEN_DIM = NUM_HEADS * HEAD_SIZE
DTYPE = torch.float32


def _shape_desc() -> "torch_ops.PageBufferShapeDesc":
    """Build the descriptor shared by both formats under test."""
    desc = torch_ops.PageBufferShapeDesc()
    desc.nl = NUM_LAYERS
    desc.nb = NUM_BLOCKS
    desc.bs = BLOCK_SIZE
    desc.nh = NUM_HEADS
    desc.hs = HEAD_SIZE
    desc.element_size = DTYPE.itemsize
    desc.kv_size = 2
    return desc


def _paged_layers_5d() -> list[torch.Tensor]:
    """Per-layer ``NL_X_TWO_NB_NH_BS_HS`` buffers with deterministic content."""
    torch.manual_seed(456)
    return [
        torch.randn(2, NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_SIZE, dtype=DTYPE)
        for _ in range(NUM_LAYERS)
    ]


def _transfer(
    layers: list[torch.Tensor],
    chunks: list[torch.Tensor],
    fmt: EngineKVFormat,
    direction: TransferDirection,
) -> None:
    """Run the torch fallback block transfer over whole blocks."""
    torch_ops.multi_layer_block_kv_transfer(
        layers,
        chunks,
        torch.tensor(list(range(NUM_BLOCKS)), dtype=torch.int64),
        torch.device("cpu"),
        direction,
        _shape_desc(),
        CHUNK_TOKENS,
        fmt,
        0,
    )


def _empty_chunks() -> list[torch.Tensor]:
    """Staging chunks sized as the transfer path expects."""
    return [
        torch.zeros((2, NUM_LAYERS, CHUNK_TOKENS, HIDDEN_DIM), dtype=DTYPE)
        for _ in range(NUM_BLOCKS // BLOCKS_PER_CHUNK)
    ]


# ---------------------------------------------------------------------------
# Equivalence with NL_X_TWO_NB_NH_BS_HS
# ---------------------------------------------------------------------------


def test_gather_matches_the_squeezed_hnd_format() -> None:
    """D2H through the 6-D format produces byte-identical staging chunks."""
    layers_5d = _paged_layers_5d()
    layers_6d = [layer.unsqueeze(3).contiguous() for layer in layers_5d]

    chunks_5d = _empty_chunks()
    chunks_6d = _empty_chunks()
    _transfer(
        layers_5d, chunks_5d, EngineKVFormat.NL_X_TWO_NB_NH_BS_HS, TransferDirection.D2H
    )
    _transfer(
        layers_6d,
        chunks_6d,
        EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS,
        TransferDirection.D2H,
    )

    for got, expected in zip(chunks_6d, chunks_5d, strict=True):
        assert torch.equal(got, expected)


def test_scatter_round_trips_through_the_6d_format() -> None:
    """H2D restores the original 6-D buffers exactly."""
    layers_6d = [layer.unsqueeze(3).contiguous() for layer in _paged_layers_5d()]
    chunks = _empty_chunks()
    _transfer(
        layers_6d,
        chunks,
        EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS,
        TransferDirection.D2H,
    )

    restored = [torch.zeros_like(layer) for layer in layers_6d]
    _transfer(
        restored, chunks, EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS, TransferDirection.H2D
    )

    for got, expected in zip(restored, layers_6d, strict=True):
        assert torch.equal(got, expected)


# ---------------------------------------------------------------------------
# Format classification
# ---------------------------------------------------------------------------


def test_classified_as_a_per_layer_hnd_format() -> None:
    """The new format joins the layer-list and HND families."""
    fmt = EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS
    assert torch_ops.is_layer_list(fmt) is True
    assert torch_ops.is_cross_layer(fmt) is False
    assert torch_ops.is_kv_list(fmt) is False
    assert torch_ops.is_mla(fmt) is False


def test_per_layer_paged_shape_keeps_the_singleton_axis() -> None:
    """Pointer reconstruction uses the engine's real 6-D rank."""
    shape = torch_ops._per_layer_paged_shape(
        EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS,
        NUM_BLOCKS,
        BLOCK_SIZE,
        NUM_HEADS,
        HEAD_SIZE,
    )
    assert shape == (2, NUM_BLOCKS, NUM_HEADS, 1, BLOCK_SIZE, HEAD_SIZE)


def test_slot_mapping_transfer_is_rejected() -> None:
    """``multi_layer_kv_transfer`` refuses HND layouts, including this one."""
    with pytest.raises(NotImplementedError, match="NL_X_TWO_NB_NH_ONE_BS_HS"):
        torch_ops.multi_layer_kv_transfer(
            torch.zeros(1),
            [torch.zeros(1)],
            torch.zeros(1, dtype=torch.long),
            torch.device("cpu"),
            NUM_BLOCKS * BLOCK_SIZE,
            TransferDirection.D2H,
            EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS,
        )


# ---------------------------------------------------------------------------
# Spec geometry and discovery
# ---------------------------------------------------------------------------


def test_spec_is_discovered_by_the_registry() -> None:
    """Dropping the file in ``specs/`` is enough to register it."""
    assert (
        registry.get_spec_class(EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS)
        is NL_X_TWO_NB_NH_ONE_BS_HS_Spec
    )


def test_spec_reads_geometry_past_the_singleton_axis() -> None:
    """Accessors skip axis 3 and report the same geometry as the 5-D sibling."""
    layers = [layer.unsqueeze(3).contiguous() for layer in _paged_layers_5d()]
    spec = NL_X_TWO_NB_NH_ONE_BS_HS_Spec(layers)
    assert spec.num_layers() == NUM_LAYERS
    assert spec.num_blocks() == NUM_BLOCKS
    assert spec.block_size() == BLOCK_SIZE
    assert spec.num_heads() == NUM_HEADS
    assert spec.head_size() == HEAD_SIZE
    assert spec.hidden_dim() == HIDDEN_DIM
    assert spec.kv_size() == 2
    assert spec.tokens_per_layer() == NUM_BLOCKS * BLOCK_SIZE
    assert spec.page_buffer_size() == NUM_BLOCKS * BLOCK_SIZE
    assert spec.dtype() == DTYPE


def test_spec_geometry_matches_the_5d_sibling() -> None:
    """Every accessor agrees with ``NL_X_TWO_NB_NH_BS_HS`` on the same data."""
    layers_5d = _paged_layers_5d()
    sibling = registry.get_spec_class(EngineKVFormat.NL_X_TWO_NB_NH_BS_HS)(layers_5d)
    spec = NL_X_TWO_NB_NH_ONE_BS_HS_Spec(
        [layer.unsqueeze(3).contiguous() for layer in layers_5d]
    )
    for accessor in (
        "num_layers",
        "num_blocks",
        "block_size",
        "page_buffer_size",
        "kv_size",
        "num_heads",
        "hidden_dim",
        "head_size",
        "tokens_per_layer",
        "elements_per_layer",
    ):
        assert getattr(spec, accessor)() == getattr(sibling, accessor)(), accessor


# ---------------------------------------------------------------------------
# Squeeze guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (2, NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_SIZE),
        (2, NUM_BLOCKS, NUM_HEADS, 2, BLOCK_SIZE, HEAD_SIZE),
    ],
    ids=["5d", "non-singleton-axis"],
)
def test_squeeze_rejects_a_layout_that_is_not_6d_with_a_singleton(
    shape: tuple[int, ...],
) -> None:
    """A mismatched rank fails loudly instead of transferring wrong slots."""
    with pytest.raises(ValueError, match="NL_X_TWO_NB_NH_ONE_BS_HS"):
        torch_ops._squeeze_singleton_axis(
            EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS, [torch.zeros(shape)]
        )


def test_squeeze_leaves_other_formats_untouched() -> None:
    """Only the RBLN format is rewritten; identity is preserved elsewhere."""
    layers = _paged_layers_5d()
    assert (
        torch_ops._squeeze_singleton_axis(EngineKVFormat.NL_X_TWO_NB_NH_BS_HS, layers)
        is layers
    )
