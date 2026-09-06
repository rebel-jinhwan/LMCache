# SPDX-License-Identifier: Apache-2.0
"""Tests for the RBLN HND block transfer (``lmcache.rbln_ops``).

The guards run everywhere: they raise in ``RblnDeviceOps`` before the
extension is reached (operand shape, format, chunk-size divisibility). The
transfer tests need the extension built (``BUILD_WITH_RBLN=1``) and are
skipped otherwise. It is plain ATen, so CPU tensors exercise the real kernel;
what they cannot cover is the transfer cost and torch-rbln's copy dispatch,
which ``bench_kv_transfer_mp.py --verify`` checks on hardware.

The load-bearing test is :func:`test_chunk_matches_the_canonical_torch_path`:
a round trip alone passes under any self-consistent layout, because the same
code writes and reads the chunk. Only byte equality against the shared torch
path proves an RBLN chunk is interchangeable with one written by another
device -- the property cross-device sharing and PD disaggregation rely on.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
from lmcache.v1.platform.rbln import device_ops
from lmcache.v1.platform.rbln.device_ops import RblnDeviceOps
from lmcache.v1.platform.rbln.kv_layout import squeeze_singleton_axis
from lmcache.v1.platform.torch_ops import multi_layer_block_kv_transfer
import lmcache.lmcache_native as lmcache_native

EngineKVFormat = lmcache_native.EngineKVFormat
TransferDirection = lmcache_native.TransferDirection

NUM_LAYERS = 2
NUM_BLOCKS = 8
NUM_HEADS = 2
BLOCK_SIZE = 4
HEAD_SIZE = 8
BLOCKS_PER_CHUNK = 2
CHUNK_TOKENS = BLOCKS_PER_CHUNK * BLOCK_SIZE
DTYPE = torch.float32


def _paged_layers(fill_random: bool = False) -> list[torch.Tensor]:
    """Per-layer HND KV in the native 6-D layout the detector reports."""
    torch.manual_seed(11)
    shape = (2, NUM_BLOCKS, NUM_HEADS, 1, BLOCK_SIZE, HEAD_SIZE)
    factory = torch.randn if fill_random else torch.zeros
    return [factory(shape, dtype=DTYPE) for _ in range(NUM_LAYERS)]


def _chunks(blocks_per_chunk: int = BLOCKS_PER_CHUNK) -> list[torch.Tensor]:
    """Staging chunks sized token-major, as upstream allocates them."""
    return [
        torch.zeros(
            (2, NUM_LAYERS, blocks_per_chunk * BLOCK_SIZE, NUM_HEADS * HEAD_SIZE),
            dtype=DTYPE,
        )
        for _ in range(NUM_BLOCKS // blocks_per_chunk)
    ]


def _shape_desc() -> PageBufferShapeDesc:
    """Descriptor matching the paged layers above."""
    desc = PageBufferShapeDesc()
    desc.kv_size = 2
    desc.nl = NUM_LAYERS
    desc.nb = NUM_BLOCKS
    desc.bs = BLOCK_SIZE
    desc.nh = NUM_HEADS
    desc.hs = HEAD_SIZE
    desc.element_size = DTYPE.itemsize
    return desc


def _transfer(
    layers: list[torch.Tensor],
    chunks: list[torch.Tensor],
    direction: TransferDirection,
    skip_prefix_n_blocks: int = 0,
    block_ids: "list[int] | None" = None,
    blocks_per_chunk: int = BLOCKS_PER_CHUNK,
) -> None:
    """Run the RBLN block transfer over HND layers."""
    RblnDeviceOps().multi_layer_block_kv_transfer(
        layers,
        chunks,
        list(range(NUM_BLOCKS)) if block_ids is None else block_ids,
        torch.device("cpu"),
        direction,
        _shape_desc(),
        blocks_per_chunk * BLOCK_SIZE,
        EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS,
        skip_prefix_n_blocks,
    )


# ---------------------------------------------------------------------------
# Guards: raised in RblnDeviceOps, no extension needed
# ---------------------------------------------------------------------------


def test_unsupported_format_is_refused() -> None:
    """Only the HND layout the detector produces is validated."""
    with pytest.raises(ValueError, match="NL_X_TWO_NB_NH_ONE_BS_HS"):
        RblnDeviceOps().multi_layer_block_kv_transfer(
            _paged_layers(),
            _chunks(),
            list(range(NUM_BLOCKS)),
            torch.device("cpu"),
            TransferDirection.D2H,
            _shape_desc(),
            CHUNK_TOKENS,
            EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            0,
        )


def test_pointer_operands_are_refused() -> None:
    """The extension takes tensors; pointers are only produced by bind_native."""
    with pytest.raises(ValueError, match="tensor operands"):
        RblnDeviceOps().multi_layer_block_kv_transfer(
            torch.tensor([0, 1], dtype=torch.int64),
            [0, 1],
            list(range(NUM_BLOCKS)),
            torch.device("cpu"),
            TransferDirection.D2H,
            _shape_desc(),
            CHUNK_TOKENS,
            EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS,
            0,
        )


def test_chunk_size_must_be_a_block_multiple() -> None:
    """A ragged chunk size would mis-slice the block list."""
    with pytest.raises(ValueError, match="multiple of shape_desc.bs"):
        RblnDeviceOps().multi_layer_block_kv_transfer(
            _paged_layers(),
            _chunks(),
            list(range(NUM_BLOCKS)),
            torch.device("cpu"),
            TransferDirection.D2H,
            _shape_desc(),
            BLOCK_SIZE + 1,
            EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS,
            0,
        )


def test_missing_extension_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """There is no torch fallback: without the extension the transfer raises."""
    monkeypatch.setattr(device_ops, "rbln_ops", None)
    with pytest.raises(RuntimeError, match="BUILD_WITH_RBLN"):
        _transfer(_paged_layers(), _chunks(), TransferDirection.D2H)


# ---------------------------------------------------------------------------
# Transfer: needs lmcache.rbln_ops
# ---------------------------------------------------------------------------

needs_extension = pytest.mark.skipif(
    device_ops.rbln_ops is None, reason="lmcache.rbln_ops not built"
)


@needs_extension
@pytest.mark.parametrize("blocks_per_chunk", [1, BLOCKS_PER_CHUNK])
def test_chunk_matches_the_canonical_torch_path(blocks_per_chunk: int) -> None:
    """RBLN chunks are byte-identical to the shared torch HND path's.

    ``blocks_per_chunk == 1`` takes the one-descriptor-per-chunk leg, a larger
    value the per-(kv, layer) leg.
    """
    layers = _paged_layers(fill_random=True)
    ours = _chunks(blocks_per_chunk)
    theirs = _chunks(blocks_per_chunk)

    _transfer(layers, ours, TransferDirection.D2H, blocks_per_chunk=blocks_per_chunk)
    multi_layer_block_kv_transfer(
        squeeze_singleton_axis(layers),
        theirs,
        list(range(NUM_BLOCKS)),
        torch.device("cpu"),
        TransferDirection.D2H,
        _shape_desc(),
        blocks_per_chunk * BLOCK_SIZE,
        EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        0,
    )
    for got, expected in zip(ours, theirs, strict=True):
        assert torch.equal(got, expected)


@needs_extension
def test_round_trip_restores_the_paged_cache() -> None:
    """Gather then scatter reproduces the source exactly, blocks permuted."""
    src = _paged_layers(fill_random=True)
    dst = _paged_layers()
    chunks = _chunks()
    order = [5, 2, 7, 0, 3, 6, 1, 4]
    _transfer(src, chunks, TransferDirection.D2H, block_ids=order)
    _transfer(dst, chunks, TransferDirection.H2D, block_ids=order)
    for got, expected in zip(dst, src, strict=True):
        assert torch.equal(got, expected)


@needs_extension
@pytest.mark.parametrize("skip", [1, 2, 3])
def test_prefix_skip_leaves_leading_blocks_untouched(skip: int) -> None:
    """A whole-block prefix skip is neither read nor written."""
    src = _paged_layers(fill_random=True)
    dst = _paged_layers()
    chunks = _chunks()
    _transfer(src, chunks, TransferDirection.D2H)
    _transfer(dst, chunks, TransferDirection.H2D, skip_prefix_n_blocks=skip)
    for got, expected in zip(dst, src, strict=True):
        assert torch.count_nonzero(got[:, :skip]) == 0
        assert torch.equal(got[:, skip:], expected[:, skip:])


@needs_extension
def test_trailing_partial_chunk_is_handled() -> None:
    """A chunk holding fewer blocks than it is sized for round-trips."""
    src = _paged_layers(fill_random=True)
    dst = _paged_layers()
    chunks = _chunks()
    partial = NUM_BLOCKS - 1
    for direction, layers in (
        (TransferDirection.D2H, src),
        (TransferDirection.H2D, dst),
    ):
        _transfer(layers, chunks, direction, block_ids=list(range(partial)))
    for got, expected in zip(dst, src, strict=True):
        assert torch.equal(got[:, :partial], expected[:, :partial])
        assert torch.count_nonzero(got[:, partial:]) == 0
