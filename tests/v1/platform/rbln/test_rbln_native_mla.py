# SPDX-License-Identifier: Apache-2.0
"""Tests for the RBLN MLA block transfer (``lmcache.rbln_ops``).

The guards run everywhere: they raise in ``RblnDeviceOps`` before the
extension is reached. The transfer tests need the extension built
(``BUILD_WITH_RBLN=1``) and are skipped otherwise. It is plain ATen, so CPU
tensors exercise the real kernel; what they cannot cover is the transfer cost
and torch-rbln's copy dispatch, which ``bench_kv_transfer_mp.py --verify``
checks on hardware.

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
from lmcache.v1.platform.torch_ops import multi_layer_block_kv_transfer
import lmcache.lmcache_native as lmcache_native

EngineKVFormat = lmcache_native.EngineKVFormat
TransferDirection = lmcache_native.TransferDirection

NUM_LAYERS = 3
NUM_BLOCKS = 8
BLOCK_SIZE = 4
HEAD_SIZE = 16
BLOCKS_PER_CHUNK = 2
CHUNK_TOKENS = BLOCKS_PER_CHUNK * BLOCK_SIZE
DTYPE = torch.float32


def _layers(fill_random: bool = True) -> list[torch.Tensor]:
    """Per-layer MLA KV in the native 3-D layout the detector reports."""
    torch.manual_seed(23)
    shape = (NUM_BLOCKS, BLOCK_SIZE, HEAD_SIZE)
    factory = torch.randn if fill_random else torch.zeros
    return [factory(shape, dtype=DTYPE) for _ in range(NUM_LAYERS)]


def _chunks() -> list[torch.Tensor]:
    """Single-plane staging chunks, as upstream allocates them for MLA."""
    return [
        torch.zeros((NUM_LAYERS, CHUNK_TOKENS, HEAD_SIZE), dtype=DTYPE)
        for _ in range(NUM_BLOCKS // BLOCKS_PER_CHUNK)
    ]


def _shape_desc() -> PageBufferShapeDesc:
    """Descriptor matching the layers above."""
    desc = PageBufferShapeDesc()
    desc.kv_size = 1
    desc.nl = NUM_LAYERS
    desc.nb = NUM_BLOCKS
    desc.bs = BLOCK_SIZE
    desc.nh = 1
    desc.hs = HEAD_SIZE
    desc.element_size = DTYPE.itemsize
    return desc


def _transfer(
    layers: list[torch.Tensor],
    chunks: list[torch.Tensor],
    direction: TransferDirection,
    skip_prefix_n_blocks: int = 0,
    block_ids: "list[int] | None" = None,
    engine_kv_format: EngineKVFormat = EngineKVFormat.NL_X_NB_BS_HS,
) -> None:
    """Run the RBLN block transfer over MLA layers."""
    RblnDeviceOps().multi_layer_block_kv_transfer(
        layers,
        chunks,
        list(range(NUM_BLOCKS)) if block_ids is None else block_ids,
        torch.device("cpu"),
        direction,
        _shape_desc(),
        CHUNK_TOKENS,
        engine_kv_format,
        skip_prefix_n_blocks,
    )


# ---------------------------------------------------------------------------
# Guards: raised in RblnDeviceOps, no extension needed
# ---------------------------------------------------------------------------


def test_other_mla_layouts_are_refused() -> None:
    """is_mla() admits every MLA variant; only NL_X_NB_BS_HS has a sequence."""
    with pytest.raises(ValueError, match="NL_X_NB_BS_HS MLA layout"):
        _transfer(
            _layers(),
            _chunks(),
            TransferDirection.D2H,
            engine_kv_format=EngineKVFormat.NL_X_NB_BSV_BSS,
        )


def test_missing_extension_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """MLA has no torch fallback: without the extension the transfer raises."""
    monkeypatch.setattr(device_ops, "rbln_ops", None)
    with pytest.raises(RuntimeError, match="BUILD_WITH_RBLN"):
        _transfer(_layers(), _chunks(), TransferDirection.D2H)


# ---------------------------------------------------------------------------
# Transfer: needs lmcache.rbln_ops
# ---------------------------------------------------------------------------

needs_extension = pytest.mark.skipif(
    device_ops.rbln_ops is None, reason="lmcache.rbln_ops not built"
)


@needs_extension
def test_chunk_matches_the_canonical_torch_path() -> None:
    """RBLN MLA chunks are byte-identical to the shared torch path's."""
    layers = _layers()
    ours = _chunks()
    theirs = _chunks()

    _transfer(layers, ours, TransferDirection.D2H)
    multi_layer_block_kv_transfer(
        layers,
        theirs,
        list(range(NUM_BLOCKS)),
        torch.device("cpu"),
        TransferDirection.D2H,
        _shape_desc(),
        CHUNK_TOKENS,
        EngineKVFormat.NL_X_NB_BS_HS,
        0,
    )
    for got, expected in zip(ours, theirs, strict=True):
        assert torch.equal(got, expected)


@needs_extension
def test_round_trip_restores_the_paged_cache() -> None:
    """Gather then scatter reproduces the source exactly, blocks permuted."""
    src = _layers()
    dst = _layers(fill_random=False)
    chunks = _chunks()
    order = [5, 2, 7, 0, 3, 6, 1, 4]
    _transfer(src, chunks, TransferDirection.D2H, block_ids=order)
    _transfer(dst, chunks, TransferDirection.H2D, block_ids=order)
    for got, expected in zip(dst, src, strict=True):
        assert torch.equal(got, expected)


@needs_extension
@pytest.mark.parametrize("skip", [1, 2, 3])
def test_prefix_skip_leaves_leading_blocks_untouched(skip: int) -> None:
    """A whole-block prefix skip is neither read nor written.

    ``skip=1`` and ``skip=3`` start inside a chunk (partial H2D window);
    ``skip=2`` skips exactly one chunk.
    """
    src = _layers()
    dst = _layers(fill_random=False)
    chunks = _chunks()
    _transfer(src, chunks, TransferDirection.D2H)
    _transfer(dst, chunks, TransferDirection.H2D, skip_prefix_n_blocks=skip)
    for got, expected in zip(dst, src, strict=True):
        assert torch.count_nonzero(got[:skip]) == 0
        assert torch.equal(got[skip:], expected[skip:])


@needs_extension
def test_trailing_partial_chunk_is_handled() -> None:
    """A chunk holding fewer blocks than it is sized for round-trips."""
    src = _layers()
    dst = _layers(fill_random=False)
    chunks = _chunks()
    partial = NUM_BLOCKS - 1
    for direction, layers in (
        (TransferDirection.D2H, src),
        (TransferDirection.H2D, dst),
    ):
        _transfer(layers, chunks, direction, block_ids=list(range(partial)))
    for got, expected in zip(dst, src, strict=True):
        assert torch.equal(got[:partial], expected[:partial])
        assert torch.count_nonzero(got[partial:]) == 0


@needs_extension
def test_non_mla_layers_are_refused() -> None:
    """The 6-D attention layout under the MLA format fails loudly."""
    hnd = [torch.zeros(2, NUM_BLOCKS, 2, 1, BLOCK_SIZE, 8) for _ in range(NUM_LAYERS)]
    with pytest.raises(RuntimeError, match=r"contiguous \[NB, BS, HS\]"):
        _transfer(hnd, _chunks(), TransferDirection.D2H)


@needs_extension
def test_non_contiguous_layers_are_refused() -> None:
    """A permuted view would break the whole-block copies; fail loudly."""
    permuted = [
        torch.zeros(BLOCK_SIZE, NUM_BLOCKS, HEAD_SIZE).permute(1, 0, 2)
        for _ in range(NUM_LAYERS)
    ]
    with pytest.raises(RuntimeError, match="contiguous"):
        _transfer(permuted, _chunks(), TransferDirection.D2H)
