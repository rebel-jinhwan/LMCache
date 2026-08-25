# SPDX-License-Identifier: Apache-2.0
"""Tests for the torch head-major (``KV_2LHTD``) block transfer kernel.

``multi_layer_block_kv_transfer_head_major`` fills the ``[2, L, T, H*D]``
chunk buffer as ``[2, L, H, T, D]``. A round trip alone would pass under any
self-consistent layout, so the load-bearing tests pin the bytes: directly
against the paged tensors, and as the ``(T, H)`` permutation of the canonical
token-major chunk the shared kernel writes. CPU tensors are enough -- the
kernel is torch-only and device-independent.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
from lmcache.v1.platform.torch_ops import (
    head_major_chunk_view,
    multi_layer_block_kv_transfer,
    multi_layer_block_kv_transfer_head_major,
)
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

TWO_MAJOR = EngineKVFormat.NL_X_TWO_NB_NH_BS_HS  # [2, NB, NH, BS, HS]
BLOCKS_FIRST = EngineKVFormat.NL_X_NB_TWO_NH_BS_HS  # [NB, 2, NH, BS, HS]


def _layers(fmt: EngineKVFormat, fill_random: bool = True) -> list[torch.Tensor]:
    torch.manual_seed(3)
    shape = (
        (2, NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_SIZE)
        if fmt == TWO_MAJOR
        else (NUM_BLOCKS, 2, NUM_HEADS, BLOCK_SIZE, HEAD_SIZE)
    )
    factory = torch.randn if fill_random else torch.zeros
    return [factory(shape, dtype=DTYPE) for _ in range(NUM_LAYERS)]


def _kv(layer: torch.Tensor, fmt: EngineKVFormat, half: int) -> torch.Tensor:
    """The ``[NB, NH, BS, HS]`` plane for K (0) or V (1)."""
    return layer[half] if fmt == TWO_MAJOR else layer[:, half]


def _chunks() -> list[torch.Tensor]:
    return [
        torch.zeros((2, NUM_LAYERS, CHUNK_TOKENS, NUM_HEADS * HEAD_SIZE), dtype=DTYPE)
        for _ in range(NUM_BLOCKS // BLOCKS_PER_CHUNK)
    ]


def _shape_desc() -> PageBufferShapeDesc:
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
    fmt: EngineKVFormat,
    block_ids: list[int] | None = None,
    skip_prefix_n_blocks: int = 0,
) -> None:
    multi_layer_block_kv_transfer_head_major(
        layers,
        chunks,
        list(range(NUM_BLOCKS)) if block_ids is None else block_ids,
        torch.device("cpu"),
        direction,
        _shape_desc(),
        CHUNK_TOKENS,
        fmt,
        skip_prefix_n_blocks,
    )


@pytest.mark.parametrize(
    "fmt", [TWO_MAJOR, BLOCKS_FIRST], ids=["two_major", "nb_first"]
)
def test_chunk_is_head_major(fmt: EngineKVFormat) -> None:
    """Viewed as [2, L, H, T, D], each block's [H, BS, D] lands verbatim."""
    layers = _layers(fmt)
    chunks = _chunks()
    _transfer(layers, chunks, TransferDirection.D2H, fmt)
    view = head_major_chunk_view(chunks[1], NUM_HEADS, HEAD_SIZE)
    for half in (0, 1):
        for li in range(NUM_LAYERS):
            for pos in range(BLOCKS_PER_CHUNK):
                block = BLOCKS_PER_CHUNK + pos
                tokens = slice(pos * BLOCK_SIZE, (pos + 1) * BLOCK_SIZE)
                assert torch.equal(
                    view[half, li, :, tokens, :], _kv(layers[li], fmt, half)[block]
                )


@pytest.mark.parametrize(
    "fmt", [TWO_MAJOR, BLOCKS_FIRST], ids=["two_major", "nb_first"]
)
def test_head_major_is_the_canonical_chunk_permuted(fmt: EngineKVFormat) -> None:
    """Same values as the token-major kernel, with H and T swapped."""
    layers = _layers(fmt)
    head_major, token_major = _chunks(), _chunks()
    _transfer(layers, head_major, TransferDirection.D2H, fmt)
    multi_layer_block_kv_transfer(
        layers,
        token_major,
        list(range(NUM_BLOCKS)),
        torch.device("cpu"),
        TransferDirection.D2H,
        _shape_desc(),
        CHUNK_TOKENS,
        fmt,
        0,
    )
    for got, canonical in zip(head_major, token_major, strict=True):
        expected = canonical.unflatten(-1, (NUM_HEADS, HEAD_SIZE)).permute(
            0, 1, 3, 2, 4
        )
        assert torch.equal(head_major_chunk_view(got, NUM_HEADS, HEAD_SIZE), expected)
        # The raw buffers differ: the format is not a no-op relabel.
        assert not torch.equal(got, canonical)


@pytest.mark.parametrize(
    "fmt", [TWO_MAJOR, BLOCKS_FIRST], ids=["two_major", "nb_first"]
)
def test_round_trip_restores_the_paged_cache(fmt: EngineKVFormat) -> None:
    src = _layers(fmt)
    dst = _layers(fmt, fill_random=False)
    chunks = _chunks()
    _transfer(src, chunks, TransferDirection.D2H, fmt)
    _transfer(dst, chunks, TransferDirection.H2D, fmt)
    for got, expected in zip(dst, src, strict=True):
        assert torch.equal(got, expected)


def test_prefix_skip_spanning_chunks_leaves_leading_blocks_untouched() -> None:
    """A skip larger than one chunk is applied globally, not per chunk."""
    src = _layers(TWO_MAJOR)
    dst = _layers(TWO_MAJOR, fill_random=False)
    chunks = _chunks()
    skip = BLOCKS_PER_CHUNK + 1
    _transfer(src, chunks, TransferDirection.D2H, TWO_MAJOR)
    _transfer(dst, chunks, TransferDirection.H2D, TWO_MAJOR, skip_prefix_n_blocks=skip)
    for got, expected in zip(dst, src, strict=True):
        assert torch.count_nonzero(got[:, :skip]) == 0
        assert torch.equal(got[:, skip:], expected[:, skip:])


def test_trailing_partial_chunk_is_handled() -> None:
    """A chunk holding fewer blocks than it is sized for round-trips."""
    src = _layers(TWO_MAJOR)
    dst = _layers(TWO_MAJOR, fill_random=False)
    chunks = _chunks()
    partial = NUM_BLOCKS - 1
    ids = list(range(partial))
    _transfer(src, chunks, TransferDirection.D2H, TWO_MAJOR, block_ids=ids)
    _transfer(dst, chunks, TransferDirection.H2D, TWO_MAJOR, block_ids=ids)
    for got, expected in zip(dst, src, strict=True):
        assert torch.equal(got[:, :partial], expected[:, :partial])
        assert torch.count_nonzero(got[:, partial:]) == 0


def test_head_major_view_rejects_wrong_geometry() -> None:
    with pytest.raises(ValueError, match=r"H\*D"):
        head_major_chunk_view(_chunks()[0], NUM_HEADS + 1, HEAD_SIZE)
    strided = torch.zeros(
        (2, NUM_LAYERS, CHUNK_TOKENS, 2 * NUM_HEADS * HEAD_SIZE), dtype=DTYPE
    )[..., ::2]
    with pytest.raises(ValueError, match="contiguous"):
        head_major_chunk_view(strided, NUM_HEADS, HEAD_SIZE)


@pytest.mark.parametrize(
    "fmt",
    [
        EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,  # NHD
        EngineKVFormat.NL_X_NB_BS_HS,  # MLA
        EngineKVFormat.NL_X_NB_NH_BS_TWO_HS,  # fused K/V
    ],
    ids=["nhd", "mla", "fused"],
)
def test_non_hnd_split_formats_are_refused(fmt: EngineKVFormat) -> None:
    """Head-major only pays off, and is only defined, for HND split K/V."""
    with pytest.raises(ValueError, match="head-major"):
        _transfer(_layers(TWO_MAJOR), _chunks(), TransferDirection.D2H, fmt)


def test_chunk_size_must_be_a_block_multiple() -> None:
    with pytest.raises(ValueError, match="multiple of shape_desc.bs"):
        multi_layer_block_kv_transfer_head_major(
            _layers(TWO_MAJOR),
            _chunks(),
            list(range(NUM_BLOCKS)),
            torch.device("cpu"),
            TransferDirection.D2H,
            _shape_desc(),
            BLOCK_SIZE + 1,
            TWO_MAJOR,
            0,
        )
