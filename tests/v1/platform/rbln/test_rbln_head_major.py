# SPDX-License-Identifier: Apache-2.0
"""Tests for the optional head-major RBLN chunk layout.

``LMCACHE_RBLN_SAVE_HEAD_MAJOR=1`` makes ``RblnDeviceOps`` write the
paged cache's own ``[2, L, H, T, D]`` order into the ``[2, L, T, H*D]`` chunk
buffer instead of the canonical token-major layout, skipping the host
transpose. The load-bearing tests here are the two layout checks: a round trip
alone passes under any self-consistent layout, so the bytes are pinned against
the paged tensors directly and against the token-major chunk permuted by hand
-- which is also exactly what the in-process ``lmcache-rbln`` connector writes.

No RBLN hardware is needed: the kernels are torch-only.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
from lmcache.v1.platform.rbln.device_ops import RblnDeviceOps
from lmcache.v1.platform.rbln.kv_layout import (
    HEAD_MAJOR_ENV_VAR,
    RblnChunkLayout,
)
from lmcache.v1.platform.rbln.kv_ops import head_major_view
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


def _paged_layers(fill_random: bool = True) -> list[torch.Tensor]:
    torch.manual_seed(11)
    shape = (2, NUM_BLOCKS, NUM_HEADS, 1, BLOCK_SIZE, HEAD_SIZE)
    factory = torch.randn if fill_random else torch.zeros
    return [factory(shape, dtype=DTYPE) for _ in range(NUM_LAYERS)]


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
    ops: RblnDeviceOps,
    layers: list[torch.Tensor],
    chunks: list[torch.Tensor],
    direction: TransferDirection,
    block_ids: list[int] | None = None,
    skip_prefix_n_blocks: int = 0,
) -> None:
    ops.multi_layer_block_kv_transfer(
        layers,
        chunks,
        list(range(NUM_BLOCKS)) if block_ids is None else block_ids,
        torch.device("cpu"),
        direction,
        _shape_desc(),
        CHUNK_TOKENS,
        EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS,
        skip_prefix_n_blocks,
    )


@pytest.fixture
def head_major() -> RblnDeviceOps:
    return RblnDeviceOps(chunk_layout=RblnChunkLayout.HEAD_MAJOR)


@pytest.fixture
def token_major() -> RblnDeviceOps:
    return RblnDeviceOps(chunk_layout=RblnChunkLayout.TOKEN_MAJOR)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_chunk_is_head_major(head_major: RblnDeviceOps) -> None:
    layers = _paged_layers()
    chunks = _chunks()
    _transfer(head_major, layers, chunks, TransferDirection.D2H)

    view = head_major_view(chunks[0], NUM_HEADS, HEAD_SIZE)
    for kv in (0, 1):
        for li in range(NUM_LAYERS):
            for pos in range(BLOCKS_PER_CHUNK):
                tokens = slice(pos * BLOCK_SIZE, (pos + 1) * BLOCK_SIZE)
                assert torch.equal(
                    view[kv, li, :, tokens, :], layers[li][kv, pos, :, 0, :, :]
                )


def test_head_major_is_the_token_major_chunk_permuted(
    head_major: RblnDeviceOps, token_major: RblnDeviceOps
) -> None:
    layers = _paged_layers()
    hm, tm = _chunks(), _chunks()
    _transfer(head_major, layers, hm, TransferDirection.D2H)
    _transfer(token_major, layers, tm, TransferDirection.D2H)

    for got, canonical in zip(hm, tm, strict=True):
        expected = canonical.unflatten(-1, (NUM_HEADS, HEAD_SIZE)).permute(
            0, 1, 3, 2, 4
        )
        assert torch.equal(head_major_view(got, NUM_HEADS, HEAD_SIZE), expected)
        assert not torch.equal(got, canonical)


def test_round_trip_restores_the_paged_cache(head_major: RblnDeviceOps) -> None:
    src = _paged_layers()
    dst = _paged_layers(fill_random=False)
    chunks = _chunks()
    _transfer(head_major, src, chunks, TransferDirection.D2H)
    _transfer(head_major, dst, chunks, TransferDirection.H2D)
    for got, expected in zip(dst, src, strict=True):
        assert torch.equal(got, expected)


def test_layouts_do_not_mix(
    head_major: RblnDeviceOps, token_major: RblnDeviceOps
) -> None:
    src = _paged_layers()
    dst = _paged_layers(fill_random=False)
    chunks = _chunks()
    _transfer(head_major, src, chunks, TransferDirection.D2H)
    _transfer(token_major, dst, chunks, TransferDirection.H2D)
    assert not all(torch.equal(g, e) for g, e in zip(dst, src, strict=True))


def test_prefix_skip_leaves_leading_blocks_untouched(
    head_major: RblnDeviceOps,
) -> None:
    src = _paged_layers()
    dst = _paged_layers(fill_random=False)
    chunks = _chunks()
    _transfer(head_major, src, chunks, TransferDirection.D2H)
    _transfer(head_major, dst, chunks, TransferDirection.H2D, skip_prefix_n_blocks=1)
    for got, expected in zip(dst, src, strict=True):
        assert torch.count_nonzero(got[:, 0]) == 0
        assert torch.equal(got[:, 1:], expected[:, 1:])


def test_prefix_skip_spanning_whole_chunks(head_major: RblnDeviceOps) -> None:
    src = _paged_layers()
    dst = _paged_layers(fill_random=False)
    chunks = _chunks()
    skip = BLOCKS_PER_CHUNK + 1
    _transfer(head_major, src, chunks, TransferDirection.D2H)
    _transfer(head_major, dst, chunks, TransferDirection.H2D, skip_prefix_n_blocks=skip)
    for got, expected in zip(dst, src, strict=True):
        assert torch.count_nonzero(got[:, :skip]) == 0
        assert torch.equal(got[:, skip:], expected[:, skip:])


def test_trailing_partial_chunk_is_handled(head_major: RblnDeviceOps) -> None:
    src = _paged_layers()
    dst = _paged_layers(fill_random=False)
    chunks = _chunks()
    partial = NUM_BLOCKS - 1
    for direction, layers in (
        (TransferDirection.D2H, src),
        (TransferDirection.H2D, dst),
    ):
        _transfer(head_major, layers, chunks, direction, block_ids=list(range(partial)))
    for got, expected in zip(dst, src, strict=True):
        assert torch.equal(got[:, :partial], expected[:, :partial])
        assert torch.count_nonzero(got[:, partial:]) == 0


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_head_major_view_rejects_wrong_geometry() -> None:
    with pytest.raises(ValueError, match="H\\*D"):
        head_major_view(_chunks()[0], NUM_HEADS + 1, HEAD_SIZE)
    strided = torch.zeros(
        (2, NUM_LAYERS, CHUNK_TOKENS, 2 * NUM_HEADS * HEAD_SIZE), dtype=DTYPE
    )[..., ::2]
    with pytest.raises(ValueError, match="contiguous"):
        head_major_view(strided, NUM_HEADS, HEAD_SIZE)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "0", "false", "OFF"])
def test_default_and_falsy_values_are_token_major(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv(HEAD_MAJOR_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(HEAD_MAJOR_ENV_VAR, value)
    assert RblnDeviceOps().chunk_layout is RblnChunkLayout.TOKEN_MAJOR


@pytest.mark.parametrize("value", ["1", " true ", "YES", "on"])
def test_truthy_values_select_head_major(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(HEAD_MAJOR_ENV_VAR, value)
    assert RblnDeviceOps().chunk_layout is RblnChunkLayout.HEAD_MAJOR


def test_non_boolean_value_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(HEAD_MAJOR_ENV_VAR, "head_major")
    with pytest.raises(ValueError, match="must be 1/0"):
        RblnDeviceOps()


def test_explicit_layout_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(HEAD_MAJOR_ENV_VAR, "1")
    ops = RblnDeviceOps(chunk_layout=RblnChunkLayout.TOKEN_MAJOR)
    assert ops.chunk_layout is RblnChunkLayout.TOKEN_MAJOR


# ---------------------------------------------------------------------------
# MLA is unaffected
# ---------------------------------------------------------------------------

MLA_HEAD_SIZE = 16


def _mla_layers(fill_random: bool = True) -> list[torch.Tensor]:
    torch.manual_seed(23)
    shape = (NUM_BLOCKS, BLOCK_SIZE, MLA_HEAD_SIZE)
    factory = torch.randn if fill_random else torch.zeros
    return [factory(shape, dtype=DTYPE) for _ in range(NUM_LAYERS)]


def _mla_chunks() -> list[torch.Tensor]:
    return [
        torch.zeros((NUM_LAYERS, CHUNK_TOKENS, MLA_HEAD_SIZE), dtype=DTYPE)
        for _ in range(NUM_BLOCKS // BLOCKS_PER_CHUNK)
    ]


def _mla_transfer(
    ops: RblnDeviceOps,
    layers: list[torch.Tensor],
    chunks: list[torch.Tensor],
    direction: TransferDirection,
) -> None:
    desc = PageBufferShapeDesc()
    desc.kv_size = 1
    desc.nl = NUM_LAYERS
    desc.nb = NUM_BLOCKS
    desc.bs = BLOCK_SIZE
    desc.nh = 1
    desc.hs = MLA_HEAD_SIZE
    desc.element_size = DTYPE.itemsize
    ops.multi_layer_block_kv_transfer(
        layers,
        chunks,
        list(range(NUM_BLOCKS)),
        torch.device("cpu"),
        direction,
        desc,
        CHUNK_TOKENS,
        EngineKVFormat.NL_X_NB_BS_HS,
        0,
    )


def test_mla_ignores_the_head_major_setting(
    head_major: RblnDeviceOps, token_major: RblnDeviceOps
) -> None:
    layers = _mla_layers()
    hm, tm = _mla_chunks(), _mla_chunks()
    _mla_transfer(head_major, layers, hm, TransferDirection.D2H)
    _mla_transfer(token_major, layers, tm, TransferDirection.D2H)
    for got, expected in zip(hm, tm, strict=True):
        assert torch.equal(got, expected)

    dst = _mla_layers(fill_random=False)
    _mla_transfer(head_major, dst, tm, TransferDirection.H2D)
    for got, expected in zip(dst, layers, strict=True):
        assert torch.equal(got, expected)
