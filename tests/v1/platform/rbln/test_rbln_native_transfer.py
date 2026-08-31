# SPDX-License-Identifier: Apache-2.0
"""Tests for ``lmcache.rbln_ops``, the native token-major block transfer.

Runs on CPU tensors: the extension executes the same segment planning, staging
and pipeline order there with eager copies and swaps, which pins the layout
contract without hardware. The device transpose and the copy-stream overlap are
covered by ``bench_kv_transfer_mp.py --verify`` on an NPU.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor

# Third Party
import pytest
import torch

rbln_ops = pytest.importorskip("lmcache.rbln_ops")

NUM_LAYERS, NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_SIZE = 3, 8, 2, 4, 8
BLOCKS_PER_CHUNK = 2
CHUNK_TOKENS = BLOCKS_PER_CHUNK * BLOCK_SIZE
DTYPE = torch.bfloat16


def _layers(fill_random: bool = True) -> list[torch.Tensor]:
    torch.manual_seed(7)
    shape = (2, NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_SIZE)
    return [
        (torch.randn(shape) if fill_random else torch.zeros(shape)).to(DTYPE)
        for _ in range(NUM_LAYERS)
    ]


def _chunks() -> list[torch.Tensor]:
    return [
        torch.zeros((2, NUM_LAYERS, CHUNK_TOKENS, NUM_HEADS * HEAD_SIZE), dtype=DTYPE)
        for _ in range(NUM_BLOCKS // BLOCKS_PER_CHUNK)
    ]


def _reference(layers: list[torch.Tensor]) -> list[torch.Tensor]:
    """Token-major chunks built with plain torch indexing."""
    chunks = _chunks()
    for ci, chunk in enumerate(chunks):
        for li, layer in enumerate(layers):
            for j in range(BLOCKS_PER_CHUNK):
                block = layer[:, ci * BLOCKS_PER_CHUNK + j]  # [2, H, BS, D]
                chunk[:, li, j * BLOCK_SIZE : (j + 1) * BLOCK_SIZE] = block.permute(
                    0, 2, 1, 3
                ).flatten(-2)
    return chunks


ALL = list(range(NUM_BLOCKS))


def test_gather_matches_reference() -> None:
    layers = _layers()
    chunks = _chunks()
    rbln_ops.gather_blocks_to_chunks_token_major(layers, ALL, chunks, BLOCKS_PER_CHUNK)
    for got, expected in zip(chunks, _reference(layers), strict=True):
        assert torch.equal(got, expected)


def test_gather_handles_a_trailing_partial_chunk() -> None:
    layers = _layers()
    chunks = _chunks()
    rbln_ops.gather_blocks_to_chunks_token_major(
        layers, ALL[:-1], chunks, BLOCKS_PER_CHUNK
    )
    reference = _reference(layers)
    for got, expected in zip(chunks[:-1], reference[:-1], strict=True):
        assert torch.equal(got, expected)
    assert torch.equal(chunks[-1][:, :, :BLOCK_SIZE], reference[-1][:, :, :BLOCK_SIZE])
    assert not chunks[-1][:, :, BLOCK_SIZE:].any()


def test_gather_slices_when_staging_is_capped() -> None:
    """A one-byte cap makes every slice a single block and every run partial."""
    layers = _layers()
    chunks = _chunks()
    rbln_ops.gather_blocks_to_chunks_token_major(
        layers, ALL, chunks, BLOCKS_PER_CHUNK, max_staging_bytes=1
    )
    for got, expected in zip(chunks, _reference(layers), strict=True):
        assert torch.equal(got, expected)


def test_round_trip_restores_the_paged_cache() -> None:
    src = _layers()
    dst = _layers(fill_random=False)
    rbln_ops.scatter_chunks_to_blocks_token_major(
        dst, ALL, _reference(src), BLOCKS_PER_CHUNK, 0
    )
    for got, expected in zip(dst, src, strict=True):
        assert torch.equal(got, expected)


def test_scatter_skips_the_prefix() -> None:
    src = _layers()
    dst = _layers(fill_random=False)
    rbln_ops.scatter_chunks_to_blocks_token_major(
        dst, ALL, _reference(src), BLOCKS_PER_CHUNK, 3
    )
    for got, expected in zip(dst, src, strict=True):
        assert not got[:, :3].any()
        assert torch.equal(got[:, 3:], expected[:, 3:])


def test_scatter_slices_when_staging_is_capped() -> None:
    src = _layers()
    dst = _layers(fill_random=False)
    rbln_ops.scatter_chunks_to_blocks_token_major(
        dst, ALL, _reference(src), BLOCKS_PER_CHUNK, 0, max_staging_bytes=1
    )
    for got, expected in zip(dst, src, strict=True):
        assert torch.equal(got, expected)


def test_chunks_too_small_are_refused() -> None:
    with pytest.raises(RuntimeError):
        rbln_ops.gather_blocks_to_chunks_token_major(
            _layers(), ALL, _chunks()[:1], BLOCKS_PER_CHUNK
        )


def test_parallel_transfers_do_not_share_staging() -> None:
    """Staging is per thread: concurrent gathers and scatters stay correct."""
    layers = _layers()
    reference = _reference(layers)
    before = [layer.clone() for layer in layers]

    def gather() -> list[torch.Tensor]:
        chunks = _chunks()
        for _ in range(20):
            rbln_ops.gather_blocks_to_chunks_token_major(
                layers, ALL[:2], chunks[:1], BLOCKS_PER_CHUNK
            )
        return chunks

    def scatter() -> None:
        # Blocks 6, 7 keep receiving the bytes of blocks 0, 1 while gathers read 0, 1.
        for _ in range(20):
            rbln_ops.scatter_chunks_to_blocks_token_major(
                layers, [6, 7], reference[:1], BLOCKS_PER_CHUNK, 0
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        gathers = [pool.submit(gather) for _ in range(4)]
        scatters = [pool.submit(scatter) for _ in range(4)]
        results = [f.result() for f in gathers]
        for f in scatters:
            f.result()
    for chunks in results:
        assert torch.equal(chunks[0], reference[0])
    for layer, orig in zip(layers, before, strict=True):
        assert torch.equal(layer[:, :6], orig[:, :6])
        assert torch.equal(layer[:, 6:8], orig[:, :2])


# ---------------------------------------------------------------------------
# chunk_size == block_size: the configuration the path optimizes
# ---------------------------------------------------------------------------


def _one_block_chunks() -> list[torch.Tensor]:
    return [
        torch.zeros((2, NUM_LAYERS, BLOCK_SIZE, NUM_HEADS * HEAD_SIZE), dtype=DTYPE)
        for _ in range(NUM_BLOCKS)
    ]


def _one_block_reference(layers: list[torch.Tensor]) -> list[torch.Tensor]:
    chunks = _one_block_chunks()
    for ci, chunk in enumerate(chunks):
        for li, layer in enumerate(layers):
            chunk[:, li] = layer[:, ci].permute(0, 2, 1, 3).flatten(-2)
    return chunks


def test_gather_with_one_block_per_chunk() -> None:
    layers = _layers()
    chunks = _one_block_chunks()
    rbln_ops.gather_blocks_to_chunks_token_major(layers, ALL, chunks, 1)
    for got, expected in zip(chunks, _one_block_reference(layers), strict=True):
        assert torch.equal(got, expected)


def test_gather_with_one_block_per_chunk_in_a_single_slice() -> None:
    """Staging large enough for every block leaves the transfer one unit."""
    layers = _layers()
    chunks = _one_block_chunks()
    block_bytes = 2 * NUM_LAYERS * NUM_HEADS * BLOCK_SIZE * HEAD_SIZE * DTYPE.itemsize
    rbln_ops.gather_blocks_to_chunks_token_major(
        layers, ALL, chunks, 1, max_staging_bytes=block_bytes * NUM_BLOCKS
    )
    for got, expected in zip(chunks, _one_block_reference(layers), strict=True):
        assert torch.equal(got, expected)


def test_round_trip_with_one_block_per_chunk() -> None:
    src = _layers()
    dst = _layers(fill_random=False)
    rbln_ops.scatter_chunks_to_blocks_token_major(
        dst, ALL, _one_block_reference(src), 1, 0
    )
    for got, expected in zip(dst, src, strict=True):
        assert torch.equal(got, expected)


def test_chunk_token_count_must_match_blocks_per_chunk() -> None:
    with pytest.raises(RuntimeError, match="tokens"):
        rbln_ops.gather_blocks_to_chunks_token_major(_layers(), ALL, _chunks(), 1)
