# SPDX-License-Identifier: Apache-2.0
"""Tests for the RBLN-native head-major block transfer (``lmcache.rbln_ops``).

Two layers of coverage:

- Binding: :meth:`RblnDeviceOps.ensure_native` layers the extension over the
  torch baseline and degrades to "symbol absent" when it is not built. These
  run everywhere, with the extension stubbed.
- Kernel: ``block_kv_transfer_head_major`` moves bytes between a real RBLN
  KV cache and head-major host chunks. The reference is computed with plain
  torch indexing on the host, so the test pins the layout contract rather
  than round-tripping through the kernel twice. Needs the extension and an
  NPU; skipped otherwise.
"""

# Standard
from types import ModuleType
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.rbln.device_ops import RblnDeviceOps
import lmcache.lmcache_native as lmcache_native

TransferDirection = lmcache_native.TransferDirection

NUM_LAYERS = 2
NUM_BLOCKS = 8
NUM_HEADS = 2
BLOCK_SIZE = 4
HEAD_SIZE = 8
DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------


def test_ensure_native_binds_extension_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every public callable of ``lmcache.rbln_ops`` lands on the instance."""
    calls: list[tuple[object, ...]] = []

    def fake_kernel(*args: object, **kwargs: object) -> None:
        calls.append(args)

    stub = ModuleType("lmcache.rbln_ops")
    stub.block_kv_transfer_head_major = fake_kernel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lmcache.rbln_ops", stub)

    ops = RblnDeviceOps()
    assert not hasattr(ops, "block_kv_transfer_head_major")

    ops.ensure_native()
    ops.block_kv_transfer_head_major([], [], [], TransferDirection.D2H, 0)

    assert calls == [([], [], [], TransferDirection.D2H, 0)]
    # The torch-baseline override is untouched by the bind.
    assert (
        ops.multi_layer_block_kv_transfer.__func__
        is RblnDeviceOps.multi_layer_block_kv_transfer
    )


def test_ensure_native_without_extension_leaves_symbol_unbound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing build is a warning, not an error, and ``hasattr`` says so."""
    monkeypatch.setitem(sys.modules, "lmcache.rbln_ops", None)  # forces ImportError

    ops = RblnDeviceOps()
    ops.ensure_native()
    ops.ensure_native()  # idempotent

    assert not hasattr(ops, "block_kv_transfer_head_major")


# ---------------------------------------------------------------------------
# Kernel (hardware)
# ---------------------------------------------------------------------------


def _rbln_available() -> bool:
    try:
        return hasattr(torch, "rbln") and torch.rbln.is_available()
    except Exception:
        return False


requires_npu = pytest.mark.skipif(
    not _rbln_available(), reason="needs an RBLN NPU (torch.rbln unavailable)"
)


@pytest.fixture
def native_ops() -> RblnDeviceOps:
    """``RblnDeviceOps`` with the compiled extension bound, or skip."""
    pytest.importorskip("lmcache.rbln_ops", reason="lmcache.rbln_ops not built")
    ops = RblnDeviceOps()
    ops.ensure_native()
    return ops


def _paged_layers(device: torch.device, fill_random: bool) -> list[torch.Tensor]:
    """Per-layer KV in the native 6-D layout, materialised on ``device``."""
    torch.manual_seed(7)
    shape = (2, NUM_BLOCKS, NUM_HEADS, 1, BLOCK_SIZE, HEAD_SIZE)
    factory = torch.randn if fill_random else torch.zeros
    return [factory(shape, dtype=DTYPE).to(device) for _ in range(NUM_LAYERS)]


def _reference_chunk(paged_cpu: list[torch.Tensor], blocks: list[int]) -> torch.Tensor:
    """Head-major ``[2, L, H, T, D]`` chunk built with plain torch indexing."""
    chunk_tokens = len(blocks) * BLOCK_SIZE
    out = torch.empty((2, NUM_LAYERS, NUM_HEADS, chunk_tokens, HEAD_SIZE), dtype=DTYPE)
    for layer, kv in enumerate(paged_cpu):
        squeezed = kv.squeeze(3)  # [2, NB, NH, BS, HS]
        for i, b in enumerate(blocks):
            tok = slice(i * BLOCK_SIZE, (i + 1) * BLOCK_SIZE)
            out[:, layer, :, tok, :] = squeezed[:, b]
    return out


@requires_npu
@pytest.mark.parametrize("blocks_per_chunk", [1, 2], ids=["coalesced", "per-head"])
def test_d2h_matches_torch_reference(
    native_ops: RblnDeviceOps, blocks_per_chunk: int
) -> None:
    """Store: the device blocks land in the head-major chunk byte-for-byte."""
    device = torch.device("rbln")
    paged = _paged_layers(device, fill_random=True)
    block_ids = [5, 2, 7, 0][: 2 * blocks_per_chunk]
    chunk_numel = 2 * NUM_LAYERS * NUM_HEADS * blocks_per_chunk * BLOCK_SIZE * HEAD_SIZE
    chunks = [torch.zeros(chunk_numel, dtype=DTYPE) for _ in range(2)]

    native_ops.block_kv_transfer_head_major(
        paged, chunks, block_ids, TransferDirection.D2H, 0
    )

    paged_cpu = [kv.cpu() for kv in paged]
    for chunk_idx, chunk in enumerate(chunks):
        blocks = block_ids[
            chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk
        ]
        expected = _reference_chunk(paged_cpu, blocks)
        assert torch.equal(chunk.view_as(expected), expected)


@requires_npu
def test_h2d_scatters_and_honours_prefix_skip(native_ops: RblnDeviceOps) -> None:
    """Retrieve: chunk bytes land in the addressed blocks; skipped ones stay 0."""
    device = torch.device("rbln")
    paged = _paged_layers(device, fill_random=False)
    blocks_per_chunk = 2
    block_ids = [1, 4, 6, 3]
    skip = 1
    torch.manual_seed(3)
    chunk_shape = (2, NUM_LAYERS, NUM_HEADS, blocks_per_chunk * BLOCK_SIZE, HEAD_SIZE)
    chunks = [torch.randn(chunk_shape, dtype=DTYPE).contiguous() for _ in range(2)]

    native_ops.block_kv_transfer_head_major(
        paged, chunks, block_ids, TransferDirection.H2D, skip
    )

    paged_cpu = [kv.cpu().squeeze(3) for kv in paged]
    for flat, b in enumerate(block_ids):
        chunk = chunks[flat // blocks_per_chunk]
        i = flat % blocks_per_chunk
        tok = slice(i * BLOCK_SIZE, (i + 1) * BLOCK_SIZE)
        for layer, kv in enumerate(paged_cpu):
            got = kv[:, b]  # [2, NH, BS, HS]
            if flat < skip:
                assert torch.count_nonzero(got) == 0
            else:
                assert torch.equal(got, chunk[:, layer, :, tok, :])
    # Blocks never addressed are untouched.
    untouched = sorted(set(range(NUM_BLOCKS)) - set(block_ids))
    for kv in paged_cpu:
        assert torch.count_nonzero(kv[:, untouched]) == 0


@requires_npu
def test_geometry_errors_are_raised_before_any_dma(
    native_ops: RblnDeviceOps,
) -> None:
    """Malformed operands fail with a clear error instead of an OOB copy."""
    device = torch.device("rbln")
    paged = _paged_layers(device, fill_random=False)
    good_chunk = torch.zeros(
        2 * NUM_LAYERS * NUM_HEADS * BLOCK_SIZE * HEAD_SIZE, dtype=DTYPE
    )

    with pytest.raises(RuntimeError, match="divisible"):
        native_ops.block_kv_transfer_head_major(
            paged, [good_chunk, good_chunk], [0, 1, 2], TransferDirection.D2H, 0
        )
    with pytest.raises(RuntimeError, match="smaller than"):
        native_ops.block_kv_transfer_head_major(
            paged, [good_chunk[:-1]], [0], TransferDirection.D2H, 0
        )
    with pytest.raises(RuntimeError, match="out of range"):
        native_ops.block_kv_transfer_head_major(
            paged, [good_chunk], [NUM_BLOCKS], TransferDirection.D2H, 0
        )
    with pytest.raises(RuntimeError, match="6-D"):
        native_ops.block_kv_transfer_head_major(
            [kv.squeeze(3) for kv in paged], [good_chunk], [0], TransferDirection.D2H, 0
        )
