# SPDX-License-Identifier: Apache-2.0
"""Tests for the RBLN-native block transfer (``lmcache.rbln_ops``).

Two layers of coverage:

- Binding: :meth:`RblnDeviceOps.ensure_native` layers the extension over the
  torch baseline -- ``multi_layer_block_kv_transfer`` shadows the torch method,
  ``block_kv_transfer_mla`` is added -- and degrades to the torch baseline when
  it is not built. These run everywhere, with the extension stubbed.
- Kernel: the HND entry moves bytes between a real RBLN cache
  (``[2, NB, NH, 1, BS, HS]``) and canonical token-major ``[2, L, T, NH*HS]``
  chunks; ``block_kv_transfer_mla`` does the same for the MLA cache
  (``[NB, BS, HS]``) and ``[L, T, HS]`` chunks. References are computed with
  plain torch indexing on the host, so the tests pin the layout contract rather
  than round-tripping through the kernel twice. Need the extension and an NPU;
  skipped otherwise.
"""

# Standard
from types import ModuleType
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.rbln.device_ops import RblnDeviceOps
from lmcache.v1.platform.rbln.kv_ops import gather_blocks_to_chunk
import lmcache.lmcache_native as lmcache_native

TransferDirection = lmcache_native.TransferDirection
EngineKVFormat = lmcache_native.EngineKVFormat
PageBufferShapeDesc = lmcache_native.PageBufferShapeDesc

NUM_LAYERS = 2
NUM_BLOCKS = 8
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
    stub.block_kv_transfer_mla = fake_kernel  # type: ignore[attr-defined]
    stub.multi_layer_block_kv_transfer = fake_kernel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lmcache.rbln_ops", stub)

    ops = RblnDeviceOps()
    assert not hasattr(ops, "block_kv_transfer_mla")
    torch_method = RblnDeviceOps.multi_layer_block_kv_transfer
    assert ops.multi_layer_block_kv_transfer.__func__ is torch_method  # type: ignore[attr-defined]

    ops.ensure_native()
    ops.block_kv_transfer_mla([], [], [], TransferDirection.D2H, 0)

    assert calls == [([], [], [], TransferDirection.D2H, 0)]
    # The DeviceOps entry is shadowed on the instance so the mp gather/scatter
    # path reaches the native kernel; the class method stays the torch fallback.
    assert "multi_layer_block_kv_transfer" in vars(ops)
    assert RblnDeviceOps.multi_layer_block_kv_transfer is torch_method


def test_ensure_native_without_extension_leaves_symbol_unbound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing build is a warning, not an error, and ``hasattr`` says so."""
    monkeypatch.setitem(sys.modules, "lmcache.rbln_ops", None)  # forces ImportError

    ops = RblnDeviceOps()
    ops.ensure_native()
    ops.ensure_native()  # idempotent

    assert not hasattr(ops, "block_kv_transfer_mla")
    assert "multi_layer_block_kv_transfer" not in vars(ops)


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


def _mla_layers(device: torch.device, fill_random: bool) -> list[torch.Tensor]:
    """Per-layer MLA latent planes ``[NB, BS, HS]``, materialised on ``device``."""
    torch.manual_seed(7)
    shape = (NUM_BLOCKS, BLOCK_SIZE, HEAD_SIZE)
    factory = torch.randn if fill_random else torch.zeros
    return [factory(shape, dtype=DTYPE).to(device) for _ in range(NUM_LAYERS)]


def _reference_chunk(paged_cpu: list[torch.Tensor], blocks: list[int]) -> torch.Tensor:
    """Canonical ``[L, T, HS]`` MLA chunk built with plain torch indexing."""
    chunk_tokens = len(blocks) * BLOCK_SIZE
    out = torch.empty((NUM_LAYERS, chunk_tokens, HEAD_SIZE), dtype=DTYPE)
    for layer, kv in enumerate(paged_cpu):
        for i, b in enumerate(blocks):
            out[layer, i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE] = kv[b]
    return out


@requires_npu
@pytest.mark.parametrize("blocks_per_chunk", [1, 2])
def test_d2h_matches_torch_reference(
    native_ops: RblnDeviceOps, blocks_per_chunk: int
) -> None:
    """Store: the device blocks land in the ``[L, T, HS]`` chunk byte-for-byte."""
    device = torch.device("rbln")
    paged = _mla_layers(device, fill_random=True)
    block_ids = [5, 2, 7, 0][: 2 * blocks_per_chunk]
    chunk_numel = NUM_LAYERS * blocks_per_chunk * BLOCK_SIZE * HEAD_SIZE
    chunks = [torch.zeros(chunk_numel, dtype=DTYPE) for _ in range(2)]

    native_ops.block_kv_transfer_mla(paged, chunks, block_ids, TransferDirection.D2H, 0)

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
    paged = _mla_layers(device, fill_random=False)
    blocks_per_chunk = 2
    block_ids = [1, 4, 6, 3]
    skip = 1
    torch.manual_seed(3)
    chunk_shape = (NUM_LAYERS, blocks_per_chunk * BLOCK_SIZE, HEAD_SIZE)
    chunks = [torch.randn(chunk_shape, dtype=DTYPE).contiguous() for _ in range(2)]

    native_ops.block_kv_transfer_mla(
        paged, chunks, block_ids, TransferDirection.H2D, skip
    )

    paged_cpu = [kv.cpu() for kv in paged]
    for flat, b in enumerate(block_ids):
        chunk = chunks[flat // blocks_per_chunk]
        i = flat % blocks_per_chunk
        tok = slice(i * BLOCK_SIZE, (i + 1) * BLOCK_SIZE)
        for layer, kv in enumerate(paged_cpu):
            got = kv[b]  # [BS, HS]
            if flat < skip:
                assert torch.count_nonzero(got) == 0
            else:
                assert torch.equal(got, chunk[layer, tok])
    # Blocks never addressed are untouched.
    untouched = sorted(set(range(NUM_BLOCKS)) - set(block_ids))
    for kv in paged_cpu:
        assert torch.count_nonzero(kv[untouched]) == 0


@requires_npu
def test_geometry_errors_are_raised_before_any_dma(
    native_ops: RblnDeviceOps,
) -> None:
    """Malformed operands fail with a clear error instead of an OOB copy."""
    device = torch.device("rbln")
    paged = _mla_layers(device, fill_random=False)
    good_chunk = torch.zeros(NUM_LAYERS * BLOCK_SIZE * HEAD_SIZE, dtype=DTYPE)

    with pytest.raises(RuntimeError, match="divisible"):
        native_ops.block_kv_transfer_mla(
            paged, [good_chunk, good_chunk], [0, 1, 2], TransferDirection.D2H, 0
        )
    with pytest.raises(RuntimeError, match="smaller than"):
        native_ops.block_kv_transfer_mla(
            paged, [good_chunk[:-1]], [0], TransferDirection.D2H, 0
        )
    with pytest.raises(RuntimeError, match="out of range"):
        native_ops.block_kv_transfer_mla(
            paged, [good_chunk], [NUM_BLOCKS], TransferDirection.D2H, 0
        )
    with pytest.raises(RuntimeError, match="3-D"):
        native_ops.block_kv_transfer_mla(
            [kv.unsqueeze(0) for kv in paged],
            [good_chunk],
            [0],
            TransferDirection.D2H,
            0,
        )


# ---------------------------------------------------------------------------
# Kernel (hardware): HND layout through the DeviceOps entry
# ---------------------------------------------------------------------------

NUM_HEADS = 2


def _hnd_layers(device: torch.device, fill_random: bool) -> list[torch.Tensor]:
    """Per-layer HND caches ``[2, NB, NH, 1, BS, HS]``, materialised on ``device``."""
    torch.manual_seed(11)
    shape = (2, NUM_BLOCKS, NUM_HEADS, 1, BLOCK_SIZE, HEAD_SIZE)
    factory = torch.randn if fill_random else torch.zeros
    return [factory(shape, dtype=DTYPE).to(device) for _ in range(NUM_LAYERS)]


def _hnd_shape_desc() -> PageBufferShapeDesc:
    """Shape descriptor for the HND fixture, populated the way the mp path does."""
    desc = PageBufferShapeDesc()
    desc.kv_size = 2
    desc.nl = NUM_LAYERS
    desc.nb = NUM_BLOCKS
    desc.bs = BLOCK_SIZE
    desc.nh = NUM_HEADS
    desc.hs = HEAD_SIZE
    desc.element_size = torch.empty((), dtype=DTYPE).element_size()
    return desc


def _hnd_chunk(blocks_per_chunk: int) -> torch.Tensor:
    """Zeroed canonical token-major chunk ``[2, L, T, NH*HS]``."""
    return torch.zeros(
        (2, NUM_LAYERS, blocks_per_chunk * BLOCK_SIZE, NUM_HEADS * HEAD_SIZE),
        dtype=DTYPE,
    )


def _native_hnd_transfer(
    native_ops: RblnDeviceOps,
    paged: list[torch.Tensor],
    chunks: list[torch.Tensor],
    block_ids: list[int],
    blocks_per_chunk: int,
    direction: lmcache_native.TransferDirection,
    skip_prefix_n_blocks: int,
) -> None:
    native_ops.multi_layer_block_kv_transfer(
        paged,
        chunks,
        block_ids,
        paged[0].device,
        direction,
        _hnd_shape_desc(),
        blocks_per_chunk * BLOCK_SIZE,
        EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS,
        skip_prefix_n_blocks,
    )


@requires_npu
@pytest.mark.parametrize("blocks_per_chunk", [1, 2])
def test_hnd_d2h_matches_torch_reference(
    native_ops: RblnDeviceOps, blocks_per_chunk: int
) -> None:
    """Store: HND blocks land in the token-major chunk exactly as the torch path."""
    device = torch.device("rbln")
    paged = _hnd_layers(device, fill_random=True)
    block_ids = [5, 2, 7, 0][: 2 * blocks_per_chunk]
    chunks = [_hnd_chunk(blocks_per_chunk) for _ in range(2)]

    _native_hnd_transfer(
        native_ops, paged, chunks, block_ids, blocks_per_chunk, TransferDirection.D2H, 0
    )

    paged_cpu = [kv.cpu().squeeze(3) for kv in paged]
    for chunk_idx, chunk in enumerate(chunks):
        blocks = block_ids[
            chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk
        ]
        expected = _hnd_chunk(blocks_per_chunk)
        gather_blocks_to_chunk(paged_cpu, blocks, expected)
        assert torch.equal(chunk, expected)


@requires_npu
def test_hnd_h2d_scatters_and_honours_prefix_skip(
    native_ops: RblnDeviceOps,
) -> None:
    """Retrieve: chunk tokens land in the addressed HND blocks; skipped stay 0."""
    device = torch.device("rbln")
    paged = _hnd_layers(device, fill_random=False)
    blocks_per_chunk = 2
    block_ids = [1, 4, 6, 3]
    skip = 1
    torch.manual_seed(5)
    chunks = [torch.randn_like(_hnd_chunk(blocks_per_chunk)) for _ in range(2)]

    _native_hnd_transfer(
        native_ops,
        paged,
        chunks,
        block_ids,
        blocks_per_chunk,
        TransferDirection.H2D,
        skip,
    )

    paged_cpu = [kv.cpu().squeeze(3) for kv in paged]
    for flat, b in enumerate(block_ids):
        chunk = chunks[flat // blocks_per_chunk]
        tok = slice(
            (flat % blocks_per_chunk) * BLOCK_SIZE,
            (flat % blocks_per_chunk + 1) * BLOCK_SIZE,
        )
        for layer, kv in enumerate(paged_cpu):
            for k in range(2):
                got = kv[k, b]  # [NH, BS, HS]
                if flat < skip:
                    assert torch.count_nonzero(got) == 0
                    continue
                expected = (
                    chunk[k, layer, tok]
                    .view(BLOCK_SIZE, NUM_HEADS, HEAD_SIZE)
                    .permute(1, 0, 2)
                )
                assert torch.equal(got, expected)
    untouched = sorted(set(range(NUM_BLOCKS)) - set(block_ids))
    for kv in paged_cpu:
        assert torch.count_nonzero(kv[:, untouched]) == 0


@requires_npu
def test_hnd_rejects_unknown_format(native_ops: RblnDeviceOps) -> None:
    """A format neither HND nor MLA is refused before any DMA."""
    device = torch.device("rbln")
    paged = _hnd_layers(device, fill_random=False)
    with pytest.raises(RuntimeError, match="supports only"):
        native_ops.multi_layer_block_kv_transfer(
            paged,
            [_hnd_chunk(1)],
            [0],
            device,
            TransferDirection.D2H,
            _hnd_shape_desc(),
            BLOCK_SIZE,
            EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            0,
        )
