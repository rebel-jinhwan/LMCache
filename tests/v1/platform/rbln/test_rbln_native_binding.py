# SPDX-License-Identifier: Apache-2.0
"""Tests for binding ``lmcache.rbln_ops`` over the torch kernels.

The extension only exists on a host with a rebel runtime, so what CI can pin is
the *decision*: when ``RblnDeviceOps`` uses the native kernel, when it falls
back, and what it hands the kernel. The kernel itself is a hardware test.

The declines carry the weight. Every case :func:`native_can_serve` rejects is
one where the kernel would address the wrong bytes rather than raise, so a gate
that silently stopped working would be data corruption, not a slowdown.
"""

# Standard
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
from lmcache.v1.platform.rbln import device_ops as device_ops_module
from lmcache.v1.platform.rbln.device_ops import RblnDeviceOps, native_can_serve
import lmcache.lmcache_native as lmcache_native

EngineKVFormat = lmcache_native.EngineKVFormat
TransferDirection = lmcache_native.TransferDirection

NUM_LAYERS = 2
NUM_BLOCKS = 4
NUM_HEADS = 2
BLOCK_SIZE = 4
HEAD_SIZE = 8
BLOCKS_PER_CHUNK = 2
CHUNK_TOKENS = BLOCKS_PER_CHUNK * BLOCK_SIZE
DTYPE = torch.float32


@pytest.fixture
def paged_layers() -> list[torch.Tensor]:
    """Per-layer KV in the native 6-D layout the detector reports."""
    torch.manual_seed(3)
    shape = (2, NUM_BLOCKS, NUM_HEADS, 1, BLOCK_SIZE, HEAD_SIZE)
    return [torch.randn(shape, dtype=DTYPE) for _ in range(NUM_LAYERS)]


@pytest.fixture
def chunks() -> list[torch.Tensor]:
    """Staging chunks, sized token-major as upstream allocates them."""
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


class _RecordingOps(RblnDeviceOps):
    """An instance with the native symbol bound, as ``bind_native`` leaves it."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []
        self.block_kv_transfer = self._record  # type: ignore[assignment]

    def _record(
        self,
        paged_layers: list[torch.Tensor],
        chunks: list[torch.Tensor],
        block_ids: list[int],
        direction: int,
        skip_prefix_n_blocks: int = 0,
    ) -> None:
        self.calls.append(
            {
                "paged_layers": paged_layers,
                "chunks": chunks,
                "block_ids": block_ids,
                "direction": direction,
                "skip_prefix_n_blocks": skip_prefix_n_blocks,
            }
        )


def _transfer(
    ops: RblnDeviceOps,
    layers: list[torch.Tensor],
    chunks: list[torch.Tensor],
    direction: TransferDirection = TransferDirection.D2H,
    block_ids: "list[int] | None" = None,
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
        0,
    )


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------


def test_missing_extension_leaves_the_torch_kernels_in_place() -> None:
    """No compiled extension is the ordinary case off an RBLN host."""
    ops = RblnDeviceOps()
    ops.ensure_native()
    assert getattr(ops, "block_kv_transfer", None) is None


# ---------------------------------------------------------------------------
# Declining
# ---------------------------------------------------------------------------


def test_cpu_paged_tensors_cannot_be_served(
    paged_layers: list[torch.Tensor], chunks: list[torch.Tensor]
) -> None:
    """The kernel DMAs from an RBLN vaddr; a CPU pointer is not one."""
    assert (
        native_can_serve(
            paged_layers, chunks, list(range(NUM_BLOCKS)), BLOCKS_PER_CHUNK
        )
        is False
    )


def _fake_rbln(contiguous: bool = True) -> Any:
    return SimpleNamespace(
        device=SimpleNamespace(type="rbln"), is_contiguous=lambda: contiguous
    )


def test_non_contiguous_operands_cannot_be_served(
    chunks: list[torch.Tensor],
) -> None:
    """A strided operand would be walked as if it were packed."""
    blocks = list(range(NUM_BLOCKS))
    assert (
        native_can_serve(
            [_fake_rbln(contiguous=False)], chunks, blocks, BLOCKS_PER_CHUNK
        )
        is False
    )
    strided = [chunk[:, :, ::2] for chunk in chunks]
    assert native_can_serve([_fake_rbln()], strided, blocks, BLOCKS_PER_CHUNK) is False


def test_device_chunks_cannot_be_served() -> None:
    """The chunk is the host end of the DMA, so it must live on the host."""
    assert (
        native_can_serve(
            [_fake_rbln()], [_fake_rbln()], list(range(NUM_BLOCKS)), BLOCKS_PER_CHUNK
        )
        is False
    )


def test_ragged_block_list_cannot_be_served(chunks: list[torch.Tensor]) -> None:
    """The kernel derives the chunk index from the flat block position.

    A block list that does not fill every chunk would make the tail land at
    another chunk's offsets, so a partial transfer stays on the torch path.
    """
    assert (
        native_can_serve(
            [_fake_rbln()], chunks, list(range(NUM_BLOCKS - 1)), BLOCKS_PER_CHUNK
        )
        is False
    )


def test_cpu_operands_fall_back_and_still_gather(
    paged_layers: list[torch.Tensor], chunks: list[torch.Tensor]
) -> None:
    """Declining must reach the torch kernels, not skip the transfer."""
    ops = _RecordingOps()
    _transfer(ops, paged_layers, chunks)
    assert ops.calls == []
    assert torch.count_nonzero(chunks[0]) > 0


# ---------------------------------------------------------------------------
# Dispatching
# ---------------------------------------------------------------------------


@pytest.fixture
def servable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make CPU operands eligible: dispatch is unreachable without an NPU
    otherwise. The predicate itself is asserted separately above."""
    monkeypatch.setattr(
        device_ops_module, "native_can_serve", lambda *args, **kwargs: True
    )


def test_dispatch_passes_flat_blocks_and_an_int_direction(
    servable: None, paged_layers: list[torch.Tensor], chunks: list[torch.Tensor]
) -> None:
    """The kernel takes a plain int, not the pybind enum from ``lmcache.c_ops``."""
    ops = _RecordingOps()
    _transfer(ops, paged_layers, chunks, direction=TransferDirection.H2D)

    (call,) = ops.calls
    assert call["block_ids"] == list(range(NUM_BLOCKS))
    assert call["direction"] == int(TransferDirection.H2D)
    assert type(call["direction"]) is int
    assert call["chunks"] == chunks


def test_dispatch_squeezes_the_singleton_axis_and_skips_the_torch_gather(
    servable: None, paged_layers: list[torch.Tensor], chunks: list[torch.Tensor]
) -> None:
    """A chunk the recording stub never filled proves the torch gather did not
    also run, and the squeezed shape proves the kernel gets the 5-D form its
    stride arithmetic assumes."""
    ops = _RecordingOps()
    _transfer(ops, paged_layers, chunks)

    assert len(ops.calls) == 1
    assert ops.calls[0]["paged_layers"][0].shape == (
        2,
        NUM_BLOCKS,
        NUM_HEADS,
        BLOCK_SIZE,
        HEAD_SIZE,
    )
    assert all(torch.count_nonzero(chunk) == 0 for chunk in chunks)
