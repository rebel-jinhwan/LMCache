# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``lmcache.rbln_ops`` adapter.

The compiled extension only exists on a host with a rebel runtime, so what CI
can pin is the *decision*: when the adapter takes the native path, when it
declines, and what it hands the kernel when it does. The kernel's own
correctness is a hardware test.

The declines matter more than they look. Every one of them is a case where the
native kernel would compute an address from geometry that does not hold --
addressing wrong bytes rather than raising -- so a gate that silently stopped
working would be a data-corruption bug, not a performance regression.
"""

# Standard
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.ops_types import (
    EngineKVFormat,
    PageBufferShapeDesc,
    TransferDirection,
)
from lmcache.v1.platform.rbln import native_kv_transfer
from lmcache.v1.platform.rbln.device_ops import RblnDeviceOps

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
    """Per-layer KV in the squeezed 5-D form the native kernel takes."""
    torch.manual_seed(3)
    shape = (2, NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_SIZE)
    return [torch.randn(shape, dtype=DTYPE) for _ in range(NUM_LAYERS)]


@pytest.fixture
def chunks() -> list[torch.Tensor]:
    """Staging chunks, sized token-major as upstream allocates them."""
    return [
        torch.zeros((2, NUM_LAYERS, CHUNK_TOKENS, NUM_HEADS * HEAD_SIZE), dtype=DTYPE)
        for _ in range(NUM_BLOCKS // BLOCKS_PER_CHUNK)
    ]


class _RecordingModule:
    """Stand-in for ``lmcache.rbln_ops`` that records one call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def head_major_block_kv_transfer(
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


@pytest.fixture
def native(monkeypatch: pytest.MonkeyPatch) -> _RecordingModule:
    """Install a recording module and make CPU tensors look eligible.

    Pointing the device predicate at CPU is the only way to reach the dispatch
    path without an NPU; the predicate itself is asserted separately in
    :func:`test_cpu_paged_tensors_are_not_eligible`.
    """
    module = _RecordingModule()
    monkeypatch.setattr(native_kv_transfer, "load_native_module", lambda: module)
    monkeypatch.setattr(
        native_kv_transfer,
        "is_native_paged_tensor",
        lambda tensor: tensor.is_contiguous(),
    )
    monkeypatch.delenv(native_kv_transfer.ENV_RBLN_NATIVE_KV_TRANSFER, raising=False)
    return module


def _try(
    paged_layers: list[torch.Tensor],
    chunks: list[torch.Tensor],
    *,
    block_ids: list[int] | None = None,
    direction: TransferDirection = TransferDirection.D2H,
    skip_prefix_n_blocks: int = 0,
) -> bool:
    return native_kv_transfer.try_head_major_block_kv_transfer(
        paged_layers=paged_layers,
        chunks=chunks,
        block_ids=list(range(NUM_BLOCKS)) if block_ids is None else block_ids,
        blocks_per_chunk=BLOCKS_PER_CHUNK,
        direction=direction,
        skip_prefix_n_blocks=skip_prefix_n_blocks,
    )


# ---------------------------------------------------------------------------
# Declining
# ---------------------------------------------------------------------------


def test_missing_extension_declines(
    monkeypatch: pytest.MonkeyPatch,
    paged_layers: list[torch.Tensor],
    chunks: list[torch.Tensor],
) -> None:
    """No compiled extension is the ordinary case off an RBLN host."""
    monkeypatch.setattr(native_kv_transfer, "load_native_module", lambda: None)
    assert _try(paged_layers, chunks) is False


def test_env_kill_switch_declines(
    monkeypatch: pytest.MonkeyPatch,
    native: _RecordingModule,
    paged_layers: list[torch.Tensor],
    chunks: list[torch.Tensor],
) -> None:
    """``LMCACHE_RBLN_NATIVE_KV_TRANSFER=0`` forces the torch kernels."""
    monkeypatch.setenv(native_kv_transfer.ENV_RBLN_NATIVE_KV_TRANSFER, "0")
    assert _try(paged_layers, chunks) is False
    assert native.calls == []


def test_cpu_paged_tensors_are_not_eligible(
    paged_layers: list[torch.Tensor],
    chunks: list[torch.Tensor],
) -> None:
    """The kernel DMAs from an RBLN vaddr; a CPU pointer is not one."""
    assert native_kv_transfer.is_native_paged_tensor(paged_layers[0]) is False
    assert native_kv_transfer.is_native_chunk_tensor(chunks[0]) is True


def test_non_contiguous_chunk_is_not_eligible(chunks: list[torch.Tensor]) -> None:
    """A strided chunk would be walked as if it were packed."""
    assert native_kv_transfer.is_native_chunk_tensor(chunks[0][:, :, ::2]) is False


def test_device_chunk_is_not_eligible(chunks: list[torch.Tensor]) -> None:
    """The chunk is the host end of the DMA, so it must live on the host."""
    on_device = SimpleNamespace(
        device=SimpleNamespace(type="rbln"), is_contiguous=lambda: True
    )
    assert native_kv_transfer.is_native_chunk_tensor(on_device) is False  # type: ignore[arg-type]


def test_ragged_block_list_declines(
    native: _RecordingModule,
    paged_layers: list[torch.Tensor],
    chunks: list[torch.Tensor],
) -> None:
    """The kernel derives the chunk index from the flat block position.

    A block list that does not fill every chunk would therefore make the last
    chunk's blocks land at another chunk's offsets, so a partial transfer stays
    on the torch path.
    """
    assert _try(paged_layers, chunks, block_ids=list(range(NUM_BLOCKS - 1))) is False
    assert native.calls == []


# ---------------------------------------------------------------------------
# Dispatching
# ---------------------------------------------------------------------------


def test_dispatch_passes_flat_blocks_and_int_direction(
    native: _RecordingModule,
    paged_layers: list[torch.Tensor],
    chunks: list[torch.Tensor],
) -> None:
    """The kernel takes plain ints, not the pybind enum from ``lmcache.c_ops``."""
    assert _try(paged_layers, chunks, direction=TransferDirection.H2D) is True

    (call,) = native.calls
    assert call["block_ids"] == list(range(NUM_BLOCKS))
    assert call["direction"] == int(TransferDirection.H2D)
    assert type(call["direction"]) is int
    assert call["chunks"] == chunks
    assert call["paged_layers"] == paged_layers


def test_device_ops_skips_the_torch_kernels_when_native_ran(
    native: _RecordingModule,
    chunks: list[torch.Tensor],
) -> None:
    """``RblnDeviceOps`` must not gather twice — the chunks stay untouched.

    The native kernel writes through raw pointers, so a chunk the recording
    stub never filled proves the torch gather did not also run.
    """
    torch.manual_seed(5)
    layers = [
        torch.randn((2, NUM_BLOCKS, NUM_HEADS, 1, BLOCK_SIZE, HEAD_SIZE), dtype=DTYPE)
        for _ in range(NUM_LAYERS)
    ]
    desc = PageBufferShapeDesc()
    desc.kv_size = 2
    desc.nl = NUM_LAYERS
    desc.nb = NUM_BLOCKS
    desc.bs = BLOCK_SIZE
    desc.nh = NUM_HEADS
    desc.hs = HEAD_SIZE
    desc.element_size = DTYPE.itemsize

    RblnDeviceOps().multi_layer_block_kv_transfer(
        layers,
        chunks,
        list(range(NUM_BLOCKS)),
        torch.device("cpu"),
        TransferDirection.D2H,
        desc,
        CHUNK_TOKENS,
        EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS,
        0,
    )

    assert len(native.calls) == 1
    # The singleton axis is squeezed before the kernel sees the layers.
    assert native.calls[0]["paged_layers"][0].shape == (
        2,
        NUM_BLOCKS,
        NUM_HEADS,
        BLOCK_SIZE,
        HEAD_SIZE,
    )
    assert all(torch.count_nonzero(chunk) == 0 for chunk in chunks)
