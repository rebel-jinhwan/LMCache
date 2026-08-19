# SPDX-License-Identifier: Apache-2.0
"""The only module in LMCache that imports the rebel runtime.

Everything else on the RBLN path reaches the device through torch. Direct
storage is the exception: moving device virtual memory to NVMe has no
torch-level expression, so ``rebel.rds`` has to be called directly. Keeping
that in one module makes the coupling a single documented surface, and the lazy
imports mean a host without the runtime pays nothing.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any, Sequence

# Third Party
import torch

if TYPE_CHECKING:
    # Third Party
    from rebel.rds import Chunk, Stream

# ``Chunk`` / ``Stream`` exist only under TYPE_CHECKING, so they are not listed.
__all__ = [
    "create_device_tensor_from_ptr",
    "mark_device_updated",
    "rds_chunk",
    "rds_stream",
    "vmem",
]


def is_available() -> bool:
    """Return whether the rebel runtime's direct-storage API can be imported."""
    try:
        # Third Party
        import rebel.rds  # noqa: F401
    except Exception:
        return False
    return True


def vmem() -> Any:
    """The ``rebel._C.vmem`` handle backing RDS DMA.

    Callers use it to fetch the device buffers for a vaddr
    (``get_device_buffers``), to own the host->device sync before a write
    (``sync_to_device``), and -- under ``vmem.debug`` -- to bind RDS areas.
    """
    # Third Party
    from rebel import _C

    return _C.vmem


def create_device_tensor_from_ptr(
    data_ptr: int, shape: Sequence[int], dtype: torch.dtype
) -> torch.Tensor:
    """Wrap an already-bound device vaddr as a ``torch.Tensor``.

    The tensor does not own the region; whoever bound the vaddr still frees it.
    Reaches a private torch-rbln symbol because no public equivalent is exported
    yet.
    """
    # Third Party
    from torch_rbln.device.device_tensor_utils import _create_tensor_from_ptr

    return _create_tensor_from_ptr(data_ptr, shape, dtype)


def rds_chunk(size: int, device: int) -> "Chunk":
    """Allocate an RDS chunk of ``size`` bytes on torch device ``device``."""
    # Third Party
    from rebel import rds

    return rds.Chunk(size=size, device=device)


def rds_stream(device: int, node_id: int) -> "Stream":
    """Create an RDS stream for batching async chunk reads and writes."""
    # Third Party
    from rebel import rds

    return rds.Stream(device=device, node_id=node_id)


def mark_device_updated(data_ptr: int) -> None:
    """Declare the device side of ``data_ptr`` the current state of the vmem.

    A stream ``Chunk.read`` DMAs NVMe -> device vmem without touching the sync
    state; only the synchronous read path does that internally. Without this the
    buffer keeps its pre-read contents and restored KV is silently stale.
    """
    # Third Party
    from rebel import rds

    rds.mark_device_updated(data_ptr)
