# SPDX-License-Identifier: Apache-2.0
"""RBLN KV-cache IPC wrapper (dma-buf based) -- scaffold.

The exporting worker turns the tensor's device VA into a dma-buf fd
(``rblnExportMemoryByDva``); the importing server maps that fd into its own
context (``rblnImportBoMemory``) and rebuilds a tensor view over it.

Constraints from the UMD side (to be confirmed as the driver work lands):

- Export requires the buffer to have been allocated with ``num_task == 1``.
- Import requires the exported buffer to be ``DRAM && PRIVATE``.
- Import goes through the DRM-backed ``rbln_bo`` path, so it needs kernel
  >= 6.2.
- Passing the fd between processes needs ``SCM_RIGHTS`` (or ``pidfd_getfd``);
  the raw integer in ``handle`` is only valid inside the exporting process.

Both constructors below raise ``NotImplementedError`` until torch_rbln exposes
the export/import entry points.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# Third Party
import torch

# First Party
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper


class RblnIPCWrapper(DeviceIPCWrapper):
    """Ship an RBLN KV tensor across the multiprocess wire as a dma-buf fd."""

    device_type: ClassVar[str] = "rbln"

    @classmethod
    def wrap(cls, tensor: torch.Tensor) -> "RblnIPCWrapper":
        """Factory used by :func:`lmcache.v1.platform.resolve_kv_wrapper_factory`.

        Args:
            tensor: An RBLN tensor allocated with ``num_task == 1``.

        Returns:
            A new wrapper ready for the wire.
        """
        return cls(tensor)

    def __init__(self, tensor: torch.Tensor) -> None:
        # TODO(rbln): tensor.data_ptr() -> rblnExportMemoryByDva -> dma-buf fd.
        # Store fd, nbytes, dtype/shape/stride/storage_offset, device_uuid.
        raise NotImplementedError(
            "RblnIPCWrapper export is not implemented: torch_rbln does not "
            "yet expose dma-buf export for device tensors."
        )

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct the tensor in this process via dma-buf import."""
        # TODO(rbln): rblnImportBoMemory(fd) -> dva -> tensor view
        # (dtype/shape/stride/storage_offset). Keep the bo alive as owner.
        raise NotImplementedError(
            "RblnIPCWrapper import is not implemented: torch_rbln does not "
            "yet expose dma-buf import for device tensors."
        )
