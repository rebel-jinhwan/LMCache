# SPDX-License-Identifier: Apache-2.0
"""RBLN host-memory backend: ``torch.rbln.host_register`` (cudaHostRegister's
counterpart). Pages are pinned once and later copies inside the range skip the
per-command-buffer pin; the region's contents are untouched.
"""

# Future
from __future__ import annotations

# Standard
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend

logger = init_logger(__name__)


class RblnPinMemoryBackend(PinMemoryBackend):
    """Pin host memory for RBLN DMA by registering it with the runtime.

    A range is registered on the current device only, as ``host_register``
    does; :meth:`unpin_memory` unregisters it on that same device.

    Attributes:
        _registered: Start address -> device index of every range this backend
            registered, so :meth:`unpin_memory` only hands back pins it took.
    """

    def __init__(self) -> None:
        """Create the backend; it needs nothing beyond ``torch.rbln``."""
        self._registered: dict[int, int] = {}
        self._lock = threading.Lock()

    def pin_memory(self, ptr: int, size: int, flags: int = 0) -> bool:
        """Register a host region with the runtime for DMA.

        Args:
            ptr: Raw pointer (``data_ptr``) to the memory region.  Need not be
                page-aligned for the pin itself, but only page-aligned copy
                operands take the device-VA path.
            size: Size in bytes of the region.
            flags: Ignored.  RBLN has no host-registration flags; the argument
                is kept for the :class:`PinMemoryBackend` signature.

        Returns:
            True when the range is registered, False when the range is empty
            or ``host_register`` declined it (no RBLN device, an overlap with a
            live registration, a UMD without ``rblnRegisterHostMemory``).  A
            False leaves the region usable, just on the per-copy pin path.
        """
        del flags
        if ptr <= 0 or size <= 0:
            return False
        device = torch.rbln.current_device()
        if not torch.rbln.host_register(ptr, size, device=device):
            logger.warning(
                "RblnPinMemoryBackend: host_register(ptr=%#x, size=%d, device=%d) "
                "declined; the range stays on the per-copy pin path",
                ptr,
                size,
                device,
            )
            return False
        with self._lock:
            self._registered[ptr] = device
        return True

    def unpin_memory(self, ptr: int) -> bool:
        """Unregister a range pinned by :meth:`pin_memory`.

        The runtime drains the device's pending transfers before unpinning.

        Args:
            ptr: Raw pointer previously passed to :meth:`pin_memory`.

        Returns:
            True when the range is unregistered, False when this backend never
            registered it or the runtime refused.
        """
        with self._lock:
            device = self._registered.pop(ptr, None)
        if device is None:
            return False
        try:
            torch.rbln.host_unregister(ptr, device=device)
        except RuntimeError:
            logger.exception(
                "RblnPinMemoryBackend: host_unregister(ptr=%#x, device=%d) failed",
                ptr,
                device,
            )
            return False
        return True

    @property
    def is_pin_supported(self) -> bool:
        """Registration is part of torch-rbln; always available.

        Returns:
            True.
        """
        return True
