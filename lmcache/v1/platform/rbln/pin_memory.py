# SPDX-License-Identifier: Apache-2.0
"""RBLN host-memory backend: populate the mapping instead of registering it.

Every other :class:`~lmcache.v1.platform.base.pin_memory.PinMemoryBackend`
hands an existing host address to a runtime that records it -- CUDA's
``cudaHostRegister``, XPU's ``sycl::malloc_host``.  The RBLN runtime exposes no
such entry point: its only DMA-able-host-memory API (``torch.rbln
.huge_host_empty``) allocates a fresh buffer, which is no use for a region
another process already created and owns.

What the RBLN copy path actually wants from a host buffer is (a) a page-aligned
address, or it stages the transfer through a bounce buffer, and (b) pages that
are already faulted in, or the first touch takes a fault per 4 KiB *inside* the
transfer.  A POSIX SHM pool satisfies (a) by construction -- ``mmap`` returns a
page-aligned base and LMCache's ``AddressManager`` carves 4 KiB-aligned slots
out of it -- so (b) is what is left for this backend, and ``MADV_POPULATE_WRITE``
is how a mapping gets it without writing a byte of its content.

This is the same work the mp worker used to do downstream before RBLN mp moved
into LMCache; the pool-wide call site here (``EngineDrivenContextShm``, right
after the worker maps the server-owned pool) is the one it used.
"""

# Future
from __future__ import annotations

# Standard
import ctypes
import errno
import mmap
import os
import sys

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.pin_memory import PinMemoryBackend

logger = init_logger(__name__)

#: ``MADV_POPULATE_WRITE`` from Linux' ``include/uapi/asm-generic/mman-common.h``.
#: Populates the page tables of a mapping as if it had been written to, without
#: modifying its contents.  Added in Linux 5.14; older kernels report ``EINVAL``.
_MADV_POPULATE_WRITE = 23

_PAGE_SIZE = os.sysconf("SC_PAGESIZE")

#: ``MADV_POPULATE_WRITE`` reports ``EINTR`` when a signal arrives mid-walk,
#: having populated some prefix of the range.  Re-issuing the whole range is
#: cheap (populated pages are skipped), so a few retries absorb that.
_EINTR_RETRIES = 3


def _page_span(ptr: int, size: int) -> tuple[int, int]:
    """Round a byte range out to the pages that contain it.

    ``madvise`` rejects an unaligned start address, so the span is grown rather
    than trimmed: the page holding the first byte and the page holding the last
    byte are both part of the caller's mapping, so widening to them can never
    reach an unmapped page.

    Args:
        ptr: First byte of the region.
        size: Length of the region in bytes.

    Returns:
        ``(aligned_ptr, aligned_size)`` covering ``[ptr, ptr + size)``.
    """
    start = ptr & ~(_PAGE_SIZE - 1)
    end = (ptr + size + _PAGE_SIZE - 1) & ~(_PAGE_SIZE - 1)
    return start, end - start


class RblnPinMemoryBackend(PinMemoryBackend):
    """Prefault host mappings so RBLN DMA never faults inside a transfer.

    Attributes:
        _libc: ``ctypes`` handle on libc with ``madvise`` bound, or ``None``
            when it could not be loaded.
        _supported: Whether ``MADV_POPULATE_WRITE`` works on this kernel,
            decided once by :meth:`_probe`.
    """

    def __init__(self) -> None:
        """Bind ``madvise`` and probe ``MADV_POPULATE_WRITE`` once.

        Both the load and the probe are treated as capability questions, not
        errors: a failure leaves the backend reporting
        ``is_pin_supported = False``, which callers already handle by keeping
        their pageable path.
        """
        self._libc = self._load_libc()
        self._supported = self._probe()
        if not self._supported:
            logger.info(
                "RblnPinMemoryBackend: MADV_POPULATE_WRITE unavailable; "
                "host mappings stay lazily faulted"
            )

    @staticmethod
    def _load_libc() -> ctypes.CDLL | None:
        """Load libc with ``madvise`` bound to its C signature.

        Returns:
            The handle, or ``None`` off Linux or when the load fails.
        """
        if sys.platform != "linux":
            return None
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            libc.madvise.restype = ctypes.c_int
            return libc
        except (AttributeError, OSError) as exc:
            logger.debug("RblnPinMemoryBackend: failed to bind madvise: %s", exc)
            return None

    def _probe(self) -> bool:
        """Test ``MADV_POPULATE_WRITE`` on a throwaway page.

        Probing a private anonymous page rather than a caller's region keeps
        the answer free of any side effect on real data.

        Returns:
            True if the kernel accepted the advice.
        """
        if self._libc is None:
            return False
        try:
            probe = mmap.mmap(-1, _PAGE_SIZE)
        except (OSError, ValueError) as exc:
            logger.debug("RblnPinMemoryBackend: probe mapping failed: %s", exc)
            return False
        try:
            view = (ctypes.c_uint8 * 1).from_buffer(probe)
            addr = ctypes.addressof(view)
            del view  # drop the export before the mapping is closed
            return self._madvise(addr, _PAGE_SIZE)
        finally:
            probe.close()

    def _madvise(self, ptr: int, size: int) -> bool:
        """Issue ``madvise(MADV_POPULATE_WRITE)`` over an aligned span.

        Args:
            ptr: Page-aligned start address.
            size: Page-multiple length in bytes.

        Returns:
            True when the kernel populated the span.
        """
        if self._libc is None:
            return False
        for _ in range(_EINTR_RETRIES):
            ctypes.set_errno(0)
            if (
                self._libc.madvise(
                    ctypes.c_void_p(ptr),
                    ctypes.c_size_t(size),
                    _MADV_POPULATE_WRITE,
                )
                == 0
            ):
                return True
            err = ctypes.get_errno()
            if err != errno.EINTR:
                logger.debug(
                    "RblnPinMemoryBackend: madvise(ptr=%#x, size=%d) failed: %s",
                    ptr,
                    size,
                    os.strerror(err),
                )
                return False
        return False

    def pin_memory(self, ptr: int, size: int, flags: int = 0) -> bool:
        """Fault in every page of a host region so DMA into it does not.

        The region's contents are untouched, so this is safe to call on a
        shared pool that already holds cached KV -- which is what the mp worker
        does when it maps the server-owned SHM pool.

        Args:
            ptr: Raw pointer (``data_ptr``) to the memory region.  Need not be
                page-aligned; the span is widened to whole pages.
            size: Size in bytes of the region.
            flags: Ignored.  RBLN has no host-registration flags; the argument
                is kept for the :class:`PinMemoryBackend` signature.

        Returns:
            True when the whole region is now populated, False when the kernel
            lacks ``MADV_POPULATE_WRITE`` or refused the range (for instance
            ``ENOMEM`` on a pool larger than free memory).  A False leaves the
            region usable, just lazily faulted.
        """
        del flags
        if not self._supported or ptr <= 0 or size <= 0:
            return False
        aligned_ptr, aligned_size = _page_span(ptr, size)
        return self._madvise(aligned_ptr, aligned_size)

    def unpin_memory(self, ptr: int) -> bool:
        """Release nothing: population holds no resource to hand back.

        The pages stay resident until the mapping itself goes away.  Dropping
        them here would mean ``MADV_DONTNEED``, which discards the contents of
        a shared mapping -- the cached KV the pool exists to hold.

        Args:
            ptr: Raw pointer previously passed to :meth:`pin_memory`.

        Returns:
            True when this backend is active, mirroring
            :attr:`is_pin_supported`.
        """
        del ptr
        return self._supported

    @property
    def is_pin_supported(self) -> bool:
        """Whether host mappings can be populated on this kernel.

        Returns:
            True on Linux 5.14+, where ``MADV_POPULATE_WRITE`` exists.
        """
        return self._supported
