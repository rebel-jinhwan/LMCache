# SPDX-License-Identifier: Apache-2.0
"""Staging allocator for the RBLN RDS backend.

Hands out one full RBLN vmem area per :meth:`~RDSMemoryAllocator.allocate`,
recycled through a per-size free list, and owns the single ``rebel.rds.Chunk``
those areas are written to. A whole area rather than sub-ranges of one buffer
because ``rds`` transfers an *area* and requires the transfer size to equal the
area's size.

Where inside the chunk an object lands belongs to
:class:`~lmcache.v1.storage_backend.rds_backend.NvmeOffsetAllocator` instead --
see that class for why the two are split.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any, List, Optional, Union
import os
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
    get_size_bytes,
)
from lmcache.v1.platform.rbln import rds_runtime

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)

#: ``rds`` validates that an area's size and a chunk's file offset are both
#: multiples of this, so every staging buffer is rounded up to it.
RDS_ALIGN = 4096

#: Device vmem, in MiB, that RDS staging may hold at once. Both the in-flight
#: write cap and the read destinations come out of this one budget.
ENV_STAGING_CAP_MB = "LMCACHE_RBLN_RDS_STAGING_CAP_MB"

_DEFAULT_STAGING_CAP_MB = 2048


def _align_up(n: int, align: int) -> int:
    return (n + align - 1) // align * align


def staging_cap_bytes() -> int:
    """Device-vmem budget for RDS staging, in bytes (``<= 0`` disables caps)."""
    return int(os.environ.get(ENV_STAGING_CAP_MB, _DEFAULT_STAGING_CAP_MB)) * 1024**2


def kv_chunk_bytes(metadata: "Optional[LMCacheMetadata]") -> int:
    """Bytes of one KV chunk's staging buffer, from the engine metadata.

    Returns 0 when the geometry is unknown, which callers read as "do not cap".
    """
    if metadata is None:
        return 0
    try:
        return int(get_size_bytes([torch.Size(metadata.kv_shape)], [metadata.kv_dtype]))
    except Exception:  # noqa: BLE001 — unknown geometry -> disable capping
        return 0


def store_inflight_writes(metadata: "Optional[LMCacheMetadata]") -> int:
    """Concurrent in-flight store writes (K) the staging budget allows.

    Each in-flight write holds one chunk of staging, so K takes **half** the
    budget: reads allocate destinations from the same pool and can run during a
    store. Returns 0 when the budget or geometry is unknown.
    """
    cap_bytes = staging_cap_bytes()
    chunk_bytes = kv_chunk_bytes(metadata)
    if cap_bytes <= 0 or chunk_bytes <= 0:
        return 0
    return max(1, cap_bytes // chunk_bytes // 2)


class RDSMemoryAllocator(MemoryAllocatorInterface):
    """Pool of RBLN vmem staging areas, plus the shared ``rds.Chunk``.

    Each :meth:`allocate` hands out one full vmem area sized to the object's
    4096-aligned physical bytes, so its ``data_ptr()`` starts an area whose size
    equals the transfer size. :meth:`free` pushes it into a size-keyed pool for
    reuse, bounding live device memory at *peak concurrent* objects rather than
    at *total* allocations.
    """

    #: Not configurable: the node does not isolate the NVMe destination -- a
    #: write lands at ``(chunk_id, file_offset)`` regardless.
    _NODE_ID: int = 0

    def __init__(
        self,
        chunk_size: int,
        device: Optional[str] = None,
    ) -> None:
        """Bind the runtime and allocate the backing chunk.

        Args:
            chunk_size: Size of the ``rds.Chunk`` in bytes; must be a multiple
                of :data:`RDS_ALIGN`.
            device: Torch device the staging vmem is bound to. Defaults to
                ``"rbln:0"``.

        Raises:
            ValueError: If ``chunk_size`` is not 4096-aligned.
        """
        if device is None:
            device = "rbln:0"
        if chunk_size % RDS_ALIGN != 0:
            raise ValueError(
                f"chunk_size={chunk_size} must be a multiple of {RDS_ALIGN}"
            )

        torch_device_id = torch.device(device).index
        if torch_device_id is None:
            torch_device_id = 0

        self._device: str = device
        self._torch_device_id: int = torch_device_id
        self._chunk_size: int = chunk_size

        self._lock = threading.Lock()
        # phy_nbytes -> free uint8 tensors, each its own vmem area. Strong
        # references, so torch never reclaims them.
        self._tensor_pool: dict[int, list[torch.Tensor]] = {}

        # Bound here so a missing runtime fails at construction, not mid-transfer.
        self._vmem = rds_runtime.vmem()
        # Freed in ``close``: a wrapped tensor does not own its region.
        self._vaddrs: List[int] = []

        # Warm up the vmem manager BEFORE creating the chunk: a first
        # ``torch.empty(device='rbln')`` after the chunk exists invalidates its
        # registration, and later chunk ops fail with "unknown chunk_id".
        _ = torch.empty(RDS_ALIGN, dtype=torch.uint8, device=device)
        del _

        self._chunk: Optional[Any] = rds_runtime.rds_chunk(
            size=chunk_size, device=torch_device_id
        )
        logger.info(
            "Allocated RDS chunk id=%d size=%d on rbln device %d (node_id=%d)",
            self._chunk.chunk_id,
            chunk_size,
            torch_device_id,
            self._NODE_ID,
        )

    # ------------------------------------------------------------------
    # Backend-facing helpers
    # ------------------------------------------------------------------

    @property
    def chunk(self) -> Any:
        """The backing ``rds.Chunk``.

        Raises:
            RuntimeError: If the allocator has been closed.
        """
        if self._chunk is None:
            raise RuntimeError("RDSMemoryAllocator is closed")
        return self._chunk

    @property
    def vmem(self) -> Any:
        """The vmem handle backing RDS DMA.

        The backend uses it to fetch the buffers ``Chunk.write`` / ``Chunk.read``
        transfer, and to own the ``sync_to_device`` call before a write.
        """
        return self._vmem

    @property
    def torch_device_id(self) -> int:
        return self._torch_device_id

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def pool_size(self, phy_nbytes: Optional[int] = None) -> int:
        """Free-list length for one size class, or across all of them."""
        with self._lock:
            if phy_nbytes is None:
                return sum(len(v) for v in self._tensor_pool.values())
            return len(self._tensor_pool.get(phy_nbytes, []))

    def make_stream(self) -> Any:
        """Create an RDS stream bound to this allocator's device and node."""
        return rds_runtime.rds_stream(
            device=self._torch_device_id, node_id=self._NODE_ID
        )

    # ------------------------------------------------------------------
    # MemoryAllocatorInterface
    # ------------------------------------------------------------------

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        """Hand out a vmem staging buffer, reused from the pool when possible.

        ``metadata.address`` stays 0; the backend's ``NvmeOffsetAllocator``
        stamps the store path's ``file_offset`` onto it, and read destinations
        keep the 0.
        """
        shapes_l, dtypes_l = self._adapt_shapes_and_dtypes(shapes, dtypes)
        logical_nbytes = get_size_bytes(shapes_l, dtypes_l)
        phy_nbytes = _align_up(logical_nbytes, RDS_ALIGN)

        with self._lock:
            bucket = self._tensor_pool.get(phy_nbytes)
            raw_data = bucket.pop() if bucket else None

        if raw_data is None:
            # A bound vmem region wrapped as a tensor satisfies both
            # ``MemoryObj.tensor`` and ``Chunk.write``/``read``, with no
            # process-wide eager-malloc mode.
            vaddr = self._vmem.debug.allocate_and_bind_single_device(
                self._torch_device_id, phy_nbytes, self._NODE_ID
            )
            self._vaddrs.append(vaddr)
            raw_data = rds_runtime.create_device_tensor_from_ptr(
                vaddr, [phy_nbytes], torch.uint8
            )
        else:
            # The connector's gather writes through a reshaped view and does
            # not touch every byte of the area -- alignment padding survives it.
            # Fresh areas hold zeros there; recycled ones hold the previous
            # chunk's KV, which would ride along into the write.
            raw_data.zero_()

        meta = MemoryObjMetadata(
            shape=shapes_l[0],
            dtype=dtypes_l[0],
            address=0,
            phy_size=phy_nbytes,
            ref_count=1,
            pin_count=0,
            fmt=fmt,
            shapes=shapes_l,
            dtypes=dtypes_l,
        )
        return TensorMemoryObj(raw_data=raw_data, metadata=meta, parent_allocator=self)

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        """All-or-nothing :meth:`allocate` for ``batch_size`` objects."""
        objs: List[MemoryObj] = []
        for _ in range(batch_size):
            mo = self.allocate(shapes, dtypes, fmt)
            if mo is None:
                self.batched_free(objs)
                return None
            objs.append(mo)
        return objs

    def free(
        self,
        memory_obj: MemoryObj,
        allocator_type: Optional[str] = None,
    ) -> None:
        """Return an object's area to the pool, at most once.

        Every MemoryObj is freed twice -- ``ref_count_down`` reaching 0, then
        ``__del__`` on collection -- and by the second call the area is usually
        live inside a *new* MemoryObj, so a second push would alias two objects
        onto one area. ``TensorMemoryAllocator.free`` uses the same guard.
        """
        if getattr(memory_obj, "parent_allocator", None) is not self:
            return  # never handed out by this allocator; not ours to pool
        if not memory_obj.is_valid():
            return
        # ``.metadata`` is a property taking ``memory_obj.lock``, which
        # ``ref_count_down`` already holds -- reading it here deadlocks.
        raw = getattr(memory_obj, "raw_data", None)
        if raw is not None:
            with self._lock:
                self._tensor_pool.setdefault(memory_obj.meta.phy_size, []).append(raw)
        memory_obj.invalidate()

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ) -> None:
        for mo in memory_objs:
            self.free(mo)

    def memcheck(self) -> bool:
        """The pool is usable exactly while the backing chunk is open."""
        return self._chunk is not None

    def close(self) -> None:
        """Close the chunk and release every bound vmem region."""
        if self._chunk is not None:
            self._chunk.close()
            self._chunk = None
        with self._lock:
            self._tensor_pool.clear()
        for vaddr in self._vaddrs:
            try:
                self._vmem.debug.free(vaddr)
            except Exception:  # noqa: BLE001 — best effort during teardown
                logger.debug("failed to free RDS staging vaddr %d", vaddr)
        self._vaddrs.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 — never raise from a finalizer
            pass

    def __str__(self) -> str:
        return "RDSMemoryAllocator"
