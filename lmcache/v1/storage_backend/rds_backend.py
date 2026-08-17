# SPDX-License-Identifier: Apache-2.0
"""RBLN Direct Storage (RDS) backend: device vmem straight to NVMe.

The RBLN analogue of
:class:`~lmcache.v1.storage_backend.gds_backend.GdsBackend`: both skip host RAM,
because the KV chunk is already in device memory and routing it through a CPU
buffer costs a copy in each direction the storage path does not need.

What differs is who answers "where does this object go". GDS writes a file per
key and lets the filesystem answer it; ``rebel.rds`` exposes a flat, fixed-size
region with no filesystem, so :class:`NvmeOffsetAllocator` answers it here.

Scope:

- one fixed-size ``rebel.rds.Chunk`` per rank, sized from ``max_local_disk_size``
- in-memory metadata only, so a restart starts cold
- discard-only eviction: a full chunk drops the LRU entry and reuses its range,
  and ``pin`` / ``unpin`` are no-ops
"""

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Union, cast
import asyncio
import os
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    AddressManager,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.platform.rbln import rds_runtime
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.rds_memory_allocator import (
    RDS_ALIGN,
    RDSMemoryAllocator,
    store_inflight_writes,
)

logger = init_logger(__name__)

#: NVMe chunk size per rank, in GB, when ``max_local_disk_size`` is unset. Each
#: rank reserves its own, so this times the rank count must fit the device.
DEFAULT_CHUNK_SIZE_GB = 64

#: Override for the in-flight write cap K, the number of store batches that may
#: hold staging at once. See :meth:`RDSBackend.batched_submit_put_task`.
ENV_MAX_INFLIGHT_WRITES = "LMCACHE_RBLN_RDS_MAX_INFLIGHT_WRITES"


class NvmeOffsetAllocator:
    """Allocates byte ranges inside the backing ``rds.Chunk``.

    An :class:`AddressManager` over the chunk's byte-offset space, the same
    structure ``GDSL1MemoryManager`` uses over its slab file. Its default 4096
    alignment is exactly what ``rds`` requires of an area size and file offset.

    Owned by :class:`RDSBackend` rather than by the memory allocator: a range
    lives for exactly as long as its cache entry, so both the reservation and
    the release are driven by cache policy, which lives on the backend.
    """

    def __init__(
        self, chunk_size: int, align_bytes: int = AddressManager.ALIGN_BYTES
    ) -> None:
        self._chunk_size = chunk_size
        self._address_manager = AddressManager(chunk_size, align_bytes)

    def reserve(self, nbytes: int) -> Optional[int]:
        """Reserve ``nbytes`` and return its ``file_offset``, or None if full.

        A ``None`` is not fatal: the backend turns it into an LRU eviction and
        retries.
        """
        try:
            file_offset, _ = self._address_manager.allocate(nbytes)
        except RuntimeError:
            logger.warning(
                "RDS chunk full: need %d bytes, %d of %d free",
                nbytes,
                self._address_manager.get_free_size(),
                self._chunk_size,
            )
            return None
        return file_offset

    def release(self, file_offset: int, nbytes: int) -> None:
        """Return a range to the free space.

        Only for offsets of entries already popped from the cache, so a range is
        released exactly once.
        """
        self._address_manager.free(file_offset, nbytes)

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def bytes_in_use(self) -> int:
        return self._address_manager.total_allocated_size


@dataclass(frozen=True)
class _RdsEntry:
    """One stored chunk: where it is on NVMe and how to reconstruct its object."""

    file_offset: int
    size: int
    shape: torch.Size
    dtype: torch.dtype
    fmt: MemoryFormat


class RDSBackend(AllocatorBackendInterface):
    """LMCache storage backend that persists KV chunks to NVMe via RDS.

    1. :meth:`allocate` hands out a device staging area and reserves the NVMe
       range its contents will occupy; the connector then copies KV into
       ``memory_obj.tensor`` device-to-device.
    2. :meth:`batched_submit_put_task` enqueues one ``chunk.write`` per object
       inside a single stream and synchronises on stream exit.
    3. :meth:`get_blocking` allocates a fresh staging area, issues one
       ``chunk.read`` inside its own stream, and returns the populated object.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        dst_device: str = "rbln",
        local_cpu_backend: Optional[Any] = None,
    ) -> None:
        """Construct the backend.

        Args:
            config: Cache engine config; ``max_local_disk_size`` sizes the chunk.
            metadata: Engine metadata; supplies the worker index the staging
                vmem is pinned to and the KV geometry the write cap derives from.
            loop: Event loop the async writes are scheduled on.
            dst_device: Accepted for interface parity and ignored -- staging
                always lands in RBLN vmem on this worker's own NPU.
            local_cpu_backend: Accepted and ignored: RDS writes device vmem
                straight to NVMe, so there is no host buffer to share.
        """
        super().__init__(dst_device=dst_device)
        self.config = config
        self.metadata = metadata
        self.loop = loop

        self.memory_allocator: RDSMemoryAllocator = self.initialize_allocator(
            config, metadata
        )
        # Sized from the allocator's own chunk, so the two cannot disagree
        # about where it ends.
        self.nvme_offsets = NvmeOffsetAllocator(self.memory_allocator.chunk_size)

        self.hot_lock = threading.Lock()
        self.hot_cache: OrderedDict[CacheEngineKey, _RdsEntry] = OrderedDict()

        self.put_lock = threading.Lock()
        self.put_tasks: set[CacheEngineKey] = set()

        override = os.environ.get(ENV_MAX_INFLIGHT_WRITES)
        k = max(1, int(override)) if override is not None else 0
        self._write_slots = threading.BoundedSemaphore(
            k or store_inflight_writes(metadata) or 1
        )

    # ------------------------------------------------------------------
    # AllocatorBackendInterface
    # ------------------------------------------------------------------

    def initialize_allocator(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ) -> RDSMemoryAllocator:
        """Build the staging allocator on this worker's NPU."""
        # Not ``dst_device``: ``CreateStorageBackends`` derives that from
        # ``is_cuda_worker()`` and falls back to "cpu", which would stage the
        # KV chunk on the host.
        device = f"rbln:{metadata.local_worker_id}"
        return RDSMemoryAllocator(
            chunk_size=self._chunk_size_bytes(config),
            device=device,
        )

    @staticmethod
    def _chunk_size_bytes(config: LMCacheEngineConfig) -> int:
        """Size of the backing ``rds.Chunk``, from ``max_local_disk_size`` (GB).

        RDS is this platform's disk tier, so it takes the same capacity knob
        ``LocalDiskBackend`` uses. Per rank, like the chunk it sizes.
        """
        size_gb = float(getattr(config, "max_local_disk_size", 0) or 0)
        if size_gb <= 0:
            size_gb = DEFAULT_CHUNK_SIZE_GB
        # A fractional GB would miss the 4096 multiple rds requires.
        return int(size_gb * 1024**3) // RDS_ALIGN * RDS_ALIGN

    def get_memory_allocator(self) -> MemoryAllocatorInterface:
        return self.memory_allocator

    def get_allocator_backend(self) -> "RDSBackend":
        return self

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        return self._allocate_for_store(shapes, dtypes, fmt, eviction)

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        """All-or-nothing :meth:`_allocate_for_store` for ``batch_size`` objects.

        Each object evicts for itself, so a batch that needs N evictions costs
        one pass rather than N re-runs of the whole batch.
        """
        objs: list[MemoryObj] = []
        for _ in range(batch_size):
            mo = self._allocate_for_store(shapes, dtypes, fmt, eviction)
            if mo is None:
                for allocated in objs:
                    self._free_for_store(allocated)
                return None
            objs.append(mo)
        return objs

    def _allocate_for_store(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat,
        eviction: bool,
    ) -> Optional[MemoryObj]:
        """A staging buffer plus the NVMe range its contents will be written to.

        Buffer first, offset second: an offset reserved before a staging failure
        would be leaked, whereas a buffer whose reservation fails is handed back
        to the pool.

        The NVMe chunk is the scarce resource, so a full chunk discards LRU
        entries to reclaim their offsets and retries, giving up once there is
        nothing left to evict. The retry sits on ``reserve`` rather than around
        the whole allocation because that is the only failure eviction can
        undo: eviction frees NVMe ranges, never staging areas.
        """
        mo = self.memory_allocator.allocate(shapes, dtypes, fmt)
        if mo is None:
            return None
        nbytes = mo.get_physical_size()
        file_offset = self.nvme_offsets.reserve(nbytes)
        while file_offset is None and eviction and self._evict_one_lru():
            file_offset = self.nvme_offsets.reserve(nbytes)
        if file_offset is None:
            mo.ref_count_down()  # -> memory_allocator.free, back into the pool
            return None
        mo.metadata.address = file_offset
        return mo

    def _free_for_store(self, memory_obj: MemoryObj) -> None:
        """Undo :meth:`_allocate_for_store` -- release the range, then the buffer.

        Only for objects whose contents never reached NVMe. Once a write lands,
        the range outlives the MemoryObj and eviction releases it instead.
        """
        self.nvme_offsets.release(
            memory_obj.metadata.address, memory_obj.get_physical_size()
        )
        memory_obj.ref_count_down()

    def _evict_one_lru(self) -> bool:
        """Discard the LRU stored chunk to free its NVMe range.

        The evicted key becomes a miss and is recomputed; there is no
        write-back. :meth:`get_blocking` moves a hit to MRU so a key with an
        in-flight read is not the LRU, and a key mid-store is not in the cache
        yet, so neither is evicted here.

        Returns:
            ``False`` when there is nothing left to evict.
        """
        with self.hot_lock:
            try:
                key, entry = self.hot_cache.popitem(last=False)  # oldest = LRU
            except KeyError:
                return False
        self.nvme_offsets.release(entry.file_offset, entry.size)
        logger.info(
            "RDS eviction: discarded LRU key=%s (freed offset=%d size=%d)",
            key.to_string(),
            entry.file_offset,
            entry.size,
        )
        return True

    # ------------------------------------------------------------------
    # StorageBackendInterface -- lookup
    # ------------------------------------------------------------------

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.hot_lock:
            return key in self.hot_cache

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.put_lock:
            return key in self.put_tasks

    # ------------------------------------------------------------------
    # StorageBackendInterface -- put
    # ------------------------------------------------------------------

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Union[List[Future], None]:
        if len(keys) != len(objs):
            raise ValueError(
                f"keys ({len(keys)}) and objs ({len(objs)}) length mismatch"
            )
        if not keys:
            return []
        keys_list = list(keys)
        for mo in objs:
            mo.ref_count_up()
        with self.put_lock:
            self.put_tasks.update(keys_list)
        # Backpressure: acquired on the worker thread so an over-limit submit
        # blocks here instead of piling up staging.
        self._write_slots.acquire()
        future = asyncio.run_coroutine_threadsafe(
            self._async_batched_write(keys_list, objs, on_complete_callback),
            self.loop,
        )
        return [future]

    async def _async_batched_write(
        self,
        keys: List[CacheEngineKey],
        objs: List[MemoryObj],
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]],
    ) -> None:
        try:
            maybe_entries = [self._snapshot_entry(mo) for mo in objs]
            if any(entry is None for entry in maybe_entries):
                # The chunk cannot hold this batch. Give back what the entries
                # that did fit reserved, so a full chunk costs a skipped store
                # rather than a leak.
                logger.warning("RDS chunk full: dropping a batch of %d keys", len(keys))
                for entry in maybe_entries:
                    if entry is not None:
                        self.nvme_offsets.release(entry.file_offset, entry.size)
                return
            entries = cast(List[_RdsEntry], maybe_entries)
            try:
                await asyncio.to_thread(self._sync_batched_write, objs, entries)
            except Exception:
                logger.error(
                    "RDS batched write failed for %d keys", len(keys), exc_info=True
                )
                # Nothing will ever evict a range that never reached the
                # cache, so release it here or it is lost.
                for entry in entries:
                    self.nvme_offsets.release(entry.file_offset, entry.size)
                return
            # A re-store displaces the previous entry, whose range would
            # otherwise be stranded. Released outside ``hot_lock``, as
            # ``_evict_one_lru`` does.
            displaced: List[_RdsEntry] = []
            with self.hot_lock:
                for key, entry in zip(keys, entries, strict=True):
                    previous = self.hot_cache.get(key)
                    if previous is not None:
                        displaced.append(previous)
                    self.hot_cache[key] = entry
            for previous in displaced:
                self.nvme_offsets.release(previous.file_offset, previous.size)
            if on_complete_callback is not None:
                for key in keys:
                    try:
                        on_complete_callback(key)
                    except Exception:
                        logger.error(
                            "on_complete_callback failed for key %s",
                            key.to_string(),
                            exc_info=True,
                        )
        finally:
            for mo in objs:
                mo.ref_count_down()
            with self.put_lock:
                self.put_tasks.difference_update(keys)
            # Released only after the staging above was freed, so the cap
            # reflects live writes.
            self._write_slots.release()

    def _snapshot_entry(self, memory_obj: MemoryObj) -> Optional[_RdsEntry]:
        """Pin down where this object goes on NVMe, and what shape comes back.

        An object from :meth:`_allocate_for_store` already carries its range in
        ``metadata.address``, stamped when it was allocated. An object from
        another allocator carries *that* allocator's address, which means
        nothing here, so its range is reserved now -- evicting LRU entries to
        make room, exactly as the allocate path does.

        Args:
            memory_obj: The object about to be written.

        Returns:
            The entry to store it under, or ``None`` when the chunk is full and
            there is nothing left to evict.
        """
        meta = memory_obj.metadata
        nbytes = memory_obj.get_physical_size()
        if getattr(memory_obj, "parent_allocator", None) is self.memory_allocator:
            file_offset = meta.address
        else:
            reserved = self.nvme_offsets.reserve(nbytes)
            while reserved is None and self._evict_one_lru():
                reserved = self.nvme_offsets.reserve(nbytes)
            if reserved is None:
                return None
            file_offset = reserved
        return _RdsEntry(
            file_offset=file_offset,
            size=nbytes,
            shape=meta.shape,
            dtype=meta.dtype,
            fmt=meta.fmt,
        )

    def _transfer_area(
        self,
        op: Callable[..., None],
        vaddr: int,
        file_offset: int,
        stream: Any,
    ) -> None:
        """Run ``op`` (``chunk.write`` / ``chunk.read``) over the area at ``vaddr``.

        ``Chunk.write`` / ``Chunk.read`` take a device ``Buffer`` rather than a
        vaddr, so walk the area's buffers and lay them out contiguously from
        ``file_offset``. Neither direction syncs for the caller: writes need
        ``sync_to_device`` beforehand, reads need ``mark_device_updated`` after.
        """
        foff = file_offset
        for buf in self.memory_allocator.vmem.get_device_buffers(vaddr):
            op(buf, file_offset=foff, stream=stream)
            foff += buf.size

    def _sync_batched_write(
        self,
        objs: List[MemoryObj],
        entries: List[_RdsEntry],
    ) -> None:
        """Issue every ``chunk.write`` inside a single stream and synchronise.

        Addresses ``raw_data`` -- the vaddr of the whole area -- rather than
        ``tensor``, which may be a view onto part of it: the DMA moves buffer
        handles fetched from that vaddr, not an address range.

        An object this backend did not allocate has no vmem entry behind it, so
        it yields no buffers and cannot be a write source; it is copied into a
        staging area first. That is the ordinary case whenever a host pool is
        registered in front, and it costs one full-chunk copy per store.
        """
        chunk = self.memory_allocator.chunk
        staged: List[MemoryObj] = []
        try:
            with self.memory_allocator.make_stream() as stream:
                for mo, entry in zip(objs, entries, strict=True):
                    source = self._as_write_source(mo, staged)
                    vaddr = source.raw_data.data_ptr()
                    # ``Chunk.write`` does not sync host->device; we own it.
                    self.memory_allocator.vmem.sync_to_device(vaddr)
                    self._transfer_area(chunk.write, vaddr, entry.file_offset, stream)
        finally:
            # The stream is drained on exit, so the areas are free to reuse.
            for mo in staged:
                mo.ref_count_down()

    def _as_write_source(self, mo: MemoryObj, staged: List[MemoryObj]) -> MemoryObj:
        """Return ``mo`` if it is one of our areas, else a staged copy of it.

        Args:
            mo: The object the storage manager allocated and filled.
            staged: Collects the areas allocated here, for the caller to
                release once the writes are drained.

        Returns:
            MemoryObj: An object whose ``raw_data`` is a bound vmem area,
            so ``get_device_buffers`` can hand out its transfer handles.

        Raises:
            RuntimeError: If the staging pool cannot serve the copy.
        """
        if getattr(mo, "parent_allocator", None) is self.memory_allocator:
            return mo
        staging = self.memory_allocator.allocate(
            mo.metadata.shape, cast(torch.dtype, mo.metadata.dtype), mo.metadata.fmt
        )
        if staging is None:
            raise RuntimeError(
                "RDS write staging pool exhausted while copying a "
                f"{mo.metadata.phy_size}-byte object this backend did not "
                "allocate. Increase max_local_disk_size (chunk is currently "
                f"{self.memory_allocator.chunk_size} bytes)."
            )
        staged.append(staging)
        cast(torch.Tensor, staging.tensor).copy_(cast(torch.Tensor, mo.tensor))
        return staging

    # ------------------------------------------------------------------
    # StorageBackendInterface -- get
    # ------------------------------------------------------------------

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Single-key read -- the one-element case of :meth:`batched_get_blocking`.

        Only the LRU touch is extra: a lone hit has to be promoted so eviction
        targets genuinely old chunks.
        """
        with self.hot_lock:
            if key in self.hot_cache:
                self.hot_cache.move_to_end(key)
        return self.batched_get_blocking([key])[0]

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        with self.hot_lock:
            entries = [self.hot_cache.get(k) for k in keys]

        results: List[Optional[MemoryObj]] = []
        ops: List[tuple[MemoryObj, _RdsEntry, int]] = []
        for idx, entry in enumerate(entries):
            if entry is None:
                results.append(None)
                continue
            # No NVMe range is reserved: the read fills this buffer from the
            # stored entry's own offset and never writes it back.
            mo = self.memory_allocator.allocate(entry.shape, entry.dtype, entry.fmt)
            if mo is None:
                # The cache promised these entries, so surface the failure
                # here rather than returning ``None`` for some keys and
                # corrupting the model output downstream.
                raise RuntimeError(
                    f"RDS batched_get_blocking: staging pool exhausted while "
                    f"serving key={keys[idx].to_string()} ({idx + 1}/"
                    f"{len(keys)}). Increase max_local_disk_size (chunk is "
                    f"currently {self.memory_allocator.chunk_size} bytes)."
                )
            results.append(mo)
            ops.append((mo, entry, idx))

        if not ops:
            return results

        chunk = self.memory_allocator.chunk
        try:
            with self.memory_allocator.make_stream() as stream:
                for mo, entry, _ in ops:
                    self._transfer_area(
                        chunk.read,
                        mo.raw_data.data_ptr(),
                        entry.file_offset,
                        stream,
                    )
            # A stream read does not mark the data as the current device
            # state; without this the connector reads back pre-read contents.
            for mo, _entry, _ in ops:
                rds_runtime.mark_device_updated(mo.raw_data.data_ptr())
        except Exception:
            logger.error("RDS batched read failed", exc_info=True)
            with self.hot_lock:
                for _mo, _entry, idx in ops:
                    self.hot_cache.pop(keys[idx], None)
            for mo, _entry, idx in ops:
                mo.ref_count_down()
                results[idx] = None
        return results

    # ------------------------------------------------------------------
    # StorageBackendInterface -- pin / remove
    # ------------------------------------------------------------------

    def pin(self, key: CacheEngineKey) -> bool:
        """Pinning is not supported: eviction only ever discards."""
        return False

    def unpin(self, key: CacheEngineKey) -> bool:
        """Pinning is not supported: eviction only ever discards."""
        return False

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """Drop a stored chunk and reclaim its NVMe range.

        Returns:
            ``True`` if a chunk was removed, ``False`` if the key was absent.
        """
        with self.hot_lock:
            entry = self.hot_cache.pop(key, None)
        if entry is None:
            return False
        self.nvme_offsets.release(entry.file_offset, entry.size)
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.memory_allocator.close()
        logger.info("RDS backend closed")

    def __str__(self) -> str:
        return "RDSBackend"
