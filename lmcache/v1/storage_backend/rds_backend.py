# SPDX-License-Identifier: Apache-2.0
"""RBLN Direct Storage (RDS) backend: device vmem straight to NVMe.

The RBLN analogue of ``GdsBackend``. GDS writes a file per key and lets the
filesystem place it; ``rebel.rds`` exposes a flat region with no filesystem, so
an ``AddressManager`` over the chunk's byte offsets places it here.

One fixed-size ``rebel.rds.Chunk`` per rank, in-memory metadata only (a restart
starts cold), discard-only eviction, ``pin`` / ``unpin`` are no-ops.
"""

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Union
import asyncio
import threading
import time

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
    ENV_STAGING_CAP_MB,
    RDS_ALIGN,
    RDSMemoryAllocator,
)

logger = init_logger(__name__)


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
                vmem is pinned to.
            loop: Accepted for interface parity and ignored -- writes are
                synchronous, see :meth:`batched_submit_put_task`.
            dst_device: Accepted for interface parity and ignored -- staging
                always lands in RBLN vmem on this worker's own NPU.
            local_cpu_backend: Accepted and ignored: RDS writes device vmem
                straight to NVMe, so there is no host buffer to share.
        """
        super().__init__(dst_device=dst_device)
        self.config = config
        self.metadata = metadata

        self.memory_allocator: RDSMemoryAllocator = self.initialize_allocator(
            config, metadata
        )
        # A range outlives its MemoryObj and dies with its cache entry, so the
        # offset space lives here with the cache rather than on the allocator.
        # AddressManager's 4096 alignment is what rds requires of both.
        self.nvme_offsets = AddressManager(self.memory_allocator.chunk_size)

        self.hot_lock = threading.Lock()
        self.hot_cache: OrderedDict[CacheEngineKey, _RdsEntry] = OrderedDict()

        self.put_lock = threading.Lock()
        self.put_tasks: set[CacheEngineKey] = set()

        # GdsBackend's knobs and defaults.
        extra_config = config.extra_config or {}
        self._max_alloc_attempts: int = extra_config.get("max_alloc_attempts", 10)
        self._alloc_attempt_delay_secs: float = extra_config.get(
            "allocation_attempt_delay_secs", 0.1
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
        ``CreateStorageBackends`` only builds this backend when the knob is
        positive, so there is no default to fall back to.
        """
        # A fractional GB would miss the 4096 multiple rds requires.
        return int(config.max_local_disk_size * 1024**3) // RDS_ALIGN * RDS_ALIGN

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
        """One staging area plus its NVMe range, waiting out a full pool.

        Args:
            shapes: Logical tensor shape or shapes to allocate.
            dtypes: Logical tensor dtype or dtypes to allocate.
            fmt: Memory format stamped onto the returned object's metadata.
            eviction: Whether a full NVMe chunk may discard LRU entries.
            busy_loop: Whether to retry while the staging pool is at its cap.

        Returns:
            The object, or ``None`` if the pool stayed full for every attempt.
        """
        return self._allocate_for_store(shapes, dtypes, fmt, eviction, busy_loop)

    def _allocate_staging(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat,
        busy_loop: bool,
    ) -> Optional[MemoryObj]:
        """Take one area from the pool, waiting for in-flight writes to return
        one if the cap is reached. ``GdsBackend.allocate`` waits the same way.
        """
        max_attempts = self._max_alloc_attempts if busy_loop else 1
        for num_attempts in range(1, max_attempts + 1):
            mo = self.memory_allocator.allocate(shapes, dtypes, fmt)
            if mo is not None:
                return mo
            if num_attempts < max_attempts and self._alloc_attempt_delay_secs > 0:
                time.sleep(self._alloc_attempt_delay_secs)
        logger.warning(
            "RDS staging allocation failed after %d attempt(s): pool holds "
            "%d of %d bytes. Returning None.",
            max_attempts,
            self.memory_allocator.bound_bytes,
            self.memory_allocator.cap_bytes,
        )
        return None

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        """All-or-nothing allocation of ``batch_size`` objects.

        Args:
            shapes: Logical tensor shape or shapes to allocate.
            dtypes: Logical tensor dtype or dtypes to allocate.
            batch_size: How many objects to hand out.
            fmt: Memory format stamped onto each object's metadata.
            eviction: Whether a full NVMe chunk may discard LRU entries.
            busy_loop: Whether to wait on a staging pool at its cap.

        Returns:
            Every object, or ``None`` if any one of them could not be had.
        """
        objs: list[MemoryObj] = []
        for _ in range(batch_size):
            mo = self._allocate_for_store(shapes, dtypes, fmt, eviction, busy_loop)
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
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        """A staging buffer plus the NVMe range its contents will be written to.

        Buffer first, offset second: an offset reserved before a staging failure
        would be leaked, since nothing would reference it.

        The two shortages fail differently: a full pool is waited on, a full
        chunk evicts and retries, then gives up -- waiting cannot conjure a
        range eviction just failed to find.
        """
        mo = self._allocate_staging(shapes, dtypes, fmt, busy_loop)
        if mo is None:
            return None
        nbytes = mo.get_physical_size()
        file_offset = self._reserve_offset(nbytes)
        while file_offset is None and eviction and self._evict_one_lru():
            file_offset = self._reserve_offset(nbytes)
        if file_offset is None:
            mo.ref_count_down()  # -> memory_allocator.free, back into the pool
            return None
        mo.metadata.address = file_offset
        return mo

    def _reserve_offset(self, nbytes: int) -> Optional[int]:
        """Reserve ``nbytes`` of the chunk, or ``None`` when it is full.

        ``AddressManager`` raises on a full space; the caller wants that as a
        value, because a full chunk is an eviction to try, not an error.
        """
        try:
            file_offset, _ = self.nvme_offsets.allocate(nbytes)
        except RuntimeError:
            logger.warning(
                "RDS chunk full: need %d bytes, %d free",
                nbytes,
                self.nvme_offsets.get_free_size(),
            )
            return None
        return file_offset

    def _free_for_store(self, memory_obj: MemoryObj) -> None:
        """Undo :meth:`_allocate_for_store` -- release the range, then the buffer.

        Only for objects whose contents never reached NVMe. Once a write lands,
        the range outlives the MemoryObj and eviction releases it instead.
        """
        self.nvme_offsets.free(
            memory_obj.metadata.address, memory_obj.get_physical_size()
        )
        memory_obj.ref_count_down()

    def _evict_one_lru(self) -> bool:
        """Discard the LRU stored chunk to free its NVMe range, or return
        ``False`` when there is nothing left to evict. The key becomes a miss
        and is recomputed; there is no write-back tier below this one.
        """
        with self.hot_lock:
            try:
                key, entry = self.hot_cache.popitem(last=False)  # oldest = LRU
            except KeyError:
                return False
        self.nvme_offsets.free(entry.file_offset, entry.size)
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
        """Write every object to NVMe, then record the keys. Blocking.

        Deliberately synchronous. The DMA shares the node's PCIe with the
        expert all-to-all, so a write that outlives this call lands on top of
        the next forward's CCL traffic and inflates prefill. Returning only
        once ``chunk.write`` has drained removes that overlap by construction,
        with no barrier to place and nothing in flight to bound. Nothing is
        lost by waiting: ``StorageManager.batched_put`` discards the futures,
        and the batch's writes are already parallel inside one ``rds.Stream``.

        No ``ref_count_up`` here on purpose: ``batched_put`` holds its own
        reference across this call and drops it only after we return. The async
        version needed one because that reference could fall away mid-write.
        """
        if len(keys) != len(objs):
            raise ValueError(
                f"keys ({len(keys)}) and objs ({len(objs)}) length mismatch"
            )
        if not keys:
            return []
        keys_list = list(keys)
        with self.put_lock:
            self.put_tasks.update(keys_list)
        try:
            entries = [self._snapshot_entry(mo) for mo in objs]
            try:
                self._sync_batched_write(objs, entries)
            except Exception:
                logger.error(
                    "RDS batched write failed for %d keys", len(keys), exc_info=True
                )
                # No eviction can reach a range that never entered hot_cache.
                for entry in entries:
                    self.nvme_offsets.free(entry.file_offset, entry.size)
                return []
            # A re-store strands the displaced entry's range. Freed outside
            # ``hot_lock``, as ``_evict_one_lru`` does.
            displaced: List[_RdsEntry] = []
            with self.hot_lock:
                for key, entry in zip(keys_list, entries, strict=True):
                    previous = self.hot_cache.get(key)
                    if previous is not None:
                        displaced.append(previous)
                    self.hot_cache[key] = entry
            for previous in displaced:
                self.nvme_offsets.free(previous.file_offset, previous.size)
            if on_complete_callback is not None:
                for key in keys_list:
                    try:
                        on_complete_callback(key)
                    except Exception:
                        logger.error(
                            "on_complete_callback failed for key %s",
                            key.to_string(),
                            exc_info=True,
                        )
        finally:
            with self.put_lock:
                self.put_tasks.difference_update(keys_list)
        return []

    @staticmethod
    def _snapshot_entry(memory_obj: MemoryObj) -> _RdsEntry:
        """Where this object goes on NVMe, and what shape comes back.

        ``metadata.address`` carries the range ``_allocate_for_store`` reserved.
        Every object reaching here has one: ``StorageManager`` re-allocates each
        batch through ``get_allocator_backend()``, which is this backend.
        """
        meta = memory_obj.metadata
        if meta.dtype is None:
            # A restore rebuilds the tensor from this dtype. Defaulting it would
            # hand back float32-shaped garbage instead of failing.
            raise ValueError(
                "MemoryObj reached the RDS write path without a dtype; only "
                "objects from this backend's allocator can be stored"
            )
        return _RdsEntry(
            file_offset=meta.address,
            size=memory_obj.get_physical_size(),
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

        Addresses ``raw_data`` -- the whole area's vaddr -- not ``tensor``,
        which may be a view onto part of it: the DMA moves buffer handles
        fetched from that vaddr, not an address range.
        """
        chunk = self.memory_allocator.chunk
        with self.memory_allocator.make_stream() as stream:
            for mo, entry in zip(objs, entries, strict=True):
                vaddr = mo.raw_data.data_ptr()
                # ``Chunk.write`` does not sync host->device; the caller owns it.
                self.memory_allocator.vmem.sync_to_device(vaddr)
                self._transfer_area(chunk.write, vaddr, entry.file_offset, stream)

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
                # A restore wants every matched chunk's destination at once, so
                # a long prompt can outgrow the cap. Report a miss, as GDS does
                # on a full slab: the caller truncates its hit prefix here.
                # Retrying is pointless -- the pool holds this batch's own
                # destinations, which only the caller can release.
                logger.warning(
                    "RDS staging pool full serving key=%s (%d/%d); reporting a "
                    "miss so the caller recomputes from here. Raise %s to keep "
                    "longer prefixes.",
                    keys[idx].to_string(),
                    idx + 1,
                    len(keys),
                    ENV_STAGING_CAP_MB,
                )
                results.append(None)
                continue
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
        self.nvme_offsets.free(entry.file_offset, entry.size)
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.memory_allocator.close()
        logger.info("RDS backend closed")

    def __str__(self) -> str:
        return "RDSBackend"
