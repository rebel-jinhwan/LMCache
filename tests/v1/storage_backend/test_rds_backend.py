# SPDX-License-Identifier: Apache-2.0
"""Tests for the RBLN RDS storage backend.

No NPU and no rebel runtime: :mod:`lmcache.v1.platform.rbln.rds_runtime` is
replaced by a fake whose "device areas" are CPU tensors and whose "NVMe chunk"
is a bytearray that copies real bytes at real offsets. A store followed by a
read therefore proves the offsets the backend computes address what it wrote.

The range-lifetime tests carry the rest of the weight. A range released twice
hands the same NVMe bytes to two live keys; one never released is lost for the
chunk's lifetime. Neither raises, so each path is asserted on ``bytes_in_use``.
"""

# Standard
from typing import Any, Optional
from unittest.mock import MagicMock
import asyncio
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend import rds_backend as rds_backend_module
from lmcache.v1.storage_backend import rds_memory_allocator as allocator_module
from lmcache.v1.storage_backend.rds_backend import RDSBackend
from lmcache.v1.storage_backend.rds_memory_allocator import (
    RDS_ALIGN,
    RDSMemoryAllocator,
)

CHUNK_SIZE = 1 * 1024 * 1024  # 1 MiB of "NVMe"
KV_SHAPE = torch.Size([2, 2, 8, 16])  # 512 elements
KV_DTYPE = torch.float32


# ---------------------------------------------------------------------------
# A fake rebel.rds
# ---------------------------------------------------------------------------


class _FakeBuffer:
    """One contiguous device buffer, backed by a CPU tensor."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self.size = tensor.numel()


class _FakeChunk:
    """A flat NVMe region that really copies bytes at the given offsets."""

    chunk_id = 7

    def __init__(self, size: int) -> None:
        self.storage = bytearray(size)
        self.closed = False
        self.writes = 0
        self.reads = 0

    def write(self, buf: _FakeBuffer, file_offset: int, stream: Any) -> None:
        self.writes += 1
        self.storage[file_offset : file_offset + buf.size] = (
            buf.tensor.numpy().tobytes()
        )

    def read(self, buf: _FakeBuffer, file_offset: int, stream: Any) -> None:
        self.reads += 1
        window = self.storage[file_offset : file_offset + buf.size]
        buf.tensor.copy_(torch.frombuffer(bytearray(window), dtype=torch.uint8))

    def close(self) -> None:
        self.closed = True


class _FakeStream:
    def __enter__(self) -> "_FakeStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class _FakeRuntime:
    """Stands in for ``lmcache.v1.platform.rbln.rds_runtime``."""

    def __init__(self) -> None:
        self.areas: dict[int, torch.Tensor] = {}  # data_ptr -> backing tensor
        self.chunk: Optional[_FakeChunk] = None
        self.synced: list[int] = []
        self.marked: list[int] = []
        self._next_vaddr = 1

        debug = MagicMock()
        debug.allocate_and_bind_single_device.side_effect = self._bind
        debug.free.side_effect = lambda vaddr: None
        self.vmem_handle = MagicMock()
        self.vmem_handle.debug = debug
        self.vmem_handle.get_device_buffers.side_effect = self._device_buffers
        self.vmem_handle.sync_to_device.side_effect = self.synced.append

        self._pending: dict[int, torch.Tensor] = {}

    def _bind(self, device_id: int, nbytes: int, node_id: int) -> int:
        vaddr = self._next_vaddr
        self._next_vaddr += 1
        self._pending[vaddr] = torch.zeros(nbytes, dtype=torch.uint8)
        return vaddr

    def _device_buffers(self, data_ptr: int) -> list[_FakeBuffer]:
        return [_FakeBuffer(self.areas[data_ptr])]

    # -- the rds_runtime surface --------------------------------------

    def vmem(self) -> Any:
        return self.vmem_handle

    def create_device_tensor_from_ptr(
        self, vaddr: int, shape: list[int], dtype: torch.dtype
    ) -> torch.Tensor:
        tensor = self._pending.pop(vaddr)
        self.areas[tensor.data_ptr()] = tensor
        return tensor

    def rds_chunk(self, size: int, device: int) -> _FakeChunk:
        self.chunk = _FakeChunk(size)
        return self.chunk

    def rds_stream(self, device: int, node_id: int) -> _FakeStream:
        return _FakeStream()

    def mark_device_updated(self, data_ptr: int) -> None:
        self.marked.append(data_ptr)


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> _FakeRuntime:
    """Install the fake runtime and pin staging to CPU tensors."""
    fake = _FakeRuntime()
    monkeypatch.setattr(allocator_module, "rds_runtime", fake)
    monkeypatch.setattr(rds_backend_module, "rds_runtime", fake)

    original = RDSMemoryAllocator.__init__

    def cpu_init(self: RDSMemoryAllocator, chunk_size: int, device: Any = None) -> None:
        # The warm-up is a real ``torch.empty`` on the staging device, so CPU
        # stands in for the NPU.
        original(self, chunk_size=chunk_size, device="cpu")

    monkeypatch.setattr(RDSMemoryAllocator, "__init__", cpu_init)
    return fake


@pytest.fixture
def loop() -> Any:
    """A background event loop for the async write path."""
    new_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=new_loop.run_forever, daemon=True)
    thread.start()
    yield new_loop
    new_loop.call_soon_threadsafe(new_loop.stop)
    thread.join(timeout=5)
    new_loop.close()


def _config(max_local_disk_size: float = CHUNK_SIZE / 1024**3) -> Any:
    config = MagicMock()
    config.max_local_disk_size = max_local_disk_size
    config.extra_config = {}
    return config


def _metadata() -> Any:
    metadata = MagicMock()
    metadata.local_worker_id = 0
    metadata.kv_shape = KV_SHAPE
    metadata.kv_dtype = KV_DTYPE
    return metadata


@pytest.fixture
def backend(runtime: _FakeRuntime, loop: Any) -> Any:
    """A backend whose staging and NVMe are both fakes."""
    made = RDSBackend(_config(), _metadata(), loop)
    yield made
    made.close()


def _key(name: str) -> CacheEngineKey:
    return CacheEngineKey("model", 1, 0, hash(name) & 0xFFFFFFFF, KV_DTYPE)


def _store(backend: Any, key: CacheEngineKey, fill: float) -> None:
    """Allocate, fill, and synchronously complete one store."""
    mo = backend.allocate(KV_SHAPE, KV_DTYPE, MemoryFormat.KV_2LTD)
    assert mo is not None
    mo.tensor.fill_(fill)
    futures = backend.batched_submit_put_task([key], [mo])
    assert futures is not None
    for future in futures:
        future.result(timeout=10)
    mo.ref_count_down()


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_store_then_read_returns_the_stored_bytes(
    backend: Any, runtime: _FakeRuntime
) -> None:
    """The offsets the backend computes address what it wrote."""
    key = _key("a")
    _store(backend, key, fill=1.5)
    assert backend.contains(key)

    got = backend.get_blocking(key)
    assert got is not None
    assert torch.equal(got.tensor, torch.full(KV_SHAPE, 1.5, dtype=KV_DTYPE))
    assert runtime.chunk is not None and runtime.chunk.reads == 1
    got.ref_count_down()


def test_the_backend_allocates_for_its_own_writes(backend: Any) -> None:
    """``rds`` writes areas it bound, so the manager must allocate from here.

    ``StorageManager.batched_put`` re-allocates each batch through the
    destination's ``get_allocator_backend()``, so reporting itself is what
    keeps a foreign ``MemoryObj`` -- which has no vmem entry, hence no
    transferable buffer -- from ever reaching the write path.
    """
    assert backend.get_allocator_backend() is backend
    mo = backend.allocate(KV_SHAPE, KV_DTYPE, MemoryFormat.KV_2LTD)
    assert mo is not None
    assert mo.parent_allocator is backend.memory_allocator
    # The NVMe range is reserved and stamped at allocation, which is where the
    # write path reads it from.
    assert mo.metadata.address is not None
    assert backend.nvme_offsets.bytes_in_use == mo.get_physical_size()
    mo.ref_count_down()


def test_two_keys_do_not_overlap_on_nvme(backend: Any) -> None:
    """Distinct keys get distinct ranges, so neither read sees the other."""
    first, second = _key("a"), _key("b")
    _store(backend, first, fill=1.0)
    _store(backend, second, fill=2.0)

    got_first = backend.get_blocking(first)
    got_second = backend.get_blocking(second)
    assert got_first is not None and got_second is not None
    assert float(got_first.tensor.flatten()[0]) == 1.0
    assert float(got_second.tensor.flatten()[0]) == 2.0
    got_first.ref_count_down()
    got_second.ref_count_down()


def test_missing_key_reads_as_none(backend: Any) -> None:
    assert backend.get_blocking(_key("absent")) is None


def test_read_marks_the_staging_area_device_updated(
    backend: Any, runtime: _FakeRuntime
) -> None:
    """A stream read does not update the vmem sync state on its own.

    Without the call, the connector reads back pre-read contents and the
    restored KV is silently stale.
    """
    key = _key("a")
    _store(backend, key, fill=3.0)
    got = backend.get_blocking(key)
    assert got is not None
    assert runtime.marked == [got.raw_data.data_ptr()]
    got.ref_count_down()


# ---------------------------------------------------------------------------
# Range lifetime
# ---------------------------------------------------------------------------


def test_store_reserves_and_eviction_releases(backend: Any) -> None:
    key = _key("a")
    _store(backend, key, fill=1.0)
    reserved = backend.nvme_offsets.bytes_in_use
    assert reserved > 0

    assert backend._evict_one_lru() is True
    assert backend.nvme_offsets.bytes_in_use == 0
    assert backend.contains(key) is False
    assert backend._evict_one_lru() is False


def test_remove_releases_the_range(backend: Any) -> None:
    key = _key("a")
    _store(backend, key, fill=1.0)
    assert backend.remove(key) is True
    assert backend.nvme_offsets.bytes_in_use == 0
    assert backend.remove(key) is False


def test_restoring_a_live_key_releases_the_displaced_range(backend: Any) -> None:
    """A re-store leaves the old entry unreachable; its range must come back."""
    key = _key("a")
    _store(backend, key, fill=1.0)
    after_first = backend.nvme_offsets.bytes_in_use
    _store(backend, key, fill=2.0)

    assert backend.nvme_offsets.bytes_in_use == after_first
    got = backend.get_blocking(key)
    assert got is not None
    assert float(got.tensor.flatten()[0]) == 2.0
    got.ref_count_down()


def test_failed_write_releases_its_range(
    backend: Any, runtime: _FakeRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing else can free a range that never reached the cache."""

    def explode(*args: Any, **kwargs: Any) -> None:
        raise OSError("nvme write failed")

    monkeypatch.setattr(runtime.chunk, "write", explode)
    _store(backend, _key("a"), fill=1.0)

    assert backend.nvme_offsets.bytes_in_use == 0
    assert backend.contains(_key("a")) is False


def test_allocation_evicts_lru_when_the_chunk_is_full(backend: Any) -> None:
    """The chunk cap is the scarce resource: allocate must reclaim, not fail."""
    stored: list[CacheEngineKey] = []
    while True:
        key = _key("k%d" % len(stored))
        mo = backend.allocate(KV_SHAPE, KV_DTYPE, MemoryFormat.KV_2LTD, eviction=False)
        if mo is None:
            break
        backend._free_for_store(mo)
        _store(backend, key, fill=float(len(stored)))
        stored.append(key)
        assert len(stored) < 1000, "chunk never filled"

    oldest = stored[0]
    assert backend.contains(oldest)
    mo = backend.allocate(KV_SHAPE, KV_DTYPE, MemoryFormat.KV_2LTD)
    assert mo is not None, "eviction should have made room"
    assert backend.contains(oldest) is False
    backend._free_for_store(mo)


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def test_chunk_size_comes_from_max_local_disk_size() -> None:
    """RDS is the disk tier, so it takes the disk tier's capacity knob."""
    assert RDSBackend._chunk_size_bytes(_config(max_local_disk_size=2)) == 2 * 1024**3


def test_chunk_size_is_aligned_and_defaulted() -> None:
    fractional = RDSBackend._chunk_size_bytes(_config(max_local_disk_size=0.5001))
    assert fractional % RDS_ALIGN == 0

    default = RDSBackend._chunk_size_bytes(_config(max_local_disk_size=0))
    assert default == rds_backend_module.DEFAULT_CHUNK_SIZE_GB * 1024**3


# ---------------------------------------------------------------------------
# The staging cap
# ---------------------------------------------------------------------------


def test_staging_pool_refuses_to_grow_past_its_cap(
    runtime: _FakeRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Binding on demand has no ceiling of its own, so the cap has to be it.

    ``CuFileMemoryAllocator`` gets this for free: it sub-allocates a slab it
    registered once, so a caller that asks for more than fits is told ``None``.
    Here every miss binds a fresh area, and a long prompt's allocation loop
    runs once per chunk before a single write is submitted -- the in-flight
    write semaphore is downstream of all of it and cannot bound this.
    """
    monkeypatch.setattr(allocator_module, "staging_cap_bytes", lambda: 2 * RDS_ALIGN)
    allocator = RDSMemoryAllocator(chunk_size=CHUNK_SIZE)
    try:
        first = allocator.allocate(KV_SHAPE, KV_DTYPE, MemoryFormat.KV_2LTD)
        second = allocator.allocate(KV_SHAPE, KV_DTYPE, MemoryFormat.KV_2LTD)
        assert first is not None
        assert second is not None
        assert allocator.bound_bytes == 2 * RDS_ALIGN

        assert allocator.allocate(KV_SHAPE, KV_DTYPE, MemoryFormat.KV_2LTD) is None
        assert allocator.bound_bytes == 2 * RDS_ALIGN
    finally:
        allocator.close()


def test_a_recycled_area_is_handed_out_at_the_cap(
    runtime: _FakeRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap counts device memory held, not objects outstanding."""
    monkeypatch.setattr(allocator_module, "staging_cap_bytes", lambda: RDS_ALIGN)
    allocator = RDSMemoryAllocator(chunk_size=CHUNK_SIZE)
    try:
        first = allocator.allocate(KV_SHAPE, KV_DTYPE, MemoryFormat.KV_2LTD)
        assert first is not None
        assert allocator.allocate(KV_SHAPE, KV_DTYPE, MemoryFormat.KV_2LTD) is None

        first.ref_count_down()  # -> free, area back in the pool
        recycled = allocator.allocate(KV_SHAPE, KV_DTYPE, MemoryFormat.KV_2LTD)
        assert recycled is not None
        assert allocator.bound_bytes == RDS_ALIGN
    finally:
        allocator.close()


def test_a_restore_past_the_cap_reports_a_miss_rather_than_raising(
    runtime: _FakeRuntime, loop: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The caller truncates its hit prefix at ``None``; a raise kills the request.

    ``GdsBackend`` reports the same miss when its slab cannot hold another read
    destination, and ``_process_tokens_internal`` already recomputes from there.
    """
    backend = RDSBackend(_config(), _metadata(), loop)
    try:
        first, second = _key("first"), _key("second")
        _store(backend, first, 1.0)
        _store(backend, second, 2.0)

        # Room for exactly one destination, with two keys asked for.
        monkeypatch.setattr(
            backend.memory_allocator,
            "allocate",
            _exhausting_after(backend.memory_allocator.allocate, 1),
        )
        got = backend.batched_get_blocking([first, second])
        assert got[0] is not None
        assert got[1] is None
        assert torch.allclose(got[0].tensor, torch.ones_like(got[0].tensor))
        got[0].ref_count_down()
    finally:
        backend.close()


def _exhausting_after(real: Any, budget: int) -> Any:
    """Wrap ``real`` so it returns ``None`` after ``budget`` successful calls."""
    served = 0

    def limited(*args: Any, **kwargs: Any) -> Any:
        nonlocal served
        if served >= budget:
            return None
        served += 1
        return real(*args, **kwargs)

    return limited


def test_allocate_waits_for_the_pool_before_reporting_it_full(
    runtime: _FakeRuntime, loop: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full pool is a wait, not an instant truncation -- as GDS treats its slab.

    ``busy_loop=False`` opts out, because ``allocate_and_copy_objects`` runs the
    whole batch on the caller's thread and would otherwise stall it.
    """
    monkeypatch.setattr(allocator_module, "staging_cap_bytes", lambda: RDS_ALIGN)
    config = _config()
    config.extra_config = {
        "max_alloc_attempts": 3,
        "allocation_attempt_delay_secs": 0,
    }
    backend = RDSBackend(config, _metadata(), loop)
    held = None
    try:
        held = backend.allocate(KV_SHAPE, KV_DTYPE, MemoryFormat.KV_2LTD)
        assert held is not None

        attempts = []
        pool_allocate = backend.memory_allocator.allocate

        def counting(*args: Any, **kwargs: Any) -> Any:
            attempts.append(1)
            return pool_allocate(*args, **kwargs)

        monkeypatch.setattr(backend.memory_allocator, "allocate", counting)

        assert backend.allocate(KV_SHAPE, KV_DTYPE, MemoryFormat.KV_2LTD) is None
        assert len(attempts) == 3

        attempts.clear()
        no_wait = backend.allocate(
            KV_SHAPE, KV_DTYPE, MemoryFormat.KV_2LTD, busy_loop=False
        )
        assert no_wait is None
        assert len(attempts) == 1
    finally:
        if held is not None:
            held.ref_count_down()
        backend.close()
