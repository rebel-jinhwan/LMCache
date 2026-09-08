# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the RBLN pin-memory backend (no RBLN hardware needed).

The backend's contract, per
``docs/design/v1/platform/rbln/README.md`` and its own docstrings:

- ``pin_memory`` faults a host region in without changing its contents, and
  accepts an address that is not page-aligned.
- ``unpin_memory`` releases nothing, so a pinned region keeps its data.
- Where ``MADV_POPULATE_WRITE`` is unavailable the whole surface degrades to
  "unsupported" rather than raising.
"""

# Standard
from collections.abc import Iterator
import ctypes
import mmap
import os

# Third Party
import pytest

# First Party
from lmcache.v1.platform.rbln import RblnDeviceSpec
from lmcache.v1.platform.rbln.pin_memory import RblnPinMemoryBackend

PAGE_SIZE = os.sysconf("SC_PAGESIZE")
REGION_PAGES = 3
SENTINEL = b"lmcache-rbln-pin"


@pytest.fixture
def backend() -> RblnPinMemoryBackend:
    """A backend instance, skipping the test when the kernel is too old."""
    instance = RblnPinMemoryBackend()
    if not instance.is_pin_supported:
        pytest.skip("MADV_POPULATE_WRITE unavailable on this kernel")
    return instance


@pytest.fixture
def region() -> Iterator[tuple[int, int]]:
    """A private anonymous mapping carrying a sentinel, as ``(ptr, size)``."""
    size = REGION_PAGES * PAGE_SIZE
    mapping = mmap.mmap(-1, size)
    mapping.write(SENTINEL)
    view = (ctypes.c_uint8 * 1).from_buffer(mapping)
    ptr = ctypes.addressof(view)
    del view  # a live export would block mapping.close()
    try:
        yield ptr, size
    finally:
        mapping.close()


def _read(ptr: int, length: int) -> bytes:
    """Read ``length`` bytes back out of a raw address."""
    return bytes((ctypes.c_uint8 * length).from_address(ptr))


def test_spec_selects_the_rbln_backend() -> None:
    """The device spec routes host pinning to this backend."""
    spec = RblnDeviceSpec()

    assert spec.pin_memory_backend is RblnPinMemoryBackend
    assert spec.is_pin_supported == RblnPinMemoryBackend().is_pin_supported


def test_pin_memory_populates_without_touching_contents(
    backend: RblnPinMemoryBackend, region: tuple[int, int]
) -> None:
    """A pinned region reports success and keeps every byte it held."""
    ptr, size = region

    assert backend.pin_memory(ptr, size) is True
    assert _read(ptr, len(SENTINEL)) == SENTINEL


def test_pin_memory_accepts_an_unaligned_address(
    backend: RblnPinMemoryBackend, region: tuple[int, int]
) -> None:
    """An interior, unaligned span still succeeds.

    ``madvise`` rejects an unaligned start with ``EINVAL``, so this only
    passes if the backend widens the span to whole pages.
    """
    ptr, size = region
    unaligned = ptr + 17

    assert backend.pin_memory(unaligned, size - PAGE_SIZE) is True


def test_pin_memory_rejects_an_empty_region(backend: RblnPinMemoryBackend) -> None:
    """Nothing to populate is not an error, just False."""
    assert backend.pin_memory(0, PAGE_SIZE) is False
    assert backend.pin_memory(1 << 20, 0) is False
    assert backend.pin_memory(1 << 20, -1) is False


def test_unpin_memory_keeps_the_region_readable(
    backend: RblnPinMemoryBackend, region: tuple[int, int]
) -> None:
    """Unpinning drops no pages, so the region's contents survive it."""
    ptr, size = region
    backend.pin_memory(ptr, size)

    assert backend.unpin_memory(ptr) is True
    assert _read(ptr, len(SENTINEL)) == SENTINEL


def test_degrades_to_unsupported_without_madvise(
    monkeypatch: pytest.MonkeyPatch, region: tuple[int, int]
) -> None:
    """Off Linux the backend reports unsupported instead of raising."""
    monkeypatch.setattr("lmcache.v1.platform.rbln.pin_memory.sys.platform", "darwin")
    ptr, size = region

    unsupported = RblnPinMemoryBackend()

    assert unsupported.is_pin_supported is False
    assert unsupported.pin_memory(ptr, size) is False
    assert unsupported.unpin_memory(ptr) is False
