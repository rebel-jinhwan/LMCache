# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the RBLN pin-memory backend (no RBLN hardware needed).

The backend's contract, per
``docs/design/v1/platform/rbln/README.md`` and its own docstrings:

- With torch-rbln's ``register_host_memory`` available, ``pin_memory`` registers
  the range and ``unpin_memory`` unregisters exactly what was registered; a
  refused registration is reported as False, never as a pretend pin.
- Without it the surface reports "unsupported" rather than raising.

torch-rbln is replaced by a fake registrar so the suite runs on any host.
"""

# Standard
from collections.abc import Iterator
import ctypes
import mmap
import os
import types

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.rbln import RblnDeviceSpec
from lmcache.v1.platform.rbln.pin_memory import RblnPinMemoryBackend

PAGE_SIZE = os.sysconf("SC_PAGESIZE")
REGION_PAGES = 3
SENTINEL = b"lmcache-rbln-pin"


class FakeRegistrar:
    """Stands in for ``torch.rbln.register_host_memory`` / ``unregister_host_memory``.

    Records every call and can be told to refuse, so the tests can see which
    strategy the backend took.
    """

    def __init__(self, refuse: bool = False) -> None:
        self.refuse = refuse
        self.registered: list[tuple[int, int]] = []
        self.unregistered: list[int] = []

    def register(self, ptr: int, size: int) -> None:
        if self.refuse:
            raise RuntimeError("register_host_memory: overlaps a registered range")
        self.registered.append((ptr, size))

    def unregister(self, ptr: int) -> None:
        if ptr not in {p for p, _ in self.registered}:
            raise RuntimeError("unregister_host_memory: not a registration")
        self.unregistered.append(ptr)


def _install(monkeypatch: pytest.MonkeyPatch, fake: FakeRegistrar) -> None:
    """Point ``torch.rbln``'s register/unregister_host_memory at ``fake``."""
    rbln = getattr(torch, "rbln", None)
    if rbln is None:
        rbln = types.SimpleNamespace()
        monkeypatch.setattr(torch, "rbln", rbln, raising=False)
    monkeypatch.setattr(rbln, "register_host_memory", fake.register, raising=False)
    monkeypatch.setattr(rbln, "unregister_host_memory", fake.unregister, raising=False)


@pytest.fixture
def registrar(monkeypatch: pytest.MonkeyPatch) -> FakeRegistrar:
    """Install a fake torch-rbln registrar that accepts everything."""
    fake = FakeRegistrar()
    _install(monkeypatch, fake)
    return fake


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


# --- registration path ------------------------------------------------------


def test_pin_registers_with_torch_rbln(
    registrar: FakeRegistrar, region: tuple[int, int]
) -> None:
    """With torch-rbln present the range is registered, not populated."""
    ptr, size = region
    backend = RblnPinMemoryBackend()

    assert backend.is_pin_supported is True
    assert backend.pin_memory(ptr, size) is True
    assert registrar.registered == [(ptr, size)]
    assert _read(ptr, len(SENTINEL)) == SENTINEL


def test_unpin_unregisters_what_pin_registered(
    registrar: FakeRegistrar, region: tuple[int, int]
) -> None:
    """Unpinning hands the pin back exactly once."""
    ptr, size = region
    backend = RblnPinMemoryBackend()
    backend.pin_memory(ptr, size)

    assert backend.unpin_memory(ptr) is True
    assert registrar.unregistered == [ptr]
    # A second unpin has nothing registered to release and must not reach the
    # runtime again.
    assert backend.unpin_memory(ptr) is False
    assert registrar.unregistered == [ptr]


def test_refused_registration_reports_false(
    monkeypatch: pytest.MonkeyPatch, region: tuple[int, int]
) -> None:
    """A registrar that refuses makes pin_memory False, with the data intact."""
    fake = FakeRegistrar(refuse=True)
    _install(monkeypatch, fake)
    ptr, size = region
    backend = RblnPinMemoryBackend()

    assert backend.pin_memory(ptr, size) is False
    assert fake.registered == []
    assert _read(ptr, len(SENTINEL)) == SENTINEL
    # Nothing was registered, so unpin must not reach the runtime.
    assert backend.unpin_memory(ptr) is False
    assert fake.unregistered == []


def test_pin_rejects_an_empty_region_before_registering(
    registrar: FakeRegistrar,
) -> None:
    """Nothing to pin is False, and never a runtime call."""
    backend = RblnPinMemoryBackend()

    assert backend.pin_memory(0, PAGE_SIZE) is False
    assert backend.pin_memory(1 << 20, 0) is False
    assert registrar.registered == []
