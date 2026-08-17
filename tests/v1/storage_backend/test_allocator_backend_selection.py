# SPDX-License-Identifier: Apache-2.0
"""Allocator selection when a tier allocates its own objects.

A storage tier that DMAs device memory (GDS, RDS) allocates the objects it
writes, so a deployment can leave the host pool out entirely. The selection
used to hardcode ``LocalCPUBackend`` and raised ``KeyError`` for that shape.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.storage_manager import StorageManager


def _backend(owns_allocator: bool) -> MagicMock:
    spec = AllocatorBackendInterface if owns_allocator else object
    return MagicMock(spec=spec)


def _manager(names: list[str], enable_pd: bool = False) -> StorageManager:
    manager = StorageManager.__new__(StorageManager)
    manager.enable_pd = enable_pd
    manager.storage_backends = {
        name: _backend(name != "PlainBackend") for name in names
    }
    return manager


def _selected(manager: StorageManager) -> str:
    backend = manager._get_allocator_backend(MagicMock())  # noqa: SLF001
    return next(n for n, b in manager.storage_backends.items() if b is backend)


def test_host_pool_still_wins_when_present() -> None:
    assert _selected(_manager(["LocalCPUBackend", "GdsBackend"])) == "LocalCPUBackend"


def test_a_tier_that_owns_an_allocator_serves_without_a_host_pool() -> None:
    assert _selected(_manager(["GdsBackend"])) == "GdsBackend"
    assert _selected(_manager(["RDSBackend"])) == "RDSBackend"


def test_first_registered_allocator_owner_wins() -> None:
    assert _selected(_manager(["PlainBackend", "RDSBackend"])) == "RDSBackend"


def test_pd_still_wins() -> None:
    assert (
        _selected(_manager(["PDBackend", "LocalCPUBackend"], enable_pd=True))
        == "PDBackend"
    )


def test_no_allocator_anywhere_is_a_named_error() -> None:
    with pytest.raises(RuntimeError, match="owns a memory allocator"):
        _selected(_manager(["PlainBackend"]))
