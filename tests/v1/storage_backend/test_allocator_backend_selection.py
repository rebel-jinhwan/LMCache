# SPDX-License-Identifier: Apache-2.0
"""Tests for which backend ``StorageManager`` picks as its allocator.

``RDSBackend`` has to win over ``LocalCPUBackend``, which is what distinguishes
it from ``GdsBackend`` — an ``AllocatorBackendInterface`` that is never selected
here, because cuFile writes whatever buffer it is handed. ``rebel.rds`` cannot,
and ``max_local_cpu_size`` defaults to 5 GB, so the default RDS configuration
has a host pool sitting in front of it.

Exercised on an uninitialised instance: the real constructor builds an event
loop, a metrics registry and a worker, none of which take part in this decision.
"""

# Standard
from collections import OrderedDict
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.abstract_backend import (
    AllocatorBackendInterface,
    StorageBackendInterface,
)
from lmcache.v1.storage_backend.storage_manager import StorageManager


def _backend(name: str) -> StorageBackendInterface:
    backend = MagicMock(spec=AllocatorBackendInterface)
    backend.configure_mock(**{"__str__.return_value": name})
    return backend


def _manager(names: list[str], enable_pd: bool = False) -> StorageManager:
    manager = object.__new__(StorageManager)
    manager.storage_backends = OrderedDict((n, _backend(n)) for n in names)
    manager.enable_pd = enable_pd
    return manager


def _selected(manager: StorageManager) -> str:
    return str(manager._get_allocator_backend(MagicMock()))


def test_rds_wins_over_the_host_pool() -> None:
    """The default config has both, and RDS cannot write a host MemoryObj."""
    assert _selected(_manager(["LocalCPUBackend", "RDSBackend"])) == "RDSBackend"


def test_rds_is_selected_without_a_host_pool() -> None:
    """`max_local_cpu_size=0` used to KeyError on a backend never configured."""
    assert _selected(_manager(["RDSBackend"])) == "RDSBackend"


def test_pd_still_wins_over_rds() -> None:
    """P/D is a different mode, and owns its own shared buffer."""
    manager = _manager(["PDBackend", "RDSBackend"], enable_pd=True)
    assert _selected(manager) == "PDBackend"


def test_local_cpu_still_wins_when_rds_is_absent() -> None:
    """Unchanged for every configuration that does not enable RDS."""
    assert _selected(_manager(["LocalCPUBackend", "GdsBackend"])) == "LocalCPUBackend"
    assert _selected(_manager(["LocalCPUBackend", "MaruBackend"])) == "LocalCPUBackend"


def test_maru_still_wins_without_a_host_pool() -> None:
    assert _selected(_manager(["MaruBackend"])) == "MaruBackend"


def test_gds_alone_still_raises() -> None:
    """Left as-is deliberately.

    GDS without a host pool has always been a KeyError, and whether cuFile can
    serve as the primary allocator is a question for someone working on GDS —
    not something this change should decide by widening the fallback.
    """
    with pytest.raises(KeyError, match="LocalCPUBackend"):
        _selected(_manager(["GdsBackend"]))
