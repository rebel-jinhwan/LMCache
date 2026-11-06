# SPDX-License-Identifier: Apache-2.0
"""Host-synchronous event IPC for RBLN.

``torch.rbln.Event(interprocess=True)`` exists but ``ipc_handle()`` raises
until torch-rbln ships interprocess events. Ordering is therefore done on the
host: recording an event drains the recording process's device work before
the handle leaves it, so the receiver has nothing left to wait for.

    producer                                  consumer
    record_event -> torch.rbln.synchronize()
    export_event -> b""        --- MQ --->    import_event, wait_event: no-op

This is correct because every handle travels over the request/response queue
after ``record_event`` returned. It costs a device-wide synchronize per store
and retrieve on each side; swap in the native events once they exist.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any

# Third Party
import torch

# First Party
from lmcache.v1.platform.base.event_ipc import EventIPCBackend


class _HostSyncedEvent:
    """An event whose recorded work has already completed."""


class RblnHostSyncEventIPCBackend(EventIPCBackend):
    """Event IPC that orders cross-process work by synchronizing on record."""

    device_type: str = "rbln"

    def check_event_support(self, device: object) -> None:
        """Require ``torch.rbln`` synchronization.

        Args:
            device: Target RBLN device.

        Raises:
            RuntimeError: If ``torch.rbln`` is unavailable.
        """
        if not callable(getattr(getattr(torch, "rbln", None), "synchronize", None)):
            raise RuntimeError(
                "Device backend 'rbln' does not support host-synchronized events: "
                "torch.rbln.synchronize is unavailable."
            )

    def create_event(self, device: object) -> object:
        """Create an event; it carries no device state."""
        return _HostSyncedEvent()

    def export_event(self, event: object, device: object) -> bytes:
        """Return an empty handle; recording already drained the producer."""
        return b""

    def import_event(self, handle: bytes, device: object) -> object:
        """Return an already-completed event."""
        return _HostSyncedEvent()

    def record_event(self, event: object, stream: Any) -> None:
        """Wait until all work queued on ``stream``'s device has finished.

        Args:
            event: Event being recorded (unused; it carries no state).
            stream: ``torch.rbln`` stream whose device is drained.
        """
        torch.rbln.synchronize(stream.device)  # type: ignore[attr-defined]

    def wait_event(self, event: object, stream: object) -> None:
        """No-op: the producer finished before exporting the handle."""

    def query_event(self, event: object) -> bool:
        """Return ``True``; recorded work has completed."""
        return True

    def synchronize_event(self, event: object, device: object) -> None:
        """No-op: recorded work has completed."""
