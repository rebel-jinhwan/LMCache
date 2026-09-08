# SPDX-License-Identifier: Apache-2.0
"""RBLN device-event IPC backend (host-synchronizing).

RBLN has no cross-process device event.  An in-process ``torch.rbln.Event``
is a ``(RblnContext, job seq)`` pair whose ``seq`` is only meaningful inside
the context that submitted the job, so it cannot be shipped to another
process the way a ``cudaIpcEventHandle`` can.

LMCache only ever uses events in the order
``record -> export -> (MQ message) -> import -> wait/query/synchronize``.
This backend therefore folds the cross-process ordering into the message
itself: :meth:`RblnEventIPCBackend.export_event` blocks the host until the
recorded work completes and returns an empty handle, and every import-side
operation treats the event as already complete.  Correctness is preserved;
the cost is that the exporting side loses stream/host overlap at the export
point.

Replace ``export_event`` / ``import_event`` with a real cross-context fence
once the UMD exposes one; the rest of the file stays as-is.

See ``docs/design/v1/platform/rbln/README.md``.
"""

# Future
from __future__ import annotations

# Third Party
import torch


class _CompletedEvent:
    """Import-side stand-in for an event that has already completed."""

    def query(self) -> bool:
        """Return ``True``: the producer synchronized before exporting."""
        return True

    def synchronize(self) -> None:
        """No-op: nothing left to wait for."""

    def wait(self, stream: object) -> None:
        """No-op: nothing left to order ``stream`` against."""


class RblnEventIPCBackend:
    """Event IPC backend that orders cross-process work via host sync.

    Satisfies :class:`lmcache.v1.platform.base.event_ipc.EventIPCBackend`.
    """

    device_type: str = "rbln"

    def check_event_support(self, device: object) -> None:
        """Validate that ``torch.rbln`` exposes an ``Event`` type.

        Args:
            device: Device that will create events (unused; RBLN events are
                process-wide).

        Raises:
            RuntimeError: If ``torch.rbln.Event`` is unavailable.
        """
        if not hasattr(torch.rbln, "Event"):
            raise RuntimeError(
                "RBLN event IPC requires torch.rbln.Event; the installed "
                "torch_rbln does not expose it."
            )

    def create_event(self, device: object) -> object:
        """Create an in-process ``torch.rbln.Event``."""
        return torch.rbln.Event()

    def record_event(self, event: object, stream: object) -> None:
        """Record ``event`` on ``stream``."""
        event.record(stream)  # type: ignore[attr-defined]

    def export_event(self, event: object, device: object) -> bytes:
        """Block until ``event`` completes, then return an empty handle.

        The completion is carried by the MQ message that follows the export,
        so the receiving process needs no payload.
        """
        # ponytail: host-side wait stands in for a cross-context fence; swap
        # for a real UMD fence export when one exists.
        event.synchronize()  # type: ignore[attr-defined]
        return b""

    def import_event(self, handle: bytes, device: object) -> object:
        """Return an already-completed event (see :meth:`export_event`)."""
        return _CompletedEvent()

    def wait_event(self, event: object, stream: object) -> None:
        """Order ``stream`` after ``event`` (no-op for imported events)."""
        event.wait(stream)  # type: ignore[attr-defined]

    def query_event(self, event: object) -> bool:
        """Return whether ``event`` has completed."""
        return bool(event.query())  # type: ignore[attr-defined]

    def synchronize_event(self, event: object, device: object) -> None:
        """Block the host until ``event`` completes."""
        event.synchronize()  # type: ignore[attr-defined]
