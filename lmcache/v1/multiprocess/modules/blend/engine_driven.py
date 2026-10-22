# SPDX-License-Identifier: Apache-2.0
"""EngineDrivenBlendModule: CacheBlend over the engine-driven transfer path.

The server cannot reach the worker's device on this path, so the retrieve is
split in two RPCs around the worker's own scatter:

1. ``CB_PREPARE_RETRIEVE`` selects the matches to scatter (the same repeat,
   allocation, and stash rules as ``CB_RETRIEVE_PRE_COMPUTED``) and exports
   their read-locked chunks through the instance's transport.
2. The worker re-RoPEs and scatters them
   (``transfer_context.blend_scatter``).
3. ``CB_COMMIT_RETRIEVE`` releases the read locks and, on success, records
   the applied ranges.

Fingerprints are registered from the engine-driven ``COMMIT_STORE``, which
this module shadows the same way ``BlendModule`` shadows ``STORE``.
"""

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING
import threading

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_coordinator.blend_client import BlendCoordinatorClient

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.multiprocess.custom_types import (
    CBMatchResult,
    CBPrepareRetrieveResponse,
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import InstanceLivenessTarget
from lmcache.v1.multiprocess.modules.blend.module import BlendModuleBase
from lmcache.v1.multiprocess.modules.blend.retrieve import RetrieveReason
from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
    EngineDrivenTransferModule,
)
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler

logger = init_logger(__name__)

#: Engine-driven registrations hold one object group, so blend reads group 0.
_READ_GIDS = (0,)


@dataclass
class _PendingCBRead:
    """Read locks a prepared retrieve holds until the worker commits.

    Attributes:
        obj_keys: Read-locked object keys, one per match.
        l1_owners: L1 owner map to release them with.
        applied: ``(hash, cur_st, cur_ed, destination blocks)`` per match,
            recorded as applied when the worker reports success.
    """

    obj_keys: list[ObjectKey]
    l1_owners: dict[ObjectKey, int] | None
    applied: set[tuple[bytes, int, int, tuple]]


def _dest_blocks(r: CBMatchResult, block_ids: list[int], block_size: int) -> tuple:
    """Destination blocks match ``r`` writes into."""
    return tuple(block_ids[r.cur_st // block_size : (r.cur_ed - 1) // block_size + 1])


class EngineDrivenBlendModule(BlendModuleBase, InstanceLivenessTarget):
    """Blend over engine-driven transfer: shadows ``COMMIT_STORE`` to
    register fingerprints; serves the CB lookup plus the prepare/commit
    retrieve pair the worker brackets its own scatter with."""

    def __init__(
        self,
        ctx: MPCacheServerContext,
        engine_driven_transfer: EngineDrivenTransferModule,
        coordinator: "BlendCoordinatorClient | None" = None,
        enable_segmented_prefix: bool = False,
        enable_dedup_content: bool = False,
        enable_reorder: bool = False,
    ):
        super().__init__(
            ctx,
            coordinator=coordinator,
            enable_segmented_prefix=enable_segmented_prefix,
            enable_dedup_content=enable_dedup_content,
            enable_reorder=enable_reorder,
        )
        self._transfer_module = engine_driven_transfer
        # Keyed by (instance_id, request_id, worker_id): at TP>1 each rank
        # prepares and commits its own retrieve.
        self._pending_reads: dict[tuple[int, str, int | None], _PendingCBRead] = {}
        self._pending_reads_lock = threading.Lock()

    def report_status(self) -> dict:
        """Return the in-flight lookup and prepared-retrieve counts."""
        with self._pending_reads_lock:
            pending = len(self._pending_reads)
        return {
            "active_cb_lookups": len(self._cb_jobs),
            "pending_cb_reads": pending,
        }

    @request_handler(
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        cpu_data: bytes,
    ) -> bool:
        """Engine-driven store commit, then register the committed chunks as
        match fingerprints.

        The commit has completed on the host when the transfer module
        returns, so the registration jobs go straight to the drainer queue.

        Returns:
            The underlying ``EngineDrivenTransfer.commit_store`` result.

        Raises:
            ValueError: If the instance is not registered.
        """
        success, stored_mask = self._transfer_module.commit_store_with_chunk_mask(
            key, instance_id, cpu_data
        )
        # The matcher is engine-shared; only worker 0 registers.
        if key.worker_id in (0, None):
            self._enqueue_stored_fingerprints(key, stored_mask, None)
        return success

    @request_handler(
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
    def cb_prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        match_results: list[CBMatchResult],
        block_ids: list[list[int]],
        instance_id: int,
    ) -> CBPrepareRetrieveResponse:
        """Select the matches to scatter and export their chunks.

        All-or-nothing like ``CB_RETRIEVE_PRE_COMPUTED``: defers while the
        allocation covers no matched token, fails (``scatter_ran=False``)
        when it covers only part, and skips ranges already scattered into
        the same blocks. A response with ``matches`` holds read locks until
        :meth:`cb_commit_retrieve`.

        Args:
            key: The request key.
            match_results: The lookup's matches, any order.
            block_ids: The request's block table, one engine group.
            instance_id: The worker's engine-driven registration.

        Returns:
            The matches to scatter with their exported chunks, or no matches
            and the wire bool.

        Raises:
            ValueError: If the instance is not registered or ``block_ids``
                holds more than one engine group.
        """
        block_size = self._transfer_module.get_block_size(instance_id)
        if len(block_ids) != 1:
            raise ValueError(
                "engine-driven blend supports one KV group, got "
                f"{len(block_ids)} block tables"
            )
        blocks = block_ids[0]
        pending_key = (instance_id, key.request_id, key.worker_id)
        # A prepare without its commit (the worker failed in between) must
        # not strand its locks.
        with self._pending_reads_lock:
            stale = self._pending_reads.pop(pending_key, None)
        if stale is not None:
            self._release_pending_read(stale)

        matches = sorted(match_results, key=lambda r: r.cur_st)
        prior = self._cb_applied_match_ranges.get((key.request_id, key.worker_id))
        if prior:
            matches = [
                r
                for r in matches
                if (r.hash, r.cur_st, r.cur_ed, _dest_blocks(r, blocks, block_size))
                not in prior
            ]
            if not matches:
                return self._no_scatter(key, RetrieveReason.ALREADY_APPLIED, 0)

        slot_bound = len(blocks) * block_size
        if any(r.cur_ed > slot_bound for r in matches):
            if all(r.cur_st >= slot_bound for r in matches):
                # No matched token is forwarded this step; the follow-up
                # full-alloc call scatters everything, locks stay held.
                return self._no_scatter(
                    key, RetrieveReason.MATCHES_BEYOND_ALLOC, len(matches)
                )
            # The client blends its own match list on any True return, so a
            # partial scatter would read unpopulated KV: full recompute.
            return self._no_scatter(
                key, RetrieveReason.MATCHES_STRADDLE_ALLOC, len(matches)
            )

        obj_keys, l1_owners = self._take_cb_read_keys(key, matches, _READ_GIDS)
        if not obj_keys:
            return self._no_scatter(key, RetrieveReason.NO_OBJECT_KEYS, len(matches))

        self._event_bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_START,
                session_id=key.request_id,
                metadata={
                    "num_chunks": len(matches),
                    "model_name": key.model_name,
                    "worker_id": key.worker_id,
                },
            )
        )
        storage_manager = self._ctx.storage_manager
        read_keys, memory_objs = storage_manager.unsafe_read(obj_keys, l1_owners)
        exported = (
            self._transfer_module.export_read_objects(instance_id, memory_objs)
            if len(memory_objs) == len(obj_keys)
            else None
        )
        if exported is None or not exported.success:
            # The client degrades the request and no retrieve will read these.
            storage_manager.finish_read_prefetched(read_keys, l1_owners=l1_owners)
            logger.warning(
                "CB prepare retrieve: %d/%d prefetched object(s) readable for "
                "request %s; degrading to full recompute",
                len(memory_objs),
                len(obj_keys),
                key.request_id,
            )
            self._publish_retrieve_end(key, success=False)
            return CBPrepareRetrieveResponse(scatter_ran=False)

        with self._pending_reads_lock:
            self._pending_reads[pending_key] = _PendingCBRead(
                obj_keys=obj_keys,
                l1_owners=l1_owners,
                applied={
                    (r.hash, r.cur_st, r.cur_ed, _dest_blocks(r, blocks, block_size))
                    for r in matches
                },
            )
        return CBPrepareRetrieveResponse(
            scatter_ran=True,
            matches=matches,
            data=exported.data,
            context=exported.context,
        )

    @request_handler(
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    )
    def cb_commit_retrieve(
        self, key: IPCCacheServerKey, instance_id: int, scattered: bool
    ) -> bool:
        """Release a prepared retrieve's read locks.

        Args:
            key: The request key the retrieve was prepared with.
            instance_id: The worker's engine-driven registration.
            scattered: Whether the worker wrote every prepared match. Only
                then are the ranges recorded as applied.

        Returns:
            ``scattered``, or ``False`` when no prepared retrieve is pending
            for the key (already released, e.g. by a reap).
        """
        with self._pending_reads_lock:
            pending = self._pending_reads.pop(
                (instance_id, key.request_id, key.worker_id), None
            )
        if pending is None:
            logger.warning(
                "CB commit retrieve: no prepared retrieve for request %s "
                "(instance %d, worker %s)",
                key.request_id,
                instance_id,
                key.worker_id,
            )
            return False
        self._release_pending_read(pending)
        if scattered:
            self._record_applied_ranges(key, pending.applied)
        self._publish_retrieve_end(key, success=scattered)
        return scattered

    def close(self) -> None:
        """Release every prepared retrieve's read locks, then the base state."""
        with self._pending_reads_lock:
            pending = list(self._pending_reads.values())
            self._pending_reads.clear()
        for entry in pending:
            self._release_pending_read(entry)
        super().close()

    def drop_instance_state(self, instance_id: int) -> None:
        """Release the prepared retrieves of a reaped instance
        (InstanceLivenessTarget hook)."""
        with self._pending_reads_lock:
            stale_keys = [k for k in self._pending_reads if k[0] == instance_id]
            stale = [self._pending_reads.pop(k) for k in stale_keys]
        for entry in stale:
            self._release_pending_read(entry)
        if stale:
            logger.info(
                "Released %d prepared CB retrieve(s) for reaped instance %d",
                len(stale),
                instance_id,
            )

    def _release_pending_read(self, pending: _PendingCBRead) -> None:
        """Release the one read lock per key a prepared retrieve consumed."""
        self._ctx.storage_manager.finish_read_prefetched(
            pending.obj_keys, l1_owners=pending.l1_owners
        )

    def _no_scatter(
        self, key: IPCCacheServerKey, reason: RetrieveReason, num_matches: int
    ) -> CBPrepareRetrieveResponse:
        """End the request without a scatter (``CB_RETRIEVE_NOOP`` when reuse
        was lost)."""
        logger.debug(
            "CB prepare retrieve: no scatter (%s) for request %s, %d match(es)",
            reason.value,
            key.request_id,
            num_matches,
        )
        if reason.publish:
            self._event_bus.publish(
                Event(
                    event_type=EventType.CB_RETRIEVE_NOOP,
                    session_id=key.request_id,
                    metadata={"reason": reason.value, "dropped_matches": num_matches},
                )
            )
        self._event_bus.publish(
            Event(event_type=EventType.CB_REQUEST_END, session_id=key.request_id)
        )
        return CBPrepareRetrieveResponse(scatter_ran=reason.scatter_ran)

    def _publish_retrieve_end(self, key: IPCCacheServerKey, success: bool) -> None:
        """Close the request's retrieve span and the request."""
        self._event_bus.publish(
            Event(
                event_type=EventType.CB_RETRIEVE_END,
                session_id=key.request_id,
                metadata={"success": success, "worker_id": key.worker_id},
            )
        )
        self._event_bus.publish(
            Event(event_type=EventType.CB_REQUEST_END, session_id=key.request_id)
        )
