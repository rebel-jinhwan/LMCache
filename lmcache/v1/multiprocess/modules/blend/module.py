# SPDX-License-Identifier: Apache-2.0
"""Blend modules: the transfer-neutral base plus the LMCache-driven module."""

# Standard
from collections import OrderedDict
from queue import Queue
from typing import TYPE_CHECKING, Any
import threading
import time
import weakref

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_coordinator.blend_client import BlendCoordinatorClient

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.custom_types import CBMatchResult, IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import InstanceLivenessTarget
from lmcache.v1.multiprocess.modules.blend import reorder as blend_reorder
from lmcache.v1.multiprocess.modules.blend.lookup import (
    LookupMixin,
    _CBUnifiedJob,
)
from lmcache.v1.multiprocess.modules.blend.matcher import BlendTokenRangeMatcher
from lmcache.v1.multiprocess.modules.blend.read_set import (
    _cb_chunk_major_object_keys,
)
from lmcache.v1.multiprocess.modules.blend.registration import RegistrationMixin
from lmcache.v1.multiprocess.modules.blend.retrieve import (
    RetrieveMixin,
    _DeviceEvent,
)
from lmcache.v1.multiprocess.modules.blend.rope import _CBRopeState
from lmcache.v1.multiprocess.modules.blend.store import (
    CB_FINGERPRINTS_KIND,
    FingerprintMixin,
    FpJob,
    StoreMixin,
)
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.request_handler import HandlerType, request_handler
from lmcache.v1.multiprocess.session import Session

logger = init_logger(__name__)

_BLEND_PROTOCOL_VERSION = 1


def _handshake_response(client_version: int) -> tuple[int, bool]:
    return (
        _BLEND_PROTOCOL_VERSION,
        client_version == _BLEND_PROTOCOL_VERSION,
    )


class BlendModuleBase(LookupMixin, FingerprintMixin):
    """Transfer-mode-neutral blend state and RPCs: the fingerprint matcher,
    the unified lookup, the reorder planner, and the protocol handshake.

    A concrete module adds the transfer-mode half — how stored chunks reach
    fingerprint registration and how matched chunks reach the paged KV.
    """

    #: ``Session.extras`` key: ``{"read_locks": N, "per_hash": {hash: keys}}``
    #: — the sparse lookup's read-lock reservation, consumed by exactly one
    #: ``extras.pop`` (the retrieve, or :meth:`_release_unretrieved_locks`).
    UNRETRIEVED_KEYS_EXTRA = "cb.unretrieved_read_locked_keys"

    def __init__(
        self,
        ctx: MPCacheServerContext,
        coordinator: "BlendCoordinatorClient | None" = None,
        enable_segmented_prefix: bool = False,
        enable_dedup_content: bool = False,
        enable_reorder: bool = False,
    ):
        self._ctx = ctx
        # Retain the gapped prefix on a mid-prefix L2 retrieve failure
        # instead of truncating at the gap.
        self._segmented_prefix = enable_segmented_prefix
        # Fleet-wide fingerprint directory; None => purely local matching.
        self._coordinator = coordinator

        self._token_range_matcher = BlendTokenRangeMatcher(
            ctx.chunk_size, dedup_content=enable_dedup_content
        )
        # Reorder planner (copy one cached prompt). Off with a coordinator: a
        # plan sees only this server's prompts.
        reorder = enable_reorder and coordinator is None
        if enable_reorder and not reorder:
            logger.warning("--enable-blend-reorder is ignored with a coordinator")
        self._prompt_store = blend_reorder.PromptStore() if reorder else None
        if reorder:  # compile the probe's numba kernels now, not in a plan
            self._token_range_matcher.match_sub_sequence([0] * ctx.chunk_size)
        self._event_bus = ctx.event_bus

        # Backstop for locks the retrieve never consumed.
        ctx.session_manager.destroy_listeners.append(self._release_unretrieved_locks)

        # Non-blocking cb_unified_lookup poll state (submit-once, poll-on-recall)
        # so the handler never holds a worker thread across the L2->L1 loads.
        self._cb_jobs: dict[str, _CBUnifiedJob] = {}
        self._cb_jobs_lock = threading.Lock()

        # Async fingerprint registration: the store path enqueues, the worker
        # drains. The request_id is the enqueuing store's, so the registration
        # event is attributed to it rather than to the drainer.
        self._fingerprint_queue: "Queue[FpJob]" = Queue()
        self._fingerprint_stop = threading.Event()
        self._fingerprint_worker = threading.Thread(
            target=self._drain_fingerprint_queue,
            name="cb-fingerprint-worker",
            daemon=True,
        )
        self._fingerprint_worker.start()

        # In-flight fingerprint hashes; storage_gate keeps these from eviction.
        self._pending_fp_hashes: set[bytes] = set()
        self._pending_fp_lock = threading.Lock()

        # Lazy eviction strikes; evict only at threshold so async re-store
        # can refresh the bucket first.
        self._stale_strike: dict[bytes, int] = {}
        self._STALE_STRIKE_THRESHOLD = 2

        # vLLM may call retrieve twice per request; already-scattered ranges
        # are skipped. Bounded LRU keyed by (request id, WORKER id): at TP>1
        # each worker scatters into its own KV buffers, so the key must
        # include the worker or later ranks skip work they never did.
        self._cb_applied_match_ranges: "OrderedDict[tuple[str, int | None], set[tuple[bytes, int, int, tuple]]]" = OrderedDict()  # noqa: E501

    @property
    def context(self) -> MPCacheServerContext:
        return self._ctx

    @request_handler()
    def cb_protocol_handshake(self, client_version: int) -> tuple[int, bool]:
        return _handshake_response(client_version)

    @request_handler(HandlerType.BLOCKING)
    def cb_reorder_plan(
        self,
        key: IPCCacheServerKey,
        keep_prefix: int,
        budget_ms: int,
    ) -> list[int]:
        """Order ``key.token_ids`` so the prompt starts with an exact copy of
        a cached prompt from the key's namespace, keeping the first
        ``keep_prefix`` tokens in place (``reorder.plan``).

        Returns:
            The reordered token ids, or an empty list to keep the prompt.
        """
        store = self._prompt_store
        if store is None:
            return []
        deadline = time.monotonic() + budget_ms / 1e3
        P = list(key.token_ids)
        try:
            ns = store.resolve(key.model_name, key.world_size, key.cache_salt)
            if ns is None:  # the salt must match; empty name / zero size = any
                return []
            hits = self._token_range_matcher.match_sub_sequence(P)
            cands, base = store.candidates(
                ns,
                [h.hash for h in hits],
                self._ctx.token_hasher.compute_chunk_hashes(P),
            )
            perm = blend_reorder.plan(
                P, cands, self._ctx.chunk_size, base, deadline, keep_prefix
            )
        except Exception:
            logger.exception("CB reorder: planning %s failed", key.request_id)
            return []
        return [P[i] for i in perm] if perm else []

    def _release_unretrieved_locks(self, session: Session) -> None:
        """Release read locks the request's retrieve never consumed.

        Without this, a request that never sends CB_RETRIEVE_PRE_COMPUTED
        would pin its sparse-prefetched chunks in L1 forever. Session
        destruction is the safe release point: no retrieve can arrive after.
        """
        stash = session.extras.pop(self.UNRETRIEVED_KEYS_EXTRA, None)
        if not stash:
            return
        keys = [key for hash_keys in stash["per_hash"].values() for key in hash_keys]
        # No retrieve consumed anything, so the full reservation is still held.
        self._ctx.storage_manager.finish_read_prefetched(
            keys, read_locks=stash["read_locks"], l1_owners=stash.get("l1_owners")
        )
        logger.info(
            "Released %d unretrieved read lock(s) for ended request %s",
            len(keys),
            session.request_id,
        )

    def close(self) -> None:
        listeners = self._ctx.session_manager.destroy_listeners
        if self._release_unretrieved_locks in listeners:
            listeners.remove(self._release_unretrieved_locks)
        self._fingerprint_stop.set()
        if self._coordinator is not None:
            # Joins the client's daemon thread and closes its httpx.Client.
            self._coordinator.close()

    def _take_cb_read_keys(
        self,
        key: IPCCacheServerKey,
        matches: list[CBMatchResult],
        read_gids: tuple[int, ...],
    ) -> tuple[list[ObjectKey], dict[ObjectKey, int] | None]:
        """Consume the lookup's read-lock stash for one retrieve.

        Takes the stash once (later calls re-resolve the keys) and releases
        the locks the lookup took for hashes none of ``matches`` reads.

        Args:
            key: The request key; ``worker_id`` selects this rank's keys.
            matches: The matches this retrieve reads.
            read_gids: The blend read groups, ascending.

        Returns:
            ``(obj_keys, l1_owners)``: this rank's object keys, chunk-major
            (``len(read_gids)`` consecutive keys per match), and the L1 owner
            map to read and release them with (``None`` for a single L1).
        """
        # ``get``, not ``get_or_create``: a retrieve after session end must
        # not recreate ownership state.
        session = self._ctx.session_manager.get(key.request_id)
        stash = (
            session.extras.pop(self.UNRETRIEVED_KEYS_EXTRA, None)
            if session is not None
            else None
        )
        cached = stash["per_hash"] if stash else None
        stash_read_locks = stash["read_locks"] if stash else 1
        l1_owners = (
            stash.get("l1_owners")
            if stash
            else session.extras.get("cb.sparse_l1_owners")
            if session
            else None
        )
        n_read = len(read_gids)
        if cached is not None and all(r.hash in cached for r in matches):
            # The lookup cached all-ranks obj keys (group-major, rank-minor);
            # select THIS rank's key per read group or TP>1 mispairs ranks.
            if key.worker_id is not None and key.world_size > 1:
                ws = key.world_size
                obj_keys = [
                    cached[r.hash][g * ws + key.worker_id]
                    for r in matches
                    for g in range(n_read)
                ]
            else:
                obj_keys = [k for r in matches for k in cached[r.hash]]
        else:
            # One key per (hash, read group), chunk-major like the cached path.
            obj_keys = _cb_chunk_major_object_keys(
                key, [r.hash for r in matches], read_gids
            )

        # The connector may have dropped matches after the lookup read-locked
        # the full found set; release those orphan locks now (disjoint from
        # obj_keys, which the retrieve still consumes).
        if cached is not None:
            retrieved_hashes = {r.hash for r in matches}
            orphan_keys = [
                k for h, ks in cached.items() if h not in retrieved_hashes for k in ks
            ]
            if orphan_keys:
                # Nothing will read these keys: release every lock the lookup
                # took (N per key).
                self._ctx.storage_manager.finish_read_prefetched(
                    orphan_keys, read_locks=stash_read_locks, l1_owners=l1_owners
                )
                logger.debug(
                    "CB released %d prefetched-but-unretrieved keys (req=%s)",
                    len(orphan_keys),
                    key.request_id,
                )
        return obj_keys, l1_owners

    def _record_applied_ranges(
        self,
        key: IPCCacheServerKey,
        applied_now: set[tuple[bytes, int, int, tuple]],
    ) -> None:
        """Record scattered ranges for the repeat-call guard (bounded LRU).

        Args:
            key: The request key; entries are per ``(request_id, worker_id)``.
            applied_now: ``(hash, cur_st, cur_ed, destination blocks)`` of
                each range this call scattered.
        """
        if not applied_now:
            return
        applied_ranges = self._cb_applied_match_ranges
        applied_key = (key.request_id, key.worker_id)
        applied_ranges.setdefault(applied_key, set()).update(applied_now)
        applied_ranges.move_to_end(applied_key)
        # One entry per (request, worker): scale the cap by world size to
        # cover the same number of requests as at TP1.
        cap = 4096 * max(1, int(key.world_size or 1))
        while len(applied_ranges) > cap:
            applied_ranges.popitem(last=False)


class BlendModule(
    BlendModuleBase,
    RegistrationMixin,
    RetrieveMixin,
    StoreMixin,
    InstanceLivenessTarget,
):
    """Paged-aware blend over LMCache-driven transfer. Wraps
    LMCacheDrivenTransfer STORE to register fingerprints; serves CB
    rope/lookup/retrieve RPCs; reads cross-module GPU state via
    :class:`LMCacheDrivenTransferModule.cache_contexts`."""

    def __init__(
        self,
        ctx: MPCacheServerContext,
        lmcache_driven_transfer: LMCacheDrivenTransferModule,
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
        self._transfer_module = lmcache_driven_transfer
        self._cb_rope_state: dict[int, _CBRopeState] = {}
        # Stream-ordered fingerprint enqueue: the store's host func fires
        # after the L1 commit and feeds the base drainer's queue.
        lmcache_driven_transfer.register_host_func(
            CB_FINGERPRINTS_KIND, self._fingerprint_queue.put_nowait, FpJob
        )

        # Request-invariant retrieve-plan specs per GPU context. The cached
        # tuple holds the rope_state it was resolved against so a
        # re-registration invalidates by identity.
        self._cb_plan_invariants: "weakref.WeakKeyDictionary[Any, tuple]" = (
            weakref.WeakKeyDictionary()
        )
        # Pinned + device slot-mapping staging per GPU context, grown on demand.
        self._cb_slot_staging: "weakref.WeakKeyDictionary[Any, tuple]" = (
            weakref.WeakKeyDictionary()
        )
        # Completion event per context; the next retrieve host-syncs it before
        # reusing the shared slot/temp buffers.
        self._cb_plan_done_events: "weakref.WeakKeyDictionary[Any, _DeviceEvent]" = (
            weakref.WeakKeyDictionary()
        )
        # Retrieve-owned stream per context: keeps retrieves out of the shared
        # stream's store gather/commit traffic.
        self._cb_retrieve_streams: "weakref.WeakKeyDictionary[Any, Any]" = (
            weakref.WeakKeyDictionary()
        )
        self._cb_retrieve_cupy_streams: "weakref.WeakKeyDictionary[Any, Any]" = (
            weakref.WeakKeyDictionary()
        )

    def report_status(self) -> dict:
        cache_contexts = self._transfer_module.context_entries_snapshot()

        def _meta(iid: int) -> "tuple[str, int] | None":
            entry = cache_contexts.get(iid)
            return (entry.model_name, entry.world_size) if entry is not None else None

        return {
            "registered_cb_rope_instances": list(self._cb_rope_state.keys()),
            "cb_rope_meta": {str(iid): _meta(iid) for iid in self._cb_rope_state},
            "active_cb_lookups": len(self._cb_jobs),
        }

    def close(self) -> None:
        super().close()
        self._cb_rope_state.clear()

    def drop_instance_state(self, instance_id: int) -> None:
        """Drop CB rope state for a reaped instance (InstanceLivenessTarget
        hook); the GPU cache context is owned by LMCacheDrivenTransferModule."""
        if self._cb_rope_state.pop(instance_id, None) is not None:
            logger.info("Dropped CB rope state for reaped instance %d", instance_id)
