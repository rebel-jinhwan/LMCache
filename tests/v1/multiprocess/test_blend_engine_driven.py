# SPDX-License-Identifier: Apache-2.0
"""CacheBlend over the engine-driven transfer path.

Three layers, each against its public surface:

* ``scatter_cb_matches_to_paged_kv`` -- the worker writes matches at token
  granularity and K-only re-RoPEs the shifted ones.
* ``EngineDrivenBlendModule`` -- the server's prepare/commit pair keeps the
  lookup's read locks balanced and applies the all-or-nothing rules.
* ``EngineDrivenTransferContext.submit_cb_retrieve`` -- the worker brackets
  its scatter with the two RPCs.
"""

# Standard
from typing import Any
from unittest.mock import MagicMock
import pickle

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_device_type
from lmcache.v1.multiprocess.custom_types import (
    CBMatchResult,
    CBPrepareRetrieveResponse,
    IPCCacheServerKey,
    PrepareRetrieveResponse,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.modules.blend import EngineDrivenBlendModule
from lmcache.v1.multiprocess.transfer_context.base import gather_paged_kv_to_cpu
from lmcache.v1.multiprocess.transfer_context.blend_scatter import (
    CBRopeSpec,
    normalize_cos_sin_cache,
    scatter_cb_matches_to_paged_kv,
)
from lmcache.v1.platform.torch_ops.pos_kernels import rotary_embedding_k_fused

_NUM_LAYERS = 2
_NUM_BLOCKS = 8
_BLOCK_SIZE = 4
_NUM_HEADS = 2
_HEAD_SIZE = 8
_CHUNK = 8  # tokens per LMCache chunk (two blocks)
_MAX_POS = 64


@pytest.fixture(autouse=True)
def _detect_vllm_layouts_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep vLLM format discovery independent of the host running the test."""
    monkeypatch.setattr(
        "lmcache.v1.gpu_connector.kv_format.detectors.vllm.torch_device_type",
        torch_device_type if torch_device_type != "cpu" else "cuda",
    )


def _nhd_kv_caches() -> dict[str, torch.Tensor]:
    """Per-layer vLLM NHD KV: ``[2, num_blocks, block_size, heads, head]``."""
    return {
        f"layer_{i}": torch.randn(2, _NUM_BLOCKS, _BLOCK_SIZE, _NUM_HEADS, _HEAD_SIZE)
        for i in range(_NUM_LAYERS)
    }


def _mla_kv_caches(hidden: int) -> dict[str, torch.Tensor]:
    """Per-layer MLA KV: ``[num_blocks, block_size, hidden]``."""
    return {
        f"layer_{i}": torch.randn(_NUM_BLOCKS, _BLOCK_SIZE, hidden)
        for i in range(_NUM_LAYERS)
    }


def _token_major(kv_caches: dict[str, torch.Tensor], block_ids: list[int]):
    """Token-major canonical view of ``block_ids`` (one gathered chunk)."""
    (chunk,) = gather_paged_kv_to_cpu(kv_caches, block_ids, len(block_ids))
    return chunk


def _cos_sin_cache(rot_dim: int) -> torch.Tensor:
    """Standard RoPE cache: ``[max_pos, rot_dim]``, cos half then sin half."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rot_dim, 2).float() / rot_dim))
    freqs = torch.outer(torch.arange(_MAX_POS).float(), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def _rope_rows(raw: torch.Tensor, start: int, cache: torch.Tensor) -> torch.Tensor:
    """Apply RoPE at positions ``start..`` to ``raw`` (``[L, n, heads, d]``).

    Implemented as an undo/redo from position 0, whose rotation is the
    identity, so it reuses the reference torch op.
    """
    rows = raw.transpose(0, 1).reshape(raw.shape[1], -1, raw.shape[-1]).clone()
    n = rows.shape[0]
    rotary_embedding_k_fused(
        torch.zeros(n, dtype=torch.long),
        torch.arange(start, start + n),
        rows,
        raw.shape[-1],
        cache,
        True,
    )
    return rows.reshape(n, raw.shape[0], *raw.shape[2:]).transpose(0, 1)


def _match(old_st: int, cur_st: int, n: int, tag: bytes = b"h") -> CBMatchResult:
    return CBMatchResult(
        old_st=old_st, old_ed=old_st + n, cur_st=cur_st, cur_ed=cur_st + n, hash=tag
    )


# ---------------------------------------------------------------------------
# Worker scatter
# ---------------------------------------------------------------------------


def test_scatter_writes_only_the_matched_tokens() -> None:
    """A match that starts and ends mid-block overwrites exactly its tokens:
    the rest of each partly covered block, and every untouched block, keep
    their content."""
    kv = _nhd_kv_caches()
    block_ids = [5, 2, 7, 0]
    before = _token_major(kv, block_ids)
    chunk = torch.randn(2, _NUM_LAYERS, _CHUNK, _NUM_HEADS * _HEAD_SIZE)
    # Six tokens at query positions [3, 9): blocks 0..2, the first and the
    # last only partly.
    match = _match(old_st=16, cur_st=3, n=6)

    scatter_cb_matches_to_paged_kv(kv, block_ids, [chunk], [match], rope=None)

    after = _token_major(kv, block_ids)
    expected = before.clone()
    expected[:, :, 3:9] = chunk[:, :, :6]
    torch.testing.assert_close(after, expected)


def test_scatter_rotates_shifted_keys_to_their_new_position() -> None:
    """A shifted match's K ends up RoPE'd at ``cur_st`` (as if computed
    there); V is copied unrotated."""
    kv = _nhd_kv_caches()
    block_ids = [1, 3, 6]
    cache = _cos_sin_cache(_HEAD_SIZE)
    rope = CBRopeSpec(cache, head_size=_HEAD_SIZE, is_neox_style=True)
    raw_k = torch.randn(_NUM_LAYERS, _CHUNK, _NUM_HEADS, _HEAD_SIZE)
    value = torch.randn(_NUM_LAYERS, _CHUNK, _NUM_HEADS * _HEAD_SIZE)
    old_st, cur_st = 24, 2
    chunk = torch.stack(
        [_rope_rows(raw_k, old_st, cache).reshape(_NUM_LAYERS, _CHUNK, -1), value]
    )

    scatter_cb_matches_to_paged_kv(
        kv, block_ids, [chunk], [_match(old_st, cur_st, _CHUNK)], rope=rope
    )

    after = _token_major(kv, block_ids)
    expected_k = _rope_rows(raw_k, cur_st, cache).reshape(_NUM_LAYERS, _CHUNK, -1)
    torch.testing.assert_close(after[0, :, cur_st : cur_st + _CHUNK], expected_k)
    torch.testing.assert_close(after[1, :, cur_st : cur_st + _CHUNK], value)


def test_scatter_without_rope_copies_shifted_keys_as_is() -> None:
    """A NoPE model (``rope=None``) relocates K without rotating it."""
    kv = _nhd_kv_caches()
    block_ids = [4, 2]
    chunk = torch.randn(2, _NUM_LAYERS, _CHUNK, _NUM_HEADS * _HEAD_SIZE)

    scatter_cb_matches_to_paged_kv(
        kv, block_ids, [chunk], [_match(old_st=32, cur_st=0, n=_CHUNK)], rope=None
    )

    torch.testing.assert_close(_token_major(kv, block_ids), chunk)


def test_scatter_rotates_only_the_mla_rope_window() -> None:
    """MLA: the trailing rope window is re-RoPE'd; the latent content dims
    are copied untouched."""
    latent, rot_dim = 12, 4
    kv = _mla_kv_caches(latent + rot_dim)
    block_ids = [0, 1, 2]
    cache = _cos_sin_cache(rot_dim)
    rope = CBRopeSpec(cache, head_size=rot_dim, is_neox_style=True, rot_offset=latent)
    content = torch.randn(_NUM_LAYERS, _CHUNK, latent)
    raw_rope = torch.randn(_NUM_LAYERS, _CHUNK, 1, rot_dim)
    old_st, cur_st = 40, 1
    chunk = torch.cat(
        [content, _rope_rows(raw_rope, old_st, cache).reshape(_NUM_LAYERS, _CHUNK, -1)],
        dim=-1,
    )

    scatter_cb_matches_to_paged_kv(
        kv, block_ids, [chunk], [_match(old_st, cur_st, _CHUNK)], rope=rope
    )

    rows = _token_major(kv, block_ids)[:, cur_st : cur_st + _CHUNK]
    torch.testing.assert_close(rows[..., :latent], content)
    torch.testing.assert_close(
        rows[..., latent:],
        _rope_rows(raw_rope, cur_st, cache).reshape(_NUM_LAYERS, _CHUNK, -1),
    )


def test_scatter_writes_several_matches_in_one_pass() -> None:
    """Adjacent and separate matches land at their own positions."""
    kv = _nhd_kv_caches()
    block_ids = [7, 6, 5, 4, 3]
    before = _token_major(kv, block_ids)
    chunks = [
        torch.randn(2, _NUM_LAYERS, _CHUNK, _NUM_HEADS * _HEAD_SIZE) for _ in range(3)
    ]
    matches = [
        _match(old_st=8, cur_st=1, n=5, tag=b"a"),
        _match(old_st=0, cur_st=6, n=3, tag=b"b"),
        _match(old_st=16, cur_st=13, n=7, tag=b"c"),
    ]

    scatter_cb_matches_to_paged_kv(kv, block_ids, chunks, matches, rope=None)

    expected = before.clone()
    for r, chunk in zip(matches, chunks, strict=True):
        expected[:, :, r.cur_st : r.cur_ed] = chunk[:, :, : r.cur_ed - r.cur_st]
    torch.testing.assert_close(_token_major(kv, block_ids), expected)


def test_scatter_rejects_a_match_beyond_the_block_table() -> None:
    kv = _nhd_kv_caches()
    chunk = torch.randn(2, _NUM_LAYERS, _CHUNK, _NUM_HEADS * _HEAD_SIZE)
    with pytest.raises(ValueError, match="exceeds"):
        scatter_cb_matches_to_paged_kv(
            kv, [0, 1], [chunk], [_match(old_st=0, cur_st=4, n=_CHUNK)], rope=None
        )


def test_normalize_cos_sin_cache_strips_mscale() -> None:
    """A YaRN-style magnitude is divided out; a unit cache is returned as is."""
    cache = _cos_sin_cache(_HEAD_SIZE)
    assert normalize_cos_sin_cache(cache) is cache
    torch.testing.assert_close(normalize_cos_sin_cache(cache * 1.3), cache)


# ---------------------------------------------------------------------------
# Server module
# ---------------------------------------------------------------------------

_SERVER_CHUNK = 256
_SERVER_BLOCK = 16
_N_CHUNKS = 3


class _LockCountingStorage:
    """Storage fake that counts read locks per key.

    The sparse prefetch locks every submitted key; ``finish_read_prefetched``
    is the only release and raises on over-release. ``unreadable`` keys are
    missing from ``unsafe_read``.
    """

    def __init__(self) -> None:
        self.locks: dict[Any, int] = {}
        self.unreadable: set[Any] = set()

    def submit_prefetch_task(self, spec: Any, external_request_id: Any = None) -> Any:
        if spec.fetching_policy == "full":
            n = int(getattr(spec, "num_kv_readers", 1) or 1)
            for row in spec.key_groups:
                for key in row.keys:
                    self.locks[key] = self.locks.get(key, 0) + n
        handle = MagicMock()
        handle.key_groups = list(spec.key_groups)
        handle.total_requested_keys = 0
        return handle

    def query_prefetch_status(self, handle: Any) -> Any:
        # First Party
        from lmcache.lmcache_native import Bitmap
        from lmcache.v1.distributed.api import PrefetchResult

        def rows() -> list[Any]:
            return [Bitmap(len(row.keys), len(row.keys)) for row in handle.key_groups]

        return PrefetchResult(
            hit_cells=rows(),
            l1_hit_cells=[Bitmap(len(row.keys)) for row in handle.key_groups],
            l2_hit_cells=rows(),
        )

    def unsafe_read(self, keys: list[Any], l1_owners: Any = None) -> tuple:
        good = [k for k in keys if k not in self.unreadable]
        return good, [MagicMock(name=f"obj-{i}") for i in range(len(good))]

    def finish_read_prefetched(
        self, keys: list[Any], read_locks: int = 1, l1_owners: Any = None
    ) -> None:
        for key in keys:
            held = self.locks.get(key, 0)
            if held < read_locks:
                raise AssertionError(f"over-release on {key}")
            if held == read_locks:
                del self.locks[key]
            else:
                self.locks[key] = held - read_locks

    def outstanding(self) -> int:
        return sum(self.locks.values())


def _server_ctx(storage: _LockCountingStorage) -> MagicMock:
    """Mock server context with a real SessionManager and one attention group.

    The prefix leg finds no chunk (no prefix coverage); sessions hash token
    ranges for real, as the store path's fingerprint registration needs.
    """
    # First Party
    from lmcache.v1.distributed.api import AttnWindowDesc
    from lmcache.v1.multiprocess.session import SessionManager
    from lmcache.v1.multiprocess.token_hasher import TokenHasher

    ctx = MagicMock()
    ctx.chunk_size = _SERVER_CHUNK
    ctx.storage_manager = storage
    ctx.event_bus.has_subscribers.return_value = False
    ctx.token_hasher.compute_chunk_hashes.return_value = []
    ctx.layout_desc_registry.find_group_layout_descs.return_value = {0: MagicMock()}
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1], world_size=1, group_kinds=("attention",)
    )
    ctx.session_manager = SessionManager(
        hasher=TokenHasher(chunk_size=_SERVER_CHUNK), ttl=3600.0, cleanup_interval=None
    )
    return ctx


def _transfer_module() -> MagicMock:
    transfer = MagicMock()
    transfer.get_block_size.return_value = _SERVER_BLOCK
    transfer.export_read_objects.return_value = PrepareRetrieveResponse(
        success=True, data=b"chunks", context={}
    )
    return transfer


@pytest.fixture
def blend_server() -> Any:
    storage = _LockCountingStorage()
    transfer = _transfer_module()
    blend = EngineDrivenBlendModule(_server_ctx(storage), transfer)
    yield blend, storage, transfer
    blend.close()


def _store_key(worker_id: int, tokens: list[int]) -> IPCCacheServerKey:
    return IPCCacheServerKey(
        model_name="m",
        world_size=2,
        num_kv_readers=1,
        worker_id=worker_id,
        token_ids=tuple(tokens),
        start=0,
        end=len(tokens),
        request_id=f"store-{worker_id}",
    )


def _shifted_lookup(
    blend: EngineDrivenBlendModule, tokens: list[int], request_id: str = "lookup"
) -> tuple[IPCCacheServerKey, list[CBMatchResult]]:
    """Look ``tokens`` up behind a 128-token prefix.

    Returns ``(key, matches)``; the sparse leg read-locks the matches.
    """
    query = list(range(50_000, 50_128)) + tokens
    key = IPCCacheServerKey(
        model_name="m",
        world_size=1,
        num_kv_readers=1,
        worker_id=None,
        token_ids=tuple(query),
        start=0,
        end=len(query),
        request_id=request_id,
    )
    result = blend.cb_unified_lookup(key, tp_size=1)
    assert result is not None
    return key, result.non_prefix_segments


def _lookup(
    blend: EngineDrivenBlendModule, transfer: MagicMock, request_id: str
) -> tuple[IPCCacheServerKey, list[CBMatchResult]]:
    """Store ``_N_CHUNKS + 1`` chunks (chunk 0 stays with the prefix leg) and
    look them up shifted. Returns ``(key, matches)`` for ``_N_CHUNKS``
    read-locked matches."""
    tokens = list(range(1000, 1000 + (_N_CHUNKS + 1) * _SERVER_CHUNK))
    transfer.commit_store_with_chunk_mask.return_value = (True, [True] * 4)
    blend.commit_store(_store_key(0, tokens), 1, b"")
    key, matches = _shifted_lookup(blend, tokens, request_id)
    assert len(matches) == _N_CHUNKS
    return key, matches


def _full_table(key: IPCCacheServerKey) -> list[list[int]]:
    return [list(range(-(-len(key.token_ids) // _SERVER_BLOCK)))]


def test_prepare_exports_the_matches_and_commit_releases_their_locks(
    blend_server: Any,
) -> None:
    blend, storage, transfer = blend_server
    key, matches = _lookup(blend, transfer, "req")
    assert storage.outstanding() == _N_CHUNKS

    response = blend.cb_prepare_retrieve(key, matches, _full_table(key), 1)

    assert response.scatter_ran is True
    assert response.matches == sorted(matches, key=lambda r: r.cur_st)
    assert response.data == b"chunks"
    assert len(transfer.export_read_objects.call_args.args[1]) == _N_CHUNKS
    # The locks stay held while the worker scatters.
    assert storage.outstanding() == _N_CHUNKS

    assert blend.cb_commit_retrieve(key, 1, scattered=True) is True
    assert storage.outstanding() == 0


def test_a_repeat_prepare_into_the_same_blocks_is_already_applied(
    blend_server: Any,
) -> None:
    blend, _, transfer = blend_server
    key, matches = _lookup(blend, transfer, "req")
    blend.cb_prepare_retrieve(key, matches, _full_table(key), 1)
    blend.cb_commit_retrieve(key, 1, scattered=True)
    transfer.export_read_objects.reset_mock()

    again = blend.cb_prepare_retrieve(key, matches, _full_table(key), 1)

    assert again == CBPrepareRetrieveResponse(scatter_ran=True)
    transfer.export_read_objects.assert_not_called()


def test_a_failed_scatter_is_not_recorded_as_applied(blend_server: Any) -> None:
    blend, storage, transfer = blend_server
    key, matches = _lookup(blend, transfer, "req")
    blend.cb_prepare_retrieve(key, matches, _full_table(key), 1)

    assert blend.cb_commit_retrieve(key, 1, scattered=False) is False
    assert storage.outstanding() == 0

    # Not applied: after a fresh lookup re-locks the chunks, the same ranges
    # are prepared again instead of being skipped.
    key, matches = _lookup(blend, transfer, "req")
    retry = blend.cb_prepare_retrieve(key, matches, _full_table(key), 1)
    assert len(retry.matches) == _N_CHUNKS


def test_prepare_defers_while_no_match_is_allocated(blend_server: Any) -> None:
    """All matches beyond the allocated slots: defer, keep the locks."""
    blend, storage, transfer = blend_server
    key, matches = _lookup(blend, transfer, "req")

    response = blend.cb_prepare_retrieve(key, matches, [[0, 1]], 1)

    assert response == CBPrepareRetrieveResponse(scatter_ran=True)
    assert storage.outstanding() == _N_CHUNKS


def test_prepare_fails_when_the_allocation_splits_the_matches(
    blend_server: Any,
) -> None:
    """Some matches allocated, some not: the client must degrade."""
    blend, storage, transfer = blend_server
    key, matches = _lookup(blend, transfer, "req")
    first_end = min(r.cur_ed for r in matches)
    table = [list(range(-(-first_end // _SERVER_BLOCK)))]

    response = blend.cb_prepare_retrieve(key, matches, table, 1)

    assert response == CBPrepareRetrieveResponse(scatter_ran=False)


def test_unreadable_objects_fail_the_prepare_and_release_the_rest(
    blend_server: Any,
) -> None:
    blend, storage, transfer = blend_server
    key, matches = _lookup(blend, transfer, "req")
    storage.unreadable.add(next(iter(storage.locks)))
    # An unreadable object holds no read lock to release.
    storage.locks.pop(next(iter(storage.unreadable)))

    response = blend.cb_prepare_retrieve(key, matches, _full_table(key), 1)

    assert response == CBPrepareRetrieveResponse(scatter_ran=False)
    assert storage.outstanding() == 0
    transfer.export_read_objects.assert_not_called()


def test_reaping_an_instance_releases_its_prepared_retrieve(
    blend_server: Any,
) -> None:
    blend, storage, transfer = blend_server
    key, matches = _lookup(blend, transfer, "req")
    blend.cb_prepare_retrieve(key, matches, _full_table(key), 1)

    blend.drop_instance_state(1)

    assert storage.outstanding() == 0
    assert blend.cb_commit_retrieve(key, 1, scattered=True) is False


def test_prepare_rejects_more_than_one_kv_group(blend_server: Any) -> None:
    blend, _, transfer = blend_server
    key, matches = _lookup(blend, transfer, "req")
    with pytest.raises(ValueError, match="one KV group"):
        blend.cb_prepare_retrieve(key, matches, [[0], [1]], 1)


def test_commit_store_registers_only_the_committed_chunks(blend_server: Any) -> None:
    """Chunk 0 belongs to the prefix leg and chunk 1 was not committed, so
    only chunk 2 becomes matchable."""
    blend, _, transfer = blend_server
    transfer.commit_store_with_chunk_mask.return_value = (True, [True, False, True])
    tokens = list(range(1000, 1000 + 3 * _SERVER_CHUNK))

    assert blend.commit_store(_store_key(0, tokens), 1, b"") is True

    _, matches = _shifted_lookup(blend, tokens)
    assert [(r.old_st, r.old_ed) for r in matches] == [
        (2 * _SERVER_CHUNK, 3 * _SERVER_CHUNK)
    ]


def test_commit_store_from_another_worker_registers_nothing(
    blend_server: Any,
) -> None:
    """The matcher is engine-shared: only worker 0 registers fingerprints."""
    blend, _, transfer = blend_server
    transfer.commit_store_with_chunk_mask.return_value = (True, [True, True, True])
    tokens = list(range(1000, 1000 + 3 * _SERVER_CHUNK))

    assert blend.commit_store(_store_key(1, tokens), 1, b"") is True

    assert _shifted_lookup(blend, tokens)[1] == []


# ---------------------------------------------------------------------------
# Worker transfer context
# ---------------------------------------------------------------------------


def _registered_worker(req_client: MagicMock) -> Any:
    """An engine-driven worker context registered on the pickle transport."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import EngineDrivenTransferContext

    register = MagicMock()
    register.result.return_value = RegisterEngineDrivenContextResponse()
    req_client.register_kv_cache_engine_driven_context.return_value = register
    context = EngineDrivenTransferContext(1, req_client)
    context.register(
        kv_caches=_nhd_kv_caches(),
        model_name="m",
        world_size=1,
        blocks_in_chunk=_CHUNK // _BLOCK_SIZE,
        mq_timeout=1.0,
    )
    return context


def _resolved(value: Any) -> MagicMock:
    future = MagicMock()
    future.result.return_value = value
    return future


def _worker_key() -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        "m", 1, 0, list(range(16)), start=0, end=16, request_id="req"
    )


def test_submit_cb_retrieve_scatters_and_commits() -> None:
    req_client = MagicMock()
    context = _registered_worker(req_client)
    kv = _nhd_kv_caches()
    block_ids = [[3, 1, 6, 0]]
    chunk = torch.randn(2, _NUM_LAYERS, _CHUNK, _NUM_HEADS * _HEAD_SIZE)
    match = _match(old_st=32, cur_st=5, n=_CHUNK)
    req_client.cb_prepare_retrieve.return_value = _resolved(
        CBPrepareRetrieveResponse(
            scatter_ran=True, matches=[match], data=pickle.dumps([chunk])
        )
    )
    req_client.cb_commit_retrieve.return_value = _resolved(True)
    key = _worker_key()

    ok = context.submit_cb_retrieve(key, kv, block_ids, [match], rope=None)

    assert ok.result() is True
    after = _token_major(kv, block_ids[0])
    torch.testing.assert_close(after[:, :, 5 : 5 + _CHUNK], chunk)
    req_client.cb_commit_retrieve.assert_called_once_with(key, 1, True)


def test_submit_cb_retrieve_without_matches_skips_the_commit() -> None:
    req_client = MagicMock()
    context = _registered_worker(req_client)
    req_client.cb_prepare_retrieve.return_value = _resolved(
        CBPrepareRetrieveResponse(scatter_ran=False)
    )

    ok = context.submit_cb_retrieve(
        _worker_key(), _nhd_kv_caches(), [[0, 1]], [_match(0, 4, 4)], rope=None
    )

    assert ok.result() is False
    req_client.cb_commit_retrieve.assert_not_called()


def test_submit_cb_retrieve_reports_a_failed_scatter_to_the_server() -> None:
    """A scatter error is committed as not scattered (locks released, range
    not applied) and degrades the request."""
    req_client = MagicMock()
    context = _registered_worker(req_client)
    # The exported chunk is shorter than the match: the scatter rejects it.
    short = torch.randn(2, _NUM_LAYERS, 2, _NUM_HEADS * _HEAD_SIZE)
    match = _match(old_st=0, cur_st=0, n=_CHUNK)
    req_client.cb_prepare_retrieve.return_value = _resolved(
        CBPrepareRetrieveResponse(
            scatter_ran=True, matches=[match], data=pickle.dumps([short])
        )
    )
    req_client.cb_commit_retrieve.return_value = _resolved(False)
    key = _worker_key()

    ok = context.submit_cb_retrieve(key, _nhd_kv_caches(), [[0, 1]], [match], None)

    assert ok.result() is False
    req_client.cb_commit_retrieve.assert_called_once_with(key, 1, False)
