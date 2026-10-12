# SPDX-License-Identifier: Apache-2.0
"""Engine-driven multi-group transfer over kernel-paged KV groups.

vLLM may page a group's tensor at a kernel block size smaller than its block
size (``kernel_block_size``): engine block ``b`` then covers tensor pages
``b*k .. b*k+k-1``. vLLM-RBLN does this for a sliding-window group, paging
4096-token blocks as 128-token windows. These tests pin that registration
reads such a group as kernel-paged rather than compressed, that store and
retrieve move tensor pages, and that a sliding window of one page keeps only
each chunk's last page.

Tensors are ``[2, NB, X, X, HS]`` with the two middle axes equal, so the block
size is the same whichever of NHD / HND the detector resolves.
"""

# Standard
from typing import Any
from unittest.mock import MagicMock

# Third Party
import torch

# First Party
from lmcache.v1.multiprocess.custom_types import (
    RegisterEngineDrivenContextPayload,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.transfer_context import worker_transfer
from lmcache.v1.multiprocess.transfer_context.pickle import EngineDrivenContextPickle
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
)

NUM_BLOCKS = 4  # engine block IDs per group
BLOCK_TOKENS = 8  # tokens one engine block ID covers
PAGES_PER_BLOCK = 2  # kernel pages per engine block in the paged group
PAGE_TOKENS = BLOCK_TOKENS // PAGES_PER_BLOCK
HEAD_SIZE = 4
BLOCKS_IN_CHUNK = 2
CHUNK_TOKENS = BLOCKS_IN_CHUNK * BLOCK_TOKENS


def _kv(seed: int = 0) -> dict[str, torch.Tensor]:
    """A full-attention layer and a kernel-paged sliding-window layer."""
    torch.manual_seed(seed)
    return {
        "full": torch.randn(2, NUM_BLOCKS, BLOCK_TOKENS, BLOCK_TOKENS, HEAD_SIZE),
        "swa": torch.randn(
            2, NUM_BLOCKS * PAGES_PER_BLOCK, PAGE_TOKENS, PAGE_TOKENS, HEAD_SIZE
        ),
    }


def _group_infos() -> list[EngineGroupInfo]:
    return [
        EngineGroupInfo(
            engine_group_id=0,
            layer_indices=(0,),
            tokens_per_block=BLOCK_TOKENS,
            num_blocks=NUM_BLOCKS,
        ),
        EngineGroupInfo(
            engine_group_id=1,
            layer_indices=(1,),
            tokens_per_block=BLOCK_TOKENS,
            sw_size_tokens=PAGE_TOKENS,
            num_blocks=NUM_BLOCKS,
        ),
    ]


def _register(
    kv: dict[str, torch.Tensor],
) -> tuple[EngineDrivenTransferContext, RegisterEngineDrivenContextPayload]:
    """Register ``kv`` and return the context and the payload it sent."""
    sent: list[Any] = []

    def _register_rpc(payload):
        sent.append(payload)
        future = MagicMock()
        future.result.return_value = RegisterEngineDrivenContextResponse(
            shm_name="lmcache_l1_pool_x", pool_size=4096
        )
        return future

    req_client = MagicMock()
    req_client.register_kv_cache_engine_driven_context.side_effect = _register_rpc
    ctx = EngineDrivenTransferContext(1, req_client)
    ctx.register(
        kv_caches=kv,
        model_name="m",
        world_size=1,
        blocks_in_chunk=BLOCKS_IN_CHUNK,
        mq_timeout=1.0,
        engine_group_infos=_group_infos(),
    )
    return ctx, sent[0]


def test_kernel_paged_window_group_registers_in_pages() -> None:
    """The paged group is not refused as compressed; its counts are in pages."""
    ctx, payload = _register(_kv())

    full, swa = ctx._group_states
    assert full.kernel_pages_per_block == 1
    assert full.blocks_in_chunk == full.blocks_per_window == BLOCKS_IN_CHUNK
    assert swa.kernel_pages_per_block == PAGES_PER_BLOCK
    assert swa.blocks_in_chunk == CHUNK_TOKENS // PAGE_TOKENS
    assert swa.blocks_per_window == 1
    assert payload.group_layouts[0].window_tokens == CHUNK_TOKENS
    assert payload.group_layouts[1].window_tokens == PAGE_TOKENS


def test_store_expands_engine_block_ids_to_pages(monkeypatch) -> None:
    """Gather receives page IDs ``b*k .. b*k+k-1`` for the paged group only."""
    ctx, _ = _register(_kv())
    pctx = EngineDrivenContextPickle(
        metadata=MagicMock(), req_client=MagicMock(), mq_timeout=0.1
    )
    ctx._engine_driven_context = pctx
    monkeypatch.setattr(pctx, "prepare_store", lambda _k, _i: None)
    monkeypatch.setattr(pctx, "commit_store", lambda _k, _i, _c: True)
    gathered: list[list[int]] = []

    def _fake_gather(_kv, block_ids, _blocks_in_chunk, **_kwargs: Any):
        gathered.append(block_ids)
        return []

    monkeypatch.setattr(worker_transfer, "gather_paged_kv_to_cpu", _fake_gather)

    future = ctx.submit_store("req", MagicMock(), _kv(), [[3, 1], [3, 1]], None, 2)

    assert future.result(timeout=1) is True
    assert gathered == [[3, 1], [6, 7, 2, 3]]


def test_window_group_round_trip_restores_each_chunks_last_page(
    monkeypatch,
) -> None:
    """Store then retrieve restores the trailing page of each chunk only."""
    ctx, _ = _register(_kv())
    pctx = EngineDrivenContextPickle(
        metadata=MagicMock(), req_client=MagicMock(), mq_timeout=0.1
    )
    ctx._engine_driven_context = pctx
    stored: dict[str, Any] = {}
    monkeypatch.setattr(pctx, "prepare_store", lambda _k, _i: None)
    monkeypatch.setattr(
        pctx,
        "commit_store",
        lambda _k, _i, chunks: stored.update(chunks=chunks) or True,
    )
    monkeypatch.setattr(
        pctx, "prepare_retrieve_multigroup", lambda _k, _i: stored["chunks"]
    )
    monkeypatch.setattr(pctx, "commit_retrieve", lambda _k, _i: True)

    src = _kv(seed=1)
    dst = {name: torch.zeros_like(t) for name, t in src.items()}
    # Two chunks of two engine blocks each: [2, 0] and [3, 1].
    block_ids = [[2, 0, 3, 1], [2, 0, 3, 1]]
    assert ctx.submit_store("req", MagicMock(), src, block_ids, None, 2).result(1)
    assert ctx.submit_retrieve("req", MagicMock(), dst, block_ids, None, 2).result(1)

    # Full attention: every engine block comes back.
    assert torch.equal(dst["full"], src["full"])
    # A one-page window keeps each chunk's last page: block 0 -> page 1 and
    # block 1 -> page 3. Every other page stays untouched.
    restored = [
        p for p in range(NUM_BLOCKS * PAGES_PER_BLOCK) if dst["swa"][:, p].any()
    ]
    assert restored == [1, 3]
    for page in restored:
        assert torch.equal(dst["swa"][:, page], src["swa"][:, page])
