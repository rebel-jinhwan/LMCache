# SPDX-License-Identifier: Apache-2.0
"""Device-staged L1 round trip through RblnCacheContext (hardware).

The objects come from
``tests/v1/distributed/device_staged_l1_fixture.py``: each lends a fresh
device buffer and keeps a host copy of what a store gathered, so this checks
the cache context's half of the ``DeviceStagedMemoryObj`` hook without any
vendor medium.

The KV lives in this process, so no IPC export is needed: the cache context
gets plain device tensors through a pass-through wrapper. The tensors are bound
with ``bind_device_memory``, which lays them out flat on chiplet 0 rather than
sharded like a real vLLM KV cache. That is fine for correctness, but timings
from this test say nothing about real KV transfer.

Two KV groups, as in ``test_rbln_cache_context``:

- group 0: full attention, ``[NB, 2, H, 1, BS, D]``;
- group 1: sliding window of one kernel page, ``[NB * K, 2, H, 1, BS / K, D]``.

A chunk is stored into the objects (D2D gather into the lent buffer), the KV
is wiped, and the chunk is retrieved into other blocks (D2D scatter from a
buffer the object refilled). Restored pages must match the stored ones bit for
bit, and nothing else may be written.
"""

# Standard
from math import prod
from typing import Any, cast

# Third Party
import pytest
import torch

NB, H, BS, D, K = 6, 2, 128, 64, 4


def _hardware_ready() -> bool:
    try:
        # Third Party
        import torch_rbln  # noqa: F401
    except ImportError:
        return False
    return torch.rbln.device_count() > 0  # type: ignore[attr-defined]


pytestmark = pytest.mark.skipif(
    not _hardware_ready(), reason="needs torch-rbln and an NPU"
)


class _Local:
    """Pass-through stand-in for an IPC wrapper over an in-process tensor."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def to_tensor(self) -> torch.Tensor:
        return self._tensor

    def close(self) -> None:
        return


def _bound(shape: tuple[int, ...]) -> torch.Tensor:
    # Third Party
    from torch_rbln.memory import bind_device_memory

    tensor = torch.empty(prod(shape), dtype=torch.bfloat16, device="rbln:0")
    bind_device_memory(tensor)
    return tensor.view(shape)


@pytest.mark.parametrize("chunk_blocks", [1, 2])
def test_chunk_round_trips_through_lent_buffers(chunk_blocks: int) -> None:
    # First Party
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo
    from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
        get_layout_desc,
    )
    from lmcache.v1.multiprocess.object_group_transfer import (
        downsample_and_stage_block_ids,
    )
    from lmcache.v1.platform.devices.rbln.cache_context import RblnCacheContext
    from tests.v1.distributed.device_staged_l1_fixture import (
        BudgetMemoryManager,
        HostCopyObject,
    )
    import lmcache.lmcache_native as native

    full = _bound((NB, 2, H, 1, BS, D))
    sw = _bound((NB * K, 2, H, 1, BS // K, D))
    gen = torch.Generator().manual_seed(chunk_blocks)
    full_ref = torch.randn(full.shape, generator=gen).to(torch.bfloat16)
    sw_ref = torch.randn(sw.shape, generator=gen).to(torch.bfloat16)
    full.copy_(full_ref)
    sw.copy_(sw_ref)
    torch.rbln.synchronize()  # type: ignore[attr-defined]

    chunk_tokens = BS * chunk_blocks
    ctx = RblnCacheContext(
        cast(Any, [_Local(full), _Local(sw)]),
        lmcache_tokens_per_chunk=chunk_tokens,
        engine_group_infos=[
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=(0,),
                tokens_per_block=BS,
                num_blocks=NB,
            ),
            EngineGroupInfo(
                engine_group_id=1,
                layer_indices=(1,),
                tokens_per_block=BS,
                sw_size_tokens=BS // K,
                num_blocks=NB,
            ),
        ],
        separate_object_groups=True,
    )
    manager = ctx.kv_layer_groups_manager
    memory = BudgetMemoryManager(1 << 30, 4096)
    store_blocks = list(range(2, 2 + chunk_blocks))
    retrieve_blocks = list(range(chunk_blocks))

    def transfer(blocks: list[int], objects: dict, direction) -> None:
        staged = downsample_and_stage_block_ids(
            ctx, [blocks for _ in manager.kernel_groups]
        )
        for og_idx, obj in objects.items():
            ctx.transfer_object_group(
                staged,
                [obj],
                object_group_id=og_idx,
                batch_size=1,
                skip_first_n_tokens=0,
                direction=direction,
                transfer_key="test",
            )

    objects: dict[int, HostCopyObject] = {}
    for og_idx in range(len(manager.object_groups)):
        _, (obj,) = memory.allocate(get_layout_desc(ctx, chunk_tokens, og_idx), 1)
        assert isinstance(obj, HostCopyObject)
        objects[og_idx] = obj
    transfer(store_blocks, objects, native.TransferDirection.D2H)
    for obj in objects.values():
        assert obj.transfers == [True] and obj.saved is not None
        assert obj.saved.any(), "the store gathered nothing into the buffer"

    full.copy_(torch.zeros(full.shape, dtype=torch.bfloat16))
    sw.copy_(torch.zeros(sw.shape, dtype=torch.bfloat16))
    torch.rbln.synchronize()  # type: ignore[attr-defined]
    transfer(retrieve_blocks, objects, native.TransferDirection.H2D)
    torch.rbln.synchronize()  # type: ignore[attr-defined]
    full_host, sw_host = full.cpu(), sw.cpu()

    # Full attention: every block of the chunk, bit for bit, nothing else.
    for src, dst in zip(store_blocks, retrieve_blocks, strict=True):
        assert torch.equal(full_host[dst], full_ref[src]), (src, dst)
    for block in set(range(NB)) - set(retrieve_blocks):
        assert not full_host[block].any(), block
    # Sliding window: only the chunk's last kernel page.
    src_page = store_blocks[-1] * K + K - 1
    dst_page = retrieve_blocks[-1] * K + K - 1
    assert torch.equal(sw_host[dst_page], sw_ref[src_page])
    for page in set(range(NB * K)) - {dst_page}:
        assert not sw_host[page].any(), page

    assert all(obj.transfers == [True, False] for obj in objects.values())
    memory.free(list(objects.values()))
    assert memory.get_memory_usage()[0] == 0
