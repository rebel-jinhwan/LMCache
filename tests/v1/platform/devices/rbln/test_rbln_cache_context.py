# SPDX-License-Identifier: Apache-2.0
"""RblnCacheContext moves the right pages of a worker's KV (hardware).

A spawned process plays the LMCache server: it opens a worker's two KV groups
from their descriptors and stores / retrieves one chunk of each.

- group 0: full attention, ``[NB, 2, H, 1, BS, D]``;
- group 1: sliding window of one kernel page, paged at ``BS / K`` tokens as
  ``[NB * K, 2, H, 1, BS / K, D]`` (vLLM ``kernel_block_size``, as vLLM-RBLN
  pages gpt-oss).

Every block / page holds its own index, so a chunk's values name the pages it
was gathered from and the exporter can see which pages a retrieve wrote.

Needs a torch-rbln with ``torch.rbln.ipc`` and read/write access to
``/dev/accel/accel0``.
"""

# Standard
from types import SimpleNamespace
from typing import Any
import multiprocessing
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
from lmcache.v1.platform.devices.rbln.ipc_wrapper import (
    RblnIPCWrapper,
    is_rbln_memory_ipc_available,
)

NB, H, BS, D, K = 4, 2, 128, 64, 4
FULL_BASE = 100
STORE_BLOCK, RETRIEVE_BLOCK = 2, 1
FULL_RETRIEVED, SW_RETRIEVED = 50.0, 60.0


def _hardware_ready() -> bool:
    return is_rbln_memory_ipc_available() and os.access(
        "/dev/accel/accel0", os.R_OK | os.W_OK
    )


def _exportable(shape: tuple[int, ...]) -> torch.Tensor:
    numel = 1
    for size in shape:
        numel *= size
    buffer = torch.rbln.ipc.empty_exportable((numel * 2,), torch.int8, device=0)  # type: ignore[attr-defined]
    return buffer.view(torch.bfloat16).view(shape)


def _server(descriptors: list[bytes], conn: Any) -> None:
    """Store one chunk of each group, report its values, then retrieve one."""
    # First Party
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo
    from lmcache.v1.multiprocess.object_group_transfer import (
        downsample_and_stage_block_ids,
    )
    from lmcache.v1.platform.devices.rbln.cache_context import RblnCacheContext
    import lmcache.lmcache_native as native

    infos = [
        EngineGroupInfo(
            engine_group_id=0, layer_indices=(0,), tokens_per_block=BS, num_blocks=NB
        ),
        EngineGroupInfo(
            engine_group_id=1,
            layer_indices=(1,),
            tokens_per_block=BS,
            sw_size_tokens=BS // K,
            num_blocks=NB,
        ),
    ]
    ctx = RblnCacheContext(
        [DeviceIPCWrapper.Deserialize(d) for d in descriptors],
        lmcache_tokens_per_chunk=BS,
        engine_group_infos=infos,
        separate_object_groups=True,
    )
    manager = ctx.kv_layer_groups_manager
    engine_group = [g.engine_group_idx for g in manager.kernel_groups]
    pages = {
        engine_group[i]: g.kernel_pages_per_block
        for i, g in enumerate(manager.kernel_groups)
    }

    def objects(fill: dict[int, float]) -> list[tuple[int, Any, list[Any]]]:
        """One host memory object per object group, region per kernel group."""
        out = []
        for og_idx, og in enumerate(manager.object_groups):
            regions = []
            for kg in og.kernel_group_indices:
                shape, dtype = ctx.get_kernel_group_shape_dtype(BS, kg)
                regions.append((kg, shape, dtype))
            nbytes = sum(s.numel() * d.itemsize for _, s, d in regions)
            raw = torch.zeros(nbytes, dtype=torch.uint8)
            views = []
            offset = 0
            for kg, shape, dtype in regions:
                size = shape.numel() * dtype.itemsize
                view = raw[offset : offset + size].view(dtype).view(shape)
                if engine_group[kg] in fill:
                    view.fill_(fill[engine_group[kg]])
                views.append((engine_group[kg], view))
                offset += size
            out.append((og_idx, SimpleNamespace(raw_tensor=raw), views))
        return out

    def run(block: int, direction: Any, fill: dict[int, float]) -> list[Any]:
        staged = downsample_and_stage_block_ids(
            ctx, [[block] for _ in manager.kernel_groups]
        )
        objs = objects(fill)
        for og_idx, obj, _ in objs:
            ctx.transfer_object_group(
                staged,
                [obj],
                object_group_id=og_idx,
                batch_size=1,
                skip_first_n_tokens=0,
                direction=direction,
                transfer_key="test",
            )
        return objs

    stored = {
        group: sorted(set(view.float().flatten().tolist()))
        for _, _, views in run(STORE_BLOCK, native.TransferDirection.D2H, {})
        for group, view in views
    }
    run(
        RETRIEVE_BLOCK,
        native.TransferDirection.H2D,
        {0: FULL_RETRIEVED, 1: SW_RETRIEVED},
    )
    torch.rbln.synchronize()  # type: ignore[attr-defined]
    ctx.close()
    conn.send((pages, stored))


@pytest.mark.skipif(not _hardware_ready(), reason="needs torch.rbln.ipc and an NPU")
def test_server_moves_full_blocks_and_the_last_kernel_page() -> None:
    """Store reads the chunk's pages; retrieve writes exactly those pages."""
    full = _exportable((NB, 2, H, 1, BS, D))
    sw = _exportable((NB * K, 2, H, 1, BS // K, D))
    for block in range(NB):
        full[block].fill_(FULL_BASE + block)
    for page in range(NB * K):
        sw[page].fill_(page)
    torch.rbln.synchronize()  # type: ignore[attr-defined]
    descriptors = [
        DeviceIPCWrapper.Serialize(RblnIPCWrapper.wrap(t)) for t in (full, sw)
    ]

    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=_server, args=(descriptors, child))
    proc.start()
    pages, stored = parent.recv()
    proc.join(timeout=120)
    assert proc.exitcode == 0

    # The sliding-window group is read as kernel-paged, K pages per block.
    assert pages == {0: 1, 1: K}
    # Store: the full block, and only the last page of the sliding block.
    assert stored == {
        0: [float(FULL_BASE + STORE_BLOCK)],
        1: [float(STORE_BLOCK * K + K - 1)],
    }
    # Retrieve: the full block is overwritten; of the sliding block only the
    # window's page is, and every other page keeps its index.
    full_host, sw_host = full.float().cpu(), sw.float().cpu()
    for block in range(NB):
        expected = FULL_RETRIEVED if block == RETRIEVE_BLOCK else FULL_BASE + block
        assert bool((full_host[block] == expected).all()), block
    window_page = RETRIEVE_BLOCK * K + K - 1
    for page in range(NB * K):
        expected = SW_RETRIEVED if page == window_page else page
        assert bool((sw_host[page] == expected).all()), page
