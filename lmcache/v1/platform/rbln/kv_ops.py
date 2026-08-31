# SPDX-License-Identifier: Apache-2.0
"""Block KV transfer for RBLN (MLA layout).

HND token-major transfers run in the native extension ``lmcache.rbln_ops``
(``csrc/rbln``): device-to-device gather into device staging, the head<->token
swap on the device (``torch.rbln.swap_axes_12``), and host copies on a separate
stream pipelined against the next slice.
"""

# Future
from __future__ import annotations

# Standard
from typing import Sequence

# Third Party
import torch


def gather_blocks_to_chunk_mla(
    paged_layers: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    dst: torch.Tensor,
) -> None:
    """Gather whole MLA paged blocks into a single-plane chunk.

    The shared torch MLA path stages through ``torch.empty(device=...)`` and
    ``index_select(out=...)``; on RBLN a raw empty device tensor is a lazy
    SHM tensor, so ops against it take the CPU-fallback path instead of the
    chip. This sequence builds the gather result *functionally* --
    ``index_select`` per layer, ``stack`` across layers, both device-native
    v2v kernels on RBLN -- then lands it with one ``copy_`` across the
    device boundary.

    Args:
        paged_layers: Per-layer MLA KV tensors, each ``[NB, BS, HS]``.
        block_ids: Blocks to gather, in chunk-token order.
        dst: Chunk shaped ``[L, T, HS]``. Only its leading
            ``len(block_ids) * BS`` tokens are written, so a trailing chunk
            holding fewer blocks than it was sized for is fine.
    """
    n_blocks = len(block_ids)
    _nb, block_size, head_size = paged_layers[0].shape
    idx = torch.as_tensor(block_ids, dtype=torch.long, device=paged_layers[0].device)
    # [L, B, BS, HS] on the paged tensors' device; stack decomposes to the
    # v2v cat kernel, and the result is contiguous so the view below is free.
    gathered = torch.stack(
        [torch.index_select(layer, 0, idx) for layer in paged_layers]
    )
    dst[:, : n_blocks * block_size].copy_(
        gathered.view(len(paged_layers), n_blocks * block_size, head_size)
    )


def scatter_chunk_to_blocks_mla(
    paged_layers: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    src: torch.Tensor,
    skip_prefix_n_blocks: int = 0,
) -> None:
    """Scatter a single-plane chunk back into whole MLA paged blocks.

    Mirror of :func:`gather_blocks_to_chunk_mla`: one ``.to(device)`` DMA for
    the chunk window, then a device-native ``index_copy_`` per layer.

    Args:
        paged_layers: Per-layer MLA KV tensors, each ``[NB, BS, HS]``.
        block_ids: Destination blocks, in chunk-token order.
        src: Chunk shaped ``[L, T, HS]``. Only the token windows the blocks
            map to are read, so a trailing chunk holding fewer blocks than it
            was sized for is fine.
        skip_prefix_n_blocks: Leading blocks already present in the KV cache;
            neither read from ``src`` nor written.
    """
    n_blocks = len(block_ids)
    start = min(skip_prefix_n_blocks, n_blocks)
    if start >= n_blocks:
        return
    _nb, block_size, head_size = paged_layers[0].shape
    device = paged_layers[0].device
    n_valid = n_blocks - start
    idx = torch.as_tensor(
        list(block_ids)[start:n_blocks], dtype=torch.long, device=device
    )
    # ``.to`` contiguizes the strided host window as part of the transfer,
    # so the per-layer views below are views, not copies.
    chunk_dev = src[:, start * block_size : n_blocks * block_size].to(device)
    for layer_idx, layer in enumerate(paged_layers):
        layer.index_copy_(
            0, idx, chunk_dev[layer_idx].view(n_valid, block_size, head_size)
        )
