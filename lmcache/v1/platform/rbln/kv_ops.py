# SPDX-License-Identifier: Apache-2.0
"""Block KV transfer for RBLN, tuned for its torch op ordering.

Chunks are staged in LMCache's canonical token-major wire layout,
``[2, L, T, H*D]`` -- the same bytes any other device writes -- so a chunk
stored from an RBLN cache can be restored into a non-RBLN one and back. What
is RBLN-specific here is the *sequence of torch ops* used to get there, not
the layout that comes out.

The baseline ``_transfer_per_layer_hnd`` walks layers with ``index_select``
into a scratch buffer, then ``index_copy_`` on the way back. These kernels
instead gather with one ``stack`` per block plus a single ``cat``, and scatter
with a single ``torch._foreach_copy_`` over every (block, layer, K/V) pair,
which is the op sequence RBLN's v2v kernels are tuned for.

RBLN stores heads before block tokens (HND), so a head<->token transpose is
inherent to writing a token-major chunk. It is expressed as a ``permute`` on
the view handed to ``copy_`` rather than as a materialised tensor, so no
intermediate is allocated for it.

Implemented with torch ops only -- no compiled extension. Device behaviour is
not owned here: on RBLN the same lines dispatch to the backend's v2v kernels
against a KV cache whose physical layout is sharded across chiplets, and
neither that nor the transfer cost has an equivalent when the tensors happen
to be on CPU.
"""

# Future
from __future__ import annotations

# Standard
from typing import Sequence

# Third Party
import torch


def gather_blocks_to_chunk(
    paged_layers: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    dst: torch.Tensor,
) -> None:
    """Gather whole paged blocks into a token-major chunk.

    Args:
        paged_layers: Per-layer HND KV tensors, each ``[2, NB, NH, BS, HS]``.
        block_ids: Blocks to gather, in chunk-token order.
        dst: Chunk region shaped ``[2, L, len(block_ids) * BS, H*D]``. May be
            on device (D2D) or host (D2H); ``copy_`` handles the transfer.
    """
    # Keeping the K/V axis in the per-layer view means one stack over layers
    # yields [2, L, H, BS, D] directly -- no separate k/v stacks and no
    # trailing recombine.
    pieces = [
        torch.stack([layer[:, block] for layer in paged_layers], dim=1)
        for block in block_ids
    ]
    gathered = torch.cat(pieces, dim=3) if len(pieces) > 1 else pieces[0]
    # Splitting H*D and transposing H<->T are both views, so copy_ reads
    # transposed rather than materialising a permuted intermediate.
    num_heads, head_size = gathered.shape[2], gathered.shape[4]
    dst.unflatten(-1, (num_heads, head_size)).copy_(gathered.permute(0, 1, 3, 2, 4))


def scatter_chunk_to_blocks(
    paged_layers: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    src: torch.Tensor,
    skip_prefix_n_blocks: int = 0,
) -> None:
    """Scatter a token-major chunk back into whole paged blocks.

    Args:
        paged_layers: Per-layer HND KV tensors, each ``[2, NB, NH, BS, HS]``.
        block_ids: Destination blocks, in chunk-token order.
        src: Chunk region shaped ``[2, L, len(block_ids) * BS, H*D]``.
        skip_prefix_n_blocks: Leading blocks already present in the KV cache;
            neither read from ``src`` nor written.
    """
    _kv, _nb, num_heads, block_size, head_size = paged_layers[0].shape
    tokens = src.unflatten(-1, (num_heads, head_size))
    dsts: list[torch.Tensor] = []
    srcs: list[torch.Tensor] = []
    for position, block in enumerate(block_ids):
        if position < skip_prefix_n_blocks:
            continue
        window = tokens[:, :, position * block_size : (position + 1) * block_size]
        for layer_idx, layer in enumerate(paged_layers):
            # [BS, H, D] -> [H, BS, D]: a view, so the transpose costs no
            # allocation and rides along with the copy.
            dsts.append(layer[0, block])
            srcs.append(window[0, layer_idx].permute(1, 0, 2))
            dsts.append(layer[1, block])
            srcs.append(window[1, layer_idx].permute(1, 0, 2))
    if dsts:
        torch._foreach_copy_(dsts, srcs)
