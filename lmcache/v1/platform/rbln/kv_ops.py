# SPDX-License-Identifier: Apache-2.0
"""Block KV transfer for RBLN, tuned for its torch op ordering.

Chunks are staged in LMCache's canonical token-major wire layout,
``[2, L, T, H*D]`` -- the same bytes any other device writes -- so a chunk
stored from an RBLN cache can be restored into a non-RBLN one and back. What
is RBLN-specific here is the *sequence of torch ops* used to get there, not
the layout that comes out.

The baseline ``_transfer_per_layer_hnd`` walks layers with ``index_select``
into a scratch buffer, then ``index_copy_`` on the way back. These kernels
instead gather with one ``stack`` per block, each written directly into its
slice of a single pre-sized buffer, and scatter with a single
``torch._foreach_copy_`` over every (block, layer, K/V) pair, which is the op
sequence RBLN's v2v kernels are tuned for.

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
import threading
from typing import Sequence

# Third Party
import torch


#: Host landing buffers for the device<->host leg, per thread.
#:
#: Thread-local rather than module-global on purpose: the multiprocess server
#: runs blocking handlers on a thread pool, so a store and a retrieve -- or two
#: stores -- can be in ``multi_layer_block_kv_transfer`` at the same time. A
#: shared buffer would let one overwrite the other's staged bytes between its
#: transpose and its copy, corrupting the KV cache with no error anywhere.
#: The cost is one resident buffer per (thread, chunk geometry).
_STAGING = threading.local()


def _host_staging(
    n_blocks: int, per_block: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    """Return this thread's contiguous host buffer shaped ``[n_blocks, *per_block]``.

    One buffer per (``per_block``, ``dtype``) per thread, grown to the largest
    block count asked for and sliced down, so a varying prefix skip or a
    trailing short chunk reuses the same allocation instead of adding another
    resident one. Slicing the leading dimension keeps it contiguous, which the
    per-(block, layer) copies depend on.

    Args:
        n_blocks: Blocks the caller needs room for.
        per_block: Shape of one block's staging area.
        dtype: Element type, matching the chunk and the paged tensors.

    Returns:
        torch.Tensor: A contiguous view with ``n_blocks`` leading entries,
        owned by the calling thread.
    """
    buffers = getattr(_STAGING, "buffers", None)
    if buffers is None:
        buffers = _STAGING.buffers = {}
    key = (per_block, dtype)
    buf = buffers.get(key)
    if buf is None or buf.shape[0] < n_blocks:
        buf = torch.empty((n_blocks, *per_block), dtype=dtype, device="cpu")
        buffers[key] = buf
    return buf[:n_blocks]


def gather_blocks_to_chunk(
    paged_layers: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    dst: torch.Tensor,
) -> None:
    """Gather whole paged blocks into a token-major chunk.

    Args:
        paged_layers: Per-layer HND KV tensors, each ``[2, NB, NH, BS, HS]``.
        block_ids: Blocks to gather, in chunk-token order.
        dst: Chunk shaped ``[2, L, T, H*D]``. Only its leading
            ``len(block_ids) * BS`` tokens are written, so a trailing chunk
            holding fewer blocks than it was sized for is fine. May be on
            device (D2D) or host (D2H); ``copy_`` handles the transfer.
    """
    _kv, _nb, num_heads, block_size, head_size = paged_layers[0].shape
    n_blocks = len(block_ids)
    # One buffer for the whole gather, block-major so that buf[i] is
    # contiguous: each block's stack writes straight into its slice through
    # out=, leaving nothing to concatenate afterwards. Stacking into separate
    # tensors and cat-ing them costs a second chunk-sized allocation plus a
    # full extra pass over the chunk, which measured ~2x the latency at four
    # blocks on RBLN. A non-contiguous out= would be worse than either: the
    # kernels stage it through a temporary and copy again.
    buf = torch.empty(
        n_blocks,
        2,
        len(paged_layers),
        num_heads,
        block_size,
        head_size,
        dtype=paged_layers[0].dtype,
        device=paged_layers[0].device,
    )
    # Keeping the K/V axis in the per-layer view means one stack over layers
    # yields [2, L, H, BS, D] directly -- no separate k/v stacks and no
    # trailing recombine.
    for position, block in enumerate(block_ids):
        torch.stack(
            [layer[:, block] for layer in paged_layers], dim=1, out=buf[position]
        )
    # Handing copy_ a permuted device source makes the H<->T transpose part of
    # the device->host copy, which splits it into head_size-sized runs; the
    # descriptors that come out of that are what makes a token-major chunk
    # expensive to fill from an HND cache. Landing the block-major buffer on
    # the host as one contiguous copy first, and transposing there, keeps the
    # device->host leg at full width and pays for the transpose in host memory
    # bandwidth instead.
    staged = _host_staging(
        n_blocks,
        (2, len(paged_layers), num_heads, block_size, head_size),
        buf.dtype,
    )
    staged.copy_(buf)
    # Splitting H*D, splitting the token axis into (block, token-in-block), and
    # transposing H<->T are all views, so copy_ reads transposed rather than
    # materialising a permuted intermediate. reshape() here would not be a
    # view: it would hit contiguous() on the permuted source and rebuild the
    # very chunk-sized temporary this buffer exists to avoid.
    tokens = dst.unflatten(-1, (num_heads, head_size))
    by_block = tokens[:, :, : n_blocks * block_size].unflatten(
        2, (n_blocks, block_size)
    )
    by_block.copy_(staged.permute(1, 2, 0, 4, 3, 5))


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
        src: Chunk shaped ``[2, L, T, H*D]``. Only the token windows the
            blocks map to are read, so a trailing chunk holding fewer blocks
            than it was sized for is fine.
        skip_prefix_n_blocks: Leading blocks already present in the KV cache;
            neither read from ``src`` nor written.
    """
    _kv, _nb, num_heads, block_size, head_size = paged_layers[0].shape
    n_blocks = len(block_ids)
    start = min(skip_prefix_n_blocks, n_blocks)
    if start >= n_blocks:
        return
    tokens = src.unflatten(-1, (num_heads, head_size))

    # Mirror of the gather: handing copy_ a permuted host source would make the
    # T<->H transpose part of the host->device copy and split it into
    # head_size-sized runs. Transposing on the host first, into a block-major
    # buffer, leaves every host->device copy a contiguous [H, BS, D] block.
    n_staged = n_blocks - start
    staged = _host_staging(
        n_staged,
        (2, len(paged_layers), num_heads, block_size, head_size),
        src.dtype,
    )
    by_block = tokens[:, :, start * block_size : n_blocks * block_size].unflatten(
        2, (n_staged, block_size)
    )
    staged.permute(1, 2, 0, 4, 3, 5).copy_(by_block)

    dsts: list[torch.Tensor] = []
    srcs: list[torch.Tensor] = []
    for position in range(start, n_blocks):
        block = block_ids[position]
        for layer_idx, layer in enumerate(paged_layers):
            # Both sides are contiguous [H, BS, D] now, so each copy is one run.
            dsts.append(layer[0, block])
            srcs.append(staged[position - start, 0, layer_idx])
            dsts.append(layer[1, block])
            srcs.append(staged[position - start, 1, layer_idx])
    torch._foreach_copy_(dsts, srcs)
