# SPDX-License-Identifier: Apache-2.0
"""Device-to-device block moves for device-staged L1 objects on RBLN.

A :class:`~lmcache.v1.memory_management.DeviceStagedMemoryObj` lends the
cache context a device buffer per transfer. This module moves one chunk's KV
pages between the paged KV tensors and such a buffer without leaving the
device.

The buffer's bytes are block-major, ``[B, L, 2, H, BS, D]`` for HND KV: every
(block, layer) pair is one contiguous run on both sides, so each copy is a
direct ``memcpy_v2v`` and no head<->token transpose is needed. That differs
from the canonical token-major ``[2, L, T, H*D]`` host tiers hold, which is
why a device-staged tier is closed to everything but this cache context (see
``DeviceStagedMemoryObj``). MLA KV already stages on the device in
``device_ops``, so its buffer keeps the canonical layout.

A hybrid model's full-attention KV can be laid out in the sliding window's
smaller kernel pages, so one block copy spans many runtime mappings. An RBLN
runtime that caps a device-to-device copy plan at 30 mappings syncs the whole
KV tensor through the host for each such copy: correct, but seconds per
chunk. Use a runtime without that cap for hybrid models.
"""

# Standard
from typing import Any, Sequence

# Third Party
import torch

# First Party
import lmcache.lmcache_native as lmcache_native


def transfer_blocks_on_device(
    kv_layers: Sequence[torch.Tensor],
    chunk: torch.Tensor,
    pages: Sequence[int],
    direction: lmcache_native.TransferDirection,
    shape_desc: Any,
    slots_per_chunk: int,
    engine_kv_format: lmcache_native.EngineKVFormat,
    skip_pages: int,
) -> None:
    """Move one chunk's pages between the KV and a device chunk area, D2D.

    Args:
        kv_layers: The kernel group's per-layer KV tensors.
        chunk: The group's region of the staging area, shaped like the
            canonical chunk ``[2, L, slots, H*D]`` (``[L, slots, HS]`` for MLA).
            For HND KV its bytes are reinterpreted block-major.
        pages: Tensor pages of this chunk, in token order.
        direction: ``D2H`` to store (KV -> chunk), ``H2D`` to retrieve.
        shape_desc: Paged-buffer shape descriptor (MLA path only).
        slots_per_chunk: Token slots the group stores per chunk (MLA path only).
        engine_kv_format: Engine KV layout.
        skip_pages: Leading pages a retrieve leaves untouched.

    Raises:
        ValueError: For a layout neither HND nor MLA, or a chunk whose width
            does not match the KV's heads.
    """
    # First Party
    from lmcache import device_ops
    from lmcache.v1.gpu_connector.kv_format.specs.registry import get_spec_class
    from lmcache.v1.platform.devices.rbln.kv_layout import (
        squeeze_singleton_axis,
        squeeze_singleton_axis_kv_first,
    )
    from lmcache.v1.platform.devices.rbln.kv_ops import (
        _hnd_block_pairs,
        _hnd_kv_first_block_pairs,
    )

    spec = get_spec_class(engine_kv_format)
    is_hnd = spec.is_layer_list and spec.is_hnd and not spec.is_fused_packed
    if not is_hnd:
        # The MLA path already stages on the device and crosses into the chunk
        # with one contiguous copy, which is D2D when the chunk is on device.
        device_ops.multi_layer_block_kv_transfer(
            list(kv_layers),
            [chunk],
            list(pages),
            chunk.device,
            direction,
            shape_desc,
            slots_per_chunk,
            engine_kv_format,
            skip_pages,
        )
        return

    if spec.is_two_major:
        prepare, pair = squeeze_singleton_axis_kv_first, _hnd_kv_first_block_pairs
    else:
        prepare, pair = squeeze_singleton_axis, _hnd_block_pairs
    paged = prepare(list(kv_layers))
    num_heads, block_size, head_size = paged[0].shape[2:]
    _, num_layers, slots, hidden = chunk.shape
    if hidden != num_heads * head_size or slots % block_size:
        raise ValueError(
            f"Staged chunk {tuple(chunk.shape)} does not match KV heads "
            f"{num_heads}x{head_size} with {block_size}-token pages"
        )
    # Same bytes as the chunk, viewed block-major: [B, L, 2, H, BS, D].
    staged = chunk.view(
        slots // block_size, num_layers, 2, num_heads, block_size, head_size
    )
    n_pages = len(pages)
    if int(direction) == int(lmcache_native.TransferDirection.D2H):
        kv_side, chunk_side = pair(paged, list(pages), staged[:n_pages])
        torch._foreach_copy_(chunk_side, kv_side)
        return
    start = min(skip_pages, n_pages)
    if start >= n_pages:
        return
    kv_side, chunk_side = pair(paged, list(pages)[start:], staged[start:n_pages])
    torch._foreach_copy_(kv_side, chunk_side)
