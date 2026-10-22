# SPDX-License-Identifier: Apache-2.0
"""Worker-side CacheBlend scatter for the engine-driven transfer path.

The LMCache-driven blend retrieve writes matched chunks into the paged KV
from the server process with one CUDA plan. On the engine-driven path the
server cannot reach the device, so the worker does the same work with the
block transfer op every engine-driven device already implements
(``DeviceOps.multi_layer_block_kv_transfer``):

1. Gather the destination blocks a match covers only partly (D2H), so the
   tokens outside every match keep their content.
2. On the host, K-only re-RoPE each shifted match from its stored position
   ``old_st`` to its new position ``cur_st``, and lay its token rows into a
   staging chunk over the destination blocks.
3. Scatter the staging chunk back as whole blocks (H2D).

Chunk row ``i`` holds stored position ``old_st + i`` and lands at query
position ``cur_st + i`` -- the same contract as the CUDA retrieve plan.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass

# Third Party
import numpy as np
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format import get_spec_class
from lmcache.v1.gpu_connector.utils import (
    LayoutHints,
    get_block_size,
    get_device,
    normalize_kv_and_discover_format,
)
from lmcache.v1.multiprocess.custom_types import CBMatchResult
from lmcache.v1.multiprocess.transfer_context.base import (
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from lmcache.v1.platform import synchronize_device
from lmcache.v1.platform.torch_ops.pos_kernels import (
    rotary_embedding_k_fused,
    rotary_embedding_k_fused_strided,
)
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)


def normalize_cos_sin_cache(cos_sin_cache: torch.Tensor) -> torch.Tensor:
    """Strip a baked-in rotary magnitude from a ``[max_pos, rot_dim]`` cache.

    YaRN/longrope bake an mscale ``m`` into the cache (cos² + sin² = m² ≠ 1).
    CB re-RoPE assumes a pure rotation, so an un-normalized ``m`` injects an
    ``m²`` error per K element.

    Args:
        cos_sin_cache: Cos half then sin half per position.

    Returns:
        ``cos_sin_cache`` itself when its magnitude is 1, else a rescaled copy
        in the same dtype.
    """
    cache32 = cos_sin_cache.to(torch.float32)
    half = cache32.shape[1] // 2
    magnitude = float((cache32[:, :half] ** 2 + cache32[:, half:] ** 2).mean().sqrt())
    if abs(magnitude - 1.0) < 1e-3:
        return cos_sin_cache
    logger.info(
        "CB re-RoPE: stripping rope-cache mscale=%.4f (m²=%.4f → K inflation "
        "if uncorrected) → unit magnitude",
        magnitude,
        magnitude * magnitude,
    )
    return (cache32 / magnitude).to(cos_sin_cache.dtype)


@dataclass(frozen=True)
class CBRopeSpec:
    """K re-RoPE geometry for the engine-driven blend scatter.

    Attributes:
        cos_sin_cache: ``[max_position, rot_dim]`` cos half then sin half,
            already normalized by :func:`normalize_cos_sin_cache`. Any device;
            the scatter copies only the rows of the positions it rotates to
            the host.
        head_size: K head size.
        is_neox_style: NeoX (contiguous halves) vs GPT-J (interleaved).
        rot_offset: MLA only: element offset of the rope window that trails
            the latent row. ``0`` rotates the leading ``rot_dim`` of each head.
    """

    cos_sin_cache: torch.Tensor
    head_size: int
    is_neox_style: bool
    rot_offset: int = 0


def _rerope_k_rows(
    k_rows: torch.Tensor,
    old_st: int,
    cur_st: int,
    rope: CBRopeSpec,
    fused_packed: bool,
) -> None:
    """Re-RoPE ``k_rows`` (``[num_layers, n, hidden]``) in place, in fp32.

    Args:
        k_rows: K plane rows of one match; row ``i`` is position ``old_st + i``.
        old_st: Stored position of row 0.
        cur_st: New position of row 0.
        rope: Rope geometry.
        fused_packed: K and V share each head as ``[K | V]`` (``2 * head_size``).

    Raises:
        ValueError: If the row width is not a whole number of heads, or the
            MLA window does not end the row.
    """
    num_tokens = k_rows.shape[1]
    hidden = k_rows.shape[2]
    cache = rope.cos_sin_cache
    rot_dim = int(cache.shape[1])
    # Copy only this match's cos/sin rows to the host: old positions as rows
    # [0, n), new positions as rows [n, 2n).
    positions = torch.cat(
        [
            torch.arange(old_st, old_st + num_tokens),
            torch.arange(cur_st, cur_st + num_tokens),
        ]
    )
    cos_sin32 = cache[positions.to(cache.device)].to(device="cpu", dtype=torch.float32)
    old_pos = torch.arange(num_tokens)
    new_pos = torch.arange(num_tokens, 2 * num_tokens)
    # Tokens first: the torch rope op broadcasts cos/sin as [n, 1, rot_dim/2].
    # A fresh contiguous buffer, so the per-head reshape below is a view the
    # op writes into.
    rows = torch.empty((num_tokens, k_rows.shape[0], hidden), dtype=torch.float32)
    rows.copy_(k_rows.transpose(0, 1))
    if rope.rot_offset > 0:
        if rope.rot_offset + rot_dim != hidden:
            raise ValueError(
                f"CB re-RoPE: MLA rope window ({rope.rot_offset}, {rot_dim}) "
                f"does not end the {hidden}-wide latent row."
            )
        rotary_embedding_k_fused(
            old_pos,
            new_pos,
            rows[..., rope.rot_offset :],
            rot_dim,
            cos_sin32,
            rope.is_neox_style,
        )
    else:
        per_head = rope.head_size * (2 if fused_packed else 1)
        if hidden % per_head != 0:
            raise ValueError(
                f"CB re-RoPE: row width {hidden} is not a multiple of the "
                f"per-head width {per_head} (fused_packed={fused_packed})."
            )
        rotary_embedding_k_fused_strided(
            old_pos,
            new_pos,
            rows.reshape(num_tokens, -1, per_head),
            rope.head_size,
            per_head,
            cos_sin32,
            rope.is_neox_style,
        )
    k_rows.copy_(rows.transpose(0, 1))


def scatter_cb_matches_to_paged_kv(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[int],
    chunks: list[torch.Tensor],
    matches: list[CBMatchResult],
    rope: CBRopeSpec | None,
    layout_hints: LayoutHints | None = None,
    engine_kv_format: "lmcache_native.EngineKVFormat" | None = None,
    engine_type: EngineType = EngineType.VLLM,
) -> None:
    """Write blend matches into the paged KV at token granularity.

    Every token outside the matches keeps its content, so matches need not
    be block-aligned and may share a block with recomputed tokens. Returns
    after the device writes complete.

    Args:
        kv_caches: Per-layer KV tensors of the single KV group.
        block_ids: The request's block table (logical block -> paged block).
        chunks: One canonical chunk per match (``[2, L, T, H*D]`` for split
            K/V, ``[L, T, HS]`` for single-plane formats).
        matches: Matches to write, ``cur_st`` order, no overlap; match ``i``
            reads rows ``[0, cur_ed - cur_st)`` of ``chunks[i]``.
        rope: Re-RoPE geometry, or ``None`` for a NoPE model (no rotation).
        layout_hints: Optional engine layout hints.
        engine_kv_format: Optional pre-detected KV format.
        engine_type: Serving engine that produced the KV tensors.

    Raises:
        ValueError: If the chunk and match counts differ, a match exceeds the
            block table or its chunk, or the K plane cannot be re-RoPE'd
            (quantized K, or a geometry mismatch).
    """
    if len(chunks) != len(matches):
        raise ValueError(f"{len(chunks)} chunk(s) for {len(matches)} match(es)")
    if not matches:
        return
    fmt, normalized = normalize_kv_and_discover_format(
        list(kv_caches.values()), engine_type, layout_hints=layout_hints
    )
    if engine_kv_format is None:
        engine_kv_format = fmt
    block_size = get_block_size(normalized, engine_kv_format)
    device = get_device(normalized)

    span_end = max(r.cur_ed for r in matches)
    if span_end > len(block_ids) * block_size:
        raise ValueError(
            f"match end {span_end} exceeds the {len(block_ids)}-block table "
            f"(block_size={block_size})"
        )
    covered = np.zeros(-(-span_end // block_size) * block_size, dtype=bool)
    for r, chunk in zip(matches, chunks, strict=True):
        if r.cur_ed - r.cur_st > chunk.shape[-2]:
            raise ValueError(
                f"match [{r.cur_st}, {r.cur_ed}) is longer than its "
                f"{chunk.shape[-2]}-token chunk"
            )
        covered[r.cur_st : r.cur_ed] = True
    dest = sorted(
        {
            b
            for r in matches
            for b in range(r.cur_st // block_size, (r.cur_ed - 1) // block_size + 1)
        }
    )
    row_of_block = {b: i * block_size for i, b in enumerate(dest)}
    partial = [
        b for b in dest if not covered[b * block_size : (b + 1) * block_size].all()
    ]

    staging_shape = list(chunks[0].shape)
    staging_shape[-2] = len(dest) * block_size
    staging = torch.empty(staging_shape, dtype=chunks[0].dtype)
    if partial:
        # A partial block's other tokens may still be in flight from the
        # engine's previous step; read them only once they have landed.
        synchronize_device(device)
        (gathered,) = gather_paged_kv_to_cpu(
            kv_caches,
            [block_ids[b] for b in partial],
            len(partial),
            layout_hints=layout_hints,
            engine_kv_format=engine_kv_format,
            engine_type=engine_type,
        )
        synchronize_device(device)
        for i, b in enumerate(partial):
            row = row_of_block[b]
            staging[..., row : row + block_size, :] = gathered[
                ..., i * block_size : (i + 1) * block_size, :
            ]

    match_rows: list[torch.Tensor] = []
    for r, chunk in zip(matches, chunks, strict=True):
        n = r.cur_ed - r.cur_st
        row = row_of_block[r.cur_st // block_size] + r.cur_st % block_size
        rows = staging[..., row : row + n, :]
        rows.copy_(chunk[..., :n, :])
        match_rows.append(rows)

    shifted = [
        (r, rows)
        for r, rows in zip(matches, match_rows, strict=True)
        if r.old_st != r.cur_st
    ]
    if rope is not None and shifted:
        if not staging.dtype.is_floating_point:
            raise ValueError(f"CB re-RoPE: cannot rotate a {staging.dtype} K plane")
        split_kv = staging.dim() == 4
        fused_packed = get_spec_class(engine_kv_format).is_fused_packed
        for r, rows in shifted:
            _rerope_k_rows(
                rows[0] if split_kv else rows, r.old_st, r.cur_st, rope, fused_packed
            )

    scatter_cpu_to_paged_kv(
        kv_caches,
        [block_ids[b] for b in dest],
        [staging],
        len(dest),
        layout_hints=layout_hints,
        engine_kv_format=engine_kv_format,
        engine_type=engine_type,
    )
    # The staging chunk is a local: the H2D copies must finish before it is
    # freed, and the caller releases the server's read locks next.
    synchronize_device(device)
