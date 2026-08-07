# SPDX-License-Identifier: Apache-2.0
"""Degenerate-axis removal for raw engine KV caches.

Some attention backends allocate a per-layer KV tensor with an extra axis of
length 1 between the head and block-token axes -- vLLM-RBLN does, and the
shape is otherwise an ordinary split-K/V paged cache. A size-1 axis carries no
bytes and no stride the rest of LMCache needs, so squeezing it is a free view
that leaves a 5-D per-layer cache the ordinary rank-5 classification already
understands.

Metadata-only (zero-copy) preprocessing, in the same family as
:mod:`~lmcache.v1.gpu_connector.kv_format.contiguity`: engine- and
format-agnostic, and it never decides *which* format the result is. Squeezing
leaves the head/block-token order untouched, so HND vs NHD is still resolved
downstream from ``layout_hints`` exactly as it is for a natively 5-D cache.
"""

# Future
from __future__ import annotations

# Standard
from typing import Sequence

# Third Party
import torch

#: Rank of a per-layer KV tensor that still carries the degenerate axis.
_SINGLETON_KV_NDIM = 6

#: Axis that is always 1 and is squeezed away.
SINGLETON_KV_AXIS = 3


def has_singleton_kv_axis(tensor: torch.Tensor) -> bool:
    """Return whether ``tensor`` is a per-layer KV cache with a degenerate axis.

    Args:
        tensor: Candidate per-layer KV tensor.

    Returns:
        bool: ``True`` for a 6-D tensor whose leading axis is the K/V pair and
        whose axis 3 is a singleton.
    """
    return (
        tensor.ndim == _SINGLETON_KV_NDIM
        and tensor.shape[0] == 2
        and tensor.shape[SINGLETON_KV_AXIS] == 1
    )


def squeeze_singleton_kv_axis(
    kv_caches: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """Return 5-D views of per-layer KV tensors carrying a degenerate axis.

    Args:
        kv_caches: Per-layer tensors shaped ``[2, NB, X, 1, Y, HS]``, where
            ``X`` / ``Y`` are the head and block-token axes in either order.

    Returns:
        list[torch.Tensor]: Views shaped ``[2, NB, X, Y, HS]``, sharing storage
        with the inputs.

    Raises:
        ValueError: If a tensor is not 6-D with a singleton at axis 3.
    """
    views: list[torch.Tensor] = []
    for tensor in kv_caches:
        if not has_singleton_kv_axis(tensor):
            raise ValueError(
                "a squeezable per-layer KV cache must be [2, NB, X, 1, Y, HS]; "
                "got " + str(tuple(tensor.shape))
            )
        views.append(tensor.squeeze(SINGLETON_KV_AXIS))
    return views
