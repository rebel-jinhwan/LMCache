# SPDX-License-Identifier: Apache-2.0
"""The native RBLN KV layouts and the views/validators used to address them.

vLLM-RBLN allocates each attention layer as
``[2, num_blocks, num_kv_heads, 1, block_size, head_size]`` -- HND with an
extra singleton axis between heads and block tokens that the RBLN attention
backend requires. Axis 3 is always 1, so the tensor is byte- and
stride-identical to a ``[2, NB, NH, BS, HS]`` layout, and squeezing it is a
free view.

Its MLA attention backend (``vllm_rbln/v1/attention/backends/mla``) instead
allocates ``[num_blocks, block_size, head_size]`` -- a single latent plane
with no K/V split and no head axis, which the vLLM detector classifies as
``EngineKVFormat.NL_X_NB_BS_HS``. There is nothing to squeeze there;
:func:`validate_mla_layers` only pins the rank so a layout drift fails loudly
at the transfer boundary instead of mis-addressing bytes.

Detection does not squeeze: the layout is registered as its own
``EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS``, so the vLLM detector classifies
what vLLM-RBLN actually allocated and holds no RBLN knowledge beyond that
shape signature. :func:`squeeze_singleton_axis` is applied one layer lower, by
:class:`~lmcache.v1.platform.rbln.device_ops.RblnDeviceOps`, where the paged
tensors are indexed to move bytes.
"""

# Future
from __future__ import annotations

# Standard
from enum import Enum
from typing import Sequence
import os

# Third Party
import torch

#: Rank of the native RBLN per-layer KV tensor.
RBLN_KV_NDIM = 6

#: Rank of the native RBLN per-layer MLA KV tensor (``[NB, BS, HS]``).
RBLN_MLA_KV_NDIM = 3

#: Axis of the native layout that is always 1 and is squeezed away.
RBLN_SINGLETON_AXIS = 3

#: Environment variable enabling head-major staging chunks (``1`` / ``0``).
HEAD_MAJOR_ENV_VAR = "LMCACHE_RBLN_SAVE_HEAD_MAJOR"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off", ""})


class RblnChunkLayout(Enum):
    """Byte order of an HND staging chunk. Same ``[2, L, T, H*D]`` buffer either
    way; a chunk must be read back under the layout it was written with."""

    TOKEN_MAJOR = "token_major"
    """Canonical ``[2, L, T, H*D]``; interchangeable with other devices."""

    HEAD_MAJOR = "head_major"
    """``[2, L, H, T, D]``; no host transpose, RBLN-only."""

    @classmethod
    def from_env(cls) -> "RblnChunkLayout":
        """Resolve the layout from ``LMCACHE_RBLN_SAVE_HEAD_MAJOR`` (1/0).

        Returns:
            RblnChunkLayout: ``HEAD_MAJOR`` when truthy, else ``TOKEN_MAJOR``.

        Raises:
            ValueError: If the value is neither truthy nor falsy.
        """
        raw = os.environ.get(HEAD_MAJOR_ENV_VAR, "").strip().lower()
        if raw in _TRUE_VALUES:
            return cls.HEAD_MAJOR
        if raw in _FALSE_VALUES:
            return cls.TOKEN_MAJOR
        raise ValueError(
            f"{HEAD_MAJOR_ENV_VAR}={raw!r} must be 1/0 (or true/false, yes/no, on/off)"
        )


def is_rbln_kv_layout(tensor: torch.Tensor) -> bool:
    """Return whether ``tensor`` is a native RBLN per-layer KV cache.

    Args:
        tensor: Candidate per-layer KV tensor.

    Returns:
        bool: ``True`` for a 6-D tensor whose leading axis is the K/V pair and
        whose axis 3 is a singleton.
    """
    return (
        tensor.ndim == RBLN_KV_NDIM
        and tensor.shape[0] == 2
        and tensor.shape[RBLN_SINGLETON_AXIS] == 1
    )


def squeeze_singleton_axis(
    kv_caches: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """Return 5-D HND views of native 6-D RBLN KV tensors.

    Strict by contract: the caller has already established that these tensors
    came from vLLM-RBLN (the detected ``EngineKVFormat`` says so), so anything
    else is a bug and fails loudly rather than being passed through.

    Args:
        kv_caches: Per-layer tensors shaped ``[2, NB, NH, 1, BS, HS]``.

    Returns:
        list[torch.Tensor]: Views shaped ``[2, NB, NH, BS, HS]``, sharing
        storage with the inputs.

    Raises:
        ValueError: If a tensor is not 6-D with a singleton at axis 3.
    """
    views: list[torch.Tensor] = []
    for tensor in kv_caches:
        if not is_rbln_kv_layout(tensor):
            raise ValueError(
                "RBLN KV caches must be [2, NB, NH, 1, BS, HS]; got "
                + str(tuple(tensor.shape))
            )
        views.append(tensor.squeeze(RBLN_SINGLETON_AXIS))
    return views


def validate_mla_layers(
    kv_caches: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """Return MLA per-layer KV tensors after pinning their rank.

    Mirrors :func:`squeeze_singleton_axis`'s strictness for the MLA layout:
    the caller has already established the detected format is
    ``NL_X_NB_BS_HS``, so any other rank is a layout drift and fails loudly.
    Unlike the HND path there is no axis to squeeze -- the tensors are
    returned unchanged.

    Args:
        kv_caches: Per-layer tensors shaped ``[NB, BS, HS]``.

    Returns:
        list[torch.Tensor]: The same tensors, as a list.

    Raises:
        ValueError: If a tensor is not 3-D.
    """
    for tensor in kv_caches:
        if tensor.ndim != RBLN_MLA_KV_NDIM:
            raise ValueError(
                "RBLN MLA KV caches must be [NB, BS, HS]; got "
                + str(tuple(tensor.shape))
            )
    return list(kv_caches)
