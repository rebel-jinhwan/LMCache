# SPDX-License-Identifier: Apache-2.0
"""Normalization for the native RBLN KV layout.

vLLM-RBLN allocates each layer as
``[2, num_blocks, num_kv_heads, 1, block_size, head_size]`` -- HND with an
extra singleton axis between heads and block tokens that the RBLN attention
backend requires. Axis 3 is always 1, so the tensor is byte- and
stride-identical to the registered ``NL_X_TWO_NB_NH_BS_HS`` layout one axis
short, and squeezing it is a free view.

Both entry points that see raw engine tensors normalize through here:

- :class:`~lmcache.v1.gpu_connector.rbln_connector.VLLMPagedMemRBLNConnectorV2`
  for the in-process path, which also needs the 5-D views for its own slot
  indexing, and so calls :func:`squeeze_singleton_axis` directly.
- the vLLM format detector, which covers the multiprocess path -- its
  ``register`` / gather / scatter helpers all resolve layouts through
  ``normalize_kv_and_discover_format`` rather than through a connector. The
  detector holds no RBLN knowledge of its own: it calls
  :meth:`~lmcache.v1.platform.base.device_spec.DeviceSpec.normalize_kv_caches`,
  which :class:`~lmcache.v1.platform.rbln.RblnDeviceSpec` routes to
  :func:`normalize_kv_caches` below.

Keeping the rule in one place means the two paths cannot drift.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Sequence

# Third Party
import torch

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache

#: Rank of the native RBLN per-layer KV tensor.
RBLN_KV_NDIM = 6

#: Axis of the native layout that is always 1 and is squeezed away.
RBLN_SINGLETON_AXIS = 3


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

    Strict: every tensor must be in the native layout. Callers that hold KV
    caches they know came from vLLM-RBLN use this and get a loud failure on
    anything else; :func:`normalize_kv_caches` is the tolerant variant.

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


def normalize_kv_caches(kv_caches: "DiscoverableKVCache") -> "DiscoverableKVCache":
    """Squeeze the native layout if that is what ``kv_caches`` holds.

    Tolerant by contract: format discovery hands every KV structure it sees to
    the device spec, including the 5-D views the in-process connector has
    already squeezed, so anything that is not a flat list of native per-layer
    tensors passes through untouched for the ordinary branches to classify.

    Args:
        kv_caches: Raw KV cache structure as the engine handed it over.

    Returns:
        DiscoverableKVCache: 5-D per-layer views when the input was the native
        6-D layout, otherwise ``kv_caches`` unchanged.
    """
    if (
        isinstance(kv_caches, list)
        and kv_caches
        and all(
            isinstance(tensor, torch.Tensor) and is_rbln_kv_layout(tensor)
            for tensor in kv_caches
        )
    ):
        return squeeze_singleton_axis(kv_caches)
    return kv_caches
