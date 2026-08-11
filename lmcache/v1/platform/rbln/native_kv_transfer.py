# SPDX-License-Identifier: Apache-2.0
"""Native RBLN block-transfer adapter.

``lmcache.rbln_ops`` is the compiled form of
:mod:`lmcache.v1.platform.rbln.kv_ops`: same head-major chunk contract, but the
copies are issued as rebel runtime DMAs instead of being expressed as torch
views.  The torch implementation has to build one view per (block, layer, kv)
to hand ``_foreach_copy_`` its operand lists, which is the dominant cost once a
chunk spans more than one block; the native path computes the same addresses
arithmetically and submits them directly.

The adapter is fail-open, like the MUSA one:
:func:`try_head_major_block_kv_transfer` returns ``False`` for anything it
cannot serve and the caller falls back to the torch kernels, which remain the
reference implementation and the only path unit tests exercise.

Unlike MUSA's ``musa_aiter``, this extension is not a separate optional
package -- ``RblnProfile`` builds it only when it found a rebel runtime to link
against, so its mere presence is the opt-in.
``LMCACHE_RBLN_NATIVE_KV_TRANSFER=0`` forces the torch path on a build that
does have it.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, Sequence
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.ops_types import TransferDirection

logger = init_logger(__name__)

ENV_RBLN_NATIVE_KV_TRANSFER = "LMCACHE_RBLN_NATIVE_KV_TRANSFER"

_DISABLED_VALUES = {"0", "false", "no", "off"}

#: Resolved once: ``False`` before the first attempt, then the module or
#: ``None``.  ``lmcache.rbln_ops`` is absent on every non-RBLN build, so the
#: import must not be retried per transfer.
_native_module: Any | None | bool = False


def is_native_kv_transfer_enabled() -> bool:
    """Return whether the native path may be used at all."""
    return (
        os.environ.get(ENV_RBLN_NATIVE_KV_TRANSFER, "").lower() not in _DISABLED_VALUES
    )


def load_native_module() -> Any | None:
    """Import ``lmcache.rbln_ops`` once, returning ``None`` when unavailable."""
    global _native_module
    if _native_module is False:
        try:
            # First Party
            import lmcache.rbln_ops as native

            _native_module = native
        except ImportError:
            logger.info(
                "lmcache.rbln_ops is not built; RBLN block transfer uses the "
                "torch kernels in lmcache.v1.platform.rbln.kv_ops."
            )
            _native_module = None
    return _native_module  # type: ignore[return-value]


def is_native_paged_tensor(tensor: torch.Tensor) -> bool:
    """Return whether a paged layer can be addressed by the native kernel.

    The kernel walks the buffer by arithmetic on ``data_ptr()``, so it needs a
    contiguous RBLN tensor -- a view with holes in it would address the wrong
    bytes rather than fail.
    """
    return tensor.device.type == "rbln" and tensor.is_contiguous()


def is_native_chunk_tensor(tensor: torch.Tensor) -> bool:
    """Return whether a chunk can be a native DMA host endpoint.

    The kernel issues ``rbln_memcpy_{v2h,h2v}_async``, whose non-device side is
    a host address, so a chunk that is itself on the NPU is not eligible.
    """
    return tensor.device.type == "cpu" and tensor.is_contiguous()


def try_head_major_block_kv_transfer(
    *,
    paged_layers: Sequence[torch.Tensor],
    chunks: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    blocks_per_chunk: int,
    direction: TransferDirection,
    skip_prefix_n_blocks: int,
) -> bool:
    """Try the native kernel for a whole block transfer.

    Args:
        paged_layers: Per-layer RBLN KV tensors, each contiguous and shaped
            ``[2, NB, NH, BS, HS]`` (singleton axis already squeezed).
        chunks: Staging chunks, read and written head-major.
        block_ids: Flat paged-block IDs in chunk-token order.
        blocks_per_chunk: Blocks each chunk holds.
        direction: ``D2H`` to store, ``H2D`` to retrieve.
        skip_prefix_n_blocks: Leading flat blocks neither read nor written.

    Returns:
        ``True`` when the native kernel ran and the caller should skip the
        torch fallback; ``False`` when the extension is missing, disabled, or
        the operands are outside what it can address.
    """
    if not is_native_kv_transfer_enabled():
        return False
    module = load_native_module()
    if module is None:
        return False
    if not paged_layers or not chunks or blocks_per_chunk <= 0:
        return False
    # The kernel derives a chunk index from the flat block position, so it can
    # only serve an exactly rectangular transfer. A ragged tail (fewer blocks
    # than the last chunk holds) goes back to the torch path.
    if len(block_ids) != len(chunks) * blocks_per_chunk:
        return False
    if not all(is_native_paged_tensor(layer) for layer in paged_layers):
        return False
    if not all(is_native_chunk_tensor(chunk) for chunk in chunks):
        return False

    module.head_major_block_kv_transfer(
        list(paged_layers),
        list(chunks),
        [int(block) for block in block_ids],
        int(direction),
        int(skip_prefix_n_blocks),
    )
    return True
