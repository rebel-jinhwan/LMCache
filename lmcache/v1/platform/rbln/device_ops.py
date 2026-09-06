# SPDX-License-Identifier: Apache-2.0
"""RBLN ops backend: block transfer tuned for the device's copy engine.

Every op except :meth:`RblnDeviceOps.multi_layer_block_kv_transfer` is
inherited from :class:`DeviceOps`, which routes to the pure torch
implementations in :mod:`lmcache.v1.platform.torch_ops`. That baseline is safe
on RBLN: ``lmcache_memcpy_async`` takes its tensor-mode branch for non-CUDA
devices, and the completion / event recorders degrade to immediate
publication, with ordering supplied by the transfer context's
``torch_dev.synchronize()``.

Block transfer is overridden for the op sequence, not for the layout. Chunks
keep LMCache's canonical token-major wire layout (``[2, L, T, H*D]`` for HND,
``[L, T, HS]`` for MLA), so a chunk written from an RBLN cache is
byte-compatible with every other device -- the case that matters for
cross-device KV sharing and PD disaggregation.

Both layouts are moved by the compiled ``lmcache.rbln_ops`` extension
(``csrc/rbln``, built with ``BUILD_WITH_RBLN=1``); there is no torch fallback,
without the extension the transfer raises. What the extension replaces is the
shared path's strided device indexing:

- **HND** (``NL_X_TWO_NB_NH_ONE_BS_HS``): RBLN stores heads before block
  tokens, so the extension gathers whole blocks device-to-device, swaps the
  head and token axes with one compiled device program, and copies each
  chunk's bytes straight into the chunk.
- **MLA** (``NL_X_NB_BS_HS``, one latent plane, no head axis): no transpose
  to hoist; what costs on RBLN is the number of device<->host copies, so the
  extension batches a chunk's blocks into a persistent device staging buffer
  with whole-block D2D copies and crosses the boundary once per chunk. The
  shared torch path instead issues one ``index_select`` / ``index_copy_`` per
  layer, each a separate submission with an index read-back and a CPU
  fallback behind it.

Both layouts require the engine's KV caches to be real device tensors
(vLLM-RBLN: ``VLLM_RBLN_USE_DEVICE_TENSOR=1``). With the default compile-mode
allocation the per-layer tensors are ``meta`` and any transfer -- this
module's or the shared path's -- dies at the first host copy with "Cannot
copy out of meta tensor".
"""

# Future
from __future__ import annotations

# Standard
from types import ModuleType
from typing import ClassVar, cast

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
from lmcache.v1.platform.rbln.kv_layout import squeeze_singleton_axis
import lmcache.lmcache_native as lmcache_native

try:
    # First Party
    from lmcache import rbln_ops
except ImportError:  # built only with torch-rbln present (BUILD_WITH_RBLN=1)
    rbln_ops = None  # type: ignore[assignment]

logger = init_logger(__name__)

#: The native vLLM-RBLN per-layer HND attention format the vLLM detector
#: reports (``[2, NB, NH, 1, BS, HS]``).
_HND_FORMAT = lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS

#: The MLA layout vLLM-RBLN's MLA attention backend allocates (``[NB, BS, HS]``).
_MLA_FORMAT = lmcache_native.EngineKVFormat.NL_X_NB_BS_HS


def _require_rbln_ops() -> ModuleType:
    """Return ``lmcache.rbln_ops``, failing loudly when it was not built.

    Returns:
        ModuleType: The compiled extension.

    Raises:
        RuntimeError: If ``lmcache.rbln_ops`` was not built.
    """
    if rbln_ops is None:
        raise RuntimeError(
            "lmcache.rbln_ops is not built; install LMCache with torch-rbln "
            "present (BUILD_WITH_RBLN=1)"
        )
    return rbln_ops


class RblnDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "rbln"

    def multi_layer_block_kv_transfer(
        self,
        paged_buffer_ptrs_tensor: "torch.Tensor | list[torch.Tensor]",
        lmcache_objects_ptrs: "list[int] | list[torch.Tensor]",
        block_ids: "torch.Tensor | list[int]",
        device: "torch.device | str",
        direction: lmcache_native.TransferDirection,
        shape_desc: PageBufferShapeDesc,
        lmcache_chunk_size: int,
        engine_kv_format: lmcache_native.EngineKVFormat,
        skip_prefix_n_blocks: int,
    ) -> None:
        """Move whole paged blocks between RBLN KV and token-major chunks.

        Args:
            paged_buffer_ptrs_tensor: Native per-layer KV tensors --
                ``[2, NB, NH, 1, BS, HS]`` (HND attention) or contiguous
                ``[NB, BS, HS]`` (MLA).
            lmcache_objects_ptrs: Staging chunks in the canonical token-major
                layout -- ``[2, L, T, H*D]`` (HND) or ``[L, T, HS]`` (MLA).
            block_ids: Flat paged-block IDs in chunk-token order.
            device: Device the transfer runs on. Unused; taken from the
                tensors.
            direction: ``D2H`` to store, ``H2D`` to retrieve.
            shape_desc: Paged-buffer shape descriptor.
            lmcache_chunk_size: Tokens per staging chunk.
            engine_kv_format: Engine KV layout; must be the HND or MLA format.
            skip_prefix_n_blocks: Leading blocks neither read nor written.

        Raises:
            ValueError: If the operands are not tensor lists, the format is
                neither the validated HND nor the MLA layout, an HND paged
                tensor is not in the native ``[2, NB, NH, 1, BS, HS]`` shape,
                or the direction is unknown.
            RuntimeError: If ``lmcache.rbln_ops`` was not built, or an MLA
                paged tensor is not a contiguous ``[NB, BS, HS]``.
        """
        del device  # taken from the operands
        if isinstance(paged_buffer_ptrs_tensor, torch.Tensor) or not all(
            isinstance(obj, torch.Tensor) for obj in lmcache_objects_ptrs
        ):
            raise ValueError(
                "RBLN block transfer requires tensor operands; the pointer "
                "form is only produced for backends bound through "
                "bind_native, and lmcache.rbln_ops takes tensors."
            )
        is_mla = lmcache_native.is_mla(engine_kv_format)
        if is_mla:
            # is_mla() admits every MLA layout; only NL_X_NB_BS_HS has an RBLN
            # op sequence, so reject the others with a format error rather
            # than a shape mismatch inside the extension.
            if int(engine_kv_format) != int(_MLA_FORMAT):
                raise ValueError(
                    "RBLN block transfer supports only the "
                    f"{_MLA_FORMAT.name} MLA layout; got {engine_kv_format!r}"
                )
        elif int(engine_kv_format) != int(_HND_FORMAT):
            raise ValueError(
                "RBLN block transfer supports only "
                f"{_HND_FORMAT.name} and {_MLA_FORMAT.name}; "
                f"got {engine_kv_format!r}"
            )

        paged_layers = cast("list[torch.Tensor]", list(paged_buffer_ptrs_tensor))
        if not is_mla:
            # The HND format keeps the singleton axis the RBLN attention
            # backend requires; drop it here, where the bytes are actually
            # addressed. MLA has nothing to squeeze; the extension pins its
            # rank and contiguity.
            paged_layers = squeeze_singleton_axis(paged_layers)
        chunks = cast("list[torch.Tensor]", list(lmcache_objects_ptrs))
        flat_blocks = (
            [int(b) for b in block_ids.tolist()]
            if isinstance(block_ids, torch.Tensor)
            else [int(b) for b in block_ids]
        )

        block_size = int(shape_desc.bs)
        if block_size <= 0 or lmcache_chunk_size % block_size != 0:
            raise ValueError(
                "lmcache_chunk_size must be a positive multiple of shape_desc.bs"
            )
        blocks_per_chunk = lmcache_chunk_size // block_size

        is_d2h = int(direction) == int(lmcache_native.TransferDirection.D2H)
        if not is_d2h and int(direction) != int(lmcache_native.TransferDirection.H2D):
            raise ValueError(f"Unsupported transfer direction: {direction!r}")

        native = _require_rbln_ops()
        if is_d2h:
            gather = (
                native.gather_blocks_to_chunks_mla
                if is_mla
                else native.gather_blocks_to_chunks_hnd
            )
            gather(paged_layers, flat_blocks, chunks, blocks_per_chunk)
        else:
            scatter = (
                native.scatter_chunks_to_blocks_mla
                if is_mla
                else native.scatter_chunks_to_blocks_hnd
            )
            scatter(
                paged_layers,
                flat_blocks,
                chunks,
                blocks_per_chunk,
                skip_prefix_n_blocks,
            )
