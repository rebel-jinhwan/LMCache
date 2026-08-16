# SPDX-License-Identifier: Apache-2.0
"""RBLN ops backend: block transfer tuned for the device's torch op ordering.

Every op except :meth:`RblnDeviceOps.multi_layer_block_kv_transfer` is
inherited from :class:`DeviceOps`, which routes to the pure torch
implementations in :mod:`lmcache.v1.platform.torch_ops`. That baseline is safe
on RBLN: ``lmcache_memcpy_async`` takes its tensor-mode branch for non-CUDA
devices, and the completion / event recorders degrade to immediate
publication, with ordering supplied by the transfer context's
``torch_dev.synchronize()``.

Block transfer is overridden for the op sequence, not for the layout. Chunks
keep LMCache's canonical token-major wire layout (``[2, L, T, H*D]``), so a
chunk written from an RBLN cache is byte-compatible with every other device --
the case that matters for cross-device KV sharing and PD disaggregation.

:mod:`lmcache.v1.platform.rbln.kv_ops` holds the torch kernels that do it, and
they stay the reference implementation and the fallback. When
``lmcache.rbln_ops`` is built, :meth:`RblnDeviceOps.ensure_native` binds it over
them exactly as CUDA and XPU bind theirs: the device<->host leg becomes rebel
runtime DMAs and the HND<->token-major transpose runs in C++ rather than as a
torch strided copy. The bytes it produces are the same. Operands it cannot
address fall back; see :func:`native_can_serve`.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar, cast

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
from lmcache.v1.platform.rbln.kv_layout import squeeze_singleton_axis
from lmcache.v1.platform.rbln.kv_ops import (
    gather_blocks_to_chunk,
    scatter_chunk_to_blocks,
)
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)

#: The only layout this path is validated for: the native vLLM-RBLN per-layer
#: HND format the vLLM detector reports for an RBLN KV cache.
_SUPPORTED_FORMAT = lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS


def native_can_serve(
    paged_layers: "list[torch.Tensor]",
    chunks: "list[torch.Tensor]",
    block_ids: "list[int]",
    blocks_per_chunk: int,
) -> bool:
    """Return whether the native kernel can address these operands.

    Each rejection is a case where the kernel would compute an address from
    geometry that does not hold, so declining keeps a mismatch a slow path
    rather than silent corruption:

    - It walks both sides by arithmetic on ``data_ptr()``, and its copies are
      ``rbln_memcpy_{v2h,h2v}_async``, so a paged layer must be a contiguous
      RBLN tensor and a chunk a contiguous host one.
    - It derives a chunk index from the flat block position, so a block list
      that does not fill every chunk would place the tail at another chunk's
      offsets. The torch path re-strides the short chunk instead.
    """
    if not paged_layers or not chunks or blocks_per_chunk <= 0:
        return False
    if len(block_ids) != len(chunks) * blocks_per_chunk:
        return False
    if not all(
        layer.device.type == "rbln" and layer.is_contiguous() for layer in paged_layers
    ):
        return False
    return all(chunk.device.type == "cpu" and chunk.is_contiguous() for chunk in chunks)


class RblnDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "rbln"

    def ensure_native(self) -> None:
        """Bind ``lmcache.rbln_ops`` over the torch baseline, if it was built.

        Soft-fail as on CUDA and XPU: the extension links the rebel runtime, so
        its absence is the ordinary case. It adds
        ``block_kv_transfer``, a name no torch method has, which is
        how :meth:`multi_layer_block_kv_transfer` knows which one it holds.
        """
        if self._native_bound:
            return
        self._native_bound = True  # set early to prevent repeated attempts
        try:
            # First Party
            import lmcache.rbln_ops as native
        except ImportError:
            logger.info(
                "lmcache.rbln_ops not built; RblnDeviceOps stays on the torch "
                "kernels in lmcache.v1.platform.rbln.kv_ops."
            )
            return
        self.bind_native(native)

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
            paged_buffer_ptrs_tensor: Native per-layer HND KV tensors,
                ``[2, NB, NH, 1, BS, HS]``.
            lmcache_objects_ptrs: Staging chunks, each in the canonical
                token-major layout ``[2, L, T, H*D]``.
            block_ids: Flat paged-block IDs in chunk-token order.
            device: Device the transfer runs on. Unused; taken from the
                tensors.
            direction: ``D2H`` to store, ``H2D`` to retrieve.
            shape_desc: Paged-buffer shape descriptor.
            lmcache_chunk_size: Tokens per staging chunk.
            engine_kv_format: Engine KV layout; must be the HND format.
            skip_prefix_n_blocks: Leading blocks neither read nor written.

        Raises:
            ValueError: If the operands are not tensor lists, the format is
                not the validated HND layout, a paged tensor is not in the
                native ``[2, NB, NH, 1, BS, HS]`` shape, or the direction is
                unknown.
        """
        del device  # taken from the operands
        if isinstance(paged_buffer_ptrs_tensor, torch.Tensor) or not all(
            isinstance(obj, torch.Tensor) for obj in lmcache_objects_ptrs
        ):
            raise ValueError(
                "RBLN block transfer requires tensor operands. The pointer "
                "form reconstructs tensors from a packed pointer tensor plus "
                "shape_desc, which lmcache.rbln_ops does not take -- it "
                "addresses the tensors it is handed."
            )
        if int(engine_kv_format) != int(_SUPPORTED_FORMAT):
            raise ValueError(
                "RBLN block transfer supports only "
                f"{_SUPPORTED_FORMAT.name}; got {engine_kv_format!r}"
            )

        # The format keeps the singleton axis the RBLN attention backend
        # requires; drop it here, where the bytes are actually addressed.
        paged_layers = squeeze_singleton_axis(
            cast("list[torch.Tensor]", list(paged_buffer_ptrs_tensor))
        )
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

        native = getattr(self, "block_kv_transfer", None)
        if native is not None and native_can_serve(
            paged_layers, chunks, flat_blocks, blocks_per_chunk
        ):
            native(
                paged_layers,
                chunks,
                flat_blocks,
                int(direction),
                skip_prefix_n_blocks,
            )
            return

        consumed = 0
        for chunk_idx, chunk in enumerate(chunks):
            blocks = flat_blocks[
                chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk
            ]
            if not blocks:
                break
            if is_d2h:
                gather_blocks_to_chunk(paged_layers, blocks, chunk)
            else:
                # The prefix skip is global across the transfer; translate it
                # into this chunk's local block offset.
                local_skip = min(len(blocks), max(0, skip_prefix_n_blocks - consumed))
                scatter_chunk_to_blocks(
                    paged_layers, blocks, chunk, skip_prefix_n_blocks=local_skip
                )
            consumed += len(blocks)
