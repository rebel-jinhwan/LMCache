# SPDX-License-Identifier: Apache-2.0
"""RBLN cache context for LMCache-driven multiprocess transfer.

The server opens the worker's KV tensors from :class:`RblnIPCWrapper`
descriptors and moves blocks with the same tensor-level gather/scatter the
engine-driven path runs in the worker (:class:`RblnDeviceOps`). Chunks are
written straight into, and read straight from, the host memory objects, so
there is no device staging buffer.

Everything not tied to the device -- layer-group discovery, the CPU block-ID
buffer, shape bookkeeping -- comes from :class:`CPUCacheContext`.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any

# Third Party
import torch

# First Party
from lmcache import device_ops
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.platform.devices.cpu.cache_context import CPUCacheContext
import lmcache.lmcache_native as lmcache_native

if TYPE_CHECKING:
    # Standard
    from collections.abc import Sequence

    # First Party
    from lmcache.v1.memory_management import MemoryObj
    from lmcache.v1.multiprocess.custom_types import KVCache
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

logger = init_logger(__name__)


class RblnCacheContext(CPUCacheContext):
    """Server-side view of a worker's RBLN KV cache."""

    device_type = "rbln"

    def __init__(
        self,
        kv_caches: KVCache,
        lmcache_tokens_per_chunk: int = 256,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: "Sequence[EngineGroupInfo]" = (),
        engine_type: EngineType = EngineType.VLLM,
        separate_object_groups: bool = False,
        full_sw_kv: bool = False,
    ) -> None:
        super().__init__(
            kv_caches,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
            layout_hints=layout_hints,
            engine_group_infos=engine_group_infos,
            engine_type=engine_type,
            separate_object_groups=separate_object_groups,
            full_sw_kv=full_sw_kv,
        )
        self._wrappers = kv_caches
        self.device_ = self.kv_caches_[0].device
        self._rbln_stream = torch.rbln.current_stream(self.device_)  # type: ignore[attr-defined]
        attn_desc = self.kv_layer_groups_manager_.get_attn_desc()
        if not all(
            attn_desc.is_full_attention(idx)
            for idx in range(self.kv_layer_groups_manager_.num_object_groups)
        ):
            raise ValueError(
                "RBLN LMCache-driven transfer supports full-attention KV only"
            )

    @property
    def stream(self) -> Any:
        """Returns the ``torch.rbln`` stream the server enqueues on."""
        return self._rbln_stream

    def close(self) -> None:
        """Release the imported KV mappings."""
        self.kv_caches_ = []
        for wrapper in self._wrappers:
            wrapper.close()

    def transfer_object_group(
        self,
        block_ids_gpu: list[torch.Tensor],
        memory_objs: Sequence[MemoryObj | None],
        *,
        object_group_id: int,
        batch_size: int,
        skip_first_n_tokens: int,
        direction: lmcache_native.TransferDirection,
        transfer_key: str,
        block_ids_host: Sequence[Sequence[int]] = (),
    ) -> None:
        """Copy whole blocks between the KV cache and the memory objects.

        Each chunk is gathered into (or scattered from) its memory object's
        region for the kernel group, in LMCache's token-major layout.

        Args:
            block_ids_gpu: Staged block IDs per kernel group (host tensors).
            memory_objs: One memory object per chunk; ``None`` entries are
                skipped on store.
            object_group_id: Object group being copied.
            batch_size: Unused; chunks are copied one at a time.
            skip_first_n_tokens: Leading tokens not written on retrieve.
            direction: ``D2H`` to store, ``H2D`` to retrieve.
            transfer_key: Unused; no phase timing on this path.
            block_ids_host: Unused; ``block_ids_gpu`` already lives on host.

        Raises:
            ValueError: If a retrieve has a missing memory object.
        """
        del batch_size, transfer_key, block_ids_host
        is_h2d = direction == lmcache_native.TransferDirection.H2D
        chunk_tokens = self.lmcache_tokens_per_chunk
        manager = self.kv_layer_groups_manager_
        object_group = manager.object_groups[object_group_id]
        for chunk_idx, memory_obj in enumerate(memory_objs):
            if memory_obj is None:
                if is_h2d:
                    raise ValueError(
                        f"MemoryObj for chunk {chunk_idx} is None; cannot retrieve"
                    )
                continue
            chunk_start = chunk_idx * chunk_tokens
            if is_h2d and chunk_start + chunk_tokens <= skip_first_n_tokens:
                continue
            raw = memory_obj.raw_tensor
            if raw is None:
                raise ValueError(f"MemoryObj for chunk {chunk_idx} has no host tensor")
            for kernel_group_idx in object_group.kernel_group_indices:
                group = manager.kernel_groups[kernel_group_idx]
                blocks_per_chunk = self.calculate_num_blocks(
                    chunk_tokens, kernel_group_idx
                )
                block_ids = block_ids_gpu[kernel_group_idx][
                    chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk
                ]
                shape, dtype = self.get_kernel_group_shape_dtype(
                    chunk_tokens, kernel_group_idx
                )
                offset = self.get_kernel_group_offset_in_object(
                    object_group_id, kernel_group_idx
                )
                nbytes = shape.numel() * dtype.itemsize
                chunk = raw.view(torch.uint8)[offset : offset + nbytes]
                skip_blocks = max(0, skip_first_n_tokens - chunk_start) // max(
                    1, chunk_tokens // blocks_per_chunk
                )
                device_ops.multi_layer_block_kv_transfer(
                    [self.kv_caches_[i] for i in group.layer_indices],
                    [chunk.view(dtype).view(shape)],
                    block_ids,
                    self.device_,
                    direction,
                    self.get_shape_desc(kernel_group_idx),
                    chunk_tokens,
                    self.get_engine_kv_format(kernel_group_idx),
                    skip_blocks if is_h2d else 0,
                )

    def _check_shm_capacity(self) -> None:
        """The KV cache lives in device memory, not ``/dev/shm``."""
