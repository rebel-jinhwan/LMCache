# SPDX-License-Identifier: Apache-2.0
"""Type stubs for the ``lmcache.rbln_ops`` extension (``csrc/rbln``)."""

# Third Party
import torch

def gather_blocks_to_chunks_mla(
    paged_layers: list[torch.Tensor],
    block_ids: list[int],
    chunks: list[torch.Tensor],
    blocks_per_chunk: int,
) -> None:
    """Gather whole MLA paged blocks ``[NB, BS, HS]`` into chunks ``[L, T, HS]``."""

def scatter_chunks_to_blocks_mla(
    paged_layers: list[torch.Tensor],
    block_ids: list[int],
    chunks: list[torch.Tensor],
    blocks_per_chunk: int,
    skip_prefix_n_blocks: int = 0,
) -> None:
    """Scatter chunks ``[L, T, HS]`` back into whole MLA paged blocks."""
