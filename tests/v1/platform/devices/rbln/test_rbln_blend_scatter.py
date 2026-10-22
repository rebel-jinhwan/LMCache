# SPDX-License-Identifier: Apache-2.0
"""Engine-driven CacheBlend scatter on a real RBLN NPU.

``scatter_cb_matches_to_paged_kv`` reaches the device only through
``RblnDeviceOps.multi_layer_block_kv_transfer``: it reads back the partly
covered destination blocks and writes the patched blocks back. These tests
run it over device-resident paged KV in both RBLN layouts -- the 6-D HND
cache and the MLA cache -- with a match that starts and ends mid-block and
is shifted, so the read-back, the host-side re-RoPE, and the write-back are
all exercised. ``torch.rbln.explain()`` fails the test if a copy is served
from the host copy instead of the NPU.

They skip unless ``torch.rbln`` reports a usable NPU.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.custom_types import CBMatchResult
from lmcache.v1.multiprocess.transfer_context.base import gather_paged_kv_to_cpu
from lmcache.v1.multiprocess.transfer_context.blend_scatter import (
    CBRopeSpec,
    scatter_cb_matches_to_paged_kv,
)
from lmcache.v1.platform.devices.rbln import RblnDeviceSpec
from lmcache.v1.platform.torch_ops.pos_kernels import rotary_embedding_k_fused

NUM_LAYERS = 2
NUM_BLOCKS = 8
NUM_HEADS = 2
BLOCK_SIZE = 4
HEAD_SIZE = 8
MLA_LATENT = 12
MLA_ROPE = 4
CHUNK_TOKENS = 8
MAX_POS = 64
DTYPE = torch.bfloat16

# A 6-token match from stored position 16 to query position 3: it starts in
# block 0 and ends in block 2, so blocks 0 and 2 are read back.
OLD_ST, CUR_ST, N_TOK = 16, 3, 6
BLOCK_IDS = [5, 2, 7, 0]

pytestmark = [
    pytest.mark.rbln,
    pytest.mark.skipif(
        not RblnDeviceSpec().is_available(), reason="RBLN NPU is required"
    ),
]


def _bound_device_tensor(source: torch.Tensor) -> torch.Tensor:
    """Copy ``source`` into a new ``rbln:0`` tensor with bound device memory."""
    tensor = torch.empty(source.shape, dtype=source.dtype, device="rbln:0")
    torch.rbln.bind_device_memory(tensor)
    tensor.copy_(source)
    return tensor


def _cos_sin_cache(rot_dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rot_dim, 2).float() / rot_dim))
    freqs = torch.outer(torch.arange(MAX_POS).float(), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def _rope_at(raw: torch.Tensor, start: int, cache: torch.Tensor) -> torch.Tensor:
    """RoPE ``raw`` (``[L, n, heads, d]``, fp32) at positions ``start..``."""
    rows = raw.transpose(0, 1).reshape(raw.shape[1], -1, raw.shape[-1]).clone()
    n = rows.shape[0]
    rotary_embedding_k_fused(
        torch.zeros(n, dtype=torch.long),
        torch.arange(start, start + n),
        rows,
        raw.shape[-1],
        cache,
        True,
    )
    return rows.reshape(n, raw.shape[0], *raw.shape[2:]).transpose(0, 1)


def _scatter_on_device(
    layer_shape: tuple[int, ...], chunk: torch.Tensor, rope: CBRopeSpec
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter one shifted match into device KV; return the token-major view
    of ``BLOCK_IDS`` before and after."""
    torch.manual_seed(7)
    kv = {
        f"layer_{i}": _bound_device_tensor(torch.randn(layer_shape, dtype=DTYPE))
        for i in range(NUM_LAYERS)
    }
    torch.rbln.synchronize()
    (before,) = gather_paged_kv_to_cpu(kv, BLOCK_IDS, len(BLOCK_IDS))
    torch.rbln.synchronize()
    match = CBMatchResult(
        old_st=OLD_ST,
        old_ed=OLD_ST + N_TOK,
        cur_st=CUR_ST,
        cur_ed=CUR_ST + N_TOK,
        hash=b"h",
    )
    with torch.rbln.explain() as region:
        scatter_cb_matches_to_paged_kv(kv, BLOCK_IDS, [chunk], [match], rope=rope)
    assert region.verdict()["clean"], region.report()
    (after,) = gather_paged_kv_to_cpu(kv, BLOCK_IDS, len(BLOCK_IDS))
    torch.rbln.synchronize()
    return before, after


def _outside_match(num_tokens: int) -> torch.Tensor:
    mask = torch.ones(num_tokens, dtype=torch.bool)
    mask[CUR_ST : CUR_ST + N_TOK] = False
    return mask


def test_hnd_scatter_on_device() -> None:
    """6-D HND cache: K is re-RoPE'd to the query position, V is copied, and
    every other token of the touched blocks is kept."""
    cache = _cos_sin_cache(HEAD_SIZE)
    raw_k = torch.randn(NUM_LAYERS, CHUNK_TOKENS, NUM_HEADS, HEAD_SIZE)
    stored_k = _rope_at(raw_k, OLD_ST, cache).reshape(NUM_LAYERS, CHUNK_TOKENS, -1)
    value = torch.randn(NUM_LAYERS, CHUNK_TOKENS, NUM_HEADS * HEAD_SIZE)
    chunk = torch.stack([stored_k, value]).to(DTYPE)

    before, after = _scatter_on_device(
        (2, NUM_BLOCKS, NUM_HEADS, 1, BLOCK_SIZE, HEAD_SIZE),
        chunk,
        CBRopeSpec(cache, head_size=HEAD_SIZE, is_neox_style=True),
    )

    keep = _outside_match(after.shape[2])
    assert torch.equal(after[:, :, keep], before[:, :, keep])
    rows = slice(CUR_ST, CUR_ST + N_TOK)
    assert torch.equal(after[1, :, rows], chunk[1, :, :N_TOK])
    expected_k = _rope_at(raw_k, CUR_ST, cache).reshape(NUM_LAYERS, CHUNK_TOKENS, -1)
    torch.testing.assert_close(
        after[0, :, rows].float(), expected_k[:, :N_TOK], atol=3e-2, rtol=3e-2
    )


def test_mla_scatter_on_device() -> None:
    """MLA cache: only the trailing rope window is re-RoPE'd; the latent
    content is copied, and every other token of the touched blocks is kept."""
    cache = _cos_sin_cache(MLA_ROPE)
    content = torch.randn(NUM_LAYERS, CHUNK_TOKENS, MLA_LATENT)
    raw_rope = torch.randn(NUM_LAYERS, CHUNK_TOKENS, 1, MLA_ROPE)
    stored_rope = _rope_at(raw_rope, OLD_ST, cache).reshape(
        NUM_LAYERS, CHUNK_TOKENS, -1
    )
    chunk = torch.cat([content, stored_rope], dim=-1).to(DTYPE)

    before, after = _scatter_on_device(
        (NUM_BLOCKS, BLOCK_SIZE, MLA_LATENT + MLA_ROPE),
        chunk,
        CBRopeSpec(
            cache, head_size=MLA_ROPE, is_neox_style=True, rot_offset=MLA_LATENT
        ),
    )

    keep = _outside_match(after.shape[1])
    assert torch.equal(after[:, keep], before[:, keep])
    rows = after[:, CUR_ST : CUR_ST + N_TOK]
    assert torch.equal(rows[..., :MLA_LATENT], chunk[:, :N_TOK, :MLA_LATENT])
    expected = _rope_at(raw_rope, CUR_ST, cache).reshape(NUM_LAYERS, CHUNK_TOKENS, -1)
    torch.testing.assert_close(
        rows[..., MLA_LATENT:].float(), expected[:, :N_TOK], atol=3e-2, rtol=3e-2
    )
