# SPDX-License-Identifier: Apache-2.0
"""Check real RBLN DRAM transfers against the canonical CPU chunk format.

Run with TORCH_RBLN_EAGER_MALLOC=1 so small tensors use device DRAM instead
of host SHM. The Buildkite preflight rejects missing hardware and dummy mode;
ordinary CPU/CUDA test runs skip this module's hardware cases.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.devices.rbln import RblnDeviceSpec
from lmcache.v1.platform.devices.rbln.device_ops import RblnDeviceOps
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
from lmcache.v1.platform.torch_ops import multi_layer_block_kv_transfer
import lmcache.lmcache_native as lmcache_native

pytestmark = pytest.mark.rbln

NUM_LAYERS = 2
NUM_BLOCKS = 8
NUM_HEADS = 2
BLOCK_SIZE = 32
HEAD_SIZE = 128
CHUNK_TOKENS = 2 * BLOCK_SIZE
BLOCK_IDS = [6, 1, 4, 0, 3]


def _host_layers(layout: str, dtype: torch.dtype) -> list[torch.Tensor]:
    """Create reproducible CPU KV caches in native HND or MLA format.

    Args:
        layout: Native HND attention or MLA cache layout.
        dtype: Element type of the returned tensors.

    Returns:
        One CPU tensor per layer, containing reproducible random values.
    """
    shape = (
        (2, NUM_BLOCKS, NUM_HEADS, 1, BLOCK_SIZE, HEAD_SIZE)
        if layout == "hnd"
        else (NUM_BLOCKS, BLOCK_SIZE, HEAD_SIZE)
    )
    generator = torch.Generator().manual_seed(17)
    return [
        torch.randn(shape, generator=generator).to(dtype) for _ in range(NUM_LAYERS)
    ]


def _chunks(layout: str, dtype: torch.dtype) -> list[torch.Tensor]:
    """Allocate three host chunks, including a trailing half-full chunk.

    Args:
        layout: Native HND attention or MLA cache layout.
        dtype: Element type of the returned tensors.

    Returns:
        Zeroed canonical token-major CPU chunks for the selected blocks.
    """
    shape = (
        (2, NUM_LAYERS, CHUNK_TOKENS, NUM_HEADS * HEAD_SIZE)
        if layout == "hnd"
        else (NUM_LAYERS, CHUNK_TOKENS, HEAD_SIZE)
    )
    return [torch.zeros(shape, dtype=dtype) for _ in range(3)]


def _shape_desc(layout: str) -> PageBufferShapeDesc:
    """Describe the test's two-byte HND or MLA elements for block transfer.

    Args:
        layout: Native HND attention or MLA cache layout.

    Returns:
        Descriptor matching the paged layers and chunk capacity.
    """
    desc = PageBufferShapeDesc()
    desc.kv_size = 2 if layout == "hnd" else 1
    desc.nl = NUM_LAYERS
    desc.nb = NUM_BLOCKS
    desc.bs = BLOCK_SIZE
    desc.nh = NUM_HEADS if layout == "hnd" else 1
    desc.hs = HEAD_SIZE
    desc.element_size = 2
    return desc


def _format(layout: str) -> lmcache_native.EngineKVFormat:
    """Return the upstream-supported native RBLN format for a test layout.

    Args:
        layout: Native HND attention or MLA cache layout.

    Returns:
        Engine KV format accepted by the RBLN device operations.
    """
    if layout == "hnd":
        return lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS
    return lmcache_native.EngineKVFormat.NL_X_NB_BS_HS


@pytest.fixture
def rbln_device() -> torch.device:
    """Return logical NPU 0, skipping unless real device DRAM is available.

    Returns:
        The first logical device exposed by the RBLN runtime.

    Note:
        Missing hardware, dummy mode or disabled eager allocation skips the test.
    """
    if not RblnDeviceSpec().is_available():
        pytest.skip("requires an RBLN NPU")
    if torch.rbln.is_dummy_device():
        pytest.skip("requires real RBLN hardware")
    if os.environ.get("TORCH_RBLN_EAGER_MALLOC") != "1":
        pytest.skip("set TORCH_RBLN_EAGER_MALLOC=1 before starting Python")
    return torch.device("rbln:0")


@pytest.mark.parametrize("layout", ["hnd", "mla"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("skip_prefix", [0, 3])
def test_device_transfer_preserves_canonical_chunks_and_untouched_blocks(
    rbln_device: torch.device, layout: str, dtype: torch.dtype, skip_prefix: int
) -> None:
    """Gather matches CPU bytes; scatter changes only the requested suffix.

    Args:
        rbln_device: Allocated real NPU with eager DRAM allocation enabled.
        layout: Native HND attention or MLA cache layout.
        dtype: Two-byte cache element type.
        skip_prefix: Cached blocks to leave unchanged, crossing a chunk boundary.
    """
    host = _host_layers(layout, dtype)
    reference = _chunks(layout, dtype)
    reference_layers = [layer.squeeze(3) for layer in host] if layout == "hnd" else host
    reference_format = (
        lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS
        if layout == "hnd"
        else _format(layout)
    )
    multi_layer_block_kv_transfer(
        reference_layers,
        reference,
        BLOCK_IDS,
        torch.device("cpu"),
        lmcache_native.TransferDirection.D2H,
        _shape_desc(layout),
        CHUNK_TOKENS,
        reference_format,
        0,
    )
    source = [layer.to(rbln_device) for layer in host]
    chunks = _chunks(layout, dtype)
    ops = RblnDeviceOps()
    ops.multi_layer_block_kv_transfer(
        source,
        chunks,
        BLOCK_IDS,
        rbln_device,
        lmcache_native.TransferDirection.D2H,
        _shape_desc(layout),
        CHUNK_TOKENS,
        _format(layout),
        0,
    )
    torch.rbln.synchronize()
    for got, expected in zip(chunks, reference, strict=True):
        assert torch.equal(got, expected)

    restored = [torch.full_like(layer, 7).to(rbln_device) for layer in host]
    ops.multi_layer_block_kv_transfer(
        restored,
        chunks,
        BLOCK_IDS,
        rbln_device,
        lmcache_native.TransferDirection.H2D,
        _shape_desc(layout),
        CHUNK_TOKENS,
        _format(layout),
        skip_prefix,
    )
    torch.rbln.synchronize()
    selected = BLOCK_IDS[skip_prefix:]
    for got, original in zip(restored, host, strict=True):
        expected = torch.full_like(original, 7)
        if layout == "hnd":
            expected[:, selected] = original[:, selected]
        else:
            expected[selected] = original[selected]
        assert torch.equal(got.cpu(), expected)


@pytest.mark.parametrize("layout", ["hnd", "mla"])
def test_concurrent_device_gathers_keep_chunks_separate(
    rbln_device: torch.device, layout: str
) -> None:
    """Concurrent MP transfer workers must not overwrite each other's staging.

    Args:
        rbln_device: Allocated real NPU with eager DRAM allocation enabled.
        layout: Native HND attention or MLA cache layout.
    """
    host = _host_layers(layout, torch.float16)
    source = [layer.to(rbln_device) for layer in host]
    ops = RblnDeviceOps()

    def gather(block_ids: list[int]) -> torch.Tensor:
        chunk = _chunks(layout, torch.float16)[:1]
        ops.multi_layer_block_kv_transfer(
            source,
            chunk,
            block_ids,
            rbln_device,
            lmcache_native.TransferDirection.D2H,
            _shape_desc(layout),
            CHUNK_TOKENS,
            _format(layout),
            0,
        )
        return chunk[0]

    groups = [BLOCK_IDS[:2], BLOCK_IDS[2:4]] * 4
    reference = [gather(group) for group in groups]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(gather, groups))
    for got, expected in zip(results, reference, strict=True):
        assert torch.equal(got, expected)
