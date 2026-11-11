# SPDX-License-Identifier: Apache-2.0
"""``DeviceStagedMemoryObj`` without a device: the placeholder surface.

The cache context's half of the hook runs on hardware in
``tests/v1/platform/devices/rbln/test_rbln_device_staged.py``.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.memory_management import DeviceStagedMemoryObj
from tests.v1.distributed.device_staged_l1_fixture import (
    BudgetMemoryManager,
    HostCopyObject,
)


def test_device_staged_object_is_a_placeholder() -> None:
    layout = MemoryLayoutDesc([torch.Size([2, 4, 256, 512])], [torch.bfloat16])
    memory = BudgetMemoryManager(1 << 30, 4096)
    _, (obj,) = memory.allocate(layout, 1)
    assert isinstance(obj, DeviceStagedMemoryObj) and isinstance(obj, HostCopyObject)
    assert obj.tensor is None and obj.raw_tensor is None
    assert obj.get_physical_size() == 2 * 4 * 256 * 512 * 2
    assert obj.get_shapes() == [torch.Size([2, 4, 256, 512])]
    with pytest.raises(NotImplementedError, match="device-staged"):
        obj.data_ptr  # noqa: B018
    with pytest.raises(ValueError, match="nothing to retrieve"):
        obj.transfer_through_device(torch.device("cpu"), False, lambda _: None)
    memory.free([obj])
    assert not obj.is_valid()


def test_store_then_retrieve_through_a_cpu_buffer() -> None:
    layout = MemoryLayoutDesc([torch.Size([4, 8])], [torch.float32])
    memory = BudgetMemoryManager(1 << 20, 4096)
    _, (obj,) = memory.allocate(layout, 1)
    assert isinstance(obj, HostCopyObject)
    data = torch.arange(32, dtype=torch.float32)

    def gather(area: torch.Tensor) -> None:
        area[: data.numel() * 4].copy_(data.view(torch.uint8))

    obj.transfer_through_device(torch.device("cpu"), True, gather)
    restored = torch.empty_like(data)

    def scatter(area: torch.Tensor) -> None:
        restored.view(torch.uint8).copy_(area[: data.numel() * 4])

    obj.transfer_through_device(torch.device("cpu"), False, scatter)
    assert torch.equal(restored, data)
    assert obj.transfers == [True, False]
    memory.free([obj])
