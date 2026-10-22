# SPDX-License-Identifier: Apache-2.0
"""RBLN LMCache-driven building blocks: IPC wrapper and event backend.

The descriptor, fd hand-off and event tests replace ``torch.rbln`` with stubs
and run anywhere. ``test_server_opens_worker_tensor_across_processes`` needs a
torch-rbln with ``torch.rbln.ipc`` and read/write access to
``/dev/accel/accel0``.
"""

# Standard
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
import multiprocessing
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.base.ipc_wire import RblnIPCData, TensorMetadata
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
from lmcache.v1.platform.devices.rbln.event_ipc import RblnHostSyncEventIPCBackend
from lmcache.v1.platform.devices.rbln.ipc_wrapper import (
    RblnIPCWrapper,
    fetch_fds,
    is_rbln_memory_ipc_available,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_rbln_tensor() -> Any:
    """Duck-typed RBLN layer view: rows 2..5 of a 64-byte fp16 buffer."""
    view = torch.empty(32, dtype=torch.float16)[8:24].view(4, 4)
    return SimpleNamespace(
        device=SimpleNamespace(type="rbln", index=0),
        dtype=view.dtype,
        shape=view.shape,
        stride=view.stride,
        storage_offset=view.storage_offset,
        untyped_storage=view.untyped_storage,
        new_empty=view.new_empty,
    )


@pytest.fixture
def stub_rbln(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Stub ``torch.rbln.ipc`` export and the physical-NPU lookup.

    The stub export hands out the read end of a pipe as the "dma-buf" fd, so
    a test can check which file the importer received.
    """
    pipes: list[tuple[int, int]] = []

    def export_tensor(tensor: Any) -> Any:
        r, w = os.pipe()
        pipes.append((r, w))
        return SimpleNamespace(fds=[r], payload=b"layout", nbytes=tensor.nbytes)

    ipc = SimpleNamespace(export_tensor=export_tensor, open_tensor=lambda h, f: None)
    monkeypatch.setattr(torch, "rbln", SimpleNamespace(ipc=ipc), raising=False)
    rsd = SimpleNamespace(get_physical_device_ids=lambda index: [index + 3])
    with patch.dict(
        "sys.modules",
        {
            "torch_rbln": SimpleNamespace(),
            "torch_rbln._internal": SimpleNamespace(),
            "torch_rbln._internal.rsd_utils": rsd,
        },
    ):
        yield pipes
    for _, w in pipes:
        os.close(w)


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------


def test_descriptor_round_trips_without_fds(stub_rbln: Any) -> None:
    """The wire descriptor restores the wrapper; fds never ride it."""
    wrapper = RblnIPCWrapper.wrap(_fake_rbln_tensor())
    data = wrapper.to_wire()

    decoded = DeviceIPCWrapper.Deserialize(DeviceIPCWrapper.Serialize(wrapper))

    assert isinstance(decoded, RblnIPCWrapper)
    assert decoded == wrapper
    assert decoded.to_wire() == data
    assert data.payload == b"layout"
    assert data.num_fds == 1
    assert decoded.device_uuid == "rbln-npu:3"


def test_a_layer_view_exports_its_whole_allocation(stub_rbln: Any) -> None:
    """The export covers the shared buffer; the view rides the metadata."""
    data = RblnIPCWrapper.wrap(_fake_rbln_tensor()).to_wire()

    assert data.nbytes == 64
    assert data.metadata.shape == (4, 4)
    assert data.metadata.stride == (4, 1)
    assert data.metadata.storage_offset == 8


def test_descriptor_rejects_a_view_beyond_the_export() -> None:
    """A view larger than the exported bytes is refused at decode."""
    metadata = TensorMetadata(
        dtype="float16",
        shape=(4, 8),
        stride=(8, 1),
        storage_offset=0,
        device_uuid="rbln-npu:0",
    )
    with pytest.raises(ValueError, match="exceeds RBLN export"):
        RblnIPCData(
            version=1,
            metadata=metadata,
            export_id=b"\0" * 16,
            payload=b"",
            nbytes=63,
            num_fds=1,
            fd_socket="s",
        )


def test_wrap_rejects_a_non_rbln_tensor(stub_rbln: Any) -> None:
    """Only RBLN tensors can be exported."""
    with pytest.raises(ValueError, match="rbln tensor"):
        RblnIPCWrapper.wrap(torch.zeros(4))


def test_wrap_requires_torch_rbln_ipc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``torch.rbln.ipc`` the export fails with a clear error."""
    monkeypatch.setattr(torch, "rbln", SimpleNamespace(), raising=False)
    assert is_rbln_memory_ipc_available() is False
    with pytest.raises(RuntimeError, match="torch.rbln.ipc is not available"):
        RblnIPCWrapper.wrap(_fake_rbln_tensor())


# ---------------------------------------------------------------------------
# fd hand-off
# ---------------------------------------------------------------------------


def test_importer_receives_the_exported_fds_once(stub_rbln: Any) -> None:
    """The exporter passes its fds over SCM_RIGHTS, then forgets them."""
    data = RblnIPCWrapper.wrap(_fake_rbln_tensor()).to_wire()

    fds = fetch_fds(data.fd_socket, data.export_id, data.num_fds)
    try:
        # The received fd is the same pipe the stub exported.
        os.write(stub_rbln[0][1], b"kv")
        assert os.read(fds[0], 2) == b"kv"
    finally:
        for fd in fds:
            os.close(fd)

    with pytest.raises(RuntimeError, match="0 of 1"):
        fetch_fds(data.fd_socket, data.export_id, data.num_fds)


def test_exports_are_served_independently(stub_rbln: Any) -> None:
    """Each export id yields its own fds."""
    first = RblnIPCWrapper.wrap(_fake_rbln_tensor()).to_wire()
    second = RblnIPCWrapper.wrap(_fake_rbln_tensor()).to_wire()
    assert first.fd_socket == second.fd_socket
    assert first.export_id != second.export_id

    fds = fetch_fds(second.fd_socket, second.export_id, 1)
    fds += fetch_fds(first.fd_socket, first.export_id, 1)
    try:
        os.write(stub_rbln[1][1], b"2")
        os.write(stub_rbln[0][1], b"1")
        assert os.read(fds[0], 1) == b"2"
        assert os.read(fds[1], 1) == b"1"
    finally:
        for fd in fds:
            os.close(fd)


# ---------------------------------------------------------------------------
# Event backend
# ---------------------------------------------------------------------------


def test_record_drains_the_streams_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recording synchronizes the stream's device; the handle carries nothing."""
    synced: list[object] = []
    monkeypatch.setattr(
        torch, "rbln", SimpleNamespace(synchronize=synced.append), raising=False
    )
    backend = RblnHostSyncEventIPCBackend()
    backend.check_event_support("rbln:0")
    event = backend.create_event("rbln:0")

    backend.record_event(event, SimpleNamespace(device="rbln:1"))
    handle = backend.export_event(event, "rbln:1")

    assert synced == ["rbln:1"]
    assert handle == b""
    imported = backend.import_event(handle, "rbln:1")
    backend.wait_event(imported, None)
    assert backend.query_event(imported) is True


def test_event_support_requires_torch_rbln(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``torch.rbln.synchronize`` the backend fails closed."""
    monkeypatch.setattr(torch, "rbln", SimpleNamespace(), raising=False)
    with pytest.raises(RuntimeError, match="synchronize is unavailable"):
        RblnHostSyncEventIPCBackend().check_event_support("rbln:0")


# ---------------------------------------------------------------------------
# Hardware: a second process opens the exported tensor
# ---------------------------------------------------------------------------


def _hardware_ready() -> bool:
    return is_rbln_memory_ipc_available() and os.access(
        "/dev/accel/accel0", os.R_OK | os.W_OK
    )


def _importer(descriptor: bytes, conn: Any) -> None:
    """Open the descriptor, report what it holds, then overwrite it."""
    # First Party
    from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper

    wrapper = DeviceIPCWrapper.Deserialize(descriptor)
    tensor = wrapper.to_tensor()
    conn.send(tensor.cpu())
    tensor.copy_(torch.full(tensor.shape, 7.0, dtype=tensor.dtype))
    torch.rbln.synchronize()  # type: ignore[attr-defined]
    wrapper.close()
    conn.send("done")


@pytest.mark.skipif(not _hardware_ready(), reason="needs torch.rbln.ipc and an NPU")
def test_server_opens_worker_tensor_across_processes() -> None:
    """The importer sees the exporter's data and its writes reach the exporter.

    The exported tensor is the second of two layer views into one buffer, as
    vLLM-RBLN allocates them.
    """
    layer_bytes = 16 * 256 * 2
    buffer = torch.rbln.ipc.empty_exportable((2 * layer_bytes,), torch.int8, device=0)  # type: ignore[attr-defined]
    layers = [
        buffer[i * layer_bytes : (i + 1) * layer_bytes]
        .view(torch.float16)
        .view(16, 256)
        for i in range(2)
    ]
    expected = torch.arange(16 * 256, dtype=torch.float16).reshape(16, 256)
    layers[0].zero_()
    layers[1].copy_(expected)
    torch.rbln.synchronize()  # type: ignore[attr-defined]
    kv = layers[1]
    descriptor = DeviceIPCWrapper.Serialize(RblnIPCWrapper.wrap(kv))

    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=_importer, args=(descriptor, child))
    proc.start()
    seen = parent.recv()
    assert parent.recv() == "done"
    proc.join(timeout=60)

    assert proc.exitcode == 0
    assert torch.equal(seen, expected)
    assert bool((kv.cpu() == 7.0).all())
    assert bool((layers[0].cpu() == 0).all())
