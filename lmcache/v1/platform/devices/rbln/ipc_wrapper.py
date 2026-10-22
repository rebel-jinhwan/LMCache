# SPDX-License-Identifier: Apache-2.0
"""RBLN KV-cache IPC wrapper over ``torch.rbln.ipc`` dma-buf exports.

``torch.rbln.ipc.export_tensor`` returns an opaque layout ``payload`` plus one
dma-buf fd per device area. The payload rides the data-only wire descriptor;
the fds cannot, so the exporting process serves them over ``SCM_RIGHTS``:

    worker (exporter)                         LMCache server (importer)
    wrap(tensor)
      export_tensor -> payload, fds
      park fds under a random export id
      listen on an abstract Unix socket
    REGISTER_KV_CACHE(descriptor) ----------> to_tensor()
                                                connect(fd_socket)
                     <------------------------  send(export_id)
      send_fds(fds), close parked fds ------->  recv_fds
                                                open_tensor(payload, fds)

The worker listens before the registration message leaves, so the importer
never races the exporter. Each export is served once: a re-registration (for
example after a server restart) exports again.

The socket is abstract (no filesystem path), so both processes must share a
network namespace. The exporter serves only peers with its own uid
(``SO_PEERCRED``).
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar
import os
import socket
import struct
import threading
import uuid

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform.base.ipc_wire import RblnIPCData, WireIPCData
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper

logger = init_logger(__name__)

# Reply header: number of fds attached (0 means the export id is unknown).
_REPLY = struct.Struct("<I")
_EXPORT_ID_BYTES = 16


def is_rbln_memory_ipc_available() -> bool:
    """Return whether this torch-rbln exposes ``torch.rbln.ipc``.

    Returns:
        ``True`` when ``torch.rbln.ipc`` provides tensor export and open.
    """
    ipc = getattr(getattr(torch, "rbln", None), "ipc", None)
    return callable(getattr(ipc, "export_tensor", None)) and callable(
        getattr(ipc, "open_tensor", None)
    )


class _FdServer:
    """Hands parked dma-buf fds to the importer, once per export id."""

    def __init__(self) -> None:
        self.name = f"lmcache-rbln-ipc-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        self._parked: dict[bytes, list[int]] = {}
        self._lock = threading.Lock()
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind("\0" + self.name)
        self._sock.listen()
        threading.Thread(
            target=self._serve, name="lmcache-rbln-ipc-fds", daemon=True
        ).start()

    def park(self, export_id: bytes, fds: list[int]) -> None:
        """Hold ``fds`` until the importer of ``export_id`` collects them."""
        with self._lock:
            self._parked[export_id] = fds

    def _serve(self) -> None:
        while True:
            conn, _ = self._sock.accept()
            with conn:
                try:
                    self._serve_one(conn)
                except OSError:
                    logger.exception("Serving RBLN dma-buf fds failed")

    def _serve_one(self, conn: socket.socket) -> None:
        creds = conn.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
        )
        _, peer_uid, _ = struct.unpack("3i", creds)
        if peer_uid != os.getuid():
            logger.warning("Refusing RBLN fd request from uid %d", peer_uid)
            return
        export_id = _recv_exact(conn, _EXPORT_ID_BYTES)
        with self._lock:
            fds = self._parked.pop(export_id, None)
        if fds is None:
            conn.sendall(_REPLY.pack(0))
            return
        try:
            socket.send_fds(conn, [_REPLY.pack(len(fds))], fds)
        finally:
            for fd in fds:
                os.close(fd)


_fd_server: _FdServer | None = None
_fd_server_lock = threading.Lock()


def _get_fd_server() -> _FdServer:
    global _fd_server
    with _fd_server_lock:
        if _fd_server is None:
            _fd_server = _FdServer()
        return _fd_server


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            raise ConnectionError("RBLN fd socket closed mid-message")
        data += chunk
    return data


def fetch_fds(fd_socket: str, export_id: bytes, num_fds: int) -> list[int]:
    """Collect the dma-buf fds of ``export_id`` from its exporter.

    Args:
        fd_socket: Abstract socket name from the descriptor.
        export_id: Export to collect.
        num_fds: Number of fds the descriptor announced.

    Returns:
        The received fds, owned by the caller.

    Raises:
        RuntimeError: If the exporter no longer holds the export or sends a
            different number of fds.
    """
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.connect("\0" + fd_socket)
        conn.sendall(export_id)
        msg, fds, _, _ = socket.recv_fds(conn, _REPLY.size, num_fds)
    (sent,) = _REPLY.unpack(msg)
    if sent != num_fds or len(fds) != num_fds:
        for fd in fds:
            os.close(fd)
        raise RuntimeError(
            f"RBLN exporter {fd_socket} returned {len(fds)} of {num_fds} dma-buf "
            "fd(s); the export was already collected or the worker restarted"
        )
    return list(fds)


class RblnIPCWrapper(DeviceIPCWrapper):
    """Ships one RBLN KV tensor from ``torch.rbln.ipc.empty_exportable``."""

    device_type: ClassVar[str] = "rbln"

    @classmethod
    def wrap(cls, tensor: torch.Tensor) -> "RblnIPCWrapper":
        """Export ``tensor`` for the LMCache server.

        Args:
            tensor: An RBLN tensor covering a whole exportable allocation.

        Returns:
            A wrapper ready for the multiprocess wire.
        """
        return cls(tensor)

    def __init__(self, tensor: torch.Tensor) -> None:
        """Export the allocation behind ``tensor`` and park its fds.

        ``export_tensor`` only accepts a tensor spanning a whole allocation,
        while vLLM-RBLN hands out per-layer views into one shared buffer. The
        whole storage is exported, and the view (dtype, shape, stride, offset)
        travels in the descriptor metadata.

        Args:
            tensor: An RBLN tensor whose storage was allocated by
                ``torch.rbln.ipc.empty_exportable``. Export pins its device
                layout, so the engine must have placed it (run the model on
                it) before registering.

        Raises:
            ValueError: If ``tensor`` is not an RBLN tensor.
            RuntimeError: If ``torch.rbln.ipc`` is missing or rejects the
                storage (not allocated as exportable).
        """
        if tensor.device.type != "rbln":
            raise ValueError(
                f"RblnIPCWrapper needs an rbln tensor, got {tensor.device}"
            )
        if not is_rbln_memory_ipc_available():
            raise RuntimeError(
                "torch.rbln.ipc is not available; LMCache-driven transfer on RBLN "
                "needs a torch-rbln with cross-process device memory support"
            )
        storage = tensor.untyped_storage()
        whole = tensor.new_empty((0,), dtype=torch.uint8).set_(
            storage, 0, (storage.nbytes(),), (1,)
        )
        exported = torch.rbln.ipc.export_tensor(whole)  # type: ignore[attr-defined]
        server = _get_fd_server()
        self._export_id = uuid.uuid4().bytes
        server.park(self._export_id, list(exported.fds))
        self._fd_socket = server.name
        self._payload = bytes(exported.payload)
        self._nbytes = int(exported.nbytes)
        self._num_fds = len(exported.fds)
        self._tensor: torch.Tensor | None = None

        self.handle = self._export_id
        self.dtype = tensor.dtype
        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.storage_offset = int(tensor.storage_offset())
        device_index = tensor.device.index
        self.device_uuid = self._get_device_uuid(
            0 if device_index is None else device_index
        )

    def to_wire(self) -> RblnIPCData:
        """Return the descriptor; the fds stay with the exporter."""
        return RblnIPCData(
            version=1,
            metadata=self.wire_metadata(),
            export_id=self._export_id,
            payload=self._payload,
            nbytes=self._nbytes,
            num_fds=self._num_fds,
            fd_socket=self._fd_socket,
        )

    @classmethod
    def from_wire(cls, data: WireIPCData) -> "RblnIPCWrapper":
        """Restore a descriptor without collecting fds or opening memory."""
        if not isinstance(data, RblnIPCData):
            raise ValueError("Expected RBLN IPC descriptor")
        obj = object.__new__(cls)
        obj.restore_metadata(data.metadata)
        obj.handle = data.export_id
        obj._export_id = data.export_id
        obj._payload = data.payload
        obj._nbytes = data.nbytes
        obj._num_fds = data.num_fds
        obj._fd_socket = data.fd_socket
        obj._tensor = None
        return obj

    def to_tensor(self) -> torch.Tensor:
        """Open the exported tensor in this process.

        The first call collects the fds from the exporter and maps the memory;
        later calls return the same tensor.

        Returns:
            A tensor aliasing the exporter's device memory.

        Raises:
            RuntimeError: If the fds cannot be collected or the runtime
                rejects the import.
        """
        if self._tensor is not None:
            return self._tensor
        device_index = self._get_device_index_from_uuid(self.device_uuid)
        fds = fetch_fds(self._fd_socket, self._export_id, self._num_fds)
        try:
            handle = torch.rbln.ipc.IpcHandle(  # type: ignore[attr-defined]
                fds=fds,
                payload=self._payload,
                nbytes=self._nbytes,
                shape=torch.Size((self._nbytes,)),
                dtype=torch.uint8,
                device_index=device_index,
            )
            whole = torch.rbln.ipc.open_tensor(  # type: ignore[attr-defined]
                handle, fds, device=device_index
            )
        finally:
            for fd in fds:
                os.close(fd)
        self._tensor = whole.view(self.dtype).as_strided(
            self.shape, self.stride, self.storage_offset
        )
        return self._tensor

    def close(self) -> None:
        """Drop the imported tensor; the mapping is released with it."""
        self._tensor = None

    @classmethod
    def _get_device_uuid(cls, device_index: int) -> str:
        """Name the physical NPUs behind logical device ``device_index``.

        ``torch.rbln`` device properties carry no UUID, and logical indices
        differ between processes with different ``RBLN_DEVICES``.
        """
        # Third Party
        from torch_rbln._internal.rsd_utils import get_physical_device_ids

        ids = get_physical_device_ids(device_index)
        if ids is None:
            raise RuntimeError(f"rbln:{device_index} has no physical NPU mapping")
        return "rbln-npu:" + ",".join(str(i) for i in ids)
