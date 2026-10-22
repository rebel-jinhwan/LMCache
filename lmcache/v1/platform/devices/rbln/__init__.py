# SPDX-License-Identifier: Apache-2.0
"""RBLN (Rebellions NPU) platform primitives.

Registers :class:`RblnDeviceSpec` with the device-detection registry so
LMCache resolves ``torch.rbln`` as an accelerator instead of falling back
to the CPU stub.  ``torch.rbln`` is contributed by the ``torch_rbln``
package through a torch backend entry point, so it is visible on a bare
``import torch`` -- no explicit ``import torch_rbln`` is required here.

Multiprocess (MP) transfer is **engine-driven** by default. With a torch-rbln
that provides ``torch.rbln.ipc`` (cross-process device memory),
``mp_transfer_mode=lmcache_driven`` is available as an opt-in: the worker
exports its KV tensors (:class:`~.ipc_wrapper.RblnIPCWrapper`) and the server
moves blocks itself (:class:`~.cache_context.RblnCacheContext`), ordered by
host synchronization (:class:`~.event_ipc.RblnHostSyncEventIPCBackend`) until
torch-rbln exports interprocess events.

See ``docs/design/v1/platform/devices/rbln/README.md`` for the full contract.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any

# First Party
from lmcache.v1.multiprocess.transfer_mode import MPTransferMode
from lmcache.v1.platform.base.device_spec import DeviceSpec

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base.cache_context import BaseCacheContext
    from lmcache.v1.platform.base.device_ops import DeviceOps
    from lmcache.v1.platform.base.event_ipc import EventIPCBackend
    from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper

# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class RblnDeviceSpec(DeviceSpec):
    """RBLN device specification for the detection registry."""

    _event_backend_cache: "EventIPCBackend | None" = None

    @property
    def device_type(self) -> str:
        return "rbln"

    @property
    def torch_module_name(self) -> str:
        return "rbln"

    @property
    def ops_cls(self) -> type[DeviceOps]:
        # First Party
        from lmcache.v1.platform.devices.rbln.device_ops import RblnDeviceOps

        return RblnDeviceOps

    def is_available(self) -> bool:
        """Check RBLN availability without importing ``lmcache.__init__``.

        ``torch.rbln.is_available()`` raises (rather than returning
        ``False``) when the runtime cannot register a physical NPU -- for
        example when another process already holds the device or a stale
        allocation survives.  Detection runs on every LMCache start,
        including on hosts with no free NPU, so the exception is swallowed
        and reported as "unavailable"; letting it escape would abort import
        for every co-tenant process on the box.

        Returns:
            bool: ``True`` when ``torch.rbln`` is present and reports at
            least one usable device, ``False`` otherwise.
        """
        try:
            # Third Party
            import torch

            return hasattr(torch, "rbln") and torch.rbln.is_available()
        except Exception:
            return False

    def is_lmcache_driven_available(self) -> bool:
        """Report whether torch-rbln can share device memory across processes.

        Returns:
            bool: ``True`` when ``torch.rbln.ipc`` is present.
        """
        # First Party
        from lmcache.v1.platform.devices.rbln.ipc_wrapper import (
            is_rbln_memory_ipc_available,
        )

        return is_rbln_memory_ipc_available()

    def default_mp_transfer_mode(self) -> MPTransferMode:
        """LMCache-driven is opt-in; ``auto`` stays engine-driven.

        The engine must allocate KV with ``torch.rbln.ipc.empty_exportable``
        and register it only after the model has placed it, which ``auto``
        cannot verify.
        """
        return MPTransferMode.ENGINE_DRIVEN

    @property
    def ipc_wrapper_cls(self) -> type[DeviceIPCWrapper] | None:
        # First Party
        from lmcache.v1.platform.devices.rbln.ipc_wrapper import RblnIPCWrapper

        return RblnIPCWrapper

    def create_cache_context(self, *args: Any, **kwargs: Any) -> "BaseCacheContext":
        """Create the RBLN cache context for LMCache-driven transfer."""
        # First Party
        from lmcache.v1.platform.devices.rbln.cache_context import RblnCacheContext

        return RblnCacheContext(*args, **kwargs)

    @property
    def event_ipc_backend(self) -> "EventIPCBackend":
        """Return the host-synchronous RBLN event backend."""
        backend = self._event_backend_cache
        if backend is None:
            # First Party
            from lmcache.v1.platform.devices.rbln.event_ipc import (
                RblnHostSyncEventIPCBackend,
            )

            backend = RblnHostSyncEventIPCBackend()
            self._event_backend_cache = backend
        return backend
