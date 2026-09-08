# SPDX-License-Identifier: Apache-2.0
"""RBLN (Rebellions NPU) platform primitives.

Registers :class:`RblnDeviceSpec` with the device-detection registry so
LMCache resolves ``torch.rbln`` as an accelerator instead of falling back
to the CPU stub.  ``torch.rbln`` is contributed by the ``torch_rbln``
package through a torch backend entry point, so it is visible on a bare
``import torch`` -- no explicit ``import torch_rbln`` is required here.

Scope: **engine-driven** multiprocess (MP) transfer is supported.  The
**LMCache-driven** path is scaffolded but not yet enabled: the event IPC
backend (:mod:`lmcache.v1.platform.rbln.event_ipc`, host-synchronizing) and
the dma-buf IPC wrapper (:mod:`lmcache.v1.platform.rbln.ipc_wrapper`, stub)
are registered on the spec, while
:meth:`RblnDeviceSpec.is_handle_transfer_available` stays ``False`` until
the wrapper can actually export/import device memory.  Requesting
``mp_transfer_mode=lmcache_driven`` therefore still fails fast at its
documented validation point.

See ``docs/design/v1/platform/rbln/README.md`` for the full contract.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache.v1.platform.base.device_spec import DeviceSpec

if TYPE_CHECKING:
    # First Party
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
        from lmcache.v1.platform.rbln.device_ops import RblnDeviceOps

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

    def is_handle_transfer_available(self) -> bool:
        """Report that RBLN cannot yet ship KV tensors as IPC handles.

        Flip to ``True`` once :class:`~lmcache.v1.platform.rbln.ipc_wrapper.
        RblnIPCWrapper` implements dma-buf export/import.  Until then,
        ``mp_transfer_mode=lmcache_driven`` fails at its documented
        validation point instead of at ``NotImplementedError`` later on.

        Returns:
            bool: Always ``False``.
        """
        return False

    @property
    def event_ipc_backend(self) -> "EventIPCBackend":
        """Return the host-synchronizing RBLN event IPC backend."""
        backend = self._event_backend_cache
        if backend is None:
            # First Party
            from lmcache.v1.platform.rbln.event_ipc import RblnEventIPCBackend

            backend = RblnEventIPCBackend()
            self._event_backend_cache = backend
        return backend

    @property
    def ipc_wrapper_cls(self) -> "type[DeviceIPCWrapper]":
        """Return the dma-buf IPC wrapper class (export/import still stubbed)."""
        # First Party
        from lmcache.v1.platform.rbln.ipc_wrapper import RblnIPCWrapper

        return RblnIPCWrapper
