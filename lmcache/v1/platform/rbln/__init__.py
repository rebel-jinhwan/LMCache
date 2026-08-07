# SPDX-License-Identifier: Apache-2.0
"""RBLN (Rebellions NPU) platform primitives.

Registers :class:`RblnDeviceSpec` with the device-detection registry so
LMCache resolves ``torch.rbln`` as an accelerator instead of falling back
to the CPU stub.  ``torch.rbln`` is contributed by the ``torch_rbln``
package through a torch backend entry point, so it is visible on a bare
``import torch`` -- no explicit ``import torch_rbln`` is required here.

Scope: **engine-driven** multiprocess (MP) transfer only.  ``torch.rbln``
exposes device discovery and ``synchronize()`` but no ``Stream`` / ``Event``
types, so the LMCache-driven path (which needs cross-process event IPC and
an IPC handle wrapper) cannot be supported.  The spec therefore reports
:meth:`RblnDeviceSpec.is_handle_transfer_available` as ``False`` and leaves
``ipc_wrapper_cls`` / ``event_ipc_backend`` at their ``None`` defaults, so
requesting ``mp_transfer_mode=lmcache_driven`` fails fast with a clear
error rather than crashing deeper in the transfer path.

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
    from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache
    from lmcache.v1.platform.base.device_ops import DeviceOps

# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class RblnDeviceSpec(DeviceSpec):
    """RBLN device specification for the detection registry."""

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

    def normalize_kv_caches(
        self, kv_caches: DiscoverableKVCache
    ) -> DiscoverableKVCache:
        """Squeeze vLLM-RBLN's singleton axis so the detectors see a 5-D cache.

        vLLM-RBLN allocates ``[2, NB, NH, 1, BS, HS]`` per layer -- HND with an
        extra axis the RBLN attention backend requires. Axis 3 is always 1, so
        removing it is a free view onto identical bytes, and what is left is an
        ordinary per-layer cache the vLLM detector already classifies.

        This is the multiprocess path's entry point: ``compute_kv_layout``,
        ``gather_paged_kv_to_cpu`` and ``scatter_cpu_to_paged_kv`` resolve
        layouts through format discovery and never touch a connector.

        Args:
            kv_caches: Raw KV cache structure as vLLM-RBLN handed it over.

        Returns:
            DiscoverableKVCache: 5-D per-layer views when the input was the
            native 6-D layout, otherwise ``kv_caches`` unchanged.
        """
        # First Party
        from lmcache.v1.platform.rbln.kv_layout import normalize_kv_caches

        return normalize_kv_caches(kv_caches)

    def is_handle_transfer_available(self) -> bool:
        """Report that RBLN cannot ship KV tensors as IPC handles.

        The base class defaults to ``True``; RBLN overrides it to ``False``
        because ``torch.rbln`` exposes no ``Event`` type, so the ordered
        cross-process publication the LMCache-driven path depends on cannot
        be expressed.  Returning ``False`` keeps
        ``mp_transfer_mode=lmcache_driven`` failing at its documented
        validation point instead of at an attribute lookup later on.

        Returns:
            bool: Always ``False``.
        """
        return False
