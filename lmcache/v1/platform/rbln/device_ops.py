# SPDX-License-Identifier: Apache-2.0
"""RBLN ops backend: inherit the torch baseline unchanged.

:class:`RblnDeviceOps` gives the registry a ``device_type="rbln"`` entry so
``resolve_device_ops("rbln")`` returns a real spec instead of raising.  Every
op is inherited from :class:`DeviceOps` via MRO, which routes to the pure
torch implementations in :mod:`lmcache.v1.platform.torch_ops`.

The torch baseline is safe on RBLN: ``lmcache_memcpy_async`` takes its
tensor-mode branch for non-CUDA devices, and the completion / event
recorders degrade to immediate (unordered) publication.  Ordering is
preserved by the engine-driven transfer context, which calls
``torch_dev.synchronize()`` around gather and scatter.

Unlike :class:`~lmcache.v1.platform.musa.device_ops.MusaDeviceOps`, this class
does **not** override :meth:`multi_layer_block_kv_transfer` to reach a native
kernel.  RBLN's kernels stage each chunk head-major (``[2, L, H, T, D]``) while
every caller of that method assumes upstream's token-major ``[2, L, T, H*D]``,
so routing them through here would silently reinterpret KV bytes.  The native
path is therefore inseparable from a head-major transfer context and stays out
of tree.  See ``docs/design/v1/platform/rbln/README.md``.
"""

# Future
from __future__ import annotations

# Standard
from typing import ClassVar

# First Party
from lmcache.v1.platform.base.device_ops import DeviceOps


class RblnDeviceOps(DeviceOps):
    device_type: ClassVar[str] = "rbln"
