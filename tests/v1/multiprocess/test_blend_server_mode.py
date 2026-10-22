# SPDX-License-Identifier: Apache-2.0
"""Which blend module ``--engine-type blend`` composes per transfer mode."""

# Standard
from typing import Any, Literal
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess import server as server_mod
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.modules import blend as blend_pkg
from lmcache.v1.multiprocess.transfer_mode import MPTransferMode

TransferModeName = Literal["lmcache_driven", "engine_driven", "auto"]


class _FakeLMCacheDriven:
    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx


class _FakeEngineDriven:
    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx


class _FakeBlend:
    """Records the transfer module the server hands to the blend module."""

    def __init__(self, ctx: Any, transfer: Any, **_kwargs: Any) -> None:
        self.transfer = transfer


class _FakeLMCacheDrivenBlend(_FakeBlend):
    pass


class _FakeEngineDrivenBlend(_FakeBlend):
    pass


@pytest.fixture
def stub_server_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub module constructors; the transfer modules stay distinct classes
    because the server isinstance-checks them."""
    monkeypatch.setattr(server_mod, "LookupModule", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(server_mod, "P2PController", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(server_mod, "ManagementModule", MagicMock())
    monkeypatch.setattr(server_mod, "LMCacheDrivenTransferModule", _FakeLMCacheDriven)
    monkeypatch.setattr(server_mod, "EngineDrivenTransferModule", _FakeEngineDriven)
    monkeypatch.setattr(blend_pkg, "BlendModule", _FakeLMCacheDrivenBlend)
    monkeypatch.setattr(blend_pkg, "EngineDrivenBlendModule", _FakeEngineDrivenBlend)


def _blend_module(supported_transfer_mode: TransferModeName) -> _FakeBlend:
    modules = server_mod._build_modules(
        MagicMock(name="ctx"),
        MPServerConfig(
            engine_type="blend",
            supported_transfer_mode=supported_transfer_mode,
            kv_event_log_size=0,
        ),
        MagicMock(url=""),
    )
    (blend,) = [m for m in modules if isinstance(m, _FakeBlend)]
    return blend


@pytest.mark.parametrize(
    ("mode", "host_default", "blend_cls", "transfer_cls"),
    [
        pytest.param(
            "engine_driven",
            MPTransferMode.LMCACHE_DRIVEN,
            _FakeEngineDrivenBlend,
            _FakeEngineDriven,
            id="engine_driven",
        ),
        pytest.param(
            "lmcache_driven",
            MPTransferMode.ENGINE_DRIVEN,
            _FakeLMCacheDrivenBlend,
            _FakeLMCacheDriven,
            id="lmcache_driven",
        ),
        pytest.param(
            "auto",
            MPTransferMode.ENGINE_DRIVEN,
            _FakeEngineDrivenBlend,
            _FakeEngineDriven,
            id="auto-engine-driven-host",
        ),
        pytest.param(
            "auto",
            MPTransferMode.LMCACHE_DRIVEN,
            _FakeLMCacheDrivenBlend,
            _FakeLMCacheDriven,
            id="auto-lmcache-driven-host",
        ),
    ],
)
def test_blend_follows_the_transfer_mode(
    stub_server_modules: None,
    monkeypatch: pytest.MonkeyPatch,
    mode: TransferModeName,
    host_default: MPTransferMode,
    blend_cls: type,
    transfer_cls: type,
) -> None:
    """An explicit mode picks its blend module; ``auto`` follows the path the
    host's workers default to (RBLN and CPU: engine-driven)."""
    spec = MagicMock()
    spec.default_mp_transfer_mode.return_value = host_default
    monkeypatch.setattr(server_mod, "current_device_spec", spec)

    blend = _blend_module(mode)

    assert type(blend) is blend_cls
    assert isinstance(blend.transfer, transfer_cls)
