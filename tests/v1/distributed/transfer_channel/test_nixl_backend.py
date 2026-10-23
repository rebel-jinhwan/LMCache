# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for how the nixl transfer channel creates its NIXL backend.

A fake ``nixl._api`` module stands in for the NIXL bindings, so these tests run
without nixl installed. The end-to-end transfer tests against a real nixl
runtime live in ``test_nixl_impl.py``.
"""

# Standard
from unittest.mock import MagicMock
import itertools
import sys
import types

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.transfer_channel import TransferChannelContext
from lmcache.v1.distributed.transfer_channel.factory import (
    create_transfer_channel_context,
)

# Distinct ports so the handshake servers of consecutive tests never clash.
_port_counter = itertools.count(18900)


@pytest.fixture
def fake_nixl(monkeypatch) -> types.SimpleNamespace:
    """Install a fake ``nixl._api`` module whose agent is a ``MagicMock``."""
    api = types.SimpleNamespace(
        nixl_agent=MagicMock(name="nixl_agent"),
        nixl_agent_config=MagicMock(name="nixl_agent_config"),
    )
    monkeypatch.setitem(sys.modules, "nixl", types.ModuleType("nixl"))
    monkeypatch.setitem(sys.modules, "nixl._api", api)
    return api


def _create_nixl_context(**kwargs) -> TransferChannelContext:
    url = f"127.0.0.1:{next(_port_counter)}"
    return create_transfer_channel_context(
        transfer_channel_type="nixl",
        l1_memory_desc=L1MemoryDesc(ptr=0x10000, size=4096, align_bytes=256),
        listen_url=url,
        advertise_url=url,
        **kwargs,
    )


def test_default_backend_is_ucx_with_no_params(fake_nixl):
    ctx = _create_nixl_context()
    try:
        # No backend is created through the agent config ...
        fake_nixl.nixl_agent_config.assert_called_once_with(backends=[])
        # ... only the explicit UCX one, with empty params as before.
        agent = fake_nixl.nixl_agent.return_value
        agent.create_backend.assert_called_once_with("UCX", {})
    finally:
        ctx.close()


def test_named_backend_is_created_with_its_params(fake_nixl):
    ctx = _create_nixl_context(backend="RBLN", backend_params={"listen_ip": "all"})
    try:
        agent = fake_nixl.nixl_agent.return_value
        agent.create_backend.assert_called_once_with("RBLN", {"listen_ip": "all"})
    finally:
        ctx.close()


def test_backend_is_created_before_l1_is_registered(fake_nixl):
    ctx = _create_nixl_context(backend="RBLN", backend_params={"listen_ip": "all"})
    try:
        calls = [name for name, _, _ in fake_nixl.nixl_agent.return_value.mock_calls]
        assert calls.index("create_backend") < calls.index("register_memory")
    finally:
        ctx.close()
