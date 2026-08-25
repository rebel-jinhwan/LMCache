# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F811 -- fixtures are imported from the sibling module by name
"""Chunk format (``MemoryFormat``) plumbing on the engine-driven MP path.

The layout of a staged chunk is a property of the *store*: nothing in the
bytes says whether a ``[2, L, T, H*D]`` buffer holds token-major ``KV_2LTD``
or head-major ``KV_2LHTD``. These tests pin the contract that makes the two
safe to coexist in one codebase:

* the worker announces its format at registration and the server stamps it
  on every object it allocates (``MemoryLayoutDesc.fmt`` -> L1 allocation);
* the server refuses to hand back an object stored under another format
  instead of letting the worker scatter it as garbage;
* the worker refuses head-major up front for engine layouts it cannot be
  produced from (NHD, MLA, fused, lmcache-driven path);
* gather / scatter round-trip under ``KV_2LHTD`` and produce the head-major
  permutation of the canonical chunk.
"""

# Standard
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock
import pickle

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.multiprocess.custom_types import RegisterEngineDrivenContextPayload
from lmcache.v1.multiprocess.mq import msgspec_decode, msgspec_encode
from lmcache.v1.multiprocess.transfer_context.base import (
    SUPPORTED_CHUNK_FORMATS,
    compute_kv_layout,
    gather_paged_kv_to_cpu,
    parse_chunk_format,
    scatter_cpu_to_paged_kv,
    validate_chunk_format,
)
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
    create_transfer_context,
)
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.torch_ops import head_major_chunk_view
import lmcache
import lmcache.lmcache_native as lmcache_native

# Local
from .test_engine_driven_transfer import (  # noqa: F401 -- fixtures
    ServerModuleFactory,
    _default_key,
    _default_register_payload,
    _make_hnd_kv_caches,
    _make_kv_caches,
    server_module_factory,
    stub_lmcache_native,
)

EngineKVFormat = lmcache_native.EngineKVFormat
HND_HINTS: dict[str, Any] = {"kv_layout": "HND"}


# ---------------------------------------------------------------------------
# Format vocabulary
# ---------------------------------------------------------------------------


def test_kv_2lhtd_shares_token_dim_with_kv_2ltd() -> None:
    """Same logical shape, so token counting must agree."""
    assert MemoryFormat.KV_2LHTD.token_dim() == MemoryFormat.KV_2LTD.token_dim()


@pytest.mark.parametrize("name", ["KV_2LTD", "kv_2lhtd", " KV_2LHTD "])
def test_parse_chunk_format_accepts_supported_names(name: str) -> None:
    assert parse_chunk_format(name) in SUPPORTED_CHUNK_FORMATS


@pytest.mark.parametrize("name", ["KV_T2D", "head_major", "", "BINARY"])
def test_parse_chunk_format_rejects_other_formats(name: str) -> None:
    with pytest.raises(ValueError, match="Unsupported chunk format"):
        parse_chunk_format(name)


def test_validate_head_major_requires_split_hnd() -> None:
    validate_chunk_format(
        MemoryFormat.KV_2LHTD, EngineKVFormat.NL_X_TWO_NB_NH_BS_HS, kv_size=2
    )
    validate_chunk_format(
        MemoryFormat.KV_2LHTD, EngineKVFormat.NL_X_NB_TWO_NH_BS_HS, kv_size=2
    )
    with pytest.raises(ValueError, match="head-major"):
        validate_chunk_format(
            MemoryFormat.KV_2LHTD, EngineKVFormat.NL_X_TWO_NB_BS_NH_HS, kv_size=2
        )
    with pytest.raises(ValueError, match="head-major"):
        validate_chunk_format(
            MemoryFormat.KV_2LHTD, EngineKVFormat.NL_X_TWO_NB_NH_BS_HS, kv_size=1
        )
    # Token-major is always fine.
    validate_chunk_format(
        MemoryFormat.KV_2LTD, EngineKVFormat.NL_X_TWO_NB_BS_NH_HS, kv_size=2
    )


def test_memory_layout_desc_defaults_to_token_major() -> None:
    """Descriptors built before the field existed keep their old meaning."""
    desc = MemoryLayoutDesc(shapes=[torch.Size([2, 2, 8, 16])], dtypes=[torch.float32])
    assert desc.fmt is MemoryFormat.KV_2LTD


def test_memory_layout_desc_fmt_survives_the_wire() -> None:
    desc = MemoryLayoutDesc(
        shapes=[torch.Size([2, 2, 8, 16])],
        dtypes=[torch.float32],
        fmt=MemoryFormat.KV_2LHTD,
    )
    decoded = msgspec_decode(msgspec_encode(desc, MemoryLayoutDesc), MemoryLayoutDesc)
    assert decoded.fmt is MemoryFormat.KV_2LHTD
    assert decoded.shapes == desc.shapes


def test_register_payload_defaults_to_token_major() -> None:
    """An older worker that omits the field is a token-major worker."""
    payload = _default_register_payload()
    assert payload.chunk_format == "KV_2LTD"
    decoded = msgspec_decode(
        msgspec_encode(payload, RegisterEngineDrivenContextPayload),
        RegisterEngineDrivenContextPayload,
    )
    assert decoded.chunk_format == "KV_2LTD"


# ---------------------------------------------------------------------------
# Server side: registration stamps the format, retrieve enforces it
# ---------------------------------------------------------------------------


def _head_major_payload(instance_id: int = 1) -> RegisterEngineDrivenContextPayload:
    base = _default_register_payload(instance_id=instance_id)
    return RegisterEngineDrivenContextPayload(
        instance_id=base.instance_id,
        model_name=base.model_name,
        world_size=base.world_size,
        block_size=base.block_size,
        num_layers=base.num_layers,
        hidden_dim_size=base.hidden_dim_size,
        dtype_str=base.dtype_str,
        use_mla=base.use_mla,
        chunk_format="KV_2LHTD",
    )


def test_server_register_stamps_chunk_format_on_layout(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    module, _, _, ctx = server_module_factory(chunk_size=16)
    module.register_kv_cache_engine_driven_context(_head_major_payload(instance_id=1))
    layout = ctx.layout_desc_registry.find("m", 1)
    assert layout is not None
    assert layout.fmt is MemoryFormat.KV_2LHTD
    # The shape is the same buffer the token-major layout uses.
    assert layout.shapes[0] == torch.Size([2, 2, 16, 16])


def test_server_register_rejects_unknown_chunk_format(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    module, _, _, _ = server_module_factory()
    base = _default_register_payload()
    bad = RegisterEngineDrivenContextPayload(
        instance_id=base.instance_id,
        model_name=base.model_name,
        world_size=base.world_size,
        block_size=base.block_size,
        num_layers=base.num_layers,
        hidden_dim_size=base.hidden_dim_size,
        dtype_str=base.dtype_str,
        use_mla=base.use_mla,
        chunk_format="KV_T2D",
    )
    with pytest.raises(ValueError, match="Unsupported chunk format"):
        module.register_kv_cache_engine_driven_context(bad)


def test_server_register_rejects_head_major_for_mla(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    module, _, _, _ = server_module_factory()
    base = _default_register_payload()
    mla = RegisterEngineDrivenContextPayload(
        instance_id=base.instance_id,
        model_name=base.model_name,
        world_size=base.world_size,
        block_size=base.block_size,
        num_layers=base.num_layers,
        hidden_dim_size=base.hidden_dim_size,
        dtype_str=base.dtype_str,
        use_mla=True,
        chunk_format="KV_2LHTD",
    )
    with pytest.raises(ValueError, match="split K/V"):
        module.register_kv_cache_engine_driven_context(mla)


def test_server_reserve_write_carries_chunk_format(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    """The layout handed to the storage manager is what allocation stamps."""
    mock_storage = MagicMock()
    mock_storage.reserve_write.return_value = {}
    mock_session = MagicMock()
    mock_session.get_hashes.return_value = [b"h"]
    module, _, _, _ = server_module_factory(
        mock_storage=mock_storage, mock_session=mock_session
    )
    module.register_kv_cache_engine_driven_context(_head_major_payload(instance_id=2))
    module.commit_store(_default_key(), 2, pickle.dumps([torch.ones(2, 2, 8, 16)]))
    layout_desc = mock_storage.reserve_write.call_args.args[1]
    assert layout_desc.fmt is MemoryFormat.KV_2LHTD


@pytest.mark.parametrize(
    ("stored_fmt", "worker_payload", "expect_success"),
    [
        (MemoryFormat.KV_2LHTD, _head_major_payload, True),
        (MemoryFormat.KV_2LTD, _head_major_payload, False),
        (MemoryFormat.KV_2LHTD, _default_register_payload, False),
        (MemoryFormat.KV_2LTD, _default_register_payload, True),
    ],
)
def test_server_pickle_retrieve_enforces_chunk_format(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
    stored_fmt: MemoryFormat,
    worker_payload: Any,
    expect_success: bool,
) -> None:
    """A chunk is handed back only under the format it was written with."""
    mock_storage = MagicMock()
    mock_memory_obj = MagicMock()
    mock_memory_obj.tensor = torch.ones(2, 2, 8, 16)
    mock_memory_obj.metadata.fmt = stored_fmt

    @contextmanager
    def _read_prefetched_results(_keys: Any) -> Any:
        yield [mock_memory_obj]

    mock_storage.read_prefetched_results.side_effect = _read_prefetched_results
    mock_session = MagicMock()
    mock_session.get_hashes.return_value = [b"h"]
    module, _, _, _ = server_module_factory(
        mock_storage=mock_storage, mock_session=mock_session
    )
    module.register_kv_cache_engine_driven_context(worker_payload(instance_id=3))

    response = module.prepare_retrieve(_default_key(), 3)
    assert response.success is expect_success
    if not expect_success:
        assert response.data == b""


def test_server_shm_retrieve_enforces_chunk_format(
    stub_lmcache_native: Any,
    server_module_factory: ServerModuleFactory,
) -> None:
    # Local
    from .test_engine_driven_transfer import _make_storage_manager_config

    mock_storage = MagicMock()
    mock_memory_obj = MagicMock()
    mock_memory_obj.tensor = torch.zeros(2, 2, 8, 16)
    mock_memory_obj.shm_offset = 0
    mock_memory_obj.shm_byte_length = 2048
    mock_memory_obj.metadata.fmt = MemoryFormat.KV_2LTD
    mock_storage.unsafe_read.side_effect = lambda obj_keys: (
        obj_keys,
        [mock_memory_obj for _ in obj_keys],
    )
    mock_session = MagicMock()
    mock_session.get_hashes.return_value = [b"h"]
    module, _, _, _ = server_module_factory(
        storage_manager_config=_make_storage_manager_config(
            shm_name="lmcache_test_fmt_pool", pool_size=4096
        ),
        mock_storage=mock_storage,
        mock_session=mock_session,
    )
    module.register_kv_cache_engine_driven_context(_head_major_payload(instance_id=4))

    response = module.prepare_retrieve(_default_key(), 4)
    assert response.success is False
    # The refused read must not leave the prefetch lock held.
    mock_storage.finish_read_prefetched.assert_called_once()


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------


def test_worker_context_defaults_to_token_major() -> None:
    assert EngineDrivenTransferContext().chunk_format is MemoryFormat.KV_2LTD


def test_create_transfer_context_passes_chunk_format_to_engine_driven() -> None:
    kv = {"layer_0": torch.randn(2, 4, 2, 4, 8)}
    context = create_transfer_context(
        kv, mode="engine_driven", chunk_format=MemoryFormat.KV_2LHTD
    )
    assert isinstance(context, EngineDrivenTransferContext)
    assert context.chunk_format is MemoryFormat.KV_2LHTD


def test_create_transfer_context_refuses_head_major_on_lmcache_driven() -> None:
    kv = {"layer_0": torch.randn(2, 4, 2, 4, 8)}
    with pytest.raises(ValueError, match="engine-driven"):
        create_transfer_context(
            kv, mode="lmcache_driven", chunk_format=MemoryFormat.KV_2LHTD
        )


def test_worker_register_refuses_head_major_for_nhd_layout() -> None:
    """The mismatch is caught before anything is sent to the server."""
    context = EngineDrivenTransferContext(chunk_format=MemoryFormat.KV_2LHTD)
    send_request = MagicMock()
    with pytest.raises(ValueError, match="head-major"):
        context.register(
            instance_id=1,
            kv_caches=_make_kv_caches(),
            model_name="m",
            world_size=1,
            blocks_in_chunk=2,
            mq_client=MagicMock(),
            mq_timeout=1.0,
            send_request=send_request,
        )
    send_request.assert_not_called()


def test_worker_register_announces_chunk_format() -> None:
    context = EngineDrivenTransferContext(chunk_format=MemoryFormat.KV_2LHTD)
    future = MagicMock()
    future.result.return_value = None
    send_request = MagicMock(return_value=future)
    context.register(
        instance_id=1,
        kv_caches=_make_hnd_kv_caches(),
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        mq_client=MagicMock(),
        mq_timeout=1.0,
        send_request=send_request,
        layout_hints=HND_HINTS,
    )
    payload = send_request.call_args.args[2][0]
    assert payload.chunk_format == "KV_2LHTD"
    assert context.engine_driven_context.layout_desc.fmt is MemoryFormat.KV_2LHTD


# ---------------------------------------------------------------------------
# Gather / scatter under KV_2LHTD
# ---------------------------------------------------------------------------


@pytest.fixture
def torch_device_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the gather/scatter utilities against the torch baseline backend.

    ``gather_paged_kv_to_cpu`` resolves ``lmcache.device_ops`` at call time.
    The layout contract under test is device-independent, so pin the torch
    implementation rather than whatever accelerator this host detected.
    """
    monkeypatch.setattr(lmcache, "device_ops", DeviceOps())


def test_gather_scatter_head_major_roundtrip_and_layout(
    torch_device_ops: None,
) -> None:
    source = _make_hnd_kv_caches(2, 8, 4, 2, 8)
    (_bs, _nl, _hd, _dt, fmt, _kv) = compute_kv_layout(source, layout_hints=HND_HINTS)
    assert fmt == EngineKVFormat.NL_X_TWO_NB_NH_BS_HS
    blocks_per_chunk = 2
    head_major = gather_paged_kv_to_cpu(
        source,
        [0, 1],
        blocks_per_chunk,
        layout_hints=HND_HINTS,
        engine_kv_format=fmt,
        chunk_format=MemoryFormat.KV_2LHTD,
    )
    token_major = gather_paged_kv_to_cpu(
        source, [0, 1], blocks_per_chunk, layout_hints=HND_HINTS, engine_kv_format=fmt
    )
    # Same buffer shape, head-major bytes == token-major bytes with (T, H) swapped.
    assert head_major[0].shape == token_major[0].shape
    expected = token_major[0].unflatten(-1, (2, 8)).permute(0, 1, 3, 2, 4)
    assert torch.equal(head_major_chunk_view(head_major[0], 2, 8), expected)

    destination = {k: torch.zeros_like(v) for k, v in source.items()}
    scatter_cpu_to_paged_kv(
        destination,
        [4, 5],
        head_major,
        blocks_per_chunk,
        layout_hints=HND_HINTS,
        engine_kv_format=fmt,
        chunk_format=MemoryFormat.KV_2LHTD,
    )
    for name in source:
        assert torch.equal(source[name][:, 0], destination[name][:, 4])
        assert torch.equal(source[name][:, 1], destination[name][:, 5])


def test_gather_head_major_refuses_nhd_layout(torch_device_ops: None) -> None:
    source = _make_kv_caches()
    with pytest.raises(ValueError, match="head-major"):
        gather_paged_kv_to_cpu(source, [0, 1], 2, chunk_format=MemoryFormat.KV_2LHTD)


# ---------------------------------------------------------------------------
# vLLM adapter configuration
# ---------------------------------------------------------------------------


def test_adapter_extra_config_defaults_to_token_major() -> None:
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        ExtraConfigDefault,
        _resolve_extra_config,
    )

    cfg = _resolve_extra_config(None)
    assert parse_chunk_format(cfg[ExtraConfigDefault.chunk_format.name]) is (
        MemoryFormat.KV_2LTD
    )


def test_adapter_extra_config_selects_head_major() -> None:
    # First Party
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        ExtraConfigDefault,
        _resolve_extra_config,
    )

    cfg = _resolve_extra_config({"lmcache.mp.chunk_format": "KV_2LHTD"})
    assert parse_chunk_format(cfg[ExtraConfigDefault.chunk_format.name]) is (
        MemoryFormat.KV_2LHTD
    )
