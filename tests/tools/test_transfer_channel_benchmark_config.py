# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the transfer-channel benchmark's torch-free CLI config."""

# Standard
import argparse

# Third Party
import pytest

# First Party
from lmcache.tools.transfer_channel_benchmark.config import (
    BenchmarkConfig,
    add_benchmark_arguments,
    build_config,
)


def _build(argv: list[str]) -> BenchmarkConfig:
    parser = argparse.ArgumentParser()
    add_benchmark_arguments(parser)
    return build_config(parser.parse_args(["--role", "client", *argv]))


def test_nixl_backend_defaults_to_ucx_without_params():
    cfg = _build([])
    assert cfg.nixl_backend == "UCX"
    assert cfg.nixl_backend_params == {}


def test_nixl_backend_params_are_parsed():
    cfg = _build(
        [
            "--nixl-backend",
            "RBLN",
            "--nixl-backend-params",
            '{"listen_ip": "10.0.0.2"}',
        ]
    )
    assert cfg.nixl_backend == "RBLN"
    assert cfg.nixl_backend_params == {"listen_ip": "10.0.0.2"}


@pytest.mark.parametrize("raw", ["{", '["listen_ip"]', '{"listen_port": 26000}'])
def test_nixl_backend_params_must_be_json_object_of_strings(raw):
    with pytest.raises(ValueError, match="nixl.backend.params"):
        _build(["--nixl-backend-params", raw])
