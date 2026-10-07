#!/usr/bin/env bash
# Source inside an ephemeral CI container whose RBLN stack is already installed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
# Each test Pod has its own checkout, separate from the pipeline upload job.
# shellcheck source=.buildkite/k3_tests/common_scripts/helpers.sh
source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"
merge_pr_base_branch
mkdir -p "${ARTIFACT_DIR}"
export LMCACHE_TRACK_USAGE=false
export TORCH_RBLN_EAGER_MALLOC=1
export LMCACHE_RBLN_UPSTREAM_PATCHES=0
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"

# Use the image's pinned torch/RBLN stack and build tools without resolving
# dependencies against public indexes or installing downstream patch overlays.
echo "--- Build the checked-out LMCache with common native extensions"
NO_GPU_EXT=1 SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.0.0+rbln.ci \
    python -m pip install --no-deps --no-build-isolation -e .
python lmcache/v1/multiprocess/transport/grpc_impl/_proto_gen/_generate.py
python -m pip freeze > "${ARTIFACT_DIR}/pip-freeze.txt"
git rev-parse HEAD > "${ARTIFACT_DIR}/lmcache-commit.txt"

# Failure here must fail the lane, rather than let hardware tests skip green.
echo "--- Check the allocated NPU and LMCache backend"
rbln-stat 2>&1 | tee "${ARTIFACT_DIR}/rbln-stat.txt"
python - <<'PY' 2>&1 | tee "${ARTIFACT_DIR}/runtime-preflight.txt"
import importlib.metadata
import os

import torch

import lmcache

if not hasattr(torch, "rbln") or not torch.rbln.is_available():
    raise RuntimeError("No usable RBLN NPU was allocated to this container")
if torch.rbln.is_dummy_device():
    raise RuntimeError("The RBLN CI lanes require real hardware, not dummy mode")
if lmcache.torch_device_type != "rbln":
    raise RuntimeError(f"LMCache selected {lmcache.torch_device_type!r}, expected rbln")

for package in ("torch", "torch-rbln", "rebel-compiler", "lmcache"):
    print(f"{package}={importlib.metadata.version(package)}")
print(f"lmcache_source={lmcache.__file__}")
print(f"RBLN_DEVICES={os.environ.get('RBLN_DEVICES', '<DRA managed>')}")
print(f"rbln_device_count={torch.rbln.device_count()}")
host = torch.arange(24, dtype=torch.float16).reshape(2, 3, 4)
device = host.to("rbln:0")
torch.rbln.synchronize()
if not torch.equal(device.cpu(), host):
    raise RuntimeError("RBLN host/device copy did not preserve the input")
print("rbln_probe=ok")
PY
