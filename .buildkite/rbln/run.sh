#!/usr/bin/env bash
# Three upstream test scopes, using the same pre-provisioned RBLN image.
set -euo pipefail

LANE="${1:?Usage: run.sh hw-smoke|smoke|unit}"
case "${LANE}" in
    hw-smoke|smoke|unit) ;;
    *) echo "Unknown RBLN lane: ${LANE}" >&2; exit 2 ;;
esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="${SCRIPT_DIR}/../../rbln-ci-artifacts/${LANE}"
# shellcheck source=.buildkite/rbln/common-setup.sh
source "${SCRIPT_DIR}/common-setup.sh"

PYTEST_ARGS=(-q -rs --maxfail=1 --timeout=600 --durations=20
    "--junitxml=${ARTIFACT_DIR}/junit.xml")
if [[ "${LANE}" == hw-smoke ]]; then
    TEST_PATHS=(tests/v1/platform/devices/rbln
        tests/v1/gpu_connector/test_kv_format_detection.py)
else
    # Other vendors import their own runtimes during collection; marker
    # filtering alone cannot exclude those imports. RBLN has no Triton backend.
    PYTEST_ARGS+=(
        -m "not cuda and not xpu and not musa and not npu and not neuron and not sglang"
        --ignore=tests/v1/platform/devices/cuda
        --ignore=tests/v1/platform/devices/xpu
        --ignore=tests/v1/platform/devices/musa
        --ignore=tests/v1/platform/devices/npu
        --ignore=tests/v1/platform/devices/neuron
        --ignore=tests/v1/compute/attention/test_triton_kernels.py
    )
    if [[ "${LANE}" == smoke ]]; then
        TEST_PATHS=(tests/v1/compute tests/v1/platform tests/v1/lmcache_native
            tests/v1/gpu_connector tests/v1/cache_controller
            tests/v1/internal_api_server tests/v1/shm_allocator
            tests/v1/lookup_client tests/v1/plugin tests/v1/cli)
    else
        TEST_PATHS=(tests)
        # Match the vendor CI exclusions for CUDA kernels, NIXL/EIC services,
        # multi-node disaggregation, and the non-unit performance harness.
        PYTEST_ARGS+=(
            --ignore=tests/disagg
            --ignore=tests/skipped
            --ignore=tests/benchmarks
            --ignore=tests/v1/test_pos_kernels.py
            --ignore=tests/v1/test_device_id_race.py
            --ignore=tests/v1/test_nixl_batched_contains.py
            --ignore=tests/v1/test_nixl_multipath.py
            --ignore=tests/v1/storage_backend/test_eic.py
        )
    fi
fi

echo "--- Run RBLN ${LANE}"
python -m pytest "${PYTEST_ARGS[@]}" "${TEST_PATHS[@]}" \
    2>&1 | tee "${ARTIFACT_DIR}/pytest.log"

if [[ "${LANE}" == hw-smoke ]]; then
    # A later runtime-discovery failure must not turn hardware coverage into skips.
    python - "${ARTIFACT_DIR}/junit.xml" <<'CHECK'
import sys
import xml.etree.ElementTree as ET

cases = [
    case for case in ET.parse(sys.argv[1]).getroot().iter("testcase")
    if case.get("classname", "").endswith("test_rbln_hw_block_transfer")
]
if not cases or any(case.find("skipped") is not None for case in cases):
    raise SystemExit("RBLN hardware tests were missing or skipped")
CHECK
    bash "${SCRIPT_DIR}/server-smoke.sh" "${ARTIFACT_DIR}"
fi
