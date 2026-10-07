#!/usr/bin/env bash
# A fresh Kubernetes Pod gives each lane its own loopback ports.
set -euo pipefail
ARTIFACT_DIR="${1:?Usage: server-smoke.sh artifact-directory}"
SERVER_LOG="${ARTIFACT_DIR}/lmcache-server.log"
SERVER_PID=""

wait_for_exit() {
    local seconds="$1" elapsed
    for ((elapsed = 0; elapsed < seconds; elapsed++)); do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            wait "${SERVER_PID}"
            return $?
        fi
        sleep 1
    done
    return 124
}

cleanup() {
    local status=$?
    trap - EXIT
    set +e
    if [[ -n "${SERVER_PID}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null
        wait_for_exit 10
        if [[ $? -eq 124 ]]; then
            kill -KILL "${SERVER_PID}" 2>/dev/null
            wait "${SERVER_PID}" 2>/dev/null
        fi
    fi
    if [[ "${status}" -ne 0 ]]; then
        tail -n 200 "${SERVER_LOG}" >&2
    fi
    exit "${status}"
}
trap cleanup EXIT

echo "--- Start the RBLN multiprocess server"
LMCACHE_DEVICE_BACKEND=rbln lmcache server \
    --host 127.0.0.1 --port 6555 \
    --http-host 127.0.0.1 --http-port 7555 \
    --l1-size-gb 0.25 --no-l1-use-lazy --eviction-policy LRU \
    --chunk-size 128 --disable-metrics > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

healthy=0
for ((attempt = 0; attempt < 120; attempt++)); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "LMCache server exited before becoming healthy" >&2
        exit 1
    fi
    if curl --max-time 2 -fsS http://127.0.0.1:7555/healthcheck >/dev/null 2>&1; then
        healthy=1
        break
    fi
    sleep 1
done
if [[ "${healthy}" -ne 1 ]]; then
    echo "LMCache server did not become healthy" >&2
    exit 1
fi

kill "${SERVER_PID}"
status=0
wait_for_exit 30 || status=$?
if [[ "${status}" -eq 124 ]]; then
    echo "LMCache server did not stop within 30 seconds" >&2
    exit 1
fi
SERVER_PID=""
if [[ "${status}" -ne 0 && "${status}" -ne 143 ]]; then
    echo "LMCache server exited with status ${status}" >&2
    exit 1
fi
grep -q 'LMCache HTTP server stopped' "${SERVER_LOG}"
echo "RBLN server startup and shutdown passed"
