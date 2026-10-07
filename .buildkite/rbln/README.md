# RBLN Buildkite CI

Run upstream LMCache on the Rebellions Kubernetes workers. This ports the
three lane scopes from [lmcache-rbln#217](https://github.com/rebellions-sw/lmcache-rbln/pull/217)
without installing its downstream monkey patches or test overlays.

| Lane | Builds | Checks |
| --- | --- | --- |
| `hw-smoke` | All | Runtime/device discovery, RBLN platform tests, real DRAM HND/MLA block transfers, MP server health and shutdown |
| `smoke` | PR | MetaX-style compute, platform, native, connector and API subset |
| `unit` | Non-PR, including manual and scheduled | Whole `tests/` tree with the exclusions listed in `run.sh` |

Each step requests one NPU through the `npu1` ResourceClaimTemplate. All lanes
share a Buildkite concurrency group of size one, including across builds.
Kubernetes DRA controls device visibility; the pipeline does not override
physical device IDs with `RBLN_DEVICES=0`. Logical device `rbln:0` is used by tests.
`TORCH_RBLN_EAGER_MALLOC=1` makes device tensors reside in DRAM, rather than
letting small copies pass through host SHM. Missing hardware or dummy mode fails
the preflight before pytest can skip the real-device tests.

## Infrastructure setup

1. Configure the Buildkite Kubernetes controller to watch `rbln-queue`, and make
   the `npu1` ResourceClaimTemplate available in its job namespace.
2. Allow the Rebellions worker nodes to pull
   `ghcr.io/rebellions-sw/rbln-test:latest`. If the package is private, configure
   GHCR authentication on the job service account (`imagePullSecrets`) or
   controller `pod-spec-patch`. Keep pull credentials in Kubernetes; the image
   address alone does not grant access or restrict pulls to these workers.
3. The test image defaults to `ghcr.io/rebellions-sw/rbln-test:latest`.
   Optionally set `RBLN_CI_IMAGE` in the pipeline environment to select a specific
   tag, digest, or internal mirror. `imagePullPolicy: Always` checks the registry
   on each Pod start so updates to `latest` are used. The queue routes image pulls
   and test execution to the Rebellions workers.
4. Paste `buildkite-pipeline.yml` into the pipeline's Steps editor. The upload
   job uses the controller's default image and needs Bash, Git and Buildkite
   agent, but no NPU. It uses the shared PR-base merge/path-filter helper.
5. Enable PR builds and push builds for `dev` in Buildkite. Add a schedule there
   if nightly runs are wanted. The YAML selects lanes; it does not create GitHub
   webhooks or schedules. Give operators Build & Read access to start builds.

If the controller actually watches `rbln-npu`, change the bootstrap queue in
`buildkite-pipeline.yml` and set `RBLN_CI_QUEUE=rbln-npu` in the pipeline
environment. Both upload and test jobs must reach the same infrastructure.

For example, to select the image explicitly:

```text
RBLN_CI_IMAGE=ghcr.io/rebellions-sw/rbln-test:latest
RBLN_CI_QUEUE=rbln-queue
```

See Buildkite's [PodSpec](https://buildkite.com/docs/agent/self-hosted/agent-stack-k8s/podspec)
and [custom image](https://buildkite.com/docs/agent/self-hosted/agent-stack-k8s/custom-images)
documentation for the controller configuration.

## Image contract

The selected image must be compatible with the worker's driver and contain:

- Python 3.12 with `pip`; compatible pinned `torch`, `torch-rbln` and
  `rebel-compiler`, with automatic `torch.rbln` registration.
- LMCache's `requirements/build.txt`, `requirements/common.txt`, and
  `requirements/test.txt`, plus `pytest-timeout`, installed **while preserving
  those RBLN stack pins**. Keep the image dependencies in sync with `dev`.
- Bash, Git, curl, `rbln-stat`, a C++ compiler and Python development headers.
  The container user must be able to build and install into the active Python.

Use a clean runtime without `lmcache-rbln` import hooks. The lanes also set
`LMCACHE_RBLN_UPSTREAM_PATCHES=0` so upstream behavior is what gets tested.
Each job builds the checked-out source with `NO_GPU_EXT=1`, `--no-deps` and
`--no-build-isolation`, preserving the image's runtime and avoiding private
package-index credentials in jobs. Common native extensions are required;
`NO_NATIVE_EXT=1` cannot run these tests. The local SCM version override is for
CI builds only. gRPC bindings are generated after the editable install.

No host Docker socket, privileged container or host filesystem mount is needed.
Each Pod gets an 8 GiB `/dev/shm` mount within a 32 GiB memory limit.

## Validation and results

Start a build against the branch containing these files. Inspect the hardware
smoke step first: checkout, image pull, DRA allocation, source build, real-device
tests, and server health must all succeed. A PR build also runs `smoke`; a manual
non-PR build runs `unit`. All results are strict failures, with no soft-fail rule.

Artifacts under `rbln-ci-artifacts/<lane>/` include the tested commit, installed
versions, runtime probe, pytest log and JUnit report; hardware smoke also saves
the MP server log. The health check proves startup/shutdown, not model inference
or end-to-end vLLM KV reuse. No model downloads or RDMA are required.

Inside an already provisioned container with an allocated NPU:

```bash
bash .buildkite/rbln/run.sh hw-smoke
bash .buildkite/rbln/run.sh smoke
bash .buildkite/rbln/run.sh unit
```

The broad lanes deliberately keep upstream's non-native-layout transfer tests.
At introduction, upstream RBLN rejects those layouts; #217 passed them using a
downstream fallback patch. Those failures must be fixed separately in upstream,
not masked by this CI. The blocks-first layout overlay from #217 is likewise
outside this PR. Unlike that incubation workflow, this lane does not suppress
turboquant tests for older drivers or the occasionally flaky multi-client MQ
test; failures remain visible in the full suite.
