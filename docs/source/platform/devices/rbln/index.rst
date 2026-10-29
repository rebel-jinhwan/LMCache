.. SPDX-License-Identifier: Apache-2.0

Rebellions NPU
==============

The ``rbln`` backend runs LMCache on Rebellions NPUs, with
`vLLM-RBLN <https://github.com/RBLN-SW/vllm-rbln>`_ as the serving engine. It
implements the community ``DeviceSpec`` and ``DeviceOps`` interfaces for
multiprocess (MP) **engine-driven** transfers: on store, the vLLM-RBLN worker
gathers KV blocks from its paged cache on the NPU into the LMCache server's
host memory, and on retrieve it scatters them back. KV moves with
``torch.rbln`` tensor operations.

With a ``torch-rbln`` that provides ``torch.rbln.ipc``, the backend also
supports **LMCache-driven** transfers as an opt-in mode: the LMCache server
opens the worker's KV cache on the same NPU and moves blocks itself. See
`LMCache-driven transfer (opt-in)`_.

Prerequisites
-------------

Install the RBLN driver and the matching runtime stack (``torch-rbln`` and
``rebel-compiler``) by following the
`RBLN SDK documentation <https://docs.rbln.ai>`_, then install ``vllm-rbln``.
Do not let the LMCache installation replace the vendor's PyTorch.

Build LMCache's common C++ extensions against that PyTorch installation. RBLN
adds no native extension; the build selects the RBLN profile when
``torch_rbln`` is installed, and ``BUILD_WITH_RBLN=1`` forces it:

.. code-block:: bash

   BUILD_WITH_RBLN=1 pip install -e . --no-build-isolation

Use an environment with the repository's build dependencies installed.

Running vLLM-RBLN with LMCache
------------------------------

As on GPUs, start an LMCache server and point vLLM-RBLN at it with
``LMCacheMPConnector``. With ``N`` as the vLLM block size:

.. code-block:: bash

   lmcache server --port 6555 --l1-size-gb 32 --eviction-policy LRU \
       --chunk-size N --supported-transfer-mode engine_driven

   vllm serve <model> \
       --model-impl vllm \
       --block-size N \
       --kv-transfer-config '{
           "kv_connector": "LMCacheMPConnector",
           "kv_role": "kv_both",
           "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
           "kv_connector_extra_config": {
               "lmcache.mp.host": "tcp://127.0.0.1",
               "lmcache.mp.port": 6555
           }
       }'

Three settings are specific to RBLN:

* ``--supported-transfer-mode engine_driven`` on the server. The server
  defaults to ``lmcache_driven``, which does not register the engine-driven
  handlers the RBLN worker calls. ``auto`` also works, as it enables both
  paths.
* ``--model-impl vllm``. KV connectors work only with vLLM's model
  implementation. With the default ``auto``, vLLM-RBLN can choose a model
  path that never hands its KV caches to a connector.
* The LMCache chunk size must be a positive multiple of the vLLM block size,
  because the backend moves whole blocks, so set ``--chunk-size``
  explicitly. Setting the block size equal to the chunk size is recommended.

LMCache detects ``torch.rbln`` and selects ``rbln`` automatically when an NPU
can be opened. To pin the backend, set:

.. code-block:: bash

   export LMCACHE_DEVICE_BACKEND=rbln

Pinning is recommended. Without it, a process that cannot open an NPU (for
example, because another process holds the device) falls back to the CPU stub
without an error. With it, startup fails instead.

LMCache-driven transfer (opt-in)
--------------------------------

In LMCache-driven mode, the vLLM-RBLN worker exports its KV cache as dma-buf
device memory when it registers, and the LMCache server opens that memory on
the same NPU. On store and retrieve, the server copies blocks between the
opened KV cache and its host memory with DMA, so the worker no longer moves KV
itself. ``auto`` never selects this mode on RBLN, because it depends on how the
engine allocates its KV cache, which LMCache cannot check.

Requirements:

* A ``torch-rbln`` and ``rebel-compiler`` pair that provides
  ``torch.rbln.ipc``. Check it with:

  .. code-block:: bash

     python -c "import torch; print(hasattr(torch.rbln, 'ipc'))"

* A vLLM-RBLN with ``--rbln-use-exportable-kv-cache``. The flag allocates the
  KV cache with ``torch.rbln.ipc.empty_exportable``, the only memory
  ``torch.rbln.ipc.export_tensor`` accepts. It also refuses to start unless
  vLLM-RBLN's dynamic-KV resize is active, because only the resize registers
  the KV cache after the compiled model has placed it, and an export pins the
  memory's layout at that moment. The flag is on the
  `feat/exportable-kv-cache <https://github.com/RBLN-SW/vllm-rbln/tree/feat/exportable-kv-cache>`_
  branch and not merged yet. Install that branch into the environment that
  already holds the RBLN stack and the vLLM version the branch requires;
  ``--no-deps`` keeps the installed ``torch-rbln`` and ``rebel-compiler``:

  .. code-block:: bash

     git clone -b feat/exportable-kv-cache https://github.com/RBLN-SW/vllm-rbln.git
     pip install --no-deps -e ./vllm-rbln

* The worker and the server must run on the same host, as the same user, and
  in the same network namespace. The worker hands the dma-buf file
  descriptors to the server over an abstract Unix socket.

Testing with ``vllm bench serve``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following steps start the LMCache server, vLLM-RBLN, and a benchmark
client, each in its own terminal. They use ``Qwen/Qwen3-0.6B`` and a block
size of 1024.

**Terminal 1: LMCache server.** Start the server in ``lmcache_driven`` mode,
with the chunk size equal to the vLLM block size:

.. code-block:: bash

   export LMCACHE_DEVICE_BACKEND=rbln

   lmcache server \
       --host 127.0.0.1 \
       --port 6555 \
       --l1-size-gb 32 \
       --no-l1-use-lazy \
       --eviction-policy LRU \
       --chunk-size 1024 \
       --supported-transfer-mode lmcache_driven

The HTTP endpoint for health checks and metrics listens on
``http://127.0.0.1:8080``. ``--no-l1-use-lazy`` is needed because lazy L1
allocation requires pinned host memory, which RBLN does not support.

**Terminal 2: vLLM-RBLN.** Pass ``--rbln-use-exportable-kv-cache`` and set
``lmcache.mp.mp_transfer_mode`` to ``lmcache_driven``:

.. code-block:: bash

   export LMCACHE_DEVICE_BACKEND=rbln

   vllm serve Qwen/Qwen3-0.6B \
       --model-impl vllm \
       --rbln-use-exportable-kv-cache \
       --max-model-len 4096 \
       --block-size 1024 \
       --max-num-seqs 4 \
       --max-num-batched-tokens 1024 \
       --no-enable-prefix-caching \
       --kv-transfer-config '{
           "kv_connector": "LMCacheMPConnector",
           "kv_role": "kv_both",
           "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
           "kv_connector_extra_config": {
               "lmcache.mp.host": "tcp://127.0.0.1",
               "lmcache.mp.port": 6555,
               "lmcache.mp.mp_transfer_mode": "lmcache_driven"
           }
       }'

``--no-enable-prefix-caching`` turns off vLLM's own prefix cache, so a repeated
prompt can skip prefill only through LMCache. Wait until the server answers:

.. code-block:: bash

   curl -s http://127.0.0.1:8000/v1/models

**Terminal 3: benchmark.** Run the same random workload twice. The first run
stores every full chunk into LMCache. The second run uses the same ``--seed``,
so it sends the same prompts and retrieves their KV from LMCache:

.. code-block:: bash

   # Run 1: cold, stores KV into LMCache.
   vllm bench serve \
       --backend vllm \
       --model Qwen/Qwen3-0.6B \
       --host 127.0.0.1 \
       --port 8000 \
       --dataset-name random \
       --random-input-len 3072 \
       --random-output-len 32 \
       --num-prompts 32 \
       --max-concurrency 4 \
       --seed 0

   # Run 2: warm, the same prompts hit LMCache.
   vllm bench serve \
       --backend vllm \
       --model Qwen/Qwen3-0.6B \
       --host 127.0.0.1 \
       --port 8000 \
       --dataset-name random \
       --random-input-len 3072 \
       --random-output-len 32 \
       --num-prompts 32 \
       --max-concurrency 4 \
       --seed 0

**Checking the result.** The LMCache server log of terminal 1 confirms the
transfer path. The KV cache registers through the LMCache-driven module, and
stores and retrieves report their time:

.. code-block:: text

   LMCache INFO: Registered KV cache for GPU ID ... with 28 layers (lmcache_driven_transfer.py:...)
   LMCache INFO: Stored 1024 tokens in 0.019 seconds (lmcache_driven_transfer.py:...)
   LMCache INFO: Retrieved 3072 tokens in 0.052 seconds (lmcache_driven_transfer.py:...)

The server's chunk counters show what run 1 stored and run 2 read back:

.. code-block:: bash

   curl -s http://127.0.0.1:8080/metrics | grep -E '^lmcache_mp_l1_(write|read)_chunks_total'

With 32 prompts of 3072 tokens, both counters reach 96, three chunks per
prompt. Stores complete after the responses return, so the write counter can
lag run 1 by a few seconds.

To compare with engine-driven transfer, restart both servers with
``--supported-transfer-mode engine_driven``, set
``"lmcache.mp.mp_transfer_mode": "engine_driven"``, drop
``--rbln-use-exportable-kv-cache``, and run the same two benchmarks.

Example results on one RBLN-CR13 NPU with the commands above:

.. list-table::
   :header-rows: 1

   * - Mode
     - Server store, 1024 tokens
     - Server retrieve, 3072 tokens
     - Mean TTFT, run 1
     - Mean TTFT, run 2
   * - ``engine_driven``
     - 188 ms
     - 234 ms
     - 235 ms
     - 1293 ms
   * - ``lmcache_driven``
     - 16 ms
     - 50 ms
     - 128 ms
     - 141 ms

The server times are the mean of the ``Stored`` and ``Retrieved`` log lines.
Qwen3-0.6B prefills 3072 tokens about as fast as LMCache retrieves them, so
run 2 is not faster than run 1 here. This setup checks the transfer path and
compares the two modes. It does not show the TTFT benefit of a cache hit,
which needs a model whose prefill costs more than the retrieve.

Supported KV cache layouts
--------------------------

The backend accepts the per-layer KV cache layouts that vLLM-RBLN allocates:

* Blocks-first HND: ``[num_blocks, 2, num_kv_heads, 1, block_size, head_size]``.
* K/V-first HND: ``[2, num_blocks, num_kv_heads, 1, block_size, head_size]``.
* MLA: ``[num_blocks, block_size, head_size]``.

Any other layout raises ``ValueError``.

Capability limits
-----------------

LMCache-driven MP transfer is opt-in. On the worker, ``auto`` (the default)
resolves to ``engine_driven``. Selecting ``lmcache_driven`` fails at
validation when ``torch.rbln.ipc`` is missing. When it is present, LMCache
still cannot check the engine-side requirements listed in
`LMCache-driven transfer (opt-in)`_.

The server does not check the device. With ``--supported-transfer-mode
lmcache_driven`` (its default), it serves only LMCache-driven workers, and an
engine-driven RBLN worker needs ``engine_driven`` or ``auto``.

In LMCache-driven mode, each store and retrieve synchronizes the whole device
on both sides, because ``torch.rbln`` events cannot be shared across processes
yet.

Validation
----------

Run the backend tests on a host with an RBLN NPU:

.. code-block:: bash

   python -m pytest -q -rs tests/v1/platform/devices/rbln

Tests that need an NPU skip when none can be opened; skipped cases are not
hardware validation. The ``rbln`` Buildkite pipeline
(``.buildkite/k3_tests/rbln``) runs these tests and an end-to-end vLLM-RBLN
check on RBLN hardware.
