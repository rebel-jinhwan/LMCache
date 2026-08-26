# RBLN Benchmarks

RBLN's supported transfer path is multiprocess engine-driven store/retrieve.
Synthetic `.to("rbln:0")` tensors land in host SHM and run on CPU, so this
bench boots a real vLLM `LLM` and times the transfer **inside the worker** on
`worker.model_runner.kv_caches`.

`bench_kv_transfer_mp.py` follows `lmcache-rbln/benchmark/bench_kv_transfer_mp.py`:
`LLM.collective_rpc` calls `gather_paged_kv_to_cpu` / `scatter_cpu_to_paged_kv`
on the native 6-D DRAM KV. Each sweep cell is timed twice — unbound
`RblnDeviceOps` (torch) vs `ensure_native()` (`lmcache.rbln_ops`).

Run on a Rebellions RBLN host with `torch-rbln`, vLLM-RBLN, and LMCache built
with the RBLN profile (`BUILD_WITH_RBLN=1` or auto-detect when `torch_rbln` is
installed). Pin the NPU with ``RBLN_DEVICES`` (the ``CUDA_VISIBLE_DEVICES``
analogue); on a shared box, omitting it makes torch-rbln register every
device and abort on occupied ones:

```bash
python -c "import torch, torch_rbln; print(torch.rbln.is_available())"
pytest -q tests/v1/platform/rbln tests/test_build_profiles_rbln.py tests/benchmarks/test_rbln_mp_benchmark.py -rs
RBLN_DEVICES=0 python benchmarks/rbln/bench_kv_transfer_mp.py \
    --sweep-blocks 1,2,4 --warmup 5 --iters 10 --min-speedup 1.2 --out /tmp/kv_transfer.json
```

The LLM boot (`Qwen/Qwen3-Coder-30B-A3B-Instruct`, `block_size=1024`,
`max_model_len=8192`, …) is hardcoded in `_LLM_KWARGS`; edit that dict for a
smaller compile. Require `block_size < max_model_len`; `--blocks-per-chunk 1`
(default) means `chunk_size == block_size`, matching the serve. The model
compiles on first run. Latency is mean±std over the timed iters (after
warmup). The run passes when native is at least `--min-speedup` faster than
torch on every gather and scatter cell (`--min-speedup 0` disables the gate).
For review, attach the full command, rebel / `torch-rbln` / vLLM-RBLN
versions, and the printed tables.
