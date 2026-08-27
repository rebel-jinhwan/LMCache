# SPDX-License-Identifier: Apache-2.0
"""mp KV-transfer microbench for LMCache's RBLN path, on REAL vLLM device KV.

Boots a vLLM ``LLM`` on RBLN and times LMCache's engine-driven mp transfer
*inside the worker* (via ``collective_rpc`` on ``worker.model_runner.kv_caches``)
by calling ``gather_paged_kv_to_cpu`` / ``scatter_cpu_to_paged_kv`` on the real
native 6-D RBLN KV tensors. Synthetic ``.to("rbln:0")`` tensors land in host
SHM and run on CPU, so they do not match a serve.

Each sweep cell is timed twice: an unbound ``RblnDeviceOps`` (torch fallback)
and one that called ``ensure_native()`` (``lmcache.rbln_ops``).
"""

# Standard
import argparse
import json
import os

# Boot recipe mirrors lmcache-rbln/benchmark/bench_kv_transfer_mp.py — these
# MUST be set before vllm is imported (RblnPlatform reads them at
# class-definition time).
_BOOT_ENV = {
    "VLLM_RBLN_USE_VLLM_MODEL": "1",
    # Without this vLLM-RBLN keeps the worker KV on the ``meta`` device (its
    # default ``VLLM_RBLN_COMPILE_MODEL`` path) and nothing can be copied.
    "VLLM_RBLN_USE_DEVICE_TENSOR": "1",
    "TORCH_RBLN_DEPLOY": "ON",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    # collective_rpc rejects a Callable payload unless this is set (it falls
    # back to pickle to ship _bench_on_worker to the worker process).
    "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
}

# Fixed serve-shaped LLM so the worker allocates real 6-D DRAM KV. These are
# not bench knobs: change them here if a different compile is needed.
_LLM_KWARGS = {
    "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "max_model_len": 8192,
    "max_num_seqs": 4,
    "block_size": 1024,
    "gpu_memory_utilization": 0.6,
    "enable_prefix_caching": True,
    "enable_chunked_prefill": True,
    "max_num_batched_tokens": 128,
    "trust_remote_code": True,  # some serve models ship custom config/modeling
}


def _bench_on_worker(worker, cfg: dict) -> dict:
    """Runs on each vLLM worker; benches mp transfer on real KV.

    Self-contained (all imports inline) so it survives cloudpickle to the
    spawned worker. Returns a picklable dict of geometry + timings.
    """
    # Standard
    import statistics
    import time

    # Third Party
    import torch

    # First Party
    from lmcache.v1.multiprocess.transfer_context.base import (
        gather_paged_kv_to_cpu,
        scatter_cpu_to_paged_kv,
    )
    from lmcache.v1.platform.rbln.device_ops import RblnDeviceOps
    import lmcache

    mr = worker.model_runner
    raw = list(mr.kv_caches)
    if not raw:
        return {"error": "worker.model_runner.kv_caches is empty"}

    names = list(getattr(mr, "kv_cache_names", None) or [])
    if len(names) != len(raw):
        names = [f"layer.{i}" for i in range(len(raw))]
    # Native 6-D RBLN KV ([2, NB, NH, 1, BS, HD]) is passed as-is: LMCache's
    # vLLM format detector recognises it directly (no HND squeeze / layout
    # hint needed), the same tensors the mp connector registers.
    kv = dict(zip(names, raw, strict=True))

    l0 = raw[0]
    shp = tuple(l0.shape)
    nb, bs = (shp[1], shp[4]) if l0.dim() == 6 else (shp[1], shp[3])
    info = {
        "n_layers": len(raw),
        "raw_shape": shp,
        "num_blocks": nb,
        "block_size": bs,
        "dtype": str(l0.dtype),
        "device": str(l0.device),
        "data_ptr_hex": hex(l0.data_ptr()),
        "is_contiguous": bool(l0.is_contiguous()),
    }

    torch_ops = RblnDeviceOps()
    native_ops = RblnDeviceOps()
    native_ops.ensure_native()
    if "multi_layer_block_kv_transfer" not in vars(native_ops):
        info["error"] = (
            "lmcache.rbln_ops is not bound; install LMCache with "
            "BUILD_WITH_RBLN=1 so torch vs native can be compared"
        )
        return info

    blocks_per_chunk = max(1, int(cfg.get("blocks_per_chunk", 1)))
    warmup = int(cfg.get("warmup", 3))
    iters = int(cfg["iters"])

    # Reuse the destination chunks across iterations, as a serve reuses its
    # server-owned SHM slots, so the timing excludes malloc/first-touch. The
    # first gather allocates and fully writes every chunk, which faults in all
    # of its pages; nothing further is needed to warm them.
    _out_cache: dict[int, list] = {}

    def _gather(n_chunks: int):
        block_ids = list(range(n_chunks * blocks_per_chunk))
        out = _out_cache.get(n_chunks)
        if out is None:
            out = gather_paged_kv_to_cpu(kv, block_ids, blocks_per_chunk)
            _out_cache[n_chunks] = out
            return out
        return gather_paged_kv_to_cpu(kv, block_ids, blocks_per_chunk, out=out)

    def _scatter(n_chunks: int, chunks: list) -> None:
        block_ids = list(range(n_chunks * blocks_per_chunk))
        scatter_cpu_to_paged_kv(kv, block_ids, chunks, blocks_per_chunk)

    def _stats(fn) -> tuple[float, float]:
        """Return (mean_ms, std_ms) over `iters` timed calls after `warmup`."""
        for _ in range(warmup):
            fn()
        torch.rbln.synchronize()
        samples = []
        for _ in range(iters):
            torch.rbln.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.rbln.synchronize()
            samples.append((time.perf_counter() - t0) * 1e3)
        mean = statistics.mean(samples)
        std = statistics.stdev(samples) if len(samples) > 1 else 0.0
        return mean, std

    def _sweep() -> list[dict]:
        results = []
        for n_block in cfg["sweep_blocks"]:
            n_chunks = n_block // blocks_per_chunk
            if n_chunks < 1 or n_chunks * blocks_per_chunk > nb:
                continue
            chunks = _gather(n_chunks)
            moved = sum(c.numel() * c.element_size() for c in chunks)
            gbps = moved / 1e9
            g_ms, g_std = _stats(lambda nc=n_chunks: _gather(nc))
            s_ms, s_std = _stats(lambda nc=n_chunks, ch=chunks: _scatter(nc, ch))
            results.append(
                {
                    "n_block": n_chunks * blocks_per_chunk,
                    "n_chunks": n_chunks,
                    "moved_mb": moved / 1e6,
                    "gather_ms": g_ms,
                    "gather_std": g_std,
                    "gather_gbps": (gbps / (g_ms / 1e3)) if g_ms else None,
                    "scatter_ms": s_ms,
                    "scatter_std": s_std,
                    "scatter_gbps": (gbps / (s_ms / 1e3)) if s_ms else None,
                }
            )
        return results

    previous_ops = lmcache.device_ops
    try:
        lmcache.device_ops = torch_ops
        torch_rows = _sweep()
        lmcache.device_ops = native_ops
        native_rows = _sweep()
    finally:
        lmcache.device_ops = previous_ops

    info["blocks_per_chunk"] = blocks_per_chunk
    info["torch"] = torch_rows
    info["native"] = native_rows
    return info


def main() -> int:
    """Boot vLLM, run the worker bench, print torch vs native tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blocks-per-chunk",
        type=int,
        default=1,
        help="paged blocks per LMCache chunk (1 => chunk_size == block_size, "
        "matching the production serve)",
    )
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="untimed warmup calls before the timed iters (per direction)",
    )
    parser.add_argument(
        "--sweep-blocks",
        default="1,2,4,8",
        help="comma list of #blocks to transfer per direction",
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=1.2,
        help="required native/torch speedup per cell; 0 disables the gate",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    for k, v in _BOOT_ENV.items():
        os.environ.setdefault(k, v)

    # Third Party
    # Imported only after env is set (RblnPlatform reads VLLM_RBLN_* at import).
    from vllm import LLM, SamplingParams

    print(
        f"[boot] building LLM(model={_LLM_KWARGS['model']}) — this compiles on RBLN ..."
    )
    llm = LLM(**_LLM_KWARGS)
    try:
        # A tiny generation guarantees KV caches are fully initialized.
        llm.generate(["Hello"], SamplingParams(max_tokens=1, temperature=0.0))
        cfg = {
            "iters": args.iters,
            "warmup": args.warmup,
            "blocks_per_chunk": args.blocks_per_chunk,
            "sweep_blocks": [int(x) for x in args.sweep_blocks.split(",") if x],
        }
        out = llm.collective_rpc(_bench_on_worker, args=(cfg,))
        # Report BEFORE shutdown — the engine teardown can end the process
        # before trailing stdout flushes, so don't leave results after it.
        passed = _report(out, args)
    finally:
        llm.llm_engine.engine_core.shutdown()
    return 0 if passed else 1


def _f(x: float | None, digits: int = 2) -> str:
    """Format an optional float; ``'-'`` when absent."""
    return f"{x:.{digits}f}" if x is not None else "-"


def _ms_std(row: dict, direction: str) -> str:
    """``mean±std`` ms for one direction, or ``-`` when absent."""
    mean = row.get(f"{direction}_ms")
    if mean is None:
        return "-"
    return f"{mean:.2f}±{row.get(f'{direction}_std', 0.0):.2f}"


def _speedup(torch_ms: float | None, native_ms: float | None) -> float | None:
    """Torch-over-native speedup; ``None`` when either sample is missing."""
    if not torch_ms or not native_ms:
        return None
    return torch_ms / native_ms


def _zip_rows(
    torch_rows: list[dict], native_rows: list[dict]
) -> list[tuple[dict, dict]]:
    """Pair torch/native rows that share ``n_block``."""
    native_by_block = {row["n_block"]: row for row in native_rows}
    return [
        (torch_row, native_by_block[torch_row["n_block"]])
        for torch_row in torch_rows
        if torch_row["n_block"] in native_by_block
    ]


def _rows(pairs: list[tuple[dict, dict]], direction: str) -> None:
    print(
        f"{'n_block':>7} {'moved_MB':>9} {'torch_ms':>13} {'native_ms':>13} "
        f"{'speedup':>8} {'torch_GB/s':>10} {'native_GB/s':>11}"
    )
    for torch_row, native_row in pairs:
        speedup = _speedup(
            torch_row.get(f"{direction}_ms"), native_row.get(f"{direction}_ms")
        )
        print(
            f"{torch_row['n_block']:>7} {torch_row['moved_mb']:>9.2f} "
            f"{_ms_std(torch_row, direction):>13} "
            f"{_ms_std(native_row, direction):>13} "
            f"{_f(speedup, 3):>8} "
            f"{_f(torch_row.get(f'{direction}_gbps'), 3):>10} "
            f"{_f(native_row.get(f'{direction}_gbps'), 3):>11}"
        )


def _gate_passed(pairs: list[tuple[dict, dict]], min_speedup: float) -> bool:
    """Return True when every gather/scatter cell meets ``min_speedup``."""
    if min_speedup <= 0:
        return True
    for torch_row, native_row in pairs:
        for direction in ("gather", "scatter"):
            speedup = _speedup(
                torch_row.get(f"{direction}_ms"), native_row.get(f"{direction}_ms")
            )
            if speedup is None or speedup < min_speedup:
                return False
    return True


def _report(out: list, args: argparse.Namespace) -> bool:
    """Print per-worker tables. Return True when the speedup gate passes."""
    passed = True
    for i, info in enumerate(out):
        print(f"\n==== worker {i} : REAL vLLM device KV caches ====", flush=True)
        print(
            f"layers={info.get('n_layers')} raw_shape={info.get('raw_shape')} "
            f"num_blocks={info.get('num_blocks')} block_size={info.get('block_size')} "
            f"dtype={info.get('dtype')}"
        )
        print(
            f"device={info.get('device')} data_ptr={info.get('data_ptr_hex')} "
            f"contiguous={info.get('is_contiguous')} "
            f"blocks_per_chunk={info.get('blocks_per_chunk')}"
        )
        if info.get("error"):
            print(f"ERROR: {info['error']}")
            passed = False
            continue

        pairs = _zip_rows(info["torch"], info["native"])
        print("\nGATHER (store / D2H)")
        _rows(pairs, "gather")
        print("\nSCATTER (retrieve / H2D)")
        _rows(pairs, "scatter")
        if not _gate_passed(pairs, args.min_speedup):
            print(
                f"\nSPEEDUP GATE FAILED: native must be >={args.min_speedup:.3f}x "
                "torch on every gather/scatter cell"
            )
            passed = False
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[out] wrote {args.out}", flush=True)
    return passed


if __name__ == "__main__":
    raise SystemExit(main())
