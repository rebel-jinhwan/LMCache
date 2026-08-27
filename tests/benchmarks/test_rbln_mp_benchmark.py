# SPDX-License-Identifier: Apache-2.0

# First Party
from benchmarks.rbln.bench_kv_transfer_mp import (
    _gate_passed,
    _speedup,
    _zip_rows,
)


def test_speedup_is_torch_over_native() -> None:
    """A faster native run reports a speedup greater than 1."""
    assert _speedup(1.0, 0.5) == 2.0
    assert _speedup(1.0, None) is None
    assert _speedup(0.0, 0.5) is None


def test_zip_rows_pairs_matching_block_counts() -> None:
    """Rows are joined on ``n_block``; unmatched native cells are dropped."""
    torch_rows = [
        {"n_block": 1, "gather_ms": 2.0, "scatter_ms": 3.0},
        {"n_block": 4, "gather_ms": 8.0, "scatter_ms": 9.0},
    ]
    native_rows = [
        {"n_block": 1, "gather_ms": 1.0, "scatter_ms": 1.5},
        {"n_block": 2, "gather_ms": 1.0, "scatter_ms": 1.0},
    ]

    pairs = _zip_rows(torch_rows, native_rows)

    assert len(pairs) == 1
    assert pairs[0][0]["n_block"] == 1
    assert pairs[0][1]["gather_ms"] == 1.0


def test_gate_requires_speedup_on_every_direction() -> None:
    """The gate fails when scatter is below the required speedup."""
    pairs = [
        (
            {"n_block": 1, "gather_ms": 2.0, "scatter_ms": 2.0},
            {"n_block": 1, "gather_ms": 1.0, "scatter_ms": 1.9},
        )
    ]

    assert _gate_passed(pairs, min_speedup=0.0) is True
    assert _gate_passed(pairs, min_speedup=1.2) is False
