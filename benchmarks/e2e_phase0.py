"""
VeriTree Phase 0: End-to-End Reproducibility Script
Exercises the Phase 0 toolchain & storage tenets end-to-end using REAL data
from `data/data_pipleine.go` (Go downloader, 4-8 workers) + prebinned .bin files.

Does NOT synthesize data or results. Skips datasets whose .bin file is absent.
Writes results/phase0_e2e.json traceability manifest only for datasets actually present.
"""

import json
import os
import sys
import time
from typing import Dict, Any, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "benchmarks"))

from data_loader import load_benchmark_dataset  # noqa: E402

SPEC_MAP = {
    "higgs": (11_000_000, 28),
    "covertype": (581_012, 54),
    "adult": (48_842, 14),
    "creditcard": (284_807, 30),
    "unsw_nb15": (257_673, 42),
    "california_housing": (20_640, 8),
}


def mmap_load_and_time(name: str) -> Optional[Dict[str, Any]]:
    """Load a dataset via mmap column-store and time column access. Returns None if .bin absent."""
    path = os.path.join("data", f"{name}_binned_uint8.bin")
    if not os.path.exists(path):
        print(f"[SKIP] {name}: no pre-binned .bin at {path} — run the Go downloader first")
        return None

    loader, g, h = load_benchmark_dataset(name)
    loader.open()
    try:
        n, d = loader.num_samples, loader.num_features
        start = time.perf_counter_ns()
        col_sums = []
        for f in range(d):
            col = loader.get_column(f)
            col_sums.append(int(col.sum()))
        elapsed_ns = time.perf_counter_ns() - start
        return {
            "dataset": name,
            "num_samples": n,
            "num_features": d,
            "dtype": str(loader._mmap_obj.dtype),
            "mmap_bytes": loader.expected_bytes,
            "column_access_wall_ns": elapsed_ns,
            "column_access_wall_us": elapsed_ns / 1_000.0,
            "first_column_sum": col_sums[0] if col_sums else 0,
        }
    finally:
        loader.close()


def main() -> None:
    present: Dict[str, Any] = {}
    for name in SPEC_MAP:
        result = mmap_load_and_time(name)
        if result is not None:
            present[name] = result

    results: Dict[str, Any] = {
        "phase": 0,
        "description": "Workspace architecture & toolchain initialization",
        "tenets": [
            "64-byte L1 cache-line alignment",
            "AVX2 + FMA SIMD",
            "Strict IEEE-754 determinism (-fno-fast-math)",
            "Zero-copy mmap-backed uint8_t column store",
            "No data fabrication — real .bin files required",
        ],
        "toolchains": {
            "python": sys.version.split()[0],
        },
        "datasets_present": present,
        "datasets_present_count": len(present),
        "datasets_total_claimed": len(SPEC_MAP),
        "seed": 42,
    }

    out_dir = os.path.join(REPO_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase0_e2e.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[Phase 0 e2e] {len(present)}/{len(SPEC_MAP)} datasets present")
    print(f"[Phase 0 e2e] Wrote {out_path}")


if __name__ == "__main__":
    main()
