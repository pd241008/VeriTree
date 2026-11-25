"""
VeriTree Phase 0: Prebinning Script (v2)
Converts raw downloaded CSVs/archives into pre-binned uint8_t mmap matrices.

Layout:
  data/datasets/         - raw downloaded files (CSVs, zips, gz)
  data/preprocessed/     - *_binned_uint8.bin + *_shape.json

No data fabrication. Bins derived deterministically via quantile equal-frequency
binning into 256 bins (uint8 range).

Usage:
  python3 benchmarks/prebin_datasets.py
"""

import csv
import gzip
import io
import json
import os
import sys
import zipfile
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "datasets")
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")

NUM_BINS = 256


def _out_path(name: str) -> str:
    return os.path.join(PREPROCESSED_DIR, f"{name}_binned_uint8.bin")


def _shape_path(name: str) -> str:
    return os.path.join(PREPROCESSED_DIR, f"{name}_shape.json")


def _open_maybe_gz(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _read_csv_rows(path, delimiter=",", skipinitialspace=False):
    rows = []
    with _open_maybe_gz(path) as f:
        reader = csv.reader(f, delimiter=delimiter, skipinitialspace=skipinitialspace)
        for row in reader:
            if row and any(c.strip() for c in row):
                rows.append(row)
    return rows


def _quantile_bins(col: np.ndarray, num_bins: int) -> np.ndarray:
    col = col.astype(np.float64)
    finite = col[np.isfinite(col)]
    if finite.size == 0:
        return np.zeros(col.shape, dtype=np.uint8)
    qs = np.linspace(0, 100, num_bins + 1)
    edges = np.percentile(finite, qs)
    edges = np.unique(edges)
    if edges.size < 2:
        return np.zeros(col.shape, dtype=np.uint8)
    return np.digitize(col, edges[1:-1]).astype(np.uint8)


def _label_encode_col(col: list) -> np.ndarray:
    uniq = {}
    out = []
    for v in col:
        key = v.strip() if isinstance(v, str) else str(v)
        if key == "" or key == "?":
            key = "__MISSING__"
        if key not in uniq:
            uniq[key] = len(uniq)
        out.append(uniq[key])
    arr = np.array(out, dtype=np.float64)
    return _quantile_bins(arr, NUM_BINS)


def _write_shape(name: str, n: int, d: int):
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    with open(_shape_path(name), "w") as f:
        json.dump({"num_samples": n, "num_features": d}, f)
    print(f"[SHAPE] {name} -> ({n},{d})")


def _save_bin(name: str, binned: np.ndarray):
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    binned.tofile(_out_path(name))
    print(f"[PREBIN] {name} -> {os.path.basename(_out_path(name))} shape={binned.shape}")


def _load_numeric_csv(raw_path, num_cols):
    if raw_path.endswith(".gz"):
        with gzip.open(raw_path, "rt") as f:
            data = np.loadtxt(f, delimiter=",")
    else:
        data = np.loadtxt(raw_path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.shape[1] >= num_cols, f"expected >= {num_cols} cols, got {data.shape[1]}"
    return data[:, :num_cols]


def prebin_higgs(raw_path):
    """HIGGS: 29 cols, col0=label, cols1-28=features."""
    data = _load_numeric_csv(raw_path, 29)
    n = data.shape[0]
    feats = data[:, 1:]
    binned = np.zeros((n, feats.shape[1]), dtype=np.uint8)
    for j in range(feats.shape[1]):
        binned[:, j] = _quantile_bins(feats[:, j], NUM_BINS)
    _save_bin("higgs", binned)
    _write_shape("higgs", n, binned.shape[1])


def prebin_covertype(raw_path):
    """Covertype: 55 cols, col54=label, cols0-53=features."""
    data = _load_numeric_csv(raw_path, 55)
    n = data.shape[0]
    feats = data[:, 0:54]
    binned = np.zeros((n, 54), dtype=np.uint8)
    for j in range(54):
        binned[:, j] = _quantile_bins(feats[:, j], NUM_BINS)
    _save_bin("covertype", binned)
    _write_shape("covertype", n, 54)


def prebin_adult(raw_path):
    """Adult: 15 cols mixed, last col=label, 14 features."""
    rows = _read_csv_rows(raw_path, skipinitialspace=True)
    n = len(rows)
    feature_cols = list(range(14))  # cols 0-13, exclude label col 14
    data = np.zeros((n, 14), dtype=np.float64)
    numeric_cols = [0, 2, 4, 10, 11, 12]
    categorical_cols = [1, 3, 5, 6, 7, 8, 9, 13]
    for i, row in enumerate(rows):
        for k, c in enumerate(feature_cols):
            val = row[c].strip() if c < len(row) else "?"
            if c in numeric_cols:
                try:
                    data[i, k] = float(val) if val not in ("?", "") else np.nan
                except ValueError:
                    data[i, k] = np.nan
    for k, c in enumerate(feature_cols):
        if c in categorical_cols:
            col_vals = [row[c].strip() if c < len(row) else "?" for row in rows]
            data[:, k] = _label_encode_col(col_vals)
    binned = np.zeros((n, 14), dtype=np.uint8)
    for j in range(14):
        binned[:, j] = _quantile_bins(data[:, j], NUM_BINS)
    _save_bin("adult", binned)
    _write_shape("adult", n, 14)


def prebin_creditcard(raw_path):
    """CreditCard Fraud: 30 cols (Time,V1-V28,Amount), last col=Class label."""
    rows = _read_csv_rows(raw_path)
    n = len(rows)
    # cols 0-28 are features (Time, V1-V28, Amount), col 29 = Class
    d = 29
    data = np.zeros((n, d), dtype=np.float64)
    for i, row in enumerate(rows):
        for j in range(d):
            val = row[j].strip() if j < len(row) else "0"
            try:
                data[i, j] = float(val)
            except ValueError:
                data[i, j] = 0.0
    binned = np.zeros((n, d), dtype=np.uint8)
    for j in range(d):
        binned[:, j] = _quantile_bins(data[:, j], NUM_BINS)
    _save_bin("creditcard", binned)
    _write_shape("creditcard", n, d)


def prebin_unsw_nb15(raw_path):
    """UNSW-NB15: all cols except last (label)."""
    rows = _read_csv_rows(raw_path)
    n = len(rows)
    if n == 0:
        print("[PREBIN] UNSW-NB15: empty, skipping")
        return
    d = len(rows[0]) - 1
    data = np.zeros((n, d), dtype=np.float64)
    for i, row in enumerate(rows):
        for j in range(d):
            val = row[j].strip() if j < len(row) else "?"
            try:
                data[i, j] = float(val)
            except ValueError:
                data[i, j] = np.nan
    for j in range(d):
        col = data[:, j]
        if np.isnan(col).any():
            str_col = [row[j].strip() if j < len(row) else "?" for row in rows]
            data[:, j] = _label_encode_col(str_col)
    binned = np.zeros((n, d), dtype=np.uint8)
    for j in range(d):
        binned[:, j] = _quantile_bins(data[:, j], NUM_BINS)
    _save_bin("unsw_nb15", binned)
    _write_shape("unsw_nb15", n, d)


def prebin_california_housing():
    from sklearn.datasets import fetch_california_housing
    bunch = fetch_california_housing(as_frame=True)
    X = bunch.data.values
    n, d = X.shape
    binned = np.zeros((n, d), dtype=np.uint8)
    for j in range(d):
        binned[:, j] = _quantile_bins(X[:, j], NUM_BINS)
    _save_bin("california_housing", binned)
    _write_shape("california_housing", n, d)


def prebin_epsilon(raw_path):
    """Epsilon from OpenML parquet: 2000 features + class label."""
    import pandas as pd
    df = pd.read_parquet(raw_path)
    # Last column is typically the target
    n = len(df)
    d = df.shape[1] - 1
    data = df.iloc[:, :d].values.astype(np.float64)
    binned = np.zeros((n, d), dtype=np.uint8)
    for j in range(d):
        binned[:, j] = _quantile_bins(data[:, j], NUM_BINS)
    _save_bin("epsilon", binned)
    _write_shape("epsilon", n, d)


def main():
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

    jobs = [
        ("HIGGS", os.path.join(RAW_DIR, "HIGGS.csv.gz"), prebin_higgs),
        ("Covertype", os.path.join(RAW_DIR, "covtype.data.gz"), prebin_covertype),
        ("Adult", os.path.join(RAW_DIR, "adult.data"), prebin_adult),
        ("CreditCard", os.path.join(RAW_DIR, "creditcard.csv"), prebin_creditcard),
        ("UNSW-NB15", os.path.join(RAW_DIR, "UNSW_NB15_training-set.csv"), prebin_unsw_nb15),
    ]
    for name, raw, fn in jobs:
        if os.path.exists(raw):
            fn(raw)
        else:
            print(f"[SKIP] {name}: raw file not found: {raw}")

    prebin_california_housing()

    # Epsilon from OpenML parquet (if downloaded)
    epsilon_pq = "/tmp/kilo/epsilon_check.pq"
    if os.path.exists(epsilon_pq):
        prebin_epsilon(epsilon_pq)
    else:
        print("[SKIP] Epsilon: parquet not found at /tmp/kilo/epsilon_check.pq")


if __name__ == "__main__":
    main()
