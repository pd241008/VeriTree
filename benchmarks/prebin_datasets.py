"""
VeriTree Phase 0: Prebinning Script
Converts raw downloaded CSVs/archives (from data/data_pipleine.go) into
pre-binned uint8_t mmap matrices expected by benchmarks/data_loader.py:
  data/<dataset>_binned_uint8.bin  (row-major, shape [N, D], uint8)

No data is fabricated — bins are derived deterministically from real values
via quantile-based equal-frequency binning into 256 bins (uint8 range).

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

NUM_BINS = 256  # uint8 range


def _open_maybe_gz(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _read_csv_rows(path, has_header=False, delimiter=",", skipinitialspace=False):
    rows = []
    with _open_maybe_gz(path) as f:
        reader = csv.reader(f, delimiter=delimiter, skipinitialspace=skipinitialspace)
        if has_header:
            next(reader, None)
        for row in reader:
            if row and any(c.strip() for c in row):
                rows.append(row)
    return rows


def _quantile_bins(col: np.ndarray, num_bins: int) -> np.ndarray:
    """Bin a 1-D numeric column into num_bins quantile-based bins (0..num_bins-1)."""
    col = col.astype(np.float64)
    finite = col[np.isfinite(col)]
    if finite.size == 0:
        return np.zeros(col.shape, dtype=np.uint8)
    # Unique quantile edges; fall back to linspace if all values identical
    qs = np.linspace(0, 100, num_bins + 1)
    edges = np.percentile(finite, qs)
    edges = np.unique(edges)
    if edges.size < 2:
        return np.zeros(col.shape, dtype=np.uint8)
    return np.digitize(col, edges[1:-1]).astype(np.uint8)


def _label_encode_col(col: list) -> np.ndarray:
    """Label-encode a string column, treating '?'/empty as a distinct bin."""
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


def _load_numeric_csv(raw_path, num_cols):
    """Load an all-numeric (possibly gzipped) CSV with no header into float64 array."""
    if raw_path.endswith(".gz"):
        with gzip.open(raw_path, "rt") as f:
            data = np.loadtxt(f, delimiter=",")
    else:
        data = np.loadtxt(raw_path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.shape[1] >= num_cols, f"expected >= {num_cols} cols, got {data.shape[1]}"
    return data[:, :num_cols]


def _write_shape(name: str, n: int, d: int):
    """Write a sidecar JSON with actual (N, D) for the loader to trust."""
    shape_path = os.path.join(DATA_DIR, f"{name}_shape.json")
    with open(shape_path, "w") as f:
        json.dump({"num_samples": n, "num_features": d}, f)
    print(f"[SHAPE] {name} -> {shape_path}  ({n},{d})")


def prebin_higgs(raw_path, out_path):
    """HIGGS: 29 cols, col0=label, cols1-28=features, no header."""
    data = _load_numeric_csv(raw_path, 29)
    n = data.shape[0]
    d = data.shape[1] - 1  # all columns except label
    feats = data[:, 1:]
    binned = np.zeros((n, d), dtype=np.uint8)
    for j in range(d):
        binned[:, j] = _quantile_bins(feats[:, j], NUM_BINS)
    binned.tofile(out_path)
    _write_shape("higgs", n, d)
    print(f"[PREBIN] HIGGS -> {out_path}  shape=({n},{d})")


def prebin_covertype(raw_path, out_path):
    """Covertype: 55 cols, col54=label, cols0-53=features, no header."""
    data = _load_numeric_csv(raw_path, 55)
    n = data.shape[0]
    d = 54
    feats = data[:, 0:54]
    binned = np.zeros((n, d), dtype=np.uint8)
    for j in range(d):
        binned[:, j] = _quantile_bins(feats[:, j], NUM_BINS)
    binned.tofile(out_path)
    _write_shape("covertype", n, d)
    print(f"[PREBIN] Covertype -> {out_path}  shape=({n},{d})")


def prebin_adult(raw_path, out_path):
    """
    Adult: 15 cols, mixed types, no header, missing " ?", spaces around commas.
    Cols:
      0 age, 1 workclass, 2 fnlwgt, 3 education, 4 education-num,
      5 marital-status, 6 occupation, 7 relationship, 8 race, 9 sex,
      10 capital-gain, 11 capital-loss, 12 hours-per-week, 13 native-country,
      14 label
    Numeric: 0,2,4,10,11,12  (6 cols)
    Categorical: 1,3,5,6,7,8,9,13  (8 cols)
    Total features: 14
    """
    rows = _read_csv_rows(raw_path, skipinitialspace=True)
    n = len(rows)
    numeric_cols = [0, 2, 4, 10, 11, 12]
    categorical_cols = [1, 3, 5, 6, 7, 8, 9, 13]
    feature_cols = numeric_cols + categorical_cols
    d = len(feature_cols)
    data = np.zeros((n, d), dtype=np.float64)
    for i, row in enumerate(rows):
        for k, c in enumerate(feature_cols):
            val = row[c].strip() if c < len(row) else "?"
            if c in numeric_cols:
                try:
                    data[i, k] = float(val) if val not in ("?", "") else np.nan
                except ValueError:
                    data[i, k] = np.nan
            # categorical columns are label-encoded in the second pass below
    # Label-encode categorical columns properly
    for k, c in enumerate(feature_cols):
        if c in categorical_cols:
            col_vals = [row[c].strip() if c < len(row) else "?" for row in rows]
            data[:, k] = _label_encode_col(col_vals)
    binned = np.zeros((n, d), dtype=np.uint8)
    for j in range(d):
        binned[:, j] = _quantile_bins(data[:, j], NUM_BINS)
    binned.tofile(out_path)
    _write_shape("adult", n, d)
    print(f"[PREBIN] Adult -> {out_path}  shape=({n},{d})")


def prebin_unsw_nb15(raw_path, out_path):
    """
    UNSW-NB15: use the training-set CSV. 43 features + label, no header,
    comma-separated, some string columns.
    We take all columns except the last (label) and bin them.
    Non-numeric columns are label-encoded.
    """
    rows = _read_csv_rows(raw_path)
    n = len(rows)
    if n == 0:
        print(f"[PREBIN] UNSW-NB15: empty, skipping")
        return
    ncols = len(rows[0])
    d = ncols - 1  # exclude label
    data = np.zeros((n, d), dtype=np.float64)
    for i, row in enumerate(rows):
        for j in range(d):
            val = row[j].strip() if j < len(row) else "?"
            try:
                data[i, j] = float(val)
            except ValueError:
                data[i, j] = np.nan
    # For any column that had non-numeric values, label-encode
    for j in range(d):
        col = data[:, j]
        if np.isnan(col).any():
            str_col = [row[j].strip() if j < len(row) else "?" for row in rows]
            data[:, j] = _label_encode_col(str_col)
    binned = np.zeros((n, d), dtype=np.uint8)
    for j in range(d):
        binned[:, j] = _quantile_bins(data[:, j], NUM_BINS)
    binned.tofile(out_path)
    _write_shape("unsw_nb15", n, d)
    print(f"[PREBIN] UNSW-NB15 -> {out_path}  shape=({n},{d})")


def prebin_california_housing(out_path):
    """Fetch via sklearn and bin 8 features."""
    from sklearn.datasets import fetch_california_housing
    bunch = fetch_california_housing(as_frame=True)
    X = bunch.data.values  # (20640, 8)
    n, d = X.shape
    binned = np.zeros((n, d), dtype=np.uint8)
    for j in range(d):
        binned[:, j] = _quantile_bins(X[:, j], NUM_BINS)
    binned.tofile(out_path)
    _write_shape("california_housing", n, d)
    print(f"[PREBIN] California Housing -> {out_path}  shape=({n},{d})")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Raw → pre-binned conversions
    jobs = [
        ("HIGGS", os.path.join(DATA_DIR, "HIGGS.csv.gz"),
         os.path.join(DATA_DIR, "higgs_binned_uint8.bin"), prebin_higgs),
        ("Covertype", os.path.join(DATA_DIR, "covtype.data.gz"),
         os.path.join(DATA_DIR, "covertype_binned_uint8.bin"), prebin_covertype),
        ("Adult", os.path.join(DATA_DIR, "adult.data"),
         os.path.join(DATA_DIR, "adult_binned_uint8.bin"), prebin_adult),
        ("UNSW-NB15", os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv"),
         os.path.join(DATA_DIR, "unsw_nb15_binned_uint8.bin"), prebin_unsw_nb15),
    ]
    for name, raw, out, fn in jobs:
        if os.path.exists(raw):
            fn(raw, out)
        else:
            print(f"[SKIP] {name}: raw file not found: {raw}")

    # California Housing via sklearn
    cal_out = os.path.join(DATA_DIR, "california_housing_binned_uint8.bin")
    prebin_california_housing(cal_out)

    # CreditCard Fraud — requires Kaggle CLI (manual), skip if absent
    cc_out = os.path.join(DATA_DIR, "creditcard_binned_uint8.bin")
    if not os.path.exists(cc_out):
        print(f"[SKIP] CreditCard Fraud: needs Kaggle CLI download to {cc_out}")


if __name__ == "__main__":
    main()
