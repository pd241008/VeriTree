"""
VeriTree Phase 0: Prebinning Script (v3)
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
    feature_cols = list(range(14))
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


def prebin_yearpredictionmsd(raw_path):
    """YearPredictionMSD: zip → txt, comma-separated, 91 cols (year label + 90 features)."""
    with zipfile.ZipFile(raw_path) as z:
        txt_name = [n for n in z.namelist() if n.endswith(".txt")][0]
        with z.open(txt_name) as f:
            data = np.loadtxt(f, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n = data.shape[0]
    feats = data[:, 1:]
    binned = np.zeros((n, feats.shape[1]), dtype=np.uint8)
    for j in range(feats.shape[1]):
        binned[:, j] = _quantile_bins(feats[:, j], NUM_BINS)
    _save_bin("yearpredictionmsd", binned)
    _write_shape("yearpredictionmsd", n, binned.shape[1])


def prebin_susy(raw_path):
    """SUSY: csv.gz, comma-separated, 19 cols (label + 18 features)."""
    data = _load_numeric_csv(raw_path, 19)
    n = data.shape[0]
    feats = data[:, 1:]
    binned = np.zeros((n, feats.shape[1]), dtype=np.uint8)
    for j in range(feats.shape[1]):
        binned[:, j] = _quantile_bins(feats[:, j], NUM_BINS)
    _save_bin("susy", binned)
    _write_shape("susy", n, binned.shape[1])


def prebin_bank_marketing(raw_path):
    """Bank Marketing: zip → semicolon-separated, quoted, header, last col=y."""
    with zipfile.ZipFile(raw_path) as z:
        csv_name = [n for n in z.namelist() if n.endswith("bank-additional-full.csv")][0]
        with z.open(csv_name) as f:
            text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
            reader = csv.reader(text, delimiter=";")
            header = next(reader)
            rows = [row for row in reader if row and any(c.strip() for c in row)]
    n = len(rows)
    d = len(header) - 1  # exclude label col
    data = np.zeros((n, d), dtype=np.float64)
    numeric_cols = set()
    categorical_cols = set()
    for j, h in enumerate(header[:-1]):
        if h.strip('"') in ("age", "duration", "campaign", "pdays", "previous",
                            "emp.var.rate", "cons.price.idx", "cons.conf.idx",
                            "euribor3m", "nr.employed"):
            numeric_cols.add(j)
        else:
            categorical_cols.add(j)
    for i, row in enumerate(rows):
        for j in range(d):
            val = row[j].strip().strip('"') if j < len(row) else "?"
            if j in numeric_cols:
                try:
                    data[i, j] = float(val) if val not in ("?", "") else np.nan
                except ValueError:
                    data[i, j] = np.nan
            else:
                data[i, j] = np.nan  # placeholder, label-encoded below
    for j in categorical_cols:
        col_vals = [row[j].strip().strip('"') if j < len(row) else "?" for row in rows]
        data[:, j] = _label_encode_col(col_vals)
    binned = np.zeros((n, d), dtype=np.uint8)
    for j in range(d):
        binned[:, j] = _quantile_bins(data[:, j], NUM_BINS)
    _save_bin("bank_marketing", binned)
    _write_shape("bank_marketing", n, d)


def prebin_diabetes(raw_path):
    """Diabetes 130-US: zip → comma-separated, header, mixed types, missing ?, last col=readmitted."""
    with zipfile.ZipFile(raw_path) as z:
        csv_name = [n for n in z.namelist() if n.endswith("diabetic_data.csv")][0]
        with z.open(csv_name) as f:
            text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
            reader = csv.reader(text)
            header = next(reader)
            rows = [row for row in reader if row and any(c.strip() for c in row)]
    n = len(rows)
    d = len(header) - 1  # exclude label col
    data = np.zeros((n, d), dtype=np.float64)
    numeric_cols = set()
    categorical_cols = set()
    for j, h in enumerate(header[:-1]):
        h_clean = h.strip()
        if h_clean in ("time_in_hospital", "num_lab_procedures", "num_procedures",
                       "num_medications", "number_outpatient", "number_emergency",
                       "number_inpatient", "number_diagnoses"):
            numeric_cols.add(j)
        else:
            categorical_cols.add(j)
    for i, row in enumerate(rows):
        for j in range(d):
            val = row[j].strip() if j < len(row) else "?"
            if j in numeric_cols:
                try:
                    data[i, j] = float(val) if val not in ("?", "") else np.nan
                except ValueError:
                    data[i, j] = np.nan
            else:
                data[i, j] = np.nan
    for j in categorical_cols:
        col_vals = [row[j].strip() if j < len(row) else "?" for row in rows]
        data[:, j] = _label_encode_col(col_vals)
    binned = np.zeros((n, d), dtype=np.uint8)
    for j in range(d):
        binned[:, j] = _quantile_bins(data[:, j], NUM_BINS)
    _save_bin("diabetes", binned)
    _write_shape("diabetes", n, d)


def prebin_epsilon(raw_path):
    """Epsilon from OpenML ARFF: 2000 numeric features + class label."""
    rows = []
    data_section = False
    with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.upper() == "@DATA":
                data_section = True
                continue
            if not data_section or not line or line.startswith("%") or line.startswith("@"):
                continue
            rows.append(line.split(","))
    n = len(rows)
    d = len(rows[0]) - 1 if n > 0 else 0
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
    _save_bin("epsilon", binned)
    _write_shape("epsilon", n, d)


def prebin_cicids2017(raw_dir):
    """CICIDS2017: multiple daily CSV captures from centralized_datasets.

    Memory-efficient single pass: reads each file once, converts values to
    float immediately, accumulates per-column using array.array('d') (8 bytes
    per float, no Python object overhead). Then converts to numpy for binning.
    """
    import array
    import glob
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not files:
        print(f"[SKIP] CICIDS2017: no CSV files in {raw_dir}")
        return
    header = None
    col_arrays: list = []
    n = 0
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            h = next(reader, None)
            if header is None:
                header = h
                d = len(header) - 1
                col_arrays = [array.array("d") for _ in range(d)]
            for row in reader:
                if not row or not any(c.strip() for c in row):
                    continue
                n += 1
                for j in range(d):
                    val = row[j].strip() if j < len(row) else "?"
                    try:
                        col_arrays[j].append(
                            float(val)
                            if val not in ("?", "", "Infinity", "-Infinity", "NaN")
                            else np.nan
                        )
                    except ValueError:
                        col_arrays[j].append(np.nan)
    if n == 0:
        print("[SKIP] CICIDS2017: no data rows")
        return
    d = len(header) - 1
    data = np.zeros((n, d), dtype=np.float64)
    for j in range(d):
        data[:, j] = np.array(col_arrays[j], dtype=np.float64)
    del col_arrays
    binned = np.zeros((n, d), dtype=np.uint8)
    for j in range(d):
        binned[:, j] = _quantile_bins(data[:, j], NUM_BINS)
    _save_bin("cicids2017", binned)
    _write_shape("cicids2017", n, d)


def prebin_nsl_kdd(raw_path):
    """NSL-KDD: comma-separated, 42 cols, no header, last col=label.

    Streams row-by-row; accumulates per-column float lists.
    """
    rows = _read_csv_rows(raw_path)
    n = len(rows)
    if n == 0:
        print("[SKIP] NSL-KDD: empty, skipping")
        return
    d = len(rows[0]) - 1
    col_lists = [[] for _ in range(d)]
    for row in rows:
        for j in range(d):
            val = row[j].strip() if j < len(row) else "?"
            try:
                col_lists[j].append(float(val) if val not in ("?", "") else np.nan)
            except ValueError:
                col_lists[j].append(np.nan)
    data = np.zeros((n, d), dtype=np.float64)
    for j in range(d):
        data[:, j] = np.array(col_lists[j], dtype=np.float64)
    binned = np.zeros((n, d), dtype=np.uint8)
    for j in range(d):
        binned[:, j] = _quantile_bins(data[:, j], NUM_BINS)
    _save_bin("nsl_kdd", binned)
    _write_shape("nsl_kdd", n, d)


def prebin_rta(raw_path):
    """RTA Dataset: comma-separated, header, last col=label.

    Streams row-by-row; accumulates per-column float lists.
    """
    rows = _read_csv_rows(raw_path)
    n = len(rows)
    if n == 0:
        print("[SKIP] RTA: empty, skipping")
        return
    header = rows[0]
    data_rows = rows[1:]
    n = len(data_rows)
    d = len(header) - 1
    col_lists = [[] for _ in range(d)]
    for row in data_rows:
        for j in range(d):
            val = row[j].strip() if j < len(row) else "?"
            try:
                col_lists[j].append(float(val) if val not in ("?", "") else np.nan)
            except ValueError:
                col_lists[j].append(np.nan)
    data = np.zeros((n, d), dtype=np.float64)
    for j in range(d):
        data[:, j] = np.array(col_lists[j], dtype=np.float64)
    binned = np.zeros((n, d), dtype=np.uint8)
    for j in range(d):
        binned[:, j] = _quantile_bins(data[:, j], NUM_BINS)
    _save_bin("rta", binned)
    _write_shape("rta", n, d)


def main():
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

    jobs = [
        ("HIGGS", os.path.join(RAW_DIR, "HIGGS.csv.gz"), prebin_higgs),
        ("Covertype", os.path.join(RAW_DIR, "covtype.data.gz"), prebin_covertype),
        ("Adult", os.path.join(RAW_DIR, "adult.data"), prebin_adult),
        ("CreditCard", os.path.join(RAW_DIR, "creditcard.csv"), prebin_creditcard),
        ("UNSW-NB15", os.path.join(RAW_DIR, "UNSW_NB15_training-set.csv"), prebin_unsw_nb15),
        ("YearPredictionMSD", os.path.join(RAW_DIR, "YearPredictionMSD.txt.zip"), prebin_yearpredictionmsd),
        ("SUSY", os.path.join(RAW_DIR, "SUSY.csv.gz"), prebin_susy),
        ("Bank Marketing", os.path.join(RAW_DIR, "bank-additional.zip"), prebin_bank_marketing),
        ("Diabetes", os.path.join(RAW_DIR, "dataset_diabetes.zip"), prebin_diabetes),
    ]
    for name, raw, fn in jobs:
        if os.path.exists(raw):
            fn(raw)
        else:
            print(f"[SKIP] {name}: raw file not found: {raw}")

    prebin_california_housing()

    # Epsilon from OpenML ARFF
    epsilon_arff = "/tmp/kilo/epsilon_check.arff"
    if os.path.exists(epsilon_arff):
        prebin_epsilon(epsilon_arff)
    else:
        print("[SKIP] Epsilon: ARFF not found at /tmp/kilo/epsilon_check.arff")

    # Centralized datasets
    cicids_dir = "/root/workspace/workspace/03-Code/Projects/Legacy/centralized_datasets/raw/cicids2017"
    if os.path.isdir(cicids_dir):
        prebin_cicids2017(cicids_dir)
    else:
        print(f"[SKIP] CICIDS2017: directory not found: {cicids_dir}")

    nsl_kdd = "/root/workspace/workspace/03-Code/Projects/Legacy/centralized_datasets/raw/nsl-kdd/nsl-kdd-train.csv"
    if os.path.exists(nsl_kdd):
        prebin_nsl_kdd(nsl_kdd)
    else:
        print(f"[SKIP] NSL-KDD: file not found: {nsl_kdd}")

    rta = "/root/workspace/workspace/03-Code/Projects/Legacy/centralized_datasets/raw/rta/RTA Dataset.csv"
    if os.path.exists(rta):
        prebin_rta(rta)
    else:
        print(f"[SKIP] RTA: file not found: {rta}")


if __name__ == "__main__":
    main()
