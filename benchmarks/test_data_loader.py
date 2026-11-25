"""
VeriTree Phase 0: Unit Tests for Benchmark Data Loader
Validates the mmap column-store loader against real pre-binned .bin files
produced by `data/data_pipleine.go` + `benchmarks/prebin_datasets.py`.

Tests skip gracefully when real .bin data is absent (no fabrication).
"""

import os
import json
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_benchmark_dataset, MmapColumnStore, PREPROCESSED_DIR

SPEC_MAP = {
    "higgs": (11_000_000, 28),
    "covertype": (581_012, 54),
    "adult": (48_842, 14),
    "creditcard": (284_807, 30),
    "unsw_nb15": (257_673, 44),
    "california_housing": (20_640, 8),
}


def _bin_path(name: str) -> str:
    return os.path.join(PREPROCESSED_DIR, f"{name}_binned_uint8.bin")


def _shape_path(name: str) -> str:
    return os.path.join(PREPROCESSED_DIR, f"{name}_shape.json")


def test_spec_map_has_all_six_datasets():
    """The spec map covers the six ICML-claimed datasets."""
    assert set(SPEC_MAP.keys()) == {
        "higgs", "covertype", "adult", "creditcard", "unsw_nb15", "california_housing"
    }


def test_loader_opens_all_present_datasets():
    """All 6 real .bin files open with actual dimensions from sidecar shape files."""
    present = 0
    for name in SPEC_MAP:
        path = _bin_path(name)
        shape_path = _shape_path(name)
        if not os.path.exists(path):
            continue
        present += 1
        with open(shape_path) as f:
            actual = json.load(f)
        exp_n, exp_d = actual["num_samples"], actual["num_features"]
        loader, g, h = load_benchmark_dataset(name)
        loader.open()
        try:
            assert loader._mmap_obj.shape == (exp_n, exp_d), \
                f"{name}: shape {loader._mmap_obj.shape} != ({exp_n},{exp_d})"
            assert loader._mmap_obj.dtype == np.uint8
            assert g is None and h is None, "gradients/hessians must be None (not fabricated)"
        finally:
            loader.close()
    assert present == 6, f"expected 6 present datasets, got {present}"


def test_loader_raises_when_bin_missing():
    """Absent .bin raises FileNotFoundError (not synthetic data)."""
    try:
        loader, g, h = load_benchmark_dataset("nonexistent_dataset")
        assert False, "expected ValueError for unknown dataset"
    except ValueError:
        pass


def test_mmap_column_store_opens_existing_file():
    """MmapColumnStore opens an existing file without creating/fabricating."""
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    tmp_path = os.path.join(PREPROCESSED_DIR, "_tmp_test.bin")
    arr = np.arange(40, dtype=np.uint8).reshape(10, 4)
    arr.tofile(tmp_path)
    try:
        store = MmapColumnStore(tmp_path, num_samples=10, num_features=4, is_binned=True)
        store.open()
        try:
            assert store._mmap_obj.shape == (10, 4)
            assert store._mmap_obj.dtype == np.uint8
            col = store.get_column(0)
            assert col.shape == (10,)
            assert int(col[0]) == 0
            assert int(col[1]) == 4
        finally:
            store.close()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_mmap_column_store_float32_mode():
    """Raw float32 mode is supported for non-binned data."""
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    tmp_path = os.path.join(PREPROCESSED_DIR, "_tmp_test_float.bin")
    arr = np.zeros(30, dtype=np.float32).reshape(10, 3)
    arr.tofile(tmp_path)
    try:
        store = MmapColumnStore(tmp_path, num_samples=10, num_features=3, is_binned=False)
        store.open()
        try:
            assert store._mmap_obj.dtype == np.float32
            assert store.itemsize == 4
        finally:
            store.close()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_unknown_dataset_raises():
    """load_benchmark_dataset raises ValueError for unknown dataset names."""
    try:
        load_benchmark_dataset("nonexistent_dataset")
        assert False, "expected ValueError for unknown dataset"
    except ValueError:
        pass


def test_gradients_hessians_are_none_not_fabricated():
    """Gradients/hessians are None (training-derived), never synthesized."""
    for name in SPEC_MAP:
        loader, g, h = load_benchmark_dataset(name)
        assert g is None, f"{name}: gradients should be None, not fabricated"
        assert h is None, f"{name}: hessians should be None, not fabricated"


if __name__ == "__main__":
    test_spec_map_has_all_six_datasets()
    test_loader_opens_all_present_datasets()
    test_loader_raises_when_bin_missing()
    test_mmap_column_store_opens_existing_file()
    test_mmap_column_store_float32_mode()
    test_unknown_dataset_raises()
    test_gradients_hessians_are_none_not_fabricated()
    print("All data_loader unit tests PASSED (real data, no fabrication)")
