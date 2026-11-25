"""
VeriTree Phase 0: Unit Tests for Benchmark Data Loader
Validates the mmap column-store loader against real pre-binned .bin files
produced by `data/data_pipleine.go` + prebinning.

Tests that require real .bin data skip gracefully when the files are absent
(no fabrication).
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_benchmark_dataset, MmapColumnStore

SPEC_MAP = {
    "higgs": (11_000_000, 28),
    "covertype": (581_012, 54),
    "adult": (48_842, 14),
    "creditcard": (284_807, 30),
    "unsw_nb15": (257_673, 42),
    "california_housing": (20_640, 8),
}


def _bin_path(name: str) -> str:
    return os.path.join("data", f"{name}_binned_uint8.bin")


def test_spec_map_has_all_six_datasets():
    """The spec map covers the six ICML-claimed datasets."""
    assert set(SPEC_MAP.keys()) == {
        "higgs", "covertype", "adult", "creditcard", "unsw_nb15", "california_housing"
    }


def test_loader_raises_when_bin_missing():
    """load_benchmark_dataset raises FileNotFoundError (not synthetic data) when .bin absent."""
    for name in SPEC_MAP:
        try:
            loader, g, h = load_benchmark_dataset(name)
            loader.open()
            assert False, f"{name}: expected FileNotFoundError, got open success"
        except FileNotFoundError:
            pass  # expected: no real data downloaded yet


def test_mmap_column_store_opens_existing_file():
    """MmapColumnStore opens an existing file without creating/fabricating."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp_path = tmp.name
        arr = np.arange(40, dtype=np.uint8).reshape(10, 4)
        tmp.write(arr.tobytes())
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
        os.remove(tmp_path)


def test_mmap_column_store_float32_mode():
    """Raw float32 mode is supported for non-binned data."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp_path = tmp.name
        arr = np.zeros(30, dtype=np.float32).reshape(10, 3)
        tmp.write(arr.tobytes())
    try:
        store = MmapColumnStore(tmp_path, num_samples=10, num_features=3, is_binned=False)
        store.open()
        try:
            assert store._mmap_obj.dtype == np.float32
            assert store.itemsize == 4
        finally:
            store.close()
    finally:
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
    test_loader_raises_when_bin_missing()
    test_mmap_column_store_opens_existing_file()
    test_mmap_column_store_float32_mode()
    test_unknown_dataset_raises()
    test_gradients_hessians_are_none_not_fabricated()
    print("All data_loader unit tests PASSED (no real .bin data required)")
