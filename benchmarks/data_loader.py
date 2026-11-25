"""
VeriTree Mmap Binary Column-Store Loader
Supports zero-copy memory-mapped access for pre-binned uint8_t feature matrices.
Real data is produced by `data/data_pipleine.go` (Go downloader, 4-8 workers)
which downloads raw CSVs/UCI archives into `data/`, then a prebinning step
converts them to `data/<dataset>_binned_uint8.bin`.

This loader does NOT synthesize or fabricate data. If a `.bin` file is missing
it raises FileNotFoundError with a clear message pointing at the downloader.
"""

import os
import mmap
import struct
import numpy as np
from typing import Tuple, Optional

class MmapColumnStore:
    def __init__(self, data_path: str, num_samples: int, num_features: int, is_binned: bool = True):
        self.data_path = data_path
        self.num_samples = num_samples
        self.num_features = num_features
        self.is_binned = is_binned
        self.dtype = np.uint8 if is_binned else np.float32
        self.itemsize = 1 if is_binned else 4
        self.expected_bytes = num_samples * num_features * self.itemsize
        self._mmap_obj: Optional[np.memmap] = None

    def open(self) -> np.memmap:
        """Open an existing memory-mapped matrix. Never creates/fabricates data."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Pre-binned data file not found: {self.data_path}\n"
                f"Run: cd data && go run data_pipleine.go -download -workers 8\n"
                f"Then prebin the raw CSV to {self.data_path}"
            )
        self._mmap_obj = np.memmap(
            self.data_path,
            dtype=self.dtype,
            mode="r",
            shape=(self.num_samples, self.num_features)
        )
        return self._mmap_obj

    def get_column(self, col_idx: int) -> np.ndarray:
        """Zero-copy slice of a single feature column."""
        assert self._mmap_obj is not None, "Call open() before accessing data"
        return self._mmap_obj[:, col_idx]

    def close(self):
        if self._mmap_obj is not None:
            del self._mmap_obj
            self._mmap_obj = None

def load_benchmark_dataset(
    dataset_name: str,
    base_dir: str = "data"
) -> Tuple[MmapColumnStore, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Standard loader returning (mmap_features, gradients, hessians) for:
    HIGGS (N=11M, D=28), Covertype (N=581k, D=54), Adult Income (N=48k, D=14),
    CreditCard Fraud (N=284k, D=30), UNSW-NB15 (N=257k, D=42), California Housing (N=20k, D=8).

    Gradients/hessians are training-derived and are NOT synthesized here.
    They are returned as None until the training pipeline supplies them.
    """
    spec_map = {
        "higgs": (11000000, 28),
        "covertype": (581012, 54),
        "adult": (48842, 14),
        "creditcard": (284807, 30),
        "unsw_nb15": (257673, 42),
        "california_housing": (20640, 8)
    }
    key = dataset_name.lower()
    if key not in spec_map:
        raise ValueError(f"Unknown benchmark: {dataset_name}. Expected one of {list(spec_map.keys())}")

    n_samples, n_feats = spec_map[key]
    path = os.path.join(base_dir, f"{key}_binned_uint8.bin")
    loader = MmapColumnStore(path, n_samples, n_feats, is_binned=True)

    # Gradients/hessians come from the training pipeline, not the feature store.
    # Return None to avoid fabricating labels.
    return loader, None, None
