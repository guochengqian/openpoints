import importlib.util
from pathlib import Path

import numpy as np


def _load_data_util():
    path = Path(__file__).resolve().parents[1] / "dataset" / "data_util.py"
    spec = importlib.util.spec_from_file_location("data_util", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fnv_hash_vec_distinguishes_negative_mirrored_coordinates():
    data_util = _load_data_util()
    coords = np.array([
        [1, 3, 5],
        [-1, 3, 5],
        [1, -3, 5],
        [1, 3, -5],
    ], dtype=np.int64)

    hashes = data_util.fnv_hash_vec(coords)

    assert len(np.unique(hashes)) == len(coords)


def test_fnv_hash_vec_does_not_mutate_input():
    data_util = _load_data_util()
    coords = np.array([[-2, 0, 1], [2, 0, 1]], dtype=np.int64)
    before = coords.copy()

    data_util.fnv_hash_vec(coords)

    np.testing.assert_array_equal(coords, before)
