"""W46: tests for the ImageNet-32 training subset.  Run: .venv/bin/python qks-tn-library-W46/tests/test_imagenet.py"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import run_ff_experiment as rfe  # noqa: E402


def test_subset_is_fixed_sorted_unique_and_spans_classes():
    n = 1281167
    a = rfe.subset_indices(n, 20000)
    b = rfe.subset_indices(n, 20000)
    assert np.array_equal(a, b) and len(a) == 20000 and len(np.unique(a)) == 20000
    assert np.all(np.diff(a) > 0) and a.min() >= 0 and a.max() < n
    # class-sorted data (1281 rows per class here): the subset must reach most classes, not the first few
    assert len(np.unique(a // 1281)) > 900


def test_subset_larger_than_n_is_everything():
    assert np.array_equal(rfe.subset_indices(100, 500), np.arange(100))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
