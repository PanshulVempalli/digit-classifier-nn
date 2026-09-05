"""
Loads the real handwritten digit dataset bundled with scikit-learn
(a subset of the UCI ML "Optical Recognition of Handwritten Digits"
dataset -- 1,797 real scanned digits, each 8x8 pixels, 0-9). Bundled
locally with the library, so no network access is needed to load it.
"""

import numpy as np
from sklearn.datasets import load_digits


def load_digit_data(test_fraction=0.2, seed=0):
    digits = load_digits()
    X = digits.data.astype(np.float64)  # (1797, 64), pixel values 0-16
    y = digits.target  # (1797,), integers 0-9

    # Scale pixel values to [0, 1] -- keeps activations in a sane range
    # for the network, same idea as normalising image data for any
    # gradient-based model.
    X = X / 16.0

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    indices = rng.permutation(n)
    n_test = int(n * test_fraction)
    test_idx, train_idx = indices[:n_test], indices[n_test:]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
