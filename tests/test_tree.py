from __future__ import annotations

import numpy as np
import pytest

from numcompute_stream.tree import DecisionTreeClassifier, _gini


def test_gini_pure() -> None:
    assert _gini(np.array([10.0, 0.0])) == 0.0


def test_tree_partial_fit_predict(binary_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = binary_data
    tree = DecisionTreeClassifier(max_depth=4, min_samples_split=4, random_state=0)
    for start in range(0, X.shape[0], 15):
        tree.partial_fit(X[start : start + 15], y[start : start + 15])
    pred = tree.predict(X)
    assert pred.shape == y.shape


def test_tree_unknown_label() -> None:
    tree = DecisionTreeClassifier()
    tree.partial_fit(np.zeros((3, 2)), np.array([0, 1, 0]))
    with pytest.raises(ValueError):
        tree.partial_fit(np.zeros((2, 2)), np.array([0, 5]))


def test_tree_not_fitted_predict() -> None:
    tree = DecisionTreeClassifier()
    with pytest.raises(RuntimeError):
        tree.predict(np.zeros((2, 2)))


def test_tree_proba_sums_to_one(binary_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = binary_data
    tree = DecisionTreeClassifier(max_depth=3, random_state=1)
    tree.partial_fit(X, y)
    proba = tree.predict_proba(X[:10])
    assert np.allclose(proba.sum(axis=1), 1.0)
