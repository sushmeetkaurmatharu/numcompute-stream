from __future__ import annotations

import numpy as np
import pytest

from numcompute_stream.ensemble import BaggingClassifier
from numcompute_stream.metrics import accuracy_score
from numcompute_stream.preprocessing import StreamingStandardScaler
from numcompute_stream.tree import DecisionTreeClassifier


def test_accuracy_empty() -> None:
    assert accuracy_score(np.array([]), np.array([])) == 0.0


def test_scaler_constant_feature() -> None:
    scaler = StreamingStandardScaler()
    X = np.ones((5, 3))
    scaler.partial_fit(X)
    out = scaler.transform(X)
    assert np.allclose(out, 0.0)


def test_tree_single_class() -> None:
    X = np.random.default_rng(0).normal(size=(20, 3))
    y = np.zeros(20, dtype=int)
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    tree.partial_fit(X, y)
    pred = tree.predict(X)
    assert np.all(pred == 0)


def test_bagging_n_estimators_invalid() -> None:
    with pytest.raises(ValueError):
        BaggingClassifier(n_estimators=0)


def test_nan_in_features_imputed_via_pipeline() -> None:
    X = np.array([[1.0, np.nan], [2.0, 3.0]])
    y = np.array([0, 1])
    from numcompute_stream.preprocessing import StreamingSimpleImputer
    from numcompute_stream.pipeline import Pipeline

    pipe = Pipeline(
        [
            ("imp", StreamingSimpleImputer()),
            ("clf", DecisionTreeClassifier(max_depth=2, random_state=0)),
        ]
    )
    pipe.partial_fit(X, y)
    pred = pipe.predict(np.array([[np.nan, 2.0]]))
    assert pred.shape == (1,)
