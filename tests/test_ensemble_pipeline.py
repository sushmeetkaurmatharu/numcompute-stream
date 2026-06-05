from __future__ import annotations

import numpy as np
import pytest

from numcompute_stream.ensemble import BaggingClassifier, RandomForestClassifier
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StreamingStandardScaler
from numcompute_stream.tree import DecisionTreeClassifier


def test_bagging_streaming(multiclass_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = multiclass_data
    model = BaggingClassifier(n_estimators=3, max_depth=4, random_state=0)
    for start in range(0, 60, 20):
        model.partial_fit(X[start : start + 20], y[start : start + 20])
    pred = model.predict(X[60:])
    assert pred.shape[0] == 30


def test_random_forest_partial_fit(multiclass_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = multiclass_data
    rf = RandomForestClassifier(n_estimators=3, random_state=0)
    rf.partial_fit(X[:45], y[:45])
    rf.partial_fit(X[45:], y[45:])
    assert rf.predict(X[:5]).shape[0] == 5


def test_pipeline_end_to_end(multiclass_data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = multiclass_data
    pipe = Pipeline(
        [
            ("scale", StreamingStandardScaler()),
            ("clf", DecisionTreeClassifier(max_depth=4, random_state=0)),
        ]
    )
    for start in range(0, 80, 20):
        pipe.partial_fit(X[start : start + 20], y[start : start + 20])
    assert pipe.predict(X[80:]).shape[0] == 10


def test_pipeline_not_fitted() -> None:
    pipe = Pipeline(
        [
            ("scale", StreamingStandardScaler()),
            ("clf", DecisionTreeClassifier()),
        ]
    )
    with pytest.raises(RuntimeError):
        pipe.predict(np.zeros((2, 3)))
