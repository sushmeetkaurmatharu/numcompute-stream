from __future__ import annotations

import numpy as np
import pytest

from numcompute_stream.preprocessing import StreamingSimpleImputer, StreamingStandardScaler
from numcompute_stream.stats import RunningHistogram, WelfordAccumulator


def test_welford_matches_batch() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [np.nan, 8.0]])
    acc = WelfordAccumulator(2)
    acc.partial_fit(X)
    mean_batch = np.nanmean(X, axis=0)
    assert np.allclose(acc.mean_, mean_batch, equal_nan=True)


def test_welford_zero_variance() -> None:
    acc = WelfordAccumulator(2)
    acc.partial_fit(np.ones((4, 2)))
    assert np.allclose(acc.var_, 0.0)


def test_histogram_update() -> None:
    hist = RunningHistogram(n_bins=4)
    hist.partial_fit(np.array([0.0, 1.0, 2.0, 3.0]))
    assert hist.counts_.sum() == 4.0


def test_streaming_scaler_partial_fit() -> None:
    scaler = StreamingStandardScaler()
    X1 = np.array([[0.0, 0.0], [2.0, 2.0]])
    X2 = np.array([[4.0, 4.0]])
    scaler.partial_fit(X1)
    scaler.partial_fit(X2)
    Z = scaler.transform(np.array([[2.0, 2.0]]))
    assert Z.shape == (1, 2)


def test_scaler_transform_before_fit() -> None:
    scaler = StreamingStandardScaler()
    with pytest.raises(RuntimeError):
        scaler.transform(np.zeros((2, 2)))


def test_imputer_nan_fill() -> None:
    imp = StreamingSimpleImputer()
    X = np.array([[1.0, np.nan], [np.nan, 3.0]])
    imp.partial_fit(X)
    out = imp.transform(np.array([[np.nan, np.nan]]))
    assert not np.any(np.isnan(out))
