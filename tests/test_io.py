from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from numcompute_stream.io import load_csv, split_features_labels, stream_chunks


def test_load_csv_shape() -> None:
    path = Path(__file__).resolve().parents[1] / "demo" / "data" / "wine_stream.csv"
    data = load_csv(path)
    assert data.ndim == 2
    assert data.shape[0] >= 10


def test_split_features_labels() -> None:
    data = np.array([[1.0, 2.0, 0], [3.0, 4.0, 1]])
    X, y = split_features_labels(data, label_col=-1)
    assert X.shape == (2, 2)
    assert y.tolist() == [0.0, 1.0]


def test_stream_chunks_count() -> None:
    X = np.arange(20).reshape(10, 2).astype(float)
    y = np.arange(10)
    chunks = list(stream_chunks(X, y, chunk_size=3))
    assert len(chunks) == 4
    assert chunks[0][0].shape[0] == 3


def test_stream_chunks_mismatched_y() -> None:
    X = np.zeros((5, 2))
    y = np.zeros(3)
    with pytest.raises(ValueError):
        list(stream_chunks(X, y, chunk_size=2))
