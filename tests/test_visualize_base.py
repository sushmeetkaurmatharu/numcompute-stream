from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from numcompute_stream.base import check_X_y, encode_labels, resolve_tie_max
from numcompute_stream.metrics import MetricTracker
from numcompute_stream.visualise import plot_confusion_matrix, plot_metric_history, plot_model_comparison


def test_check_X_y_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        check_X_y(np.zeros((3, 2)), np.zeros(2))


def test_encode_labels() -> None:
    enc, classes = encode_labels(np.array(["a", "b", "a"]))
    assert list(classes) == ["a", "b"]
    assert list(enc) == [0, 1, 0]


def test_resolve_tie_max() -> None:
    assert resolve_tie_max(np.array([1.0, 3.0, 3.0])) == 1


def test_plot_metric_history() -> None:
    tracker = MetricTracker()
    tracker.record_chunk(np.array([0, 1]), np.array([0, 1]))
    ax = plot_metric_history(tracker.history_, metric="running_accuracy")
    assert ax is not None


def test_plot_model_comparison() -> None:
    h1 = [{"step": 0, "running_accuracy": 0.5}]
    h2 = [{"step": 0, "running_accuracy": 0.7}]
    ax = plot_model_comparison({"A": h1, "B": h2})
    assert ax is not None


def test_plot_confusion_matrix() -> None:
    cm = np.array([[3.0, 1.0], [0.0, 2.0]])
    ax = plot_confusion_matrix(cm, labels=["0", "1"])
    assert ax is not None
