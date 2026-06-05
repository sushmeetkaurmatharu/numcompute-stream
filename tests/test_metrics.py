from __future__ import annotations

import numpy as np

from numcompute_stream.metrics import (
    MetricTracker,
    StreamingMetrics,
    accuracy_score,
    confusion_matrix_vectorised,
)


def test_accuracy_perfect() -> None:
    y = np.array([0, 1, 1, 0])
    assert accuracy_score(y, y) == 1.0


def test_confusion_matrix_vectorised() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    cm = confusion_matrix_vectorised(y_true, y_pred, 2)
    assert cm[0, 0] == 1.0
    assert cm[0, 1] == 1.0
    assert cm[1, 1] == 2.0


def test_metric_tracker_streaming() -> None:
    tracker = MetricTracker()
    tracker.record_chunk(np.array([0, 1]), np.array([0, 0]), step=0)
    tracker.record_chunk(np.array([1, 1]), np.array([1, 1]), step=1)
    assert tracker.accuracy == 0.75
    assert len(tracker.history_) == 2


def test_streaming_metrics_result_keys() -> None:
    m = StreamingMetrics()
    m.update(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    r = m.result()
    assert "precision" in r and "f1" in r and "confusion_matrix" in r


def test_streaming_metrics_rolling_window() -> None:
    m = StreamingMetrics(rolling_window=2)
    classes = np.array([0, 1])
    m.update(np.array([0, 0]), np.array([0, 0]), classes=classes)  # chunk acc 1.0
    m.update(np.array([0, 1]), np.array([1, 0]), classes=classes)  # chunk acc 0.0
    r = m.result()
    assert r["rolling_accuracy"] == 0.5
    assert m.chunk_history_[-1]["rolling_accuracy"] == 0.5


def test_streaming_metrics_multiclass_auc() -> None:
    m = StreamingMetrics()
    y = np.array([0, 1, 2, 0, 1, 2])
    pred = np.array([0, 1, 2, 0, 1, 2])
    scores = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.85, 0.1, 0.05],
            [0.1, 0.85, 0.05],
            [0.05, 0.1, 0.85],
        ]
    )
    m.update(y, pred, y_score_chunk=scores, classes=np.array([0, 1, 2]))
    auc = m.result()["auc"]
    assert isinstance(auc, float)
    assert auc > 0.9
