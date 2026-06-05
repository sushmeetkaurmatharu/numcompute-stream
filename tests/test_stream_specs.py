from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.metrics import StreamingAccuracy, StreamingMetrics, f1_score, precision_score, recall_score
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import OneHotEncoder, StandardScaler
from numcompute_stream.stats import StreamingStats, streaming_mean, update_stats
from numcompute_stream.stream import StreamTrainer, memory_footprint_mb
from numcompute_stream.tree import DecisionTreeClassifier, _entropy, _gini
from numcompute_stream.visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)


def test_streaming_metrics_reset_result() -> None:
    m = StreamingMetrics()
    m.update(np.array([0, 1, 0]), np.array([0, 0, 1]))
    r1 = m.result()
    assert r1["accuracy"] > 0
    m.reset()
    assert m.result()["accuracy"] == 0.0


def test_update_stats_api() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    state = update_stats(X)
    state = update_stats(np.array([[5.0, 6.0]]), state)
    assert state.mean().shape == (2,)


def test_streaming_mean_helper() -> None:
    mean, state = streaming_mean(np.array([[1.0, 0.0]]))
    assert mean.shape == (2,)
    assert state is not None


def test_stream_trainer_fit_score() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = (X[:, 0] > 0).astype(int)
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", DecisionTreeClassifier(max_depth=3, random_state=0)),
    ])
    trainer = StreamTrainer(pipe)
    log = trainer.fit_chunk(X[:10], y[:10])
    assert "cumulative_accuracy" in log
    assert "memory_mb" in log
    trainer.score_chunk(X[10:20], y[10:20])
    assert len(trainer.chunk_logs) == 2


def test_memory_footprint() -> None:
    mem = memory_footprint_mb()
    assert "heap_current_mb" in mem
    assert "memory_mb" in mem


def test_one_hot_encoder_incremental() -> None:
    enc = OneHotEncoder(categorical_cols=[0])
    enc.partial_fit(np.array([["a"], ["b"]]))
    enc.partial_fit(np.array([["c"]]))
    out = enc.transform(np.array([["a"], ["c"]]))
    assert out.shape[0] == 2


def test_entropy_criterion() -> None:
    tree = DecisionTreeClassifier(max_depth=3, criterion="entropy", random_state=0)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]], dtype=float)
    y = np.array([0, 0, 1, 1, 1])
    tree.partial_fit(X, y)
    assert tree.predict(X).shape == (5,)


def test_ensemble_classifier_base() -> None:
    ens = EnsembleClassifier(n_estimators=2, max_depth=2, random_state=0)
    X = np.random.default_rng(1).normal(size=(30, 3))
    y = np.array([0] * 15 + [1] * 15)
    ens.partial_fit(X, y)
    assert ens.predict(X).shape == (30,)


def test_batch_precision_recall_f1() -> None:
    y_t = np.array([0, 0, 1, 1])
    y_p = np.array([0, 1, 1, 1])
    assert 0 <= precision_score(y_t, y_p, n_classes=2) <= 1
    assert 0 <= recall_score(y_t, y_p, n_classes=2) <= 1
    assert 0 <= f1_score(y_t, y_p, n_classes=2) <= 1


def test_streaming_accuracy_alias() -> None:
    acc = StreamingAccuracy()
    acc.update(np.array([0, 1]), np.array([0, 1]))
    assert acc.result()["accuracy"] == 1.0


def test_gini_entropy_impurity() -> None:
    counts = np.array([5.0, 5.0])
    assert _gini(counts) > 0
    assert _entropy(counts) > 0


def test_spec_visualise_functions() -> None:
    ax = plot_metric_over_time([0.5, 0.7, 0.8], "Acc", "Accuracy", show=False)
    assert ax is not None
    ax2 = compare_models([0.5, 0.6], [0.7, 0.8], ("A", "B"), show=False)
    assert ax2 is not None
    ax3 = plot_predictions_vs_ground_truth(np.array([0, 1, 1]), np.array([0, 0, 1]), show=False)
    assert ax3 is not None


def test_pipeline_spec_name() -> None:
    pipe = Pipeline([("s", StandardScaler()), ("c", DecisionTreeClassifier(max_depth=2))])
    X = np.random.default_rng(2).normal(size=(20, 3))
    y = np.array([0] * 10 + [1] * 10)
    pipe.partial_fit(X, y)
    assert pipe.predict(X).shape == (20,)
