from __future__ import annotations

import tracemalloc
from typing import Any

import numpy as np

from numcompute_stream.metrics import StreamingMetrics


def _estimate_model_bytes(pipeline: Any) -> float:
    """Sum nbytes of fitted NumPy arrays held in pipeline steps (MB)."""
    total = 0
    steps = getattr(pipeline, "steps", [])
    for _, est in steps:
        for attr in dir(est):
            if attr.endswith("_") and not attr.startswith("__"):
                val = getattr(est, attr, None)
                if isinstance(val, np.ndarray):
                    total += val.nbytes
        estimators = getattr(est, "estimators_", None)
        if estimators:
            for sub in estimators:
                for attr in dir(sub):
                    if attr.endswith("_") and not attr.startswith("__"):
                        val = getattr(sub, attr, None)
                        if isinstance(val, np.ndarray):
                            total += val.nbytes
    return total / (1024 * 1024)


def memory_footprint_mb(pipeline: Any | None = None) -> dict[str, float]:
    """Return traced heap and optional model-array memory (MB)."""
    current_mb, peak_mb = 0.0, 0.0
    if tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
        current_mb = current / (1024 * 1024)
        peak_mb = peak / (1024 * 1024)
    model_mb = _estimate_model_bytes(pipeline) if pipeline is not None else 0.0
    return {
        "heap_current_mb": current_mb,
        "heap_peak_mb": peak_mb,
        "model_arrays_mb": model_mb,
        "memory_mb": current_mb + model_mb,
    }


class StreamTrainer:
    """Orchestrate ``partial_fit`` on a pipeline; log metrics, memory, and accuracy per chunk."""

    def __init__(
        self,
        pipeline: Any,
        *,
        classes: np.ndarray | None = None,
        rolling_window: int | None = 10,
        track_memory: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.classes = np.asarray(classes) if classes is not None else None
        self.metrics = StreamingMetrics(rolling_window=rolling_window)
        self.track_memory = track_memory
        self.chunk_logs: list[dict[str, float]] = []
        self.step = 0
        if track_memory and not tracemalloc.is_tracing():
            tracemalloc.start()

    def fit_chunk(self, X_chunk: np.ndarray, y_chunk: np.ndarray) -> dict[str, float]:
        """Train on one chunk and return the score snapshot for that chunk."""
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)
        if self.classes is None:
            self.classes = np.unique(y_chunk)
        self.pipeline.partial_fit(X_chunk, y_chunk, classes=self.classes)
        return self.score_chunk(X_chunk, y_chunk)

    def score_chunk(self, X_chunk: np.ndarray, y_chunk: np.ndarray) -> dict[str, float]:
        """Predict on the chunk and update cumulative metrics."""
        pred = self.pipeline.predict(X_chunk)
        y_proba = None
        if callable(getattr(self.pipeline, "predict_proba", None)):
            y_proba = self.pipeline.predict_proba(X_chunk)
        self.metrics.update(
            y_chunk,
            pred,
            y_score_chunk=y_proba,
            classes=self.classes,
        )

        chunk_acc = float(np.mean(np.asarray(y_chunk).ravel() == np.asarray(pred).ravel()))
        log: dict[str, float] = {
            "step": float(self.step),
            "chunk_accuracy": chunk_acc,
            "cumulative_accuracy": self.metrics.accuracy,
            "rolling_accuracy": self.metrics.rolling_accuracy,
        }
        if self.track_memory:
            mem = memory_footprint_mb(self.pipeline)
            log.update(mem)
        self.chunk_logs.append(log)
        self.step += 1
        return log

    def results(self) -> dict[str, Any]:
        """Full metric summary after all chunks."""
        out = self.metrics.result()
        out["chunk_logs"] = self.chunk_logs
        return out
