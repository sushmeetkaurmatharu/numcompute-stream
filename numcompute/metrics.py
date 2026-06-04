from __future__ import annotations

from collections import deque

import numpy as np

from numcompute_stream.base import encode_labels


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Batch accuracy (fraction correct)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def confusion_matrix_vectorised(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Confusion matrix for integer labels ``0 .. n_classes-1`` (vectorised)."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    idx = y_true * n_classes + y_pred
    counts = np.bincount(idx, minlength=n_classes * n_classes)
    return counts.reshape(n_classes, n_classes).astype(float)


def _metrics_from_cm(cm: np.ndarray) -> tuple[float, float, float]:
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return float(np.mean(prec)), float(np.mean(rec)), float(np.mean(f1))


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, *, n_classes: int) -> float:
    """Macro precision from a single batch."""
    y_enc, _ = encode_labels(np.asarray(y_true), np.unique(np.concatenate([y_true, y_pred])))
    pred_enc, _ = encode_labels(np.asarray(y_pred), np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix_vectorised(y_enc, pred_enc, n_classes)
    p, _, _ = _metrics_from_cm(cm)
    return p


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, *, n_classes: int) -> float:
    """Macro recall from a single batch."""
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    y_enc, _ = encode_labels(np.asarray(y_true), labels)
    pred_enc, _ = encode_labels(np.asarray(y_pred), labels)
    cm = confusion_matrix_vectorised(y_enc, pred_enc, n_classes)
    _, r, _ = _metrics_from_cm(cm)
    return r


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, *, n_classes: int) -> float:
    """Macro F1 from a single batch."""
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    y_enc, _ = encode_labels(np.asarray(y_true), labels)
    pred_enc, _ = encode_labels(np.asarray(y_pred), labels)
    cm = confusion_matrix_vectorised(y_enc, pred_enc, n_classes)
    _, _, f1 = _metrics_from_cm(cm)
    return f1


def _auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC AUC (Mann-Whitney), vectorised."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (pos.size * neg.size))


def _auc_multiclass_ovr(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Macro one-vs-rest AUC from class probability matrix."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2 or proba.shape[0] != y_true.shape[0]:
        return float("nan")
    aucs: list[float] = []
    for c in range(proba.shape[1]):
        binary = (y_true == c).astype(int)
        if binary.sum() == 0 or binary.sum() == binary.size:
            continue
        aucs.append(_auc_binary(binary, proba[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


class StreamingMetrics:
    """Accumulate accuracy, precision, recall, F1, confusion matrix, and AUC over chunks."""

    def __init__(
        self,
        *,
        n_classes: int | None = None,
        rolling_window: int | None = 10,
    ) -> None:
        self.n_classes_ = n_classes
        self.classes_: np.ndarray | None = None
        self.rolling_window = rolling_window
        self.correct_: int = 0
        self.total_: int = 0
        self.cm_: np.ndarray | None = None
        self._y_true_auc: list[int] = []
        self._y_score_auc: list[np.ndarray] = []
        self._rolling: deque[float] = deque(maxlen=rolling_window) if rolling_window else deque()
        self.chunk_history_: list[dict[str, float]] = []

    def update(
        self,
        y_true_chunk: np.ndarray,
        y_pred_chunk: np.ndarray,
        *,
        y_score_chunk: np.ndarray | None = None,
        classes: np.ndarray | None = None,
    ) -> StreamingMetrics:
        """Spec: ``update(y_true_chunk, y_pred_chunk)``."""
        y_true = np.asarray(y_true_chunk).ravel()
        y_pred = np.asarray(y_pred_chunk).ravel()
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape.")

        if classes is not None:
            self.classes_ = np.unique(np.asarray(classes))
        elif self.classes_ is None:
            self.classes_ = np.unique(np.concatenate([y_true, y_pred]))
        y_enc, _ = encode_labels(y_true, self.classes_)
        pred_enc, _ = encode_labels(y_pred, self.classes_)
        n_classes = int(self.classes_.size)
        self.n_classes_ = n_classes
        if self.cm_ is None:
            self.cm_ = np.zeros((n_classes, n_classes), dtype=float)
        elif self.cm_.shape != (n_classes, n_classes):
            raise ValueError("Class count changed during streaming; call reset() first.")

        self.cm_ += confusion_matrix_vectorised(y_enc, pred_enc, n_classes)
        self.correct_ += int(np.sum(y_enc == pred_enc))
        self.total_ += int(y_enc.size)

        chunk_acc = float(np.mean(y_enc == pred_enc))
        if self.rolling_window:
            self._rolling.append(chunk_acc)

        if y_score_chunk is not None:
            scores = np.asarray(y_score_chunk, dtype=float)
            if scores.ndim == 1 and n_classes == 2 and scores.shape[0] == y_enc.shape[0]:
                self._y_true_auc.extend(y_enc.tolist())
                self._y_score_auc.append(scores)
            elif scores.ndim == 2 and scores.shape == (y_enc.shape[0], n_classes):
                self._y_true_auc.extend(y_enc.tolist())
                self._y_score_auc.append(scores)

        self.chunk_history_.append(
            {
                "chunk_accuracy": chunk_acc,
                "running_accuracy": self.accuracy,
                "rolling_accuracy": self.rolling_accuracy,
            }
        )
        return self

    def reset(self) -> None:
        """Spec: clear all accumulated state."""
        self.correct_ = 0
        self.total_ = 0
        self.cm_ = None
        self.classes_ = None
        self._y_true_auc.clear()
        self._y_score_auc.clear()
        self._rolling.clear()
        self.chunk_history_.clear()

    def result(self) -> dict[str, float | np.ndarray]:
        """Spec: return cumulative accuracy, precision, recall, F1, CM, AUC."""
        if self.cm_ is None or self.n_classes_ is None:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "confusion_matrix": np.zeros((0, 0)),
                "auc": float("nan"),
                "rolling_accuracy": 0.0,
            }
        prec, rec, f1 = _metrics_from_cm(self.cm_)
        auc = float("nan")
        if self._y_true_auc and self._y_score_auc:
            y_all = np.array(self._y_true_auc, dtype=int)
            scores_all = np.vstack(self._y_score_auc)
            if scores_all.shape[1] == 2:
                auc = _auc_binary(y_all, scores_all[:, 1])
            else:
                auc = _auc_multiclass_ovr(y_all, scores_all)
        return {
            "accuracy": self.accuracy,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": self.cm_.copy(),
            "auc": auc,
            "rolling_accuracy": self.rolling_accuracy,
        }

    @property
    def accuracy(self) -> float:
        if self.total_ == 0:
            return 0.0
        return self.correct_ / self.total_

    @property
    def rolling_accuracy(self) -> float:
        if not self._rolling:
            return self.accuracy
        return float(np.mean(self._rolling))


class StreamingAccuracy(StreamingMetrics):
    """Spec-aligned alias focused on accuracy; inherits full ``update`` / ``reset`` / ``result``."""


class StreamingPrecision(StreamingMetrics):
    """Spec-aligned alias; use ``result()['precision']`` after ``update`` calls."""


class StreamingRecall(StreamingMetrics):
    """Spec-aligned alias; use ``result()['recall']`` after ``update`` calls."""


class StreamingF1(StreamingMetrics):
    """Spec-aligned alias; use ``result()['f1']`` after ``update`` calls."""


class MetricTracker(StreamingMetrics):
    """Backward-compatible tracker with ``record_chunk`` snapshots."""

    def record_chunk(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        step: int | None = None,
        y_score: np.ndarray | None = None,
        classes: np.ndarray | None = None,
    ) -> dict[str, float]:
        self.update(y_true, y_pred, y_score_chunk=y_score, classes=classes)
        snap = self.chunk_history_[-1].copy()
        snap["step"] = float(step if step is not None else len(self.chunk_history_) - 1)
        snap["running_error"] = 1.0 - snap["running_accuracy"]
        return snap

    @property
    def history_(self) -> list[dict[str, float]]:
        return self.chunk_history_
