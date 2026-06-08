from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from numcompute_stream.base import encode_labels
from numcompute_stream.ensemble import BaggingClassifier, RandomForestClassifier
from numcompute_stream.metrics import accuracy_score, confusion_matrix_vectorised
from numcompute_stream.preprocessing import StreamingOneHotEncoder
from numcompute_stream.stats import WelfordAccumulator
from numcompute_stream.tree import DecisionTreeClassifier


def _timeit(func, *args, repeats: int = 5, **kwargs) -> float:
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000.0)


def _print_row(label: str, vec_ms: float, loop_ms: float) -> None:
    speedup = loop_ms / vec_ms if vec_ms > 0 else float("inf")
    print(f"{label:<35} {vec_ms:<18.4f} {loop_ms:<12.4f} {speedup:<8.2f}x")


def confusion_matrix_loop(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=float)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[t, p] += 1.0
    return cm


def encode_labels_loop(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    lookup = {c: i for i, c in enumerate(classes.tolist())}
    out = np.empty(y.shape[0], dtype=int)
    for i, v in enumerate(y):
        out[i] = lookup[v]
    return out


def welford_loop(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_features = X.shape[1]
    count = np.zeros(n_features, dtype=int)
    mean = np.zeros(n_features, dtype=float)
    m2 = np.zeros(n_features, dtype=float)
    for row in X:
        for j in range(n_features):
            v = row[j]
            if np.isnan(v):
                continue
            count[j] += 1
            delta = v - mean[j]
            mean[j] += delta / count[j]
            m2[j] += delta * (v - mean[j])
    return mean, m2


def onehot_loop(encoder: StreamingOneHotEncoder, X: np.ndarray) -> np.ndarray:
    widths = [cats.size for cats in encoder.categories_]
    offsets = np.cumsum([0] + widths)
    out = np.zeros((X.shape[0], int(offsets[-1])), dtype=float)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            cats = encoder.categories_[j]
            if cats.size == 0:
                continue
            for k, cat in enumerate(cats):
                if X[i, j] == cat:
                    out[i, int(offsets[j] + k)] = 1.0
                    break
    return out


def ensemble_vote_loop(votes: np.ndarray, classes: np.ndarray) -> np.ndarray:
    out = np.empty(votes.shape[1], dtype=classes.dtype)
    for i in range(votes.shape[1]):
        labels, counts = np.unique(votes[:, i], return_counts=True)
        out[i] = labels[int(np.argmax(counts))]
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    n_samples, n_features, n_classes = 800, 6, 3
    X = rng.normal(size=(n_samples, n_features))
    y = rng.integers(0, n_classes, size=n_samples)

    tree = DecisionTreeClassifier(max_depth=6, min_samples_split=4, random_state=0)
    for start in range(0, n_samples, 100):
        tree.partial_fit(X[start : start + 100], y[start : start + 100])

    y_pred = tree.predict(X[:200])
    y_true = y[:200]

    print(f"CPU: {platform.processor()} | Python {sys.version.split()[0]} | NumPy {np.__version__}")
    print()
    print(f"{'Operation':<35} {'Vectorised (ms)':<18} {'Loop (ms)':<12} {'Speedup':<8}")
    print("-" * 75)
    _print_row(
        "Confusion matrix (200 samples)",
        _timeit(confusion_matrix_vectorised, y_true, y_pred, n_classes),
        _timeit(confusion_matrix_loop, y_true, y_pred, n_classes),
    )

    classes = np.arange(n_classes)
    labels = rng.integers(0, n_classes, size=5000)
    _print_row(
        "Label encoding (5000 samples)",
        _timeit(lambda: encode_labels(labels, classes)[0]),
        _timeit(lambda: encode_labels_loop(labels, classes)),
    )

    Xw = rng.normal(size=(2000, 12))
    Xw[rng.random((2000, 12)) < 0.05] = np.nan
    _print_row(
        "Welford partial_fit (2000 x 12)",
        _timeit(lambda: WelfordAccumulator(12).partial_fit(Xw)),
        _timeit(lambda: welford_loop(Xw)),
    )

    Xcat = rng.integers(0, 40, size=(5000, 8))
    enc = StreamingOneHotEncoder()
    enc.partial_fit(Xcat)
    _print_row(
        "OneHotEncoder transform (5000 x 8)",
        _timeit(enc.transform, Xcat),
        _timeit(lambda: onehot_loop(enc, Xcat)),
    )

    bag = BaggingClassifier(n_estimators=5, max_depth=4, random_state=0)
    bag.partial_fit(X[:400], y[:400])
    votes = np.stack([est.predict(X[400:600]) for est in bag.estimators_], axis=0)

    _print_row(
        "Ensemble majority vote (200 samples)",
        _timeit(bag.predict, X[400:600]),
        _timeit(lambda: ensemble_vote_loop(votes, bag.classes_)),
    )

    models = {
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=0),
        "Bagging(5)": BaggingClassifier(n_estimators=5, max_depth=4, random_state=0),
        "RandomForest(5)": RandomForestClassifier(n_estimators=5, max_depth=4, random_state=0),
    }
    chunk = 80
    print()
    print("Streaming running accuracy (base vs ensemble):")
    for name, model in models.items():
        correct = 0
        total = 0
        for start in range(0, n_samples, chunk):
            end = min(start + chunk, n_samples)
            model.partial_fit(X[start:end], y[start:end])
            pred = model.predict(X[start:end])
            correct += int(np.sum(pred == y[start:end]))
            total += end - start
        print(f"  {name:<20} running accuracy = {correct / total:.4f}")


if __name__ == "__main__":
    main()
