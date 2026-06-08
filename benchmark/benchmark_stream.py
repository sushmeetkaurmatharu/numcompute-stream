from __future__ import annotations

import platform
import sys
import time

import numpy as np

from numcompute_stream.metrics import accuracy_score, confusion_matrix_vectorised
from numcompute_stream.tree import DecisionTreeClassifier


def _timeit(func, *args, repeats: int = 5, **kwargs) -> float:
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000.0)


def confusion_matrix_loop(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=float)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[t, p] += 1.0
    return cm


def predict_loop(tree: DecisionTreeClassifier, X: np.ndarray) -> np.ndarray:
    return tree.predict(X)


def predict_batch(tree: DecisionTreeClassifier, X: np.ndarray) -> np.ndarray:
    # Same API; included for timing parity with ensemble voting below
    return tree.predict(X)


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

    vec_ms = _timeit(confusion_matrix_vectorised, y_true, y_pred, n_classes)
    loop_ms = _timeit(confusion_matrix_loop, y_true, y_pred, n_classes)
    speedup = loop_ms / vec_ms if vec_ms > 0 else float("inf")

    print(f"CPU: {platform.processor()} | Python {sys.version.split()[0]} | NumPy {np.__version__}")
    print()
    print(f"{'Operation':<35} {'Vectorised (ms)':<18} {'Loop (ms)':<12} {'Speedup':<8}")
    print("-" * 75)
    print(f"{'Confusion matrix (200 samples)':<35} {vec_ms:<18.4f} {loop_ms:<12.4f} {speedup:<8.2f}x")

    # Tree depth vs ensemble (streaming chunks)
    from numcompute_stream.ensemble import BaggingClassifier, RandomForestClassifier

    models = {
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=0),
        "Bagging(5)": BaggingClassifier(n_estimators=5, max_depth=4, random_state=0),
        "RandomForest(5)": RandomForestClassifier(n_estimators=5, max_depth=4, random_state=0),
    }
    chunk = 80
    print()
    print("Streaming accuracy after each chunk (last chunk accuracy):")
    for name, model in models.items():
        correct = 0
        total = 0
        for start in range(0, n_samples, chunk):
            end = min(start + chunk, n_samples)
            model.partial_fit(X[start:end], y[start:end])
            pred = model.predict(X[start:end])
            correct += int(np.sum(pred == y[start:end]))
            total += end - start
        acc = correct / total
        print(f"  {name:<20} running accuracy = {acc:.4f}")


if __name__ == "__main__":
    main()
