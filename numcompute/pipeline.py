from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class StreamTransformer(Protocol):
    def partial_fit(self, X: np.ndarray, /) -> Any: ...
    def transform(self, X: np.ndarray, /) -> np.ndarray: ...


@runtime_checkable
class StreamClassifier(Protocol):
    def partial_fit(self, X: np.ndarray, y: np.ndarray, /, classes: np.ndarray | None = None) -> Any: ...
    def predict(self, X: np.ndarray, /) -> np.ndarray: ...


def _has_partial_fit(obj: object) -> bool:
    return callable(getattr(obj, "partial_fit", None))


class Pipeline:
    """Spec name: chain transformers + classifier with ``partial_fit``."""

    def __init__(self, steps: list[tuple[str, object]]) -> None:
        if len(steps) < 2:
            raise ValueError("Pipeline requires at least two steps (preprocess + model).")
        names = [s[0] for s in steps]
        if len(set(names)) != len(names):
            raise ValueError("Pipeline step names must be unique.")
        self.steps = steps
        self._fitted = False

    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray | None = None) -> Pipeline:
        Xt = np.asarray(X, dtype=float)
        for name, est in self.steps[:-1]:
            if not _has_partial_fit(est):
                raise ValueError(f"Step '{name}' must implement partial_fit.")
            est.partial_fit(Xt)
            Xt = est.transform(Xt)
        model = self.steps[-1][1]
        if not _has_partial_fit(model):
            raise ValueError("Final estimator must implement partial_fit.")
        model.partial_fit(Xt, y, classes=classes)
        self._fitted = True
        return self

    def update(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray | None = None) -> Pipeline:
        return self.partial_fit(X, y, classes=classes)

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xt = np.asarray(X, dtype=float)
        for _, est in self.steps[:-1]:
            Xt = est.transform(Xt)
        return Xt

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Pipeline is not fitted; call partial_fit first.")
        return self.steps[-1][1].predict(self.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Pipeline is not fitted; call partial_fit first.")
        model = self.steps[-1][1]
        if not callable(getattr(model, "predict_proba", None)):
            raise AttributeError("Final estimator does not support predict_proba.")
        return model.predict_proba(self.transform(X))


StreamPipeline = Pipeline
