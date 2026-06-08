from __future__ import annotations

import numpy as np

from numcompute_stream.base import check_X_y, encode_labels
from numcompute_stream.tree import DecisionTreeClassifier


class EnsembleClassifier:
    """Base class managing ``n_estimators`` decision trees for streaming ensembles."""

    def __init__(
        self,
        *,
        n_estimators: int = 5,
        max_depth: int = 5,
        min_samples_split: int = 4,
        criterion: str = "gini",
        random_state: int | None = 42,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.random_state = random_state
        self.estimators_: list[DecisionTreeClassifier] = []
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self._rng = np.random.default_rng(random_state)

    def _make_estimator(self, index: int, **kwargs: object) -> DecisionTreeClassifier:
        rs = None if self.random_state is None else self.random_state + index
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            criterion=self.criterion,
            random_state=rs,
            **kwargs,
        )

    def _init_estimators(self, **kwargs: object) -> None:
        if not self.estimators_:
            self.estimators_ = [self._make_estimator(i, **kwargs) for i in range(self.n_estimators)]

    def _bootstrap_partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n = X.shape[0]
        for est in self.estimators_:
            idx = self._rng.integers(0, n, size=n)
            est.partial_fit(X[idx], y[idx], classes=self.classes_)

    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray | None = None) -> EnsembleClassifier:
        """Update each estimator with a bootstrap-resampled view of the incoming chunk."""
        X, y = check_X_y(X, y)
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
        if classes is not None:
            self.classes_ = np.unique(np.asarray(classes))
        elif self.classes_ is None:
            self.classes_ = np.unique(y)
        self._init_estimators()
        self._bootstrap_partial_fit(X, y)
        return self

    def update(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray | None = None) -> EnsembleClassifier:
        """Alias for ``partial_fit``."""
        return self.partial_fit(X, y, classes=classes)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Majority vote across estimators; returns shape ``(n_samples,)``."""
        if not self.estimators_ or self.classes_ is None:
            raise RuntimeError("EnsembleClassifier is not fitted.")
        X = np.asarray(X, dtype=float)
        votes = np.stack([est.predict(X) for est in self.estimators_], axis=0)
        n_samples = X.shape[0]
        n_classes = int(self.classes_.size)
        vote_idx, _ = encode_labels(votes.ravel(), self.classes_)
        vote_idx = vote_idx.reshape(votes.shape)
        offsets = vote_idx * n_samples + np.arange(n_samples)
        counts = np.bincount(offsets.ravel(), minlength=n_classes * n_samples).reshape(
            n_classes, n_samples
        ).T
        return self.classes_[np.argmax(counts, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average class probabilities; returns shape ``(n_samples, n_classes)``."""
        if not self.estimators_ or self.classes_ is None:
            raise RuntimeError("EnsembleClassifier is not fitted.")
        return np.mean(np.stack([est.predict_proba(X) for est in self.estimators_], axis=0), axis=0)


class BaggingClassifier(EnsembleClassifier):
    """Bootstrap aggregating ensemble."""

    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray | None = None) -> BaggingClassifier:
        """Update all bagged trees from the new chunk."""
        super().partial_fit(X, y, classes=classes)
        return self


class RandomForestClassifier(EnsembleClassifier):
    """Random forest with per-tree feature subsampling."""

    def __init__(
        self,
        *,
        n_estimators: int = 5,
        max_depth: int = 6,
        min_samples_split: int = 4,
        max_features: str | int | float = "sqrt",
        criterion: str = "gini",
        random_state: int | None = 42,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            random_state=random_state,
        )
        self.max_features = max_features

    def _resolve_max_features(self, n_features: int) -> int:
        if isinstance(self.max_features, int):
            return max(1, min(self.max_features, n_features))
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(max(n_features, 2))))
        raise ValueError("max_features must be int, float, 'sqrt', or 'log2'.")

    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray | None = None) -> RandomForestClassifier:
        """Update all trees using bootstrap rows plus per-tree feature subsampling."""
        X, y = check_X_y(X, y)
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
        if classes is not None:
            self.classes_ = np.unique(np.asarray(classes))
        elif self.classes_ is None:
            self.classes_ = np.unique(y)
        mf = self._resolve_max_features(self.n_features_in_)
        if not self.estimators_:
            self.estimators_ = [self._make_estimator(i, max_features=mf) for i in range(self.n_estimators)]
        self._bootstrap_partial_fit(X, y)
        return self
