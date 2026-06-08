from __future__ import annotations

import warnings

import numpy as np

from numcompute_stream.stats import WelfordAccumulator


class StreamingStandardScaler:
    """Z-score scaler with Welford running mean/variance."""

    def __init__(self) -> None:
        self._acc: WelfordAccumulator | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_features_: int | None = None

    def partial_fit(self, X: np.ndarray) -> StreamingStandardScaler:
        """Update running mean/std from chunk ``X`` of shape ``(n_samples, n_features)``."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D with shape (n_samples, n_features).")
        if self._acc is None:
            self.n_features_ = X.shape[1]
            self._acc = WelfordAccumulator(self.n_features_)
        elif X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features; got {X.shape[1]}.")
        self._acc.partial_fit(X)
        self.mean_ = self._acc.mean_.copy()
        std = self._acc.std_
        if np.any(std == 0):
            warnings.warn(
                "Zero variance detected; scale set to 1 for those features.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.scale_ = np.where(std == 0, 1.0, std)
        return self

    def update(self, X: np.ndarray) -> StreamingStandardScaler:
        """Alias for ``partial_fit``."""
        return self.partial_fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return z-scored ``X`` using fitted ``mean_`` and ``scale_``."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before transform.")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def partial_fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on ``X`` and return transformed chunk."""
        return self.partial_fit(X).transform(X)


class StreamingSimpleImputer:
    """Fill NaNs using running per-feature means."""

    def __init__(self) -> None:
        self._acc: WelfordAccumulator | None = None
        self.statistics_: np.ndarray | None = None

    def partial_fit(self, X: np.ndarray) -> StreamingSimpleImputer:
        """Update per-feature means used to fill NaNs."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D.")
        if self._acc is None:
            self._acc = WelfordAccumulator(X.shape[1])
        self._acc.partial_fit(X)
        stats = self._acc.mean_.copy()
        self.statistics_ = np.where(np.isnan(stats), 0.0, stats)
        return self

    def update(self, X: np.ndarray) -> StreamingSimpleImputer:
        """Alias for ``partial_fit``."""
        return self.partial_fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Replace NaNs with fitted column means; returns shape ``(n_samples, n_features)``."""
        if self.statistics_ is None:
            raise RuntimeError("Imputer must be fitted before transform.")
        X = np.asarray(X, dtype=float).copy()
        nan_mask = np.isnan(X)
        if np.any(nan_mask):
            X[nan_mask] = np.take(self.statistics_, np.where(nan_mask)[1])
        return X


class StreamingOneHotEncoder:
    """Expand categorical columns; categories grow with each ``partial_fit``."""

    def __init__(self, categorical_cols: list[int] | None = None) -> None:
        self.categorical_cols = categorical_cols
        self.categories_: list[np.ndarray] | None = None
        self.n_features_in_: int | None = None

    def partial_fit(self, X: np.ndarray) -> StreamingOneHotEncoder:
        """Update per-column category vocabularies from the incoming chunk."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2-D.")
        self.n_features_in_ = X.shape[1]
        cols = self.categorical_cols if self.categorical_cols is not None else list(range(X.shape[1]))
        if self.categories_ is None:
            self.categories_ = [np.array([], dtype=object) for _ in range(X.shape[1])]
        for j in cols:
            existing = self.categories_[j]
            current = np.unique(X[:, j].astype(object))
            merged = np.concatenate([existing.astype(object), current]) if existing.size else current
            # Fall back to string-order sorting when mixed types are not directly sortable.
            try:
                self.categories_[j] = np.unique(merged)
            except TypeError:
                self.categories_[j] = np.array(sorted(set(merged.tolist()), key=lambda v: str(v)), dtype=object)
        return self

    def update(self, X: np.ndarray) -> StreamingOneHotEncoder:
        """Alias for ``partial_fit``."""
        return self.partial_fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """One-hot encode fitted categorical columns; unseen values become all-zero rows."""
        if self.categories_ is None:
            raise RuntimeError("OneHotEncoder must be fitted before transform.")
        X = np.asarray(X)
        blocks: list[np.ndarray] = []
        for j in range(X.shape[1]):
            cats = self.categories_[j]
            if cats.size == 0:
                continue
            col = X[:, j]
            block = (col[:, np.newaxis] == cats[np.newaxis, :]).astype(float)
            blocks.append(block)
        return np.hstack(blocks) if blocks else np.empty((X.shape[0], 0))


# Spec-aligned aliases
StandardScaler = StreamingStandardScaler
Imputer = StreamingSimpleImputer
OneHotEncoder = StreamingOneHotEncoder
