from __future__ import annotations

from collections import deque

import numpy as np


class WelfordAccumulator:
    """Online mean and variance per feature (axis=0 over samples)."""

    def __init__(self, n_features: int) -> None:
        if n_features <= 0:
            raise ValueError("n_features must be positive.")
        self.n_features = n_features
        self.count_: int = 0
        self.mean_ = np.zeros(n_features, dtype=float)
        self.m2_ = np.zeros(n_features, dtype=float)

    def partial_fit(self, X: np.ndarray) -> WelfordAccumulator:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"X must have shape (n, {self.n_features}).")
        for row in X:
            mask = ~np.isnan(row)
            if not np.any(mask):
                continue
            self.count_ += 1
            delta = np.where(mask, row - self.mean_, 0.0)
            self.mean_ += delta / self.count_
            delta2 = np.where(mask, row - self.mean_, 0.0)
            self.m2_ += delta * delta2
        return self

    def update(self, X: np.ndarray) -> WelfordAccumulator:
        return self.partial_fit(X)

    @property
    def var_(self) -> np.ndarray:
        if self.count_ < 2:
            return np.zeros(self.n_features, dtype=float)
        return self.m2_ / (self.count_ - 1)

    @property
    def std_(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.var_, 0.0))


class RunningHistogram:
    """Fixed-bin histogram with optional sliding-window recomputation."""

    def __init__(self, n_bins: int = 16, window_size: int | None = None) -> None:
        if n_bins <= 0:
            raise ValueError("n_bins must be positive.")
        self.n_bins = n_bins
        self.window_size = window_size
        self.counts_ = np.zeros(n_bins, dtype=float)
        self.edges_: np.ndarray | None = None
        self._min = np.inf
        self._max = -np.inf
        self._recent: deque[np.ndarray] = deque()

    def partial_fit(self, values: np.ndarray) -> RunningHistogram:
        values = np.asarray(values, dtype=float).ravel()
        values = values[~np.isnan(values)]
        if values.size == 0:
            return self
        if self.window_size is not None:
            self._recent.append(values.copy())
            while len(self._recent) > self.window_size:
                self._recent.popleft()
            values = np.concatenate(list(self._recent))
            self.counts_ = np.zeros(self.n_bins, dtype=float)
            self._min, self._max = np.inf, -np.inf
        self._min = min(self._min, float(np.min(values)))
        self._max = max(self._max, float(np.max(values)))
        if self._max == self._min:
            self.counts_ = np.zeros(self.n_bins, dtype=float)
            self.counts_[0] = float(values.size)
            self.edges_ = np.array([self._min, self._max + 1e-9])
            return self
        edges = np.linspace(self._min, self._max, self.n_bins + 1)
        self.edges_ = edges
        idx = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, self.n_bins - 1)
        self.counts_ = np.bincount(idx, minlength=self.n_bins).astype(float)
        return self

    def update(self, values: np.ndarray) -> RunningHistogram:
        return self.partial_fit(values)

    def quantile(self, q: float) -> float:
        if self.edges_ is None or self.counts_.sum() == 0:
            return float("nan")
        if q <= 0:
            return float(self.edges_[0])
        if q >= 1:
            return float(self.edges_[-1])
        cdf = np.cumsum(self.counts_) / self.counts_.sum()
        idx = min(int(np.searchsorted(cdf, q, side="right")), self.n_bins - 1)
        return float((self.edges_[idx] + self.edges_[idx + 1]) / 2.0)


class StreamingStats:
    """Bundle of streaming mean/variance and per-feature histograms."""

    def __init__(self, n_features: int, *, n_bins: int = 16, window_size: int | None = None) -> None:
        self.welford = WelfordAccumulator(n_features)
        self.histograms = [RunningHistogram(n_bins=n_bins, window_size=window_size) for _ in range(n_features)]

    def update_stats(self, X_chunk: np.ndarray) -> StreamingStats:
        """Spec API: update all streaming statistics from a chunk."""
        X = np.asarray(X_chunk, dtype=float)
        if X.ndim != 2:
            raise ValueError("X_chunk must be 2-D.")
        if X.shape[1] != self.welford.n_features:
            raise ValueError("Feature count mismatch for StreamingStats.")
        self.welford.partial_fit(X)
        for j in range(X.shape[1]):
            self.histograms[j].partial_fit(X[:, j])
        return self

    def mean(self) -> np.ndarray:
        return self.welford.mean_.copy()

    def variance(self) -> np.ndarray:
        return self.welford.var_.copy()

    def quantile(self, q: float, feature: int = 0) -> float:
        return self.histograms[feature].quantile(q)


def update_stats(X_chunk: np.ndarray, state: StreamingStats | None = None) -> StreamingStats:
    """Spec: functional entrypoint to create or update streaming statistics."""
    X = np.asarray(X_chunk, dtype=float)
    if X.ndim != 2:
        raise ValueError("X_chunk must be 2-D.")
    if state is None:
        state = StreamingStats(X.shape[1])
    return state.update_stats(X)


def streaming_mean(X_chunk: np.ndarray, state: StreamingStats | None = None) -> tuple[np.ndarray, StreamingStats]:
    """Chunk-based running mean (returns mean vector and updated state)."""
    state = update_stats(X_chunk, state)
    return state.mean(), state


def streaming_variance(X_chunk: np.ndarray, state: StreamingStats | None = None) -> tuple[np.ndarray, StreamingStats]:
    """Chunk-based running variance."""
    state = update_stats(X_chunk, state)
    return state.variance(), state


def streaming_quantile(
    X_chunk: np.ndarray,
    q: float,
    *,
    feature: int = 0,
    state: StreamingStats | None = None,
) -> tuple[float, StreamingStats]:
    """Chunk-based approximate quantile from streaming histogram."""
    state = update_stats(X_chunk, state)
    return state.quantile(q, feature=feature), state
