from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from numcompute_stream.base import check_X_y, encode_labels, resolve_tie_max


def _gini(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    return float(1.0 - np.sum(p * p))


def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


def _impurity(counts: np.ndarray, criterion: str) -> float:
    if criterion == "entropy":
        return _entropy(counts)
    if criterion == "gini":
        return _gini(counts)
    raise ValueError("criterion must be 'gini' or 'entropy'.")


def _best_split(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    feature_indices: np.ndarray | None = None,
    *,
    criterion: str = "gini",
) -> tuple[int, float, float] | None:
    """Find the split with maximum impurity reduction (vectorised per feature)."""
    n, d = X.shape
    if n < 2:
        return None
    if feature_indices is None:
        feature_indices = np.arange(d)
    parent_counts = np.bincount(y, minlength=n_classes).astype(float)
    parent_imp = _impurity(parent_counts, criterion)
    best_gain = 0.0
    best: tuple[int, float, float] | None = None

    for j in feature_indices:
        order = np.argsort(X[:, j], kind="mergesort")
        x_sorted = X[order, j]
        y_sorted = y[order]
        left_counts = np.zeros(n_classes, dtype=float)
        right_counts = parent_counts.copy()
        i = 0
        while i < n:
            while i + 1 < n and x_sorted[i + 1] == x_sorted[i]:
                cls = y_sorted[i]
                left_counts[cls] += 1.0
                right_counts[cls] -= 1.0
                i += 1
            cls = y_sorted[i]
            left_counts[cls] += 1.0
            right_counts[cls] -= 1.0
            i += 1
            if i >= n or left_counts.sum() == 0 or right_counts.sum() == 0:
                continue
            thr = 0.5 * (x_sorted[i - 1] + x_sorted[i])
            n_left = left_counts.sum()
            n_right = right_counts.sum()
            gain = parent_imp - (n_left / n) * _impurity(left_counts, criterion) - (n_right / n) * _impurity(
                right_counts, criterion
            )
            if gain > best_gain + 1e-12:
                best_gain = gain
                best = (int(j), float(thr), float(gain))
    return best


@dataclass
class _TreeNode:
    """Internal tree node."""

    is_leaf: bool = True
    feature: int = -1
    threshold: float = 0.0
    left: _TreeNode | None = None
    right: _TreeNode | None = None
    class_counts: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    buffer_X: list[np.ndarray] = field(default_factory=list)
    buffer_y: list[np.ndarray] = field(default_factory=list)

    def prediction(self) -> int:
        return resolve_tie_max(self.class_counts)

    def n_samples(self) -> int:
        return int(self.class_counts.sum())


class DecisionTreeClassifier:
    """Binary decision tree supporting chunk-wise ``partial_fit``."""

    def __init__(
        self,
        *,
        max_depth: int = 5,
        min_samples_split: int = 6,
        min_samples_leaf: int = 2,
        max_features: int | None = None,
        criterion: str = "gini",
        random_state: int | None = 42,
    ) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1.")
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root_: _TreeNode | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self._rng = np.random.default_rng(random_state)
        self.depth_ = 0

    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray | None = None) -> DecisionTreeClassifier:
        X, y = check_X_y(X, y)
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features; got {X.shape[1]}.")
        class_ref = self.classes_ if self.classes_ is not None else classes
        y_enc, self.classes_ = encode_labels(y, class_ref)
        n_classes = int(self.classes_.size)
        if self.root_ is None:
            self.root_ = _TreeNode(class_counts=np.zeros(n_classes, dtype=float))
        for xi, yi in zip(X, y_enc, strict=True):
            self._add_sample(self.root_, xi, int(yi), depth=0, n_classes=n_classes)
        return self

    def update(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray | None = None) -> DecisionTreeClassifier:
        return self.partial_fit(X, y, classes=classes)

    def _feature_subset(self, n_features: int) -> np.ndarray:
        if self.max_features is None or self.max_features >= n_features:
            return np.arange(n_features)
        k = max(1, int(self.max_features))
        return self._rng.choice(n_features, size=k, replace=False)

    def _add_sample(self, node: _TreeNode, x: np.ndarray, y: int, *, depth: int, n_classes: int) -> None:
        node.class_counts[y] += 1.0
        if node.is_leaf:
            node.buffer_X.append(x.copy())
            node.buffer_y.append(np.array([y], dtype=int))
            if node.n_samples() >= self.min_samples_split and depth < self.max_depth:
                self._try_split(node, depth, n_classes)
            return
        if x[node.feature] <= node.threshold:
            child = node.left
        else:
            child = node.right
        assert child is not None
        self._add_sample(child, x, y, depth=depth + 1, n_classes=n_classes)

    def _try_split(self, node: _TreeNode, depth: int, n_classes: int) -> None:
        if not node.buffer_X:
            return
        Xb = np.vstack(node.buffer_X)
        yb = np.concatenate(node.buffer_y).astype(int)
        if Xb.shape[0] < self.min_samples_split:
            return
        feat_idx = self._feature_subset(Xb.shape[1])
        split = _best_split(Xb, yb, n_classes, feature_indices=feat_idx, criterion=self.criterion)
        if split is None:
            return
        feat, thr, _ = split
        left_mask = Xb[:, feat] <= thr
        right_mask = ~left_mask
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            return
        node.is_leaf = False
        node.feature = feat
        node.threshold = thr
        node.left = _TreeNode(class_counts=np.bincount(yb[left_mask], minlength=n_classes).astype(float))
        node.right = _TreeNode(class_counts=np.bincount(yb[right_mask], minlength=n_classes).astype(float))
        node.buffer_X.clear()
        node.buffer_y.clear()
        self.depth_ = max(self.depth_, depth + 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None or self.classes_ is None:
            raise RuntimeError("DecisionTreeClassifier is not fitted.")
        X = np.asarray(X, dtype=float)
        preds = np.empty(X.shape[0], dtype=self.classes_.dtype)
        for i, row in enumerate(X):
            preds[i] = self.classes_[self._predict_row(self.root_, row)]
        return preds

    def _predict_row(self, node: _TreeNode, x: np.ndarray) -> int:
        if node.is_leaf or node.left is None or node.right is None:
            return node.prediction()
        if x[node.feature] <= node.threshold:
            return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None or self.classes_ is None:
            raise RuntimeError("DecisionTreeClassifier is not fitted.")
        X = np.asarray(X, dtype=float)
        n_classes = self.classes_.size
        proba = np.zeros((X.shape[0], n_classes), dtype=float)
        for i, row in enumerate(X):
            counts = self._node_counts(self.root_, row)
            total = counts.sum()
            proba[i] = counts / total if total > 0 else 1.0 / n_classes
        return proba

    def _node_counts(self, node: _TreeNode, x: np.ndarray) -> np.ndarray:
        if node.is_leaf or node.left is None or node.right is None:
            return node.class_counts.copy()
        if x[node.feature] <= node.threshold:
            return self._node_counts(node.left, x)
        return self._node_counts(node.right, x)
