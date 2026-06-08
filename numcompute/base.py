from __future__ import annotations

import numpy as np


def check_array(
    X: np.ndarray,
    *,
    ndim: int = 2,
    dtype: type | np.dtype = float,
    name: str = "X",
) -> np.ndarray:
    """Validate and coerce an array to the expected rank and dtype."""
    arr = np.asarray(X, dtype=dtype)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D; got shape {arr.shape}.")
    return arr


def check_X_y(
    X: np.ndarray,
    y: np.ndarray,
    *,
    allow_nan_X: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate feature matrix ``X`` and label vector ``y``."""
    X = check_array(X, ndim=2, name="X")
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D; got shape {y.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples; got {X.shape[0]} and {y.shape[0]}."
        )
    if not allow_nan_X and np.any(np.isnan(X)):
        raise ValueError("X contains NaN values.")
    return X, y


def encode_labels(y: np.ndarray, classes: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Map labels to contiguous integers ``0 .. n_classes-1``.

    Parameters
    ----------
    y
        Label vector of shape ``(n_samples,)``.
    classes
        Optional known class list; unseen labels in ``y`` raise ``ValueError``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Encoded integer labels and the class array used for mapping.
    """
    y = np.asarray(y).ravel()
    if classes is None:
        classes = np.unique(y)
    else:
        classes = np.unique(np.asarray(classes))
    if y.size == 0:
        return np.empty(0, dtype=int), classes

    numeric = np.issubdtype(classes.dtype, np.number) and np.issubdtype(y.dtype, np.number)
    if numeric:
        class_vals = classes.astype(float, copy=False)
        y_num = y.astype(float, copy=False)
        idx = np.searchsorted(class_vals, y_num)
        safe_idx = np.clip(idx, 0, max(classes.size - 1, 0))
        valid = (idx < classes.size) & (class_vals[safe_idx] == y_num)
        encoded = idx.astype(int, copy=False)
    else:
        lookup = {label: i for i, label in enumerate(classes.tolist())}
        encoded = np.fromiter((lookup.get(label, -1) for label in y), dtype=int, count=y.size)
        valid = encoded >= 0

    unknown = ~valid
    if np.any(unknown):
        bad = y[unknown][0]
        raise ValueError(f"Unknown label {bad!r} not in classes {classes.tolist()}.")
    return encoded, classes


def resolve_tie_max(values: np.ndarray) -> int:
    """Return index of maximum value; ties broken by smallest index (stable)."""
    return int(np.argmax(values))
