from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import numpy as np


def load_csv(
    path: str | Path,
    *,
    delimiter: str = ",",
    skip_header: int = 1,
    missing: str = "nan",
    dtype: type | np.dtype = float,
) -> np.ndarray:
    """Load a numeric CSV file into a 2-D array.

    Parameters
    ----------
    path
        File path.
    delimiter
        Column separator.
    skip_header
        Number of header rows to skip (0 if no header).
    missing
        Token interpreted as missing; forwarded to ``numpy.genfromtxt``.
    dtype
        Output dtype.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_samples, n_features)``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    data = np.genfromtxt(
        path,
        delimiter=delimiter,
        skip_header=skip_header,
        missing_values=missing,
        filling_values=np.nan,
        dtype=dtype,
    )
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return np.asarray(data, dtype=float)


def split_features_labels(
    data: np.ndarray,
    *,
    label_col: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a table into ``X`` and ``y`` using a column index."""
    data = check_array_2d(data)
    if label_col < 0:
        label_col = data.shape[1] + label_col
    if label_col < 0 or label_col >= data.shape[1]:
        raise ValueError(f"label_col {label_col} out of range for {data.shape[1]} columns.")
    y = data[:, label_col]
    X = np.delete(data, label_col, axis=1)
    return X, y


def check_array_2d(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be 2-D.")
    return data


def stream_chunks(
    X: np.ndarray,
    y: np.ndarray | None = None,
    *,
    chunk_size: int = 32,
) -> Generator[tuple[np.ndarray, np.ndarray | None], None, None]:
    """Yield successive chunks along the sample axis."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if y is not None:
        y = np.asarray(y)
        if y.shape[0] != n:
            raise ValueError("X and y must have the same number of rows.")
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        if y is None:
            yield X[start:end], None
        else:
            yield X[start:end], y[start:end]
