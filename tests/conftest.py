from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def binary_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def multiclass_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X = rng.normal(size=(90, 5))
    y = rng.integers(0, 3, size=90)
    return X, y
