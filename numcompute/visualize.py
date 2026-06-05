from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_metric_over_time(
    metric_values: list[float] | np.ndarray,
    title: str,
    ylabel: str,
    *,
    xlabel: str = "Chunk",
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Axes:
    """Plot a metric sequence across streaming chunks."""
    values = np.asarray(metric_values, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("metric_values must not be empty.")
    steps = np.arange(values.size)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, values, marker="o", linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    return ax


def compare_models(
    metric1: list[float] | np.ndarray,
    metric2: list[float] | np.ndarray,
    labels: tuple[str, str],
    *,
    title: str = "Model comparison",
    ylabel: str = "Metric value",
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Axes:
    """Compare two models' metric traces on the same axes."""
    m1 = np.asarray(metric1, dtype=float).ravel()
    m2 = np.asarray(metric2, dtype=float).ravel()
    n = max(m1.size, m2.size)
    steps = np.arange(n)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    if m1.size:
        ax.plot(steps[: m1.size], m1, marker="o", label=labels[0], linewidth=2)
    if m2.size:
        ax.plot(steps[: m2.size], m2, marker="s", label=labels[1], linewidth=2)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    return ax


def plot_predictions_vs_ground_truth(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Predictions vs ground truth (latest chunk)",
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Axes:
    """Scatter of predicted vs true labels (with y=x reference)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", linewidths=0.3)
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "r--", label="Perfect prediction")
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    return ax


# Legacy / convenience helpers
def plot_metric_history(history: list[dict[str, float]], *, metric: str = "running_accuracy", **kwargs: Any) -> plt.Axes:
    values = [h[metric] for h in history]
    return plot_metric_over_time(values, kwargs.get("title", "Metric over streaming chunks"), metric, show=False)


def plot_model_comparison(
    histories: dict[str, list[dict[str, float]]],
    *,
    metric: str = "running_accuracy",
    title: str = "Model comparison (streaming)",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if not histories:
        raise ValueError("histories must not be empty.")
    names = list(histories.keys())
    if len(names) < 2:
        vals = [h[metric] for h in histories[names[0]]]
        return plot_metric_over_time(vals, title, metric, ax=ax, show=False)
    m1 = [h[metric] for h in histories[names[0]]]
    m2 = [h[metric] for h in histories[names[1]]]
    return compare_models(m1, m2, (names[0], names[1]), title=title, ylabel=metric, ax=ax, show=False)


def plot_confusion_matrix(
    cm: np.ndarray,
    *,
    labels: list[str] | None = None,
    title: str = "Confusion matrix",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    cm = np.asarray(cm, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046)
    n = cm.shape[0]
    if labels is None:
        labels = [str(i) for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    return ax


def save_figure(fig: plt.Figure, path: str, **kwargs: Any) -> None:
    fig.savefig(path, bbox_inches="tight", dpi=120, **kwargs)
