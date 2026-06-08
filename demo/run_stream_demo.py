from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from numcompute_stream.io import load_csv, split_features_labels, stream_chunks
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer, StandardScaler
from numcompute_stream.ensemble import BaggingClassifier, RandomForestClassifier
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.visualise import (
    compare_models,
    plot_confusion_matrix,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
    save_figure,
)


def build_pipeline(model) -> Pipeline:
    return Pipeline([
        ("impute", Imputer()),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def main() -> None:
    data_path = Path(__file__).parent / "data" / "wine_stream.csv"
    raw = load_csv(data_path)
    X, y = split_features_labels(raw, label_col=-1)
    classes = np.unique(y)

    print("=" * 70)
    print("NumCompute Stream: Assignment 2.2 Demo")
    print("=" * 70)
    print(f"CSV I/O (io.py): {data_path.name}")
    print(f"Samples={X.shape[0]}, Features={X.shape[1]}, Classes={classes.tolist()}")
    print("Streaming: chunk_size=8, partial_fit per chunk, StreamTrainer logging\n")

    configs = {
        "DecisionTree (Gini)": DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=0),
        "DecisionTree (Entropy)": DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=0),
        "Bagging": BaggingClassifier(n_estimators=5, max_depth=4, random_state=0),
        "RandomForest": RandomForestClassifier(n_estimators=5, max_depth=4, random_state=0),
    }
    trainers = {
        name: StreamTrainer(build_pipeline(model), classes=classes, rolling_window=5)
        for name, model in configs.items()
    }

    last_chunk: tuple[np.ndarray, np.ndarray] | None = None
    for step, (X_chunk, y_chunk) in enumerate(stream_chunks(X, y, chunk_size=8)):
        print(f"--- Chunk {step + 1} ({X_chunk.shape[0]} rows) ---")
        for name, trainer in trainers.items():
            log = trainer.fit_chunk(X_chunk, y_chunk)
            print(
                f"  {name:26} chunk={log['chunk_accuracy']:.1%}  "
                f"cumulative={log['cumulative_accuracy']:.1%}  "
                f"memory_mb={log.get('memory_mb', 0):.3f}"
            )
        last_chunk = (X_chunk, y_chunk)
        print()

    print("=" * 70)
    print("FINAL metrics (StreamingMetrics.result)")
    print("=" * 70)
    for name, trainer in trainers.items():
        r = trainer.results()
        print(
            f"{name}: acc={r['accuracy']:.1%} prec={r['precision']:.1%} "
            f"rec={r['recall']:.1%} f1={r['f1']:.1%} auc={r['auc']}"
        )

    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)

    tree_cum = [log["cumulative_accuracy"] for log in trainers["DecisionTree (Gini)"].chunk_logs]
    rf_cum = [log["cumulative_accuracy"] for log in trainers["RandomForest"].chunk_logs]
    ax1 = plot_metric_over_time(
        tree_cum,
        "Cumulative accuracy — Decision Tree",
        "Accuracy",
        show=False,
    )
    save_figure(ax1.figure, out / "plot_metric_over_time_tree.png")
    plt.close(ax1.figure)

    ax2 = compare_models(tree_cum, rf_cum, ("DecisionTree", "RandomForest"), ylabel="Accuracy", show=False)
    save_figure(ax2.figure, out / "compare_models.png")
    plt.close(ax2.figure)

    best = max(trainers, key=lambda n: trainers[n].results()["accuracy"])
    if last_chunk is not None:
        X_last, y_last = last_chunk
        pred_last = trainers[best].pipeline.predict(X_last)
        ax3 = plot_predictions_vs_ground_truth(y_last, pred_last, title=f"Latest chunk — {best}", show=False)
        save_figure(ax3.figure, out / "plot_predictions_vs_ground_truth.png")
        plt.close(ax3.figure)

    cm = trainers[best].results()["confusion_matrix"]
    ax4 = plot_confusion_matrix(cm, labels=[str(c) for c in classes], title=f"Confusion matrix — {best}")
    save_figure(ax4.figure, out / "confusion_matrix.png")
    plt.close(ax4.figure)

    print(f"\nFigures saved: {out.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
