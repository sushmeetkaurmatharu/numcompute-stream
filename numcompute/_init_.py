from numcompute_stream.ensemble import BaggingClassifier, EnsembleClassifier, RandomForestClassifier
from numcompute_stream.io import load_csv, split_features_labels, stream_chunks
from numcompute_stream.metrics import (
    MetricTracker,
    StreamingAccuracy,
    StreamingF1,
    StreamingMetrics,
    StreamingPrecision,
    StreamingRecall,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from numcompute_stream.pipeline import Pipeline, StreamPipeline
from numcompute_stream.preprocessing import Imputer, OneHotEncoder, StandardScaler
from numcompute_stream.stats import StreamingStats, streaming_mean, streaming_quantile, streaming_variance, update_stats
from numcompute_stream.stream import StreamTrainer, memory_footprint_mb
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)

__all__ = [
    "BaggingClassifier",
    "DecisionTreeClassifier",
    "EnsembleClassifier",
    "Imputer",
    "MetricTracker",
    "OneHotEncoder",
    "Pipeline",
    "RandomForestClassifier",
    "StandardScaler",
    "StreamPipeline",
    "StreamTrainer",
    "StreamingAccuracy",
    "StreamingF1",
    "StreamingMetrics",
    "StreamingPrecision",
    "StreamingRecall",
    "StreamingStats",
    "accuracy_score",
    "compare_models",
    "f1_score",
    "load_csv",
    "memory_footprint_mb",
    "plot_metric_over_time",
    "plot_predictions_vs_ground_truth",
    "precision_score",
    "recall_score",
    "split_features_labels",
    "stream_chunks",
    "streaming_mean",
    "streaming_quantile",
    "streaming_variance",
    "update_stats",
]

__version__ = "1.0.0"
