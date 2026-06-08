# NumCompute Stream (Assignment 2.2)

**Course:** COMP 5004 Programming for AI and Machine Learning  
**Student:** Sushmeet Kaur Matharu  
**Task:** Programming Task 2 (Weeks 1 to 10)

## What this project is

NumCompute Stream is my streaming machine learning framework for Assignment 2.2. I built it to extend the ideas from the NumCompute work in this course into a full pipeline that learns from data as it arrives in small batches, similar to how real online systems receive records over time.

The framework focuses on decision trees and tree-based ensembles. Every major piece supports incremental learning through `partial_fit()` or `update()`. I used only **NumPy** and **matplotlib**, as required by the brief, so every algorithm choice is visible in the source code rather than hidden inside an external library.

## What I implemented and why it matters

I designed the system around three goals from the assignment brief: **streaming compatibility**, **ensemble learning**, and **clear engineering** with tests and plots.

**Streaming learning.** Instead of loading the full dataset once, the demo reads a CSV file, splits it into chunks, and trains after each chunk. Preprocessing (imputation and scaling), the model, metrics, and plots all update incrementally. This matches the specification in `Assignment2.2-specs.md` and reflects how production pipelines often behave when data never stops arriving.

**Ensemble methods.** I implemented a single decision tree, bagging with bootstrap samples, and a random forest with feature subsampling. All three share the same pipeline interface so I can compare them fairly on the same stream of wine classification data.

**Numerical stability and testing.** I handled missing values, constant features, label encoding edge cases, and empty chunks. I wrote fifty-seven unit tests that run with pytest, including cases that only appear during streaming (for example a fixed class list across chunks, unseen one-hot categories, or rolling accuracy over the last two chunks).

## Package structure and what each module does

**`numcompute_stream/io.py`**  
Loads CSV files with NumPy, separates features and labels, and provides `stream_chunks()` so the rest of the code never assumes the full matrix is in memory at once.

**`numcompute_stream/preprocessing.py`**  
`StandardScaler` uses Welford’s method for running mean and variance. `Imputer` tracks column means as new rows arrive. `OneHotEncoder` can grow its vocabulary when new categories appear in a later chunk.

**`numcompute_stream/stats.py`**  
Supports `update_stats()` for running means, variances, and histogram based quantiles, which the notebook uses before modelling to show descriptive streaming statistics.

**`numcompute_stream/tree.py`**  
`DecisionTreeClassifier` supports Gini or entropy impurity, depth limits, minimum split size, and `max_features` for random forest style subsampling. Trees grow incrementally: samples sit in leaf buffers until a split is possible.

**`numcompute_stream/ensemble.py`**  
`EnsembleClassifier` is the shared base. `BaggingClassifier` trains several trees on bootstrap samples of each chunk. `RandomForestClassifier` adds random feature subsets per tree.

**`numcompute_stream/metrics.py`**  
`StreamingMetrics` accumulates accuracy, precision, recall, F1, a confusion matrix, multiclass AUC, and optional rolling accuracy across chunks. It exposes `update()`, `reset()`, and `result()` exactly as the spec describes.

**`numcompute_stream/pipeline.py`**  
Chains transformers and a classifier. Each step receives `partial_fit()` in order, then `predict()` runs through the same chain.

**`numcompute_stream/stream.py`**  
`StreamTrainer` is the main orchestrator for demos. `fit_chunk()` trains and scores one batch; `score_chunk()` updates metrics without retraining. Logs include per chunk accuracy, cumulative accuracy, and memory estimates via `tracemalloc`.

**`numcompute_stream/visualise.py`**  
Provides `plot_metric_over_time`, `compare_models`, and `plot_predictions_vs_ground_truth`, plus helpers to save figures for the demo script.

## Deliverables in this repository

| Deliverable | Location |
|---|---|
| Source code | `numcompute_stream/` |
| Unit tests (57 tests) | `tests/` |
| Demo notebook | `demo/stream_demo.ipynb` |
| Demo script (no Jupyter required) | `demo/run_stream_demo.py` |
| Sample data | `demo/data/wine_stream.csv` |
| Benchmark | `benchmark/benchmark_stream.py` |
| Report and video | Submitted separately on Canvas |
| Submission checklist | `SUBMISSION.md` |
| Official spec copy | `Assignment2.2-specs.md` |

## How to set up the environment

Open PowerShell in the project folder and run:

```powershell
cd "ProgrammingForAI - Assignment2"
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip numpy matplotlib pytest fpdf2
pip install -e .
```

The editable install (`pip install -e .`) lets Python import `numcompute_stream` from anywhere inside the project.

## How to verify everything before you submit

These commands are what I run before uploading to Canvas:

```powershell
python -m pytest -q
python benchmark\benchmark_stream.py
python demo\run_stream_demo.py
```

**pytest** should report fifty seven passed tests.  
**benchmark** prints vectorised versus loop timing for confusion matrix, Welford stats, label encoding, one-hot encoding, and ensemble voting.  
**demo** prints per chunk logs for four models on 178 wine samples and saves plots under `demo/output/`.

Latest local verification run (Jun 8, 2026): `57 passed`, benchmark completed with vectorised speedups, and demo finished successfully with output figures saved in `demo/output/`.

## Minimal code example

This snippet shows the core idea: a pipeline inside a stream trainer, trained chunk by chunk.

```python
import numpy as np
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.ensemble import RandomForestClassifier
from numcompute_stream.stream import StreamTrainer

X = np.random.default_rng(0).normal(size=(80, 4))
y = np.random.default_rng(1).integers(0, 3, size=80)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=3)),
])
trainer = StreamTrainer(pipe)
for i in range(0, 80, 20):
    trainer.fit_chunk(X[i : i + 20], y[i : i + 20])
print(trainer.results()["accuracy"])
```

## Project layout

```text
numcompute_stream/          Main Python package
tests/                    Fifty seven pytest tests
demo/
  stream_demo.ipynb       Interactive walkthrough with plots
  run_stream_demo.py      Same story in plain Python
  data/wine_stream.csv    Full UCI wine dataset (178 rows)
  output/                 Figures created by the demo script
benchmark/
  benchmark_stream.py     Vectorised vs loop and model comparison
scripts/
  export_report_pdf.py
  export_video_script_docx.py
README.md
SUBMISSION.md
Assignment2.2-specs.md
pyproject.toml
```

## Use of generative AI

I used generative AI to help with test ideas and drafting documentation. I ran every script and test locally, read through the generated code, and adjusted design and wording so the submission reflects work I understand and can explain in the demo video.
