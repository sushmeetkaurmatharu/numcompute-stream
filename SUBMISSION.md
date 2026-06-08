# Assignment 2.2 Submission Checklist

**Student:** Sushmeet Kaur Matharu  
**Course:** COMP 5004 Programming for AI and Machine Learning

This file is my personal checklist before I upload to Canvas. It lists what is inside the repository and what I still submit as separate files on the learning management system.

## What is included in the code package

I completed every module named in `Assignment2.2-specs.md`. The table below maps each requirement to the file I wrote.

| Requirement | File | Notes |
|---|---|---|
| Core package | `numcompute_stream/` | All streaming APIs |
| Streaming trainer | `numcompute_stream/stream.py` | `fit_chunk`, `score_chunk`, memory logs |
| Decision tree | `numcompute_stream/tree.py` | Gini and entropy, depth and feature limits |
| Ensembles | `numcompute_stream/ensemble.py` | Bagging and random forest |
| Preprocessing | `numcompute_stream/preprocessing.py` | Scaler, imputer, one hot encoder |
| Statistics | `numcompute_stream/stats.py` | `update_stats` and running summaries |
| Metrics | `numcompute_stream/metrics.py` | `update`, `reset`, `result`, AUC, rolling window |
| Pipeline | `numcompute_stream/pipeline.py` | Chained `partial_fit` |
| Visualisation | `numcompute_stream/visualise.py` | Spec plots and save helpers |
| CSV I/O | `numcompute_stream/io.py` | Load, split labels, chunk iterator |
| Unit tests | `tests/` | Fifty seven tests |
| Demo notebook | `demo/stream_demo.ipynb` | Full workflow with figures |
| Demo script | `demo/run_stream_demo.py` | Same workflow without Jupyter |
| Benchmark | `benchmark/benchmark_stream.py` | Vectorised vs loop comparison |
| README | `README.md` | Setup and project overview |
| Technical report and video | Submitted separately on Canvas | Follow assignment upload rules |
| Course spec copy | `Assignment2.2-specs.md` | Provided specification |

## What I upload separately on Canvas

Canvas asks for three items. I upload them one at a time, then press Submit when all three are attached.

**1. Git repository as a zip file or a link**  
I zip the folders and files listed in the command below. I never include `.venv`, `__pycache__`, or `.pytest_cache` because those are local environment files, not part of my submission.

**2. Technical report as a PDF**  
I prepare this separately as a PDF upload for Canvas and confirm it is at most four pages.

**3. Demo video (maximum five minutes)**  
I record this separately for Canvas upload. The video shows setup, tests, the streaming demo, and matplotlib visualisations.

## Command to build the zip file

From the project folder in PowerShell:

```powershell
cd "ProgrammingForAI - Assignment2"
Compress-Archive -Path numcompute_stream, tests, demo, benchmark, scripts, README.md, SUBMISSION.md, Assignment2.2-specs.md, pyproject.toml, .gitignore -DestinationPath ..\NumCompute-Stream-A2.2-Submit.zip -Force
```

## Commands I run to verify before upload

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe benchmark\benchmark_stream.py
.\.venv\Scripts\python.exe demo\run_stream_demo.py
```

If pytest passes, the benchmark runs and reports timing comparisons, and the demo ends with `Done.`, I consider the code package ready for submission.
