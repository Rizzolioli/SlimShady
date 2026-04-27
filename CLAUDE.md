# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

**SlimShady** is a PhD research implementation of **SLIM-GSGP** (Structured Locally Interpretable Models – Geometric Semantic Genetic Programming), a symbolic regression algorithm. It also implements baseline comparators: classic GP, GSGP, and SSHC (Stochastic Search Hill Climbing). Computation is PyTorch-tensor-based throughout for speed.

## Running Experiments

There is no build step. Run experiment scripts directly:

```bash
cd main
python main_slim.py       # SLIM_GSGP (primary algorithm)
python main_gsgp.py       # GSGP baseline
python main_gp.py         # GP baseline
python main_sklearn.py    # Scikit-learn baselines
```

Experiment variants (feature selection, convex hull, mutation step analysis, missing values, etc.) follow the same pattern: `python main_slim_<variant>.py`.

There is no test suite or linter configured. Validation is done by inspecting results logged to `main/log/` as CSV files.

## Key Dependencies

PyTorch (`torch`), NumPy, Pandas, scikit-learn. No `requirements.txt` — install manually if needed.

## Architecture

### Algorithm Flow

Each algorithm follows the same pattern: instantiate an optimizer class, call `.solve()`.

```
main/main_slim.py
  └─ parametrization.py          # all hyperparameters & dataset-specific configs
  └─ datasets/data_loader.py     # loads pre-split tensors (15 seeds × 15 datasets)
  └─ algorithms/SLIM_GSGP/slim_gsgp.py   # optimizer
       └─ representations/population.py  # population of Individuals
       └─ representations/individual.py  # block-based individual (list of Trees)
       └─ operators/mutators.py          # inflate / deflate mutations
       └─ operators/selection_algorithms.py
  └─ evaluators/fitness_functions.py     # RMSE, MAE, MSE
  └─ utils/logger.py                     # CSV result logging (UUID per run)
```

### SLIM_GSGP Individual Representation

An `Individual` is a **list of GSGP trees** (blocks) combined by a configurable operator (`'sum'` or `'product'`). Mutations either **inflate** (add a new tree block) or **deflate** (remove one). Each `Tree` stores its semantic vector (output on training data) as a PyTorch tensor, enabling O(1) fitness recalculation after mutation.

### Parametrization

`main/parametrization.py` is the single source of truth for all hyperparameters. Dataset-specific overrides (e.g. `p_inflate`, `p_deflate`) are defined there as dictionaries keyed by dataset name. When adding a new experiment, update this file rather than hardcoding values in `main_slim_*.py`.

### Datasets

`datasets/data_loader.py` exposes `load_preloaded(dataset, seed)` which returns pre-split `(X_train, X_test, y_train, y_test)` as PyTorch tensors. Raw data lives in `datasets/data/`; preprocessed 80/20 splits for 15 seeds live in `datasets/pre_loaded_data/`.

### Logging

Each run writes a row to a CSV in `main/log/`, identified by a `uuid`. The logger (`utils/logger.py`) records generation-level metrics: train fitness, test fitness, diversity, tree depth, node count.

### Utilities

- `utils/utils.py` — protected arithmetic ops (division, sqrt, log), `train_test_split`
- `utils/diversity.py` — population semantic diversity
- `utils/TIE.py` — Tracking Individual Evolution
- `utils/convexhull.py` — convex hull distance analysis (used in `chull` experiments)
