# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

**SlimShady** is a PhD research implementation of **SLIM-GSGP** (Structured Locally Interpretable Models – Geometric Semantic Genetic Programming), a symbolic regression algorithm. It also implements baseline comparators: classic GP, GSGP, and SSHC (Stochastic Search Hill Climbing). Computation is PyTorch-tensor-based throughout for speed.

## Running Experiments

There is no build step. Run experiment scripts directly from the project root or `main/`:

```bash
cd main
python main_slim_normalized.py   # primary benchmark: 10 SLIM variants × 6 datasets × 30 seeds
python main_slim.py              # SLIM_GSGP (ad-hoc / single-variant runs)
python main_gsgp.py              # GSGP baseline
python main_gp.py                # GP baseline
python main_sklearn.py           # Scikit-learn baselines
```

`main_slim_normalized.py` has **resume detection**: reads `results_normalized_simplification.csv` on startup and skips `(algo, dataset, seed)` triples already present, so it is safe to interrupt and re-run.

### Smoke tests

```bash
python main/test_smoke.py   # run from the project root
```

Covers: `r2`, `compute_m_phi`, NORM1/NORM2 mutators (incl. α-scaling invariant), SymPy conversion, log-level-8 end-to-end run, `log_simplification` + `merge_simplification_logs`, geometry operators.

### Analysis pipeline (normalized experiment)

After running `main_slim_normalized.py`, run these in any order:

```bash
cd main
python plot_normalized_results.py    # convergence curves + last-gen boxplots → log/figs/
python table_normalized_results.py   # rankings table CSV + Excel → log/figs/
python performance_scatter.py        # 2D scatter: RMSE vs Size / RMSE vs M_phi → log/figs/
python simplification_effect.py      # SymPy simplification delta table + figure
python retry_simplification.py       # re-attempt previously failed simplifications
python geometry_study.py             # offspring geometry in 2D semantic space → log/geometry_study.png
```

## Key Dependencies

PyTorch (`torch`), NumPy, Pandas, scikit-learn, SymPy, openpyxl. No `requirements.txt` — install manually.

## Architecture

### Algorithm Flow

Each algorithm follows the same pattern: instantiate an optimizer class, call `.solve()`.

```
main/main_slim_normalized.py
  └─ datasets/data_loader.py          # loads pre-split tensors (30 seeds × 6 datasets)
  └─ algorithms/SLIM_GSGP/slim_gsgp.py   # optimizer class
       └─ representations/population.py  # population of Individuals
       └─ representations/individual.py  # block-based individual (list of Trees)
       └─ operators/mutators.py          # inflate / deflate mutations (incl. NORM1, NORM2)
       └─ operators/selection_algorithms.py
  └─ evaluators/fitness_functions.py  # rmse, mae, r2
  └─ utils/logger.py                  # CSV result logging, simplification logging
  └─ utils/utils.py                   # compute_m_phi, slim_individual_to_sympy, sympy_m_phi
```

### SLIM_GSGP Individual Representation

An `Individual` is a **list of GSGP Trees** (blocks) combined by a configurable operator (`'sum'` or `'product'`). Mutations either **inflate** (add a new tree block) or **deflate** (remove one). Each `Tree` stores its semantic vector (output on training data) as a PyTorch tensor, enabling O(1) fitness recalculation after mutation.

### Inflate mutation variants

| Name | Mutation step | Notes |
|---|---|---|
| `SLIM+2SIG` / `SLIM*2SIG` | `ms * (Tr1 − Tr2)` with sigmoid-based tree outputs | two random trees |
| `SLIM+1SIG` / `SLIM*1SIG` | `ms * (2*T − 1)` with sigmoid output | one random tree |
| `SLIM+ABS` / `SLIM*ABS` | `ms * (1 − 2/(1+|T|))` | one random tree |
| `SLIM+NORM1` / `SLIM*NORM1` | `ms * (2*(T−min)/(max−min) − 1)` | min-max normalizes T to [−1,1] on training; bounds frozen for test |
| `SLIM+NORM2` / `SLIM*NORM2` | `ms * α * (Tr1−Tr2)`, α = 1/max(|diff|) | scales diff to [−1,1] on training; α frozen for test |

`+` variants combine blocks with `sum`; `*` variants use `mul`.

### M_φ interpretability metric

```
M_φ = 79.1 − 0.2·ell − 0.5·no − 3.4·nnao − 4.5·nnaoc
```

- `ell` = total node count (leaves + operators)
- `no` = number of arithmetic operators
- `nnao` = non-arithmetic operators (exp, abs, sigmoid, …)
- `nnaoc` = 1 if nnao > 0 else 0

Higher = more interpretable. Can go negative for large models. Computed live via `compute_m_phi()` (`utils/utils.py`) and post-hoc on SymPy-simplified expressions via `sympy_m_phi()`.

**Analysis-side filter** (applied in all analysis/plot scripts, never in the algorithm): if SymPy makes the model larger or worse, revert to original:
```python
df['ell_after']   = df[['ell_after',   'ell_before'  ]].min(axis=1)
df['m_phi_after'] = df[['m_phi_after', 'm_phi_before']].max(axis=1)
```

### Log levels

`slim_gsgp.solve()` accepts a `log` parameter:
- `0` — disabled
- `1` — generation + train fitness only
- `8` — full: train RMSE, test RMSE, nodes_count, M_φ, no, nnao, nnaoc, MAE, R²

Log level 8 also triggers optional `simplify_elite=True`, which runs SymPy simplification on the final-generation elite and appends a row to `simplify_log_path`.

### Output CSVs (normalized experiment)

| File | Written by | Contents |
|---|---|---|
| `log/results_normalized_generations.csv` | logger, every generation | algo, run_id, dataset, seed, generation, train_fitness, timing, nodes, test_fitness, nodes_count, m_phi, no, nnao, nnaoc, mae, r2, log_level |
| `log/results_normalized_simplification.csv` | `log_simplification()`, final gen only | algo, run_id, dataset, seed, ell/m_phi/no/nnao/nnaoc before & after, genotype_before/after, simplified_ok, simp_time_s, test_rmse, test_mae, test_r2 |

### Datasets

`datasets/data_loader.py` exposes `load_preloaded(dataset, seed)` returning pre-split `(X_train, X_test, y_train, y_test)` as PyTorch tensors. Raw data: `datasets/data/`; preprocessed 80/20 splits: `datasets/pre_loaded_data/`. Six standard datasets: `concrete`, `energy`, `instanbul`, `ppb`, `resid_build_sale_price`, `toxicity`.

### Parametrization

`main/parametrization.py` is the source of truth for the non-normalized experiment scripts. For `main_slim_normalized.py`, all parameters are defined directly in the script: `VARIANTS`, `_DATASET_P_INFLATE`, and `ms = median(y_train)` (fixed scalar, not random).

### Utilities

- `utils/utils.py` — protected arithmetic, `compute_m_phi`, `slim_individual_to_sympy`, `sympy_m_phi`, `get_terminals`, `train_test_split`
- `utils/diversity.py` — population semantic diversity
- `utils/TIE.py` — Tracking Individual Evolution
- `utils/convexhull.py` — convex hull distance analysis
- `utils/logger.py` — `logger()`, `log_simplification()`, `merge_simplification_logs()`, `_SIMP_HEADER`

### Windows multiprocessing note

`main_slim_normalized.py` uses `multiprocessing.Pool` for outer parallelism (workers spawn no children). `retry_simplification.py` uses `ThreadPoolExecutor` (threads are not daemonic) so each thread can spawn a SymPy subprocess via `mp.Process + mp.Pipe`. Never use `Pool` workers to spawn subprocesses — they are daemonic and will raise "daemonic processes cannot have children".
