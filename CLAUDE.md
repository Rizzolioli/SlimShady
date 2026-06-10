# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

**SlimShady** is a PhD research implementation of **SLIM-GSGP** (Structured Locally Interpretable Models – Geometric Semantic Genetic Programming) for symbolic regression, plus baseline comparators (GP, GSGP). All computation is PyTorch-tensor-based for speed. The main research thread is a **head crossover (XO)** operator applied to the GP head tree of each individual.

## Running Experiments

No build step. Run scripts directly from the repo root (not from inside `main/`):

```bash
python main/main_slim.py          # plain SLIM-GSGP baseline
python main/main_scramble_xo.py   # periodic head XO, sweep xo_freq
python main/main_head_size.py     # periodic head XO, sweep max_head_depth
python main/main_prob_xo.py       # probabilistic head XO, sweep p_xo
python main/main_pop_xo.py        # probabilistic head XO, sweep pop/budget
python main/main_depth_cap.py     # depth-cap sweep at fixed p_xo
```

All experiment scripts use `ProcessPoolExecutor` and write to `main/log/results_<experiment>_<date>.csv`. They detect already-completed runs on restart (matching on `(algo, dataset, seed)` at `n_iter` generations), so interrupted runs are safely resumable.

### Analysis

```bash
python main/analysis/generate_results_table.py   # summary tables for all experiments
python main/analysis/generate_depth_cap_plots.py # convergence + size plots for depth-cap
python main/analysis/generate_report_figs.py     # figures for all head-XO experiments
python main/analysis/run_stn.py                  # end-to-end STN pipeline (train fitness)
python main/analysis/run_stn_test_fitness.py     # same pipeline with test fitness on nodes
python main/analysis/generate_stn_grids.py       # 3x2 grid figures from pre-built pkl files
```

STN pipeline requires `log=8` experiment output, which produces a companion `_sem_gen.csv` file alongside the main log. The `run_stn_test_fitness.py` script must be run on the machine that has these sem_gen files.

## Log CSV Format

All experiment logs are headerless CSVs with this column order:

```
col 0:  algo          (string, e.g. "SLIM+2SIG_pxo0.7_hd17")
col 1:  run_id        (UUID — shared across all tasks in one batch, not per-run)
col 2:  dataset       (string)
col 3:  seed          (int)
col 4:  gen           (int, 1-indexed)
col 5:  train_fit     (float, RMSE)
col 6:  timing        (float, seconds)
col 7:  nodes         (int, elite tree depth)
col 8:  test_fit      (float, RMSE)
col 9:  nodes_count   (int, total node count of elite individual)
col 10: log_level     (int)
```

When `log=8`, a companion `{log_path}_sem_gen.csv` is also written (elite genotype + semantics, only when elite changes).

## Architecture

### Individual Representation

A `SLIM_GSGP` `Individual` is a **list of GSGP `Tree` blocks** (`collection`). The combining operator is either `sum` or `product` over the block semantics. Key structural invariant:

- `collection[0]` is always the **original GP tree from initialization** — a `Tree` with `structure` as a **tuple** (from `rhh` initializer). It is never modified by inflate/deflate mutations.
- `collection[1:]` are inflate-mutation blocks — `Tree` objects with `structure` as a **list** `[operator_fn, tree1, tree2_or_ms, ...]`.

This distinction matters for head XO: `slim_head_crossover` checks `isinstance(h1.structure, tuple)` to guard against operating on inflated blocks. The `Tree.depth` and `Tree.nodes` attributes work for both tuple and list structures (`GSGP/representations/tree.py` lines 22–30).

### Head Crossover Operator

Defined in `algorithms/SLIM_GSGP/operators/crossover_operators.py`. Two mechanisms, both using `slim_head_crossover(FUNCTIONS, max_head_depth)`:

- **Periodic (`head_xo_freq`)**: entire offspring batch uses XO at generations `it % head_xo_freq == 0`.
- **Probabilistic (`p_xo`)**: each offspring independently uses XO with probability `p_xo`; fallback is inflate/deflate mutation.

Both are instantiated in `slim_gsgp.py` lines 249–250. If a XO offspring would exceed `max_head_depth`, the operator returns the original parents unchanged (depth cap, line 26).

### Semantics Are Cached on Trees

Each `Tree` stores its output vector (`train_semantics`, `test_semantics`) as a PyTorch tensor. `Individual.train_semantics` is a stacked 2-D tensor (blocks × n_train). Fitness evaluation is O(1) after mutation because only the new block's semantics need computing. `head_xo` pre-computes semantics eagerly so XO generations have the same cost as mutation generations.

### Variant Naming Convention

Variants are identified by `(sig, two_trees, operator, gsgp)` tuples. The canonical names are:

| Tuple | Name |
|---|---|
| `(True, True, "sum", False)` | `SLIM+2SIG` |
| `(True, False, "sum", False)` | `SLIM+1SIG` |
| `(False, False, "sum", False)` | `SLIM+ABS` |
| `(True, False, "mul", False)` | `SLIM*1SIG` |
| `(False, False, "mul", False)` | `SLIM*ABS` |

Algo strings in logs encode config as suffixes, e.g. `SLIM+2SIG_pxo0.7_hd17`. STN file names sanitize `+` and `*` via `re.sub(r'[^A-Za-z0-9_\-]', '_', algo)` → `SLIM_2SIG`, `SLIM_ABS`, etc.

### Dataset-Specific Parameters

Each experiment script defines a `_dataset_params` dict keyed by dataset name with a fallback `"other"` key. Datasets with unusual behaviour: `toxicity` uses low `p_inflate=0.1` and a tight mutation step; `concrete` uses high `p_inflate=0.5`. Seeds are 1-indexed when passed to `load_preloaded` (`seed + 1`) but 0-indexed in log files.

### STN Pipeline

Search Trajectory Networks are built in three stages:
1. `stn_prep.py` — extracts elite-change rows from `_sem_gen.csv`, joins fitness from main log, writes per-algo CSVs to `stn_data/{benchmark}/`
2. `stn_build.py` — builds `genotype`, `hypercube`, and `clustering` (k-means) STN graphs, saves as `{alg}_{model}_stn.pkl` in `stns/{benchmark}/`
3. `generate_stn_grids.py` — loads pkl files and renders 3×2 grid figures (rows = SLIM+2SIG / SLIM*ABS / SLIM*1SIG; cols = baseline p_xo=0.0 / treatment p_xo=0.7)

The three study variants for STN figures are `SLIM+2SIG`, `SLIM*ABS`, `SLIM*1SIG`. The six benchmark datasets are `toxicity`, `concrete`, `instanbul`, `ppb`, `resid_build_sale_price`, `energy`.

## Key Findings (head XO experiments)

- With `head_xo_freq=500` (periodic, 4 XO events per run): `max_head_depth=17` and `max_head_depth=25` are byte-for-byte identical — the depth cap never fires because XO on `rhh`-initialized trees (init_depth=6) naturally stays ≤ ~12 deep. `max_head_depth=5` diverges at exactly gen 500.
- With `p_xo=0.3` (probabilistic): the depth cap fires in only ~5/30 runs for most variants; hd=17 and hd=25 are nearly equivalent.
- With `p_xo=0.7` (high frequency): the cap at hd=17 becomes active — populations diverge and model sizes differ (hd=17 produces more compact models than hd=25 with similar fitness).
