"""
generate_results_table.py
=========================
Summarise all head-XO experiments: train RMSE, test RMSE, model size
at the last generation (median ± IQR across seeds).

One CSV + one LaTeX table per experiment, plus a combined CSV.
Output: main/log/latex/tables/
"""

import os
import re
import pandas as pd
import numpy as np

# ── PATHS ─────────────────────────────────────────────────────────────────────

_HERE    = os.path.dirname(os.path.abspath(__file__))
_LOG     = os.path.join(_HERE, "..", "log")
_OUT     = os.path.join(_LOG, "latex", "tables")
os.makedirs(_OUT, exist_ok=True)

# ── HELPERS ───────────────────────────────────────────────────────────────────

_COLS = {0:"algo",1:"run_id",2:"dataset",3:"seed",4:"gen",
         5:"train",6:"timing",7:"nodes",8:"test",9:"nodes_count",10:"log"}

def _load(path):
    df = pd.read_csv(path, header=None).rename(columns=_COLS)
    df["seed"] = df["seed"].astype(int)
    return df


def _last_gen(df):
    """Keep only the final generation row per (algo, dataset, seed).
    For pop_xo the last gen varies per algo; for all others it is 2000."""
    last = (df.groupby(["algo", "dataset", "seed"])["gen"]
              .max()
              .reset_index()
              .rename(columns={"gen": "last_gen"}))
    df = df.merge(last, on=["algo", "dataset", "seed"])
    return df[df["gen"] == df["last_gen"]].copy()


def _stats(df):
    """Median + IQR over seeds for train, test, nodes_count."""
    def iqr(x):
        return x.quantile(0.75) - x.quantile(0.25)

    agg = (df.groupby(["algo", "dataset"])
             .agg(
                 train_med  =("train",       "median"),
                 train_iqr  =("train",       iqr),
                 test_med   =("test",        "median"),
                 test_iqr   =("test",        iqr),
                 size_med   =("nodes_count", "median"),
                 size_iqr   =("nodes_count", iqr),
                 n_seeds    =("seed",        "nunique"),
             )
             .reset_index())
    return agg


def _fmt(med, iqr, decimals=2):
    """Format as 'med (iqr)' string."""
    fmt = f"{{:.{decimals}f}}"
    return (fmt + " ({})").format(med, fmt.format(iqr))


def _to_latex(df_stats, caption, label, config_cols):
    """
    Build a LaTeX longtable string.

    config_cols : list of extra column names extracted from the algo string
                  to display between 'variant' and the metric columns.
    """
    metric_header = (
        r"\multicolumn{2}{c}{Train RMSE} & "
        r"\multicolumn{2}{c}{Test RMSE} & "
        r"\multicolumn{2}{c}{Model Size}"
    )
    sub_header = r"Med & IQR & Med & IQR & Med & IQR"

    col_spec = "ll" + "l" * len(config_cols) + "rrrrrr"
    config_head = " & ".join(c.replace("_", " ").title() for c in config_cols)

    lines = [
        r"\begin{longtable}{" + col_spec + "}",
        r"\caption{" + caption + r"} \label{" + label + r"} \\",
        r"\toprule",
        r"Dataset & Variant & " + (config_head + " & " if config_head else "") +
        metric_header + r" \\",
        r"\cmidrule(lr){" + str(3 + len(config_cols)) + "-" +
        str(4 + len(config_cols)) + r"}"
        r"\cmidrule(lr){" + str(5 + len(config_cols)) + "-" +
        str(6 + len(config_cols)) + r"}"
        r"\cmidrule(lr){" + str(7 + len(config_cols)) + "-" +
        str(8 + len(config_cols)) + r"}",
        r" & & " + (" & " * len(config_cols)) + sub_header + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Dataset & Variant & " + (config_head + " & " if config_head else "") +
        metric_header + r" \\",
        r" & & " + (" & " * len(config_cols)) + sub_header + r" \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]

    prev_dataset = None
    for _, row in df_stats.sort_values(["dataset", "variant"] + config_cols).iterrows():
        ds = row["dataset"] if row["dataset"] != prev_dataset else ""
        if prev_dataset is not None and row["dataset"] != prev_dataset:
            lines.append(r"\midrule")
        prev_dataset = row["dataset"]

        cfg_cells = " & ".join(str(row[c]) for c in config_cols)
        train = _fmt(row["train_med"], row["train_iqr"])
        test  = _fmt(row["test_med"],  row["test_iqr"])
        size  = _fmt(row["size_med"],  row["size_iqr"], decimals=0)

        row_str = f"{ds} & {row['variant']}"
        if cfg_cells:
            row_str += f" & {cfg_cells}"
        row_str += f" & {train} & {test} & {size} \\\\"
        lines.append(row_str)

    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def _save(df_stats, csv_name, tex_name, caption, label, config_cols):
    csv_path = os.path.join(_OUT, csv_name)
    df_stats.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  CSV  -> {csv_path}")

    tex = _to_latex(df_stats, caption, label, config_cols)
    tex_path = os.path.join(_OUT, tex_name)
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(tex)
    print(f"  TEX  -> {tex_path}")


# ══════════════════════════════════════════════════════════════════════════════
# EXP 1 — Scramble / periodic XO frequency
# ══════════════════════════════════════════════════════════════════════════════

def table_scramble():
    print("\n-- Scramble XO")
    df = _load(os.path.join(_LOG, "results_scramble_xo_05052026.csv"))
    df = _last_gen(df)

    df["variant"] = df["algo"].str.extract(r"^(SLIM[+*]\w+)_head_xo")
    df["xo_freq"] = df["algo"].str.extract(r"_head_xo(\w+)$")

    stats = _stats(df)
    meta  = df[["algo","variant","xo_freq"]].drop_duplicates("algo")
    stats = stats.merge(meta, on="algo")

    _save(stats, "scramble_xo_table.csv", "scramble_xo_table.tex",
          caption="Scramble XO: median (IQR) at final generation",
          label="tab:scramble_xo",
          config_cols=["xo_freq"])


# ══════════════════════════════════════════════════════════════════════════════
# EXP 2 — Head-size sweep
# ══════════════════════════════════════════════════════════════════════════════

def table_head_size():
    print("\n-- Head-size sweep")
    df = _load(os.path.join(_LOG, "results_head_size_07052026.csv"))
    df = _last_gen(df)

    df["variant"] = df["algo"].str.extract(r"^(SLIM[+*]\w+)_hd")
    df["max_depth"] = df["algo"].str.extract(r"_hd(\d+)_xo")
    df["xo_freq"]   = df["algo"].str.extract(r"_xo(\d+)$")

    stats = _stats(df)
    meta  = df[["algo","variant","max_depth","xo_freq"]].drop_duplicates("algo")
    stats = stats.merge(meta, on="algo")

    _save(stats, "head_size_table.csv", "head_size_table.tex",
          caption="Head-size sweep: median (IQR) at final generation",
          label="tab:head_size",
          config_cols=["max_depth", "xo_freq"])


# ══════════════════════════════════════════════════════════════════════════════
# EXP 3 — Probabilistic XO
# ══════════════════════════════════════════════════════════════════════════════

def table_prob_xo():
    print("\n-- Probabilistic XO")
    df = _load(os.path.join(_LOG, "results_prob_xo_12052026.csv"))
    df = _last_gen(df)

    df["variant"] = df["algo"].str.extract(r"^(SLIM[+*]\w+)_pxo")
    df["p_xo"]    = df["algo"].str.extract(r"_pxo([0-9.]+)$")

    stats = _stats(df)
    meta  = df[["algo","variant","p_xo"]].drop_duplicates("algo")
    stats = stats.merge(meta, on="algo")

    _save(stats, "prob_xo_table.csv", "prob_xo_table.tex",
          caption="Probabilistic XO (max\\_depth=17): median (IQR) at final generation",
          label="tab:prob_xo",
          config_cols=["p_xo"])


# ══════════════════════════════════════════════════════════════════════════════
# EXP 4 — Population / budget sweep
# ══════════════════════════════════════════════════════════════════════════════

def table_pop_xo():
    print("\n-- Pop/budget sweep")
    df = _load(os.path.join(_LOG, "results_pop_xo_15052026.csv"))
    df = _last_gen(df)   # last gen varies per algo (200 / 400 / 1000)

    df["variant"] = df["algo"].str.extract(r"^(SLIM[+*]\w+)_pop")
    df["pop"]     = df["algo"].str.extract(r"_pop(\d+)_")
    df["n_iter"]  = df["algo"].str.extract(r"_iter(\d+)_")

    stats = _stats(df)
    meta  = df[["algo","variant","pop","n_iter"]].drop_duplicates("algo")
    stats = stats.merge(meta, on="algo")

    _save(stats, "pop_xo_table.csv", "pop_xo_table.tex",
          caption="Population/budget sweep (p\\_xo=0.7, max\\_depth=17): median (IQR) at final generation",
          label="tab:pop_xo",
          config_cols=["pop", "n_iter"])


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED CSV
# ══════════════════════════════════════════════════════════════════════════════

def table_combined():
    print("\n-- Combined CSV")
    frames = []

    specs = [
        ("results_scramble_xo_05052026.csv", "scramble_xo",
         r"^(SLIM[+*]\w+)_head_xo", {"xo_freq": r"_head_xo(\w+)$"}),
        ("results_head_size_07052026.csv", "head_size",
         r"^(SLIM[+*]\w+)_hd",    {"max_depth": r"_hd(\d+)_xo", "xo_freq": r"_xo(\d+)$"}),
        ("results_prob_xo_12052026.csv", "prob_xo",
         r"^(SLIM[+*]\w+)_pxo",   {"p_xo": r"_pxo([0-9.]+)$"}),
        ("results_pop_xo_15052026.csv", "pop_xo",
         r"^(SLIM[+*]\w+)_pop",   {"pop": r"_pop(\d+)_", "n_iter": r"_iter(\d+)_"}),
    ]

    for fname, exp_label, variant_pat, cfg_pats in specs:
        df = _load(os.path.join(_LOG, fname))
        df = _last_gen(df)
        df["variant"]    = df["algo"].str.extract(variant_pat)
        df["experiment"] = exp_label
        for col, pat in cfg_pats.items():
            df[col] = df["algo"].str.extract(pat)

        stats = _stats(df)
        meta_cols = ["algo", "experiment", "variant"] + list(cfg_pats.keys())
        meta  = df[meta_cols].drop_duplicates("algo")
        stats = stats.merge(meta, on="algo")
        frames.append(stats)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    # put key columns first
    front = ["experiment", "dataset", "variant", "algo",
             "train_med","train_iqr","test_med","test_iqr","size_med","size_iqr","n_seeds"]
    rest  = [c for c in combined.columns if c not in front]
    combined = combined[front + rest]

    path = os.path.join(_OUT, "all_experiments_table.csv")
    combined.to_csv(path, index=False, float_format="%.4f")
    print(f"  CSV  -> {path}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    table_scramble()
    table_head_size()
    table_prob_xo()
    table_pop_xo()
    table_combined()
    print(f"\nAll tables written to {_OUT}")
