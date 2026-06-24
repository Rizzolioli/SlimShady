"""
generate_decay_xo_tables.py
===========================
Single-table comparison of cosine² decay vs fixed p_xo=0.70 at gen=400.

Ratio = (decay_median - fixed_median) / fixed_median
  negative  →  green  (decay improves)
  positive  →  orange (decay worsens)

Bold = Mann-Whitney U significant (p < 0.05, two-sided).

Output: main/log/latex_final/decay_xo/
    improvement_table.png
    improvement_table.html
    improvement_table.csv
"""

import os, math
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_LOG_DIR   = os.path.join(_HERE, "..", "log")
_OUT       = os.path.join(_LOG_DIR, "latex_final", "decay_xo")
_DECAY_CSV = os.path.join(_LOG_DIR, "results_decay_xo_22062026.csv")
_FIXED_CSV = os.path.join(_LOG_DIR, "results_op_stats_19062026.csv")

# ── config ────────────────────────────────────────────────────────────────────
_COLS9 = {
    0:'algo',1:'run_id',2:'dataset',3:'seed',4:'gen',
    5:'train_fit',6:'timing',7:'nodes',8:'test_fit',9:'nodes_count',
    10:'inflate_n',11:'inflate_improved',12:'deflate_n',13:'deflate_improved',
    14:'xo_n',15:'xo_improved',16:'log_level',
}
_VARIANTS  = ["SLIM+2SIG", "SLIM*ABS", "SLIM*1SIG"]
_VAR_DISP  = {"SLIM+2SIG": "SLIM+2SIG", "SLIM*ABS": "SLIM*ABS", "SLIM*1SIG": "SLIM*1SIG"}
_DATASETS  = ["toxicity", "concrete", "instanbul", "ppb",
              "resid_build_sale_price", "energy"]
_DS_DISP   = {
    "toxicity":             "Toxicity",
    "concrete":             "Concrete",
    "instanbul":            "Istanbul",
    "ppb":                  "PPB",
    "resid_build_sale_price": "Resid. Build",
    "energy":               "Energy",
}
_GEN  = 400
_ALPHA = 0.05

# colour maps: white → deep green / orange
_CMAP_NEG = mcolors.LinearSegmentedColormap.from_list("neg", ["#f7fbf5","#1a7a3c"])
_CMAP_POS = mcolors.LinearSegmentedColormap.from_list("pos", ["#fffaf0","#d4580a"])

# ── loader ────────────────────────────────────────────────────────────────────
def _load(path):
    df = pd.read_csv(path, header=None).rename(columns=_COLS9)
    df["seed"] = df["seed"].astype(int)
    df = df.drop_duplicates(subset=["algo","dataset","seed","gen"], keep="last")
    df["variant"] = df["algo"].str.extract(r'^(SLIM[+*]\w+)_pop')
    return df[df["variant"].isin(_VARIANTS)]

# ── compute ratios at gen=400 ─────────────────────────────────────────────────
def compute_ratios(df_decay, df_fixed):
    rows = []
    for ds in _DATASETS:
        for var in _VARIANTS:
            entry = {"dataset": ds, "variant": var}
            for metric in ("test_fit", "nodes_count"):
                d = df_decay[(df_decay["dataset"]==ds)&(df_decay["variant"]==var)
                             &(df_decay["gen"]==_GEN)][metric].dropna()
                f = df_fixed[(df_fixed["dataset"]==ds)&(df_fixed["variant"]==var)
                             &(df_fixed["gen"]==_GEN)][metric].dropna()
                if d.empty or f.empty or f.median()==0:
                    entry[metric+"_ratio"] = float("nan")
                    entry[metric+"_p"]     = float("nan")
                    entry[metric+"_sig"]   = False
                    continue
                ratio = (d.median() - f.median()) / f.median()
                try:
                    _, p = stats.mannwhitneyu(d.values, f.values, alternative="two-sided")
                except ValueError:
                    p = float("nan")
                entry[metric+"_ratio"] = round(ratio, 4)
                entry[metric+"_p"]     = round(p, 4) if not math.isnan(p) else float("nan")
                entry[metric+"_sig"]   = (not math.isnan(p)) and (p < _ALPHA)
            rows.append(entry)
    return pd.DataFrame(rows)

# ── colour helper ─────────────────────────────────────────────────────────────
def _val_color(val, max_abs):
    if math.isnan(val):
        return "#e8e8e8"
    intensity = min(abs(val) / max_abs, 1.0) ** 0.6
    cmap = _CMAP_NEG if val < 0 else _CMAP_POS
    return mcolors.to_hex(cmap(0.2 + 0.8 * intensity))

def _text_color(bg_hex):
    r, g, b = mcolors.to_rgb(bg_hex)
    lum = 0.299*r + 0.587*g + 0.114*b
    return "white" if lum < 0.45 else "#1a1a1a"

# ── draw table ────────────────────────────────────────────────────────────────
def draw_table(df, filename):
    n_ds  = len(_DATASETS)
    n_var = len(_VARIANTS)
    n_rows = n_ds * n_var      # 18 data rows
    n_cols = 4                 # Dataset | Variant | Test RMSE | Model size

    # layout (normalised 0-1)
    col_x     = [0.00, 0.24, 0.50, 0.75]   # left edges of cols
    col_w     = [0.24, 0.26, 0.25, 0.25]   # widths
    header_h  = 0.065
    row_h     = (1.0 - header_h) / n_rows
    fig_w, fig_h = 9.5, 7.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    max_abs_fit   = df["test_fit_ratio"].abs().replace(float("nan"), 0).max()
    max_abs_nodes = df["nodes_count_ratio"].abs().replace(float("nan"), 0).max()
    max_abs_fit   = max(max_abs_fit,   0.01)
    max_abs_nodes = max(max_abs_nodes, 0.01)

    def add_rect(x, y, w, h, fc, ec="#bbbbbb", lw=0.5, zorder=1):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="square,pad=0",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
        ))

    def add_text(x, y, w, h, txt, fs=9, fw="normal", fc="#1a1a1a",
                 ha="center", va="center"):
        ax.text(x + w/2, y + h/2, txt,
                fontsize=fs, fontweight=fw, color=fc,
                ha=ha, va=va, clip_on=True)

    # ── header row ────────────────────────────────────────────────────────────
    hy = 1.0 - header_h
    header_labels = ["Dataset", "Variant", "Test RMSE ratio", "Model size ratio"]
    for ci, (lbl, cx, cw) in enumerate(zip(header_labels, col_x, col_w)):
        add_rect(cx, hy, cw, header_h, fc="#2c3e50", ec="#1a252f", lw=1.0, zorder=2)
        add_text(cx, hy, cw, header_h, lbl, fs=10, fw="bold", fc="white")

    # ── subtitle under metric headers ─────────────────────────────────────────
    subtitle_y = hy - 0.025
    for ci in [2, 3]:
        ax.text(col_x[ci] + col_w[ci]/2, subtitle_y,
                "(decay − fixed) / fixed",
                fontsize=6.5, color="#555555", ha="center", va="center",
                style="italic")

    # ── data rows ─────────────────────────────────────────────────────────────
    DS_BG = ["#f8f9fa", "#ffffff"]   # alternating dataset blocks

    for di, ds in enumerate(_DATASETS):
        ds_bg = DS_BG[di % 2]
        for vi, var in enumerate(_VARIANTS):
            ri = di * n_var + vi
            ry = 1.0 - header_h - (ri + 1) * row_h

            row = df[(df["dataset"]==ds) & (df["variant"]==var)].iloc[0]
            fit_val   = row["test_fit_ratio"]
            fit_sig   = row["test_fit_sig"]
            node_val  = row["nodes_count_ratio"]
            node_sig  = row["nodes_count_sig"]

            # Dataset label cell (merged across 3 variant rows — only draw bg/text for first)
            ds_cell_h = n_var * row_h
            ds_cell_y = 1.0 - header_h - (di * n_var + n_var) * row_h
            if vi == 0:
                add_rect(col_x[0], ds_cell_y, col_w[0], ds_cell_h,
                         fc=ds_bg, ec="#888888", lw=0.8, zorder=2)
                ax.text(col_x[0] + col_w[0]/2,
                        ds_cell_y + ds_cell_h/2,
                        _DS_DISP.get(ds, ds),
                        fontsize=10, fontweight="bold", color="#2c3e50",
                        ha="center", va="center")

            # Variant cell
            var_bg = "#eef2f7" if vi % 2 == 0 else ds_bg
            add_rect(col_x[1], ry, col_w[1], row_h, fc=var_bg,
                     ec="#cccccc", lw=0.4, zorder=2)
            ax.text(col_x[1] + 0.012, ry + row_h/2,
                    _VAR_DISP.get(var, var),
                    fontsize=9, fontweight="semibold", color="#333333",
                    ha="left", va="center")

            # Value cells
            for ci, (val, sig, ma) in enumerate(
                    [(fit_val, fit_sig, max_abs_fit),
                     (node_val, node_sig, max_abs_nodes)]):
                cx = col_x[2 + ci]
                cw = col_w[2 + ci]
                bg = _val_color(val, ma)
                add_rect(cx, ry, cw, row_h, fc=bg, ec="#cccccc", lw=0.4, zorder=2)
                if math.isnan(val):
                    txt = "—"
                    fw  = "normal"
                else:
                    txt = f"{val:+.3f}"
                    fw  = "bold" if sig else "normal"
                tc = _text_color(bg)
                # significance marker
                if sig:
                    ax.text(cx + cw - 0.008, ry + row_h*0.82, "★",
                            fontsize=6, color=tc, ha="right", va="center",
                            zorder=3)
                ax.text(cx + cw/2, ry + row_h/2, txt,
                        fontsize=9.5, fontweight=fw, color=tc,
                        ha="center", va="center", zorder=3)

        # thick separator line between dataset blocks
        sep_y = 1.0 - header_h - (di + 1) * n_var * row_h
        ax.axhline(sep_y, color="#666666", linewidth=1.0, zorder=4)

    # outer border
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 1.0 - header_h - n_rows * row_h), 1.0, header_h + n_rows * row_h,
        boxstyle="square,pad=0", fill=False, edgecolor="#333333", linewidth=1.5, zorder=5,
    ))

    # ── legend ────────────────────────────────────────────────────────────────
    legend_y = 1.0 - header_h - n_rows * row_h - 0.055
    swatch_w, swatch_h = 0.022, 0.025
    items = [
        (_CMAP_NEG(0.85), "Decay improves  (ratio < 0)"),
        (_CMAP_POS(0.85), "Decay worsens   (ratio > 0)"),
    ]
    lx = 0.02
    for color, label in items:
        add_rect(lx, legend_y, swatch_w, swatch_h, fc=mcolors.to_hex(color),
                 ec="#888", lw=0.5, zorder=6)
        ax.text(lx + swatch_w + 0.010, legend_y + swatch_h/2,
                label, fontsize=8, va="center", color="#333333")
        lx += 0.28

    ax.text(lx, legend_y + swatch_h/2,
            "★  Mann-Whitney U  p < 0.05",
            fontsize=8, va="center", color="#555555")

    # ── title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        "Improvement ratio: cosine² decay (0.70→0.30) vs fixed pₓₒ=0.70  "
        "—  gen 400",
        fontsize=11, fontweight="bold", y=1.01,
    )

    os.makedirs(_OUT, exist_ok=True)
    path = os.path.join(_OUT, filename)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    print(f"  saved: {filename}")
    plt.close(fig)

# ── HTML output ───────────────────────────────────────────────────────────────
def write_html(df, filename):
    max_abs_fit   = max(df["test_fit_ratio"].abs().replace(float("nan"),0).max(), 0.01)
    max_abs_nodes = max(df["nodes_count_ratio"].abs().replace(float("nan"),0).max(), 0.01)

    html = """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
body{font-family:Arial,sans-serif;font-size:13px;padding:20px;background:#fafafa}
table{border-collapse:collapse;box-shadow:0 1px 4px rgba(0,0,0,.15)}
th{background:#2c3e50;color:white;padding:8px 14px;font-size:13px}
td{padding:6px 14px;border:1px solid #d0d0d0;text-align:center}
.ds{font-weight:bold;font-size:13px;color:#2c3e50;background:#f0f4f8;
    vertical-align:middle;text-align:left;border-right:2px solid #aaa}
.var{text-align:left;padding-left:10px;font-size:12px;color:#333;
    background:#f8f9fa}
caption{caption-side:top;font-size:14px;font-weight:bold;
        margin-bottom:8px;color:#333}
</style></head><body>
<table>
<caption>Improvement ratio at gen 400: cosine&sup2; decay vs fixed p<sub>xo</sub>=0.70<br>
<small style="font-weight:normal">
(decay &minus; fixed) / fixed &nbsp;|&nbsp;
<span style="color:#1a7a3c">&#9646; negative = decay improves</span> &nbsp;
<span style="color:#d4580a">&#9646; positive = decay worsens</span> &nbsp;
<b>&#9733;</b> = Mann-Whitney p &lt; 0.05</small></caption>
<tr>
  <th>Dataset</th><th>Variant</th>
  <th>Test RMSE ratio</th><th>Model size ratio</th>
</tr>
"""
    for di, ds in enumerate(_DATASETS):
        row_bg = "#f8f9fa" if di % 2 == 0 else "#ffffff"
        for vi, var in enumerate(_VARIANTS):
            row = df[(df["dataset"]==ds)&(df["variant"]==var)].iloc[0]
            html += "<tr>"
            if vi == 0:
                html += (f'<td rowspan="3" class="ds">'
                         f'{_DS_DISP.get(ds, ds)}</td>')
            html += f'<td class="var">{_VAR_DISP.get(var, var)}</td>'
            for metric, ma in [("test_fit", max_abs_fit),
                                ("nodes_count", max_abs_nodes)]:
                val = row[f"{metric}_ratio"]
                sig = row[f"{metric}_sig"]
                bg  = _val_color(val, ma)
                tc  = _text_color(bg)
                txt = "—" if math.isnan(val) else f"{val:+.3f}"
                star = " &#9733;" if sig else ""
                bw = "bold" if sig else "normal"
                html += (f'<td style="background:{bg};color:{tc};'
                         f'font-weight:{bw}">{txt}{star}</td>')
            html += f'</tr>\n'

    html += "</table></body></html>"
    path = os.path.join(_OUT, filename)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"  saved: {filename}")

# ── CSV ───────────────────────────────────────────────────────────────────────
def write_csv(df):
    out = df[["dataset","variant",
              "test_fit_ratio","test_fit_sig","test_fit_p",
              "nodes_count_ratio","nodes_count_sig","nodes_count_p"]].copy()
    out["dataset"] = out["dataset"].map(_DS_DISP)
    path = os.path.join(_OUT, "improvement_table.csv")
    out.to_csv(path, index=False, float_format="%.4f")
    print(f"  saved: improvement_table.csv")

# ── terminal print ────────────────────────────────────────────────────────────
def print_table(df):
    print(f"\n{'Dataset':<14} {'Variant':<12}  {'Test RMSE':>12}  {'Nodes':>12}")
    print("-" * 55)
    for ds in _DATASETS:
        for vi, var in enumerate(_VARIANTS):
            row = df[(df["dataset"]==ds)&(df["variant"]==var)].iloc[0]
            ds_s = _DS_DISP.get(ds, ds) if vi == 0 else ""
            def fmt(v, s):
                if math.isnan(v): return "     --"
                return f"{'*' if s else ' '}{v:+.3f}"
            print(f"{ds_s:<14} {_VAR_DISP.get(var,var):<12}  "
                  f"{fmt(row['test_fit_ratio'], row['test_fit_sig']):>12}  "
                  f"{fmt(row['nodes_count_ratio'], row['nodes_count_sig']):>12}")
        print()

# ── entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df_decay = _load(_DECAY_CSV)
    df_fixed = _load(_FIXED_CSV)

    print(f"Computing improvement ratios at gen={_GEN}...")
    df = compute_ratios(df_decay, df_fixed)

    os.makedirs(_OUT, exist_ok=True)

    print("\nGenerating PNG table...")
    draw_table(df, "improvement_table.png")

    print("Generating HTML table...")
    write_html(df, "improvement_table.html")

    print("Writing CSV...")
    write_csv(df)

    print_table(df)
    print("Done. Files in:", _OUT)
