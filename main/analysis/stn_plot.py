"""
Visualise Search Trajectory Networks (STNs).
Translated from R ggraph code by Gabriela Ochoa, Josip Hrvatić & Magda Smolić-Ročak.

Loads pickle files produced by stn_build.py and saves combined PNG figures.

Layouts:
  "stress" / "kk"  — Kamada-Kawai force-directed (default)
  "fitness"         — KK for x, fitness value for y

Node size encodes either TreeSize or visit Count.
Edge width encodes transition Count.

Node types and colours  (matches R original):
  Start  → green  #4daf4a  filled square
  Medium → gray   #333333  open circle
  End    → blue   #377eb8  filled triangle
  Best   → red    #e41a1c  filled circle
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# ── VISUAL SETTINGS ───────────────────────────────────────────────────────────
_NODE = {
    #            color        alpha  marker  filled
    'Start':  ('#4daf4a',    1.0,   's',    True),
    'Medium': ('#333333',    0.45,  'o',    False),
    'End':    ('#377eb8',    1.0,   '^',    True),
    'Best':   ('#e41a1c',    1.0,   'o',    True),
}
DRAW_ORDER   = ['Medium', 'Start', 'End', 'Best']   # Best drawn last (on top)

EDGE_COLOR   = '#404040'
EDGE_ALPHA   = 0.50

NSIZE_RANGE  = (25,  250)   # matplotlib scatter: point² area
EWIDTH_RANGE = (0.3, 2.5)
FSIZE        = 11           # base font size for axes/titles
KK_THRESHOLD = 300          # nodes above this fall back to spring_layout


# ── LAYOUT ────────────────────────────────────────────────────────────────────

def _kk_or_spring(G: nx.DiGraph) -> dict:
    """Kamada-Kawai for small graphs, spring for large ones."""
    if G.number_of_nodes() <= KK_THRESHOLD:
        try:
            return nx.kamada_kawai_layout(G)
        except Exception:
            pass
    return nx.spring_layout(G, seed=42)


def _get_layout(G: nx.DiGraph, kind: str,
                x_attr: str = 'TreeSize', y_attr: str = 'Fitness') -> dict:
    """Return {node: (x, y)} position dict."""
    if kind in ('stress', 'kk'):
        return _kk_or_spring(G)

    if kind == 'fitness':
        # x is hidden; always use spring to avoid slow KK on large graphs
        base = nx.spring_layout(G, seed=42)
        fits = nx.get_node_attributes(G, 'Fitness')
        return {n: (base[n][0], fits[n]) for n in G.nodes()}

    if kind == 'bivar':
        x_vals = nx.get_node_attributes(G, x_attr)
        y_vals = nx.get_node_attributes(G, y_attr)
        return {n: (float(x_vals.get(n, 0.0)), float(y_vals.get(n, 0.0)))
                for n in G.nodes()}

    return nx.spring_layout(G, seed=42)


# ── SCALE HELPER ──────────────────────────────────────────────────────────────

def _scale(vals, out_range, v_range=None):
    lo = min(vals) if v_range is None else v_range[0]
    hi = max(vals) if v_range is None else v_range[1]
    arr = np.asarray(vals, dtype=float)
    if hi == lo:
        return np.full(len(arr), np.mean(out_range))
    return out_range[0] + (arr - lo) / (hi - lo) * (out_range[1] - out_range[0])


# ── SINGLE-PANEL PLOT ─────────────────────────────────────────────────────────

def plot_stn(ax: plt.Axes,
             G: nx.DiGraph,
             model: str,
             alg: str,
             layout:         str   = 'stress',
             node_size_attr: str   = 'TreeSize',
             size_range:     tuple = None,
             edge_range:     tuple = None,
             fitness_limits: tuple = None,
             x_attr:         str   = 'TreeSize',
             y_attr:         str   = 'Fitness',
             x_limits:       tuple = None,
             y_limits:       tuple = None,
             pos:            dict  = None):
    """
    Draw one STN panel onto `ax`.

    Parameters
    ----------
    size_range    (min, max) of node size attribute — supply for consistent scaling
                  across panels; if None, uses per-graph range.
    edge_range    (min, max) of edge Count — same purpose.
    fitness_limits  (min, max) for y-axis when layout == 'fitness'.
    """
    if G.number_of_nodes() == 0:
        ax.set_title(f"{alg}  {model}  (empty)", fontsize=FSIZE - 1)
        ax.axis('off')
        return

    if pos is None:
        pos = _get_layout(G, layout, x_attr=x_attr, y_attr=y_attr)

    # ── Edges ─────────────────────────────────────────────────────────────────
    ecounts = [d['Count'] for _, _, d in G.edges(data=True)]
    if ecounts:
        ewidths = _scale(ecounts, EWIDTH_RANGE, edge_range)
        for (u, v), w in zip(G.edges(), ewidths):
            ax.plot([pos[u][0], pos[v][0]],
                    [pos[u][1], pos[v][1]],
                    color=EDGE_COLOR, lw=float(w),
                    alpha=EDGE_ALPHA, zorder=1,
                    solid_capstyle='round')

    # ── Nodes ─────────────────────────────────────────────────────────────────
    size_attr = nx.get_node_attributes(G, node_size_attr)
    svals     = list(size_attr.values())
    scaled    = _scale(svals, NSIZE_RANGE, size_range)
    size_map  = dict(zip(size_attr.keys(), scaled))

    for ntype in DRAW_ORDER:
        color, alpha, marker, filled = _NODE[ntype]
        ns = [n for n in G.nodes() if G.nodes[n].get('Node') == ntype]
        if not ns:
            continue
        xs = [pos[n][0] for n in ns]
        ys = [pos[n][1] for n in ns]
        sz = [float(size_map.get(n, np.mean(NSIZE_RANGE))) for n in ns]

        if filled:
            ax.scatter(xs, ys, s=sz, c=color, marker=marker,
                       alpha=alpha, edgecolors='none', zorder=3)
        else:
            ax.scatter(xs, ys, s=sz, facecolors='none', edgecolors=color,
                       linewidths=0.7, marker=marker, alpha=alpha, zorder=3)

    ax.set_title(f"{alg}  [{model}]", fontsize=FSIZE - 1, pad=4)
    ax.axis('off')

    # Fitness-layout: y = fitness, x hidden
    if layout == 'fitness':
        ax.axis('on')
        ax.set_ylabel("Fitness", fontsize=FSIZE - 2)
        _ylim = y_limits or fitness_limits
        if _ylim:
            ax.set_ylim(_ylim)
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        for spine in ('top', 'right', 'bottom'):
            ax.spines[spine].set_visible(False)
        ax.grid(axis='y', linewidth=0.3, alpha=0.5)

    # Bivar-layout: both axes show node attributes
    elif layout == 'bivar':
        ax.axis('on')
        ax.set_xlabel(x_attr, fontsize=FSIZE - 2)
        ax.set_ylabel(y_attr, fontsize=FSIZE - 2)
        if x_limits:
            ax.set_xlim(x_limits)
        if y_limits:
            ax.set_ylim(y_limits)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.grid(linewidth=0.3, alpha=0.4)
        ax.tick_params(labelsize=FSIZE - 3)


# ── COMBINED FIGURE ───────────────────────────────────────────────────────────

def combined_plot(benchmark:  str,
                  stn_root:   str   = "stns",
                  plot_root:  str   = "plots",
                  layout:     str   = 'stress',
                  node_sizes: tuple = ('tree', 'node'),
                  ncols:      int   = 3,
                  x_attr:     str   = 'TreeSize',
                  y_attr:     str   = 'Fitness',
                  models:     tuple = ('genotype', 'hypercube', 'clustering')):
    """
    Load all STN pickle files for `benchmark` and produce one combined PNG per
    node_size variant. Layouts are computed once and reused across variants.

    Parameters
    ----------
    layout      'stress' | 'kk' | 'fitness' | 'bivar'
    node_sizes  tuple of size encodings to produce: 'tree' and/or 'node'
    ncols       number of columns in the grid
    models      which STN models to include: any subset of
                ('genotype', 'hypercube', 'clustering')
    """
    infolder  = os.path.join(stn_root,  benchmark)
    outfolder = os.path.join(plot_root, benchmark)
    os.makedirs(outfolder, exist_ok=True)

    pkls = sorted(f for f in os.listdir(infolder)
                  if f.endswith('.pkl') and any(m in f for m in models))
    if not pkls:
        print(f"No STN pickle files in {infolder} for models={models}")
        return

    graphs = []
    for p in pkls:
        with open(os.path.join(infolder, p), 'rb') as f:
            graphs.append(pickle.load(f))

    # ── Global limits for consistent cross-panel scaling ──────────────────────
    all_fit, all_tree, all_count, all_edge = [], [], [], []
    all_x_attr, all_y_attr = [], []
    for d in graphs:
        G = d['G']
        all_fit.extend(nx.get_node_attributes(G, 'Fitness').values())
        all_tree.extend(nx.get_node_attributes(G, 'TreeSize').values())
        all_count.extend(nx.get_node_attributes(G, 'Count').values())
        all_edge.extend(nx.get_edge_attributes(G, 'Count').values() or [1])
        if layout == 'bivar':
            all_x_attr.extend(nx.get_node_attributes(G, x_attr).values())
            all_y_attr.extend(nx.get_node_attributes(G, y_attr).values())

    fitness_limits = (min(all_fit),   max(all_fit))
    tree_range     = (min(all_tree),  max(all_tree))
    count_range    = (min(all_count), max(all_count))
    edge_range     = (min(all_edge),  max(all_edge))

    x_limits = (min(all_x_attr), max(all_x_attr)) if all_x_attr else None
    y_limits = (min(all_y_attr), max(all_y_attr)) if all_y_attr else None

    # ── Compute layout once per graph, reuse across node_size variants ────────
    print(f"  computing {layout} layouts for {len(graphs)} graphs …")
    pos_list = [_get_layout(d['G'], layout, x_attr=x_attr, y_attr=y_attr)
                for d in graphs]

    # ── Render one figure per node_size ───────────────────────────────────────
    n     = len(graphs)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    w = 4.5 * ncols + 1.8
    h = 4.2 * nrows + 0.6

    for node_size in node_sizes:
        size_range = tree_range if node_size == 'tree' else count_range
        attr       = 'TreeSize' if node_size == 'tree' else 'Count'

        fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), squeeze=False)

        for idx, (d, pos) in enumerate(zip(graphs, pos_list)):
            r, c = divmod(idx, ncols)
            # pass None for axis limits → each subplot auto-scales to its own data
            plot_stn(axes[r][c], d['G'], d['model'], d['alg'],
                     layout=layout, node_size_attr=attr,
                     size_range=size_range, edge_range=edge_range,
                     fitness_limits=None,
                     x_attr=x_attr, y_attr=y_attr,
                     x_limits=None, y_limits=None,
                     pos=pos)

        for idx in range(n, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r][c].set_visible(False)

        legend_handles = [
            mpatches.Patch(color=_NODE['Start'][0],  label='Start'),
            mpatches.Patch(color=_NODE['Medium'][0], label='Medium', alpha=0.45),
            mpatches.Patch(color=_NODE['End'][0],    label='End'),
            mpatches.Patch(color=_NODE['Best'][0],   label='Best'),
        ]
        fig.legend(handles=legend_handles,
                   loc='center right', fontsize=FSIZE,
                   framealpha=0.9, bbox_to_anchor=(1.0, 0.5))

        fig.suptitle(f"STN — {benchmark}  |  layout={layout}  node={node_size}",
                     fontsize=FSIZE + 1, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 0.88, 0.97])

        model_tag = '' if len(models) == 3 else '_' + '+'.join(models)
        fname = f"{benchmark}_{layout}{model_tag}_{node_size}_stn.png"
        fpath = os.path.join(outfolder, fname)
        fig.savefig(fpath, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fpath}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os as _os
    _HERE      = _os.path.dirname(_os.path.abspath(__file__))
    _STN_ROOT  = _os.path.join(_HERE, "..", "log", "stns")
    _PLOT_ROOT = _os.path.join(_HERE, "..", "log", "figs", "stns")

    BENCHMARKS = ["toxicity", "concrete", "instanbul", "ppb",
                  "resid_build_sale_price", "energy"]

    for benchmark in BENCHMARKS:
        print(f"\n{'='*60}")
        print(f"  Plotting STNs: {benchmark}")
        print(f"{'='*60}")
        for layout in ('stress', 'fitness'):
            combined_plot(benchmark,
                          stn_root=_STN_ROOT,
                          plot_root=_PLOT_ROOT,
                          layout=layout,
                          node_sizes=('tree', 'node'),
                          ncols=3,
                          x_attr='TreeSize',
                          y_attr='Fitness')

        # bivar: genotype + hypercube use TreeSize on X
        combined_plot(benchmark,
                      stn_root=_STN_ROOT,
                      plot_root=_PLOT_ROOT,
                      layout='bivar',
                      node_sizes=('tree', 'node'),
                      ncols=2,
                      x_attr='TreeSize',
                      y_attr='Fitness',
                      models=('genotype', 'hypercube'))

        # bivar: clustering uses ClusterID on X
        combined_plot(benchmark,
                      stn_root=_STN_ROOT,
                      plot_root=_PLOT_ROOT,
                      layout='bivar',
                      node_sizes=('tree', 'node'),
                      ncols=1,
                      x_attr='ClusterID',
                      y_attr='Fitness',
                      models=('clustering',))
