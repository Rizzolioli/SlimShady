"""
Build Search Trajectory Networks (STNs) from algorithm run traces.
Translated from R code by Gabriela Ochoa, Josip Hrvatić & Magda Smolić-Ročak.

Expected CSV format (no header) — produced by stn_prep.py:
  col 0 : Run      (int, 1-indexed)
  col 1 : Iter     (int, generation)
  col 2 : Fitness  (float, RMSE — minimisation assumed)
  col 3 : name     (str, genotype / tree string)
  col 4+: sem_i    (float, phenotype / semantic descriptor values)

Produces three pickle files per algorithm file:
  {alg}_genotype_stn.pkl    — nodes = tree string
  {alg}_hypercube_stn.pkl   — nodes = rounded semantic vector (concatenated string)
  {alg}_clustering_stn.pkl  — nodes = k-means cluster ID in the semantic space

Each pickle contains:
  G      : nx.DiGraph with node attrs (Fitness, TreeSize, Count, Node)
                        and edge attrs (Count, Type)
  best   : float, minimum fitness seen
  model  : str, model name
  alg    : str, algorithm name (from filename)
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── CONFIG ────────────────────────────────────────────────────────────────────
PHENOTYPE_ROUND_FACTOR = 1    # decimal places for hypercube node IDs
FITNESS_ROUND_FACTOR   = 4    # decimal places for fitness rounding
NRUNS                  = 10   # max runs to use per file
N_CLUSTERS             = 50   # k for k-means clustering STN
CLUSTER_RANDOM_STATE   = 42


# ── HELPERS ───────────────────────────────────────────────────────────────────

def tree_size(x: str) -> int:
    """Character-count proxy for tree complexity (mirrors R version)."""
    s = str(x)
    s = re.sub(r'\b([a-zA-Z]+)\d+\b', r'\1', s)
    s = re.sub(r'(?<![A-Za-z_])[-+]?\d*\.?\d+([eE][-+]?\d+)?', 'N', s)
    return len(s)


def create_node_id(row: np.ndarray, factor: int = PHENOTYPE_ROUND_FACTOR) -> str:
    """Round-and-concatenate a semantic vector into a string node ID."""
    scaled = np.round(row.astype(float), factor) * (10 ** factor)
    return "_".join(map(str, scaled.astype(int)))


# ── STN BUILDER ───────────────────────────────────────────────────────────────

def build_stn(df: pd.DataFrame, nruns: int = NRUNS) -> tuple:
    """
    Build a directed STN NetworkX graph from a trajectory dataframe.

    df must have columns: Run, Fitness, TreeSize, name
    Edges connect consecutive distinct nodes within each run.
    Nodes are aggregated across runs: Fitness=min, TreeSize=median, Count=visits.
    """
    df = df[df['Run'] <= nruns].copy()

    edges_raw = []
    start_nodes, end_nodes = set(), set()

    for r in range(1, nruns + 1):
        dfr = df[df['Run'] == r]
        if dfr.empty:
            continue
        seq = dfr['name'].tolist()
        start_nodes.add(seq[0])
        end_nodes.add(seq[-1])
        for i in range(len(seq) - 1):
            if seq[i] != seq[i + 1]:
                edges_raw.append((seq[i], seq[i + 1]))

    # Aggregate nodes
    node_agg = (
        df.groupby('name', sort=False)
          .agg(Fitness=('Fitness', 'min'),
               TreeSize=('TreeSize', 'median'),
               Count=('Fitness', 'count'))
          .reset_index()
    )
    best_fitness = float(node_agg['Fitness'].min())

    # Aggregate edges
    edge_agg: dict[tuple, int] = {}
    for u, v in edges_raw:
        edge_agg[(u, v)] = edge_agg.get((u, v), 0) + 1

    # Construct graph
    G = nx.DiGraph()

    for _, row in node_agg.iterrows():
        n = row['name']
        ntype = 'Medium'
        if n in end_nodes:
            ntype = 'End'
        if n in start_nodes:
            ntype = 'Start'
        if row['Fitness'] == best_fitness:
            ntype = 'Best'
        G.add_node(n,
                   Fitness=float(row['Fitness']),
                   TreeSize=float(row['TreeSize']),
                   Count=int(row['Count']),
                   Node=ntype)

    for (u, v), count in edge_agg.items():
        if u in G and v in G:
            f_u = G.nodes[u]['Fitness']
            f_v = G.nodes[v]['Fitness']
            etype = ('Improving' if f_v < f_u else
                     'Worsening' if f_v > f_u else 'Equal')
            G.add_edge(u, v, Count=count, Type=etype)

    return G, best_fitness


# ── CLUSTERING STN ─────────────────────────────────────────────────────────────

def build_clustering_stn(df_all: pd.DataFrame,
                         sem_cols: list[str],
                         nruns: int = NRUNS,
                         n_clusters: int = N_CLUSTERS,
                         random_state: int = CLUSTER_RANDOM_STATE) -> tuple:
    """
    Build a clustering-based phenotype STN.

    K-means is fit on ALL semantic vectors from df_all (all runs pooled),
    then each visit is replaced by its cluster label as node ID.

    Parameters
    ----------
    df_all     : dataframe with Run, Fitness, TreeSize, and sem_cols
    sem_cols   : list of semantic column names
    n_clusters : k for k-means (automatically capped at n_unique_points)
    """
    sem_matrix = df_all[sem_cols].values.astype(float)

    # Cap k to avoid error when fewer points than clusters
    n_unique = len(np.unique(sem_matrix, axis=0))
    k = min(n_clusters, n_unique)
    if k < n_clusters:
        print(f"  clustering: capping k from {n_clusters} to {k} (only {n_unique} unique points)")

    scaler = StandardScaler()
    sem_scaled = scaler.fit_transform(sem_matrix)

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(sem_scaled)

    df_clust = pd.DataFrame({
        'Run':      df_all['Run'].values,
        'Fitness':  df_all['Fitness'].values,
        'TreeSize': df_all['TreeSize'].values,
        'name':     [f'C{lbl}' for lbl in labels],
    })

    # Deduplicate consecutive identical clusters within each run
    df_clust = df_clust.drop_duplicates(subset=['Run', 'name'])

    G, best = build_stn(df_clust, nruns)

    # Store numeric cluster index as a node attribute for bivar plotting
    for n in G.nodes():
        try:
            G.nodes[n]['ClusterID'] = int(n[1:])   # "C42" → 42
        except (ValueError, IndexError):
            G.nodes[n]['ClusterID'] = -1

    return G, best


# ── PROCESS FOLDER ────────────────────────────────────────────────────────────

def process_folder(benchmark: str,
                   data_root:  str  = "data",
                   stn_root:   str  = "stns",
                   nruns:      int  = NRUNS,
                   n_clusters: int  = N_CLUSTERS,
                   build_clustering: bool = True):
    """
    Build genotype, hypercube, and (optionally) clustering STNs for all
    algorithm files in data/{benchmark}/.

    Parameters
    ----------
    benchmark        : dataset/benchmark name
    data_root        : folder containing the prepared CSV files
    stn_root         : folder where pickle outputs are written
    nruns            : max runs to include per algorithm
    n_clusters       : k for k-means clustering STN
    build_clustering : set False to skip the clustering STN (slow for large semantics)
    """
    infolder  = os.path.join(data_root, benchmark)
    outfolder = os.path.join(stn_root,  benchmark)
    os.makedirs(outfolder, exist_ok=True)

    files = sorted(f for f in os.listdir(infolder)
                   if f.lower().endswith(('.csv', '.txt')))
    if not files:
        print(f"No CSV/TXT files found in {infolder}")
        return

    for fname in files:
        fpath = os.path.join(infolder, fname)
        print(f"\n── {fname}")

        df = pd.read_csv(fpath, header=None)
        n_sem = df.shape[1] - 4
        sem_cols = [f'sem_{i}' for i in range(n_sem)]
        df.columns = ['Run', 'Iter', 'Fitness', 'name'] + sem_cols

        df['Fitness']  = df['Fitness'].round(FITNESS_ROUND_FACTOR)
        df['TreeSize'] = df['name'].apply(tree_size)

        algn = re.sub(r'\.(csv|txt)$', '', fname, flags=re.IGNORECASE)

        # ── Genotype STN ──────────────────────────────────────────────────────
        df_geno = (df[['Run', 'Fitness', 'TreeSize', 'name']]
                     .drop_duplicates(subset=['Run', 'name']))
        print(f"  genotype rows after dedup : {len(df_geno)}")
        G, best = build_stn(df_geno, nruns)
        print(f"  nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")

        out = os.path.join(outfolder, f"{algn}_genotype_stn.pkl")
        with open(out, 'wb') as f:
            pickle.dump(dict(G=G, best=best, model='genotype', alg=algn), f)
        print(f"  → {out}")

        # ── Hypercube STN ─────────────────────────────────────────────────────
        if not sem_cols:
            print("  no semantic columns — skipping hypercube and clustering STNs")
            continue

        df_hyp = (df[['Run', 'Fitness', 'TreeSize'] + sem_cols]
                    .drop_duplicates(subset=['Run'] + sem_cols))
        print(f"  hypercube rows after dedup: {len(df_hyp)}")

        ids = df_hyp[sem_cols].apply(lambda r: create_node_id(r.values), axis=1)
        df_hyp_named = pd.DataFrame({
            'Run':      df_hyp['Run'].values,
            'Fitness':  df_hyp['Fitness'].values,
            'TreeSize': df_hyp['TreeSize'].values,
            'name':     ids.values,
        })

        G, best = build_stn(df_hyp_named, nruns)
        print(f"  nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")

        out = os.path.join(outfolder, f"{algn}_hypercube_stn.pkl")
        with open(out, 'wb') as f:
            pickle.dump(dict(G=G, best=best, model='hypercube', alg=algn), f)
        print(f"  → {out}")

        # ── Clustering STN ────────────────────────────────────────────────────
        if not build_clustering:
            continue

        df_clust_in = (df[['Run', 'Fitness', 'TreeSize'] + sem_cols]
                         .drop_duplicates(subset=['Run'] + sem_cols))
        print(f"  clustering k={n_clusters}, input rows: {len(df_clust_in)}")

        G, best = build_clustering_stn(df_clust_in, sem_cols, nruns, n_clusters)
        print(f"  nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")

        out = os.path.join(outfolder, f"{algn}_clustering_stn.pkl")
        with open(out, 'wb') as f:
            pickle.dump(dict(G=G, best=best, model='clustering', alg=algn), f)
        print(f"  → {out}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    _HERE     = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = os.path.join(_HERE, "..", "log", "stn_data")
    STN_ROOT  = os.path.join(_HERE, "..", "log", "stns")

    benchmarks = ["koza-1", "nguyen-5", "nguyen-6", "rbsp",
                  "concrete", "istanbul", "ppb"]
    benchmark = benchmarks[5]   # change as needed

    process_folder(
        benchmark,
        data_root=DATA_ROOT,
        stn_root=STN_ROOT,
        nruns=5,             # 5 seeds used in SlimShady experiments
        n_clusters=50,
        build_clustering=True,
    )
