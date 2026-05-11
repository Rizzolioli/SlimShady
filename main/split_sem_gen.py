"""
Split an existing joint log=8 CSV into:
  - main CSV    (in-place): metrics only, uniform 11 columns
  - _sem_gen CSV (new file): genotype + semantics, 8 columns

Target file: results_head_size_07052026.csv  (default, matches main_head_size.py)

Existing log=8 row formats:
  12 cols  "same" row  → [algo, run_id, loader, seed, gen, train_fit, timing, nodes,
                           test_fit, nodes_count, "same", log]
  14 cols  "changed"   → [algo, run_id, loader, seed, gen, train_fit, timing, nodes,
                           test_fit, nodes_count, tree_repr, train_sem, test_sem, log]

After split:
  main (11 cols):     cols 0-9 + log  (semantics and genotype removed)
  sem_gen (8 cols):   cols 0-4 + tree_repr + train_sem + test_sem  (only for "changed" rows)

Usage:
    python split_sem_gen.py                    # splits the default head_size log
    python split_sem_gen.py path/to/file.csv   # splits a specific file
"""
import csv
import os
import sys

_DEFAULT_LOG = os.path.join(os.path.dirname(__file__), "log", "results_scramble_xo_05052026.csv")

SAME_NCOLS    = 12
CHANGED_NCOLS = 14


def split_file(path):
    base         = path[:-4] if path.endswith('.csv') else path
    sem_gen_path = base + '_sem_gen.csv'

    main_rows    = []
    sem_gen_rows = []
    skipped      = 0

    with open(path, 'r', newline='', encoding='utf-8') as f:
        for row in csv.reader(f):
            n = len(row)
            if n == CHANGED_NCOLS:
                main_rows.append(row[:10] + [row[13]])          # 11 cols — semantics/genotype stripped
                sem_gen_rows.append(row[:5] + row[10:13])       # 8 cols — key + tree_repr + semantics
            elif n == SAME_NCOLS:
                main_rows.append(row[:10] + [row[11]])          # 11 cols — "same" marker dropped
            else:
                main_rows.append(row)                           # unexpected format, keep as-is
                skipped += 1

    with open(path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(main_rows)

    with open(sem_gen_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(sem_gen_rows)

    note = f"  ({skipped} rows with unexpected column count kept as-is)" if skipped else ""
    print(f"{os.path.basename(path)}: {len(main_rows)} main rows, "
          f"{len(sem_gen_rows)} sem_gen rows → {os.path.basename(sem_gen_path)}{note}")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_LOG

    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    split_file(path)
    print("Done.")
