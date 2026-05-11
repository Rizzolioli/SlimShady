"""
Split existing joint log=8 CSVs into:
  - main CSV    (in-place): metrics only, uniform 11 columns
  - _sem_gen CSV (new file): genotype + semantics, 8 columns

Existing log=8 row formats:
  12 cols  "same" row  → [algo, run_id, loader, seed, gen, train_fit, timing, nodes,
                           test_fit, nodes_count, "same", log]
  14 cols  "changed"   → [algo, run_id, loader, seed, gen, train_fit, timing, nodes,
                           test_fit, nodes_count, tree_repr, train_sem, test_sem, log]

After split:
  main (11 cols):     cols 0-9 + log (col 11 or 13 respectively)
  sem_gen (8 cols):   cols 0-4 + tree_repr + train_sem + test_sem  (only for "changed" rows)

Usage:
    python split_sem_gen.py path/to/results.csv [another.csv ...]
    python split_sem_gen.py          # processes all eligible CSVs in main/log/
"""
import csv
import os
import sys
from glob import glob

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
                main_rows.append(row[:10] + [row[13]])          # 11 cols
                sem_gen_rows.append(row[:5] + row[10:13])       # 8 cols
            elif n == SAME_NCOLS:
                main_rows.append(row[:10] + [row[11]])          # 11 cols
            else:
                main_rows.append(row)                           # unknown format, keep as-is
                skipped += 1

    with open(path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(main_rows)

    with open(sem_gen_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(sem_gen_rows)

    note = f"  ({skipped} rows with unexpected column count kept as-is)" if skipped else ""
    print(f"{os.path.basename(path)}: {len(main_rows)} main rows, "
          f"{len(sem_gen_rows)} sem_gen rows → {os.path.basename(sem_gen_path)}{note}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        log_dir = os.path.join(os.path.dirname(__file__), 'log')
        paths = sorted(
            p for p in glob(os.path.join(log_dir, '*.csv'))
            if not p.endswith('_sem_gen.csv')
            and not os.path.basename(p).startswith('settings')
        )

    if not paths:
        print("No CSV files found.")
        sys.exit(0)

    for path in paths:
        split_file(path)

    print("Done.")
