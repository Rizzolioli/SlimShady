"""
run_extended_experiments.py
============================
Runs the 30-seed extensions of the four head-XO experiments in sequence:

    1. main_prob_xo.py    - p_xo in {0.3, 0.5, 0.7}
    2. main_pop_xo.py     - budget reallocation across (pop_size, n_iter)
    3. main_op_stats.py   - operator improvement-rate tracking
    4. main_decay_xo.py   - cosine^2 p_xo decay schedule

Each script already saturates all CPU cores internally
(ProcessPoolExecutor(max_workers=os.cpu_count())), so they are run one at a
time as separate subprocesses rather than concurrently. A failure in one
script does not stop the others; a pass/fail summary is printed at the end.

Usage:
    python main/run_extended_experiments.py
    python main/run_extended_experiments.py --only prob_xo,decay_xo
    python main/run_extended_experiments.py --dry-run
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

_MAIN_DIR = Path(__file__).resolve().parent

EXPERIMENTS = [
    ("prob_xo",  _MAIN_DIR / "main_prob_xo.py"),
    ("pop_xo",   _MAIN_DIR / "main_pop_xo.py"),
    ("op_stats", _MAIN_DIR / "main_op_stats.py"),
    ("decay_xo", _MAIN_DIR / "main_decay_xo.py"),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated subset of experiment labels to run, e.g. "
             "'prob_xo,decay_xo'. Default: run all four, in order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the experiments that would run, in order, without executing them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    experiments = EXPERIMENTS
    if args.only:
        wanted = [label.strip() for label in args.only.split(",")]
        known = {label for label, _ in EXPERIMENTS}
        unknown = [label for label in wanted if label not in known]
        if unknown:
            print(f"Unknown experiment label(s): {unknown}. "
                  f"Valid labels: {sorted(known)}")
            sys.exit(1)
        experiments = [(label, path) for label, path in EXPERIMENTS if label in wanted]

    print("Experiment run order:")
    for label, path in experiments:
        print(f"  - {label:<10} ({path.name})")

    if args.dry_run:
        print("\n--dry-run: no experiments executed.")
        return

    results = []
    wall0 = time.time()

    for label, path in experiments:
        print(f"\n{'=' * 80}")
        print(f"Starting {label} ({path.name}) at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 80}\n")

        t0 = time.time()
        proc = subprocess.run([sys.executable, str(path)], cwd=str(_MAIN_DIR.parent))
        elapsed = time.time() - t0

        ok = proc.returncode == 0
        results.append((label, ok, elapsed))
        status = "OK" if ok else f"FAILED (exit {proc.returncode})"
        print(f"\n--- {label} finished: {status} in {elapsed:.1f}s ---")

    total_elapsed = time.time() - wall0

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    for label, ok, elapsed in results:
        status = "OK" if ok else "FAILED"
        print(f"  {label:<10} {status:<8} {elapsed:>10.1f}s")
    print(f"\nTotal wall time: {total_elapsed:.1f}s")

    if any(not ok for _, ok, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
