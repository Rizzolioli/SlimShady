"""
install_baselines.py — Install all GP baseline dependencies.

Run once before main_baselines.py:
    python main/install_baselines.py

PySR note: requires Julia. If Julia is not on PATH, this script will attempt
to install it automatically. After Julia is available, PySR downloads its
Julia packages on the first run (~5 min, once only).
"""

import subprocess
import sys
import shutil


def pip(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *packages])


PACKAGES = [
    # (pip_name,    import_name,   description)
    ("gplearn",    "gplearn",     "GPLearn — sklearn-compatible tree GP"),
    ("pyoperon",   "pyoperon",    "Operon  — high-performance GP"),
    ("pysr",       "pysr",        "PySR    — Julia-backed symbolic regression  (+Julia)"),
    # GP-GOMEA: no PyPI wheel — requires manual C++ build from source
    #   https://github.com/marcovirgolin/GP-GOMEA
]


def install_julia_if_needed():
    """Install Julia via juliaup if not already on PATH."""
    if shutil.which("julia"):
        print("  Julia already on PATH — skipping Julia install.")
        return True

    print("  Julia not found. Attempting to install via juliaup …")

    if sys.platform == "win32":
        winget = shutil.which("winget")
        if winget:
            r = subprocess.run(
                ["winget", "install", "--id", "Julialang.Juliaup", "-e", "--silent"],
                capture_output=True)
            if r.returncode == 0:
                print("  Julia installed via winget. Open a new terminal and re-run this script.")
                return False   # PATH update requires new shell
        print("  Install Julia manually from: https://julialang.org/downloads/")
        return False

    else:  # macOS / Linux
        # juliaup non-interactive installer
        r = subprocess.run(
            ["curl", "-fsSL", "https://install.julialang.org"],
            capture_output=True, text=True)
        if r.returncode == 0:
            r2 = subprocess.run(
                ["sh", "-s", "--", "--yes"],
                input=r.stdout, text=True, capture_output=True)
            if r2.returncode == 0:
                # juliaup adds to ~/.bashrc / ~/.zshrc; try sourcing PATH update
                julia_bin = shutil.which("julia") or \
                    subprocess.run(["bash", "-lc", "which julia"],
                                   capture_output=True, text=True).stdout.strip()
                if julia_bin:
                    print(f"  Julia installed at {julia_bin}")
                    return True
                print("  Julia installed. Open a new terminal and re-run this script.")
                return False

        # macOS homebrew fallback
        brew = shutil.which("brew")
        if brew:
            print("  Trying: brew install julia …")
            r = subprocess.run(["brew", "install", "julia"], capture_output=True)
            if r.returncode == 0 and shutil.which("julia"):
                print("  Julia installed via Homebrew.")
                return True

        print("  Could not auto-install Julia.")
        print("  Install manually: https://julialang.org/downloads/")
        return False


def init_pysr_julia():
    """Trigger PySR's one-time Julia package download."""
    print("  Initialising PySR Julia environment (may take ~5 min on first run) …")
    try:
        from pysr import PySRRegressor
        # instantiating with niterations=1 triggers Julia setup without a real fit
        reg = PySRRegressor(niterations=1, verbosity=0)
        # access the julia_project attribute to trigger environment setup
        _ = reg.julia_project
        print("  PySR Julia environment ready.")
    except Exception as e:
        print(f"  PySR Julia init: {e}")
        print("  If Julia is freshly installed, open a new terminal and run:")
        print("    python -c \"from pysr import PySRRegressor; PySRRegressor(niterations=1, verbosity=0)\"")


if __name__ == "__main__":
    results = {}

    for pip_name, import_name, desc in PACKAGES:
        print(f"\n{'─'*60}")
        print(f"Installing: {desc}")
        print(f"  pip install {pip_name}")
        try:
            pip(pip_name)
            results[pip_name] = "OK"
        except subprocess.CalledProcessError as e:
            results[pip_name] = f"FAILED ({e})"
            print(f"  !! Failed: {e}")

    # Julia setup for PySR
    print(f"\n{'─'*60}")
    print("Setting up Julia for PySR …")
    if results.get("pysr") == "OK":
        if install_julia_if_needed():
            init_pysr_julia()
    else:
        print("  Skipping (pysr not installed).")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Installation summary:")
    for pip_name, _, _ in PACKAGES:
        status = results.get(pip_name, "SKIPPED")
        mark = "✓" if status == "OK" else "✗"
        print(f"  {mark}  {pip_name:20s}  {status}")

    print()
    print("Verifying imports …")
    for pip_name, import_name, _ in PACKAGES:
        try:
            __import__(import_name)
            print(f"  OK  {pip_name} (import {import_name})")
        except ImportError as e:
            print(f"  --  {pip_name}  ({e})")

    print()
    print("Not on PyPI (build from source if needed):")
    print("  GP-GOMEA  https://github.com/marcovirgolin/GP-GOMEA")
