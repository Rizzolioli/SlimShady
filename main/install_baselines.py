"""
install_baselines.py — Install all GP baseline dependencies.

Run once before main_baselines.py:
    python main/install_baselines.py

PySR note: requires Julia. If Julia is not on PATH, this script will attempt
to install it automatically via `juliaup` (Windows/Linux/macOS installer).
After Julia is available, `pysr` calls `PySRRegressor().julia_project` on first
use which downloads the Julia packages — this takes a few minutes once.
"""

import subprocess
import sys
import shutil


def pip(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *packages])


def run(cmd, **kwargs):
    return subprocess.run(cmd, **kwargs)


PACKAGES = [
    # package name(s)            description
    (["gplearn"],                 "GPLearn — sklearn-compatible tree GP"),
    (["operon-sklearn"],          "Operon — high-performance GP"),
    (["pygpgomea"],               "GP-GOMEA — linkage-learning GP"),
    (["pysr"],                    "PySR — Julia-backed symbolic regression"),
    (["itea-sklearn"],            "ITEA — Interaction-Transformation EA"),
]


def install_julia_if_needed():
    """Install Julia via juliaup if not already on PATH."""
    if shutil.which("julia"):
        print("  Julia already on PATH — skipping Julia install.")
        return True

    print("  Julia not found. Attempting to install via juliaup …")
    if sys.platform == "win32":
        # winget is the simplest path on Windows
        winget = shutil.which("winget")
        if winget:
            r = run(["winget", "install", "--id", "Julialang.Juliaup", "-e", "--silent"],
                    capture_output=True)
            if r.returncode == 0:
                print("  Julia installed via winget. Restart your shell to put julia on PATH.")
                return True
        # fallback: direct installer
        print("  winget not found. Download and run the Julia installer manually:")
        print("    https://julialang.org/downloads/")
        return False
    else:
        # Linux / macOS
        r = run(["curl", "-fsSL", "https://install.julialang.org"], capture_output=True, text=True)
        if r.returncode == 0:
            r2 = run(["sh", "-c", r.stdout], capture_output=True)
            if r2.returncode == 0:
                print("  Julia installed. You may need to restart your shell.")
                return True
        print("  Could not auto-install Julia. Install manually: https://julialang.org/downloads/")
        return False


def init_pysr_julia():
    """
    Run PySR's Julia environment setup (downloads Julia packages).
    Only needed once; safe to re-run.
    """
    print("  Initialising PySR Julia environment (may take a few minutes on first run) …")
    try:
        from pysr import PySRRegressor
        PySRRegressor(verbosity=0).julia_project   # triggers Julia package installation
        print("  PySR Julia environment ready.")
    except Exception as e:
        print(f"  PySR Julia init failed: {e}")
        print("  You can initialise manually with:")
        print("    python -c \"from pysr import PySRRegressor; PySRRegressor()\"")


if __name__ == "__main__":
    results = {}

    for pkgs, desc in PACKAGES:
        print(f"\n{'─'*60}")
        print(f"Installing: {desc}")
        print(f"  pip install {' '.join(pkgs)}")
        try:
            pip(*pkgs)
            results[pkgs[0]] = "OK"
        except subprocess.CalledProcessError as e:
            results[pkgs[0]] = f"FAILED ({e})"
            print(f"  !! Failed: {e}")

    # PySR needs Julia
    print(f"\n{'─'*60}")
    print("Setting up Julia for PySR …")
    if results.get("pysr") == "OK":
        julia_ok = install_julia_if_needed()
        if julia_ok:
            init_pysr_julia()
    else:
        print("  Skipping Julia setup (pysr not installed).")

    # Summary
    print(f"\n{'='*60}")
    print("Installation summary:")
    for pkg, status in results.items():
        mark = "✓" if status == "OK" else "✗"
        print(f"  {mark}  {pkg:20s}  {status}")

    print("\nVerifying imports …")
    for lib, import_name in [
        ("gplearn",        "gplearn"),
        ("operon-sklearn", "operon"),
        ("pygpgomea",      "pygpgomea"),
        ("pysr",           "pysr"),
        ("itea-sklearn",   "itea"),
    ]:
        try:
            __import__(import_name)
            print(f"  OK  {lib}")
        except ImportError:
            print(f"  --  {lib}  (not importable)")
