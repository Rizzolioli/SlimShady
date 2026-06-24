"""
install_baselines.py — Install all GP baseline dependencies.

Run once before main_baselines.py:
    python main/install_baselines.py

PySR note: requires Julia. If Julia is not on PATH, this script will attempt
to install it automatically. After Julia is available, PySR downloads its
Julia packages on the first run (~5 min, once only).

GP-GOMEA note: no PyPI wheel. This script clones and builds it from source.
Requires: git, cmake ≥ 3.14, C++17 compiler (Xcode CLT on macOS, gcc/g++ on Linux).
"""

import os
import subprocess
import sys
import shutil


def pip(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *packages])


def _verify_import(pip_name, fallback_import_name):
    """
    Try to import a package. For packages whose pip name differs from the
    Python module name (e.g. pyGPGOMEA), discover the real module name from
    the distribution's top_level.txt metadata, then fall back to guessing.
    Returns the importable name on success, None on failure.
    """
    import importlib.metadata as _im, importlib as _il

    # 1. look up the real module name from installed metadata
    dist_candidates = [pip_name, pip_name.lower(), pip_name.upper(),
                       "pyGPGOMEA", "pygpgomea"]
    top_levels = []
    for dist_name in dict.fromkeys(dist_candidates):   # deduplicate, keep order
        try:
            text = _im.distribution(dist_name).read_text("top_level.txt")
            if text:
                top_levels = [m.strip() for m in text.splitlines() if m.strip()]
                break
        except Exception:
            pass

    # 2. try metadata-discovered names first, then the fallback name
    for name in dict.fromkeys(top_levels + [fallback_import_name]):
        try:
            _il.import_module(name)
            return name
        except ImportError:
            pass
    return None


PACKAGES = [
    # (pip_name,  import_name,  description)
    ("gplearn",   "gplearn",    "GPLearn — sklearn-compatible tree GP"),
    ("pyoperon",  "pyoperon",   "Operon  — high-performance GP"),
    ("pysr",      "pysr",       "PySR    — Julia-backed symbolic regression  (+Julia)"),
    ("tomli",     "tomli",      "tomli   — TOML parser (needed by juliacall/juliapkg)"),
]


# ── pyoperon macOS libzstd fix ─────────────────────────────────────────────────

def fix_pyoperon_macos():
    """
    pyoperon wheels are linked against a CI build path for libzstd.
    On macOS, the library must live inside the active conda env.
    Fix: conda install zstd, or brew install zstd + symlink.
    """
    if sys.platform != "darwin":
        return

    try:
        import pyoperon  # noqa: F401
        return   # already importable — nothing to do
    except ImportError:
        pass

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    target = os.path.join(conda_prefix, "lib", "libzstd.1.dylib") if conda_prefix else ""

    # 1) conda install zstd
    if conda_prefix and not os.path.exists(target):
        conda = shutil.which("conda")
        if conda:
            print("  [macOS] libzstd missing — installing via conda …")
            r = subprocess.run(
                [conda, "install", "-c", "conda-forge", "zstd", "-y"],
                capture_output=False)
            if r.returncode == 0 and os.path.exists(target):
                print("  libzstd installed via conda.")
                return

    # 2) brew install zstd + symlink into conda env (or system lib)
    brew = shutil.which("brew")
    if brew:
        # find brew's libzstd
        r = subprocess.run(
            ["brew", "--prefix", "zstd"], capture_output=True, text=True)
        if r.returncode != 0:
            print("  [macOS] zstd not in brew — installing …")
            subprocess.run(["brew", "install", "zstd"])
            r = subprocess.run(
                ["brew", "--prefix", "zstd"], capture_output=True, text=True)

        if r.returncode == 0:
            brew_lib = os.path.join(r.stdout.strip(), "lib", "libzstd.1.dylib")
            if os.path.exists(brew_lib):
                if conda_prefix and target and not os.path.exists(target):
                    os.symlink(brew_lib, target)
                    print(f"  libzstd symlinked: {brew_lib} → {target}")
                    return
                # fall back: set DYLD_LIBRARY_PATH hint
                brew_lib_dir = os.path.dirname(brew_lib)
                print(f"\n  NOTE: libzstd found at {brew_lib_dir}")
                print("  If pyoperon still fails to import, run:")
                print(f"    export DYLD_LIBRARY_PATH={brew_lib_dir}:$DYLD_LIBRARY_PATH")
                return

    print("  [macOS] Could not auto-fix libzstd.")
    print("  Try: conda install -c conda-forge zstd")


# ── GP-GOMEA source build ──────────────────────────────────────────────────────

def _find_python_package_root(repo_path, depth=3):
    """Walk repo_path up to `depth` levels deep and return the first directory
    that contains setup.py or pyproject.toml."""
    for root, dirs, files in os.walk(repo_path):
        # prune hidden dirs and keep depth bounded
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        level = root.replace(repo_path, "").count(os.sep)
        if level >= depth:
            dirs.clear()
            continue
        if "setup.py" in files or "pyproject.toml" in files:
            return root
    return None


def _cmake_compile(repo_path):
    """
    Run cmake configure + build from repo_path.
    Returns (success: bool, ext_paths: list[str]).
    ext_paths contains all .so/.pyd files produced in the build directory.
    """
    import glob as _glob

    build_dir = os.path.join(repo_path, "_pybuild")
    # wipe stale cmake cache so a failed previous configure doesn't poison this run
    cmake_cache = os.path.join(build_dir, "CMakeCache.txt")
    if os.path.exists(cmake_cache):
        os.remove(cmake_cache)
    os.makedirs(build_dir, exist_ok=True)

    # determine prefix path so cmake FindArmadillo / FindPython can locate headers
    prefix_path = os.environ.get("CONDA_PREFIX", "")
    if not prefix_path:
        brew = shutil.which("brew")
        if brew:
            r0 = subprocess.run(["brew", "--prefix"], capture_output=True, text=True)
            if r0.returncode == 0:
                prefix_path = r0.stdout.strip()

    cmake_cmd = [
        "cmake", "..",
        f"-DPYTHON_EXECUTABLE={sys.executable}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_R_BINDING=OFF",   # we only need the Python binding; R not required
    ]
    if prefix_path:
        cmake_cmd.append(f"-DCMAKE_PREFIX_PATH={prefix_path}")

    print("  cmake configure …")
    r = subprocess.run(cmake_cmd, cwd=build_dir)
    if r.returncode != 0:
        print("  !! cmake configure failed.")
        _print_gpgomea_manual_hint(repo_path)
        return False, []

    cpu = os.cpu_count() or 2
    print(f"  cmake build (j={cpu}) …")
    r = subprocess.run(["cmake", "--build", ".", f"-j{cpu}"], cwd=build_dir)
    if r.returncode != 0:
        print("  !! cmake build failed.")
        _print_gpgomea_manual_hint(repo_path)
        return False, []

    exts = (_glob.glob(os.path.join(build_dir, "**", "*.so"),  recursive=True) +
            _glob.glob(os.path.join(build_dir, "**", "*.pyd"), recursive=True))
    if not exts:
        print("  !! cmake succeeded but no .so/.pyd found in build dir.")
        print("  Build dir contents:")
        for root, _, files in os.walk(build_dir):
            for f in files:
                if f.endswith((".so", ".pyd", ".dylib")):
                    print(f"    {os.path.join(root, f)}")
    return True, exts


def _print_gpgomea_manual_hint(repo_path):
    conda_prefix = os.environ.get("CONDA_PREFIX", "$CONDA_PREFIX")
    print("  Manual build:")
    print(f"    conda install -c conda-forge armadillo boost   # C++ deps")
    print(f"    cd {repo_path}")
    print(f"    mkdir -p _pybuild && cd _pybuild")
    print(f"    cmake .. -DPYTHON_EXECUTABLE={sys.executable} \\")
    print(f"             -DCMAKE_BUILD_TYPE=Release \\")
    print(f"             -DBUILD_R_BINDING=OFF \\")
    print(f"             -DCMAKE_PREFIX_PATH={conda_prefix}")
    print(f"    cmake --build . -j$(nproc)")
    print("  macOS: make sure Xcode CLT is installed:  xcode-select --install")


def build_gpgomea(clone_dir=None):
    """
    Clone and build GP-GOMEA from https://github.com/marcovirgolin/GP-GOMEA.

    Prerequisites checked automatically: git, cmake, C++ compiler, pybind11.
    The package is installed into the current Python environment via pip install .
    """
    print("\nBuilding GP-GOMEA from source …")

    # ── auto-install cmake if missing ─────────────────────────────────────────
    if not shutil.which("cmake"):
        print("  cmake not found — attempting auto-install …")
        installed = False
        conda = shutil.which("conda")
        if conda:
            print("  conda install -c conda-forge cmake …")
            r = subprocess.run([conda, "install", "-c", "conda-forge", "cmake", "-y"],
                               capture_output=False)
            installed = r.returncode == 0 and shutil.which("cmake")
        if not installed:
            brew = shutil.which("brew")
            if brew:
                print("  brew install cmake …")
                r = subprocess.run(["brew", "install", "cmake"], capture_output=False)
                installed = r.returncode == 0 and shutil.which("cmake")
        if not installed:
            print("  !! cmake install failed. Run manually:")
            print("       conda install -c conda-forge cmake")
            print("       # or: brew install cmake")
            return False

    # ── check remaining prerequisites ─────────────────────────────────────────
    missing = []
    if not shutil.which("git"):
        missing.append("git  (brew install git  /  conda install git)")
    for cc in ("g++", "clang++", "c++"):
        if shutil.which(cc):
            break
    else:
        missing.append("C++17 compiler  (Xcode CLT: xcode-select --install)")

    if missing:
        print("  !! Missing prerequisites:")
        for m in missing:
            print(f"       {m}")
        print("  GP-GOMEA build skipped.")
        return False

    # pybind11 must be a Python package (not just the system headers)
    try:
        import pybind11  # noqa: F401
    except ImportError:
        print("  Installing pybind11 (Python package) …")
        pip("pybind11")

    # Armadillo (C++ linear algebra — required by GP-GOMEA's CMakeLists.txt)
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    arma_header  = os.path.join(conda_prefix, "include", "armadillo") if conda_prefix else ""
    if not (arma_header and os.path.exists(arma_header)):
        print("  Armadillo not found — installing …")
        arma_installed = False

        # prefer mamba (faster solver), fall back to conda, then brew
        for pkg_mgr in filter(None, [shutil.which("mamba"), shutil.which("conda")]):
            if conda_prefix:
                print(f"  {os.path.basename(pkg_mgr)} install -c conda-forge armadillo …")
                r = subprocess.run(
                    [pkg_mgr, "install", "-c", "conda-forge", "armadillo", "-y"],
                    capture_output=False)
                if r.returncode == 0 and os.path.exists(arma_header):
                    arma_installed = True
                    break
                print(f"  {os.path.basename(pkg_mgr)} failed — trying next option …")

        if not arma_installed:
            brew = shutil.which("brew")
            if brew:
                print("  brew install armadillo …")
                r = subprocess.run(["brew", "install", "armadillo"], capture_output=False)
                arma_installed = r.returncode == 0

        if not arma_installed:
            print("  !! Could not auto-install Armadillo. Install manually:")
            print("       conda install -c conda-forge armadillo")
            print("       # or: brew install armadillo")

    # Boost (system/filesystem/program_options + Boost.Python + Boost.NumPy)
    boost_header = os.path.join(conda_prefix, "include", "boost") if conda_prefix else ""
    if not (boost_header and os.path.exists(boost_header)):
        print("  Boost not found — installing …")
        boost_installed = False

        for pkg_mgr in filter(None, [shutil.which("mamba"), shutil.which("conda")]):
            if conda_prefix:
                print(f"  {os.path.basename(pkg_mgr)} install -c conda-forge boost …")
                r = subprocess.run(
                    [pkg_mgr, "install", "-c", "conda-forge", "boost", "-y"],
                    capture_output=False)
                if r.returncode == 0 and os.path.exists(boost_header):
                    boost_installed = True
                    break
                print(f"  {os.path.basename(pkg_mgr)} failed — trying next option …")

        if not boost_installed:
            brew = shutil.which("brew")
            if brew:
                print("  brew install boost boost-python3 …")
                r = subprocess.run(["brew", "install", "boost", "boost-python3"],
                                   capture_output=False)
                boost_installed = r.returncode == 0

        if not boost_installed:
            print("  !! Could not auto-install Boost. Install manually:")
            print("       conda install -c conda-forge boost")
            print("       # or: brew install boost boost-python3")

    # ── clone ──────────────────────────────────────────────────────────────────
    if clone_dir is None:
        clone_dir = os.path.join(os.path.expanduser("~"), ".gpgomea_src")
    repo_url  = "https://github.com/marcovirgolin/GP-GOMEA.git"
    repo_path = os.path.join(clone_dir, "GP-GOMEA")
    os.makedirs(clone_dir, exist_ok=True)

    if os.path.exists(repo_path):
        print(f"  Repo exists at {repo_path} — pulling latest …")
        subprocess.run(["git", "-C", repo_path, "pull", "--quiet"], check=False)
    else:
        print(f"  Cloning {repo_url} → {repo_path} …")
        r = subprocess.run(
            ["git", "clone", "--depth=1", "--quiet", repo_url, repo_path])
        if r.returncode != 0:
            print("  !! Clone failed. Check network / git config.")
            return False

    # ── locate install root (setup.py / pyproject.toml may be in a subdir) ───
    install_dir = _find_python_package_root(repo_path, depth=3)
    if install_dir:
        print(f"  Found Python package at: {install_dir}")

    # ── patch src/CMakeLists.txt: make R binding optional ────────────────────
    # The Python binding (gpgomea_python) is defined before the R block, so
    # cmake configure succeeds and produces the .so we need even without R.
    src_cmake = os.path.join(repo_path, "src", "CMakeLists.txt")
    if os.path.exists(src_cmake):
        with open(src_cmake, "r") as _f:
            _txt = _f.read()
        _old = 'message(FATAL_ERROR "R executable not found in PATH; cannot build R binding.")'
        _new = 'message(STATUS "R not found — skipping R binding (Python binding already defined).")'
        if _old in _txt:
            with open(src_cmake, "w") as _f:
                _f.write(_txt.replace(_old, _new))
            print("  Patched src/CMakeLists.txt: R binding made optional.")

    # ── step 1: cmake (always — pip install alone produces a pure-Python wheel
    #            with no compiled extension, causing ImportError at runtime) ───
    cmake_ok, built_exts = _cmake_compile(repo_path)
    if not cmake_ok:
        return False

    # ── step 2: pip install (installs the Python wrapper into site-packages) ──
    if install_dir:
        print("  pip install . (Python wrapper) …")
        subprocess.run([sys.executable, "-m", "pip", "install", "."],
                       cwd=install_dir)

    # ── step 3: copy compiled extensions into the installed package dir ───────
    # pyGPGOMEA/__init__.py uses relative imports, so the .so must live
    # alongside the Python files in site-packages/pyGPGOMEA/, not at top level.
    import sysconfig
    platlib = sysconfig.get_path("platlib")
    if built_exts:
        pkg_dir  = os.path.join(platlib, "pyGPGOMEA")
        dest_dir = pkg_dir if os.path.isdir(pkg_dir) else platlib
        for ext in built_exts:
            dest = os.path.join(dest_dir, os.path.basename(ext))
            shutil.copy2(ext, dest)
            print(f"  Copied {os.path.basename(ext)} → {dest_dir}")
    else:
        print("  !! cmake produced no extensions — see build output above.")
        return False

    # ── step 4: verify import ─────────────────────────────────────────────────
    import importlib
    importlib.invalidate_caches()
    for mod_name in ("pyGPGOMEA", "pygpgomea"):
        try:
            importlib.import_module(mod_name)
            print(f"  import {mod_name}  OK")
            return True
        except ImportError:
            pass

    print("  !! pyGPGOMEA not importable after build.")
    print(f"  Extensions placed in: {dest_dir}")
    print("  Check that the .so name matches what __init__.py imports.")
    print(f"  Hint: ls {dest_dir}")
    return False


# ── Julia / PySR setup ────────────────────────────────────────────────────────

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
                return False
        print("  Install Julia manually from: https://julialang.org/downloads/")
        return False

    else:   # macOS / Linux — juliaup non-interactive installer
        r = subprocess.run(
            ["curl", "-fsSL", "https://install.julialang.org"],
            capture_output=True, text=True)
        if r.returncode == 0:
            r2 = subprocess.run(
                ["sh", "-s", "--", "--yes"],
                input=r.stdout, text=True, capture_output=True)
            if r2.returncode == 0:
                julia_bin = shutil.which("julia") or \
                    subprocess.run(
                        ["bash", "-lc", "which julia"],
                        capture_output=True, text=True).stdout.strip()
                if julia_bin:
                    print(f"  Julia installed at {julia_bin}")
                    return True
                print("  Julia installed. Open a new terminal and re-run this script.")
                return False

        # macOS Homebrew fallback
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
        import juliapkg
        juliapkg.resolve()
        print("  PySR Julia environment ready.")
    except Exception as e:
        print(f"  PySR Julia init: {e}")
        print("  If Julia is freshly installed, open a new terminal and run:")
        print("    python -c \"import juliapkg; juliapkg.resolve()\"")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}

    # 1. pip packages
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

    # 2. pyoperon macOS libzstd fix
    print(f"\n{'─'*60}")
    print("Checking pyoperon shared-library linkage (macOS) …")
    fix_pyoperon_macos()

    # 3. Julia / PySR
    print(f"\n{'─'*60}")
    print("Setting up Julia for PySR …")
    if results.get("pysr") == "OK":
        if install_julia_if_needed():
            init_pysr_julia()
    else:
        print("  Skipping (pysr not installed).")

    # 4. GP-GOMEA from source
    print(f"\n{'─'*60}")
    results["pygpgomea"] = "OK" if build_gpgomea() else "FAILED (see above)"

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Installation summary:")
    all_items = [(p, i, d) for p, i, d in PACKAGES] + \
                [("pygpgomea", "pyGPGOMEA", "GP-GOMEA — linkage-learning GP (source build)")]
    for pip_name, _, desc in all_items:
        status = results.get(pip_name, "SKIPPED")
        mark = "✓" if status == "OK" else "✗"
        print(f"  {mark}  {pip_name:20s}  {status}")

    print()
    print("Verifying imports …")
    for pip_name, import_name, _ in all_items:
        found = _verify_import(pip_name, import_name)
        if found:
            print(f"  OK  {pip_name} (import {found})")
        else:
            print(f"  --  {pip_name}  (not importable)")
