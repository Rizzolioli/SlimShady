"""
paper_fig_common.py — shared palette + save-triplet (png/pdf/tikz) helper for
the paper-ready figures under main/paper_figures/.

Color scheme: 8 mutation families x 2 shades (light = SLIM+ / sum,
dark = SLIM* / mul), so the same variant has the same color across every
figure in the set (geometry grid, evolution grids, both Pareto fronts).
Extends the 7-family palette already used in plot_normalized_results.py
with one more pair for NORMFIX, reusing its established highlight color
(#E91E63 family) from plot_normfix_study.py / plot_pareto_mphi_rmse.py.
"""
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import matplot2tikz
except ImportError:  # pragma: no cover
    matplot2tikz = None

ALL_VARIANTS = [
    'SLIM+2SIG',    'SLIM*2SIG',
    'SLIM+ABS',     'SLIM*ABS',
    'SLIM+1SIG',    'SLIM*1SIG',
    'SLIM+NORM1',   'SLIM*NORM1',
    'SLIM+NORM2',   'SLIM*NORM2',
    'SLIM+NORMROB', 'SLIM*NORMROB',
    'SLIM+NORM12',  'SLIM*NORM12',
    'SLIM+NORMFIX', 'SLIM*NORMFIX',
]

DATASETS = ['concrete', 'energy', 'instanbul', 'ppb', 'resid_build_sale_price', 'toxicity']
DS_LABELS = {
    'concrete':               'Concrete',
    'energy':                 'Energy',
    'instanbul':              'Istanbul',
    'ppb':                    'PPB',
    'resid_build_sale_price': 'Resid. Build',
    'toxicity':               'Toxicity',
}

# (light hex = sum '+', dark hex = mul '*')
FAMILY_PALETTE = {
    '2SIG':    ('#aec7e8', '#1f77b4'),
    'ABS':     ('#ff9896', '#d62728'),
    '1SIG':    ('#98df8a', '#2ca02c'),
    'NORM1':   ('#c5b0d5', '#9467bd'),
    'NORM2':   ('#c49c94', '#8c564b'),
    'NORMROB': ('#ffbb78', '#ff7f0e'),
    'NORM12':  ('#f7b6d2', '#e377c2'),
    'NORMFIX': ('#f48fb1', '#E91E63'),
}

FAMILY_ORDER = ['2SIG', 'ABS', '1SIG', 'NORM1', 'NORM2', 'NORMROB', 'NORM12', 'NORMFIX']

# Variant draw order for legends/tables/subplot rows: grouped by family, sum then mul
VARIANT_ORDER = [f'SLIM{op}{fam}' for fam in FAMILY_ORDER for op in ('+', '*')]


def variant_family(algo: str) -> str:
    """'SLIM+NORM12' -> 'NORM12'."""
    return algo[5:] if algo[:5] in ('SLIM+', 'SLIM*') else algo.split('SLIM', 1)[-1]


def variant_color(algo: str) -> str:
    """Same family -> same hue; '+' (sum) = light shade, '*' (mul) = dark shade."""
    family = variant_family(algo)
    light, dark = FAMILY_PALETTE[family]
    return light if '+' in algo else dark


def variant_linestyle(algo: str) -> str:
    """Solid for sum ('+'), dashed for mul ('*') — used when color alone isn't enough."""
    return '-' if '+' in algo else '--'


def variant_marker(algo: str) -> str:
    """Triangle for sum ('+'), square for mul ('*') — shape encodes operator independent of
    color, so e.g. all SLIM* variants share one shape across a scatter plot."""
    return '^' if '+' in algo else 's'


def save_all(fig, out_dir: str, name: str, tikz_ok: bool = True, dpi: int = 300):
    """Save fig as <name>.png, <name>.pdf, and (if tikz_ok) <name>.tex via matplot2tikz.

    A matplot2tikz failure (unsupported artist, etc.) is caught and warned about
    rather than raised, so png/pdf are never lost because tikz export choked.
    """
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f'{name}.png')
    pdf_path = os.path.join(out_dir, f'{name}.pdf')
    tex_path = os.path.join(out_dir, f'{name}.tex')

    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f'  Saved -> {png_path}')
    print(f'  Saved -> {pdf_path}')

    if tikz_ok:
        if matplot2tikz is None:
            warnings.warn(f'[{name}] matplot2tikz not importable — skipping .tex export')
        else:
            try:
                code = matplot2tikz.get_tikz_code(fig)
                with open(tex_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                print(f'  Saved -> {tex_path}')
            except Exception as exc:  # noqa: BLE001 — best-effort tikz export
                warnings.warn(f'[{name}] matplot2tikz export failed, png/pdf still saved: {exc!r}')
