r"""
cauchy_schwarz.py
=================
Quantifies the Cauchy-Schwarz looseness of the sqrt(n)*Delta_n diagnostic
across four representative semi-synthetic designs on FRED-MD (n = 400).

Produces:
    (1) Table 3 in the paper (printed to stdout as LaTeX)
    (2) fig13_cauchy_schwarz.pdf (bound vs actual remainder)

The core comparison: the Cauchy-Schwarz inequality gives
    |R_n| <= ||\hat\ell - \ell_0||_{L2} * ||\hat m - m_0||_{L2} =: Delta_n
so sqrt(n) * Delta_n bounds sqrt(n) * |R_n|. In symmetric designs the bound
is approximately 5-6 times the actual remainder. In asymmetric designs, the
bound exceeds the actual remainder by two orders of magnitude, because
Ridge's large treatment-equation error is misaligned with its outcome-equation
error, driving the inner product n^(-1) sum_i (\hat\ell_i - \ell_{0,i})
(\hat m_i - m_{0,i}) toward zero despite the large product of norms.

Inputs: aggregated bias and sqrt(n)*Delta_n values from the FRED-MD
semi-synthetic runs at n = 400, copied from results tables.

Usage:
    python cauchy_schwarz.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Output setup
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      '..', 'paper', 'figuresfinal')
os.makedirs(OUT_DIR, exist_ok=True)

# Plot style
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.linewidth': 0.6,
    'axes.labelsize': 12, 'axes.titlesize': 12, 'figure.dpi': 200,
    'legend.fontsize': 9.5, 'legend.framealpha': 0.95,
    'legend.edgecolor': '0.7', 'grid.alpha': 0.25, 'grid.linewidth': 0.4,
})

LC = '#2166AC'   # Lasso colour (blue)
RC = '#B2182B'   # Ridge colour (red)

# FRED-MD semi-synthetic results at n = 400 (from aggregated simulation runs).
# Columns: (design_label, tau_value, lasso_bias, lasso_sqrtNDn, ridge_bias, ridge_sqrtNDn)
# Bias is the mean of (theta_hat - theta_0) across B = 100 replications;
# sqrt(n) * |bias| is the (noisy) approximation of the actual remainder
# sqrt(n) * |R_n| used for the bound-vs-actual comparison.
CELLS_N400 = [
    # Symmetric dense (s/p = 1.0, beta_m = beta_g)
    ('Sym. dense',      0.20,  0.0499, 6.464,  0.0386, 4.536),
    ('Sym. dense',      0.10,  0.0376, 4.625,  0.0300, 3.266),
    ('Sym. dense',      0.05,  0.0259, 3.432,  0.0229, 2.337),
    ('Sym. dense',      0.01,  0.0090, 1.709,  0.0098, 1.074),
    # Asymmetric: sparse beta_m + dense beta_g (tau_m = 2.0, s_m/p = 0.05)
    ('Asym. sparse bm', 0.20,  0.0029, 3.474, -0.0011, 11.126),
    ('Asym. sparse bm', 0.10,  0.0011, 3.171, -0.0024, 11.033),
    ('Asym. sparse bm', 0.05,  0.0001, 2.923, -0.0035, 10.983),
    # Asymmetric: dense beta_m + sparse beta_g (tau_g = 2.0, s_g/p = 0.05)
    ('Asym. dense bm',  0.10, -0.0023, 4.044,  0.0443, 5.200),
    ('Asym. dense bm',  0.05, -0.0029, 3.207,  0.0436, 4.366),
    # Symmetric sparse baseline (s/p = 0.05, tau = 2.0)
    ('Sym. sparse',     2.00,  0.0031, 2.574,  0.0697, 13.885),
]

N = 400
SQN = np.sqrt(N)


def print_latex_table():
    """Print Table 3 (four representative designs) as LaTeX."""
    # Representative subset shown in the paper
    representative = [
        ('Symmetric dense ($\\tau = 0.1$)',          'Sym. dense',      0.10),
        ('Asym.\\ sparse $\\beta_m$ ($\\tau_g = 0.1$)', 'Asym. sparse bm', 0.10),
        ('Asym.\\ dense $\\beta_m$ ($\\tau_m = 0.1$)',  'Asym. dense bm',  0.10),
        ('Symmetric sparse ($\\tau = 2.0$)',           'Sym. sparse',     2.00),
    ]

    print("% Table 3: Cauchy-Schwarz tightness")
    print("\\begin{table}[H]")
    print("\\centering \\small")
    print("\\caption{Cauchy-Schwarz tightness: bound ($\\sqrt{n}\\cdot\\Delta_n$) "
          "vs.\\ approximate actual remainder ($\\sqrt{n}\\cdot|\\text{Bias}|$) "
          "on FRED-MD ($n = 400$).}")
    print("\\label{tab:cauchy}")
    print("\\begin{tabular}{llcccc}")
    print("\\toprule")
    print("Design & Method & $\\sqrt{n}\\cdot\\Delta_n$ & "
          "$\\sqrt{n}\\cdot|\\text{Bias}|$ & Bound/Actual \\\\")
    print("\\midrule")
    for label, design_key, tau_key in representative:
        for cell in CELLS_N400:
            if cell[0] == design_key and abs(cell[1] - tau_key) < 1e-6:
                _, _, l_bias, l_snd, r_bias, r_snd = cell
                l_actual = SQN * abs(l_bias)
                r_actual = SQN * abs(r_bias)
                l_ratio = l_snd / l_actual if l_actual > 0 else float('inf')
                r_ratio = r_snd / r_actual if r_actual > 0 else float('inf')
                print(f"{label} & Lasso & {l_snd:.2f} & {l_actual:.2f} "
                      f"& ${l_ratio:.0f}\\times$ \\\\")
                print(f" & Ridge & {r_snd:.2f} & {r_actual:.2f} "
                      f"& ${r_ratio:.0f}\\times$ \\\\")
                print("\\midrule")
                break
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def make_figure():
    """Produce fig13_cauchy_schwarz.pdf: scatter of bound vs actual."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # Plot each design class with a different marker
    markers = {
        'Sym. dense':      ('o', 'Symmetric dense'),
        'Asym. sparse bm': ('s', 'Asymmetric (sparse $\\beta_m$)'),
        'Asym. dense bm':  ('^', 'Asymmetric (dense $\\beta_m$)'),
        'Sym. sparse':     ('D', 'Symmetric sparse'),
    }
    seen_lasso, seen_ridge = set(), set()

    for label, tau, l_bias, l_snd, r_bias, r_snd in CELLS_N400:
        marker, legend_label = markers[label]
        l_actual = SQN * abs(l_bias)
        r_actual = SQN * abs(r_bias)

        lasso_label = (f'Lasso, {legend_label}'
                       if legend_label not in seen_lasso else None)
        ridge_label = (f'Ridge, {legend_label}'
                       if legend_label not in seen_ridge else None)
        seen_lasso.add(legend_label)
        seen_ridge.add(legend_label)

        ax.scatter(l_actual, l_snd, marker=marker, s=90,
                   facecolors='none', edgecolors=LC, linewidths=1.5,
                   label=lasso_label)
        ax.scatter(r_actual, r_snd, marker=marker, s=90,
                   facecolors='none', edgecolors=RC, linewidths=1.5,
                   label=ridge_label)

    # Reference lines: y = x (tight), y = 10x, y = 100x
    x = np.logspace(-3, 1, 100)
    ax.plot(x, x,       color='black', lw=0.8, ls='-',
            alpha=0.4, label='tight ($y = x$)')
    ax.plot(x, 10 * x,  color='black', lw=0.8, ls='--',
            alpha=0.4, label='$y = 10x$')
    ax.plot(x, 100 * x, color='black', lw=0.8, ls=':',
            alpha=0.4, label='$y = 100x$')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(0.01, 5); ax.set_ylim(0.5, 30)
    ax.set_xlabel(r'Actual remainder $\sqrt{n}\cdot|\text{Bias}|$')
    ax.set_ylabel(r'Bound $\sqrt{n}\cdot\Delta_n$')
    ax.set_title('Cauchy-Schwarz Bound vs Actual Remainder (FRED-MD, $n = 400$)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower right', ncol=2, fontsize=8, framealpha=0.95)

    out_path = os.path.join(OUT_DIR, 'fig13_cauchy_schwarz.pdf')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f'Saved figure: {out_path}')


if __name__ == '__main__':
    print("=" * 65)
    print("  Cauchy-Schwarz Tightness Analysis (FRED-MD, n = 400)")
    print("=" * 65)
    print()
    print_latex_table()
    print()
    make_figure()
    print("Done.")
