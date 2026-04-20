"""
plot_empirical_figures.py
=========================
Standalone plotting script for all empirical figures in:

    "When Does Orthogonality Fail to Protect?
     Ridge versus Lasso as Nuisance Estimators in Double Machine Learning"

Usage:
    python plot_empirical_figures.py

Input:
    results_empirical_shifted.csv   (produced by empirical3.py)

Output (saved to ./figures_empirical/):
    fig1_r2_comparison.png/.pdf       -- primary treatment R2 (outcome + treatment)
    fig2_delta_n.png/.pdf             -- nuisance error product diagnostic
    fig3_stability.png/.pdf           -- estimate stability (primary treatment)
    fig4_signal_positioning.png/.pdf  -- signal strength positioning
    fig5_stable_theta.png/.pdf        -- theta_hat, stable treatments only
    fig6_ci_width.png/.pdf            -- CI width (primary treatment)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings('ignore')

CSV_PATH = 'results_empirical_shifted.csv'
SAVE_DIR = 'figures_empirical_regimes'
DPI      = 150

PAL    = {'lasso': '#e74c3c', 'ridge': '#2980b9'}
LABELS = {'lasso': 'LASSO',   'ridge': 'RIDGE'}

PRIMARY = {
    'FRED-MD':     'CPIAUCSL',
    'Barro-Lee':   'invsh41',
    'Goyal-Welch': 'd_p',
}

DS_ORDER = ['Barro-Lee', 'FRED-MD', 'Goyal-Welch']
METHODS  = ['lasso', 'ridge']

# --- Regime thresholds (used only for stable_long / pathology flag) ---
R2M_PATH_HI = 0.98
R2M_PATH_LO = -0.10

os.makedirs(SAVE_DIR, exist_ok=True)


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['method'] = df['method'].str.lower()
    return df


def flag_pathological(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark treatments as pathological if R2_m > 0.98 or R2_m < -0.10
    for either method. Works on the long-format DataFrame.
    """
    ridge = df[df['method'] == 'ridge'][['dataset', 'treatment', 'm_r2']].copy()
    lasso = df[df['method'] == 'lasso'][['dataset', 'treatment', 'm_r2']].copy()
    ridge.columns = ['dataset', 'treatment', 'r2m_ridge']
    lasso.columns = ['dataset', 'treatment', 'r2m_lasso']
    w = ridge.merge(lasso, on=['dataset', 'treatment'])
    w['pathological'] = (
        (w['r2m_ridge'] > R2M_PATH_HI) | (w['r2m_lasso'] > R2M_PATH_HI) |
        (w['r2m_ridge'] < R2M_PATH_LO) | (w['r2m_lasso'] < R2M_PATH_LO)
    )
    pat_map = w.set_index(['dataset', 'treatment'])['pathological'].to_dict()
    df['pathological'] = df.apply(
        lambda r: pat_map.get((r['dataset'], r['treatment']), False), axis=1)
    return df


def stable_long(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df['pathological']].copy()


def primary_subset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, t in PRIMARY.items():
        sub = df[(df['dataset'] == ds) & (df['treatment'] == t)]
        if sub.empty:
            print(f"  WARNING: primary treatment '{t}' not found for '{ds}'")
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)



def bar_label(ax, bar, val, fmt='{:.2f}', offset_frac=0.04, fontsize=9):
    h  = bar.get_height()
    y  = h + abs(h) * offset_frac if h >= 0 else h - abs(h) * offset_frac
    va = 'bottom' if h >= 0 else 'top'
    ax.text(bar.get_x() + bar.get_width() / 2, y,
            fmt.format(val), ha='center', va=va, fontsize=fontsize)


def save(fig, name: str):
    for ext in ('png', 'pdf'):
        fig.savefig(f'{SAVE_DIR}/{name}.{ext}', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {name}')



def fig1_r2_comparison(df: pd.DataFrame):
    prim     = primary_subset(df)
    datasets = [ds for ds in DS_ORDER if ds in prim['dataset'].values]
    x, width = np.arange(len(datasets)), 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    specs = [
        ('ell_r2', 'Outcome nuisance: out-of-fold $R^2$',
         '$R^2_\\ell$ (outcome regression)'),
        ('m_r2',   'Treatment nuisance: out-of-fold $R^2$',
         '$R^2_m$ (treatment regression)'),
    ]
    for ax, (metric, title, ylabel) in zip(axes, specs):
        for i, method in enumerate(METHODS):
            sub  = prim[prim['method'] == method]
            vals = [sub.loc[sub['dataset'] == ds, metric].values[0]
                    if ds in sub['dataset'].values else np.nan
                    for ds in datasets]
            bars = ax.bar(x + i * width, vals, width,
                          label=LABELS[method], color=PAL[method], alpha=0.85)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    bar_label(ax, bar, val, fmt='{:.2f}')
        ax.axhline(0, color='black', lw=1, ls='--')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(datasets, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        # Headroom above tallest bar for labels
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, yhi * 1.18)
    fig.suptitle('First-Stage Prediction: Ridge vs Lasso', fontsize=13)
    plt.tight_layout()
    save(fig, 'fig1_r2_comparison')



def fig2_delta_n(df: pd.DataFrame):
    prim     = primary_subset(df)
    datasets = [ds for ds in DS_ORDER if ds in prim['dataset'].values]
    x, width = np.arange(len(datasets)), 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, method in enumerate(METHODS):
        sub  = prim[prim['method'] == method]
        vals = [sub.loc[sub['dataset'] == ds, 'sqrt_n_delta_norm'].values[0]
                if ds in sub['dataset'].values else np.nan
                for ds in datasets]
        bars = ax.bar(x + i * width, vals, width,
                      label=LABELS[method], color=PAL[method], alpha=0.85)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                bar_label(ax, bar, val, fmt='{:.2f}', offset_frac=0.02)
    # Headroom above tallest bar for labels
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi * 1.15)
    ax.axhline(1.0, color='black', lw=1.5, ls='--', label='threshold = 1')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel(
        r'$\sqrt{n}\,\hat{\Delta}_n$  [normalised by SD($Y$)\,SD($D$)]',
        fontsize=10)
    ax.set_title('Nuisance Error Product Diagnostic', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save(fig, 'fig2_delta_n')



def fig3_stability(df: pd.DataFrame):
    prim     = primary_subset(df)
    datasets = [ds for ds in DS_ORDER if ds in prim['dataset'].values]
    x, width = np.arange(len(datasets)), 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, method in enumerate(METHODS):
        sub  = prim[prim['method'] == method]
        vals = [sub.loc[sub['dataset'] == ds, 'theta_std'].values[0]
                if ds in sub['dataset'].values else np.nan
                for ds in datasets]
        bars = ax.bar(x + i * width, vals, width,
                      label=LABELS[method], color=PAL[method], alpha=0.85)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                bar_label(ax, bar, val, fmt='{:.3f}', offset_frac=0.05)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    # Headroom above tallest bar for labels
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi * 1.5)
    ax.set_ylabel(r'SD of $\hat{\theta}$ across $S$ repetitions (log scale)',
                  fontsize=10)
    ax.set_title(
        r'Estimate Stability: SD of $\hat{\theta}$ across Cross-Fitting Splits',
        fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y', which='both')
    plt.tight_layout()
    save(fig, 'fig3_stability')



def fig4_signal_positioning(df: pd.DataFrame):
    prim     = primary_subset(df)
    ridge    = prim[prim['method'] == 'ridge']
    datasets = [ds for ds in DS_ORDER if ds in ridge['dataset'].values]

    r2_vals, tau_vals = [], []
    for ds in datasets:
        row = ridge[ridge['dataset'] == ds]
        if row.empty:
            r2_vals.append(np.nan); tau_vals.append(np.nan); continue
        tau = row['tau_hat_ell'].values[0]
        r2  = tau / (tau + 1) * 100 if not np.isnan(tau) else np.nan
        r2_vals.append(r2); tau_vals.append(tau)

    TAU_WEAK = 0.20
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(datasets, r2_vals, color='#2980b9', alpha=0.85)
    for bar, r2, tau in zip(bars, r2_vals, tau_vals):
        if r2 is None or np.isnan(r2):
            continue
        ax.text(r2 + 0.3, bar.get_y() + bar.get_height() / 2,
                f'  $R^2$={r2:.1f}%,  $\\hat{{\\tau}}$={tau:.2f}',
                va='center', fontsize=10)
    max_r2 = max((v for v in r2_vals if not np.isnan(v)), default=20)
    ax.axvline(TAU_WEAK / (TAU_WEAK + 1) * 100, color='black',
               lw=1.2, ls=':', label=f'Weak threshold ($\\hat{{\\tau}}={TAU_WEAK}$)')
    ax.set_xlabel(r'Ridge out-of-fold $R^2$, \% (outcome regression)', fontsize=10)
    ax.set_title('Signal Strength Positioning: Outcome Regression', fontsize=11)
    ax.set_xlim(0, max_r2 * 1.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    save(fig, 'fig4_signal_positioning')



def fig5_stable_theta(df: pd.DataFrame):
    stable   = stable_long(df)
    datasets = [ds for ds in DS_ORDER if ds in stable['dataset'].values]

    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(5 * len(datasets), 5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub        = stable[stable['dataset'] == ds].copy()
        treatments = sub['treatment'].unique().tolist()
        x, width   = np.arange(len(treatments)), 0.35
        all_vals   = []

        for i, method in enumerate(METHODS):
            m_sub       = sub[sub['method'] == method]
            means, stds = [], []
            for t in treatments:
                row = m_sub[m_sub['treatment'] == t]
                if len(row) > 0:
                    means.append(row['theta_mean'].values[0])
                    stds.append(row['theta_std'].values[0])
                else:
                    means.append(np.nan); stds.append(np.nan)
            ax.bar(x + i * width, means, width, yerr=stds,
                   label=LABELS[method], color=PAL[method],
                   alpha=0.85, capsize=4)
            all_vals.extend([m for m in means if not np.isnan(m)])

        if all_vals:
            lim = max(abs(v) for v in all_vals) * 1.4
            ax.set_ylim(-lim, lim)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(treatments, fontsize=9, rotation=20, ha='right')
        ax.set_title(ds, fontsize=11)
        ax.set_ylabel(r'$\hat{\theta}$  [original units]', fontsize=9)
        ax.axhline(0, color='black', lw=0.8, ls='--')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        r'$\hat{\theta}$ across Stable Treatments  (pathological excluded)',
        fontsize=12)
    plt.tight_layout()
    save(fig, 'fig5_stable_theta')



def fig6_ci_width(df: pd.DataFrame):
    prim     = primary_subset(df)
    datasets = [ds for ds in DS_ORDER if ds in prim['dataset'].values]
    x, width = np.arange(len(datasets)), 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, method in enumerate(METHODS):
        sub  = prim[prim['method'] == method]
        vals = [sub.loc[sub['dataset'] == ds, 'ci_width'].values[0]
                if ds in sub['dataset'].values else np.nan
                for ds in datasets]
        bars = ax.bar(x + i * width, vals, width,
                      label=LABELS[method], color=PAL[method], alpha=0.85)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                bar_label(ax, bar, val, fmt='{:.3f}', offset_frac=0.05)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    # Headroom above tallest bar for labels
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi * 1.5)
    ax.set_ylabel(
        r'Avg 95\% CI width  ($2\times1.96\times\hat{\sigma}_\theta$, log scale)',
        fontsize=10)
    ax.set_title('Average Confidence Interval Width: Ridge vs Lasso', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y', which='both')
    plt.tight_layout()
    save(fig, 'fig6_ci_width')



def main():
    print('=' * 65)
    print('  Empirical Figure Generator')
    print('=' * 65)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Results CSV not found: '{CSV_PATH}'\n"
            "Run empirical3.py first to generate the results file.")

    print(f'\nLoading results from {CSV_PATH}...')
    df = load_results(CSV_PATH)
    df = flag_pathological(df)
    print(f'  {len(df)} rows, {df["dataset"].nunique()} datasets, '
          f'{df["treatment"].nunique()} treatments.')

    for ds, t in PRIMARY.items():
        if df[(df['dataset'] == ds) & (df['treatment'] == t)].empty:
            print(f"  WARNING: primary treatment '{t}' missing for '{ds}'.")

    print('\nGenerating figures...')
    fig1_r2_comparison(df)
    fig2_delta_n(df)
    fig3_stability(df)
    fig4_signal_positioning(df)
    fig5_stable_theta(df)
    fig6_ci_width(df)

    print(f'\nAll figures saved to ./{SAVE_DIR}/')
    print('Done.')


if __name__ == '__main__':
    main()