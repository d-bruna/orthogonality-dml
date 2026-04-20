"""
simulation.py
=============
Simulation study for:
    "When Does Orthogonality Fail to Protect?
     Ridge versus Lasso as Nuisance Estimators in Double Machine Learning"

Implementation notes:
    - LassoCV uses a coarser alpha grid (15 values) and max_iter=3000 for speed
    - RidgeCV uses efficient leave-one-out (no change needed)
    - Replications within each cell are parallelised with joblib (n_jobs=-1)
    - Inner CV folds = 3 (sufficient for lambda selection)
    - S=3 for the scan pass, S=10 for the focused pass, S=20 for the full grid
    - Checkpointing: completed cells are saved to CSV and skipped on restart

Usage:
    python simulation.py --demo       ~3 min sanity check
    python simulation.py --scan       ~2-3 hrs, full 648-cell landscape (B=100, S=3)
    python simulation.py --focused    ~2-3 hrs, breakdown regime only (B=500, S=10)
    python simulation.py --quick      ~30-60 min, meaningful subset
    python simulation.py              full paper grid (B=500, S=20, many hours)
    python simulation.py --B 200 --S 5   override B and S for any mode
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import argparse
import os
import time
from itertools import product

from joblib import Parallel, delayed
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import logging
logging.getLogger("joblib").setLevel(logging.ERROR)


# --- Utilities ---------------------------------------------------------------

def log(msg, end='\n'):
    print(msg, end=end, flush=True)


def progress_bar(current, total, width=40):
    frac   = current / total
    filled = int(width * frac)
    return f"[{'#'*filled + '-'*(width-filled)}] {current}/{total} ({100*frac:.0f}%)"


# --- Data-generating process -------------------------------------------------

def generate_data(n, p, tau, s_ratio, rho, theta_0=1.0, seed=None):
    """
    Draw one replication from the partially linear model

        D = X @ beta_m + v
        Y = theta_0 * D + X @ beta_g + epsilon

    Coefficients: beta_j = sqrt(tau/s) for j <= s, else 0, so ||beta||^2 = tau.
    Toeplitz covariance: Sigma_jk = rho^|j-k|.
    """
    rng   = np.random.default_rng(seed)
    idx   = np.arange(p)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    L     = np.linalg.cholesky(Sigma + 1e-10 * np.eye(p))
    X     = rng.standard_normal((n, p)) @ L.T

    s      = max(1, int(s_ratio * p))
    beta_g = np.zeros(p); beta_g[:s] = np.sqrt(tau / s)
    beta_m = np.zeros(p); beta_m[:s] = np.sqrt(tau / s)

    v       = rng.standard_normal(n)
    epsilon = rng.standard_normal(n)
    m_0     = X @ beta_m
    g_0     = X @ beta_g
    D       = m_0 + v
    Y       = theta_0 * D + g_0 + epsilon
    ell_0   = theta_0 * m_0 + g_0   # E[Y|X]

    return X, D, Y, ell_0, m_0, beta_g, beta_m


# --- Nuisance fitting --------------------------------------------------------

def fit_nuisance(X_tr, y_tr, method, K_inner):
    """
    Fit a single nuisance function on the training fold.

    Lasso uses a 15-alpha grid with max_iter=3000 for speed.
    Ridge uses RidgeCV (leave-one-out, already fast).
    OLS is only feasible when p < n_train.
    """
    n_tr, p = X_tr.shape

    if method == 'lasso':
        model = LassoCV(
            alphas=np.logspace(-4, 1, 15),
            cv=K_inner,
            max_iter=3_000,
            n_jobs=1,               # outer parallelism is handled by joblib
            random_state=0,
        ).fit(X_tr, y_tr)

    elif method == 'ridge':
        model = RidgeCV(
            alphas=np.logspace(-4, 6, 60),
            cv=K_inner,
            scoring='neg_mean_squared_error',
        ).fit(X_tr, y_tr)

    elif method == 'ols':
        if p >= n_tr:
            raise ValueError(f"OLS not feasible: p={p} >= n_train={n_tr}")
        model = LinearRegression().fit(X_tr, y_tr)

    else:
        raise ValueError(f"Unknown method: {method}")

    return model


# --- DML estimator -----------------------------------------------------------

def dml_estimator(X, D, Y, method, K_outer=5, K_inner=3, seed=0):
    """
    Partialling-out DML with K_outer-fold cross-fitting.

    Returns theta_hat, its sandwich standard error, the out-of-fold
    residuals, and the out-of-fold nuisance predictions.
    """
    n          = len(Y)
    ell_resid  = np.zeros(n)
    v_resid    = np.zeros(n)
    ell_fitted = np.zeros(n)
    m_fitted   = np.zeros(n)

    kf = KFold(n_splits=K_outer, shuffle=True, random_state=seed)

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        D_tr, D_te = D[train_idx], D[test_idx]

        # Standardise on the training fold only (no leakage).
        scaler  = StandardScaler().fit(X_tr)
        X_tr_s  = scaler.transform(X_tr)
        X_te_s  = scaler.transform(X_te)

        ell_mod  = fit_nuisance(X_tr_s, Y_tr, method, K_inner)
        m_mod    = fit_nuisance(X_tr_s, D_tr, method, K_inner)
        ell_pred = ell_mod.predict(X_te_s)
        m_pred   = m_mod.predict(X_te_s)

        ell_fitted[test_idx] = ell_pred
        m_fitted[test_idx]   = m_pred
        ell_resid[test_idx]  = Y_te - ell_pred
        v_resid[test_idx]    = D_te - m_pred

    denom     = np.sum(v_resid ** 2)
    theta_hat = np.sum(v_resid * ell_resid) / denom

    # Sandwich variance estimator.
    eps_hat   = ell_resid - theta_hat * v_resid
    meat      = np.mean(v_resid ** 2 * eps_hat ** 2)
    bread     = np.mean(v_resid ** 2)
    sigma_hat = np.sqrt(meat / (bread ** 2 * n))

    return theta_hat, sigma_hat, ell_resid, v_resid, ell_fitted, m_fitted


# --- Metrics -----------------------------------------------------------------

def compute_metrics(theta_hat, sigma_hat, ell_fitted, m_fitted,
                    ell_0_true, m_0_true, theta_0=1.0):
    """
    Compute the per-replication evaluation metrics.

    Delta_n = ||ell_hat - ell_0|| * ||m_hat - m_0|| is the nuisance error
    product. R^2 is computed against the true nuisance functions, not against
    the noisy observations Y and D.
    """
    n        = len(ell_0_true)
    bias     = theta_hat - theta_0
    ci_lo    = theta_hat - 1.96 * sigma_hat
    ci_hi    = theta_hat + 1.96 * sigma_hat
    covered  = float(ci_lo <= theta_0 <= ci_hi)
    ci_width = ci_hi - ci_lo

    ell_mse      = np.mean((ell_fitted - ell_0_true) ** 2)
    m_mse        = np.mean((m_fitted   - m_0_true  ) ** 2)
    delta_n      = np.sqrt(ell_mse) * np.sqrt(m_mse)
    sqrt_n_delta = np.sqrt(n) * delta_n

    ss_ell = np.sum((ell_0_true - ell_0_true.mean()) ** 2) + 1e-12
    ss_m   = np.sum((m_0_true   - m_0_true.mean()  ) ** 2) + 1e-12
    ell_r2 = 1 - np.sum((ell_fitted - ell_0_true) ** 2) / ss_ell
    m_r2   = 1 - np.sum((m_fitted   - m_0_true  ) ** 2) / ss_m

    return dict(
        theta_hat    = theta_hat,
        bias         = bias,
        covered      = covered,
        ci_width     = ci_width,
        ell_mse      = ell_mse,
        m_mse        = m_mse,
        delta_n      = delta_n,
        sqrt_n_delta = sqrt_n_delta,
        ell_r2       = ell_r2,
        m_r2         = m_r2,
    )


# --- One replication (top-level so joblib can pickle it) ---------------------

def _one_rep(b, n, p, tau, s_ratio, rho, theta_0, S, K_outer, K_inner, methods):
    """
    Run one Monte Carlo replication with S cross-fitting repetitions.

    Averages theta_hat and sigma_hat over the S repetitions before computing
    metrics. Runs in parallel from joblib.
    """
    ols_feasible = (p < n)
    X, D, Y, ell_0, m_0, _, _ = generate_data(
        n=n, p=p, tau=tau, s_ratio=s_ratio,
        rho=rho, theta_0=theta_0, seed=b
    )
    records = []

    for method in methods:
        if method == 'ols' and not ols_feasible:
            continue

        rep_thetas, rep_sigmas = [], []
        rep_ell = np.zeros(n)
        rep_m   = np.zeros(n)

        for s_rep in range(S):
            try:
                th, sig, _, _, ef, mf = dml_estimator(
                    X, D, Y, method,
                    K_outer=K_outer, K_inner=K_inner,
                    seed=b * S + s_rep
                )
                rep_thetas.append(th)
                rep_sigmas.append(sig)
                rep_ell += ef
                rep_m   += mf
            except Exception:
                continue

        if not rep_thetas:
            continue

        k       = len(rep_thetas)
        metrics = compute_metrics(
            theta_hat  = float(np.mean(rep_thetas)),
            sigma_hat  = float(np.mean(rep_sigmas)),
            ell_fitted = rep_ell / k,
            m_fitted   = rep_m   / k,
            ell_0_true = ell_0,
            m_0_true   = m_0,
            theta_0    = theta_0,
        )
        records.append(dict(
            n=n, p=p, tau=tau, s_ratio=s_ratio, rho=rho,
            method=method, replication=b, **metrics
        ))

    return records


# --- Simulation cell (parallel over replications) ----------------------------

def run_cell(n, p, tau, s_ratio, rho,
             B=200, S=3, K_outer=5, K_inner=3,
             theta_0=1.0, methods=('lasso', 'ridge', 'ols'),
             n_jobs=-1):
    """
    Run B replications in parallel using all available cores. Each
    replication is independent, so this is embarrassingly parallel.
    """
    t0 = time.time()
    log(f"  Running {B} reps x S={S} (n_jobs={n_jobs})...", end=' ')

    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_one_rep)(
            b, n, p, tau, s_ratio, rho, theta_0, S, K_outer, K_inner, methods
        )
        for b in range(B)
    )

    elapsed = time.time() - t0
    log(f"done in {elapsed:.1f}s")

    records = [r for rep in results for r in rep]
    return pd.DataFrame(records)


# --- Aggregation -------------------------------------------------------------

def aggregate(df):
    """Collapse per-replication records to cell-level averages."""
    grp = ['n', 'p', 'tau', 's_ratio', 'rho', 'method']
    return df.groupby(grp).agg(
        bias         = ('bias',         'mean'),
        rmse         = ('bias',         lambda x: np.sqrt(np.mean(x**2))),
        coverage     = ('covered',      'mean'),
        ci_width     = ('ci_width',     'mean'),
        delta_n      = ('delta_n',      'mean'),
        sqrt_n_delta = ('sqrt_n_delta', 'mean'),
        ell_r2       = ('ell_r2',       'mean'),
        m_r2         = ('m_r2',         'mean'),
        n_reps       = ('bias',         'count'),
    ).reset_index()


# --- Checkpointing -----------------------------------------------------------

def checkpoint_path(params, save_dir='checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    fname = (f"n{params['n']}_p{params['p']}_tau{params['tau']}"
             f"_s{params['s_ratio']}_rho{params['rho']}.csv")
    return os.path.join(save_dir, fname)


def cell_already_done(params):
    return os.path.exists(checkpoint_path(params))


def save_checkpoint(df_cell, params):
    df_cell.to_csv(checkpoint_path(params), index=False)
    log(f"  Checkpoint saved -> {checkpoint_path(params)}")


def load_all_checkpoints(save_dir='checkpoints'):
    if not os.path.exists(save_dir):
        return pd.DataFrame()
    files = [f for f in os.listdir(save_dir) if f.endswith('.csv')]
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(os.path.join(save_dir, f)) for f in files]
    log(f"  Loaded {len(files)} cached cells from {save_dir}/")
    return pd.concat(dfs, ignore_index=True)


# --- Figures -----------------------------------------------------------------

def tau_to_r2(tau):
    """Population R^2 under the sigma^2 = 1 DGP is tau / (tau + 1)."""
    return tau / (tau + 1.0)


def make_figures(agg, save_dir='figures'):
    """Generate five summary figures from the aggregated results."""
    os.makedirs(save_dir, exist_ok=True)
    pal = {'lasso': '#e74c3c', 'ridge': '#2980b9', 'ols': '#27ae60'}

    def best_slice(agg):
        """Pick a representative (n, p, rho) slice for single-panel plots."""
        for n_, p_ in [(500, 200), (300, 100), (1000, 500)]:
            sub = agg[(agg.n == n_) & (agg.p == p_) & (agg.rho == 0.5)]
            if len(sub):
                return sub
        return agg

    sub    = best_slice(agg)
    s_vals = sorted(agg.s_ratio.unique())
    ncols  = len(s_vals)

    # Coverage vs signal strength.
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 5), sharey=True)
    if ncols == 1: axes = [axes]
    fig.suptitle('CI Coverage vs Signal Strength (R^2)', fontsize=13)
    for ax, s in zip(axes, s_vals):
        for method, grp in sub[sub.s_ratio == s].groupby('method'):
            grp = grp.sort_values('tau')
            r2  = grp.tau.apply(tau_to_r2) * 100
            ax.plot(r2, grp.coverage * 100, marker='o',
                    label=method.upper(), color=pal.get(method, 'grey'), lw=2)
        ax.axhline(95, color='black', ls='--', lw=1, label='95% nominal')
        ax.axhspan(0, 90, alpha=0.08, color='red')
        ax.set_xscale('log')
        ax.set_ylim(40, 101)
        ax.set_xlabel('R^2 (%)')
        ax.set_title(f's/p = {s}')
        if ax is axes[0]:
            ax.set_ylabel('Coverage (%)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{save_dir}/fig1_coverage.png', dpi=150, bbox_inches='tight')
    fig.savefig(f'{save_dir}/fig1_coverage.pdf', bbox_inches='tight')
    plt.close()
    log("  Saved fig1_coverage")

    # Rate condition sqrt(n) * Delta_n vs signal strength.
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 5))
    if ncols == 1: axes = [axes]
    fig.suptitle('Rate Condition: $\\sqrt{n}\\Delta_n$ vs Signal Strength (R^2)', fontsize=13)
    for ax, s in zip(axes, s_vals):
        for method, grp in sub[sub.s_ratio == s].groupby('method'):
            grp = grp.sort_values('tau')
            r2  = grp.tau.apply(tau_to_r2) * 100
            ax.plot(r2, grp.sqrt_n_delta, marker='o',
                    label=method.upper(), color=pal.get(method, 'grey'), lw=2)
        ax.axhline(1.0, color='black', ls='--', lw=1, label='threshold = 1')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('R^2 (%)')
        ax.set_title(f's/p = {s}')
        if ax is axes[0]:
            ax.set_ylabel('$\\sqrt{n}\\Delta_n$')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    fig.savefig(f'{save_dir}/fig2_rate_condition.png', dpi=150, bbox_inches='tight')
    fig.savefig(f'{save_dir}/fig2_rate_condition.pdf', bbox_inches='tight')
    plt.close()
    log("  Saved fig2_rate_condition")

    # |Bias| heatmap.
    tau_vals = sorted(sub.tau.unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('|Bias| heatmap (rows = s/p, cols = tau)', fontsize=13)
    for ax, method in zip(axes, ['lasso', 'ridge']):
        mat = np.full((len(s_vals), len(tau_vals)), np.nan)
        for i, s in enumerate(s_vals):
            for j, t in enumerate(tau_vals):
                row = sub[(sub.method == method) & (sub.s_ratio == s) & (sub.tau == t)]
                if len(row):
                    mat[i, j] = abs(row.bias.values[0])
        im = ax.imshow(mat, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=0.3)
        ax.set_xticks(range(len(tau_vals)))
        ax.set_xticklabels([str(t) for t in tau_vals], rotation=45, fontsize=9)
        ax.set_yticks(range(len(s_vals)))
        ax.set_yticklabels([str(s) for s in s_vals], fontsize=9)
        ax.set_xlabel('tau')
        ax.set_ylabel('s/p')
        ax.set_title(f'DML-{method.upper()}')
        plt.colorbar(im, ax=ax, label='|Bias|')
    plt.tight_layout()
    fig.savefig(f'{save_dir}/fig3_bias_heatmap.png', dpi=150, bbox_inches='tight')
    fig.savefig(f'{save_dir}/fig3_bias_heatmap.pdf', bbox_inches='tight')
    plt.close()
    log("  Saved fig3_bias_heatmap")

    # First-stage R^2 of nuisance regressions, dense regime only.
    s_max     = max(s_vals)
    dense_sub = sub[sub.s_ratio == s_max]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Out-of-fold R^2 of nuisance regressions (s/p = 1.0)', fontsize=13)
    for ax, col, title in zip(
        axes,
        ['ell_r2', 'm_r2'],
        ['Outcome nuisance $\\ell_0(X)$', 'Treatment nuisance $m_0(X)$']
    ):
        for method, grp in dense_sub.groupby('method'):
            grp = grp.sort_values('tau')
            r2  = grp.tau.apply(tau_to_r2) * 100
            ax.plot(r2, grp[col], marker='o',
                    label=method.upper(), color=pal.get(method, 'grey'), lw=2)
        ax.axhline(0, color='black', ls='--', lw=1, label='Zero predictor')
        ax.set_xscale('log')
        ax.set_xlabel('R^2 (%)')
        ax.set_ylabel('Out-of-fold R^2')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f'{save_dir}/fig4_nuisance_r2.png', dpi=150, bbox_inches='tight')
    fig.savefig(f'{save_dir}/fig4_nuisance_r2.pdf', bbox_inches='tight')
    plt.close()
    log("  Saved fig4_nuisance_r2")

    # Coverage by sample size in the weak-dense regime.
    weak_dense = agg[(agg.tau <= 0.1) & (agg.s_ratio == s_max) & (agg.rho == 0.5)]
    if len(weak_dense) > 0:
        n_vals = sorted(weak_dense.n.unique())
        fig, axes = plt.subplots(1, len(n_vals), figsize=(5*len(n_vals), 5), sharey=True)
        if len(n_vals) == 1: axes = [axes]
        fig.suptitle('Coverage in weak-dense regime by sample size', fontsize=13)
        for ax, n_ in zip(axes, n_vals):
            sub_n = weak_dense[weak_dense.n == n_]
            for method, grp in sub_n.groupby('method'):
                grp = grp.sort_values('tau')
                r2  = grp.tau.apply(tau_to_r2) * 100
                ax.plot(r2, grp.coverage * 100, marker='o',
                        label=method.upper(), color=pal.get(method, 'grey'), lw=2)
            ax.axhline(95, color='black', ls='--', lw=1)
            ax.axhspan(0, 90, alpha=0.08, color='red')
            ax.set_xscale('log')
            ax.set_ylim(40, 101)
            ax.set_xlabel('R^2 (%)')
            ax.set_title(f'n = {n_}')
            if ax is axes[0]: ax.set_ylabel('Coverage (%)')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(f'{save_dir}/fig5_coverage_by_n.png', dpi=150, bbox_inches='tight')
        fig.savefig(f'{save_dir}/fig5_coverage_by_n.pdf', bbox_inches='tight')
        plt.close()
        log("  Saved fig5_coverage_by_n")

    log(f"All figures saved to ./{save_dir}/")


# --- Simulation grids --------------------------------------------------------

FULL_GRID = dict(
    n       = [300, 500, 1000],
    p       = [100, 200, 500],
    tau     = [5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01],
    s_ratio = [0.05, 0.30, 1.0],
    rho     = [0.0, 0.5, 0.8],
)

# Full 648-cell landscape scan, fast settings (~2-3 hours on 8 cores).
SCAN_GRID = dict(
    n       = [300, 500, 1000],
    p       = [100, 200, 500],
    tau     = [5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01],
    s_ratio = [0.05, 0.30, 1.0],
    rho     = [0.0, 0.5, 0.8],
)
SCAN_B, SCAN_S = 100, 3

# Focused on breakdown regime, high precision (~2-3 hours on 8 cores).
# tau in {0.01, 0.05, 0.1, 0.2} corresponds to R^2 in {1%, 5%, 9%, 17%}.
FOCUSED_GRID = dict(
    n       = [300, 500, 1000],
    p       = [100, 200, 500],
    tau     = [0.2, 0.1, 0.05, 0.01],
    s_ratio = [0.30, 1.0],
    rho     = [0.0, 0.5, 0.8],
)
FOCUSED_B, FOCUSED_S = 500, 10

# Small subset that still gives meaningful results (~30-60 minutes).
QUICK_GRID = dict(
    n       = [300, 500],
    p       = [100, 200],
    tau     = [2.0, 0.5, 0.1, 0.01],
    s_ratio = [0.05, 1.0],
    rho     = [0.5],
)
QUICK_B, QUICK_S = 50, 5

# Minimal sanity check, ~3 min on a MacBook.
DEMO_GRID = dict(
    n       = [300],
    p       = [100],
    tau     = [1.0, 0.1],
    s_ratio = [0.05, 1.0],
    rho     = [0.5],
)
DEMO_B, DEMO_S = 30, 2

# Weak-dense regime only.
TARGETED_GRID = dict(
    n       = [300, 500],
    p       = [100, 200],
    tau     = [0.05, 0.1, 0.2],
    s_ratio = [1.0],
    rho     = [0.5],
)
TARGETED_B, TARGETED_S = 200, 5


# --- Reporting ---------------------------------------------------------------

def print_key_results(df_agg):
    """Print the breakdown-regime rows in a readable table."""
    s_max      = df_agg.s_ratio.max()
    weak_dense = df_agg[(df_agg.tau <= 0.1) & (df_agg.s_ratio == s_max)]

    if len(weak_dense) == 0:
        log("\nNo weak-dense cells in results.")
        return

    log(f"\nWeak-dense regime (s/p = {s_max}, tau <= 0.1):")
    cols = ['n', 'p', 'tau', 'rho', 'method',
            'bias', 'rmse', 'coverage', 'sqrt_n_delta', 'ell_r2']
    log(
        weak_dense[cols]
        .sort_values(['n', 'p', 'tau', 'method'])
        .to_string(index=False)
    )

    log("\nCoverage gap (Lasso - Ridge):")
    pivot = weak_dense.pivot_table(
        index=['n', 'p', 'tau', 'rho'],
        columns='method',
        values='coverage'
    ).reset_index()
    if 'lasso' in pivot.columns and 'ridge' in pivot.columns:
        pivot['gap'] = pivot['lasso'] - pivot['ridge']
        log(pivot[['n', 'p', 'tau', 'rho', 'lasso', 'ridge', 'gap']]
            .sort_values('gap')
            .to_string(index=False))


# --- Main --------------------------------------------------------------------

def main(quick=False, demo=False, targeted=False, scan=False,
         focused=False, B_override=None, S_override=None):

    n_cores = os.cpu_count() or 1

    # Select grid, B, S, and methods based on the mode.
    if demo:
        grid, B, S, mode = DEMO_GRID, DEMO_B, DEMO_S, 'DEMO'
        methods = ['lasso', 'ridge']
    elif scan:
        grid, B, S, mode = SCAN_GRID, SCAN_B, SCAN_S, 'SCAN'
        methods = ['lasso', 'ridge']   # skip OLS in the large grid
    elif focused:
        grid, B, S, mode = FOCUSED_GRID, FOCUSED_B, FOCUSED_S, 'FOCUSED'
        methods = ['lasso', 'ridge']
    elif targeted:
        grid, B, S, mode = TARGETED_GRID, TARGETED_B, TARGETED_S, 'TARGETED'
        methods = ['lasso', 'ridge']
    elif quick:
        grid, B, S, mode = QUICK_GRID, QUICK_B, QUICK_S, 'QUICK'
        methods = ['lasso', 'ridge', 'ols']
    else:
        grid, B, S, mode = FULL_GRID, 500, 20, 'FULL'
        methods = ['lasso', 'ridge', 'ols']

    if B_override is not None: B = B_override
    if S_override is not None: S = S_override

    keys  = list(grid.keys())
    cells = list(product(*grid.values()))
    total = len(cells)

    log(f"DML simulation in {mode} mode.")
    log(f"Grid: {total} cells. Reps: B={B} x S={S}. Methods: {methods}.")
    log(f"CPU cores: {n_cores}. Checkpoints in checkpoints/ (safe to resume).")

    all_dfs  = []
    skipped  = 0
    t_start  = time.time()

    for idx, cell_vals in enumerate(cells):
        params = dict(zip(keys, cell_vals))

        if cell_already_done(params):
            skipped += 1
            log(f"[{idx+1}/{total}] cached: "
                f"n={params['n']} p={params['p']} "
                f"tau={params['tau']} s/p={params['s_ratio']} "
                f"rho={params['rho']}")
            continue

        log(f"\n[{idx+1}/{total}] "
            f"n={params['n']} p={params['p']} "
            f"tau={params['tau']} s/p={params['s_ratio']} "
            f"rho={params['rho']}")

        df_cell = run_cell(
            **params, B=B, S=S,
            methods=methods, n_jobs=-1
        )
        save_checkpoint(df_cell, params)
        all_dfs.append(df_cell)

        agg_cell = aggregate(df_cell)
        log(f"  {'Method':<8} {'Bias':>8} {'RMSE':>8} "
            f"{'Cover':>7} {'sqrtNDn':>9} {'R2_ell':>8}")
        for _, row in agg_cell.iterrows():
            log(f"  {row.method:<8} {row.bias:>8.4f} {row.rmse:>8.4f} "
                f"{row.coverage:>7.3f} {row.sqrt_n_delta:>9.3f} "
                f"{row.ell_r2:>8.3f}")

    if skipped > 0:
        cached = load_all_checkpoints()
        all_dfs.append(cached)

    if not all_dfs:
        log("\nNo results to aggregate. Exiting.")
        return None

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv('results_raw.csv', index=False)
    log(f"\nRaw results:     results_raw.csv ({len(df_all)} rows)")

    df_agg = aggregate(df_all)
    df_agg.to_csv('results.csv', index=False)
    log(f"Aggregated:      results.csv ({len(df_agg)} rows)")

    print_key_results(df_agg)

    if not demo:
        log("\nGenerating figures...")
        make_figures(df_agg)
    else:
        log("\nFigures skipped in demo mode.")

    elapsed = time.time() - t_start
    log(f"\nTotal runtime: {elapsed/60:.1f} minutes.")
    return df_agg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DML simulation: Ridge vs Lasso under weak signals'
    )
    parser.add_argument('--demo',     action='store_true',
                        help='4 cells x 30 reps (~3 min)')
    parser.add_argument('--scan',     action='store_true',
                        help='Full 648-cell landscape, B=100, S=3 (~2-3 hrs)')
    parser.add_argument('--focused',  action='store_true',
                        help='Breakdown regime, B=500, S=10 (~2-3 hrs)')
    parser.add_argument('--targeted', action='store_true',
                        help='Weak-dense regime, B=200, S=5 (~25 min)')
    parser.add_argument('--quick',    action='store_true',
                        help='16 cells x 50 reps (~30 min)')
    parser.add_argument('--B', type=int, default=None,
                        help='Override Monte Carlo replications per cell')
    parser.add_argument('--S', type=int, default=None,
                        help='Override cross-fitting repetitions per replication')
    args = parser.parse_args()

    main(
        quick    = args.quick,
        demo     = args.demo,
        targeted = args.targeted,
        scan     = args.scan,
        focused  = args.focused,
        B_override = args.B,
        S_override = args.S,
    )