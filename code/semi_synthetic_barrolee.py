"""
semi_synthetic_barrolee.py
==========================
Semi-synthetic simulation using Barro-Lee growth predictor matrix.

Key feature: n=90, p≈60, p/n ≈ 0.67
This fills the gap left by FRED-MD (max p/n = 0.49) and sits in
the regime where the pure simulation showed both methods struggling.

Design:
    - X is the real Barro-Lee predictor matrix (cross-sectional, no transforms needed)
    - X is FIXED across replications; only noise (v, eps) changes
    - Tests symmetric dense, asymmetric, and symmetric sparse designs
    - B=50 (reduced for speed — n=90 means each rep is fast but
      coverage SEs are ~7pp, adequate for detecting large effects)
    - S=3, K_outer=5, K_inner=3 (matches simulation.py)

Usage:
    python semi_synthetic_barrolee.py              # full run (~30-60 min)
    python semi_synthetic_barrolee.py --demo       # quick test (~2 min)

Requirements:
    pip install numpy pandas scikit-learn joblib
    Place GrowthData.csv in the same directory.
    (Get it from R: library(hdm); data(GrowthData); write.csv(GrowthData, "GrowthData.csv", row.names=FALSE))
"""

import numpy as np
import pandas as pd
import warnings
import argparse
import os
import time
import csv
from joblib import Parallel, delayed
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import logging
logging.getLogger("joblib").setLevel(logging.ERROR)


# =============================================================================
# SETTINGS
# =============================================================================
THETA_0 = 1.0
BL_PATH = 'barrolee.csv'    # also accepts 'GrowthData.csv'


# =============================================================================
# 1. BARRO-LEE LOADING
# =============================================================================

def load_barrolee(path):
    """
    Load Barro-Lee growth dataset.

    The dataset is cross-sectional (countries), not time-series.
    No transformations needed — variables are already in levels,
    ratios, or growth rates as constructed by Barro and Lee (1994).

    The first column 'Outcome' is GDP per capita growth.
    All other numeric columns are predictors.

    Steps:
        1. Read CSV
        2. Drop the outcome column (we generate synthetic Y)
        3. Drop any non-numeric columns
        4. Drop rows with NaN
        5. Standardise to mean=0, std=1
    """
    print("Loading Barro-Lee...")
    df = pd.read_csv(path)

    # Identify outcome column
    if 'Outcome' in df.columns:
        outcome_col = 'Outcome'
    elif 'outcome' in df.columns:
        outcome_col = 'outcome'
    else:
        # Assume first column is outcome
        outcome_col = df.columns[0]
        print(f"  Warning: using '{outcome_col}' as outcome column")

    # Drop outcome (we generate synthetic Y) and intercept (constant column)
    drop_cols = [outcome_col]
    if 'intercept' in df.columns:
        drop_cols.append('intercept')
    X_df = df.drop(columns=drop_cols, errors='ignore')

    # Keep only numeric columns
    X_df = X_df.select_dtypes(include=[np.number])

    # Drop rows with any NaN
    X_df = X_df.dropna()

    # Standardise
    X = X_df.values.astype(float)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds < 1e-10] = 1.0
    X = (X - means) / stds

    col_names = list(X_df.columns)
    n, p = X.shape

    print(f"  Loaded: n={n}, p={p}, p/n={p/n:.3f}")

    # Correlation summary
    corr = np.corrcoef(X.T)
    upper = corr[np.triu_indices_from(corr, k=1)]
    print(f"  Mean |correlation|: {np.mean(np.abs(upper)):.3f}")
    print(f"  Fraction |corr| > 0.5: {np.mean(np.abs(upper) > 0.5):.3f}")
    print(f"  Fraction |corr| > 0.8: {np.mean(np.abs(upper) > 0.8):.3f}")

    return X, col_names


# =============================================================================
# 2. COEFFICIENT CONSTRUCTION
# =============================================================================

def make_beta(p, tau, sp):
    """beta_j = sqrt(tau/s) for j <= s, 0 otherwise. ||beta||^2 = tau."""
    s = max(1, int(round(sp * p)))
    beta = np.zeros(p)
    beta[:s] = np.sqrt(tau / s)
    return beta


# =============================================================================
# 3. DATA GENERATION
# =============================================================================

def generate_data(X, beta_m, beta_g, rng):
    """Generate D and Y using fixed real X. Only noise varies."""
    n = X.shape[0]
    m0 = X @ beta_m
    g0 = X @ beta_g
    ell0 = THETA_0 * m0 + g0
    v = rng.standard_normal(n)
    eps = rng.standard_normal(n)
    D = m0 + v
    Y = THETA_0 * D + g0 + eps
    return D, Y, m0, ell0


# =============================================================================
# 4. NUISANCE FITTING (matches simulation.py)
# =============================================================================

def fit_nuisance(X_tr, y_tr, method, K_inner=3):
    if method == 'lasso':
        return LassoCV(
            alphas=np.logspace(-4, 1, 15),
            cv=K_inner,
            max_iter=3000,
            n_jobs=1,
            random_state=0,
        ).fit(X_tr, y_tr)
    else:
        return RidgeCV(
            alphas=np.logspace(-4, 6, 60),
            cv=K_inner,
            scoring='neg_mean_squared_error',
        ).fit(X_tr, y_tr)


# =============================================================================
# 5. DML ESTIMATOR (matches simulation.py)
# =============================================================================

def dml_one_split(X, D, Y, m0_true, ell0_true, method,
                  K_outer=5, K_inner=3, seed=0):
    n = X.shape[0]
    ell_resid = np.zeros(n)
    v_resid = np.zeros(n)
    ell_fitted = np.zeros(n)
    m_fitted = np.zeros(n)

    kf = KFold(n_splits=K_outer, shuffle=True, random_state=seed)

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        D_tr, D_te = D[train_idx], D[test_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        ell_mod = fit_nuisance(X_tr_s, Y_tr, method, K_inner)
        m_mod = fit_nuisance(X_tr_s, D_tr, method, K_inner)

        ell_hat = ell_mod.predict(X_te_s)
        m_hat = m_mod.predict(X_te_s)

        ell_resid[test_idx] = Y_te - ell_hat
        v_resid[test_idx] = D_te - m_hat
        ell_fitted[test_idx] = ell_hat
        m_fitted[test_idx] = m_hat

    # DML estimator
    denom = np.sum(v_resid ** 2)
    theta_hat = np.sum(v_resid * ell_resid) / denom

    # Sandwich SE
    eps_hat = ell_resid - theta_hat * v_resid
    meat = np.mean(v_resid ** 2 * eps_hat ** 2)
    bread = np.mean(v_resid ** 2)
    se_hat = np.sqrt(meat / (bread ** 2 * n))

    # Nuisance error product (true)
    ell_L2 = np.sqrt(np.mean((ell_fitted - ell0_true) ** 2))
    m_L2 = np.sqrt(np.mean((m_fitted - m0_true) ** 2))
    sqrtNDn = np.sqrt(n) * ell_L2 * m_L2

    # Out-of-fold R2
    ss_ell = np.sum((ell0_true - ell0_true.mean()) ** 2) + 1e-12
    ss_m = np.sum((m0_true - m0_true.mean()) ** 2) + 1e-12
    R2_ell = 1 - np.sum((ell_fitted - ell0_true) ** 2) / ss_ell
    R2_m = 1 - np.sum((m_fitted - m0_true) ** 2) / ss_m

    return theta_hat, se_hat, sqrtNDn, R2_ell, R2_m


def dml_estimate(X, D, Y, m0_true, ell0_true, method,
                 K_outer=5, K_inner=3, S=3, seed_base=0):
    thetas, ses, sqrtNDns, R2_ells, R2_ms = [], [], [], [], []

    for s in range(S):
        th, se, snd, r2e, r2m = dml_one_split(
            X, D, Y, m0_true, ell0_true, method,
            K_outer=K_outer, K_inner=K_inner,
            seed=seed_base * S + s
        )
        thetas.append(th)
        ses.append(se)
        sqrtNDns.append(snd)
        R2_ells.append(r2e)
        R2_ms.append(r2m)

    theta_final = np.mean(thetas)
    se_final = np.sqrt(
        np.mean(np.array(ses) ** 2 + (np.array(thetas) - theta_final) ** 2)
    )

    return {
        'theta': theta_final,
        'se': se_final,
        'sqrtNDn': np.mean(sqrtNDns),
        'R2_ell': np.mean(R2_ells),
        'R2_m': np.mean(R2_ms),
    }


# =============================================================================
# 6. ONE REPLICATION
# =============================================================================

def run_one_rep(b, X, beta_m, beta_g, m0_true, ell0_true,
                K_outer, K_inner, S):
    rng = np.random.default_rng(2026 * 300 + b)
    D, Y, _, _ = generate_data(X, beta_m, beta_g, rng)

    results = {}
    for method in ['lasso', 'ridge']:
        res = dml_estimate(
            X, D, Y, m0_true, ell0_true, method,
            K_outer=K_outer, K_inner=K_inner, S=S,
            seed_base=b
        )
        ci_lo = res['theta'] - 1.96 * res['se']
        ci_hi = res['theta'] + 1.96 * res['se']
        res['covers'] = float(ci_lo <= THETA_0 <= ci_hi)
        results[method] = res

    return results


# =============================================================================
# 7. GRID
# =============================================================================

def build_grid():
    """
    Barro-Lee grid. Focused on the key comparisons.
    p ≈ 60, n = 90, p/n ≈ 0.67 — above the 0.50 threshold.
    """
    grid = []

    # --- Symmetric dense: the main test ---
    # At p/n=0.67 with dense signals, both methods should struggle
    # but Ridge should do better at weak signals
    for tau in [0.5, 0.2, 0.1, 0.05, 0.01]:
        grid.append({
            'label': 'symmetric_dense',
            'tau_m': tau, 'sp_m': 1.0,
            'tau_g': tau, 'sp_g': 1.0,
        })

    # --- Asymmetric: sparse treatment + dense weak outcome ---
    for tau_g in [0.2, 0.1, 0.05]:
        grid.append({
            'label': 'asym_sparse_m__dense_g',
            'tau_m': 2.0, 'sp_m': 0.05,
            'tau_g': tau_g, 'sp_g': 1.0,
        })

    # --- Asymmetric reverse: dense treatment + sparse outcome ---
    for tau_m in [0.1, 0.05]:
        grid.append({
            'label': 'asym_dense_m__sparse_g',
            'tau_m': tau_m, 'sp_m': 1.0,
            'tau_g': 2.0, 'sp_g': 0.05,
        })

    # --- Symmetric sparse baseline ---
    grid.append({
        'label': 'symmetric_sparse',
        'tau_m': 2.0, 'sp_m': 0.05,
        'tau_g': 2.0, 'sp_g': 0.05,
    })

    return grid


def build_demo_grid():
    return [
        {'label': 'symmetric_dense',
         'tau_m': 0.1, 'sp_m': 1.0, 'tau_g': 0.1, 'sp_g': 1.0},
        {'label': 'symmetric_sparse',
         'tau_m': 2.0, 'sp_m': 0.05, 'tau_g': 2.0, 'sp_g': 0.05},
    ]


# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Semi-synthetic DML simulation with Barro-Lee'
    )
    parser.add_argument('--demo', action='store_true',
                        help='Quick test: 2 cells, B=10, S=1')
    parser.add_argument('--B', type=int, default=None,
                        help='Override replications (default: 50)')
    parser.add_argument('--S', type=int, default=None,
                        help='Override cross-fitting reps (default: 3)')
    args = parser.parse_args()

    if args.demo:
        B, S, K_outer, K_inner = 10, 1, 5, 3
        grid = build_demo_grid()
        mode = 'DEMO'
    else:
        B, S, K_outer, K_inner = 50, 3, 5, 3
        grid = build_grid()
        mode = 'FULL'

    if args.B is not None:
        B = args.B
    if args.S is not None:
        S = args.S

    # Load data
    if not os.path.exists(BL_PATH):
        print(f"ERROR: {BL_PATH} not found.")
        print("Get it from R: library(hdm); data(GrowthData); "
              "write.csv(GrowthData, 'GrowthData.csv', row.names=FALSE)")
        return

    X, col_names = load_barrolee(BL_PATH)
    n, p = X.shape
    total = len(grid)

    print()
    print("=" * 70)
    print("  Semi-Synthetic DML Simulation (Barro-Lee)")
    print(f"  Mode      : {mode}")
    print(f"  Cells     : {total}")
    print(f"  Reps      : B={B}  x  S={S}  x  K_outer={K_outer}")
    print(f"  Barro-Lee : n={n}, p={p}, p/n={p/n:.3f}")
    print(f"  CPU cores : {os.cpu_count()}")
    print("=" * 70)
    print()

    # Output files
    logfile = open('barrolee_semi_synthetic_results.log', 'w')
    csvfile = open('barrolee_semi_synthetic_results.csv', 'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow([
        'label', 'n', 'p', 'p_over_n',
        'tau_m', 'sp_m', 'tau_g', 'sp_g',
        'method',
        'bias', 'rmse', 'cover',
        'sqrtNDn', 'R2_ell', 'R2_m'
    ])

    logfile.write("=" * 70 + '\n')
    logfile.write("  Semi-Synthetic DML Simulation (Barro-Lee)\n")
    logfile.write(f"  Cells: {total}  |  B={B}  |  S={S}  |  K={K_outer}\n")
    logfile.write(f"  Barro-Lee: n={n}, p={p}, p/n={p/n:.3f}\n")
    logfile.write("=" * 70 + '\n\n')

    t_total = time.time()

    for idx, cell in enumerate(grid):
        t0 = time.time()

        beta_m = make_beta(p, cell['tau_m'], cell['sp_m'])
        beta_g = make_beta(p, cell['tau_g'], cell['sp_g'])
        m0_true = X @ beta_m
        ell0_true = THETA_0 * m0_true + X @ beta_g

        header = (
            f"[{idx+1}/{total}]  n={n}  p={p}  p/n={p/n:.3f}  "
            f"tau_m={cell['tau_m']}  sp_m={cell['sp_m']}  "
            f"tau_g={cell['tau_g']}  sp_g={cell['sp_g']}  "
            f"[{cell['label']}]"
        )
        print(header)
        logfile.write(header + '\n')

        # Run B replications
        all_results = Parallel(n_jobs=-1)(
            delayed(run_one_rep)(
                b, X, beta_m, beta_g, m0_true, ell0_true,
                K_outer, K_inner, S
            )
            for b in range(B)
        )

        # Aggregate
        print(f"  {'Method':<10} {'Bias':>8} {'RMSE':>8} {'Cover':>8} "
              f"{'sqrtNDn':>9} {'R2_ell':>8} {'R2_m':>8}")
        print(f"  {'-'*62}")
        logfile.write(f"  {'Method':<10} {'Bias':>8} {'RMSE':>8} {'Cover':>8} "
                      f"{'sqrtNDn':>9} {'R2_ell':>8} {'R2_m':>8}\n")
        logfile.write(f"  {'-'*62}\n")

        for method in ['lasso', 'ridge']:
            thetas  = [r[method]['theta']   for r in all_results]
            covers  = [r[method]['covers']  for r in all_results]
            sqrtNDn = [r[method]['sqrtNDn'] for r in all_results]
            R2_ells = [r[method]['R2_ell']  for r in all_results]
            R2_ms   = [r[method]['R2_m']    for r in all_results]

            bias = np.mean(thetas) - THETA_0
            rmse = np.sqrt(np.mean((np.array(thetas) - THETA_0) ** 2))
            cover = np.mean(covers)
            sqrtNDn_avg = np.mean(sqrtNDn)
            R2_ell_avg = np.mean(R2_ells)
            R2_m_avg = np.mean(R2_ms)

            line = (
                f"  {method:<10} {bias:>8.4f} {rmse:>8.4f} "
                f"{cover:>8.3f} {sqrtNDn_avg:>9.3f} "
                f"{R2_ell_avg:>8.3f} {R2_m_avg:>8.3f}"
            )
            print(line)
            logfile.write(line + '\n')

            writer.writerow([
                cell['label'], n, p, f"{p/n:.4f}",
                cell['tau_m'], cell['sp_m'],
                cell['tau_g'], cell['sp_g'],
                method,
                f"{bias:.6f}", f"{rmse:.6f}", f"{cover:.3f}",
                f"{sqrtNDn_avg:.4f}", f"{R2_ell_avg:.4f}", f"{R2_m_avg:.4f}"
            ])

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s\n")
        logfile.write(f"  Done in {elapsed:.1f}s\n\n")
        logfile.flush()
        csvfile.flush()

    total_time = time.time() - t_total
    summary = (
        f"\n{'='*70}\n"
        f"  COMPLETE\n"
        f"  Total cells: {total}\n"
        f"  Total time:  {total_time:.0f}s ({total_time/60:.1f} min)\n"
        f"{'='*70}\n"
    )
    print(summary)
    logfile.write(summary)
    logfile.close()
    csvfile.close()

    print("Output files:")
    print("  barrolee_semi_synthetic_results.log")
    print("  barrolee_semi_synthetic_results.csv")


if __name__ == '__main__':
    main()
