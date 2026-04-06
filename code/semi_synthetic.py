"""
semi_synthetic.py
=================
Semi-synthetic simulation using FRED-MD predictor matrix.

What this does:
    - Loads real FRED-MD predictors (X is real economic data)
    - Generates synthetic treatment D and outcome Y using those real predictors
    - X is FIXED across replications — only noise (v, epsilon) changes
    - Tests symmetric (beta_m = beta_g) and asymmetric (beta_m ≠ beta_g) designs
    - Runs at three sample sizes: full (n≈757), n=400, n=250
      to vary p/n from 0.16 to 0.31 to 0.49

Matches simulation.py:
    K_outer=5 cross-fitting folds
    K_inner=3 inner CV for lambda
    S=3 cross-fitting repetitions
    B=100 Monte Carlo replications
    Sandwich SE, theta_0=1

Usage:
    python semi_synthetic.py                # full run
    python semi_synthetic.py --demo         # quick test (B=10, S=1)
    python semi_synthetic.py --B 200 --S 5  # custom B and S
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
FRED_PATH = 'fred_md.csv'   # place in same directory


# =============================================================================
# 1. FRED-MD LOADING AND PREPROCESSING
# =============================================================================

def fred_transform(x, tcode):
    """
    Apply McCracken-Ng (2016) transformation codes to induce stationarity.
    1: level (no transform)
    2: first difference
    3: second difference
    4: log
    5: log first difference (growth rate)
    6: log second difference
    7: delta(x_t/x_{t-1} - 1)
    """
    x = x.copy().astype(float)
    if tcode == 1:
        return x
    elif tcode == 2:
        return x.diff()
    elif tcode == 3:
        return x.diff().diff()
    elif tcode == 4:
        return np.log(x)
    elif tcode == 5:
        return np.log(x).diff()
    elif tcode == 6:
        return np.log(x).diff().diff()
    elif tcode == 7:
        return (x / x.shift(1) - 1).diff()
    else:
        return x


def load_fred_md(path):
    """
    Load FRED-MD and preprocess following McCracken & Ng (2016).

    Steps:
        1. Read CSV — row 0 has transformation codes, row 1 onward is data
        2. Apply transformation codes to each series
        3. Drop columns with >10% missing values
        4. Drop rows with any remaining NaN (from differencing)
        5. Standardise each column to mean=0, std=1

    Returns:
        X : ndarray of shape (n, p), standardised predictors
        col_names : list of column names
    """
    print("Loading FRED-MD...")
    df = pd.read_csv(path)

    # Extract transformation codes from row 0
    tcodes = {}
    for col in df.columns[1:]:
        try:
            tcodes[col] = int(float(df.iloc[0][col]))
        except (ValueError, TypeError):
            tcodes[col] = 1  # default: no transform

    # Drop transform row, rename date column
    df = df.iloc[1:].copy().reset_index(drop=True)
    df = df.rename(columns={df.columns[0]: 'date'})

    # Convert everything to numeric
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Apply McCracken-Ng transformations
    for col in df.columns[1:]:
        if col in tcodes:
            df[col] = fred_transform(df[col], tcodes[col])

    # Keep only predictor columns
    X_df = df.iloc[:, 1:].copy()

    # Drop columns with >10% missing
    missing_frac = X_df.isnull().sum() / len(X_df)
    keep_cols = missing_frac[missing_frac <= 0.10].index
    X_df = X_df[keep_cols]

    # Drop rows with any remaining NaN
    X_df = X_df.dropna()

    # Standardise to mean=0, std=1
    X = X_df.values.astype(float)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds < 1e-10] = 1.0  # avoid division by zero for constant columns
    X = (X - means) / stds

    col_names = list(X_df.columns)
    print(f"  Loaded: n={X.shape[0]}, p={X.shape[1]}, p/n={X.shape[1]/X.shape[0]:.3f}")

    # Print correlation summary
    corr = np.corrcoef(X.T)
    upper = corr[np.triu_indices_from(corr, k=1)]
    print(f"  Mean |correlation|: {np.mean(np.abs(upper)):.3f}")
    print(f"  Fraction |corr| > 0.5: {np.mean(np.abs(upper) > 0.5):.3f}")

    return X, col_names


# =============================================================================
# 2. COEFFICIENT CONSTRUCTION
# =============================================================================

def make_beta(p, tau, sp):
    """
    Construct coefficient vector.
    beta_j = sqrt(tau / s) for j = 1, ..., s
    beta_j = 0             for j = s+1, ..., p
    where s = round(sp * p).

    This gives ||beta||^2 = s * (tau/s) = tau.
    """
    s = max(1, int(round(sp * p)))
    beta = np.zeros(p)
    beta[:s] = np.sqrt(tau / s)
    return beta


# =============================================================================
# 3. SEMI-SYNTHETIC DATA GENERATION
# =============================================================================

def generate_data(X, beta_m, beta_g, rng):
    """
    Generate treatment D and outcome Y using fixed real predictors X.

    Model:
        D = X @ beta_m + v,           v ~ N(0,1)
        Y = theta_0 * D + X @ beta_g + eps,  eps ~ N(0,1)

    X is fixed. Only v and eps are random.
    This means m_0(X) = X @ beta_m and ell_0(X) = theta_0 * m_0(X) + X @ beta_g.
    """
    n = X.shape[0]

    m0 = X @ beta_m                    # E[D | X]
    g0 = X @ beta_g                    # g_0(X)
    ell0 = THETA_0 * m0 + g0           # E[Y | X]

    v = rng.standard_normal(n)
    eps = rng.standard_normal(n)

    D = m0 + v
    Y = THETA_0 * D + g0 + eps

    return D, Y, m0, ell0


# =============================================================================
# 4. NUISANCE FITTING (matches simulation.py exactly)
# =============================================================================

def fit_nuisance(X_tr, y_tr, method, K_inner=3):
    """
    Fit one nuisance function.
    Lasso: 15-point alpha grid, max_iter=3000, inner CV = K_inner folds.
    Ridge: 60-point alpha grid, inner CV = K_inner folds.
    """
    if method == 'lasso':
        model = LassoCV(
            alphas=np.logspace(-4, 1, 15),
            cv=K_inner,
            max_iter=3000,
            n_jobs=1,
            random_state=0,
        ).fit(X_tr, y_tr)

    elif method == 'ridge':
        model = RidgeCV(
            alphas=np.logspace(-4, 6, 60),
            cv=K_inner,
            scoring='neg_mean_squared_error',
        ).fit(X_tr, y_tr)

    return model


# =============================================================================
# 5. DML ESTIMATOR (matches simulation.py exactly)
# =============================================================================

def dml_one_split(X, D, Y, m0_true, ell0_true, method,
                  K_outer=5, K_inner=3, seed=0):
    """
    One cross-fitting pass.

    Steps:
        1. Split data into K_outer folds
        2. For each fold: train nuisance models on K-1 folds, predict on held-out
        3. Compute theta_hat via partialling-out
        4. Compute sandwich SE
        5. Compute sqrt(n)*Delta_n using true nuisance functions

    Returns:
        theta_hat, se_hat, sqrtNDn, R2_ell, R2_m
    """
    n = X.shape[0]

    ell_resid = np.zeros(n)    # Y - ell_hat(X)
    v_resid = np.zeros(n)      # D - m_hat(X)
    ell_fitted = np.zeros(n)   # ell_hat(X)
    m_fitted = np.zeros(n)     # m_hat(X)

    kf = KFold(n_splits=K_outer, shuffle=True, random_state=seed)

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        D_tr, D_te = D[train_idx], D[test_idx]

        # Standardise within fold (same as simulation.py)
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Fit nuisance models
        ell_mod = fit_nuisance(X_tr_s, Y_tr, method, K_inner)
        m_mod = fit_nuisance(X_tr_s, D_tr, method, K_inner)

        # Predict on test fold
        ell_hat = ell_mod.predict(X_te_s)
        m_hat = m_mod.predict(X_te_s)

        # Store
        ell_resid[test_idx] = Y_te - ell_hat
        v_resid[test_idx] = D_te - m_hat
        ell_fitted[test_idx] = ell_hat
        m_fitted[test_idx] = m_hat

    # DML estimator: theta = sum(v_resid * ell_resid) / sum(v_resid^2)
    denom = np.sum(v_resid ** 2)
    theta_hat = np.sum(v_resid * ell_resid) / denom

    # Sandwich SE (heteroskedasticity-robust)
    eps_hat = ell_resid - theta_hat * v_resid
    meat = np.mean(v_resid ** 2 * eps_hat ** 2)
    bread = np.mean(v_resid ** 2)
    se_hat = np.sqrt(meat / (bread ** 2 * n))

    # Nuisance error product: Delta_n = ||ell_hat - ell_0||_2 * ||m_hat - m_0||_2
    ell_L2 = np.sqrt(np.mean((ell_fitted - ell0_true) ** 2))
    m_L2 = np.sqrt(np.mean((m_fitted - m0_true) ** 2))
    sqrtNDn = np.sqrt(n) * ell_L2 * m_L2

    # Out-of-fold R2 for each nuisance regression
    ss_ell = np.sum((ell0_true - ell0_true.mean()) ** 2) + 1e-12
    ss_m = np.sum((m0_true - m0_true.mean()) ** 2) + 1e-12
    R2_ell = 1 - np.sum((ell_fitted - ell0_true) ** 2) / ss_ell
    R2_m = 1 - np.sum((m_fitted - m0_true) ** 2) / ss_m

    return theta_hat, se_hat, sqrtNDn, R2_ell, R2_m


def dml_estimate(X, D, Y, m0_true, ell0_true, method,
                 K_outer=5, K_inner=3, S=3, seed_base=0):
    """
    Run S cross-fitting repetitions and aggregate.
    Averaging over S reduces sensitivity to the random fold assignment.
    Aggregation follows Chernozhukov et al. (2018).
    """
    thetas = []
    ses = []
    sqrtNDns = []
    R2_ells = []
    R2_ms = []

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

    # Aggregate: mean theta, adjusted SE (accounts for cross-split variance)
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
# 6. ONE REPLICATION (called in parallel)
# =============================================================================

def run_one_rep(b, X, beta_m, beta_g, m0_true, ell0_true,
                K_outer, K_inner, S):
    """
    One Monte Carlo replication.
    X, beta_m, beta_g, m0_true, ell0_true are FIXED.
    Only noise (v, eps) changes via the random seed.
    """
    rng = np.random.default_rng(2026 * 100 + b)

    # Generate fresh noise
    D, Y, _, _ = generate_data(X, beta_m, beta_g, rng)
    # m0_true and ell0_true don't change — they depend only on X and beta

    results = {}
    for method in ['lasso', 'ridge']:
        res = dml_estimate(
            X, D, Y, m0_true, ell0_true, method,
            K_outer=K_outer, K_inner=K_inner, S=S,
            seed_base=b
        )
        # Coverage check
        ci_lo = res['theta'] - 1.96 * res['se']
        ci_hi = res['theta'] + 1.96 * res['se']
        res['covers'] = float(ci_lo <= THETA_0 <= ci_hi)
        results[method] = res

    return results


# =============================================================================
# 7. GRID DEFINITION
# =============================================================================

def build_grid():
    """
    Build the semi-synthetic grid.

    Three sample sizes (via subsampling):
        n = full (~757)  ->  p/n ≈ 0.16
        n = 400          ->  p/n ≈ 0.31
        n = 250          ->  p/n ≈ 0.49

    Coefficient designs:
        SYMMETRIC DENSE WEAK:
            beta_m = beta_g, both dense (s/p=1.0), varying tau

        ASYMMETRIC — sparse treatment, dense weak outcome:
            beta_m: tau=2.0, s/p=0.05 (sparse strong)
            beta_g: varying tau, s/p=1.0 (dense weak)
            This is the realistic case: treatment depends on few variables,
            outcome depends on everything weakly.

        ASYMMETRIC REVERSE — dense weak treatment, sparse outcome:
            beta_m: varying tau, s/p=1.0 (dense weak)
            beta_g: tau=2.0, s/p=0.05 (sparse strong)

        SYMMETRIC SPARSE STRONG (baseline):
            beta_m = beta_g, both sparse (s/p=0.05), tau=2.0
    """
    grid = []

    sample_sizes = ['full', 400, 250]

    # --- Symmetric dense weak ---
    for n_sub in sample_sizes:
        for tau in [0.2, 0.1, 0.05, 0.01]:
            grid.append({
                'label': 'symmetric_dense',
                'n_sub': n_sub,
                'tau_m': tau, 'sp_m': 1.0,
                'tau_g': tau, 'sp_g': 1.0,
            })

    # --- Asymmetric: sparse strong treatment + dense weak outcome ---
    for n_sub in sample_sizes:
        for tau_g in [0.2, 0.1, 0.05, 0.01]:
            grid.append({
                'label': 'asym_sparse_m__dense_g',
                'n_sub': n_sub,
                'tau_m': 2.0, 'sp_m': 0.05,
                'tau_g': tau_g, 'sp_g': 1.0,
            })

    # --- Asymmetric reverse: dense weak treatment + sparse strong outcome ---
    for n_sub in sample_sizes:
        for tau_m in [0.1, 0.05]:
            grid.append({
                'label': 'asym_dense_m__sparse_g',
                'n_sub': n_sub,
                'tau_m': tau_m, 'sp_m': 1.0,
                'tau_g': 2.0, 'sp_g': 0.05,
            })

    # --- Symmetric sparse strong (baseline) ---
    for n_sub in sample_sizes:
        grid.append({
            'label': 'symmetric_sparse',
            'n_sub': n_sub,
            'tau_m': 2.0, 'sp_m': 0.05,
            'tau_g': 2.0, 'sp_g': 0.05,
        })

    return grid


def build_demo_grid():
    """Small grid for testing."""
    return [
        {'label': 'symmetric_dense',         'n_sub': 'full',
         'tau_m': 0.1, 'sp_m': 1.0, 'tau_g': 0.1, 'sp_g': 1.0},
        {'label': 'asym_sparse_m__dense_g',  'n_sub': 'full',
         'tau_m': 2.0, 'sp_m': 0.05, 'tau_g': 0.1, 'sp_g': 1.0},
        {'label': 'symmetric_dense',         'n_sub': 250,
         'tau_m': 0.1, 'sp_m': 1.0, 'tau_g': 0.1, 'sp_g': 1.0},
        {'label': 'asym_sparse_m__dense_g',  'n_sub': 250,
         'tau_m': 2.0, 'sp_m': 0.05, 'tau_g': 0.1, 'sp_g': 1.0},
    ]


# =============================================================================
# 8. MAIN SIMULATION LOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Semi-synthetic DML simulation with FRED-MD'
    )
    parser.add_argument('--demo', action='store_true',
                        help='Quick test: 4 cells, B=10, S=1')
    parser.add_argument('--B', type=int, default=None,
                        help='Override Monte Carlo replications (default: 100)')
    parser.add_argument('--S', type=int, default=None,
                        help='Override cross-fitting repetitions (default: 3)')
    args = parser.parse_args()

    # Settings
    if args.demo:
        B = 10
        S = 1
        K_outer = 5
        K_inner = 3
        grid = build_demo_grid()
        mode = 'DEMO'
    else:
        B = 100
        S = 3
        K_outer = 5
        K_inner = 3
        grid = build_grid()
        mode = 'FULL'

    # Command-line overrides
    if args.B is not None:
        B = args.B
    if args.S is not None:
        S = args.S

    # ------------------------------------------------------------------
    # Load FRED-MD
    # ------------------------------------------------------------------
    if not os.path.exists(FRED_PATH):
        print(f"ERROR: {FRED_PATH} not found.")
        print(f"Place fred_md.csv in the same directory as this script.")
        return

    X_full, col_names = load_fred_md(FRED_PATH)
    n_full, p = X_full.shape

    # ------------------------------------------------------------------
    # Prepare subsampled X matrices (fixed random seed for reproducibility)
    # ------------------------------------------------------------------
    rng_sub = np.random.default_rng(42)
    all_indices = np.arange(n_full)

    # Pre-draw subsample indices
    idx_400 = rng_sub.choice(all_indices, size=400, replace=False)
    idx_250 = rng_sub.choice(all_indices, size=250, replace=False)

    X_sets = {
        'full': X_full,
        400: X_full[idx_400],
        250: X_full[idx_250],
    }

    # Re-standardise subsampled X (important — subsample may have different mean/std)
    for key in [400, 250]:
        X_sub = X_sets[key]
        means = X_sub.mean(axis=0)
        stds = X_sub.std(axis=0)
        stds[stds < 1e-10] = 1.0
        X_sets[key] = (X_sub - means) / stds

    total = len(grid)

    # ------------------------------------------------------------------
    # Print header
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  Semi-Synthetic DML Simulation (FRED-MD)")
    print(f"  Mode      : {mode}")
    print(f"  Cells     : {total}")
    print(f"  Reps      : B={B}  x  S={S}  x  K_outer={K_outer}")
    print(f"  FRED-MD   : n_full={n_full}, p={p}")
    print(f"  Subsamples: n=400 (p/n={p/400:.3f}), n=250 (p/n={p/250:.3f})")
    print(f"  CPU cores : {os.cpu_count()}")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # Open output files
    # ------------------------------------------------------------------
    logfile = open('semi_synthetic_results.log', 'w')
    csvfile = open('semi_synthetic_results.csv', 'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow([
        'label', 'n', 'p', 'p_over_n',
        'tau_m', 'sp_m', 'tau_g', 'sp_g',
        'method',
        'bias', 'rmse', 'cover',
        'sqrtNDn', 'R2_ell', 'R2_m'
    ])

    logfile.write("=" * 70 + '\n')
    logfile.write("  Semi-Synthetic DML Simulation (FRED-MD)\n")
    logfile.write(f"  Cells: {total}  |  B={B}  |  S={S}  |  K={K_outer}\n")
    logfile.write(f"  FRED-MD: n_full={n_full}, p={p}\n")
    logfile.write("=" * 70 + '\n\n')

    t_total = time.time()

    # ------------------------------------------------------------------
    # Run each cell
    # ------------------------------------------------------------------
    for idx, cell in enumerate(grid):
        t0 = time.time()

        # Get the right X matrix
        n_sub_key = cell['n_sub']
        X = X_sets[n_sub_key]
        n = X.shape[0]
        p_n = p / n

        # Construct coefficient vectors
        beta_m = make_beta(p, cell['tau_m'], cell['sp_m'])
        beta_g = make_beta(p, cell['tau_g'], cell['sp_g'])

        # Compute true nuisance functions (fixed because X is fixed)
        m0_true = X @ beta_m
        ell0_true = THETA_0 * m0_true + X @ beta_g

        # Print header
        n_label = f"n={n}" if n_sub_key == 'full' else f"n={n} (subsample)"
        header = (
            f"[{idx+1}/{total}]  {n_label}  p={p}  p/n={p_n:.3f}  "
            f"tau_m={cell['tau_m']}  sp_m={cell['sp_m']}  "
            f"tau_g={cell['tau_g']}  sp_g={cell['sp_g']}  "
            f"[{cell['label']}]"
        )
        print(header)
        logfile.write(header + '\n')

        # ----------------------------------------------------------
        # Run B replications in parallel
        # ----------------------------------------------------------
        all_results = Parallel(n_jobs=-1)(
            delayed(run_one_rep)(
                b, X, beta_m, beta_g, m0_true, ell0_true,
                K_outer, K_inner, S
            )
            for b in range(B)
        )

        # ----------------------------------------------------------
        # Aggregate across replications
        # ----------------------------------------------------------
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

            # Write to CSV
            writer.writerow([
                cell['label'], n, p, f"{p_n:.4f}",
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

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_time = time.time() - t_total
    hours = total_time / 3600

    summary = (
        f"\n{'='*70}\n"
        f"  COMPLETE\n"
        f"  Total cells: {total}\n"
        f"  Total time:  {total_time:.0f}s ({hours:.1f} hours)\n"
        f"{'='*70}\n"
    )
    print(summary)
    logfile.write(summary)

    logfile.close()
    csvfile.close()

    print("Output files:")
    print("  semi_synthetic_results.log   (human-readable)")
    print("  semi_synthetic_results.csv   (machine-readable)")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    main()
