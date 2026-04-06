"""
============================================================
  Asymmetric DML Simulation + Semi-Synthetic (FRED-MD)
  "When Does Orthogonality Fail to Protect?"
============================================================

Two parts:
  Part A — Pure simulation with asymmetric coefficients
            (beta_m ≠ beta_g in sparsity and signal strength)
  Part B — Semi-synthetic using FRED-MD predictor matrix

Matches the existing setup:
  K=5 outer folds, S=3 cross-fitting, B=100 reps
  Sandwich SE, theta_0=1

Usage:
  python asymmetric_dml.py              # runs everything
  python asymmetric_dml.py --part A     # pure simulation only
  python asymmetric_dml.py --part B     # semi-synthetic only

Requirements:
  pip install numpy pandas scikit-learn joblib

Output:
  asymmetric_results.log   (console-style log)
  asymmetric_results.csv   (machine-readable)
"""

import numpy as np
import pandas as pd
import time
import sys
import os
import csv
from sklearn.linear_model import LassoCV, RidgeCV
from joblib import Parallel, delayed

# ============================================================
# SETTINGS — change these if needed
# ============================================================
THETA_0 = 1.0
B = 100            # Monte Carlo replications per cell
S = 3              # Cross-fitting repetitions
K = 5              # Outer folds
N_JOBS = -1        # Parallel cores (-1 = all)
SEED_BASE = 2026
FRED_PATH = 'fred_md.csv'  # path to FRED-MD file

# ============================================================
# FRED-MD PREPROCESSING (McCracken & Ng 2016)
# ============================================================

def fred_transform(x, tcode):
    """Apply McCracken-Ng transformation codes to a single series.
    1: no transform
    2: first difference
    3: second difference
    4: log
    5: log first difference (= growth rate)
    6: log second difference
    7: (x_t/x_{t-1} - 1) first difference
    """
    x = x.copy().astype(float)
    if tcode == 1:
        return x
    elif tcode == 2:
        out = x.diff()
        return out
    elif tcode == 3:
        out = x.diff().diff()
        return out
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
    """Load and preprocess FRED-MD following McCracken & Ng (2016).
    Returns: X (ndarray, standardised), column names, n, p
    """
    df = pd.read_csv(path)

    # Row 0 has transformation codes
    tcodes = {}
    for col in df.columns[1:]:
        try:
            tcodes[col] = int(float(df.iloc[0][col]))
        except:
            tcodes[col] = 1  # default: no transform

    # Drop transform row, set date
    df = df.iloc[1:].copy().reset_index(drop=True)
    df = df.rename(columns={df.columns[0]: 'date'})

    # Convert to numeric
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Apply transformations
    for col in df.columns[1:]:
        if col in tcodes:
            df[col] = fred_transform(df[col], tcodes[col])

    # Drop date column
    X_df = df.iloc[:, 1:].copy()

    # Drop columns with >10% missing
    thresh = len(X_df) * 0.1
    keep = X_df.columns[X_df.isnull().sum() <= thresh]
    X_df = X_df[keep]

    # Drop rows with any NaN (mainly from differencing)
    X_df = X_df.dropna()

    # Standardise: mean 0, std 1
    X = X_df.values.astype(float)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

    print(f"  FRED-MD loaded: n={X.shape[0]}, p={X.shape[1]}, p/n={X.shape[1]/X.shape[0]:.3f}")
    return X, list(X_df.columns)


# ============================================================
# DGP AND DML CORE
# ============================================================

def make_beta(p, tau, sp):
    """beta_j = sqrt(tau/s) for j <= s, 0 otherwise. ||beta||^2 = tau."""
    s = max(1, int(round(sp * p)))
    beta = np.zeros(p)
    beta[:s] = np.sqrt(tau / s)
    return beta


def make_toeplitz(p, rho):
    """Toeplitz covariance: Sigma_jk = rho^|j-k|."""
    if rho == 0.0:
        return np.eye(p)
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def generate_data_simulated(n, p, beta_m, beta_g, Sigma, rng):
    """Generate data from PLR model with simulated X."""
    if np.allclose(Sigma, np.eye(p)):
        X = rng.standard_normal((n, p))
    else:
        L = np.linalg.cholesky(Sigma)
        X = rng.standard_normal((n, p)) @ L.T

    m0 = X @ beta_m
    ell0 = THETA_0 * m0 + X @ beta_g
    v = rng.standard_normal(n)
    eps = rng.standard_normal(n)
    D = m0 + v
    Y = THETA_0 * D + X @ beta_g + eps

    return X, Y, D, m0, ell0


def generate_data_semisynthetic(X_real, beta_m, beta_g, rng):
    """Generate data from PLR model using real X matrix.
    X is fixed; only noise (v, eps) varies across replications.
    """
    n = X_real.shape[0]
    m0 = X_real @ beta_m
    ell0 = THETA_0 * m0 + X_real @ beta_g
    v = rng.standard_normal(n)
    eps = rng.standard_normal(n)
    D = m0 + v
    Y = THETA_0 * D + X_real @ beta_g + eps

    return X_real, Y, D, m0, ell0


def dml_one_split(X, Y, D, m0_true, ell0_true, method, K, rng):
    """One cross-fitting split. Returns dict with all metrics."""
    n = X.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, K)

    ell_resid = np.zeros(n)
    v_resid = np.zeros(n)
    ell_err_sq = np.zeros(n)
    m_err_sq = np.zeros(n)
    ell_pred = np.zeros(n)
    m_pred = np.zeros(n)

    for k in range(K):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(K) if j != k])

        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        D_tr, D_te = D[train_idx], D[test_idx]

        if method == 'lasso':
            mdl_ell = LassoCV(cv=5, max_iter=10000, n_jobs=1, random_state=42)
            mdl_ell.fit(X_tr, Y_tr)
            ell_hat = mdl_ell.predict(X_te)

            mdl_m = LassoCV(cv=5, max_iter=10000, n_jobs=1, random_state=42)
            mdl_m.fit(X_tr, D_tr)
            m_hat = mdl_m.predict(X_te)
        else:
            alphas = np.logspace(-3, 5, 50)
            mdl_ell = RidgeCV(alphas=alphas, cv=5)
            mdl_ell.fit(X_tr, Y_tr)
            ell_hat = mdl_ell.predict(X_te)

            mdl_m = RidgeCV(alphas=alphas, cv=5)
            mdl_m.fit(X_tr, D_tr)
            m_hat = mdl_m.predict(X_te)

        ell_resid[test_idx] = Y_te - ell_hat
        v_resid[test_idx] = D_te - m_hat
        ell_pred[test_idx] = ell_hat
        m_pred[test_idx] = m_hat
        ell_err_sq[test_idx] = (ell_hat - ell0_true[test_idx]) ** 2
        m_err_sq[test_idx] = (m_hat - m0_true[test_idx]) ** 2

    # DML estimator
    theta_hat = np.sum(v_resid * ell_resid) / np.sum(v_resid ** 2)

    # Sandwich SE
    eps_hat = ell_resid - theta_hat * v_resid
    J = np.mean(v_resid ** 2)
    sigma2 = np.mean((v_resid * eps_hat) ** 2)
    se_hat = np.sqrt(sigma2 / (J ** 2 * X.shape[0]))

    # Nuisance error product
    ell_L2 = np.sqrt(np.mean(ell_err_sq))
    m_L2 = np.sqrt(np.mean(m_err_sq))
    Delta_n = ell_L2 * m_L2
    sqrtN_Dn = np.sqrt(X.shape[0]) * Delta_n

    # Out-of-fold R2
    ss_res_ell = np.sum((Y - ell_pred) ** 2)
    ss_tot_ell = np.sum((Y - np.mean(Y)) ** 2)
    R2_ell = 1 - ss_res_ell / ss_tot_ell

    ss_res_m = np.sum((D - m_pred) ** 2)
    ss_tot_m = np.sum((D - np.mean(D)) ** 2)
    R2_m = 1 - ss_res_m / ss_tot_m

    return theta_hat, se_hat, sqrtN_Dn, R2_ell, R2_m


def dml_estimate(X, Y, D, m0_true, ell0_true, method, K, S, rng):
    """Run S cross-fitting repetitions, aggregate per Chernozhukov et al. (2018)."""
    thetas, ses, sqrtNDns, R2_ells, R2_ms = [], [], [], [], []

    for s in range(S):
        th, se, snd, r2e, r2m = dml_one_split(
            X, Y, D, m0_true, ell0_true, method, K, rng
        )
        thetas.append(th)
        ses.append(se)
        sqrtNDns.append(snd)
        R2_ells.append(r2e)
        R2_ms.append(r2m)

    theta_final = np.mean(thetas)
    se_final = np.sqrt(
        np.mean(np.array(ses)**2 + (np.array(thetas) - theta_final)**2)
    )

    return {
        'theta': theta_final,
        'se': se_final,
        'sqrtNDn': np.mean(sqrtNDns),
        'R2_ell': np.mean(R2_ells),
        'R2_m': np.mean(R2_ms),
    }


# ============================================================
# SINGLE REPLICATION RUNNERS
# ============================================================

def run_one_rep_simulated(b, cell, Sigma):
    """One Monte Carlo rep for pure simulation."""
    rng = np.random.default_rng(SEED_BASE * 100 + cell['idx'] * 1000 + b)
    beta_m = make_beta(cell['p'], cell['tau_m'], cell['sp_m'])
    beta_g = make_beta(cell['p'], cell['tau_g'], cell['sp_g'])
    X, Y, D, m0, ell0 = generate_data_simulated(
        cell['n'], cell['p'], beta_m, beta_g, Sigma, rng
    )
    results = {}
    for method in ['lasso', 'ridge']:
        res = dml_estimate(X, Y, D, m0, ell0, method, K, S, rng)
        ci_lo = res['theta'] - 1.96 * res['se']
        ci_hi = res['theta'] + 1.96 * res['se']
        res['covers'] = float(ci_lo <= THETA_0 <= ci_hi)
        results[method] = res
    return results


def run_one_rep_semisynthetic(b, cell, X_real):
    """One Monte Carlo rep for semi-synthetic (fixed X, fresh noise)."""
    rng = np.random.default_rng(SEED_BASE * 200 + cell['idx'] * 1000 + b)
    p = X_real.shape[1]
    beta_m = make_beta(p, cell['tau_m'], cell['sp_m'])
    beta_g = make_beta(p, cell['tau_g'], cell['sp_g'])
    X, Y, D, m0, ell0 = generate_data_semisynthetic(X_real, beta_m, beta_g, rng)
    results = {}
    for method in ['lasso', 'ridge']:
        res = dml_estimate(X, Y, D, m0, ell0, method, K, S, rng)
        ci_lo = res['theta'] - 1.96 * res['se']
        ci_hi = res['theta'] + 1.96 * res['se']
        res['covers'] = float(ci_lo <= THETA_0 <= ci_hi)
        results[method] = res
    return results


# ============================================================
# CELL RUNNER (shared by both parts)
# ============================================================

def aggregate_results(all_results):
    """Aggregate B replications into summary statistics."""
    output = {}
    for method in ['lasso', 'ridge']:
        thetas  = [r[method]['theta']   for r in all_results]
        covers  = [r[method]['covers']  for r in all_results]
        sqrtNDn = [r[method]['sqrtNDn'] for r in all_results]
        R2_ells = [r[method]['R2_ell']  for r in all_results]
        R2_ms   = [r[method]['R2_m']    for r in all_results]

        output[method] = {
            'bias':    np.mean(thetas) - THETA_0,
            'rmse':    np.sqrt(np.mean((np.array(thetas) - THETA_0) ** 2)),
            'cover':   np.mean(covers),
            'sqrtNDn': np.mean(sqrtNDn),
            'R2_ell':  np.mean(R2_ells),
            'R2_m':    np.mean(R2_ms),
        }
    return output


# ============================================================
# PART A — PURE SIMULATION (Asymmetric beta_m ≠ beta_g)
# ============================================================

def build_grid_A():
    """Asymmetric pure-simulation grid."""
    grid = []

    # --- MAIN: Sparse strong treatment + dense weak outcome ---
    # "Treatment depends on few variables, outcome depends on everything weakly"
    for n in [500, 1000]:
        for p in [200, 500]:
            for rho in [0.0, 0.5]:
                for tau_g in [0.2, 0.1, 0.05]:
                    grid.append({
                        'label': 'sparse_m__dense_g',
                        'n': n, 'p': p, 'rho': rho,
                        'tau_m': 2.0, 'sp_m': 0.05,
                        'tau_g': tau_g, 'sp_g': 1.0,
                    })

    # --- REVERSE: Dense weak treatment + sparse strong outcome ---
    for n in [500, 1000]:
        for p in [200, 500]:
            for rho in [0.0, 0.5]:
                for tau_m in [0.1, 0.05]:
                    grid.append({
                        'label': 'dense_m__sparse_g',
                        'n': n, 'p': p, 'rho': rho,
                        'tau_m': tau_m, 'sp_m': 1.0,
                        'tau_g': 2.0, 'sp_g': 0.05,
                    })

    # --- MIXED: Moderate sparse treatment + moderate dense outcome ---
    for n in [500, 1000]:
        for p in [200, 500]:
            for rho in [0.0, 0.5]:
                grid.append({
                    'label': 'mixed_moderate',
                    'n': n, 'p': p, 'rho': rho,
                    'tau_m': 1.0, 'sp_m': 0.05,
                    'tau_g': 0.5, 'sp_g': 1.0,
                })

    # --- SYMMETRIC BASELINES (for comparison) ---
    for n in [500, 1000]:
        for p in [200]:
            for rho in [0.0]:
                # Both dense weak
                grid.append({
                    'label': 'symmetric_dense_weak',
                    'n': n, 'p': p, 'rho': rho,
                    'tau_m': 0.1, 'sp_m': 1.0,
                    'tau_g': 0.1, 'sp_g': 1.0,
                })
                # Both sparse strong
                grid.append({
                    'label': 'symmetric_sparse_strong',
                    'n': n, 'p': p, 'rho': rho,
                    'tau_m': 2.0, 'sp_m': 0.05,
                    'tau_g': 2.0, 'sp_g': 0.05,
                })

    return grid


def run_part_A(logfile, csvwriter):
    """Run Part A: pure simulation with asymmetric coefficients."""
    grid = build_grid_A()
    total = len(grid)
    print(f"\n{'='*70}")
    print(f"  PART A — Pure Simulation (Asymmetric)")
    print(f"  Cells: {total}  |  B={B}  |  S={S}  |  K={K}")
    print(f"{'='*70}\n")
    logfile.write(f"\n{'='*70}\n")
    logfile.write(f"  PART A — Pure Simulation (Asymmetric)\n")
    logfile.write(f"  Cells: {total}  |  B={B}  |  S={S}  |  K={K}\n")
    logfile.write(f"{'='*70}\n\n")

    for i, cell in enumerate(grid):
        cell['idx'] = i
        t0 = time.time()

        Sigma = make_toeplitz(cell['p'], cell['rho'])

        header = (
            f"[{i+1}/{total}]  n={cell['n']}  p={cell['p']}  "
            f"tau_m={cell['tau_m']}  sp_m={cell['sp_m']}  "
            f"tau_g={cell['tau_g']}  sp_g={cell['sp_g']}  "
            f"rho={cell['rho']}  [{cell['label']}]"
        )
        print(header)
        logfile.write(header + '\n')

        # Run B replications in parallel
        all_results = Parallel(n_jobs=N_JOBS)(
            delayed(run_one_rep_simulated)(b, cell, Sigma) for b in range(B)
        )

        output = aggregate_results(all_results)
        elapsed = time.time() - t0

        # Print and log
        print(f"  {'Method':<10} {'Bias':>8} {'RMSE':>8} {'Cover':>8} "
              f"{'sqrtNDn':>9} {'R2_ell':>8} {'R2_m':>8}")
        print(f"  {'-'*62}")
        logfile.write(f"  {'Method':<10} {'Bias':>8} {'RMSE':>8} {'Cover':>8} "
                      f"{'sqrtNDn':>9} {'R2_ell':>8} {'R2_m':>8}\n")
        logfile.write(f"  {'-'*62}\n")

        for method in ['lasso', 'ridge']:
            r = output[method]
            line = (
                f"  {method:<10} {r['bias']:>8.4f} {r['rmse']:>8.4f} "
                f"{r['cover']:>8.3f} {r['sqrtNDn']:>9.3f} "
                f"{r['R2_ell']:>8.3f} {r['R2_m']:>8.3f}"
            )
            print(line)
            logfile.write(line + '\n')

            # CSV row
            csvwriter.writerow([
                'A', cell['label'], cell['n'], cell['p'],
                cell['tau_m'], cell['sp_m'], cell['tau_g'], cell['sp_g'],
                cell['rho'], method,
                f"{r['bias']:.6f}", f"{r['rmse']:.6f}", f"{r['cover']:.3f}",
                f"{r['sqrtNDn']:.4f}", f"{r['R2_ell']:.4f}", f"{r['R2_m']:.4f}"
            ])

        print(f"  Done in {elapsed:.1f}s\n")
        logfile.write(f"  Done in {elapsed:.1f}s\n\n")
        logfile.flush()


# ============================================================
# PART B — SEMI-SYNTHETIC (FRED-MD predictors)
# ============================================================

def build_grid_B(p_fred):
    """Semi-synthetic grid using FRED-MD dimensions."""
    grid = []

    # --- Asymmetric: sparse treatment + dense weak outcome ---
    for tau_g in [0.2, 0.1, 0.05, 0.01]:
        grid.append({
            'label': 'fred_sparse_m__dense_g',
            'tau_m': 2.0, 'sp_m': 0.05,
            'tau_g': tau_g, 'sp_g': 1.0,
        })

    # --- Asymmetric reverse: dense weak treatment + sparse outcome ---
    for tau_m in [0.1, 0.05]:
        grid.append({
            'label': 'fred_dense_m__sparse_g',
            'tau_m': tau_m, 'sp_m': 1.0,
            'tau_g': 2.0, 'sp_g': 0.05,
        })

    # --- Symmetric baselines ---
    for tau in [0.1, 0.05]:
        grid.append({
            'label': 'fred_symmetric_dense',
            'tau_m': tau, 'sp_m': 1.0,
            'tau_g': tau, 'sp_g': 1.0,
        })

    grid.append({
        'label': 'fred_symmetric_sparse',
        'tau_m': 2.0, 'sp_m': 0.05,
        'tau_g': 2.0, 'sp_g': 0.05,
    })

    return grid


def run_part_B(logfile, csvwriter):
    """Run Part B: semi-synthetic with FRED-MD."""

    if not os.path.exists(FRED_PATH):
        print(f"\n  WARNING: {FRED_PATH} not found. Skipping Part B.")
        print(f"  Place fred_md.csv in the same directory and rerun with --part B")
        logfile.write(f"\n  WARNING: {FRED_PATH} not found. Skipping Part B.\n")
        return

    print(f"\n{'='*70}")
    print(f"  PART B — Semi-Synthetic (FRED-MD)")
    print(f"{'='*70}\n")
    logfile.write(f"\n{'='*70}\n")
    logfile.write(f"  PART B — Semi-Synthetic (FRED-MD)\n")
    logfile.write(f"{'='*70}\n\n")

    X_fred, col_names = load_fred_md(FRED_PATH)
    n_fred, p_fred = X_fred.shape

    logfile.write(f"  FRED-MD: n={n_fred}, p={p_fred}, p/n={p_fred/n_fred:.3f}\n\n")

    grid = build_grid_B(p_fred)
    total = len(grid)
    print(f"  Cells: {total}  |  B={B}  |  S={S}  |  K={K}\n")
    logfile.write(f"  Cells: {total}  |  B={B}  |  S={S}  |  K={K}\n\n")

    for i, cell in enumerate(grid):
        cell['idx'] = i + 10000  # offset to avoid seed clash with Part A
        cell['n'] = n_fred
        cell['p'] = p_fred
        cell['rho'] = 'empirical'  # real covariance, not Toeplitz
        t0 = time.time()

        header = (
            f"[{i+1}/{total}]  n={n_fred}  p={p_fred}  "
            f"tau_m={cell['tau_m']}  sp_m={cell['sp_m']}  "
            f"tau_g={cell['tau_g']}  sp_g={cell['sp_g']}  "
            f"[{cell['label']}]"
        )
        print(header)
        logfile.write(header + '\n')

        # Run B reps — X is fixed, only noise varies
        all_results = Parallel(n_jobs=N_JOBS)(
            delayed(run_one_rep_semisynthetic)(b, cell, X_fred) for b in range(B)
        )

        output = aggregate_results(all_results)
        elapsed = time.time() - t0

        print(f"  {'Method':<10} {'Bias':>8} {'RMSE':>8} {'Cover':>8} "
              f"{'sqrtNDn':>9} {'R2_ell':>8} {'R2_m':>8}")
        print(f"  {'-'*62}")
        logfile.write(f"  {'Method':<10} {'Bias':>8} {'RMSE':>8} {'Cover':>8} "
                      f"{'sqrtNDn':>9} {'R2_ell':>8} {'R2_m':>8}\n")
        logfile.write(f"  {'-'*62}\n")

        for method in ['lasso', 'ridge']:
            r = output[method]
            line = (
                f"  {method:<10} {r['bias']:>8.4f} {r['rmse']:>8.4f} "
                f"{r['cover']:>8.3f} {r['sqrtNDn']:>9.3f} "
                f"{r['R2_ell']:>8.3f} {r['R2_m']:>8.3f}"
            )
            print(line)
            logfile.write(line + '\n')

            csvwriter.writerow([
                'B', cell['label'], n_fred, p_fred,
                cell['tau_m'], cell['sp_m'], cell['tau_g'], cell['sp_g'],
                'empirical', method,
                f"{r['bias']:.6f}", f"{r['rmse']:.6f}", f"{r['cover']:.3f}",
                f"{r['sqrtNDn']:.4f}", f"{r['R2_ell']:.4f}", f"{r['R2_m']:.4f}"
            ])

        print(f"  Done in {elapsed:.1f}s\n")
        logfile.write(f"  Done in {elapsed:.1f}s\n\n")
        logfile.flush()


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    # Parse --part argument
    run_A = True
    run_B = True
    if '--part' in sys.argv:
        idx = sys.argv.index('--part')
        if idx + 1 < len(sys.argv):
            part = sys.argv[idx + 1].upper()
            run_A = (part == 'A')
            run_B = (part == 'B')

    # Open output files
    logfile = open('asymmetric_results.log', 'w')
    csvfile = open('asymmetric_results.csv', 'w', newline='')
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow([
        'part', 'label', 'n', 'p',
        'tau_m', 'sp_m', 'tau_g', 'sp_g',
        'rho', 'method',
        'bias', 'rmse', 'cover', 'sqrtNDn', 'R2_ell', 'R2_m'
    ])

    print("=" * 70)
    print("  Asymmetric DML Simulation")
    print(f"  B={B}  S={S}  K={K}  N_JOBS={N_JOBS}")
    print(f"  Parts: {'A' if run_A else ''} {'B' if run_B else ''}")
    print("=" * 70)
    logfile.write("=" * 70 + '\n')
    logfile.write("  Asymmetric DML Simulation\n")
    logfile.write(f"  B={B}  S={S}  K={K}  N_JOBS={N_JOBS}\n")
    logfile.write("=" * 70 + '\n')

    total_start = time.time()

    if run_A:
        run_part_A(logfile, csvwriter)

    if run_B:
        run_part_B(logfile, csvwriter)

    total_elapsed = time.time() - total_start
    hours = total_elapsed / 3600

    summary = f"\nTotal runtime: {total_elapsed:.0f}s ({hours:.1f} hours)\n"
    print(summary)
    logfile.write(summary)

    logfile.close()
    csvfile.close()

    print(f"Results saved to:")
    print(f"  asymmetric_results.log  (human-readable)")
    print(f"  asymmetric_results.csv  (machine-readable)")
