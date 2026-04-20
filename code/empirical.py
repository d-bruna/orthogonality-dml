"""
empirical.py
============
Empirical component for:
    "When Does Orthogonality Fail to Protect?
     Ridge versus Lasso as Nuisance Estimators in Double Machine Learning"

Purpose:
    - Load three real economic datasets from CSV files
    - Position each on the signal-strength spectrum via hat_tau = R2 / (1 - R2)
    - Compare Ridge vs Lasso first-stage prediction (out-of-fold R2)
    - Run DML with both methods and measure estimate stability
    - Approximate Delta_n from out-of-fold nuisance residuals
    - Multiple treatment variables per dataset for robustness

Design:
    - Loaders exclude treatment columns from X and return them separately.
      No standardisation is applied in the loaders — X, Y, and D are kept
      in their original units to preserve causal interpretability of theta_hat.
    - For each focal treatment, main() augments X with the other treatments
      as additional controls, so the nuisance regressions correctly partial
      out all confounders.
    - X is standardised within each cross-fitting fold only (no leakage).
    - D is standardised within each cross-fitting fold only (no leakage),
      so Ridge/Lasso regularisation is scale-invariant across treatments.
    - theta_hat is stored and reported in original units throughout.
    - Figure 5 (robustness) rescales theta_hat by SD(D) purely for display,
      expressing effects in units of one SD of D to make treatments comparable
      within a dataset. This rescaling is not applied to any stored results.
    - Each treatment's causal estimate is computed completely independently;
      the augmentation affects only nuisance inputs, not the DML estimates
      of other treatments.

Datasets and treatments:
    FRED-MD (n~757, p~112):
        Y       = INDPRO    (industrial production growth, standard DML benchmark)
        D_1     = CPIAUCSL  (CPI inflation)
        D_2     = FEDFUNDS  (federal funds rate)
        D_3     = UNRATE    (unemployment rate)
        D_4     = HOUST     (housing starts)
        D_5     = M2SL      (money supply)
        D_6     = PAYEMS    (payroll employment)
        D_7     = S&P 500   (equity returns)
        D_8     = RPI       (real personal income)
        D_9     = GS10      (10-year Treasury yield)

    Barro-Lee (n=90, p~56):
        Y       = Outcome (GDP per capita growth)
        D_1     = invsh41   (investment share)
        D_2     = gdpsh465  (initial GDP level)

    Goyal-Welch (n~68, p~11):
        Y       = y (equity premium)
        D_1     = d_p   (dividend-price ratio)
        D_2     = e_p   (earnings-price ratio)
        D_3     = b_m   (book-to-market ratio)
        D_4     = tbl   (T-bill rate)
        D_5     = dfy   (default yield spread)

Usage:
    python empirical.py
"""

import numpy as np
import pandas as pd
import warnings

from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


K_OUTER = 5     # outer cross-fitting folds
K_INNER = 3     # inner CV folds for lambda selection
S_REPS  = 20    # cross-fitting repetitions for stability
METHODS = ['lasso', 'ridge']



def load_fredmd(path):
    df = pd.read_csv(path)

    # Shift INDPRO by -1 to forecast the next month
    target_macro_var = 'INDPRO'
    df['Y'] = df[target_macro_var].shift(-1)
    df = df.dropna(subset=['Y']).reset_index(drop=True)

    Y_col      = 'Y'
    treat_cols = ['CPIAUCSL', 'FEDFUNDS', 'UNRATE', 'HOUST', 'M2SL',
                  'PAYEMS', 'S&P 500', 'RPI', 'GS10']
    treat_cols = [c for c in treat_cols if c in df.columns]

    drop_cols = [Y_col, 'date'] + treat_cols
    Y    = df[Y_col].values.astype(float)
    X_df = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])

    mask = ~(np.isnan(Y) | X_df.isna().any(axis=1).values)
    Y    = Y[mask]
    X    = X_df.values[mask].astype(float)

    treatments = {}
    for t in treat_cols:
        d = df[t].values.astype(float)[mask]
        treatments[t] = d

    print(f"  FRED-MD: n={X.shape[0]}, p={X.shape[1]}, treatments={treat_cols}")
    return X, Y, treatments

def load_barrolee(path):
    """
    Barro-Lee: 90 countries, ~62 variables.
    Outcome = 'Outcome' (GDP per capita growth).
    Treatment columns are excluded from X and returned separately.
    No standardisation applied — original units preserved.
    """
    df = pd.read_csv(path)

    Y_col      = 'Outcome'
    treat_cols = ['invsh41', 'gdpsh465', 'human65', 'pinstab1', 'govsh41', 'bmp1l']
    treat_cols = [c for c in treat_cols if c in df.columns]

    # 'intercept' is a constant column in this dataset and must be excluded
    # from X — it carries no information and would waste a regularisation slot.
    drop_cols = [Y_col, 'intercept'] + treat_cols
    Y    = df[Y_col].values.astype(float)
    X_df = df.drop(columns=drop_cols, errors='ignore').select_dtypes(
        include=[np.number])

    mask = ~(np.isnan(Y) | X_df.isna().any(axis=1).values)
    Y    = Y[mask]
    X    = X_df.values[mask].astype(float)

    treatments = {}
    for t in treat_cols:
        d = df[t].values.astype(float)[mask]
        treatments[t] = d

    print(f"  Barro-Lee: n={X.shape[0]}, p={X.shape[1]}, "
          f"treatments={treat_cols}")
    return X, Y, treatments


def load_goyal(path):
    """
    Goyal-Welch: ~68 obs, ~16 predictor columns + outcome.
    Outcome = 'y' (equity premium).
    Treatment columns are excluded from X and returned separately.
    No standardisation applied — original units preserved.
    """
    df = pd.read_csv(path)

    Y_col      = 'y'
    treat_cols = ['d_p', 'e_p', 'b_m', 'tbl', 'dfy', 'infl', 'svar',
                  'lty', 'ntis', 'tms']
    treat_cols = [c for c in treat_cols if c in df.columns]

    drop_cols = [Y_col, 'Year'] + treat_cols
    Y    = df[Y_col].values.astype(float)
    X_df = df.drop(columns=drop_cols, errors='ignore').select_dtypes(
        include=[np.number])

    mask = ~(np.isnan(Y) | X_df.isna().any(axis=1).values)
    Y    = Y[mask]
    X    = X_df.values[mask].astype(float)

    treatments = {}
    for t in treat_cols:
        d = df[t].values.astype(float)[mask]
        treatments[t] = d

    print(f"  Goyal-Welch: n={X.shape[0]}, p={X.shape[1]}, "
          f"treatments={treat_cols}")
    return X, Y, treatments



def fit_nuisance(X_tr, y_tr, method, K_inner):
    if method == 'lasso':
        return LassoCV(
            cv=K_inner,
            max_iter=10_000,
            n_jobs=1,
            random_state=0,
            precompute=True,
        ).fit(X_tr, y_tr)
    # Use an explicit KFold object so Ridge inner CV is true K-fold,
    # identical in structure to the LassoCV inner CV (not LOO).
    inner_cv = KFold(n_splits=K_inner, shuffle=True, random_state=0)
    return RidgeCV(
        alphas=np.logspace(-4, 6, 60),
        cv=inner_cv,
        scoring='neg_mean_squared_error',
    ).fit(X_tr, y_tr)



def dml_run(X, D, Y, method, K_outer, K_inner, seed):
    """
    X : augmented covariate matrix — base covariates plus other treatments
        as controls, built in main(). Focal treatment D is excluded.
    D : focal treatment vector in original units.
    Y : outcome vector in original units.

    X is standardised within each fold (fit on train, applied to test).
    D and Y are not standardised — theta_hat retains its original-unit
    causal interpretation.
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

        # Within-fold standardisation of X only — prevents leakage.
        # D and Y are kept in original units throughout.
        scaler  = StandardScaler().fit(X_tr)
        X_tr_s  = scaler.transform(X_tr)
        X_te_s  = scaler.transform(X_te)

        ell_mod = fit_nuisance(X_tr_s, Y_tr, method, K_inner)
        m_mod   = fit_nuisance(X_tr_s, D_tr, method, K_inner)

        ell_fitted[test_idx] = ell_mod.predict(X_te_s)
        m_fitted[test_idx]   = m_mod.predict(X_te_s)
        ell_resid[test_idx]  = Y_te - ell_fitted[test_idx]
        v_resid[test_idx]    = D_te - m_fitted[test_idx]

    denom     = np.sum(v_resid ** 2)
    theta_hat = np.sum(v_resid * ell_resid) / (denom + 1e-12)

    eps_hat   = ell_resid - theta_hat * v_resid
    meat      = np.mean(v_resid ** 2 * eps_hat ** 2)
    bread     = np.mean(v_resid ** 2)
    sigma_hat = np.sqrt(meat / (bread ** 2 * n + 1e-12))

    return theta_hat, sigma_hat, ell_resid, v_resid, ell_fitted, m_fitted


def analyse_treatment(ds_name, t_name, X, D, Y,
                      K_outer=K_OUTER, K_inner=K_INNER, S=S_REPS):
    """
    Run DML with Ridge and Lasso over S repetitions for one treatment.
    X is the augmented covariate matrix built in main(): base covariates
    plus all treatments except the focal one appended as controls.
    theta_hat is in original units of Y per original unit of D.
    """
    n, p = X.shape
    results = {}

    for method in METHODS:
        print(f"      {method.upper()} (S={S})...", end=' ', flush=True)

        thetas, sigmas = [], []
        ell_accum = np.zeros(n)
        m_accum   = np.zeros(n)

        for s in range(S):
            try:
                th, sig, _, _, ef, mf = dml_run(
                    X, D, Y, method, K_outer, K_inner, seed=s
                )
                thetas.append(th)
                sigmas.append(sig)
                ell_accum += ef
                m_accum   += mf
            except Exception as e:
                print(f"[rep {s} failed: {e}]", end=' ')
                continue

        if not thetas:
            print("FAILED — skipping")
            continue

        k       = len(thetas)
        ell_avg = ell_accum / k
        m_avg   = m_accum   / k

        # Out-of-fold R2
        ss_y   = np.sum((Y - Y.mean()) ** 2) + 1e-12
        ss_d   = np.sum((D - D.mean()) ** 2) + 1e-12
        ell_r2 = 1 - np.sum((ell_avg - Y) ** 2) / ss_y
        m_r2   = 1 - np.sum((m_avg   - D) ** 2) / ss_d

        # Approximate Delta_n from out-of-fold RMSE product
        ell_rmse     = np.sqrt(np.mean((ell_avg - Y) ** 2))
        m_rmse       = np.sqrt(np.mean((m_avg   - D) ** 2))
        sqrt_n_delta = np.sqrt(n) * ell_rmse * m_rmse

        # Rescale by SD(Y) and SD(D) to make the diagnostic comparable across
        # datasets and treatments with different units and scales.
        sd_y = np.std(Y)
        sd_d = np.std(D)
        sqrt_n_delta_norm = np.sqrt(n) * (ell_rmse / sd_y) * (m_rmse / sd_d)

        # Average CI width: 2 * 1.96 * sigma_avg (two-sided 95% interval).
        # sigma_hat from dml_run already equals SE(theta_hat).
        sigma_avg = float(np.mean(sigmas))
        ci_width  = 2 * 1.96 * sigma_avg

        # hat_tau from Ridge R2 only (Lasso R2 biased downward under weak signals).
        tau_hat_ell = np.nan
        tau_hat_m   = np.nan
        if method == 'ridge':
            r2c_ell     = max(min(ell_r2, 0.999), 0.001)
            r2c_m       = max(min(m_r2,   0.999), 0.001)
            tau_hat_ell = r2c_ell / (1 - r2c_ell)
            tau_hat_m   = r2c_m   / (1 - r2c_m)

        results[method] = dict(
            dataset           = ds_name,
            treatment         = t_name,
            method            = method,
            n                 = n,
            p                 = p,
            theta_mean        = float(np.mean(thetas)),
            theta_std         = float(np.std(thetas)),
            sigma_mean        = sigma_avg,
            ci_width          = float(ci_width),
            ell_r2            = float(ell_r2),
            m_r2              = float(m_r2),
            sqrt_n_delta      = float(sqrt_n_delta),       # unscaled
            sqrt_n_delta_norm = float(sqrt_n_delta_norm),  # rescaled by SD(Y) and SD(D)
            tau_hat_ell       = float(tau_hat_ell),
            tau_hat_m         = float(tau_hat_m),
            n_reps            = k,
        )

        tau_str = (f"tau_ell={tau_hat_ell:.3f}, tau_m={tau_hat_m:.3f}"
                   if method == 'ridge' else "")
        print(f"theta={results[method]['theta_mean']:.3f} "
              f"(std={results[method]['theta_std']:.3f}), "
              f"R2_ell={ell_r2:.3f}, R2_m={m_r2:.3f}, "
              f"CI_w={ci_width:.3f}, "
              f"sqrtNDn={sqrt_n_delta:.3f} "
              f"{tau_str}")

    return results



def print_summary(all_rows):
    print()
    print("=" * 110)
    print("EMPIRICAL SUMMARY")
    print("=" * 110)
    print(f"{'Dataset':<14} {'Treatment':<12} {'Method':<8} {'theta':<8} "
          f"{'std':<7} {'CI_w':<7} {'R2_ell':<8} {'R2_m':<7} "
          f"{'sqrtNDn':<9} {'tau_ell':<9} {'tau_m'}")
    print("-" * 110)

    current_ds = None
    for r in all_rows:
        if r['dataset'] != current_ds:
            if current_ds is not None:
                print()
            current_ds = r['dataset']
        te_str = f"{r['tau_hat_ell']:.3f}" if not np.isnan(r['tau_hat_ell']) else "N/A"
        tm_str = f"{r['tau_hat_m']:.3f}"   if not np.isnan(r['tau_hat_m'])   else "N/A"
        print(f"{r['dataset']:<14} {r['treatment']:<12} {r['method']:<8} "
              f"{r['theta_mean']:>7.3f}  "
              f"{r['theta_std']:>6.3f}  "
              f"{r['ci_width']:>6.3f}  "
              f"{r['ell_r2']:>7.3f}  "
              f"{r['m_r2']:>6.3f}  "
              f"{r['sqrt_n_delta']:>8.3f}  "
              f"{te_str:<9} {tm_str}")



def main():
    print("=" * 60)
    print("  Empirical Analysis: Ridge vs Lasso in DML")
    print("=" * 60)

    data_dir = '.'

    print("\nLoading datasets...")
    datasets = {}

    try:
        X, Y, treatments = load_fredmd(f'{data_dir}/fred_md_cleaned.csv')
        datasets['FRED-MD'] = (X, Y, treatments)
    except Exception as e:
        print(f"  WARNING: FRED-MD failed: {e}")

    try:
        X, Y, treatments = load_barrolee(f'{data_dir}/barrolee.csv')
        datasets['Barro-Lee'] = (X, Y, treatments)
    except Exception as e:
        print(f"  WARNING: Barro-Lee failed: {e}")

    try:
        X, Y, treatments = load_goyal(f'{data_dir}/Goyal_x_raw.csv')
        datasets['Goyal-Welch'] = (X, Y, treatments)
    except Exception as e:
        print(f"  WARNING: Goyal-Welch failed: {e}")

    if not datasets:
        raise ValueError("No datasets loaded.")

    

    all_rows = []
    for ds_name, (X_base, Y, treatments) in datasets.items():
        print(f"\n[{ds_name}]  n={X_base.shape[0]}, p_base={X_base.shape[1]}")
        for t_focal, D in treatments.items():
            # Augment X with all treatments except the focal one as controls
            others = [treatments[k].reshape(-1, 1)
                      for k in treatments if k != t_focal]
            X_comb = np.hstack([X_base] + others) if others else X_base
            print(f"  Treatment: {t_focal} (p_total={X_comb.shape[1]})")
            res = analyse_treatment(
                ds_name, t_focal, X_comb, D, Y,
                K_outer=K_OUTER, K_inner=K_INNER, S=S_REPS,
            )
            for method_res in res.values():
                all_rows.append(method_res)

    print_summary(all_rows)

    df = pd.DataFrame(all_rows)
    df.to_csv('results_empirical_shifted.csv', index=False)
    print(f"\nResults saved to results_empirical_shifted.csv ({len(df)} rows)")


    print("\nDone.")
    return df


if __name__ == '__main__':
    main()