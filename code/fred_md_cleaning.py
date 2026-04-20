"""
fred_md_cleaning.py
===================
Apply the McCracken and Ng (2016) transformations to raw FRED-MD data
and save a cleaned CSV for use in empirical.py.

Usage:
    python fred_md_cleaning.py
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def fred_transform(x, tcode):
    """
    Apply a McCracken and Ng transformation code to induce stationarity.

    Codes:
        1: level (no transform)
        2: first difference
        3: second difference
        4: log
        5: log first difference (growth rate)
        6: log second difference
        7: delta(x_t / x_{t-1} - 1)
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


def clean_fredmd(input_path, output_path='fred_md_cleaned.csv'):
    """Load raw FRED-MD, apply the per-column transformations, drop series
    with too many missing values, and save the result to CSV."""
    print(f"Loading {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Raw shape: {df.shape}")

    # Transformation codes live on the first data row.
    tcodes = {}
    for col in df.columns[1:]:
        try:
            tcodes[col] = int(float(df.iloc[0][col]))
        except (ValueError, TypeError):
            tcodes[col] = 1   # default: no transform

    df = df.iloc[1:].copy().reset_index(drop=True)
    df = df.rename(columns={df.columns[0]: 'date'})

    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print("Applying McCracken-Ng transformations...")
    for col in df.columns[1:]:
        if col in tcodes:
            df[col] = fred_transform(df[col], tcodes[col])

    # Drop columns with more than 10 percent missing, then drop residual NaN rows.
    missing_frac = df.isnull().sum() / len(df)
    keep_cols    = missing_frac[missing_frac <= 0.10].index
    dropped_cols = [col for col in df.columns if col not in keep_cols]
    df = df[keep_cols]
    print(f"  Dropped {len(dropped_cols)} columns with >10% missing.")

    before_rows = len(df)
    df = df.dropna()
    print(f"  Dropped {before_rows - len(df)} rows with remaining NaNs.")

    print(f"Cleaned shape: {df.shape}, "
          f"dates {df['date'].iloc[0]} to {df['date'].iloc[-1]}.")

    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}.")

    # Quick sanity check on a few headline series.
    key_vars      = ['INDPRO', 'Y', 'CPIAUCSL', 'FEDFUNDS', 'UNRATE']
    existing_vars = [v for v in key_vars if v in df.columns]
    if existing_vars:
        print("Summary stats for headline series:")
        for var in existing_vars:
            print(f"  {var:>10}  mean={df[var].mean():+.4f}  "
                  f"sd={df[var].std():.4f}  "
                  f"min={df[var].min():+.4f}  max={df[var].max():+.4f}")

    return df


if __name__ == '__main__':
    INPUT_FILE  = 'fred_md.csv'
    OUTPUT_FILE = 'fred_md_cleaned.csv'

    try:
        clean_fredmd(INPUT_FILE, OUTPUT_FILE)
    except FileNotFoundError:
        print(f"Could not find {INPUT_FILE}. "
              f"Place it in the current directory or provide a full path.")
    except Exception as e:
        print(f"Error: {e}")