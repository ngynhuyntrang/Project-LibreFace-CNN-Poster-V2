# Author: Ba Hoang Khanh Phan
# Created on: 12 May, 2026

import os
import pandas as pd

# Configurations
EXPECTED_FILES = [
    "simple_cnn.csv",
    "poster_v2.csv",
    "libreface.csv",
]

EXPECTED_COLUMNS = [
    "image_path", 
    "true_label", 
    "pred_label",
    "prob_1", "prob_2", "prob_3", "prob_4", 
    "prob_5", "prob_6", "prob_7"
]

EXPECTED_ROWS = 3068   # Test set size

PREDICTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../predictions")

def check_csv(filepath):
    issues = []
    filename = os.path.basename(filepath)

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return [f"Cannot read CSV: {e}"]

    # Check columns
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
        return issues

    # Check row count
    if len(df) != EXPECTED_ROWS:
        issues.append(f"Row count {len(df)} — expected {EXPECTED_ROWS}")

    # Check label range (1 to 7)
    for col in ["true_label", "pred_label"]:
        bad = df[~df[col].between(1, 7)]
        if len(bad) > 0:
            issues.append(f" {col} out of range (1-7): {len(bad)} rows")

    # Check probability columns sum to ~1.0
    prob_cols = [f"prob_{i}" for i in range(1, 8)]
    prob_sums = df[prob_cols].sum(axis=1)
    bad_probs = df[~prob_sums.between(0.99, 1.01)]
    if len(bad_probs) > 0:
        issues.append(f" Probability sums not ~1.0: {len(bad_probs)} rows")

    # Check no absolute paths
    abs_paths = df[df["image_path"].str.startswith(("/", "\\")) | 
                   df["image_path"].str.contains(":", regex=False)]
    if len(abs_paths) > 0:
        issues.append(f"Absolute paths found in image_path: {len(abs_paths)} rows")

    # Check no NaN values
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        issues.append(f"NaN values found: {nan_count} total")

    return issues


def main():
    print("\n")
    print("  CSCI323 Prediction CSV Consolidation Check")
    print("\n")

    all_passed = True

    for fname in EXPECTED_FILES:
        fpath = os.path.join(PREDICTIONS_DIR, fname)
        print(f"\n[{fname}]")

        if not os.path.exists(fpath):
            print("Not Yet Generated")
            all_passed = False
            continue

        issues = check_csv(fpath)

        if not issues:
            df = pd.read_csv(fpath)
            acc = (df["true_label"] == df["pred_label"]).mean() * 100
            print(f"Passed — {len(df)} rows — Accuracy: {acc:.2f}%")
        else:
            all_passed = False
            for issue in issues:
                print(f"  {issue}")

    print("\n")
    if all_passed:
        print("ALL CHECKS PASSED — Ready for final evaluation!")
    else:
        print("Some files are missing or have issues.")
        print("Please fix them before running the results script.")
    print("\n")


if __name__ == "__main__":
    main()