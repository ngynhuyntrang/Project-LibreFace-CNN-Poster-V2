"""
Final model comparison script for CSCI323 Facial Expression Recognition project.

Loads prediction CSVs for all three models (Simple CNN, POSTER V2, LibreFace),
computes accuracy / macro-F1 / weighted-F1, and writes:
  results/tables/accuracy_summary.csv
  results/plots/accuracy_bar.png

Usage:
    python scripts/final_comparison.py
    python scripts/final_comparison.py --predictions predictions/
"""

from __future__ import annotations
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

MODELS = [
    ("simple_cnn",  "Simple CNN"),
    ("poster_v2",   "POSTER V2"),
    ("libreface",   "LibreFace"),
]

CLASS_NAMES = [
    "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


def load_csv(predictions_dir: str, model_key: str) -> pd.DataFrame | None:
    path = os.path.join(predictions_dir, f"{model_key}_raf_db.csv")
    if not os.path.isfile(path):
        print(f"  [SKIP] {path} — file not found")
        return None
    df = pd.read_csv(path)
    print(f"  [OK]   {path}  ({len(df)} rows)")
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    y_true = df["true_label"].astype(int).values
    y_pred = df["pred_label"].astype(int).values
    return {
        "accuracy":    round(accuracy_score(y_true, y_pred) * 100, 2),
        "macro_f1":    round(f1_score(y_true, y_pred, average="macro",    zero_division=0) * 100, 2),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
        "n_samples":   len(df),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_accuracy_bar(rows: list[dict], save_path: str) -> None:
    labels   = [r["Model"] for r in rows]
    accuracy = [r["Accuracy (%)"] for r in rows]
    macro_f1 = [r["Macro F1 (%)"] for r in rows]
    weighted_f1 = [r["Weighted F1 (%)"] for r in rows]

    x      = np.arange(len(labels))
    width  = 0.25
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_acc = ax.bar(x - width, accuracy,    width, label="Accuracy",    color=colors[0])
    bars_mac = ax.bar(x,         macro_f1,    width, label="Macro F1",    color=colors[1])
    bars_wgt = ax.bar(x + width, weighted_f1, width, label="Weighted F1", color=colors[2])

    for bars in (bars_acc, bars_mac, bars_wgt):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Model Comparison on RAF-DB Test Set", fontsize=13)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare all FER models on RAF-DB test set."
    )
    parser.add_argument(
        "--predictions", type=str, default="predictions",
        help="Directory containing *_raf_db.csv files (relative to project root or absolute).",
    )
    parser.add_argument(
        "--tables", type=str, default="results/tables",
        help="Output directory for CSV tables (relative to project root or absolute).",
    )
    parser.add_argument(
        "--plots", type=str, default="results/plots",
        help="Output directory for plots (relative to project root or absolute).",
    )
    args = parser.parse_args()

    predictions_dir = resolve(args.predictions)
    tables_dir      = resolve(args.tables)
    plots_dir       = resolve(args.plots)

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir,  exist_ok=True)

    print("\n=== CSCI323 — Final Model Comparison ===\n")
    print(f"Predictions : {predictions_dir}")
    print(f"Tables      : {tables_dir}")
    print(f"Plots       : {plots_dir}\n")

    # Load & compute
    print("Loading CSVs ...")
    summary_rows = []
    for model_key, model_name in MODELS:
        df = load_csv(predictions_dir, model_key)
        if df is None:
            continue
        m = compute_metrics(df)
        summary_rows.append({
            "Model":           model_name,
            "Accuracy (%)":    m["accuracy"],
            "Macro F1 (%)":    m["macro_f1"],
            "Weighted F1 (%)": m["weighted_f1"],
            "Samples":         m["n_samples"],
        })

    if not summary_rows:
        raise RuntimeError(
            f"No prediction CSVs found in: {predictions_dir}\n"
            "Run model inference scripts first."
        )

    # Summary table
    summary_df = pd.DataFrame(summary_rows)
    table_path = os.path.join(tables_dir, "accuracy_summary.csv")
    summary_df.to_csv(table_path, index=False)

    print("\n--- Summary Table ---")
    print(summary_df.to_string(index=False))
    print(f"\n  Saved: {table_path}")

    # Bar chart
    plot_path = os.path.join(plots_dir, "accuracy_bar.png")
    plot_accuracy_bar(summary_rows, plot_path)

    print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
