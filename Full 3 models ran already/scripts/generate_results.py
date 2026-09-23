# Author: Minh Quan Dang
# Date: 12 May 2026

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    confusion_matrix
)

# Configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(BASE_DIR, "../predictions")
TABLES_DIR = os.path.join(BASE_DIR, "../results/tables")
PLOTS_DIR = os.path.join(BASE_DIR, "../results/plots")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "../data/label_map.json")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load label map
with open(LABEL_MAP_PATH) as f:
    lm = json.load(f)

EMOTIONS = [lm["idx_to_emotion"][str(i)] for i in range(0, 7)]

MODELS = ["simple_cnn", "poster_v2", "libreface"]

def load_csv(model: str):
    path = os.path.join(PRED_DIR, f"{model}_raf_db.csv")
    if not os.path.exists(path):
        print(f"Skipping {model}_raf_db.csv — file not found")
        return None
    return pd.read_csv(path)


def compute_metrics(df: pd.DataFrame):
    y_true = df["true_label"].astype(int)
    y_pred = df["pred_label"].astype(int)

    return {
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
        "per_class_f1": [round(v * 100, 2) for v in f1_score(y_true, y_pred, average=None, labels=list(range(0,7)), zero_division=0)],
        "per_class_prec": [round(v * 100, 2) for v in precision_score(y_true, y_pred, average=None, labels=list(range(0,7)), zero_division=0)],
        "per_class_rec": [round(v * 100, 2) for v in recall_score(y_true, y_pred, average=None, labels=list(range(0,7)), zero_division=0)],
        "confusion": confusion_matrix(y_true, y_pred, labels=list(range(0,7))),
        "n_samples": len(df),
    }


def plot_confusion_matrix(cm, title: str, save_path: str):
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    print("\n")
    print("  CSCI323 - Final Results Generator (RAF-DB Only)")
    print("\n")

    all_metrics = {m: {} for m in MODELS}

    for model in MODELS:
        df = load_csv(model)
        if df is None:
            continue
        all_metrics[model]["raf_db"] = compute_metrics(df)
        print(f"Processed {model}.csv: {len(df)} samples")

    print("\nGenerating Tables")

    # Summary Table
    summary_rows = []
    for model in MODELS:
        m = all_metrics[model].get("raf_db", {})
        summary_rows.append({
            "Model": model.replace("_", " ").title(),
            "Accuracy (%)": m.get("accuracy", "N/A"),
            "Macro F1 (%)": m.get("macro_f1", "N/A"),
            "Weighted F1 (%)": m.get("weighted_f1", "N/A"),
            "Samples": m.get("n_samples", "N/A")
        })

    pd.DataFrame(summary_rows).to_csv(os.path.join(TABLES_DIR, "model_comparison.csv"), index=False)
    print("  Saved: results/tables/model_comparison.csv")

    # Per-class F1
    pcf1_rows = []
    for model in MODELS:
        m = all_metrics[model].get("raf_db")
        if not m:
            continue
        row = {"Model": model.replace("_", " ").title()}
        for i, emotion in enumerate(EMOTIONS):
            row[emotion] = m["per_class_f1"][i]
        pcf1_rows.append(row)

    pd.DataFrame(pcf1_rows).to_csv(os.path.join(TABLES_DIR, "per_class_f1.csv"), index=False)
    print("  Saved: results/tables/per_class_f1.csv")

    print("\n--- Generating Plots ---")

    for model in MODELS:
        m = all_metrics[model].get("raf_db")
        if not m:
            continue
        title = f"{model.replace('_', ' ').title()} - RAF-DB"
        path = os.path.join(PLOTS_DIR, f"cm_{model}.png")
        plot_confusion_matrix(m["confusion"], title, path)

    print("\n")
    print("   Results generation completed successfully!")
    print(f"  Tables → {TABLES_DIR}")
    print(f"  Plots  → {PLOTS_DIR}")
    print("\n")


if __name__ == "__main__":
    main()