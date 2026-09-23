"""Generate dummy prediction CSVs for testing the unified eval framework.

This script fabricates plausible-looking predictions for three fake models
(``simple_cnn``, ``poster_v2``, ``libreface``) on two fake datasets
(``raf_db``, ``ck_plus``). Each model is parameterised by a different overall
accuracy so the comparison plots actually show variation.

Run from the project root:

    python scripts/make_dummy_predictions.py

This will create six CSV files under ``predictions/`` matching the standard
schema documented in ``eval/utils.py``.
"""

# Author: Minh Quan Dang
# Created on: 10 May 2026

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from eval.utils import NUM_CLASSES, PROB_COLUMNS, CLASS_LABELS, ensure_dir

RNG_SEED = 69420
PREDICTIONS_DIR = Path("predictions")


def _generate_one(
    n_samples: int,
    target_accuracy: float,
    *,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build one predictions DataFrame with approximately ``target_accuracy``."""
    y_true = rng.choice(CLASS_LABELS, size=n_samples)

    # Decide which samples will be correctly classified.
    correct_mask = rng.random(n_samples) < target_accuracy
    y_pred = y_true.copy()

    # For the incorrect ones, pick any class other than the true one.
    wrong_idx = np.where(~correct_mask)[0]
    for i in wrong_idx:
        choices = [c for c in CLASS_LABELS if c != y_true[i]]
        y_pred[i] = rng.choice(choices)

    # Build calibrated-ish probabilities: high mass on the predicted class,
    # the rest spread randomly over the other six classes and renormalised.
    probs = rng.dirichlet(alpha=np.ones(NUM_CLASSES) * 0.6, size=n_samples)
    pred_mass = rng.uniform(0.55, 0.92, size=n_samples)
    final = np.zeros((n_samples, NUM_CLASSES), dtype=float)
    for i in range(n_samples):
        # Drop the predicted column out of the dirichlet draw, rescale the
        # remaining six to (1 - pred_mass), then drop pred_mass on prediction.
        pos = CLASS_LABELS.index(y_pred[i])   # label -> array position
        rest = np.delete(probs[i], pos)
        rest = rest / rest.sum() * (1.0 - pred_mass[i])
        final[i] = np.insert(rest, pos, pred_mass[i])
        
    image_paths = [f"img_{i:05d}.jpg" for i in range(n_samples)]
    df = pd.DataFrame(
        {
            "image_path": image_paths,
            "true_label": y_true.astype(int),
            "pred_label": y_pred.astype(int),
        }
    )
    for j, col in enumerate(PROB_COLUMNS):
        df[col] = final[:, j]
    return df


def main() -> None:
    ensure_dir(PREDICTIONS_DIR)
    rng = np.random.default_rng(RNG_SEED)

    plan = [
        # (model_name, dataset_name, n_samples, target_accuracy)
        ("simple_cnn", "raf_db", 3000, 0.62),
        ("simple_cnn", "ck_plus", 600, 0.55),
        ("poster_v2",  "raf_db", 3000, 0.88),
        ("poster_v2",  "ck_plus", 600, 0.77),
        ("libreface",  "raf_db", 3000, 0.81),
        ("libreface",  "ck_plus", 600, 0.71),
    ]

    for model_name, dataset_name, n, acc in plan:
        df = _generate_one(n_samples=n, target_accuracy=acc, rng=rng)
        out = PREDICTIONS_DIR / f"{model_name}_{dataset_name}.csv"
        df.to_csv(out, index=False)
        empirical_acc = (df["true_label"] == df["pred_label"]).mean()
        print(
            f"  Wrote {out}  "
            f"(n={len(df):>4}, target_acc={acc:.2f}, empirical_acc={empirical_acc:.3f})"
        )


if __name__ == "__main__":
    main()
