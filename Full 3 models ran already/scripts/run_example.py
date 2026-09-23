"""End-to-end example: loads the dummy predictions, runs the full report.

Run from the project root:

    python scripts/make_dummy_predictions.py    # one-off, creates predictions/
    python scripts/run_example.py               # builds the full report
"""

# Author: Minh Quan Dang
# Created on: 10 May 2026

from __future__ import annotations
import sys
from pathlib import Path

# Add the project root to sys.path so `eval` is importable from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathlib import Path
from eval import Evaluator

PREDICTIONS_DIR = Path("predictions")
RESULTS_DIR = Path("results")


def main() -> None:
    evaluator = Evaluator(results_dir=RESULTS_DIR)

    # Register every (model, dataset) the team has produced. The convention is: predictions/<model_name>_<dataset_name>.csv
    plan = [
        ("simple_cnn", "raf_db"),
        ("simple_cnn", "ck_plus"),
        ("poster_v2",  "raf_db"),
        ("poster_v2",  "ck_plus"),
        ("libreface",  "raf_db"),
        ("libreface",  "ck_plus"),
    ]
    for model, dataset in plan:
        csv = PREDICTIONS_DIR / f"{model}_{dataset}.csv"
        evaluator.add_predictions(model, dataset, csv)

    # Headline comparison printed inline
    print("\nRAF-DB comparison")
    print(evaluator.compare_models("raf_db").to_string(index=False))

    print("\nCK+ comparison")
    print(evaluator.compare_models("ck_plus").to_string(index=False))

    print("\nCross-dataset gap (RAF-DB - CK+)")
    print(evaluator.cross_dataset_summary("raf_db", "ck_plus").to_string(index=False))

    # Save everything to disk in one call
    evaluator.generate_full_report()


if __name__ == "__main__":
    main()
