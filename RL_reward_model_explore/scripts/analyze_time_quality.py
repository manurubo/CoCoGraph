#!/usr/bin/env python3
"""Analyze model errors using CoCoGraph time-model predictions."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_analysis import OUTCOME_LABELS
from src.time_predictor import DEFAULT_TIME_CHECKPOINT
from src.time_quality_analysis import run_time_quality_analysis


def main():
    parser = argparse.ArgumentParser(description="Quality analysis via CoCoGraph time predictions.")
    parser.add_argument(
        "--predictions-csv",
        type=str,
        default=str(ROOT / "outputs" / "full_run_weighted" / "analysis" / "predictions.csv"),
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        default=str(ROOT / "data" / "turing_guess_level.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "full_run_weighted" / "time_quality_analysis"),
    )
    parser.add_argument(
        "--time-checkpoint",
        type=str,
        default=str(DEFAULT_TIME_CHECKPOINT),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(ROOT / "outputs" / "smoke_test" / "feature_cache"),
    )
    parser.add_argument("--model-name", type=str, default="full_run_weighted")
    args = parser.parse_args()

    df, summary = run_time_quality_analysis(
        predictions_csv=args.predictions_csv,
        data_csv=args.data_csv,
        output_dir=args.output_dir,
        model_name=args.model_name,
        time_checkpoint=args.time_checkpoint,
        cache_dir=args.cache_dir,
    )

    print(f"Saved time quality analysis to {args.output_dir}")
    print(f"Time checkpoint: {args.time_checkpoint}\n")
    for outcome, stats in summary["by_outcome"].items():
        means = stats["means"]
        print(f"{OUTCOME_LABELS[outcome]}")
        print(
            f"  time real={means.get('time_real', float('nan')):.4f}, "
            f"gen={means.get('time_generated', float('nan')):.4f}, "
            f"Δ={means.get('delta_time', float('nan')):.4f}"
        )
        print(f"  gen lower time: {stats['generated_lower_time_rate']:.1%}")
        print()

    print("Correlations:")
    for k, v in summary["correlations"].items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
