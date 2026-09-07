#!/usr/bin/env python3
"""Analyze molecular quality in model error cases."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_analysis import OUTCOME_LABELS, run_quality_analysis


def main():
    parser = argparse.ArgumentParser(description="Analyze quality descriptors in model/human outcome buckets.")
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
        default=str(ROOT / "outputs" / "full_run_weighted" / "quality_analysis"),
    )
    parser.add_argument("--model-name", type=str, default="full_run_weighted")
    parser.add_argument(
        "--cache-path",
        type=str,
        default=str(ROOT / "outputs" / "quality_descriptor_cache.json"),
    )
    args = parser.parse_args()

    df, summary = run_quality_analysis(
        predictions_csv=args.predictions_csv,
        data_csv=args.data_csv,
        output_dir=args.output_dir,
        model_name=args.model_name,
        cache_path=args.cache_path,
    )

    print(f"Saved quality analysis to {args.output_dir}\n")
    for outcome, stats in summary["by_outcome"].items():
        means = stats["means"]
        print(f"{OUTCOME_LABELS[outcome]}")
        print(f"  n={stats['n']}")
        print(f"  QED real={means.get('real_qed', float('nan')):.3f}, gen={means.get('generated_qed', float('nan')):.3f}, Δ={means.get('delta_qed', float('nan')):.3f}")
        print(f"  gen higher QED: {stats['generated_higher_qed_rate']:.1%}")
        print()


if __name__ == "__main__":
    main()
