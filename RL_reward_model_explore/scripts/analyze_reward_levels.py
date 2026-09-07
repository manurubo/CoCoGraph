#!/usr/bin/env python3
"""Analyze whether model mistakes are benign (both molecules good quality)."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality_analysis import OUTCOME_LABELS
from src.reward_level_analysis import run_reward_level_analysis
from src.time_predictor import DEFAULT_TIME_CHECKPOINT


def main():
    parser = argparse.ArgumentParser(description="Reward level analysis for benign vs real errors.")
    parser.add_argument(
        "--predictions-csv",
        type=str,
        default=str(ROOT / "outputs" / "full_run_weighted" / "analysis" / "predictions.csv"),
    )
    parser.add_argument("--data-csv", type=str, default=str(ROOT / "data" / "turing_guess_level.csv"))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "full_run_weighted" / "reward_level_analysis"),
    )
    parser.add_argument("--time-checkpoint", type=str, default=str(DEFAULT_TIME_CHECKPOINT))
    parser.add_argument("--cache-dir", type=str, default=str(ROOT / "outputs" / "smoke_test" / "feature_cache"))
    parser.add_argument("--model-name", type=str, default="full_run_weighted")
    args = parser.parse_args()

    df, summary = run_reward_level_analysis(
        predictions_csv=args.predictions_csv,
        data_csv=args.data_csv,
        output_dir=args.output_dir,
        model_name=args.model_name,
        time_checkpoint=args.time_checkpoint,
        cache_dir=args.cache_dir,
    )

    print(f"Saved to {args.output_dir}\n")
    for outcome, stats in summary["by_outcome"].items():
        print(f"=== {OUTCOME_LABELS[outcome]} (n={stats['n']}) ===")
        print(f"  both rewards high: {stats['both_rewards_high_rate']:.1%}")
        print(f"  rewards close: {stats['rewards_close_rate']:.1%}")
        print(f"  times close: {stats['times_close_rate']:.1%}")
        print(f"  → {stats['interpretation']}")
        print()


if __name__ == "__main__":
    main()
