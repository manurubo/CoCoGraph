#!/usr/bin/env python3
"""Generate dedicated human-fooled evaluation report."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import collect_predictions
from src.eval_fooled import generate_fooled_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate reward model on human-fooled comparisons.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "outputs" / "full_run" / "best_reward_model.pt"),
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        default=str(ROOT / "data" / "turing_guess_level.csv"),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(ROOT / "outputs" / "smoke_test" / "feature_cache"),
    )
    parser.add_argument(
        "--history-json",
        type=str,
        default=str(ROOT / "outputs" / "full_run" / "training_history.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "full_run" / "fooled_eval"),
    )
    args = parser.parse_args()

    df = collect_predictions(
        checkpoint_path=args.checkpoint,
        data_csv=args.data_csv,
        cache_dir=args.cache_dir,
        history_json=args.history_json,
    )
    summary = generate_fooled_report(df, args.output_dir)

    fooled = summary["fooled_subset"]
    print(f"Report saved to {args.output_dir}")
    print(f"Fooled pairwise accuracy: {fooled['pairwise_accuracy']:.1%}")
    print(f"Model ranks generated > real (fooled): {fooled['model_ranks_generated_above_real_rate']:.1%}")
    print(f"Mean margin gen-real (fooled): {fooled['mean_margin_gen_minus_real']:.4f}")
    print("\nSee fooled_eval_report.md for full breakdown.")


if __name__ == "__main__":
    main()
