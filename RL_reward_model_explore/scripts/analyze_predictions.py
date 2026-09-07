#!/usr/bin/env python3
"""Analyze reward-model predictions and generate diagnostic plots."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import run_analysis


def main():
    parser = argparse.ArgumentParser(description="Analyze reward model predictions.")
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
        help="Use player_splits from this file to match training split.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "full_run" / "analysis"),
    )
    args = parser.parse_args()

    df, summary = run_analysis(
        checkpoint_path=args.checkpoint,
        data_csv=args.data_csv,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        history_json=args.history_json,
    )

    print(f"Saved {len(df)} scored pairs to {args.output_dir}")
    print(f"Overall accuracy: {summary['overall_accuracy']:.3f}")
    print(f"Human fooled rate: {summary['human_fooled_rate']:.3f}")
    print(f"Model prefers generated rate: {summary['model_prefers_generated_rate']:.3f}")
    print(f"Mean reward real: {summary['mean_reward_real']:.4f}")
    print(f"Mean reward generated: {summary['mean_reward_generated']:.4f}")
    print(f"Mean |score diff|: {summary.get('mean_abs_score_diff', float('nan')):.4f}")
    print("\nBy split:")
    for split_name, stats in summary["by_split"].items():
        print(f"  {split_name}: acc={stats['accuracy']:.3f}, n={stats['n']}")
    print("\nBy human_fooled:")
    for fooled, stats in summary["by_human_fooled"].items():
        label = "fooled (pref gen)" if int(fooled) == 1 else "not fooled (pref real)"
        print(f"  {label}: acc={stats['accuracy']:.3f}, n={stats['n']}")
    if "fooled_eval" in summary:
        fe = summary["fooled_eval"]["fooled_subset"]
        print("\nDedicated fooled eval:")
        print(f"  ranks gen>real: {fe['model_ranks_generated_above_real_rate']:.3f}")
        print(f"  mean margin gen-real: {fe['mean_margin_gen_minus_real']:.4f}")
        print(f"  report: {args.output_dir}/fooled_eval/fooled_eval_report.md")


if __name__ == "__main__":
    main()
