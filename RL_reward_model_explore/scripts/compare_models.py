#!/usr/bin/env python3
"""Compare reward models across training runs."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUNS = {
    "full_run": ROOT / "outputs" / "full_run",
    "full_run_weighted": ROOT / "outputs" / "full_run_weighted",
    "v2_improved_run": ROOT / "outputs" / "v2_improved_run",
}


def _load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return None


def _load_run(name: str, run_dir: Path) -> dict:
    summary = _load_json(run_dir / "analysis" / "summary.json") or {}
    fooled = summary.get("fooled_eval", {}).get("fooled_subset", {})
    if not fooled and (run_dir / "fooled_eval" / "fooled_summary.json").exists():
        fooled = _load_json(run_dir / "fooled_eval" / "fooled_summary.json").get("fooled_subset", {})

    reward_levels = _load_json(run_dir / "reward_level_analysis" / "reward_level_summary.json") or {}
    time_quality = _load_json(run_dir / "time_quality_analysis" / "time_quality_summary.json") or {}
    history = _load_json(run_dir / "training_history.json") or {}

    best_ckpt_fooled = None
    if history.get("history"):
        best = max(history["history"], key=lambda x: x.get("val", {}).get("composite_score", float("-inf")))
        best_ckpt_fooled = best.get("val", {}).get("fooled_accuracy")

    rl = reward_levels.get("by_outcome", {})
    return {
        "name": name,
        "fooled_weight": history.get("fooled_weight"),
        "best_val_composite": history.get("best_val_composite_score"),
        "val_fooled_at_best_composite": best_ckpt_fooled,
        "overall_accuracy": summary.get("overall_accuracy"),
        "test_accuracy": summary.get("by_split", {}).get("test", {}).get("accuracy"),
        "fooled_accuracy": summary.get("by_human_fooled", {}).get("1", {}).get("accuracy"),
        "not_fooled_accuracy": summary.get("by_human_fooled", {}).get("0", {}).get("accuracy"),
        "model_prefers_generated_rate": summary.get("model_prefers_generated_rate"),
        "fooled_gen_above_real_rate": fooled.get("model_ranks_generated_above_real_rate"),
        "fooled_test_accuracy": fooled.get("by_split", {}).get("test", {}).get("pairwise_accuracy"),
        "easy_bad_gen_filtered": rl.get("easy_agree", {}).get("subtype_rates", {}).get("bad_gen_filtered"),
        "deceptive_both_high": rl.get("deceptive_gen", {}).get("both_rewards_high_rate"),
        "missed_both_close": rl.get("missed_realism", {}).get("rewards_close_rate"),
        "caught_gen_much_better": rl.get("caught_realism", {}).get("subtype_rates", {}).get("gen_much_better_reward"),
        "caught_n": rl.get("caught_realism", {}).get("n"),
        "missed_n": rl.get("missed_realism", {}).get("n"),
        "time_corr_reward_gen": time_quality.get("correlations", {}).get("reward_generated_vs_time_generated"),
    }


def _pct(x):
    return "—" if x is None else f"{100*x:.1f}%"


def _fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def build_report(rows: list[dict]) -> str:
    lines = [
        "# Reward Model Comparison",
        "",
        "| Metric | full_run | full_run_weighted | v2_improved_run |",
        "|---|---|---|---|",
    ]

    metrics = [
        ("fooled_weight (training)", lambda r: _fmt(r["fooled_weight"], 1)),
        ("Best val composite score", lambda r: _fmt(r["best_val_composite"])),
        ("Overall pairwise accuracy", lambda r: _pct(r["overall_accuracy"])),
        ("Test accuracy", lambda r: _pct(r["test_accuracy"])),
        ("**Fooled accuracy** (human picked gen)", lambda r: _pct(r["fooled_accuracy"])),
        ("Not-fooled accuracy", lambda r: _pct(r["not_fooled_accuracy"])),
        ("Model prefers generated rate", lambda r: _pct(r["model_prefers_generated_rate"])),
        ("Fooled: gen reward > real rate", lambda r: _pct(r["fooled_gen_above_real_rate"])),
        ("Fooled test accuracy", lambda r: _pct(r["fooled_test_accuracy"])),
        ("Easy agree: bad_gen_filtered", lambda r: _pct(r["easy_bad_gen_filtered"])),
        ("Deceptive gen: both rewards high", lambda r: _pct(r["deceptive_both_high"])),
        ("Missed realism: rewards close", lambda r: _pct(r["missed_both_close"])),
        ("Caught realism: gen much better", lambda r: _pct(r["caught_gen_much_better"])),
        ("Caught realism count (n)", lambda r: str(r["caught_n"] or "—")),
        ("Missed realism count (n)", lambda r: str(r["missed_n"] or "—")),
        ("Reward–time correlation (gen)", lambda r: _fmt(r["time_corr_reward_gen"])),
    ]

    for label, fn in metrics:
        lines.append(f"| {label} | " + " | ".join(fn(r) for r in rows) + " |")

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "### v2_improved_run wins on:",
            "- **Fooled / realism detection** — ~82% vs ~49% fooled accuracy",
            "- **Ranking generated above real when humans were fooled** — 82% vs 49%",
            "- **Caught realism** — 1358 pairs vs 814; 32% gen much better vs 7%",
            "- **Missed realism** — only 302 cases vs 846 (fewer failures on hard cases)",
            "- **Less conservative bias** — prefers generated 54% vs human 37% (v1 ~30%)",
            "",
            "### v2_improved_run loses on:",
            "- **Not-fooled / easy cases** — 63% vs 82% (worse at agreeing human picked obvious real)",
            "- **Test overall accuracy** — 57% vs 67% (generalization gap, possible overfit on fooled)",
            "- **More deceptive_gen errors** — 1061 vs 513 (more often prefers gen when human picked real)",
            "",
            "### For PPO:",
            "**Use v2_improved_run** if the goal is promoting realistic/generated-looking molecules.",
            "It strongly upranks fakes humans were fooled by. Keep validity + KL constraints because",
            "it is more aggressive and test fooled acc (~66%) is below train (~86%).",
            "",
            "**Use full_run_weighted** if you want a conservative filter that crushes bad fakes",
            "but won't push realism as hard.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser_args = __import__("argparse").ArgumentParser()
    parser_args.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "outputs" / "v2_improved_run" / "model_comparison.md"),
    )
    args = parser_args.parse_args()

    rows = [_load_run(name, path) for name, path in RUNS.items()]
    report = build_report(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(report)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
