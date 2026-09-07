"""Dedicated evaluation report for human-fooled (hard) comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def summarize_fooled_subset(df: pd.DataFrame) -> dict:
    fooled = df[df["human_fooled"] == 1].copy()
    not_fooled = df[df["human_fooled"] == 0].copy()

    fooled["model_ranks_generated_above_real"] = (
        fooled["reward_generated"] > fooled["reward_real"]
    ).astype(int)

    summary = {
        "description": "Metrics when human was fooled (preferred generated over real).",
        "n_fooled_pairs": len(fooled),
        "n_not_fooled_pairs": len(not_fooled),
        "overall_accuracy": _safe_mean(df["model_correct"]),
        "fooled_subset": {
            "pairwise_accuracy": _safe_mean(fooled["model_correct"]),
            "model_prefers_generated_rate": _safe_mean(fooled["model_prefers_generated"]),
            "model_ranks_generated_above_real_rate": _safe_mean(
                fooled["model_ranks_generated_above_real"]
            ),
            "mean_reward_generated": _safe_mean(fooled["reward_generated"]),
            "mean_reward_real": _safe_mean(fooled["reward_real"]),
            "mean_margin_gen_minus_real": _safe_mean(
                fooled["reward_generated"] - fooled["reward_real"]
            ),
            "mean_score_diff": _safe_mean(fooled["score_diff"]),
            "mean_prob_preferred": _safe_mean(fooled["prob_preferred"]),
        },
        "not_fooled_subset": {
            "pairwise_accuracy": _safe_mean(not_fooled["model_correct"]),
            "model_prefers_generated_rate": _safe_mean(not_fooled["model_prefers_generated"]),
            "mean_margin_gen_minus_real": _safe_mean(
                not_fooled["reward_generated"] - not_fooled["reward_real"]
            ),
        },
    }

    if len(fooled):
        summary["fooled_subset"]["by_split"] = (
            fooled.groupby("split")
            .agg(
                n=("model_correct", "size"),
                pairwise_accuracy=("model_correct", "mean"),
                ranks_gen_above_real=("model_ranks_generated_above_real", "mean"),
                mean_margin_gen_minus_real=(
                    "reward_generated",
                    lambda s: (fooled.loc[s.index, "reward_generated"] - fooled.loc[s.index, "reward_real"]).mean(),
                ),
            )
            .round(4)
            .to_dict("index")
        )
        summary["fooled_subset"]["by_biochemistry_level"] = (
            fooled.groupby("biochemistry_level")
            .agg(
                n=("model_correct", "size"),
                pairwise_accuracy=("model_correct", "mean"),
                ranks_gen_above_real=("model_ranks_generated_above_real", "mean"),
            )
            .round(4)
            .to_dict("index")
        )

    return summary


def _markdown_report(summary: dict) -> str:
    fooled = summary["fooled_subset"]
    not_fooled = summary["not_fooled_subset"]
    lines = [
        "# Human-Fooled Evaluation Report",
        "",
        "This report focuses on comparisons where the human was **fooled** and preferred",
        "the **generated** molecule over the real one. These are the hard cases that matter",
        "most for RL toward human-like realism.",
        "",
        "## Headline metrics",
        "",
        f"- Overall pairwise accuracy: **{summary['overall_accuracy']:.1%}**",
        f"- Fooled subset accuracy: **{fooled['pairwise_accuracy']:.1%}** (n={summary['n_fooled_pairs']})",
        f"- Not-fooled subset accuracy: **{not_fooled['pairwise_accuracy']:.1%}** (n={summary['n_not_fooled_pairs']})",
        f"- Model ranks generated > real on fooled cases: **{fooled['model_ranks_generated_above_real_rate']:.1%}**",
        "",
        "## Fooled subset reward statistics",
        "",
        f"- Mean reward(generated): {fooled['mean_reward_generated']:.4f}",
        f"- Mean reward(real): {fooled['mean_reward_real']:.4f}",
        f"- Mean margin gen − real: {fooled['mean_margin_gen_minus_real']:.4f}",
        f"- Mean preferred−rejected margin: {fooled['mean_score_diff']:.4f}",
        "",
        "## Interpretation",
        "",
        "- **Pairwise accuracy** on fooled cases = model correctly ranks the human-preferred (generated) molecule higher.",
        "- **Ranks generated > real** = unconditional check: does the model think the fake is better than the real partner?",
        "- If fooled accuracy ≈ 50%, the model is not yet learning the subtle realism signal needed for PPO.",
        "",
    ]

    if "by_split" in fooled:
        lines.extend(["## Fooled subset by split", ""])
        for split_name, stats in fooled["by_split"].items():
            lines.append(
                f"- **{split_name}**: acc={stats['pairwise_accuracy']:.1%}, "
                f"gen>real={stats['ranks_gen_above_real']:.1%}, n={stats['n']}"
            )
        lines.append("")

    if "by_biochemistry_level" in fooled:
        lines.extend(["## Fooled subset by biochemistry level", ""])
        for level, stats in fooled["by_biochemistry_level"].items():
            lines.append(
                f"- **{level}**: acc={stats['pairwise_accuracy']:.1%}, "
                f"gen>real={stats['ranks_gen_above_real']:.1%}, n={stats['n']}"
            )
        lines.append("")

    return "\n".join(lines)


def plot_fooled_report(df: pd.DataFrame, output_dir: Path):
    fooled = df[df["human_fooled"] == 1]
    not_fooled = df[df["human_fooled"] == 0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.hist(fooled["score_diff"], bins=30, color="#DD8452", alpha=0.85, edgecolor="white")
    acc = fooled["model_correct"].mean()
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"Fooled: preferred−rejected margin\nacc={acc:.3f}, n={len(fooled)}")
    ax.set_xlabel("score_diff")
    ax.set_ylabel("Count")

    ax = axes[0, 1]
    ax.hist(
        fooled["reward_generated"] - fooled["reward_real"],
        bins=30,
        color="#8172B3",
        alpha=0.85,
        edgecolor="white",
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Fooled: reward(generated) − reward(real)")
    ax.set_xlabel("Margin")
    ax.set_ylabel("Count")

    ax = axes[1, 0]
    ax.hist(not_fooled["score_diff"], bins=30, color="#4C72B0", alpha=0.6, label="Not fooled")
    ax.hist(fooled["score_diff"], bins=30, color="#DD8452", alpha=0.6, label="Fooled")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Score diff: fooled vs not fooled")
    ax.set_xlabel("score_diff")
    ax.legend()

    ax = axes[1, 1]
    categories = ["Overall", "Not fooled", "Fooled"]
    accs = [
        df["model_correct"].mean(),
        not_fooled["model_correct"].mean(),
        fooled["model_correct"].mean(),
    ]
    gen_real = [
        (df["reward_generated"] > df["reward_real"]).mean(),
        (not_fooled["reward_generated"] > not_fooled["reward_real"]).mean(),
        (fooled["reward_generated"] > fooled["reward_real"]).mean(),
    ]
    x = range(len(categories))
    width = 0.35
    ax.bar([i - width / 2 for i in x], accs, width, label="Pairwise acc", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], gen_real, width, label="Prefers gen>real", color="#DD8452")
    ax.axhline(0.5, color="#C44E52", linestyle="--", linewidth=1)
    ax.set_xticks(list(x), categories)
    ax.set_ylim(0, 1)
    ax.set_title("Fooled vs not fooled metrics")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "fooled_eval_plots.png", dpi=160)
    plt.close(fig)


def generate_fooled_report(df: pd.DataFrame, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fooled_df = df[df["human_fooled"] == 1].copy()
    fooled_df.to_csv(output_dir / "fooled_predictions.csv", index=False)

    summary = summarize_fooled_subset(df)
    (output_dir / "fooled_summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "fooled_eval_report.md").write_text(_markdown_report(summary))
    plot_fooled_report(df, output_dir)
    return summary
