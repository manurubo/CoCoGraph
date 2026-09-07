"""Classify model outcomes by absolute reward/time levels (benign vs real errors)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .quality_analysis import OUTCOME_LABELS, assign_outcome
from .time_quality_analysis import enrich_with_time_predictions


def _add_level_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    r_med = df["reward_generated"].median()
    r_q75 = df["reward_generated"].quantile(0.75)
    t_med = df["time_generated"].median()

    df["reward_real_high"] = df["reward_real"] >= r_med
    df["reward_gen_high"] = df["reward_generated"] >= r_med
    df["reward_real_very_high"] = df["reward_real"] >= r_q75
    df["reward_gen_very_high"] = df["reward_generated"] >= r_q75
    df["time_gen_low"] = df["time_generated"] <= t_med
    df["time_real_low"] = df["time_real"] <= t_med

    df["abs_reward_margin"] = (df["reward_generated"] - df["reward_real"]).abs()
    df["abs_time_delta"] = df["delta_time"].abs()
    df["reward_close"] = df["abs_reward_margin"] < df["abs_reward_margin"].quantile(0.33)
    df["time_close"] = df["abs_time_delta"] < df["abs_time_delta"].quantile(0.33)

    df["both_rewards_high"] = df["reward_real_high"] & df["reward_gen_high"]
    df["both_rewards_low"] = (~df["reward_real_high"]) & (~df["reward_gen_high"])
    df["both_rewards_close"] = df["reward_close"]
    df["both_times_close"] = df["time_close"]
    return df


def classify_deceptive_gen(row: pd.Series) -> str:
    if row["both_rewards_high"] and row["generated_lower_time"]:
        return "both_high_gen_easier"
    if row["both_rewards_high"]:
        return "both_high_rewards"
    if row["reward_gen_high"] and not row["reward_real_high"]:
        return "gen_high_real_low"
    if row["both_rewards_close"] and row["both_times_close"]:
        return "both_close_ok"
    if row["reward_gen_very_high"]:
        return "gen_very_high"
    return "other"


def classify_missed_realism(row: pd.Series) -> str:
    if row["both_rewards_close"] and row["both_times_close"]:
        return "both_close_benign"
    if row["both_rewards_high"] and row["both_times_close"]:
        return "both_high_times_close"
    if row["both_rewards_high"]:
        return "both_high_rewards"
    if row["both_rewards_close"]:
        return "rewards_close"
    if row["both_rewards_low"]:
        return "both_low"
    return "other"


def classify_caught_realism(row: pd.Series) -> str:
    if row["_gen_much_better_reward"]:
        return "gen_much_better_reward"
    if row["_gen_slightly_better"]:
        return "gen_slightly_better_close"
    if row["generated_lower_time"] and row["reward_gen_high"]:
        return "gen_better_time_and_reward"
    if row["generated_lower_time"]:
        return "gen_better_time"
    return "other"


def classify_easy_agree(row: pd.Series) -> str:
    if row["both_rewards_low"] and not row["reward_gen_high"]:
        return "bad_gen_filtered"
    if not row["reward_gen_high"] and row["reward_real_high"]:
        return "real_good_gen_bad"
    return "other"


def enrich_with_subtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = _add_level_labels(df)
    reward_margin = df["reward_generated"] - df["reward_real"]
    df["_gen_much_better_reward"] = reward_margin > df["abs_reward_margin"].quantile(0.75)
    df["_gen_slightly_better"] = (reward_margin > 0) & df["reward_close"]

    subtype = []
    for idx, r in df.iterrows():
        if r["outcome"] == "deceptive_gen":
            subtype.append(classify_deceptive_gen(r))
        elif r["outcome"] == "missed_realism":
            subtype.append(classify_missed_realism(r))
        elif r["outcome"] == "caught_realism":
            subtype.append(classify_caught_realism(r))
        elif r["outcome"] == "easy_agree":
            if r["both_rewards_low"] or (not r["reward_gen_high"] and r["reward_real_high"]):
                subtype.append("bad_gen_filtered")
            else:
                subtype.append("other")
        else:
            subtype.append("other")
    df["subtype"] = subtype
    return df


def summarize_reward_levels(df: pd.DataFrame) -> dict:
    summary = {
        "medians": {
            "reward_real": float(df["reward_real"].median()),
            "reward_generated": float(df["reward_generated"].median()),
            "time_real": float(df["time_real"].median()),
            "time_generated": float(df["time_generated"].median()),
        },
        "by_outcome": {},
    }

    for outcome, subset in df.groupby("outcome"):
        subtypes = subset["subtype"].value_counts(normalize=True).round(3).to_dict()
        summary["by_outcome"][outcome] = {
            "label": OUTCOME_LABELS.get(outcome, outcome),
            "n": int(len(subset)),
            "mean_reward_real": float(subset["reward_real"].mean()),
            "mean_reward_generated": float(subset["reward_generated"].mean()),
            "mean_time_real": float(subset["time_real"].mean()),
            "mean_time_generated": float(subset["time_generated"].mean()),
            "both_rewards_high_rate": float(subset["both_rewards_high"].mean()),
            "both_rewards_low_rate": float(subset["both_rewards_low"].mean()),
            "rewards_close_rate": float(subset["both_rewards_close"].mean()),
            "times_close_rate": float(subset["both_times_close"].mean()),
            "subtype_rates": subtypes,
            "interpretation": _interpret_outcome(outcome, subset),
        }
    return summary


def _interpret_outcome(outcome: str, subset: pd.DataFrame) -> str:
    if outcome == "deceptive_gen":
        benign = subset["both_rewards_high"].mean()
        return (
            f"{benign:.0%} have BOTH rewards above median — model 'error' may be benign "
            f"(generated looks good; human was sharper or lucky)."
        )
    if outcome == "missed_realism":
        close_r = subset["both_rewards_close"].mean()
        close_t = subset["both_times_close"].mean()
        both_high = subset["both_rewards_high"].mean()
        return (
            f"{close_r:.0%} have close rewards, {close_t:.0%} close times, "
            f"{both_high:.0%} both high — human fooled but molecules look similar quality."
        )
    if outcome == "caught_realism":
        much_better = (subset["reward_generated"] - subset["reward_real"] > subset["abs_reward_margin"].quantile(0.75)).mean()
        close = subset["both_rewards_close"].mean()
        return (
            f"{much_better:.0%} gen much better by reward; {close:.0%} close rewards — "
            f"generated at least as good, often better than real."
        )
    if outcome == "easy_agree":
        bad_gen = ((~subset["reward_gen_high"]) & subset["reward_real_high"]).mean()
        return f"{bad_gen:.0%} have low-reward generated vs high-reward real — model filters bad fakes."
    return ""


def _markdown_report(summary: dict, model_name: str) -> str:
    lines = [
        f"# Reward Level Analysis ({model_name})",
        "",
        "Goal: distinguish **benign model mistakes** (both molecules look good)",
        "from **real failures**.",
        "",
        "Definitions:",
        "- **High reward** = above dataset median",
        "- **Close rewards** = margin in bottom third of all pairs",
        "- **Close times** = |Δtime| in bottom third",
        "",
    ]

    for outcome, stats in summary["by_outcome"].items():
        lines.extend(
            [
                f"## {stats['label']} (n={stats['n']})",
                "",
                f"- Mean reward: real={stats['mean_reward_real']:.2f}, gen={stats['mean_reward_generated']:.2f}",
                f"- Mean time: real={stats['mean_time_real']:.4f}, gen={stats['mean_time_generated']:.4f}",
                f"- Both rewards high: {stats['both_rewards_high_rate']:.1%}",
                f"- Both rewards low: {stats['both_rewards_low_rate']:.1%}",
                f"- Rewards close: {stats['rewards_close_rate']:.1%}",
                f"- Times close: {stats['times_close_rate']:.1%}",
                "",
                f"**{stats['interpretation']}**",
                "",
                "Subtypes:",
            ]
        )
        for st, rate in sorted(stats["subtype_rates"].items(), key=lambda x: -x[1]):
            lines.append(f"- `{st}`: {rate:.1%}")
        lines.append("")

    lines.extend(
        [
            "## PPO takeaway",
            "",
            "- **easy_agree + bad_gen_filtered**: reward model reliably downranks bad generated molecules.",
            "- **deceptive_gen + both_high**: model prefers a good fake — OK for PPO even if human wins.",
            "- **missed_realism + both_close_benign**: human noise; generated quality ≈ real.",
            "- **caught_realism + gen_much_better**: reward model identifies genuinely strong fakes.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_reward_levels(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    focus = ["deceptive_gen", "missed_realism", "caught_realism", "easy_agree"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, outcome in zip(axes.ravel(), focus):
        s = df[df["outcome"] == outcome]
        ax.scatter(s["reward_real"], s["reward_generated"], c=s["time_generated"], cmap="viridis_r", alpha=0.4, s=12)
        lims = [min(s["reward_real"].min(), s["reward_generated"].min()), max(s["reward_real"].max(), s["reward_generated"].max())]
        ax.plot(lims, lims, "r--", linewidth=1, label="reward gen = real")
        ax.set_xlabel("Reward real")
        ax.set_ylabel("Reward generated")
        ax.set_title(f"{outcome}\nn={len(s)}")
        cb = plt.colorbar(ax.collections[0], ax=ax, fraction=0.046)
        cb.set_label("time gen")
    fig.suptitle("Reward real vs generated (color = time gen; lower= better)", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "reward_scatter_by_outcome.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, outcome in zip(axes, ["deceptive_gen", "missed_realism", "caught_realism"]):
        s = df[df["outcome"] == outcome]
        ax.scatter(s["abs_reward_margin"], s["abs_time_delta"], alpha=0.35, s=14, c="#8172B3")
        ax.set_xlabel("|reward gen − real|")
        ax.set_ylabel("|Δtime|")
        ax.set_title(f"{outcome}\n(bottom-left = both close)")
    fig.tight_layout()
    fig.savefig(output_dir / "reward_time_closeness_errors.png", dpi=160)
    plt.close(fig)

    for outcome in focus:
        s = df[df["outcome"] == outcome]
        counts = s["subtype"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        counts.plot(kind="barh", ax=ax, color="#4C72B0")
        ax.set_title(f"Subtypes: {OUTCOME_LABELS.get(outcome, outcome)}")
        ax.set_xlabel("Count")
        fig.tight_layout()
        fig.savefig(output_dir / f"subtype_{outcome}.png", dpi=160)
        plt.close(fig)


def run_reward_level_analysis(
    predictions_csv: str | Path,
    data_csv: str | Path,
    output_dir: str | Path,
    model_name: str = "model",
    time_checkpoint: str | Path | None = None,
    cache_dir: str | Path | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = Path(predictions_csv)
    time_csv = output_dir.parent / "time_quality_analysis" / "predictions_with_time.csv"

    if time_csv.exists() and pred_path.exists() and time_csv.stat().st_mtime >= pred_path.stat().st_mtime:
        df = pd.read_csv(time_csv)
        if "outcome" not in df.columns:
            df["outcome"] = df.apply(assign_outcome, axis=1)
    else:
        df = enrich_with_time_predictions(predictions_csv, data_csv, time_checkpoint, cache_dir)

    df = enrich_with_subtypes(df)
    df.to_csv(output_dir / "predictions_with_levels.csv", index=False)

    summary = summarize_reward_levels(df)
    (output_dir / "reward_level_summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "reward_level_report.md").write_text(_markdown_report(summary, model_name))
    plot_reward_levels(df, output_dir)
    return df, summary
