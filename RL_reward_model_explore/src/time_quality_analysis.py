"""Analyze model errors using CoCoGraph time-model predictions as quality proxy."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DEVICE
from .dataset import FeatureCache
from .quality_analysis import OUTCOME_LABELS, assign_outcome
from .time_predictor import DEFAULT_TIME_CHECKPOINT, load_time_predictor, predict_time_smiles


def enrich_with_time_predictions(
    predictions_csv: str | Path,
    data_csv: str | Path,
    time_checkpoint: str | Path | None = None,
    cache_dir: str | Path | None = None,
    device=DEVICE,
) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_csv)
    source = pd.read_csv(data_csv)[["id", "real_smiles", "generated_smiles"]]
    df = predictions.merge(source, on="id", how="left")
    df["outcome"] = df.apply(assign_outcome, axis=1)

    feature_cache = FeatureCache(cache_dir) if cache_dir else None
    time_model = load_time_predictor(time_checkpoint, device=device)

    time_real, time_gen = [], []
    for row in df.itertuples(index=False):
        time_real.append(predict_time_smiles(time_model, row.real_smiles, device, feature_cache))
        time_gen.append(predict_time_smiles(time_model, row.generated_smiles, device, feature_cache))

    df["time_real"] = time_real
    df["time_generated"] = time_gen
    df = df.dropna(subset=["time_real", "time_generated"]).copy()

    # Lower predicted time = easier to generate = better in CoCoGraph sampling
    df["delta_time"] = df["time_generated"] - df["time_real"]
    df["generated_lower_time"] = (df["delta_time"] < 0).astype(int)
    df["generated_much_lower_time"] = (df["delta_time"] < -0.05).astype(int)
    return df


def summarize_time_by_outcome(df: pd.DataFrame) -> dict:
    metrics = [
        "time_real",
        "time_generated",
        "delta_time",
        "reward_real",
        "reward_generated",
        "reward_margin_real_vs_gen",
    ]
    summary = {
        "quality_proxy": "CoCoGraph time prediction (lower = easier/better to generate)",
        "time_checkpoint": str(DEFAULT_TIME_CHECKPOINT),
        "n_pairs": len(df),
        "by_outcome": {},
    }

    for outcome, subset in df.groupby("outcome"):
        summary["by_outcome"][outcome] = {
            "label": OUTCOME_LABELS.get(outcome, outcome),
            "n": int(len(subset)),
            "means": {m: float(subset[m].mean()) for m in metrics if m in subset.columns},
            "generated_lower_time_rate": float(subset["generated_lower_time"].mean()),
            "generated_much_lower_time_rate": float(subset["generated_much_lower_time"].mean()),
            "reward_prefers_generated_rate": float(subset["model_prefers_generated"].mean()),
        }

    summary["correlations"] = {
        "reward_generated_vs_time_generated": float(df["reward_generated"].corr(df["time_generated"])),
        "reward_real_vs_time_real": float(df["reward_real"].corr(df["time_real"])),
        "reward_margin_vs_delta_time": float(df["reward_margin_real_vs_gen"].corr(df["delta_time"])),
        "time_margin_vs_reward_margin": float(
            (df["time_real"] - df["time_generated"]).corr(df["reward_margin_real_vs_gen"])
        ),
    }
    return summary


def _markdown_report(summary: dict, model_name: str) -> str:
    lines = [
        f"# Time-Model Quality Report ({model_name})",
        "",
        "Quality proxy: **CoCoGraph time prediction** (`GINETimePredictor`).",
        "",
        "- **Lower time** = molecule is easier to reach during diffusion (CoCoGraph treats lower as better).",
        "- **Δtime = time(generated) − time(real)**",
        "  - negative → generated looks easier/better by time model",
        "  - positive → real looks easier/better",
        "",
    ]

    for outcome, stats in summary["by_outcome"].items():
        means = stats["means"]
        lines.extend(
            [
                f"## {stats['label']} (n={stats['n']})",
                "",
                f"- Mean time real: {means.get('time_real', float('nan')):.4f}",
                f"- Mean time generated: {means.get('time_generated', float('nan')):.4f}",
                f"- Mean Δtime (gen − real): {means.get('delta_time', float('nan')):.4f}",
                f"- Generated has lower time: {stats['generated_lower_time_rate']:.1%}",
                f"- Mean reward margin (real − gen): {means.get('reward_margin_real_vs_gen', float('nan')):.4f}",
                "",
            ]
        )

    corr = summary["correlations"]
    lines.extend(
        [
            "## Correlations",
            "",
            f"- reward(generated) vs time(generated): {corr['reward_generated_vs_time_generated']:.3f}",
            f"- reward(real) vs time(real): {corr['reward_real_vs_time_real']:.3f}",
            f"- reward margin vs Δtime: {corr['reward_margin_vs_delta_time']:.3f}",
            f"- (time_real − time_gen) vs reward margin: {corr['time_margin_vs_reward_margin']:.3f}",
            "",
            "## Interpretation",
            "",
            "- If **deceptive_gen** has generated time ≪ real time, the reward model may be following the time model.",
            "- If **missed_realism** has generated time ≥ real time, humans were fooled despite generated not looking",
            "  easier by CoCoGraph — supports label noise / non-time realism cues.",
            "- If **missed_realism** has generated time ≪ real time, the reward model is under-using the time signal.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_time_analysis(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    order = ["easy_agree", "deceptive_gen", "missed_realism", "caught_realism"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, outcome in zip(axes.ravel(), order):
        subset = df[df["outcome"] == outcome]
        ax.hist(subset["time_real"], bins=25, alpha=0.6, color="#4C72B0", label="Real")
        ax.hist(subset["time_generated"], bins=25, alpha=0.6, color="#DD8452", label="Generated")
        ax.set_title(f"{OUTCOME_LABELS[outcome][:42]}...\nn={len(subset)}")
        ax.set_xlabel("Predicted time")
        ax.legend(fontsize=8)
    fig.suptitle("CoCoGraph time prediction by outcome", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "time_by_outcome.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    data = [df[df["outcome"] == o]["delta_time"].dropna() for o in order]
    ax.boxplot(data, labels=[o.replace("_", "\n") for o in order])
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Δtime (generated − real) by outcome\n(negative = generated easier by time model)")
    ax.set_ylabel("Δtime")
    fig.tight_layout()
    fig.savefig(output_dir / "time_delta_by_outcome.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    focus = [
        ("deceptive_gen", "Model prefers generated\n(human picked real)"),
        ("missed_realism", "Model prefers real\n(human was fooled)"),
    ]
    for ax, (outcome, title) in zip(axes, focus):
        subset = df[df["outcome"] == outcome]
        ax.scatter(subset["time_real"], subset["time_generated"], alpha=0.35, s=16, c="#8172B3")
        lims = [0, max(subset["time_real"].max(), subset["time_generated"].max())]
        ax.plot(lims, lims, "k--", linewidth=1, label="time gen = time real")
        ax.set_xlabel("Time real")
        ax.set_ylabel("Time generated")
        ax.set_title(f"{title}\nn={len(subset)}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "time_scatter_model_errors.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (outcome, title) in zip(axes, focus):
        subset = df[df["outcome"] == outcome]
        ax.scatter(subset["time_generated"], subset["reward_generated"], alpha=0.35, s=16, c="#DD8452", label="Generated")
        ax.scatter(subset["time_real"], subset["reward_real"], alpha=0.35, s=16, c="#4C72B0", label="Real")
        ax.set_xlabel("Time prediction")
        ax.set_ylabel("Reward model score")
        ax.set_title(f"{title}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "time_vs_reward_model_errors.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(order))
    width = 0.35
    real_means = [df[df["outcome"] == o]["time_real"].mean() for o in order]
    gen_means = [df[df["outcome"] == o]["time_generated"].mean() for o in order]
    ax.bar(x - width / 2, real_means, width, label="Real", color="#4C72B0")
    ax.bar(x + width / 2, gen_means, width, label="Generated", color="#DD8452")
    ax.set_xticks(x, [o.replace("_", "\n") for o in order], fontsize=9)
    ax.set_ylabel("Mean predicted time")
    ax.set_title("Mean CoCoGraph time prediction by outcome (lower = better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "time_mean_by_outcome.png", dpi=160)
    plt.close(fig)

    fooled = df[df["human_fooled"] == 1]
    fig, ax = plt.subplots(figsize=(8, 5))
    for outcome, color, label in [
        ("missed_realism", "#C44E52", "Model missed realism"),
        ("caught_realism", "#55A868", "Model caught realism"),
    ]:
        subset = fooled[fooled["outcome"] == outcome]
        ax.hist(subset["delta_time"], bins=30, alpha=0.55, label=f"{label} (n={len(subset)})", color=color)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Δtime (generated − real)")
    ax.set_title("Fooled cases: time margin when model missed vs caught")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "time_delta_fooled_missed_vs_caught.png", dpi=160)
    plt.close(fig)


def run_time_quality_analysis(
    predictions_csv: str | Path,
    data_csv: str | Path,
    output_dir: str | Path,
    model_name: str = "model",
    time_checkpoint: str | Path | None = None,
    cache_dir: str | Path | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = enrich_with_time_predictions(
        predictions_csv,
        data_csv,
        time_checkpoint=time_checkpoint,
        cache_dir=cache_dir,
    )
    df.to_csv(output_dir / "predictions_with_time.csv", index=False)

    summary = summarize_time_by_outcome(df)
    (output_dir / "time_quality_summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "time_quality_report.md").write_text(_markdown_report(summary, model_name))
    plot_time_analysis(df, output_dir)
    return df, summary
