"""Analyze reward-model predictions on preference pairs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import DEVICE, SEED
from .dataset import FeatureCache, load_preference_dataframe, score_graphs, split_by_player
from .eval_fooled import generate_fooled_report
from .inference import load_reward_model


def _assign_split(df: pd.DataFrame, player_splits: dict | None, seed: int = SEED) -> pd.DataFrame:
    if player_splits is not None:
        split_map = {}
        for split_name, players in player_splits.items():
            for player_id in players:
                split_map[int(player_id)] = split_name
        out = df.copy()
        out["split"] = out["player_id"].map(split_map)
        return out

    splits, players = split_by_player(df, seed=seed)
    frames = []
    for split_name, part in splits.items():
        part = part.copy()
        part["split"] = split_name
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def _score_from_cache(model, cache: FeatureCache, smiles: str, device: torch.device):
    data = cache.get(smiles)
    if data is None:
        return None
    scores = score_graphs(model, [data], device)
    return float(scores.item())


def collect_predictions(
    checkpoint_path: str | Path,
    data_csv: str | Path,
    cache_dir: str | Path,
    history_json: str | Path | None = None,
    device: torch.device = DEVICE,
) -> pd.DataFrame:
    checkpoint_path = Path(checkpoint_path)
    cache_dir = Path(cache_dir)
    model = load_reward_model(checkpoint_path, device)

    player_splits = None
    if history_json is not None and Path(history_json).exists():
        history = json.loads(Path(history_json).read_text())
        player_splits = history.get("player_splits")

    df = load_preference_dataframe(data_csv)
    df = _assign_split(df, player_splits)

    cache = FeatureCache(cache_dir)
    rows = []

    for row in df.itertuples(index=False):
        r_real = _score_from_cache(model, cache, row.real_smiles, device)
        r_gen = _score_from_cache(model, cache, row.generated_smiles, device)
        r_pref = _score_from_cache(model, cache, row.preferred_smiles, device)
        r_rej = _score_from_cache(model, cache, row.rejected_smiles, device)
        if None in (r_real, r_gen, r_pref, r_rej):
            continue

        score_diff = r_pref - r_rej
        model_prefers_generated = int(r_gen > r_real)
        model_correct = int(score_diff > 0)

        rows.append(
            {
                "id": int(row.id) if hasattr(row, "id") else None,
                "split": row.split,
                "player_id": int(row.player_id),
                "biochemistry_level": row.biochemistry_level,
                "expert_weight": float(row.expert_weight),
                "human_fooled": int(row.human_fooled),
                "human_prefers_generated": int(row.human_fooled),
                "preferred_is_generated": int(row.human_fooled),
                "reward_real": r_real,
                "reward_generated": r_gen,
                "reward_preferred": r_pref,
                "reward_rejected": r_rej,
                "score_diff": score_diff,
                "prob_preferred": float(torch.sigmoid(torch.tensor(score_diff)).item()),
                "model_prefers_generated": model_prefers_generated,
                "model_correct": model_correct,
                "model_agrees_with_human": int(model_prefers_generated == row.human_fooled),
                "reward_margin_real_vs_gen": r_real - r_gen,
            }
        )

    return pd.DataFrame(rows)


def summarize_predictions(df: pd.DataFrame) -> dict:
    summary = {
        "n_pairs": len(df),
        "overall_accuracy": df["model_correct"].mean(),
        "model_prefers_generated_rate": df["model_prefers_generated"].mean(),
        "human_fooled_rate": df["human_fooled"].mean(),
        "model_agrees_with_human_rate": df["model_agrees_with_human"].mean(),
        "mean_reward_real": df["reward_real"].mean(),
        "mean_reward_generated": df["reward_generated"].mean(),
        "mean_score_diff": df["score_diff"].mean(),
        "mean_abs_score_diff": df["score_diff"].abs().mean(),
    }

    summary["by_split"] = (
        df.groupby("split")
        .agg(
            n=("model_correct", "size"),
            accuracy=("model_correct", "mean"),
            agree_human=("model_agrees_with_human", "mean"),
            fooled_rate=("human_fooled", "mean"),
            pref_gen_rate=("model_prefers_generated", "mean"),
            mean_reward_real=("reward_real", "mean"),
            mean_reward_generated=("reward_generated", "mean"),
        )
        .round(4)
        .to_dict("index")
    )

    summary["by_human_fooled"] = (
        df.groupby("human_fooled")
        .agg(
            n=("model_correct", "size"),
            accuracy=("model_correct", "mean"),
            mean_score_diff=("score_diff", "mean"),
            mean_prob_preferred=("prob_preferred", "mean"),
        )
        .round(4)
        .to_dict("index")
    )

    summary["by_biochemistry_level"] = (
        df.groupby("biochemistry_level")
        .agg(
            n=("model_correct", "size"),
            accuracy=("model_correct", "mean"),
            fooled_rate=("human_fooled", "mean"),
            pref_gen_rate=("model_prefers_generated", "mean"),
        )
        .round(4)
        .to_dict("index")
    )

    summary["by_expert_weight"] = (
        df.groupby("expert_weight")
        .agg(
            n=("model_correct", "size"),
            accuracy=("model_correct", "mean"),
            fooled_rate=("human_fooled", "mean"),
        )
        .round(4)
        .to_dict("index")
    )

    return summary


def _save_hist(
    ax,
    values,
    title: str,
    xlabel: str,
    bins: int = 40,
    vline: float | None = None,
):
    ax.hist(values, bins=bins, color="#4C72B0", alpha=0.85, edgecolor="white")
    if vline is not None:
        ax.axvline(vline, color="#C44E52", linestyle="--", linewidth=1.5, label=f"{vline:.2f}")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")


def plot_analysis(df: pd.DataFrame, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Score diff overall, colored by correct/incorrect
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, subset, color in [
        ("Correct", df[df["model_correct"] == 1], "#55A868"),
        ("Wrong", df[df["model_correct"] == 0], "#C44E52"),
    ]:
        ax.hist(subset["score_diff"], bins=40, alpha=0.6, label=f"{label} (n={len(subset)})", color=color)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Preferred − rejected reward (Bradley-Terry margin)")
    ax.set_xlabel("score_diff")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "hist_score_diff_correctness.png", dpi=160)
    plt.close(fig)

    # 2. Real vs generated reward distributions
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["reward_real"], bins=40, alpha=0.6, label="Real", color="#4C72B0")
    ax.hist(df["reward_generated"], bins=40, alpha=0.6, label="Generated", color="#DD8452")
    ax.set_title("Reward distribution: real vs generated molecules")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "hist_reward_real_vs_generated.png", dpi=160)
    plt.close(fig)

    # 3. By human_fooled
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, fooled, title in zip(
        axes,
        [0, 1],
        ["Human NOT fooled (preferred real)", "Human fooled (preferred generated)"],
    ):
        subset = df[df["human_fooled"] == fooled]
        ax.hist(subset["score_diff"], bins=30, color="#8172B3", alpha=0.85, edgecolor="white")
        acc = subset["model_correct"].mean() if len(subset) else float("nan")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{title}\naccuracy={acc:.3f}, n={len(subset)}")
        ax.set_xlabel("score_diff (preferred − rejected)")
        ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "hist_score_diff_by_human_fooled.png", dpi=160)
    plt.close(fig)

    # 4. Real vs generated reward by human_fooled
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, fooled, title in zip(axes, [0, 1], ["Human picked real", "Human picked generated"]):
        subset = df[df["human_fooled"] == fooled]
        ax.hist(subset["reward_real"], bins=30, alpha=0.6, label="Real", color="#4C72B0")
        ax.hist(subset["reward_generated"], bins=30, alpha=0.6, label="Generated", color="#DD8452")
        ax.set_title(f"{title} (n={len(subset)})")
        ax.set_xlabel("Reward")
        ax.set_ylabel("Count")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "hist_reward_by_human_fooled.png", dpi=160)
    plt.close(fig)

    # 5. By biochemistry level
    levels = sorted(df["biochemistry_level"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    for ax, level in zip(axes, levels):
        subset = df[df["biochemistry_level"] == level]
        ax.hist(subset["score_diff"], bins=25, color="#55A868", alpha=0.85, edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        acc = subset["model_correct"].mean()
        ax.set_title(f"{level}: accuracy={acc:.3f}, n={len(subset)}")
        ax.set_xlabel("score_diff")
        ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "hist_score_diff_by_biochemistry_level.png", dpi=160)
    plt.close(fig)

    # 6. By split
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, split_name in zip(axes, ["train", "val", "test"]):
        subset = df[df["split"] == split_name]
        ax.hist(subset["score_diff"], bins=30, alpha=0.85, color="#4C72B0", edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        acc = subset["model_correct"].mean() if len(subset) else float("nan")
        ax.set_title(f"{split_name}: acc={acc:.3f}, n={len(subset)}")
        ax.set_xlabel("score_diff")
    axes[0].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "hist_score_diff_by_split.png", dpi=160)
    plt.close(fig)

    # 7. Margin real - generated
    fig, ax = plt.subplots(figsize=(8, 5))
    for fooled, color, label in [(0, "#4C72B0", "Human picked real"), (1, "#DD8452", "Human picked generated")]:
        subset = df[df["human_fooled"] == fooled]
        ax.hist(
            subset["reward_margin_real_vs_gen"],
            bins=35,
            alpha=0.6,
            color=color,
            label=f"{label} (n={len(subset)})",
        )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Reward(real) − reward(generated)")
    ax.set_xlabel("Margin")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "hist_margin_real_minus_generated.png", dpi=160)
    plt.close(fig)

    # 8. Confusion-style bar chart
    confusion = pd.crosstab(
        df["human_prefers_generated"],
        df["model_prefers_generated"],
        rownames=["Human prefers gen"],
        colnames=["Model prefers gen"],
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion.values, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Model: real", "Model: gen"])
    ax.set_yticks([0, 1], labels=["Human: real", "Human: gen"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, confusion.values[i, j], ha="center", va="center", color="black")
    ax.set_title("Human vs model preference")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_human_vs_model.png", dpi=160)
    plt.close(fig)

    # 9. Accuracy bar chart by category
    categories = {
        "Overall": df,
        "Human fooled": df[df["human_fooled"] == 1],
        "Human not fooled": df[df["human_fooled"] == 0],
        "Train": df[df["split"] == "train"],
        "Val": df[df["split"] == "val"],
        "Test": df[df["split"] == "test"],
    }
    for level in sorted(df["biochemistry_level"].unique()):
        categories[f"Level {level}"] = df[df["biochemistry_level"] == level]

    names = list(categories.keys())
    accs = [categories[k]["model_correct"].mean() for k in names]
    counts = [len(categories[k]) for k in names]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(names, accs, color="#4C72B0", alpha=0.85)
    ax.axhline(0.5, color="#C44E52", linestyle="--", label="Random baseline (50%)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Bradley-Terry pairwise accuracy by category")
    ax.tick_params(axis="x", rotation=35)
    for bar, acc, n in zip(bars, accs, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f"{acc:.2f}\n(n={n})", ha="center", va="bottom", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "bar_accuracy_by_category.png", dpi=160)
    plt.close(fig)


def run_analysis(
    checkpoint_path: str | Path,
    data_csv: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    history_json: str | Path | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_predictions(checkpoint_path, data_csv, cache_dir, history_json)
    df.to_csv(output_dir / "predictions.csv", index=False)

    summary = summarize_predictions(df)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    plot_analysis(df, output_dir)
    fooled_summary = generate_fooled_report(df, output_dir / "fooled_eval")
    summary["fooled_eval"] = fooled_summary
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return df, summary
