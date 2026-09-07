"""Analyze whether model errors correlate with molecular quality (RDKit descriptors)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors

OUTCOME_LABELS = {
    "easy_agree": "Human: real, Model: real (easy agree)",
    "deceptive_gen": "Human: real, Model: generated (model fooled by gen)",
    "missed_realism": "Human: generated, Model: real (model missed realism)",
    "caught_realism": "Human: generated, Model: generated (caught realism)",
}


def _safe_mol(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def compute_molecule_quality(smiles: str) -> dict:
    mol = _safe_mol(smiles)
    if mol is None:
        return {
            "valid": 0,
            "qed": np.nan,
            "sa_score": np.nan,
            "mw": np.nan,
            "logp": np.nan,
            "tpsa": np.nan,
            "hbd": np.nan,
            "hba": np.nan,
            "rotatable_bonds": np.nan,
            "aromatic_rings": np.nan,
            "fraction_csp3": np.nan,
            "lipinski_violations": np.nan,
        }

    try:
        from rdkit.Chem import RDConfig
        import os
        import sys

        sys.path.append(os.path.join(RDConfig.RDCodeDir, "SA_Score"))
        import sascorer

        sa = float(sascorer.calculateScore(mol))
    except Exception:
        sa = np.nan

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    violations = sum(
        [
            mw > 500,
            logp > 5,
            hbd > 5,
            hba > 10,
        ]
    )

    return {
        "valid": 1,
        "qed": float(QED.qed(mol)),
        "sa_score": sa,
        "mw": float(mw),
        "logp": float(logp),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hbd": int(hbd),
        "hba": int(hba),
        "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "lipinski_violations": int(violations),
    }


def assign_outcome(row: pd.Series) -> str:
    if row["human_fooled"] == 0 and row["model_correct"] == 1:
        return "easy_agree"
    if row["human_fooled"] == 0 and row["model_correct"] == 0:
        return "deceptive_gen"
    if row["human_fooled"] == 1 and row["model_correct"] == 0:
        return "missed_realism"
    return "caught_realism"


def enrich_with_quality(
    predictions_csv: str | Path,
    data_csv: str | Path,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_csv)
    source = pd.read_csv(data_csv)[["id", "real_smiles", "generated_smiles"]]
    df = predictions.merge(source, on="id", how="left")
    df["outcome"] = df.apply(assign_outcome, axis=1)

    cache_path = Path(cache_path) if cache_path else None
    quality_cache: dict[str, dict] = {}
    if cache_path and cache_path.exists():
        quality_cache = json.loads(cache_path.read_text())

    def get_quality(smiles: str, prefix: str) -> dict:
        if smiles not in quality_cache:
            quality_cache[smiles] = compute_molecule_quality(smiles)
        q = quality_cache[smiles]
        return {f"{prefix}_{k}": v for k, v in q.items()}

    rows = []
    for row in df.itertuples(index=False):
        record = row._asdict()
        record.update(get_quality(row.real_smiles, "real"))
        record.update(get_quality(row.generated_smiles, "generated"))
        record["delta_qed"] = record["generated_qed"] - record["real_qed"]
        record["delta_sa"] = record["generated_sa_score"] - record["real_sa_score"]
        record["generated_higher_qed"] = int(record["delta_qed"] > 0)
        record["generated_more_druglike"] = int(record["delta_qed"] > 0.05)
        rows.append(record)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(quality_cache, indent=2))

    return pd.DataFrame(rows)


def summarize_quality_by_outcome(df: pd.DataFrame) -> dict:
    metrics = [
        "real_qed",
        "generated_qed",
        "delta_qed",
        "real_sa_score",
        "generated_sa_score",
        "delta_sa",
        "real_mw",
        "generated_mw",
        "real_logp",
        "generated_logp",
    ]

    summary = {"n_by_outcome": df["outcome"].value_counts().to_dict(), "by_outcome": {}}
    for outcome, subset in df.groupby("outcome"):
        summary["by_outcome"][outcome] = {
            "label": OUTCOME_LABELS.get(outcome, outcome),
            "n": int(len(subset)),
            "means": {m: float(subset[m].mean()) for m in metrics if m in subset.columns},
            "generated_higher_qed_rate": float(subset["generated_higher_qed"].mean()),
            "generated_more_druglike_rate": float(subset["generated_more_druglike"].mean()),
        }
    return summary


def _markdown_report(summary: dict, model_name: str) -> str:
    lines = [
        f"# Quality vs Model/Human Outcome Report ({model_name})",
        "",
        "Outcome buckets:",
        "",
        "- **easy_agree**: human picked real, model picked real",
        "- **deceptive_gen**: human picked real, model picked generated (model wrong)",
        "- **missed_realism**: human picked generated, model picked real (model wrong on fooled)",
        "- **caught_realism**: human picked generated, model picked generated",
        "",
        "Quality metrics from RDKit: **QED** (drug-likeness, 0–1), **SA** (synthetic accessibility, lower=easier),",
        "plus MW, logP, etc.",
        "",
    ]

    for outcome, stats in summary["by_outcome"].items():
        means = stats["means"]
        lines.extend(
            [
                f"## {stats['label']} (n={stats['n']})",
                "",
                f"- Mean QED real: {means.get('real_qed', float('nan')):.3f}",
                f"- Mean QED generated: {means.get('generated_qed', float('nan')):.3f}",
                f"- Mean ΔQED (gen − real): {means.get('delta_qed', float('nan')):.3f}",
                f"- Generated has higher QED: {stats['generated_higher_qed_rate']:.1%}",
                f"- Mean SA real: {means.get('real_sa_score', float('nan')):.2f}",
                f"- Mean SA generated: {means.get('generated_sa_score', float('nan')):.2f}",
                f"- Mean ΔSA (gen − real): {means.get('delta_sa', float('nan')):.2f}",
                "",
            ]
        )

    lines.extend(
        [
            "## How to read this",
            "",
            "- If **deceptive_gen** cases have high generated QED, the model may be reacting to drug-likeness",
            "  rather than human realism.",
            "- If **missed_realism** cases have low generated QED, humans may have been fooled by molecules",
            "  that are not actually high-quality by cheminformatics standards (noise/guessing).",
            "- If **missed_realism** has high generated QED, the model is under-ranking genuinely good fakes.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_quality_analysis(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    order = ["easy_agree", "deceptive_gen", "missed_realism", "caught_realism"]
    colors = {"real": "#4C72B0", "generated": "#DD8452"}

    # 1. QED by outcome (real vs generated)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    for ax, outcome in zip(axes, order):
        subset = df[df["outcome"] == outcome]
        ax.hist(subset["real_qed"].dropna(), bins=25, alpha=0.6, color=colors["real"], label="Real")
        ax.hist(subset["generated_qed"].dropna(), bins=25, alpha=0.6, color=colors["generated"], label="Generated")
        ax.set_title(f"{OUTCOME_LABELS[outcome][:40]}...\nn={len(subset)}")
        ax.set_xlabel("QED")
        ax.legend(fontsize=8)
    fig.suptitle("QED distribution by outcome bucket", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "quality_qed_by_outcome.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 2. Delta QED by outcome
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [df[df["outcome"] == o]["delta_qed"].dropna() for o in order]
    ax.boxplot(data, labels=[o.replace("_", "\n") for o in order])
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("ΔQED (generated − real) by outcome")
    ax.set_ylabel("ΔQED")
    fig.tight_layout()
    fig.savefig(output_dir / "quality_delta_qed_by_outcome.png", dpi=160)
    plt.close(fig)

    # 3. Focus: model wrong buckets
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    focus = [
        ("deceptive_gen", "Model prefers generated\n(human picked real)"),
        ("missed_realism", "Model prefers real\n(human was fooled)"),
    ]
    for ax, (outcome, title) in zip(axes, focus):
        subset = df[df["outcome"] == outcome]
        ax.scatter(subset["real_qed"], subset["generated_qed"], alpha=0.35, s=16, c="#8172B3")
        lims = [0, 1]
        ax.plot(lims, lims, "k--", linewidth=1, label="QED gen = QED real")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("QED real")
        ax.set_ylabel("QED generated")
        ax.set_title(f"{title}\nn={len(subset)}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "quality_qed_scatter_model_errors.png", dpi=160)
    plt.close(fig)

    # 4. Bar chart of mean QED per outcome
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(order))
    width = 0.35
    real_means = [df[df["outcome"] == o]["real_qed"].mean() for o in order]
    gen_means = [df[df["outcome"] == o]["generated_qed"].mean() for o in order]
    ax.bar(x - width / 2, real_means, width, label="Real", color=colors["real"])
    ax.bar(x + width / 2, gen_means, width, label="Generated", color=colors["generated"])
    ax.set_xticks(x, [o.replace("_", "\n") for o in order], fontsize=9)
    ax.set_ylabel("Mean QED")
    ax.set_title("Mean drug-likeness (QED) by outcome")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "quality_mean_qed_by_outcome.png", dpi=160)
    plt.close(fig)

    # 5. SA score comparison for missed_realism vs caught_realism
    fig, ax = plt.subplots(figsize=(8, 5))
    for outcome, color in [("missed_realism", "#C44E52"), ("caught_realism", "#55A868")]:
        subset = df[df["outcome"] == outcome]
        ax.hist(
            subset["generated_sa_score"].dropna(),
            bins=25,
            alpha=0.55,
            label=f"{outcome} (n={len(subset)})",
            color=color,
        )
    ax.set_xlabel("SA score (generated molecule)")
    ax.set_title("Synthetic accessibility of generated mols:\nmodel missed vs caught realism")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "quality_sa_fooled_cases.png", dpi=160)
    plt.close(fig)


def run_quality_analysis(
    predictions_csv: str | Path,
    data_csv: str | Path,
    output_dir: str | Path,
    model_name: str = "model",
    cache_path: str | Path | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = enrich_with_quality(predictions_csv, data_csv, cache_path=cache_path)
    df.to_csv(output_dir / "predictions_with_quality.csv", index=False)

    summary = summarize_quality_by_outcome(df)
    (output_dir / "quality_summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "quality_report.md").write_text(_markdown_report(summary, model_name))

    plot_quality_analysis(df, output_dir)
    return df, summary
