#!/usr/bin/env python3
"""Turing sets: GuacaMol-style KL + PubChem KL + signed property shifts.

1) Pairwise set-vs-set KL (unsigned): chosen vs rejected, etc.
2) GuacaMol-style KL vs PubChem baseline for each set (unsigned KL, signed ΔKL).
3) Signed mean property diffs: mean(A) - mean(B) including internal_similarity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from guacamol.utils.chemistry import (
    calculate_internal_pairwise_similarities,
    calculate_pc_descriptors,
    canonicalize_list,
    continuous_kldiv,
    discrete_kldiv,
)

PC_DESCRIPTORS = [
    "BertzCT",
    "MolLogP",
    "MolWt",
    "TPSA",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "NumAliphaticRings",
    "NumAromaticRings",
]
ALL_PROPS = PC_DESCRIPTORS + ["internal_similarity"]


def unique_valid(smiles: list[str]) -> list[str]:
    return canonicalize_list(
        [s for s in smiles if isinstance(s, str) and s.strip()],
        include_stereocenters=False,
    )


def subsample(smiles: list[str], n: int, seed: int) -> list[str]:
    if len(smiles) <= n:
        return list(smiles)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(smiles), size=n, replace=False)
    return [smiles[i] for i in idx]


def load_smiles_file(path: Path, max_lines: int | None = None) -> list[str]:
    out: list[str] = []
    with path.open() as f:
        for i, line in enumerate(f):
            s = line.strip().split()[-1] if line.strip() else ""
            if s:
                out.append(s)
            if max_lines is not None and i + 1 >= max_lines:
                break
    return out


def feat_bundle(smiles: list[str]) -> dict:
    """Descriptors + max internal pairwise similarity per molecule."""
    d = calculate_pc_descriptors(smiles, PC_DESCRIPTORS)
    sim = calculate_internal_pairwise_similarities(smiles).max(axis=1)
    means = {PC_DESCRIPTORS[i]: float(np.mean(d[:, i])) for i in range(len(PC_DESCRIPTORS))}
    means["internal_similarity"] = float(np.mean(sim))
    return {"smiles": smiles, "desc": d, "sim": sim, "means": means, "n": len(smiles)}


def kl_from_feats(base: dict, samp: dict) -> tuple[float, dict[str, float]]:
    kldivs: dict[str, float] = {}
    for i in range(4):
        kldivs[PC_DESCRIPTORS[i]] = continuous_kldiv(base["desc"][:, i], samp["desc"][:, i])
    for i in range(4, 9):
        kldivs[PC_DESCRIPTORS[i]] = discrete_kldiv(base["desc"][:, i], samp["desc"][:, i])
    kldivs["internal_similarity"] = continuous_kldiv(base["sim"], samp["sim"])
    score = float(np.mean([np.exp(-v) for v in kldivs.values()]))
    return score, kldivs


def build_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    chosen, rejected = [], []
    for _, row in df.iterrows():
        left, right = row["molecule1_smiles"], row["molecule2_smiles"]
        if row["human_choice"] == "left":
            chosen.append(left)
            rejected.append(right)
        else:
            chosen.append(right)
            rejected.append(left)

    return {
        "chosen": unique_valid(chosen),
        "rejected": unique_valid(rejected),
        "fooled_gen": unique_valid(df.loc[df["human_fooled"] == 1, "generated_smiles"].tolist()),
        "caught_gen": unique_valid(df.loc[df["human_fooled"] == 0, "generated_smiles"].tolist()),
        "real": unique_valid(df["real_smiles"].tolist()),
        "generated": unique_valid(df["generated_smiles"].tolist()),
    }


def save_delta_bar(
    props: list[str],
    values: list[float],
    title: str,
    xlabel: str,
    out_png: Path,
    zero_line: bool = True,
) -> None:
    colors = ["#2A9D8F" if v < 0 else "#E76F51" for v in values]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(props[::-1], values[::-1], color=colors[::-1])
    if zero_line:
        ax.axvline(0.0, color="black", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--turing-csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "turing_guess_level.csv",
    )
    parser.add_argument(
        "--pubchem-file",
        type=Path,
        default=repo / "Data" / "CID-SMILES-filtered-lt70_0_notraining.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "outputs"
        / "turing_kl_chosen_vs_rejected",
    )
    parser.add_argument("--max-n", type=int, default=4000, help="Cap per Turing set")
    parser.add_argument("--pubchem-n", type=int, default=10000, help="PubChem baseline size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.turing_csv)
    sets = build_sets(df)
    print("Set sizes (unique valid canonical SMILES):")
    for k, v in sets.items():
        print(f"  {k:12s} {len(v)}")

    # Subsample each Turing set once (shared across analyses)
    used: dict[str, list[str]] = {}
    for name, smiles in sets.items():
        used[name] = subsample(smiles, args.max_n, args.seed + hash(name) % 1000)
        print(f"  using {name:12s} n={len(used[name])}")

    print("\nComputing features for Turing sets ...", flush=True)
    feats = {name: feat_bundle(smiles) for name, smiles in used.items()}

    # --- 1) Pairwise set-vs-set KL (unsigned) ---
    pair_defs = [
        ("chosen_vs_rejected", "rejected", "chosen"),
        ("fooled_gen_vs_caught_gen", "caught_gen", "fooled_gen"),
        ("generated_vs_real", "real", "generated"),
    ]
    pair_rows = []
    print("\n=== Pairwise set-vs-set KL (unsigned) ===")
    for cname, base_n, samp_n in pair_defs:
        # equalize n within the pair for fair KL
        n = min(feats[base_n]["n"], feats[samp_n]["n"])
        base_s = subsample(used[base_n], n, args.seed + 11)
        samp_s = subsample(used[samp_n], n, args.seed + 12)
        base_f, samp_f = feat_bundle(base_s), feat_bundle(samp_s)
        score, kldivs = kl_from_feats(base_f, samp_f)
        print(f"{cname}: score={100*score:.2f} n={n}")
        for p in ALL_PROPS:
            print(f"  {p:22s} kl={kldivs[p]:.4f}")
            pair_rows.append({"comparison": cname, "property": p, "kl": kldivs[p], "contrib": float(np.exp(-kldivs[p])), "n": n, "score": score})
    pd.DataFrame(pair_rows).to_csv(args.output_dir / "pairwise_kl.csv", index=False)

    # --- 2) Signed mean shifts ---
    print("\n=== Signed mean diffs (mean(sampled) - mean(baseline)) ===")
    mean_rows = []
    for cname, base_n, samp_n in pair_defs:
        print(f"{cname}:")
        for p in ALL_PROPS:
            delta = feats[samp_n]["means"][p] - feats[base_n]["means"][p]
            mean_rows.append(
                {
                    "comparison": cname,
                    "property": p,
                    "mean_baseline": feats[base_n]["means"][p],
                    "mean_sampled": feats[samp_n]["means"][p],
                    "delta_mean": delta,
                    "n_baseline": feats[base_n]["n"],
                    "n_sampled": feats[samp_n]["n"],
                }
            )
            print(
                f"  {p:22s} base={feats[base_n]['means'][p]:.4f}  "
                f"samp={feats[samp_n]['means'][p]:.4f}  Δ={delta:+.4f}"
            )
        deltas = [r["delta_mean"] for r in mean_rows if r["comparison"] == cname]
        save_delta_bar(
            ALL_PROPS,
            deltas,
            title=f"{cname}: mean(sampled)−mean(baseline)",
            xlabel="Signed mean difference",
            out_png=args.output_dir / f"signed_mean_{cname}.png",
        )
    pd.DataFrame(mean_rows).to_csv(args.output_dir / "signed_mean_diffs.csv", index=False)

    # --- 3) PubChem KL ---
    print(f"\nLoading PubChem from {args.pubchem_file} ...", flush=True)
    # Read more than needed then subsample (file is 1M lines)
    raw_pc = load_smiles_file(args.pubchem_file, max_lines=args.pubchem_n * 5)
    pc = subsample(unique_valid(raw_pc), args.pubchem_n, args.seed)
    print(f"PubChem baseline n={len(pc)}", flush=True)
    print("Computing PubChem features (internal_similarity is the slow step) ...", flush=True)
    pc_feats = feat_bundle(pc)

    print("\n=== GuacaMol-style KL vs PubChem ===")
    pub_rows = []
    set_scores: dict[str, float] = {}
    set_kls: dict[str, dict[str, float]] = {}
    for name in ["chosen", "rejected", "fooled_gen", "caught_gen", "real", "generated"]:
        score, kldivs = kl_from_feats(pc_feats, feats[name])
        set_scores[name] = score
        set_kls[name] = kldivs
        print(f"{name:12s} score={100*score:.2f}  n={feats[name]['n']}")
        for p in ALL_PROPS:
            pub_rows.append(
                {
                    "set": name,
                    "property": p,
                    "kl_vs_pubchem": kldivs[p],
                    "contrib": float(np.exp(-kldivs[p])),
                    "mean": feats[name]["means"][p],
                    "pubchem_mean": pc_feats["means"][p],
                    "delta_mean_vs_pubchem": feats[name]["means"][p] - pc_feats["means"][p],
                    "n": feats[name]["n"],
                    "score": score,
                }
            )
            print(f"  {p:22s} kl={kldivs[p]:.4f}  Δmean_vs_PC={feats[name]['means'][p]-pc_feats['means'][p]:+.4f}")
    pd.DataFrame(pub_rows).to_csv(args.output_dir / "kl_vs_pubchem.csv", index=False)

    # ΔKL between paired sets (signed: >0 => first set farther from PubChem)
    print("\n=== Signed ΔKL vs PubChem (KL(A)−KL(B); >0 => A farther from PubChem) ===")
    delta_defs = [
        ("chosen_minus_rejected", "chosen", "rejected"),
        ("fooled_minus_caught", "fooled_gen", "caught_gen"),
        ("generated_minus_real", "generated", "real"),
    ]
    dkl_rows = []
    for cname, a, b in delta_defs:
        d_score = set_scores[a] - set_scores[b]
        print(f"{cname}: score_A={100*set_scores[a]:.2f} score_B={100*set_scores[b]:.2f} Δscore={100*d_score:+.2f}")
        for p in ALL_PROPS:
            dkl = set_kls[a][p] - set_kls[b][p]
            dmean = feats[a]["means"][p] - feats[b]["means"][p]
            dkl_rows.append(
                {
                    "comparison": cname,
                    "property": p,
                    "kl_A": set_kls[a][p],
                    "kl_B": set_kls[b][p],
                    "delta_kl": dkl,
                    "mean_A": feats[a]["means"][p],
                    "mean_B": feats[b]["means"][p],
                    "delta_mean": dmean,
                    "score_A": set_scores[a],
                    "score_B": set_scores[b],
                    "delta_score": d_score,
                }
            )
            print(f"  {p:22s} Δkl={dkl:+.4f}  Δmean={dmean:+.4f}")
        dkls = [r["delta_kl"] for r in dkl_rows if r["comparison"] == cname]
        save_delta_bar(
            ALL_PROPS,
            dkls,
            title=f"{cname}: KL(A)−KL(B) vs PubChem",
            xlabel="ΔKL (>0 = A farther from PubChem)",
            out_png=args.output_dir / f"delta_kl_pubchem_{cname}.png",
        )
    pd.DataFrame(dkl_rows).to_csv(args.output_dir / "delta_kl_vs_pubchem.csv", index=False)

    # Summary markdown
    lines = [
        "# Turing KL analysis (pairwise + PubChem + signed means)",
        "",
        f"- Turing: `{args.turing_csv}`",
        f"- PubChem: `{args.pubchem_file}` (n={len(pc)})",
        f"- Turing cap: max_n={args.max_n}",
        "",
        "## Scores vs PubChem (higher = closer to PubChem)",
    ]
    for name in ["chosen", "rejected", "fooled_gen", "caught_gen", "real", "generated"]:
        lines.append(f"- {name}: {100*set_scores[name]:.2f}")
    lines += ["", "## Signed ΔKL vs PubChem (KL(A)−KL(B))"]
    for cname, a, b in delta_defs:
        sub = [r for r in dkl_rows if r["comparison"] == cname]
        lines.append(
            f"- **{cname}**: Δscore={100*sub[0]['delta_score']:+.2f} "
            f"(A={a} {100*sub[0]['score_A']:.2f}, B={b} {100*sub[0]['score_B']:.2f})"
        )
        ins = next(r for r in sub if r["property"] == "internal_similarity")
        lines.append(
            f"  - internal_similarity: Δkl={ins['delta_kl']:+.4f}, Δmean={ins['delta_mean']:+.4f}"
        )
    lines += [
        "",
        "## Signed mean diffs (chosen − rejected)",
    ]
    for r in mean_rows:
        if r["comparison"] == "chosen_vs_rejected":
            lines.append(f"- {r['property']}: Δmean={r['delta_mean']:+.4f}")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")

    summary = {
        "turing_csv": str(args.turing_csv),
        "pubchem_file": str(args.pubchem_file),
        "pubchem_n": len(pc),
        "max_n": args.max_n,
        "seed": args.seed,
        "set_sizes_pool": {k: len(v) for k, v in sets.items()},
        "set_sizes_used": {k: len(v) for k, v in used.items()},
        "scores_vs_pubchem": set_scores,
        "kl_vs_pubchem": set_kls,
        "pubchem_means": pc_feats["means"],
        "set_means": {k: v["means"] for k, v in feats.items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nWrote outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
