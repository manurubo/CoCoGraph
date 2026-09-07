"""Inference helpers for trained reward models."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import DEVICE, MIN_ATOMS, MAX_ATOMS
from .featurization import smiles_to_pyg_data
from .model import GINRewardModel, build_reward_model


def _remap_legacy_reward_head(state_dict: dict) -> dict:
    """Map v1 heads (Linear at index 3, no Dropout) onto the current Sequential."""
    if "reward_head.3.weight" in state_dict and "reward_head.4.weight" not in state_dict:
        remapped = dict(state_dict)
        remapped["reward_head.4.weight"] = remapped.pop("reward_head.3.weight")
        remapped["reward_head.4.bias"] = remapped.pop("reward_head.3.bias")
        return remapped
    return state_dict


def load_reward_model(checkpoint_path: str | Path, device: torch.device = DEVICE) -> GINRewardModel:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config") or {}
    model_type = config.get("model_type", "gin")
    n_player_buckets = int(config.get("n_player_buckets", 16))
    model = build_reward_model(model_type, n_player_buckets=n_player_buckets).to(device)
    state = _remap_legacy_reward_head(checkpoint["model_state_dict"])
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def score_smiles(model: GINRewardModel, smiles: str, device: torch.device = DEVICE) -> float | None:
    data = smiles_to_pyg_data(smiles, min_atoms=MIN_ATOMS, max_atoms=MAX_ATOMS)
    if data is None:
        return None
    batch = data.to(device)
    batch.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    return float(model(batch).item())


@torch.no_grad()
def compare_smiles(
    model: GINRewardModel,
    smiles_a: str,
    smiles_b: str,
    device: torch.device = DEVICE,
):
    score_a = score_smiles(model, smiles_a, device)
    score_b = score_smiles(model, smiles_b, device)
    if score_a is None or score_b is None:
        return None
    preferred = "A" if score_a > score_b else "B"
    prob_a = torch.sigmoid(torch.tensor(score_a - score_b)).item()
    return {
        "score_a": score_a,
        "score_b": score_b,
        "preferred": preferred,
        "prob_a_beats_b": prob_a,
    }


def main():
    parser = argparse.ArgumentParser(description="Score SMILES with a trained reward model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--smiles", type=str, nargs="+", required=True)
    args = parser.parse_args()

    model = load_reward_model(args.checkpoint, DEVICE)
    if len(args.smiles) == 1:
        score = score_smiles(model, args.smiles[0], DEVICE)
        print(f"{args.smiles[0]} -> reward={score}")
    else:
        result = compare_smiles(model, args.smiles[0], args.smiles[1], DEVICE)
        print(result)


if __name__ == "__main__":
    main()
