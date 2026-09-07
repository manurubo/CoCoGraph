"""Dataset and preprocessing for human preference pairs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import MAX_ATOMS, MIN_ATOMS, SEED, TRAIN_FRAC, VAL_FRAC
from .featurization import smiles_to_pyg_data


@dataclass
class PreferenceExample:
    example_id: int
    preferred_smiles: str
    rejected_smiles: str
    weight: float
    player_id: int
    biochemistry_level: str
    human_fooled: int


def load_preference_dataframe(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"real_smiles", "generated_smiles", "human_fooled", "expert_weight", "player_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in preference CSV: {missing}")

    df = df[df["real_smiles"] != df["generated_smiles"]].copy()
    df["preferred_smiles"] = df.apply(
        lambda row: row["generated_smiles"] if row["human_fooled"] == 1 else row["real_smiles"],
        axis=1,
    )
    df["rejected_smiles"] = df.apply(
        lambda row: row["real_smiles"] if row["human_fooled"] == 1 else row["generated_smiles"],
        axis=1,
    )
    df["weight"] = df["expert_weight"].astype(float)
    return df


def split_by_player(df: pd.DataFrame, seed: int = SEED, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    players = sorted(df["player_id"].unique())
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(players), generator=rng).tolist()
    shuffled = [players[i] for i in perm]

    n_train = int(len(shuffled) * train_frac)
    n_val = int(len(shuffled) * val_frac)
    train_players = set(shuffled[:n_train])
    val_players = set(shuffled[n_train : n_train + n_val])
    test_players = set(shuffled[n_train + n_val :])

    splits = {}
    for name, player_set in [("train", train_players), ("val", val_players), ("test", test_players)]:
        part = df[df["player_id"].isin(player_set)].reset_index(drop=True)
        splits[name] = part
    return splits, {"train": train_players, "val": val_players, "test": test_players}


def _cache_key(smiles: str) -> str:
    return hashlib.sha1(smiles.encode("utf-8")).hexdigest()


class FeatureCache:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / "index.json"
        self.index = self._load_index()

    def _load_index(self) -> dict:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text())
        return {}

    def _save_index(self):
        self.index_path.write_text(json.dumps(self.index, indent=2))

    def get(self, smiles: str):
        key = _cache_key(smiles)
        path = self.cache_dir / f"{key}.pt"
        if path.exists():
            return torch.load(path, weights_only=False)
        return None

    def _is_external_cache(self) -> bool:
        """Refuse writes when the cache lives outside this explore package."""
        explore_root = Path(__file__).resolve().parents[1]
        try:
            self.cache_dir.resolve().relative_to(explore_root.resolve())
            return False
        except ValueError:
            return True

    def put(self, smiles: str, data):
        if self._is_external_cache():
            return
        key = _cache_key(smiles)
        path = self.cache_dir / f"{key}.pt"
        torch.save(data, path)
        self.index[smiles] = str(path.name)
        self._save_index()


def build_feature_cache(
    smiles_list: list[str],
    cache_dir: str | Path,
    min_atoms: int = MIN_ATOMS,
    max_atoms: int = MAX_ATOMS,
) -> dict[str, int]:
    cache = FeatureCache(cache_dir)
    stats = {"cached": 0, "computed": 0, "failed": 0}

    unique_smiles = sorted(set(smiles_list))
    for smiles in unique_smiles:
        if cache.get(smiles) is not None:
            stats["cached"] += 1
            continue
        data = smiles_to_pyg_data(smiles, min_atoms=min_atoms, max_atoms=max_atoms)
        if data is None:
            stats["failed"] += 1
            continue
        cache.put(smiles, data)
        stats["computed"] += 1
    return stats


class PreferencePairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cache_dir: str | Path):
        self.cache = FeatureCache(cache_dir)
        self.examples: list[PreferenceExample] = []

        for row in df.itertuples(index=False):
            preferred = row.preferred_smiles
            rejected = row.rejected_smiles
            if self.cache.get(preferred) is None or self.cache.get(rejected) is None:
                continue
            self.examples.append(
                PreferenceExample(
                    example_id=int(row.id) if hasattr(row, "id") else len(self.examples),
                    preferred_smiles=preferred,
                    rejected_smiles=rejected,
                    weight=float(row.weight),
                    player_id=int(row.player_id),
                    biochemistry_level=str(row.biochemistry_level),
                    human_fooled=int(row.human_fooled),
                )
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        preferred = self.cache.get(ex.preferred_smiles)
        rejected = self.cache.get(ex.rejected_smiles)
        if preferred is not None and getattr(preferred, "smiles", None) is None:
            preferred.smiles = ex.preferred_smiles
        if rejected is not None and getattr(rejected, "smiles", None) is None:
            rejected.smiles = ex.rejected_smiles
        return {
            "preferred": preferred,
            "rejected": rejected,
            "weight": torch.tensor(ex.weight, dtype=torch.float32),
            "example_id": ex.example_id,
            "index": idx,
            "human_fooled": ex.human_fooled,
            "player_id": ex.player_id,
            "biochemistry_level": ex.biochemistry_level,
        }

    def get_sampler_weights(self, fooled_multiplier: float = 1.0) -> torch.Tensor:
        """Per-example weights for training sampler (oversample fooled hard pairs)."""
        weights = []
        for ex in self.examples:
            w = ex.weight
            if ex.human_fooled == 1:
                w *= fooled_multiplier
            weights.append(w)
        return torch.tensor(weights, dtype=torch.float64)


def preference_collate(batch):
    preferred = [item["preferred"] for item in batch]
    rejected = [item["rejected"] for item in batch]
    weights = torch.stack([item["weight"] for item in batch])
    human_fooled = torch.tensor([item["human_fooled"] for item in batch], dtype=torch.float32)
    player_ids = torch.tensor([item["player_id"] for item in batch], dtype=torch.long)
    indices = torch.tensor([item["index"] for item in batch], dtype=torch.long)
    return {
        "preferred": preferred,
        "rejected": rejected,
        "weights": weights,
        "human_fooled": human_fooled,
        "player_ids": player_ids,
        "indices": indices,
    }


def score_graphs(model, graphs, device):
    rewards = []
    for graph in graphs:
        graph = graph.to(device)
        graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
        out = model(graph)
        if isinstance(out, tuple):
            out = out[0]
        rewards.append(out.reshape(()))
    if not rewards:
        return torch.empty(0, device=device)
    return torch.stack(rewards)


def encode_graphs(model, graphs, device):
    embeddings = []
    for graph in graphs:
        graph = graph.to(device)
        graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
        embeddings.append(model.encode_fused(graph))
    if not embeddings:
        return torch.empty(0, device=device)
    return torch.cat(embeddings, dim=0)
