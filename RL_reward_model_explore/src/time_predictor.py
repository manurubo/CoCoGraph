"""CoCoGraph time-model inference for molecule quality analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from .config import DEVICE
from .dataset import FeatureCache, score_graphs
from .featurization import smiles_to_pyg_data

# Use the parent CoCoGraph time model implementation + checkpoint.
COCOGRAPH_ROOT = Path(__file__).resolve().parents[2]
if str(COCOGRAPH_ROOT) not in sys.path:
    sys.path.insert(0, str(COCOGRAPH_ROOT))

from lib_functions.models import GINETimePredictor  # noqa: E402


DEFAULT_TIME_CHECKPOINT = COCOGRAPH_ROOT / "models" / "BASE_time" / "model_epoch_2_slice_22.pth"


def load_time_predictor(
    checkpoint_path: str | Path | None = None,
    device: torch.device = DEVICE,
) -> GINETimePredictor:
    checkpoint_path = Path(checkpoint_path or DEFAULT_TIME_CHECKPOINT)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Time model checkpoint not found: {checkpoint_path}")

    model = GINETimePredictor().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_time_from_data(model: GINETimePredictor, data, device: torch.device = DEVICE) -> float:
    data = data.to(device)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    return float(model(data).item())


@torch.no_grad()
def predict_time_smiles(
    model: GINETimePredictor,
    smiles: str,
    device: torch.device = DEVICE,
    cache: FeatureCache | None = None,
) -> float | None:
    if cache is not None:
        data = cache.get(smiles)
    else:
        data = smiles_to_pyg_data(smiles)
    if data is None:
        return None
    return predict_time_from_data(model, data, device)


@torch.no_grad()
def predict_time_batch_smiles(
    model: GINETimePredictor,
    smiles_list: list[str],
    device: torch.device = DEVICE,
    cache: FeatureCache | None = None,
) -> dict[str, float | None]:
    scores = {}
    for smiles in smiles_list:
        scores[smiles] = predict_time_smiles(model, smiles, device=device, cache=cache)
    return scores
