"""Morgan fingerprint helpers for fingerprint-aware reward models."""

from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator
from torch import Tensor

MORGAN_BITS = 2048
MORGAN_RADIUS = 3

_FP_CACHE: dict[str, Tensor] = {}
_MORGAN_GENERATOR = None


def _generator():
    global _MORGAN_GENERATOR
    if _MORGAN_GENERATOR is None:
        _MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
            radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
        )
    return _MORGAN_GENERATOR


def compute_morgan_fp(
    smiles: str | None,
    n_bits: int = MORGAN_BITS,
    radius: int = MORGAN_RADIUS,
) -> Tensor:
    """Return a float32 bit vector of size n_bits (zeros if SMILES is invalid)."""
    if not smiles:
        return torch.zeros(n_bits, dtype=torch.float32)
    cached = _FP_CACHE.get(smiles)
    if cached is not None and cached.numel() == n_bits:
        return cached

    mol = Chem.MolFromSmiles(smiles)
    arr = np.zeros((n_bits,), dtype=np.float32)
    if mol is not None:
        try:
            if n_bits == MORGAN_BITS and radius == MORGAN_RADIUS:
                fp = _generator().GetFingerprint(mol)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            DataStructs.ConvertToNumpyArray(fp, arr)
        except Exception:
            arr[:] = 0.0

    tensor = torch.from_numpy(arr)
    _FP_CACHE[smiles] = tensor
    return tensor


def morgan_fp_from_data(data, n_bits: int = MORGAN_BITS, radius: int = MORGAN_RADIUS) -> Tensor:
    """Fetch or compute a batched [1, n_bits] fingerprint from a PyG Data object."""
    existing = getattr(data, "morgan_fp", None)
    if existing is not None:
        tensor = existing if torch.is_tensor(existing) else torch.tensor(existing, dtype=torch.float32)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.to(device=data.x.device, dtype=torch.float32)

    smiles = getattr(data, "smiles", None)
    if isinstance(smiles, (list, tuple)):
        smiles = smiles[0] if smiles else None
    fp = compute_morgan_fp(smiles, n_bits=n_bits, radius=radius)
    return fp.unsqueeze(0).to(device=data.x.device, dtype=torch.float32)
