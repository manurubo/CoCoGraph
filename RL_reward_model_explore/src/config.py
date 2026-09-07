"""Configuration for the human-preference reward model."""

from pathlib import Path

import torch

# Graph / featurization (matches CoCoGraph time model)
MAX_ATOM = 70
NNFEAT = 31
NGFEAT = 21
NHEAD = 1
ENCEL = ["B", "N", "C", "O", "F", "P", "S", "Cl", "Br", "I", "Ca", "K", "Na", "Mg", "H"]
CYCLE_MAX = 14

# Training defaults
SEED = 42
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
BATCH_SIZE = 64
NUM_EPOCHS = 80
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
LR_GAMMA = 0.98
WARMUP_EPOCHS = 3
MIN_ATOMS = 5
MAX_ATOMS = MAX_ATOM
GRAD_CLIP_NORM = 1.0
EARLY_STOP_PATIENCE = 15

# Extra loss weight when human was fooled (preferred generated over real)
FOOLED_LOSS_WEIGHT = 4.0
# Training sampler oversampling for fooled pairs (pair ordering / hard-case focus)
FOOLED_SAMPLER_WEIGHT = 3.0
# Hinge margin on fooled pairs to push preferred above rejected
FOOLED_MARGIN = 0.15
FOOLED_MARGIN_WEIGHT = 0.5

# Checkpoint selection: weight fooled val accuracy vs overall val accuracy
VAL_FOOLED_METRIC_WEIGHT = 0.65

# Default pretrained CoCoGraph time-model encoder (shared GINE backbone)
DEFAULT_TIME_CHECKPOINT = (
    Path(__file__).resolve().parents[2] / "models" / "BASE_time" / "model_epoch_2_slice_22.pth"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
