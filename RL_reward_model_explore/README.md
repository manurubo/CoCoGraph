# RL Reward Model

Self-contained human-preference reward model trained with **Bradley-Terry loss** on Turing-test data.

Everything lives in this directory: data, featurization, model, training, and inference.

## Architecture choice

Use the **time-model backbone** (`GINETimePredictor` style), **not** the main CoCoGraph generation model.

| Model | Role | Good for reward model? |
|---|---|---|
| `GINEdgeQuadrupletPredictor` | Predict edge edits during generation | No — too heavy, wrong output type |
| `GINETimePredictor` | Graph → scalar score | **Yes** — same encoder, scalar head |

This package implements `GINRewardModel`:

- 3× Enhanced GINE conv layers (same featurization as CoCoGraph)
- Global mean pooling
- Linear reward head (unbounded scalar, no sigmoid)
- Trained with Bradley-Terry pairwise loss

The reward model scores **single molecules**. During RL you compare candidates via `sigmoid(r(a) - r(b))`.

## Data

`data/turing_guess_level.csv` contains Turing-test comparisons:

- `real_smiles` vs `generated_smiles`
- `human_fooled = 1` → human preferred the generated molecule
- `human_fooled = 0` → human preferred the real molecule
- `expert_weight` → sample weight (GR=1.0, UG=0.8, HS/PH=0.5)

Pairs where both SMILES are identical are dropped.

Train/val/test split is **by player_id** to avoid leakage.

## Quick start

```bash
cd RL_reward_model
pip install -r requirements.txt

# Train (default: 80 epochs, v2 improvements — see IMPROVEMENTS.md)
python scripts/train_reward_model.py \
  --output-dir outputs/v2_improved_run \
  --cache-dir outputs/smoke_test/feature_cache \
  --epochs 80

# Score one molecule
python scripts/score_smiles.py \
  --checkpoint outputs/default_run/best_reward_model.pt \
  --smiles "CC(=O)Oc1ccccc1C(=O)O"

# Compare two molecules
python scripts/score_smiles.py \
  --checkpoint outputs/default_run/best_reward_model.pt \
  --smiles "CCO" "CCN"

# Full analysis + fooled eval report
python scripts/analyze_predictions.py \
  --checkpoint outputs/full_run/best_reward_model.pt \
  --output-dir outputs/full_run/analysis

# Dedicated fooled-only report
python scripts/eval_fooled_report.py \
  --checkpoint outputs/full_run/best_reward_model.pt \
  --output-dir outputs/full_run/fooled_eval
```

Training upweights fooled comparisons via `--fooled-weight` (default 2.5):

```bash
python scripts/train_reward_model.py \
  --fooled-weight 2.5 \
  --output-dir outputs/full_run_weighted
```

## Bradley-Terry loss

For preferred molecule `x` and rejected molecule `y`:

```
P(x ≻ y) = sigmoid(r(x) - r(y))
L = softplus(r(y) - r(x))
```

Weighted by `expert_weight` per comparison.

## Layout

```
RL_reward_model/
  data/                  # turing_guess_level.csv
  src/
    config.py            # hyperparameters
    graph_utils.py       # SMILES → NetworkX
    featurization.py     # NetworkX → PyG Data
    model.py             # GINRewardModel
    loss.py              # Bradley-Terry loss
    dataset.py           # preference pairs + feature cache
    train.py             # training loop
    inference.py         # scoring helpers
  scripts/
    train_reward_model.py
    score_smiles.py
  outputs/               # checkpoints + logs (created on train)
  checkpoints/           # optional manual checkpoint storage
```

## Notes

- Feature cache is stored under `outputs/<run>/feature_cache/` after first run.
- First epoch is slow while graphs are featurized; later epochs reuse the cache.
- For RL integration, load `best_reward_model.pt` and call `score_smiles()` or batch through `GINRewardModel`.

## Possible extensions

- **Morgan FP head** (`GINETimePredictor_MorganFP` style) if you want fingerprint-aware rewards.
- **Init from time-model weights** if you have a pretrained checkpoint.
- **Bradley-Terry with ties** if you add "can't tell" labels later.
