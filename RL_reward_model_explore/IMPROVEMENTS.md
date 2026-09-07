# Reward model v2 improvements (2026-07-08)

## Baseline issues (`full_run_weighted`)

- Overall pairwise accuracy ~70%, but **fooled accuracy ~49%** (random on hard cases).
- Strong bias toward ranking real > generated (`mean_reward_real` 2.5 vs `generated` 1.9).
- Checkpoint selection optimized val **loss**, not fooled accuracy.
- Training used uniform shuffle — fooled pairs underrepresented per epoch.
- Encoder trained from scratch despite compatible pretrained `GINETimePredictor` weights.

## Changes

| Area | Change |
|------|--------|
| **Pair ordering / sampling** | `WeightedRandomSampler` oversamples fooled pairs (`FOOLED_SAMPLER_WEIGHT=3`) |
| **Loss** | Bradley-Terry + fooled hinge margin (`FOOLED_MARGIN=0.15`) |
| **Architecture** | Dropout in reward head; encoder init from `GINETimePredictor` |
| **Training** | Batched PyG forward; warmup+cosine LR; grad clip; early stop on val composite (65% fooled acc + 35% overall) |
| **Hyperparams** | LR 5e-4, fooled loss weight 4.0, 80 epochs max, weight decay 1e-4 |

## Reproduce improved training

```bash
cd RL_reward_model
python scripts/train_reward_model.py \
  --output-dir outputs/v2_improved_run \
  --cache-dir outputs/smoke_test/feature_cache \
  --epochs 80

python scripts/analyze_predictions.py \
  --checkpoint outputs/v2_improved_run/best_reward_model.pt \
  --cache-dir outputs/smoke_test/feature_cache \
  --history-json outputs/v2_improved_run/training_history.json \
  --output-dir outputs/v2_improved_run/analysis
```
