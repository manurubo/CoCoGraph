"""Training loop for the Bradley-Terry reward model (explore copy)."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import (
    BATCH_SIZE,
    DEFAULT_TIME_CHECKPOINT,
    DEVICE,
    EARLY_STOP_PATIENCE,
    FOOLED_LOSS_WEIGHT,
    FOOLED_MARGIN,
    FOOLED_MARGIN_WEIGHT,
    FOOLED_SAMPLER_WEIGHT,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    NUM_EPOCHS,
    SEED,
    VAL_FOOLED_METRIC_WEIGHT,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)
from .dataset import (
    FeatureCache,
    PreferencePairDataset,
    build_feature_cache,
    load_preference_dataframe,
    preference_collate,
    score_graphs,
    split_by_player,
)
from .inference import load_reward_model
from .loss import (
    bradley_terry_metrics,
    combine_sample_weights,
    distillation_loss,
    generation_prior_loss,
    irm_penalty,
    preference_loss,
    spread_hinge_loss,
)
from .model import build_reward_model, load_encoder_from_time_checkpoint

ORIGINAL_V1_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "RL_reward_model"
    / "outputs"
    / "full_run_weighted"
    / "best_reward_model.pt"
)
ORIGINAL_V2_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "RL_reward_model"
    / "outputs"
    / "v2_improved_run"
    / "best_reward_model.pt"
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_json_serializable(obj):
    if isinstance(obj, dict):
        return {key: to_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(value) for value in obj]
    if isinstance(obj, set):
        return sorted(to_json_serializable(value) for value in obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _safe_metric(value: float) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    return float(value)


def composite_val_score(metrics: dict, fooled_weight: float = VAL_FOOLED_METRIC_WEIGHT) -> float:
    """Select checkpoints by fooled accuracy first, overall accuracy second."""
    fooled_acc = _safe_metric(metrics.get("fooled_accuracy"))
    overall_acc = _safe_metric(metrics.get("accuracy"))
    return fooled_weight * fooled_acc + (1.0 - fooled_weight) * overall_acc


def balanced_val_score(metrics: dict) -> float:
    """More even checkpoint score: held-out overall + fooled + not-fooled."""
    return (
        0.40 * _safe_metric(metrics.get("accuracy"))
        + 0.30 * _safe_metric(metrics.get("fooled_accuracy"))
        + 0.30 * _safe_metric(metrics.get("not_fooled_accuracy"))
    )


def _set_lr(optimizer, lr: float):
    for group in optimizer.param_groups:
        group["lr"] = lr


def _warmup_cosine_lr(epoch: int, num_epochs: int, base_lr: float, warmup_epochs: int) -> float:
    if epoch <= warmup_epochs:
        return base_lr * epoch / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _player_buckets(player_ids: torch.Tensor, n_buckets: int) -> torch.Tensor:
    return player_ids % n_buckets


def _precompute_teacher_diffs(checkpoint: str | Path, dataset: PreferencePairDataset, device) -> torch.Tensor:
    teacher = load_reward_model(checkpoint, device)
    teacher.eval()
    diffs = torch.zeros(len(dataset), dtype=torch.float32)
    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            r_pref = score_graphs(teacher, [item["preferred"]], device)
            r_rej = score_graphs(teacher, [item["rejected"]], device)
            diffs[idx] = (r_pref - r_rej).item()
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return diffs


def _precompute_time_targets(smiles_list: list[str], cache_dir: Path, device) -> dict[str, float]:
    from .time_predictor import load_time_predictor, predict_time_from_data

    cache = FeatureCache(cache_dir)
    time_model = load_time_predictor(device=device)
    targets: dict[str, float] = {}
    with torch.no_grad():
        for smiles in smiles_list:
            data = cache.get(smiles)
            if data is None:
                continue
            targets[smiles] = predict_time_from_data(time_model, data, device)
    del time_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if targets:
        values = torch.tensor(list(targets.values()), dtype=torch.float32)
        mean = float(values.mean().item())
        std = float(values.std(unbiased=False).clamp_min(1e-6).item())
        targets = {smiles: (score - mean) / std for smiles, score in targets.items()}
        print(f"Standardized time targets: mean={mean:.4f} std={std:.4f} n={len(targets)}")
    return targets


def _time_targets_for_graphs(graphs, time_targets: dict[str, float], device) -> torch.Tensor:
    values = []
    for graph in graphs:
        smiles = getattr(graph, "smiles", None)
        if isinstance(smiles, (list, tuple)):
            smiles = smiles[0] if smiles else None
        values.append(float(time_targets.get(smiles, 0.0)))
    return torch.tensor(values, device=device, dtype=torch.float32)


def compute_batch_loss(
    model,
    batch: dict,
    device,
    fooled_weight: float,
    fooled_margin: float,
    fooled_margin_weight: float,
    not_fooled_margin: float,
    not_fooled_margin_weight: float,
    calibration_weight: float,
    calibration_target: float,
    spread_weight: float,
    spread_target: float,
    irm_weight: float,
    adversarial_weight: float,
    adversarial_lambda: float,
    n_player_buckets: int,
    aux_time_weight: float,
    time_targets: dict[str, float] | None,
    teacher_diffs: torch.Tensor | None,
    distill_weight: float,
    distill_temperature: float,
):
    preferred = batch["preferred"]
    rejected = batch["rejected"]
    expert_weights = batch["weights"].to(device)
    human_fooled = batch["human_fooled"].to(device)
    player_ids = batch["player_ids"].to(device)
    sample_weights = combine_sample_weights(
        expert_weights, human_fooled, fooled_multiplier=fooled_weight
    )

    r_pref = score_graphs(model, preferred, device)
    r_rej = score_graphs(model, rejected, device)
    loss = preference_loss(
        r_pref,
        r_rej,
        sample_weights,
        human_fooled=human_fooled,
        fooled_margin=fooled_margin,
        fooled_margin_weight=fooled_margin_weight,
        not_fooled_margin=not_fooled_margin,
        not_fooled_margin_weight=not_fooled_margin_weight,
    )

    extras = {}
    if calibration_weight > 0:
        cal = generation_prior_loss(r_pref, r_rej, human_fooled, target_rate=calibration_target)
        loss = loss + calibration_weight * cal
        extras["cal"] = float(cal.detach().item())

    if spread_weight > 0:
        spr = spread_hinge_loss(r_pref, r_rej, target_gap=spread_target)
        loss = loss + spread_weight * spr
        extras["spread"] = float(spr.detach().item())

    if irm_weight > 0:
        env_ids = _player_buckets(player_ids, n_player_buckets)
        irm = irm_penalty(r_pref, r_rej, env_ids, weights=sample_weights)
        loss = loss + irm_weight * irm
        extras["irm"] = float(irm.detach().item())

    if adversarial_weight > 0 and hasattr(model, "player_logits"):
        buckets = _player_buckets(player_ids, n_player_buckets)
        logits_pref = []
        logits_rej = []
        for graph in preferred:
            graph = graph.to(device)
            graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
            logits_pref.append(model.player_logits(graph, lambda_=adversarial_lambda))
        for graph in rejected:
            graph = graph.to(device)
            graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
            logits_rej.append(model.player_logits(graph, lambda_=adversarial_lambda))
        player_logits = torch.cat(logits_pref + logits_rej, dim=0)
        player_targets = torch.cat([buckets, buckets], dim=0)
        adv = F.cross_entropy(player_logits, player_targets)
        loss = loss + adversarial_weight * adv
        extras["adv"] = float(adv.detach().item())

    if aux_time_weight > 0 and time_targets is not None and hasattr(model, "predict_time"):
        t_pref = _time_targets_for_graphs(preferred, time_targets, device)
        t_rej = _time_targets_for_graphs(rejected, time_targets, device)
        pred_pref = torch.stack(
            [
                model.predict_time(_with_batch(g, device)).reshape(())
                for g in preferred
            ]
        )
        pred_rej = torch.stack(
            [
                model.predict_time(_with_batch(g, device)).reshape(())
                for g in rejected
            ]
        )
        aux = F.mse_loss(pred_pref, t_pref) + F.mse_loss(pred_rej, t_rej)
        loss = loss + aux_time_weight * aux
        extras["aux_time"] = float(aux.detach().item())

    if distill_weight > 0 and teacher_diffs is not None:
        teacher_logit = teacher_diffs[batch["indices"].to(device)]
        student_logit = r_pref - r_rej
        kd = distillation_loss(student_logit, teacher_logit, temperature=distill_temperature)
        loss = loss + distill_weight * kd
        extras["kd"] = float(kd.detach().item())

    extras["mean_abs_diff"] = float((r_pref - r_rej).abs().mean().detach().item())
    return loss, r_pref, r_rej, human_fooled, extras


def _with_batch(graph, device):
    graph = graph.to(device)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
    return graph


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    fooled_weight: float = FOOLED_LOSS_WEIGHT,
    fooled_margin: float = FOOLED_MARGIN,
    fooled_margin_weight: float = FOOLED_MARGIN_WEIGHT,
    not_fooled_margin: float = 0.0,
    not_fooled_margin_weight: float = 0.0,
    checkpoint_metric: str = "v2_composite",
):
    model.eval()
    total_loss = 0.0
    all_pref, all_rej, all_fooled = [], [], []
    n_batches = 0

    for batch in loader:
        sample_weights = combine_sample_weights(
            batch["weights"].to(device),
            batch["human_fooled"].to(device),
            fooled_multiplier=fooled_weight,
        )
        r_pref = score_graphs(model, batch["preferred"], device)
        r_rej = score_graphs(model, batch["rejected"], device)
        loss = preference_loss(
            r_pref,
            r_rej,
            sample_weights,
            human_fooled=batch["human_fooled"].to(device),
            fooled_margin=fooled_margin,
            fooled_margin_weight=fooled_margin_weight,
            not_fooled_margin=not_fooled_margin,
            not_fooled_margin_weight=not_fooled_margin_weight,
        )
        total_loss += loss.item()
        all_pref.append(r_pref)
        all_rej.append(r_rej)
        all_fooled.append(batch["human_fooled"].to(device))
        n_batches += 1

    if n_batches == 0:
        return {
            "loss": float("nan"),
            "accuracy": float("nan"),
            "fooled_accuracy": float("nan"),
            "not_fooled_accuracy": float("nan"),
        }

    metrics = bradley_terry_metrics(
        torch.cat(all_pref),
        torch.cat(all_rej),
        torch.cat(all_fooled),
    )
    metrics["loss"] = total_loss / n_batches
    metrics["mean_abs_diff"] = float((torch.cat(all_pref) - torch.cat(all_rej)).abs().mean().item())
    metrics["composite_score"] = composite_val_score(metrics)
    metrics["balanced_score"] = balanced_val_score(metrics)
    metrics["checkpoint_score"] = (
        metrics["balanced_score"] if checkpoint_metric == "balanced" else metrics["composite_score"]
    )
    return metrics


def train(
    data_csv: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path | None = None,
    batch_size: int = BATCH_SIZE,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    seed: int = SEED,
    fooled_weight: float = FOOLED_LOSS_WEIGHT,
    fooled_sampler_weight: float = FOOLED_SAMPLER_WEIGHT,
    fooled_margin: float = FOOLED_MARGIN,
    fooled_margin_weight: float = FOOLED_MARGIN_WEIGHT,
    warmup_epochs: int = WARMUP_EPOCHS,
    grad_clip_norm: float = GRAD_CLIP_NORM,
    early_stop_patience: int = EARLY_STOP_PATIENCE,
    init_time_checkpoint: str | Path | None = DEFAULT_TIME_CHECKPOINT,
    device: torch.device = DEVICE,
    model_type: str = "gin",
    freeze_encoder: bool = False,
    not_fooled_margin: float = 0.0,
    not_fooled_margin_weight: float = 0.0,
    calibration_weight: float = 0.0,
    calibration_target: float = 0.37,
    spread_weight: float = 0.0,
    spread_target: float = 0.3,
    irm_weight: float = 0.0,
    adversarial_weight: float = 0.0,
    n_player_buckets: int = 16,
    aux_time_weight: float = 0.0,
    distill_v1_checkpoint: str | Path | None = None,
    distill_v2_checkpoint: str | Path | None = None,
    distill_alpha: float = 0.5,
    distill_weight: float = 0.0,
    distill_temperature: float = 2.0,
    checkpoint_metric: str = "v2_composite",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir or output_dir / "feature_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    set_seed(seed)

    df = load_preference_dataframe(data_csv)
    splits, player_splits = split_by_player(df, seed=seed)

    all_smiles = pd.unique(
        pd.concat([splits["train"], splits["val"], splits["test"]])[["preferred_smiles", "rejected_smiles"]].values.ravel()
    ).tolist()
    cache_stats = build_feature_cache(all_smiles, cache_dir)
    print("Feature cache stats:", cache_stats)

    datasets = {name: PreferencePairDataset(part, cache_dir) for name, part in splits.items()}

    train_sampler = WeightedRandomSampler(
        weights=datasets["train"].get_sampler_weights(fooled_multiplier=fooled_sampler_weight),
        num_samples=len(datasets["train"]),
        replacement=True,
    )
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=preference_collate,
            num_workers=0,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=preference_collate,
            num_workers=0,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=preference_collate,
            num_workers=0,
        ),
    }

    split_sizes = {name: len(ds) for name, ds in datasets.items()}
    print("Dataset sizes:", split_sizes)
    print(f"model_type={model_type} freeze_encoder={freeze_encoder} checkpoint_metric={checkpoint_metric}")

    model = build_reward_model(model_type, n_player_buckets=n_player_buckets).to(device)
    init_info = {"loaded_encoder_keys": [], "checkpoint": None}
    if init_time_checkpoint is not None and Path(init_time_checkpoint).exists():
        loaded = load_encoder_from_time_checkpoint(model, init_time_checkpoint, device=device)
        init_info = {
            "loaded_encoder_keys": loaded,
            "checkpoint": str(init_time_checkpoint),
        }
        print(f"Initialized encoder from time checkpoint ({len(loaded)} tensors): {init_time_checkpoint}")
    else:
        print("Training reward head + encoder from scratch (no time checkpoint found).")

    if freeze_encoder:
        model.freeze_encoder()
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"Froze GINE encoder; trainable params {n_train}/{n_all}")

    teacher_diffs = None
    if distill_weight > 0:
        v1_path = Path(distill_v1_checkpoint or ORIGINAL_V1_CHECKPOINT)
        v2_path = Path(distill_v2_checkpoint or ORIGINAL_V2_CHECKPOINT)
        print(f"Precomputing distillation teachers (alpha={distill_alpha})...")
        print(f"  v1: {v1_path}")
        print(f"  v2: {v2_path}")
        diffs_v1 = _precompute_teacher_diffs(v1_path, datasets["train"], device)
        diffs_v2 = _precompute_teacher_diffs(v2_path, datasets["train"], device)
        teacher_diffs = (distill_alpha * diffs_v2 + (1.0 - distill_alpha) * diffs_v1).to(device)
        print("Teacher diffs ready.")

    time_targets = None
    if aux_time_weight > 0:
        print("Precomputing time-model auxiliary targets...")
        time_targets = _precompute_time_targets(all_smiles, cache_dir, device)
        print(f"Time targets for {len(time_targets)} molecules.")

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = []
    best_val_score = float("-inf")
    best_path = output_dir / "best_reward_model.pt"
    epochs_without_improvement = 0
    last_epoch = 0
    run_config = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "fooled_weight": fooled_weight,
        "fooled_sampler_weight": fooled_sampler_weight,
        "fooled_margin": fooled_margin,
        "fooled_margin_weight": fooled_margin_weight,
        "not_fooled_margin": not_fooled_margin,
        "not_fooled_margin_weight": not_fooled_margin_weight,
        "calibration_weight": calibration_weight,
        "calibration_target": calibration_target,
        "spread_weight": spread_weight,
        "spread_target": spread_target,
        "irm_weight": irm_weight,
        "adversarial_weight": adversarial_weight,
        "aux_time_weight": aux_time_weight,
        "distill_weight": distill_weight,
        "distill_alpha": distill_alpha,
        "distill_temperature": distill_temperature,
        "model_type": model_type,
        "freeze_encoder": freeze_encoder,
        "n_player_buckets": n_player_buckets,
        "checkpoint_metric": checkpoint_metric,
        "init_time_checkpoint": init_info.get("checkpoint"),
    }

    for epoch in range(1, num_epochs + 1):
        last_epoch = epoch
        _set_lr(optimizer, _warmup_cosine_lr(epoch, num_epochs, learning_rate, warmup_epochs))
        adv_lambda = min(1.0, epoch / max(warmup_epochs + 5, 1))

        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_fooled_acc = 0.0
        train_fooled_batches = 0
        n_batches = 0

        for batch in loaders["train"]:
            optimizer.zero_grad()
            loss, r_pref, r_rej, human_fooled, _extras = compute_batch_loss(
                model,
                batch,
                device,
                fooled_weight=fooled_weight,
                fooled_margin=fooled_margin,
                fooled_margin_weight=fooled_margin_weight,
                not_fooled_margin=not_fooled_margin,
                not_fooled_margin_weight=not_fooled_margin_weight,
                calibration_weight=calibration_weight,
                calibration_target=calibration_target,
                spread_weight=spread_weight,
                spread_target=spread_target,
                irm_weight=irm_weight,
                adversarial_weight=adversarial_weight,
                adversarial_lambda=adv_lambda,
                n_player_buckets=n_player_buckets,
                aux_time_weight=aux_time_weight,
                time_targets=time_targets,
                teacher_diffs=teacher_diffs,
                distill_weight=distill_weight,
                distill_temperature=distill_temperature,
            )
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            batch_metrics = bradley_terry_metrics(r_pref, r_rej, human_fooled)
            train_loss += loss.item()
            train_acc += batch_metrics["accuracy"]
            if batch_metrics.get("fooled_n", 0) > 0:
                train_fooled_acc += batch_metrics["fooled_accuracy"]
                train_fooled_batches += 1
            n_batches += 1

        train_metrics = {
            "loss": train_loss / max(n_batches, 1),
            "accuracy": train_acc / max(n_batches, 1),
            "fooled_accuracy": train_fooled_acc / max(train_fooled_batches, 1),
        }
        val_metrics = evaluate(
            model,
            loaders["val"],
            device,
            fooled_weight=fooled_weight,
            fooled_margin=fooled_margin,
            fooled_margin_weight=fooled_margin_weight,
            not_fooled_margin=not_fooled_margin,
            not_fooled_margin_weight=not_fooled_margin_weight,
            checkpoint_metric=checkpoint_metric,
        )
        test_metrics = evaluate(
            model,
            loaders["test"],
            device,
            fooled_weight=fooled_weight,
            fooled_margin=fooled_margin,
            fooled_margin_weight=fooled_margin_weight,
            not_fooled_margin=not_fooled_margin,
            not_fooled_margin_weight=not_fooled_margin_weight,
            checkpoint_metric=checkpoint_metric,
        )

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.3f} "
            f"fooled {train_metrics['fooled_accuracy']:.3f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.3f} "
            f"fooled {val_metrics.get('fooled_accuracy', float('nan')):.3f} "
            f"not-fooled {val_metrics.get('not_fooled_accuracy', float('nan')):.3f} "
            f"score {val_metrics.get('checkpoint_score', float('nan')):.3f} | "
            f"test acc {test_metrics['accuracy']:.3f} "
            f"fooled {test_metrics.get('fooled_accuracy', float('nan')):.3f}"
        )

        val_score = val_metrics.get("checkpoint_score", float("-inf"))
        if val_score > best_val_score:
            best_val_score = val_score
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "config": run_config,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print(
                    f"Early stopping after {epoch} epochs "
                    f"(no val score improvement for {early_stop_patience} epochs)."
                )
                break

    torch.save({"model_state_dict": model.state_dict(), "epoch": last_epoch, "config": run_config}, output_dir / "last_reward_model.pt")
    wall_time_sec = time.time() - started

    metadata = to_json_serializable(
        {
            "data_csv": str(data_csv),
            "cache_dir": str(cache_dir),
            "split_sizes": split_sizes,
            "player_splits": {k: sorted(v) for k, v in player_splits.items()},
            "cache_stats": cache_stats,
            "run_config": run_config,
            "fooled_weight": fooled_weight,
            "fooled_sampler_weight": fooled_sampler_weight,
            "fooled_margin": fooled_margin,
            "fooled_margin_weight": fooled_margin_weight,
            "init_time_checkpoint": init_info,
            "best_val_checkpoint_score": best_val_score,
            "wall_time_sec": wall_time_sec,
            "epochs_ran": last_epoch,
            "history": history,
        }
    )
    (output_dir / "training_history.json").write_text(json.dumps(metadata, indent=2))
    print(f"Finished in {wall_time_sec/60:.1f} min. Best checkpoint: {best_path}")
    return model, metadata


def main():
    parser = argparse.ArgumentParser(description="Train a Bradley-Terry reward model on Turing-test preferences.")
    parser.add_argument(
        "--data-csv",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "turing_guess_level.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "default_run"),
    )
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--fooled-weight",
        type=float,
        default=FOOLED_LOSS_WEIGHT,
        help="Extra loss multiplier for comparisons where human was fooled.",
    )
    parser.add_argument(
        "--fooled-sampler-weight",
        type=float,
        default=FOOLED_SAMPLER_WEIGHT,
        help="Oversample fooled pairs in the training sampler.",
    )
    parser.add_argument("--fooled-margin", type=float, default=FOOLED_MARGIN)
    parser.add_argument("--fooled-margin-weight", type=float, default=FOOLED_MARGIN_WEIGHT)
    parser.add_argument(
        "--init-time-checkpoint",
        type=str,
        default=str(DEFAULT_TIME_CHECKPOINT),
        help="Pretrained GINETimePredictor checkpoint for encoder init (empty string to disable).",
    )
    parser.add_argument("--no-init-time", action="store_true", help="Train encoder from scratch.")
    parser.add_argument(
        "--model-type",
        type=str,
        default="gin",
        choices=[
            "gin",
            "gin_fp",
            "gin_fp_small",
            "gin_fp_multitask",
            "gin_fp_adv",
            "gin_multitask",
            "gin_adv",
        ],
    )
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--not-fooled-margin", type=float, default=0.0)
    parser.add_argument("--not-fooled-margin-weight", type=float, default=0.0)
    parser.add_argument("--calibration-weight", type=float, default=0.0)
    parser.add_argument("--calibration-target", type=float, default=0.37)
    parser.add_argument("--spread-weight", type=float, default=0.0)
    parser.add_argument("--spread-target", type=float, default=0.3)
    parser.add_argument("--irm-weight", type=float, default=0.0)
    parser.add_argument("--adversarial-weight", type=float, default=0.0)
    parser.add_argument("--n-player-buckets", type=int, default=16)
    parser.add_argument("--aux-time-weight", type=float, default=0.0)
    parser.add_argument("--distill-v1", type=str, default="")
    parser.add_argument("--distill-v2", type=str, default="")
    parser.add_argument("--distill-alpha", type=float, default=0.5)
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument(
        "--checkpoint-metric",
        type=str,
        default="v2_composite",
        choices=["v2_composite", "balanced"],
    )
    args = parser.parse_args()

    init_ckpt = None if args.no_init_time else args.init_time_checkpoint
    if init_ckpt == "":
        init_ckpt = None

    print(f"Using device: {DEVICE}")
    train(
        data_csv=args.data_csv,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        fooled_weight=args.fooled_weight,
        fooled_sampler_weight=args.fooled_sampler_weight,
        fooled_margin=args.fooled_margin,
        fooled_margin_weight=args.fooled_margin_weight,
        init_time_checkpoint=init_ckpt,
        device=DEVICE,
        model_type=args.model_type,
        freeze_encoder=args.freeze_encoder,
        not_fooled_margin=args.not_fooled_margin,
        not_fooled_margin_weight=args.not_fooled_margin_weight,
        calibration_weight=args.calibration_weight,
        calibration_target=args.calibration_target,
        spread_weight=args.spread_weight,
        spread_target=args.spread_target,
        irm_weight=args.irm_weight,
        adversarial_weight=args.adversarial_weight,
        n_player_buckets=args.n_player_buckets,
        aux_time_weight=args.aux_time_weight,
        distill_v1_checkpoint=args.distill_v1 or None,
        distill_v2_checkpoint=args.distill_v2 or None,
        distill_alpha=args.distill_alpha,
        distill_weight=args.distill_weight,
        distill_temperature=args.distill_temperature,
        checkpoint_metric=args.checkpoint_metric,
    )


if __name__ == "__main__":
    main()
