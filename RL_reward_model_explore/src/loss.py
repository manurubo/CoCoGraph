"""Bradley-Terry preference loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import FOOLED_LOSS_WEIGHT, FOOLED_MARGIN, FOOLED_MARGIN_WEIGHT


def combine_sample_weights(
    expert_weights: torch.Tensor,
    human_fooled: torch.Tensor,
    fooled_multiplier: float = FOOLED_LOSS_WEIGHT,
) -> torch.Tensor:
    """Combine expert_weight with extra weight on fooled (hard) comparisons."""
    weights = expert_weights.clone()
    fooled_mask = human_fooled > 0.5
    weights[fooled_mask] = weights[fooled_mask] * fooled_multiplier
    return weights


def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    return weights / weights.sum().clamp_min(1e-8) * weights.numel()


def bradley_terry_loss(
    reward_preferred: torch.Tensor,
    reward_rejected: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Negative log-likelihood under the Bradley-Terry model:

        P(preferred > rejected) = sigmoid(r_pref - r_rej)
    """
    logits = reward_preferred - reward_rejected
    per_sample = F.softplus(-logits)
    if weights is None:
        return per_sample.mean()
    norm_weights = _normalize_weights(weights)
    return (per_sample * norm_weights).mean()


def fooled_margin_loss(
    reward_preferred: torch.Tensor,
    reward_rejected: torch.Tensor,
    human_fooled: torch.Tensor,
    margin: float = FOOLED_MARGIN,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hinge loss on fooled pairs to enforce a minimum preferred−rejected gap."""
    fooled_mask = human_fooled > 0.5
    if not fooled_mask.any():
        return reward_preferred.new_zeros(())

    logits = reward_preferred - reward_rejected
    per_sample = F.relu(margin - logits) * fooled_mask.float()
    if weights is None:
        return per_sample[fooled_mask].mean()
    norm_weights = _normalize_weights(weights)
    fooled_weights = norm_weights * fooled_mask.float()
    return (per_sample * fooled_weights).sum() / fooled_weights.sum().clamp_min(1e-8)


def preference_loss(
    reward_preferred: torch.Tensor,
    reward_rejected: torch.Tensor,
    weights: torch.Tensor | None = None,
    human_fooled: torch.Tensor | None = None,
    fooled_margin: float = FOOLED_MARGIN,
    fooled_margin_weight: float = FOOLED_MARGIN_WEIGHT,
    not_fooled_margin: float = 0.0,
    not_fooled_margin_weight: float = 0.0,
) -> torch.Tensor:
    """Bradley-Terry loss plus optional class-conditional margin terms."""
    loss = bradley_terry_loss(reward_preferred, reward_rejected, weights)
    if human_fooled is not None and fooled_margin > 0 and fooled_margin_weight > 0:
        loss = loss + fooled_margin_weight * fooled_margin_loss(
            reward_preferred,
            reward_rejected,
            human_fooled,
            margin=fooled_margin,
            weights=weights,
        )
    if human_fooled is not None and not_fooled_margin > 0 and not_fooled_margin_weight > 0:
        loss = loss + not_fooled_margin_weight * not_fooled_margin_loss(
            reward_preferred,
            reward_rejected,
            human_fooled,
            margin=not_fooled_margin,
            weights=weights,
        )
    return loss


def not_fooled_margin_loss(
    reward_preferred: torch.Tensor,
    reward_rejected: torch.Tensor,
    human_fooled: torch.Tensor,
    margin: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hinge on not-fooled pairs: preferred (real) should beat rejected (gen) by `margin`."""
    not_fooled_mask = human_fooled < 0.5
    if not not_fooled_mask.any():
        return reward_preferred.new_zeros(())

    logits = reward_preferred - reward_rejected
    per_sample = F.relu(margin - logits) * not_fooled_mask.float()
    if weights is None:
        return per_sample[not_fooled_mask].mean()
    norm_weights = _normalize_weights(weights)
    mask_weights = norm_weights * not_fooled_mask.float()
    return (per_sample * mask_weights).sum() / mask_weights.sum().clamp_min(1e-8)


def generation_prior_loss(
    reward_preferred: torch.Tensor,
    reward_rejected: torch.Tensor,
    human_fooled: torch.Tensor,
    target_rate: float = 0.37,
) -> torch.Tensor:
    """Pull mean P(generated > real) toward the human fool rate (~37%)."""
    sign = 2.0 * human_fooled - 1.0
    r_gen_minus_real = sign * (reward_preferred - reward_rejected)
    p_gen = torch.sigmoid(r_gen_minus_real)
    return (p_gen.mean() - target_rate).pow(2)


def spread_hinge_loss(
    reward_preferred: torch.Tensor,
    reward_rejected: torch.Tensor,
    target_gap: float = 0.3,
) -> torch.Tensor:
    """Encourage a minimum mean |preferred − rejected| gap for usable RL advantages."""
    gap = (reward_preferred - reward_rejected).abs().mean()
    return F.relu(target_gap - gap)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Soft Bradley-Terry distillation on preferred−rejected logits."""
    student = student_logits.reshape(-1)
    teacher = teacher_logits.reshape(-1)
    return F.binary_cross_entropy_with_logits(
        student / temperature,
        torch.sigmoid(teacher / temperature),
    )


def irm_penalty(
    reward_preferred: torch.Tensor,
    reward_rejected: torch.Tensor,
    env_ids: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """IRM-lite: dummy-classifier gradient penalty per player-bucket environment."""
    penalties = []
    unique_envs = env_ids.unique()
    for env in unique_envs:
        mask = env_ids == env
        if int(mask.sum().item()) < 2:
            continue
        logits = reward_preferred[mask] - reward_rejected[mask]
        scale = logits.new_ones((), requires_grad=True)
        per_sample = F.softplus(-(scale * logits))
        if weights is None:
            loss_e = per_sample.mean()
        else:
            env_w = _normalize_weights(weights[mask])
            loss_e = (per_sample * env_w).mean()
        grad = torch.autograd.grad(loss_e, [scale], create_graph=True)[0]
        penalties.append(grad.pow(2))
    if not penalties:
        return reward_preferred.new_zeros(())
    return torch.stack(penalties).mean()


def bradley_terry_accuracy(
    reward_preferred: torch.Tensor,
    reward_rejected: torch.Tensor,
) -> torch.Tensor:
    return (reward_preferred > reward_rejected).float().mean()


def bradley_terry_metrics(
    reward_preferred: torch.Tensor,
    reward_rejected: torch.Tensor,
    human_fooled: torch.Tensor | None = None,
) -> dict[str, float]:
    correct = (reward_preferred > reward_rejected).float()
    metrics = {
        "accuracy": correct.mean().item(),
        "n": float(reward_preferred.numel()),
    }
    if human_fooled is None:
        return metrics

    fooled_mask = human_fooled > 0.5
    not_fooled_mask = ~fooled_mask
    if fooled_mask.any():
        metrics["fooled_accuracy"] = correct[fooled_mask].mean().item()
        metrics["fooled_n"] = float(fooled_mask.sum().item())
    else:
        metrics["fooled_accuracy"] = float("nan")
        metrics["fooled_n"] = 0.0
    if not_fooled_mask.any():
        metrics["not_fooled_accuracy"] = correct[not_fooled_mask].mean().item()
        metrics["not_fooled_n"] = float(not_fooled_mask.sum().item())
    else:
        metrics["not_fooled_accuracy"] = float("nan")
        metrics["not_fooled_n"] = 0.0
    return metrics
