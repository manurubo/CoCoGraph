"""Reward model architecture based on CoCoGraph GINETimePredictor."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

from .config import NGFEAT, NHEAD, NNFEAT
from .fingerprints import MORGAN_BITS, morgan_fp_from_data


class EnhancedGINEConv(GINEConv):
    def __init__(
        self,
        nn_module: nn.Module,
        global_feat_dim: int,
        eps: float = 0.0,
        train_eps: bool = False,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(nn_module, eps, train_eps, edge_dim, **kwargs)
        if global_feat_dim is not None:
            if isinstance(self.nn, nn.Sequential):
                inner = self.nn[0]
            else:
                inner = self.nn
            if hasattr(inner, "in_features"):
                in_channels = inner.in_features
            elif hasattr(inner, "in_channels"):
                in_channels = inner.in_channels
            else:
                raise ValueError("Could not infer input channels from nn.")
            self.global_features_transform = Linear(global_feat_dim, in_channels)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        global_features: Tensor,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        transformed_global = self.global_features_transform(global_features)
        if transformed_global.size(0) != x[1].size(0):
            transformed_global = transformed_global.repeat(x[1].size(0), 1)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r + transformed_global
        return self.nn(out)


class GINRewardModel(nn.Module):
    """
    Scalar reward model for molecules.

    Architecture mirrors CoCoGraph's GINETimePredictor (3 Enhanced GINE layers +
    global mean pooling + linear head), but outputs an unbounded reward score
    suitable for Bradley-Terry preference learning.
    """

    def __init__(self, hidden_mult: int = 2, nhead: int = NHEAD):
        super().__init__()
        hidden = hidden_mult * NNFEAT
        out_dim = hidden * 2 * nhead

        self.conv1 = EnhancedGINEConv(
            Sequential(Linear(NNFEAT, hidden), ReLU(), Linear(hidden, out_dim)),
            edge_dim=18,
            global_feat_dim=NGFEAT,
        )
        self.ff1 = Linear(out_dim, hidden)
        self.conv2 = EnhancedGINEConv(
            Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, out_dim)),
            edge_dim=18,
            global_feat_dim=NGFEAT,
        )
        self.ff2 = Linear(out_dim, hidden)
        self.conv3 = EnhancedGINEConv(
            Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, out_dim)),
            edge_dim=18,
            global_feat_dim=NGFEAT,
        )
        self.ff3 = Linear(out_dim, hidden)
        self.reward_head = nn.Sequential(
            nn.LayerNorm(hidden),
            Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            Linear(hidden // 2, 1),
        )

    def encode_graph(self, data) -> Tensor:
        x, edge_index, edge_attr, xA, dosd_distances = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.xA,
            data.dosd_distances,
        )
        batch = data.batch

        if xA.dim() == 1:
            num_graphs = int(batch.max().item()) + 1 if batch is not None and batch.numel() else 1
            if xA.numel() == num_graphs * NGFEAT:
                xA = xA.view(num_graphs, NGFEAT)
            else:
                xA = xA.unsqueeze(0)

        dosd_values = dosd_distances[edge_index[0], edge_index[1]].unsqueeze(1)
        edge_attr_added = torch.cat([edge_attr, dosd_values], dim=1)

        x = F.relu(self.conv1(x, edge_index, xA, edge_attr_added))
        x = F.relu(self.ff1(x))
        x = F.relu(self.conv2(x, edge_index, xA, edge_attr_added))
        x = F.relu(self.ff2(x))
        x = F.relu(self.conv3(x, edge_index, xA, edge_attr_added))
        x = F.relu(self.ff3(x))
        return global_mean_pool(x, batch)

    def encode_fused(self, data) -> Tensor:
        """Graph embedding that the reward head consumes (overridden by fusion models)."""
        return self.encode_graph(data)

    def forward(self, data) -> Tensor:
        return self.reward_head(self.encode_fused(data)).squeeze(-1)

    def freeze_encoder(self):
        for name, param in self.named_parameters():
            if name.startswith("conv") or name.startswith("ff"):
                param.requires_grad = False


class GINRewardModelFP(GINRewardModel):
    """GINE encoder fused with a Morgan-fingerprint MLP (GINETimePredictor_MorganFP style)."""

    uses_morgan_fp = True

    def __init__(self, hidden_mult: int = 2, nhead: int = NHEAD, morgan_bits: int = MORGAN_BITS):
        super().__init__(hidden_mult=hidden_mult, nhead=nhead)
        hidden = hidden_mult * NNFEAT
        self.morgan_fp_mlp = nn.Sequential(
            Linear(morgan_bits, 512),
            ReLU(),
            Linear(512, 256),
            ReLU(),
            Linear(256, 256),
        )
        self.fuse_fp = nn.Sequential(
            Linear(hidden + 256, hidden),
            nn.GELU(),
        )

    def encode_fused(self, data) -> Tensor:
        graph_emb = self.encode_graph(data)
        fp = morgan_fp_from_data(data)
        fp_emb = self.morgan_fp_mlp(fp.float())
        if fp_emb.dim() == 1:
            fp_emb = fp_emb.unsqueeze(0)
        if graph_emb.size(0) != fp_emb.size(0):
            fp_emb = fp_emb.expand(graph_emb.size(0), -1)
        return self.fuse_fp(torch.cat([graph_emb, fp_emb], dim=-1))


class GINRewardModelFPSmall(GINRewardModel):
    """Frozen-encoder-friendly fusion: 2048-d Morgan → 64-d bottleneck, then small head.

    The large GINETimePredictor_MorganFP MLP (~1.2M weights) memorizes 3.6k pairs.
    This bottleneck is the 'small fused head' in the two-stage plan.
    """

    uses_morgan_fp = True

    def __init__(
        self,
        hidden_mult: int = 2,
        nhead: int = NHEAD,
        morgan_bits: int = MORGAN_BITS,
        fp_dim: int = 64,
    ):
        super().__init__(hidden_mult=hidden_mult, nhead=nhead)
        hidden = hidden_mult * NNFEAT
        self.morgan_fp_mlp = nn.Sequential(
            nn.Dropout(0.25),
            Linear(morgan_bits, fp_dim),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        self.fuse_fp = nn.Sequential(
            Linear(hidden + fp_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.15),
        )

    def encode_fused(self, data) -> Tensor:
        graph_emb = self.encode_graph(data)
        fp = morgan_fp_from_data(data)
        fp_emb = self.morgan_fp_mlp(fp.float())
        if fp_emb.dim() == 1:
            fp_emb = fp_emb.unsqueeze(0)
        if graph_emb.size(0) != fp_emb.size(0):
            fp_emb = fp_emb.expand(graph_emb.size(0), -1)
        return self.fuse_fp(torch.cat([graph_emb, fp_emb], dim=-1))


class GINRewardModelFPMultiTask(GINRewardModelFP):
    """Fingerprint-fusion encoder with an auxiliary time/quality head."""

    def __init__(self, hidden_mult: int = 2, nhead: int = NHEAD, morgan_bits: int = MORGAN_BITS):
        super().__init__(hidden_mult=hidden_mult, nhead=nhead, morgan_bits=morgan_bits)
        hidden = hidden_mult * NNFEAT
        self.time_head = nn.Sequential(
            nn.LayerNorm(hidden),
            Linear(hidden, hidden // 2),
            nn.GELU(),
            Linear(hidden // 2, 1),
        )

    def predict_time(self, data) -> Tensor:
        return self.time_head(self.encode_fused(data)).squeeze(-1)


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, lambda_: float) -> Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return -ctx.lambda_ * grad_output, None


class GINRewardModelFPAdv(GINRewardModelFP):
    """Fingerprint fusion + player-bucket adversarial head (gradient reversal)."""

    def __init__(
        self,
        hidden_mult: int = 2,
        nhead: int = NHEAD,
        morgan_bits: int = MORGAN_BITS,
        n_player_buckets: int = 16,
    ):
        super().__init__(hidden_mult=hidden_mult, nhead=nhead, morgan_bits=morgan_bits)
        hidden = hidden_mult * NNFEAT
        self.n_player_buckets = n_player_buckets
        self.player_head = nn.Sequential(
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, n_player_buckets),
        )

    def player_logits(self, data, lambda_: float = 1.0) -> Tensor:
        fused = self.encode_fused(data)
        reversed_emb = _GradientReversalFn.apply(fused, lambda_)
        return self.player_head(reversed_emb)


class GINRewardModelMultiTask(GINRewardModel):
    """Original GINE encoder with an auxiliary time/quality head (no extra FP capacity)."""

    def __init__(self, hidden_mult: int = 2, nhead: int = NHEAD):
        super().__init__(hidden_mult=hidden_mult, nhead=nhead)
        hidden = hidden_mult * NNFEAT
        self.time_head = nn.Sequential(
            nn.LayerNorm(hidden),
            Linear(hidden, hidden // 2),
            nn.GELU(),
            Linear(hidden // 2, 1),
        )

    def predict_time(self, data) -> Tensor:
        return self.time_head(self.encode_fused(data)).squeeze(-1)


class GINRewardModelAdv(GINRewardModel):
    """Original GINE encoder + player-bucket adversarial head (gradient reversal)."""

    def __init__(
        self,
        hidden_mult: int = 2,
        nhead: int = NHEAD,
        n_player_buckets: int = 16,
    ):
        super().__init__(hidden_mult=hidden_mult, nhead=nhead)
        hidden = hidden_mult * NNFEAT
        self.n_player_buckets = n_player_buckets
        self.player_head = nn.Sequential(
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, n_player_buckets),
        )

    def player_logits(self, data, lambda_: float = 1.0) -> Tensor:
        fused = self.encode_fused(data)
        reversed_emb = _GradientReversalFn.apply(fused, lambda_)
        return self.player_head(reversed_emb)


def build_reward_model(model_type: str = "gin", n_player_buckets: int = 16) -> GINRewardModel:
    if model_type in ("gin", "v2", "baseline"):
        return GINRewardModel()
    if model_type in ("gin_fp", "fp"):
        return GINRewardModelFP()
    if model_type in ("gin_fp_small", "fp_small"):
        return GINRewardModelFPSmall()
    if model_type in ("gin_fp_multitask", "multitask"):
        return GINRewardModelFPMultiTask()
    if model_type in ("gin_fp_adv", "adv"):
        return GINRewardModelFPAdv(n_player_buckets=n_player_buckets)
    if model_type in ("gin_multitask", "multitask_gin"):
        return GINRewardModelMultiTask()
    if model_type in ("gin_adv", "adv_gin"):
        return GINRewardModelAdv(n_player_buckets=n_player_buckets)
    raise ValueError(f"Unknown model_type: {model_type}")


def load_encoder_from_time_checkpoint(
    model: GINRewardModel,
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> list[str]:
    """
    Initialize the shared GINE encoder from a pretrained GINETimePredictor checkpoint.
    Returns the list of loaded parameter names.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Time model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=False)
    time_state = checkpoint["model_state_dict"]
    reward_state = model.state_dict()
    loaded = []
    for key, value in time_state.items():
        if key.startswith("fc3"):
            continue
        if key in reward_state and reward_state[key].shape == value.shape:
            reward_state[key] = value
            loaded.append(key)
    model.load_state_dict(reward_state)
    return loaded
