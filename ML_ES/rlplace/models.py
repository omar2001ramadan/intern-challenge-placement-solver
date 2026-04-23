"""Graph neural policies for placement search and distillation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import GraphBatch
from .utils import graph_mean_pool


class MessagePassingLayer(nn.Module):
    """Residual message-passing layer over the cell graph."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        edge_inputs = torch.cat([hidden[src], hidden[dst], edge_weight.unsqueeze(1)], dim=1)
        messages = self.edge_mlp(edge_inputs)
        aggregated = torch.zeros_like(hidden)
        aggregated.index_add_(0, dst, messages)
        updated = self.update_mlp(torch.cat([hidden, aggregated], dim=1))
        return self.norm(hidden + updated)


class PlacementGraphEncoder(nn.Module):
    """Shared graph encoder for teacher and student policies."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([MessagePassingLayer(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch: GraphBatch) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.gelu(self.input_proj(batch.node_features))
        for layer in self.layers:
            hidden = layer(hidden, batch.edge_index, batch.edge_weight)
            hidden = self.dropout(hidden)
        pooled = graph_mean_pool(hidden, batch.batch_index, int(batch.sizes.shape[0]))
        return hidden, pooled


class PlacementPolicy(nn.Module):
    """Action-only policy network for both the teacher and distilled student."""

    def __init__(self, input_dim: int = 12, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = PlacementGraphEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, batch: GraphBatch) -> dict[str, torch.Tensor]:
        node_embeddings, _ = self.encoder(batch)
        return {"action_mean": self.policy_head(node_embeddings)}


def checkpoint_payload(model: PlacementPolicy, model_config: dict, extra: dict | None = None) -> dict:
    """Assemble a self-describing checkpoint dictionary."""
    payload = {
        "state_dict": model.state_dict(),
        "model_config": model_config,
    }
    if extra:
        payload.update(extra)
    return payload
