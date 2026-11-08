"""Transformer backbone with temporal mean pooling."""
import torch
from torch import nn


class TinyTransformer(nn.Module):
    def __init__(self, steps: int, width: int = 64, heads: int = 4,
                 layers: int = 2, dropout: float = 0.1, feedforward_dim: int | None = None):
        super().__init__()
        self.input = nn.Linear(3, width)
        self.position = nn.Parameter(torch.zeros(1, steps, width))
        nn.init.normal_(self.position, std=0.02)
        block = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=feedforward_dim or width * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 3)

    def forward(self, x):
        h = self.encoder(self.input(x) + self.position)
        return self.output(self.norm(h.mean(dim=1)))
