import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(self, length, width=128, heads=8, layers=3, dropout=.1, feedforward_dim=None):
        super().__init__()
        self.input = nn.Linear(3, width)
        self.position = nn.Parameter(torch.empty(1, length, width))
        nn.init.normal_(self.position, std=.02)
        layer = nn.TransformerEncoderLayer(width, heads, feedforward_dim or width * 2, dropout, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        # Encoder clones are independently initialized (rather than identical copies).
        for block in self.encoder.layers:
            for p in block.parameters():
                if p.ndim > 1:
                    nn.init.xavier_uniform_(p)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 3)

    def forward(self, x):
        return self.output(self.norm(self.encoder(self.input(x)+self.position).mean(1)))


class CNN(nn.Module):
    def __init__(self, length, dropout=.1):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(128, 3))

    def forward(self, x):
        return self.layers(x[:, None])
