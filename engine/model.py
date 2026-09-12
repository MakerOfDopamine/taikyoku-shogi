"""The network, on its own so it can be imported without running a training pass.

``engine.py`` and ``export_onnx.py`` share this definition; there is no second copy to
drift. Nothing here touches the GPU or the checkpoint at import time.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TaikyokuShogiBot(nn.Module):
    def __init__(self, size = 36, piece_types = 301):
        # TODO add size parameters or something
        super().__init__()

        self.size = size
        # Shape: (B, 36, 36)
        # View:  (B, 1296)
        self.emb = nn.Embedding(num_embeddings=(piece_types * 4 + 1), embedding_dim=64)
        # Shape: (B, 1296, 64)
        # Trans: (B, 64, 1296)
        # View:  (B, 64, 36, 36)

        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # Shape: (B, 128, 36, 36)
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, padding=1), # Shape: (B, 192, 36, 36)
            nn.GroupNorm(8, 192),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, padding=1), # Shape: (B, 256, 36, 36)
            nn.GroupNorm(8, 256),
            nn.ReLU()
        )
        # Shape: (B, 256, 36, 36)
        # View:  (B, 256, 1296)
        # Trans: (B, 1296, 256)

        self.pos = nn.Parameter(torch.randn((1, size ** 2, 256)) * 0.02)
        self.transformers = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=256, nhead=8, activation='gelu', batch_first=True, norm_first=True), num_layers=4)
        # Shape: (B, 1296, 256)
        # Mean:  (B, 256)
        self.lin_shared = nn.Linear(256, 256)
        self.lin_value = nn.Linear(256, 1)
        self.lin_material = nn.Linear(256, 1)

    def forward(self, x):
        BATCH_SIZE = x.shape[0]
        x = x.reshape(BATCH_SIZE, self.size ** 2)
        x = self.emb(x)

        x = x.transpose(1, 2)
        x = x.reshape((BATCH_SIZE, 64, self.size, self.size))
        x = self.conv_stem(x)

        x = x.reshape((BATCH_SIZE, 256, self.size ** 2))
        x = x.transpose(1, 2)
        x = x + self.pos
        x = self.transformers(x)

        x = x.mean(dim=1)
        x = F.relu(self.lin_shared(x))

        value = torch.tanh(self.lin_value(x))
        material = torch.tanh(self.lin_material(x))

        return value, material
