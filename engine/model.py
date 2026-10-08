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
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1),  # Shape: (B, 96, 36, 36)
            nn.GroupNorm(12, 96),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, padding=1, stride=2), # Shape: (B, 128, 18, 18)
            nn.GroupNorm(16, 128),
            nn.ReLU(),
        )
        # Shape: (B, 128, 18, 18)
        # View:  (B, 128, 324)
        # Trans: (B, 324, 128)

        self.pos = nn.Parameter(torch.randn((1, (size ** 2) // 4, 128)) * 0.02)
        self.transformers = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=128, nhead=4, activation='gelu', batch_first=True, norm_first=True), num_layers=3)
        # Shape: (B, 324, 128)
        # Mean:  (B, 128)
        self.lin_shared = nn.Linear(128, 128)
        self.lin_value = nn.Linear(128, 1)
        self.lin_material = nn.Linear(128, 1)

    def forward(self, x):
        BATCH_SIZE = x.shape[0]
        x = x.reshape(BATCH_SIZE, self.size ** 2)
        x = self.emb(x)

        x = x.transpose(1, 2)
        x = x.reshape((BATCH_SIZE, 64, self.size, self.size))
        x = self.conv_stem(x)

        x = x.reshape((BATCH_SIZE, 128, (self.size ** 2) // 4))
        x = x.transpose(1, 2)
        x = x + self.pos
        x = self.transformers(x)

        x = x.mean(dim=1)
        x = F.relu(self.lin_shared(x))

        value = torch.tanh(self.lin_value(x))
        material = torch.tanh(self.lin_material(x))

        return value, material
