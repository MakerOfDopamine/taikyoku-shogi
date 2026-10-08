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

        self.conv_norm = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1),  # Shape: (B, 96, 36, 36)
            nn.GroupNorm(12, 96),
            nn.ReLU()
        )
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, padding=1, stride=2), # Shape: (B, 128, 18, 18)
            nn.GroupNorm(16, 128),
            nn.ReLU()
        )
        # Shape: (B, 128, 18, 18)
        # View:  (B, 128, 324)
        # Trans: (B, 324, 128)

        self.unpatch = nn.Linear(128, 4 * 64)
        self.fuse = nn.Conv2d(64 + 96, 64, 1)
        self.q = nn.Linear(64, 32)
        self.k = nn.Linear(64, 32)
        self.mtype = nn.Embedding(10, 32)   # the move's special: 0 ordinary, 1-8 lion first leg, 9 trample

        self.pos = nn.Parameter(torch.randn((1, (size ** 2) // 4, 128)) * 0.02)
        self.transformers = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, activation='gelu', batch_first=True, norm_first=True), num_layers=3)
        # Shape: (B, 324, 128)
        # Mean:  (B, 128)
        self.lin_shared = nn.Linear(128, 128)
        self.lin_value = nn.Linear(128, 1)

    def forward(self, x, moves=None):
        BATCH_SIZE = x.shape[0]
        x = x.reshape(BATCH_SIZE, self.size ** 2)
        x = self.emb(x)

        x = x.transpose(1, 2)
        x = x.reshape((BATCH_SIZE, 64, self.size, self.size))
        z = self.conv_norm(x)
        x = self.conv_down(z)

        x = x.reshape((BATCH_SIZE, 128, (self.size ** 2) // 4))
        x = x.transpose(1, 2)
        x = x + self.pos
        x = self.transformers(x)

        # Policy voodoo
        t = self.unpatch(x).view(BATCH_SIZE, 18, 18, 2, 2, 64)
        t = t.permute(0, 5, 1, 3, 2, 4).reshape(BATCH_SIZE, 64, 36, 36)
        sq = self.fuse(torch.cat([t, z], 1)).flatten(2).transpose(1, 2)   # Shape: (B, 1296, 64)

        x = x.mean(dim=1)
        x = F.relu(self.lin_shared(x))

        value = torch.tanh(self.lin_value(x))
        if moves is None:
            return value, None

        qf = self.q(sq[moves[:, 0], moves[:, 1]]) + self.mtype(moves[:, 3])
        kf = self.k(sq[moves[:, 0], moves[:, 2]])
        policy = (qf * kf).sum(-1)

        return value, policy

if __name__ == "__main__":
    a = TaikyokuShogiBot()
    rows = []
    for i in range(10):
        rows.append((0, torch.randint(0, 1296, (1,)), torch.randint(0, 1296, (1,)), torch.randint(0, 10, (1,))))
    moves = torch.tensor(rows, dtype=torch.long)

    print(a.forward(torch.randint(0, 302, (36, 36)).unsqueeze(0), moves))