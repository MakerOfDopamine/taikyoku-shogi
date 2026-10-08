import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tkn_notebook import Tkn
from model import TaikyokuShogiBot
device = "cuda" if torch.cuda.is_available() else "cpu"

def save(path = 'checkpoint.pt'):
    torch.save({
        'model': tkybot.state_dict(),
        'optim': coach.state_dict()
    }, path)

def load(bot, opt):
    ckpt = torch.load('checkpoint.pt', map_location=device, weights_only=True)
    bot.load_state_dict(ckpt['model'])
    opt.load_state_dict(ckpt['optim'])

tkybot = TaikyokuShogiBot().to(device)
coach = optim.AdamW(tkybot.parameters(), 3e-4, weight_decay=0.01)
BATCH_SIZE = 64
STEP_SIZE = (256 // BATCH_SIZE)
EVAL_DATA = 0.25

load(tkybot, coach)
training_games = []
game = []
last_index = 0
mul = 1
for board, info in Tkn.unpack("selfplay.tkn"):
    x = Tkn.embedding_index(board)     # exactly what nnplay fed the network
    v = info.value                 # the network's value of this board
    z = info.result_stm            # +1 if the side to move went on to win
    if info.game_index != last_index:
        last_index = info.game_index
        training_games.append(game)
        game = []
        mul = 1
    game.append((x, v, z, mul))
    mul *= -1
training_games.append(game)

'''
for board, info in Tkn.unpack("selfplay2.tkn"):
    x = Tkn.embedding_index(board)     # exactly what nnplay fed the network
    v = info.value                 # the network's value of this board
    z = info.result_stm            # +1 if the side to move went on to win
    if info.game_index != last_index:
        last_index = info.game_index
        training_games.append(game)
        game = []
        mul = 1
    game.append((x, v, z, mul))
    mul *= -1
training_games.append(game)
'''

GAMMA = 0.999
LAMBDA = 0.99
data = []
labels = []
for g in training_games:
    dt = g[-1][2] - g[-1][1]
    accum = dt
    for move in range(len(g) - 2, -1, -1):
        accum *= GAMMA
        accum *= LAMBDA
        accum *= -1
        dt = - g[move + 1][1] - g[move][1]

        accum += dt
        data.append(g[move][0])
        labels.append(g[move][1] + accum)

data = torch.from_numpy(np.stack(data))
labels = torch.tensor(labels, dtype=torch.float32)
print(data.shape, labels.shape)
print("max", labels.abs().max())
print("mean", labels.mean(), "std", labels.std())

shuffle = torch.randperm(len(labels))
data = data[shuffle]
labels = labels[shuffle]

EPOCHS = 1
loss_func = nn.MSELoss(reduction='mean')
steps_without_accumulation = 0
for epoch in range(1, EPOCHS + 1):
    for i in range(0, len(data), BATCH_SIZE):
        data_in = data[i : i + BATCH_SIZE].to(device=device)
        outcomes = labels[i : i + BATCH_SIZE].unsqueeze(-1).to(device=device)

        result_val, _ = tkybot(data_in)
        loss_val = loss_func(result_val, outcomes)
        loss = (loss_val) / STEP_SIZE

        loss.backward()
        if steps_without_accumulation == STEP_SIZE:
            coach.step()
            coach.zero_grad()
            steps_without_accumulation = 0
            print(i, round(loss.item() * 1000, 3))
        steps_without_accumulation += 1

save("checkpoint2.pt")