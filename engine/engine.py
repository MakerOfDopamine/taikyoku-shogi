import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from unpack import unpack, embedding_index
from unpack_nn import unpack_nn
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
BATCH_SIZE = 16
STEP_SIZE = (128 // BATCH_SIZE)
EVAL_DATA = 0.25
VALUE_WEIGHT = 1.0

load(tkybot, coach)
i = 0
games = []
game = []
last_index = 0
mul = 1
for board, info in unpack_nn("selfplay.tkn"):
    x = embedding_index(board)     # exactly what nnplay fed the network
    v = info.value                 # the network's value of this board
    z = info.result_stm            # +1 if the side to move went on to win
    if info.game_index != last_index:
        last_index = info.game_index
        games.append(game)
        game = []
        mul = 1
    game.append((x, v, z, mul))
    mul *= -1
games.append(game)

GAMMA = 0.999
LAMBDA = 0.99
data = []
labels = []
for g in games:
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

data = torch.stack(tuple(map(torch.tensor, data)))
labels = torch.stack(tuple(map(torch.tensor, labels)))
print(data.shape, labels.shape)
print("max", labels.abs().max())
print("mean", labels.mean(), "std", labels.std())

'''
data = []
labels = []
results = []

test_data = []
test_labels = []
test_results = []
for board, info in unpack("./shard_0000.tkp"):
    x = embedding_index(board)          # (36,36) int, 0..603, for nn.Embedding(604, d)
    y = info.label_stm
    z = float(info.result_stm)          # +1 the side to move won, -1 lost, 0 drawn

    if random.random() < EVAL_DATA:
        test_data.append(torch.tensor(x))
        test_labels.append(torch.tensor(y))
        test_results.append(torch.tensor(z))
    else:
        data.append(torch.tensor(x))
        labels.append(torch.tensor(y))
        results.append(torch.tensor(z))

perm_data = torch.randperm(len(data))
data = torch.stack(data)[perm_data]
labels = torch.stack(labels)[perm_data]
results = torch.stack(results)[perm_data]

perm_test = torch.randperm(len(test_data))
test_data = torch.stack(test_data)[perm_test]
test_labels = torch.stack(test_labels)[perm_test]
test_results = torch.stack(test_results)[perm_test]

print(f"loaded {len(data)} samples")
print(labels.mean(), labels.std(), labels.min(), labels.max())
print(f"results: {(results > 0).float().mean():.3f} won, {(results < 0).float().mean():.3f} lost, "
      f"{(results == 0).float().mean():.3f} drawn")

loss_func = nn.MSELoss(reduction='mean')
tkybot.train()
coach.zero_grad()
steps_without_accumulation = 0
for epoch in range(1, 6):
    for i in range(0, len(data), BATCH_SIZE):
        data_in = data[i : i + BATCH_SIZE].to(device=device)
        outcomes = labels[i : i + BATCH_SIZE].unsqueeze(-1).to(device=device)
        winners = results[i : i + BATCH_SIZE].unsqueeze(-1).to(device=device)

        result_val, result_mat = tkybot(data_in)
        loss_mat = loss_func(result_mat, outcomes)
        loss_val = loss_func(result_val, winners)
        loss = (loss_mat + VALUE_WEIGHT * loss_val) / STEP_SIZE

        loss.backward()
        if steps_without_accumulation == STEP_SIZE:
            coach.step()
            coach.zero_grad()
            steps_without_accumulation = 0
            print("stepped at step", i)

        print(epoch, i, loss.item(), loss_mat.item(), loss_val.item())
        steps_without_accumulation += 1

    perm_data = torch.randperm(len(data))
    data = data[perm_data]
    labels = labels[perm_data]
    results = results[perm_data]

def r2(target, pred):
    return 1 - ((target - pred) ** 2).sum() / ((target - target.mean()) ** 2).sum()

tkybot.eval()
with torch.no_grad():
    pred_mat, pred_val = [], []
    for i in range(0, len(test_data), BATCH_SIZE):
        v, m = tkybot(test_data[i:i+BATCH_SIZE].to(device))
        pred_mat.append(m.squeeze(-1).cpu())
        pred_val.append(v.squeeze(-1).cpu())
    pred_mat = torch.cat(pred_mat)
    pred_val = torch.cat(pred_val)

    print(f"held-out material R² = {r2(test_labels, pred_mat).item():.4f}")
    print(f"held-out value    R² = {r2(test_results, pred_val).item():.4f}")
    # the value target is +/-1, so sign accuracy reads better than R2; 0.5 is chance
    print(f"held-out value sign accuracy = "
          f"{(torch.sign(pred_val) == torch.sign(test_results)).float().mean().item():.4f}")
    # the spread across positions is what the self-play softmax actually consumes
    print(f"value head spread: sd {pred_val.std().item():.4f}  "
          f"range {(pred_val.max() - pred_val.min()).item():.4f}")
tkybot.train()
save()
'''