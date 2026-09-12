"""Rewrites TransformerEncoder into the shape ONNX Runtime's attention fusion recognises.

nn.MultiheadAttention exports its packed in_proj as chunk/slice arithmetic, which ORT
cannot match, so the graph keeps a literal 1296x1296 score matrix per head per layer --
27 MB per board per layer in fp16, which is what caps the sub-batch. Splitting the packed
projection into three Linears gives the BERT pattern ORT fuses into com.microsoft.Attention.

This is a re-expression, not a reimplementation: the weights are views of the originals and
export_onnx.py asserts the outputs match the reference layer.
"""
import math

import torch
import torch.nn as nn


class BertShapedLayer(nn.Module):
    """One norm_first TransformerEncoderLayer, written out as separate q/k/v projections."""

    def __init__(self, layer: nn.TransformerEncoderLayer, nhead: int):
        super().__init__()
        mha = layer.self_attn
        e = mha.embed_dim
        self.h, self.d = nhead, e // nhead
        self.q, self.k, self.v = (nn.Linear(e, e) for _ in range(3))
        for i, lin in enumerate((self.q, self.k, self.v)):
            lin.weight = nn.Parameter(mha.in_proj_weight[i * e:(i + 1) * e].clone())
            lin.bias = nn.Parameter(mha.in_proj_bias[i * e:(i + 1) * e].clone())
        self.out = mha.out_proj
        self.norm1, self.norm2 = layer.norm1, layer.norm2
        self.lin1, self.lin2 = layer.linear1, layer.linear2

    def _heads(self, t):
        b, s, _ = t.shape
        return t.view(b, s, self.h, self.d).transpose(1, 2)

    def forward(self, x):
        y = self.norm1(x)
        q, k, v = self._heads(self.q(y)), self._heads(self.k(y)), self._heads(self.v(y))
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.d), dim=-1)
        a = (a @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], self.h * self.d)
        x = x + self.out(a)
        y = self.norm2(x)
        return x + self.lin2(torch.nn.functional.gelu(self.lin1(y)))


def rewrite(net, nhead=8):
    """In place on a copy of the network: swap every encoder layer for the BERT-shaped one."""
    net.transformers = nn.Sequential(*[BertShapedLayer(l, nhead) for l in net.transformers.layers])
    return net
