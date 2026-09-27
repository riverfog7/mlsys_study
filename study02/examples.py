"""Real file-backed functions shared by the notebook and child experiments."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def tiny(x, weight):
    hidden = x @ weight
    return torch.relu(hidden)


def stack_example(x):
    y = x + 1
    return y * 2


def two_regions(x):
    first = torch.sin(x)
    torch._dynamo.graph_break()
    return torch.cos(first)


def data_dependent(x):
    if x.sum().item() > 0:
        return torch.sin(x)
    return torch.cos(x)


def python_branch(x, extra=None):
    if extra is None:
        return x.sin()
    return x.cos() + extra


def mutate_view(x):
    view = x.view(-1)
    view.add_(1)
    return x


class ManualGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))))


class ManualLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        centered = x - mean
        variance = centered.square().mean(-1, keepdim=True)
        return centered * torch.rsqrt(variance + 1e-5) * self.weight + self.bias


class ResidualLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x, residual):
        hidden = x + residual
        return F.layer_norm(hidden, (hidden.shape[-1],), self.weight, self.bias, 1e-5)
