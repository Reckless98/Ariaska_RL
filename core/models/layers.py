# core/models/layers.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 📦 Custom Layers for PolicyNet / ValueNet
# ─────────────────────────────────────────────

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.full((out_features,), sigma_init))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


class SineLinear(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super(SineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -math.sqrt(6 / self.in_features) / self.omega_0,
                    math.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


# ─────────────────────────────────────────────
# 🧠 Utility Functions
# ─────────────────────────────────────────────

def get_phase_vector(phase_str, device="cpu"):
    """Return a one-hot encoded vector for the current phase."""
    phases = ["recon", "enumeration", "exploit", "privesc", "exfiltrate"]
    vec = torch.zeros(len(phases), device=device)
    if phase_str in phases:
        vec[phases.index(phase_str)] = 1.0
    return vec


def get_activation(name):
    """Dynamic activation selector."""
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    elif name == "relu":
        return nn.ReLU()
    elif name == "silu":
        return nn.SiLU()
    elif name == "tanh":
        return nn.Tanh()
    return nn.ReLU()

