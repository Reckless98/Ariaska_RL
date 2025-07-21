# core/models/policy_net.py — ARIASKA PolicyNet v12.0 APEX STRATEGOS
# 🧠 Context-Aware Tactical Cortex | 🎯 GPT-Backed Dynamic Strategy | ♻️ Advanced Entropy Control

import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import json
import os
import random

from core.models.layers import NoisyLinear, get_activation, get_phase_vector
from core.models.gpt_context_encoder import GPTContextEncoder
from rich.console import Console
from core.gpt_manager import GPTManager

console = Console()

def safe_tensor(data, device):
    return torch.as_tensor(data, dtype=torch.float32, device=device).clone().detach()

class PolicyNet(nn.Module):
    """
    Dueling DQN Policy Network for ARIASKA_RL
    - Modular, extensible, and PyTorch best-practice compliant
    - Supports both standard and dueling architectures
    """
    def __init__(self, input_size, output_size, hidden_size=256, dueling=True, device="cpu"):
        super().__init__()
        self.device = device
        self.dueling = dueling
        self.input_size = input_size
        self.output_size = output_size  # Add missing output_size attribute
        self.hidden_size = hidden_size
        
        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        if dueling:
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)

    def forward(self, x):
        x = x.to(self.device)
        features = self.feature(x)
        if self.dueling:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            qvals = value + (advantage - advantage.mean(dim=1, keepdim=True))
            return qvals
        else:
            return self.head(features)

    def predict(self, state, phase_vector=None, context_text=None, deterministic=True):
        """
        Predict action index by selecting argmax Q-value.
        """
        self.eval()
        with torch.no_grad():
            state_tensor = safe_tensor(state, self.device)
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)
            if phase_vector is not None:
                phase_vector = phase_vector.to(self.device)

            q_values = self.forward(state_tensor)
            action = torch.argmax(q_values, dim=-1).item()
            return action

    def entropy(self, state, phase_vector=None, context_text=None):
        """
        Estimate entropy of Q-value distribution for smarter exploration.
        """
        self.eval()
        with torch.no_grad():
            state_tensor = safe_tensor(state, self.device)
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)
            if phase_vector is not None:
                phase_vector = phase_vector.to(self.device)
            q_values = self.forward(state_tensor)
            probs = torch.softmax(q_values, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            return entropy.item()

    def inspect_distribution(self, state, phase_vector=None, context_text=None):
        """
        Print Q-value distribution and entropy for diagnostics.
        """
        self.eval()
        with torch.no_grad():
            state_tensor = safe_tensor(state, self.device)
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)
            if phase_vector is not None:
                phase_vector = phase_vector.to(self.device)
            q_values = self.forward(state_tensor)
            probs = torch.softmax(q_values, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            console.print(f"[blue]Q-values:[/blue] {q_values.cpu().numpy().tolist()}")
            console.print(f"[magenta]Softmax:[/magenta] {probs.cpu().numpy().tolist()}")
            console.print(f"[yellow]Entropy:[/yellow] {entropy.item():.4f}")

    def train_step(self, states, actions, targets, grad_clip=1.0):
        """
        DQN update: minimize (Q(s,a) - target)^2.
        """
        self.train()
        q_values = self.forward(states)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        loss = torch.nn.functional.smooth_l1_loss(q_selected, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def save(self, path):
        try:
            torch.save(self.state_dict(), path)
            console.print(f"[green]💾 PolicyNet saved at {path}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Save failed: {e}[/red]")

    def load(self, path):
        try:
            self.load_state_dict(torch.load(path, map_location=self.device))
            self.eval()
            console.print(f"[blue]✔ PolicyNet loaded from {path}[/blue]")
        except Exception as e:
            console.print(f"[red]⚠ Load failed: {e}[/red]")

# ─────────────────────────────────────────────
# 🚀 Diagnostic Mode
# ─────────────────────────────────────────────
if __name__ == "__main__":
    net = PolicyNet(input_size=512, output_size=6, dueling=True, device="cuda")
    dummy_state = torch.randn(512)
    context = "Target in exploitation phase with moderate blue team alert."
    action = net.predict(dummy_state.tolist(), context_text=context)
    net.inspect_distribution(dummy_state.tolist(), context_text=context)
    console.print(f"[bold magenta]🔹 Selected Action:[/bold magenta] {action}")
