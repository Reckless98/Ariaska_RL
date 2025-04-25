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
    ARIASKA PolicyNet v12.1 APEX STRATEGOS (DQN Mode)
    • Deep context/phase embedding
    • Noisy layers for smarter exploration
    • Entropy diagnostics for adaptive ε
    """
    def __init__(
        self,
        input_size=512,
        hidden_size=384,
        output_size=6,
        lr=7e-5,
        device="cuda",
        activation="gelu",
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = get_activation(activation)

        # Embeddings
        self.phase_embed = nn.Linear(5, hidden_size)
        self.context_embed = nn.Linear(32, hidden_size)

        # Deep Core Layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(p=0.15)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(p=0.15)

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout3 = nn.Dropout(p=0.10)

        # DQN Head: Noisy layers for exploration
        self.noisy_fc3 = NoisyLinear(hidden_size, hidden_size)
        self.noisy_fc4 = NoisyLinear(hidden_size, output_size)

        # GPT Context Encoder
        self.context_encoder = GPTContextEncoder()
        self.gpt_manager = GPTManager()

        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=600, gamma=0.92)

        self.to(self.device)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, state, phase_vector=None, context_text=None):
        """
        Forward pass with phase and GPT-encoded context injection.
        Returns Q-values for all actions.
        """
        x = self.activation(self.norm1(self.fc1(state)))
        x = self.dropout1(x)

        if phase_vector is not None:
            phase_proj = self.activation(self.phase_embed(phase_vector))
            x = x + phase_proj

        if context_text:
            context_vec = torch.tensor(
                self.context_encoder.encode(context_text),
                dtype=torch.float32,
                device=self.device
            )
            context_proj = self.activation(self.context_embed(context_vec))
            x = x + context_proj

        x = self.activation(self.norm2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.activation(self.norm3(self.fc3(x)))
        x = self.dropout3(x)

        x = self.activation(self.noisy_fc3(x))
        q_values = self.noisy_fc4(x)
        return q_values

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

            q_values = self.forward(state_tensor, phase_vector, context_text)
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
            q_values = self.forward(state_tensor, phase_vector, context_text)
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
            q_values = self.forward(state_tensor, phase_vector, context_text)
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
        self.noisy_fc3.reset_noise()
        self.noisy_fc4.reset_noise()
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
    net = PolicyNet()
    dummy_state = torch.randn(512)
    context = "Target in exploitation phase with moderate blue team alert."
    action = net.predict(dummy_state.tolist(), context_text=context)
    net.inspect_distribution(dummy_state.tolist(), context_text=context)
    console.print(f"[bold magenta]🔹 Selected Action:[/bold magenta] {action}")
