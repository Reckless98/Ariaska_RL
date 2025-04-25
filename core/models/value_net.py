# core/models/value_net.py — ARIASKA ValueNet v12.0 APEX INTEL
# 🧠 Advanced GPT-Reasoned Value Estimator | 🎯 Entropy-Aware Precision | 🌐 Contextual Phase Intelligence

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
from core.utils.local_llm_manager import LocalLLMManager

console = Console()


class ValueNet(nn.Module):
    """
    ARIASKA ValueNet v12.1 APEX INTEL (DQN Mode)
    • Deep GPT-Integrated Value Reasoning
    • Entropy-Adaptive Exploration Control
    • Phase & Context-Aware Dynamic Embedding
    • Noisy Layers + Context Vectors for Generalization
    """

    def __init__(
        self, input_size=512, hidden_size=384, device="cuda", lr=1e-4, activation="gelu"
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.activation = get_activation(activation)

        # Phase & Context Embedding
        self.phase_embed = nn.Linear(5, hidden_size)
        self.context_embed = nn.Linear(32, hidden_size)

        # Deep Encoder Stack
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(p=0.15)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(p=0.15)

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout3 = nn.Dropout(p=0.1)

        # Noisy Regression Head
        self.noisy_fc = NoisyLinear(hidden_size, 1)

        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1500
        )

        # GPT Context Encoder
        self.context_encoder = GPTContextEncoder()

        self.gpt_manager = GPTManager()

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
        Forward pass through encoder with phase & GPT-context integration.
        Returns value estimate for the state and feature representation.
        """
        x = self.activation(self.norm1(self.fc1(state)))
        x = self.dropout1(x)

        if phase_vector is not None:
            phase_proj = self.activation(self.phase_embed(phase_vector))
            x = x + phase_proj

        if context_text:
            ctx_vec = torch.tensor(
                self.context_encoder.encode(context_text),
                dtype=torch.float32,
                device=self.device,
            )
            ctx_proj = self.activation(self.context_embed(ctx_vec))
            x = x + ctx_proj

        x = self.activation(self.norm2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.activation(self.norm3(self.fc3(x)))
        x = self.dropout3(x)

        # Store features for entropy calculation and explainability
        features = x
        
        value = self.noisy_fc(x)
        return value, features

    def predict(self, state_tensor, phase_vector=None, context_text=None):
        """
        Predict scalar value for a state.
        """
        self.eval()
        with torch.no_grad():
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)
            if phase_vector is not None:
                phase_vector = phase_vector.to(self.device)
            value, _ = self.forward(
                state_tensor.to(self.device), phase_vector, context_text
            )
        return value.item()

    def train_step(self, states, targets, context_texts=None, grad_clip=1.0):
        """
        DQN value update: minimize (V(s) - target)^2.
        """
        self.train()
        batch_size = states.size(0)
        total_loss = 0.0

        for i in range(batch_size):
            ctx_text = context_texts[i] if context_texts else None
            predicted, _ = self.forward(states[i], context_text=ctx_text)
            loss = F.smooth_l1_loss(predicted.squeeze(), targets[i])

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
            self.optimizer.step()

            total_loss += loss.item()

        self.scheduler.step()
        self.noisy_fc.reset_noise()

        avg_loss = total_loss / batch_size
        console.print(f"[cyan]🎯 ValueNet Avg Training Loss:[/cyan] {avg_loss:.4f}")
        return avg_loss

    def estimate_entropy(self, state_tensor, phase_vector=None, context_text=None):
        """
        Compute entropy as a measure of uncertainty using feature norm.
        """
        self.eval()
        with torch.no_grad():
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)
            if phase_vector is not None:
                phase_vector = phase_vector.to(self.device)

            _, features = self.forward(
                state_tensor.to(self.device), phase_vector, context_text
            )
            norm = features.norm(dim=1)
            entropy = torch.exp(-norm)
            return entropy.item()

    def entropy_heatmap(self, states_batch, context_texts=None):
        """
        Generate entropy scores across a batch of states with context awareness.
        """
        self.eval()
        entropies = []
        with torch.no_grad():
            for idx, s in enumerate(states_batch):
                ctx_text = context_texts[idx] if context_texts else None
                st = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(
                    0
                )
                _, features = self.forward(st, context_text=ctx_text)
                norm = features.norm(dim=1)
                entropies.append(torch.exp(-norm).item())
        return entropies

    def inspect_value(self, state_tensor, phase_vector=None, context_text=None):
        """
        Display the current value estimate in a sleek, context-aware format.
        """
        self.eval()
        with torch.no_grad():
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)
            if phase_vector is not None:
                phase_vector = phase_vector.to(self.device)

            value, _ = self.forward(
                state_tensor.to(self.device), phase_vector, context_text
            )
            console.print(
                f"[bold green]📈 Estimated Value:[/bold green] {value.item():.3f}"
            )

    def _gpt_explain_value(self, value_estimate, features, context_text=None):
        """
        Use GPTManager to explain why this value might have been assigned, enriched by context.
        """
        feature_summary = f"Feature Norm: {features.norm().item():.3f}"
        context_note = (
            f"Context: {context_text}" if context_text else "General analysis."
        )
        prompt = f"""
You are an AI reinforcement learning strategist.
Explain why a state received a value estimate of {value_estimate:.3f}.
{context_note}
{feature_summary}
Respond concisely in one insightful sentence.
"""
        try:
            response = self.gpt_manager.gpt_request(prompt, task_type="analysis")
            return self.gpt_manager._sanitize_output(response)
        except Exception as e:
            console.print(f"[yellow]⚠ GPT value explanation failed: {e}[/yellow]")
            return "No insight provided."

    def save(self, path):
        """
        Save the ValueNet model state securely.
        """
        try:
            torch.save(self.state_dict(), path)
            console.print(f"[cyan]💾 ValueNet saved to {path}[/cyan]")
        except Exception as e:
            console.print(f"[red]❌ Failed to save ValueNet: {e}[/red]")

    def load(self, path):
        """
        Load the ValueNet model state with validation.
        """
        try:
            self.load_state_dict(torch.load(path, map_location=self.device))
            self.eval()
            console.print(f"[green]✔ ValueNet loaded from {path}[/green]")
        except Exception as e:
            console.print(f"[red]⚠ Failed to load ValueNet: {e}[/red]")


# ─────────────────────────────────────────────
# 🚀 Diagnostic Mode
# ─────────────────────────────────────────────
if __name__ == "__main__":
    vn = ValueNet()
    dummy_state = torch.randn(1, 512)
    phase_vec = get_phase_vector("exploit", device=vn.device)
    context = "Simulating privilege escalation on Linux server."

    value, features = vn.forward(
        dummy_state.to(vn.device), phase_vec, context_text=context
    )
    explanation = vn._gpt_explain_value(value.item(), features, context_text=context)

    vn.inspect_value(dummy_state, context_text=context)
    console.print(f"[bold magenta]🧠 GPT Insight:[/bold magenta] {explanation}")
