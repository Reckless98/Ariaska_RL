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

console = Console()

def safe_tensor(data, device):
    return torch.as_tensor(data, dtype=torch.float32, device=device).clone().detach()

class PolicyNet(nn.Module):
    """
    ARIASKA PolicyNet v12.0 APEX STRATEGOS
    • GPT-Enhanced Contextual Awareness
    • Adaptive Entropy & Temperature Tuning
    • NoisyLinear Exploration + Dynamic Phase Embedding
    • Token-Efficient GPT Tactical Insights
    """

    def __init__(
        self,
        input_size=512,
        hidden_size=288,
        output_size=6,
        lr=9e-5,
        device="cuda",
        activation="silu",
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

        # Core Layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(p=0.10)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(p=0.10)

        # Decision Head
        self.noisy_fc3 = NoisyLinear(hidden_size, hidden_size)
        self.noisy_fc4 = NoisyLinear(hidden_size, output_size)

        # Control Parameters
        self.entropy_beta = 0.02
        self.temperature = 1.0
        self.use_dynamic_temp = True
        self.inject_context = True

        # GPT Context Encoder
        self.context_encoder = GPTContextEncoder()

        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=600, gamma=0.92)

        self.to(self.device)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)

    def forward(self, state, phase_vector=None, context_text=None):
        """
        Forward pass with phase and GPT-encoded context injection.
        """
        x = self.activation(self.norm1(self.fc1(state)))
        x = self.dropout1(x)

        if phase_vector is not None:
            phase_proj = self.activation(self.phase_embed(phase_vector))
            x = x + phase_proj

        if self.inject_context and context_text:
            context_vec = torch.tensor(
                self.context_encoder.encode(context_text),
                dtype=torch.float32,
                device=self.device
            )
            context_proj = self.activation(self.context_embed(context_vec))
            x = x + context_proj

        x = self.activation(self.norm2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.activation(self.noisy_fc3(x))
        logits = self.noisy_fc4(x)

        if self.use_dynamic_temp:
            logits = logits / self.temperature

        return logits

    def predict(self, state, phase_vector=None, context_text=None, deterministic=True):
        """
        Predict action index with optional GPT tactical explanation.
        """
        self.eval()
        with torch.no_grad():
            state_tensor = safe_tensor(state, self.device)
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)
            if phase_vector is not None:
                phase_vector = phase_vector.to(self.device)

            logits = self.forward(state_tensor, phase_vector, context_text)
            probs = F.softmax(logits, dim=-1)

            action = (
                torch.argmax(probs, dim=-1).item()
                if deterministic
                else torch.multinomial(probs, 1).item()
            )

            if random.random() < 0.08:
                self._gpt_action_explain(action, probs.squeeze().tolist(), context_text)
    
            return action

    def _gpt_action_explain(self, action_idx, probs, context_text):
        """
        Token-efficient GPT explanation of action choice.
        """
        prompt = f"""
Summarize why action index {action_idx} is optimal.
Probabilities: {probs}
Context: {context_text or "Standard offensive operation."}
Respond in one short tactical sentence.
"""
        try:
            result = subprocess.run(
                ["sgpt", "--model", "gpt-4.1-nano", "--temperature", "0.25", "--role", "aria", prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
            explanation = result.stdout.strip()
            console.print(f"[dim cyan]🎯 GPT Tactical Insight:[/dim cyan] {explanation}")
        except Exception as e:
            console.print(f"[yellow]⚠ GPT insight failed: {e}[/yellow]")

    def train_step(self, states, actions, advantages, entropy_beta=None, grad_clip=0.5):
        """
        Policy gradient update with advanced entropy regulation.
        """
        self.train()
        logits = self.forward(states)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        entropy = -(log_probs * probs).sum(dim=-1).mean()

        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        policy_loss = -(selected_log_probs * advantages.detach()).mean()
        total_loss = policy_loss - (entropy_beta or self.entropy_beta) * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        self.optimizer.step()
        self.scheduler.step()

        self.noisy_fc3.reset_noise()
        self.noisy_fc4.reset_noise()

        if self.use_dynamic_temp:
            self._adjust_temperature(entropy.item())

        return total_loss.item(), entropy.item()

    def _adjust_temperature(self, entropy_val):
        """
        Dynamically adjust temperature based on entropy.
        """
        target_entropy = 0.85
        self.temperature *= (1.0 + (target_entropy - entropy_val))
        self.temperature = max(0.3, min(1.8, self.temperature))

    def adjust_entropy_beta(self, factor):
        self.entropy_beta = max(0.005, min(0.05, self.entropy_beta * factor))
        console.print(f"[cyan]🔧 Entropy beta adjusted to {self.entropy_beta:.4f}[/cyan]")

    def inspect_distribution(self, state, context_text=None):
        """
        Visualize action probabilities for a given state and context.
        """
        self.eval()
        with torch.no_grad():
            state_tensor = safe_tensor(state, self.device).unsqueeze(0)
            context_vector = None
            if context_text:
                context_vector = torch.tensor(self.context_encoder.encode(context_text), dtype=torch.float32).unsqueeze(0).to(self.device)

            logits = self.forward(state_tensor, context_texts=[context_text])
            probs = F.softmax(logits, dim=-1).squeeze()

            console.print("[bold cyan]🎯 Action Probability Distribution:[/bold cyan]")
            for idx, prob in enumerate(probs):
                console.print(f"  Action {idx}: [green]{prob:.4f}[/green]")

    def uncertainty_score(self, state, context_text=None):
        """
        Calculate entropy as a measure of uncertainty in decision-making.
        """
        self.eval()
        with torch.no_grad():
            state_tensor = safe_tensor(state, self.device).unsqueeze(0)
            logits = self.forward(state_tensor, context_texts=[context_text])
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(log_probs * probs).sum(dim=-1)
            return entropy.item()

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
