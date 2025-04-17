# core/models/policy_net.py — ARIASKA Cognitive Cortex v5.1+

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


class PolicyNet(nn.Module):
    """
    ARIASKA PolicyNet v5.1 — Multi-phase Decision Cortex
    • Phase-aware attention + noisy exploration + entropy control
    • Modular phase embeddings, temperature logic, dropout gates
    """
    def __init__(self, input_size=512, hidden_size=256, output_size=5, lr=1e-4, device="cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # Phase embedding: 5D vector → projection
        self.phase_embed = nn.Linear(5, hidden_size)

        # Encoder layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(p=0.15)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(p=0.15)

        # Noisy output layer
        self.noisy_fc3 = NoisyLinear(hidden_size, hidden_size)
        self.noisy_fc4 = NoisyLinear(hidden_size, output_size)

        self.entropy_beta = 0.01
        self.temperature = 1.0
        self.use_dynamic_temp = True
        self.inject_phase_direct = True

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=800, gamma=0.96)

        self.to(self.device)
        self._init_weights()
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)

    def forward(self, state, phase_vector=None):
        """
        Forward pass with optional phase embedding.
        """
        x = F.relu(self.norm1(self.fc1(state)))
        x = self.dropout1(x)

        if phase_vector is not None:
            phase_proj = F.relu(self.phase_embed(phase_vector))
            x = x + phase_proj if self.inject_phase_direct else x * phase_proj

        x = F.relu(self.norm2(self.fc2(x)))
        x = self.dropout2(x)

        x = F.relu(self.noisy_fc3(x))
        logits = self.noisy_fc4(x)

        if self.use_dynamic_temp:
            logits = logits / self.temperature

        return logits

    def predict(self, state, deterministic=True):
        self.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)

            logits = self.forward(state_tensor)
            probs = F.softmax(logits, dim=-1)

            return torch.argmax(probs, dim=-1).item() if deterministic else torch.multinomial(probs, 1).item()

    def train_step(self, states, actions, advantages, entropy_beta=0.01, grad_clip=0.5):
        self.train()
        logits = self.forward(states)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)
        entropy = -(log_probs * probs).sum(dim=-1).mean()

        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        policy_loss = -(selected_log_probs * advantages.detach()).mean()
        total_loss = policy_loss - entropy_beta * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        self.optimizer.step()
        self.scheduler.step()

        self.noisy_fc3.reset_noise()
        self.noisy_fc4.reset_noise()

        if self.use_dynamic_temp:
            self._update_temperature(entropy.item())

        return total_loss.item(), entropy.item()

    def _update_temperature(self, entropy_val):
        self.temperature = max(0.5, min(1.5, 1.0 + (0.8 - entropy_val)))

    def inspect_distribution(self, state):
        self.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)
            logits = self.forward(state_tensor)
            probs = F.softmax(logits, dim=-1).squeeze()
            print("\n[🧠 Policy Distribution]")
            for i, p in enumerate(probs):
                print(f"  Action {i}: {p:.4f}")

    def uncertainty_score(self, state):
        self.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.forward(state_tensor)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(log_probs * probs).sum(dim=-1)
            return entropy.item()

    def adjust_temperature(self, factor):
        self.temperature = max(0.1, min(2.0, self.temperature * factor))

    def set_entropy_beta(self, value):
        self.entropy_beta = value

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()
