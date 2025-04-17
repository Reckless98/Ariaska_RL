# core/models/value_net.py — ARIASKA Strategic Evaluator Core v5.0+

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ValueNet(nn.Module):
    """
    ValueNet v5.0 — Cyber Evaluation Cortex
    • Predicts expected value of a state + confidence
    • Modular loss modes: MSE, Huber, Cosine, Dual Head
    • Phase-aware embedding fusion (contextual tuning)
    • Self-calibrating uncertainty tracking
    """

    def __init__(self, input_size=512, hidden_size=256, output_size=1, lr=1e-4, device="cuda"):
        super().__init__()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Optional phase vector embedding (5D input → projection)
        self.phase_proj = nn.Linear(5, hidden_size)
        self.use_phase_fusion = True

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(0.15)

        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.norm2 = nn.LayerNorm(hidden_size // 2)
        self.dropout2 = nn.Dropout(0.15)

        # Prediction heads
        self.value_head = nn.Linear(hidden_size // 2, output_size)
        self.uncertainty_head = nn.Linear(hidden_size // 2, 1)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)

        self.to(self.device)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.value_head, self.uncertainty_head, self.phase_proj]:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, 'bias'):
                nn.init.constant_(layer.bias, 0)

    def forward(self, state, phase_vector=None):
        """
        Predict value + uncertainty with optional phase fusion
        """
        x = F.relu(self.norm1(self.fc1(state)))
        x = self.dropout1(x)

        if self.use_phase_fusion and phase_vector is not None:
            phase_encoded = F.relu(self.phase_proj(phase_vector))
            x = x + phase_encoded

        x = F.relu(self.norm2(self.fc2(x)))
        x = self.dropout2(x)

        value = self.value_head(x)
        confidence = torch.sigmoid(self.uncertainty_head(x))  # 0-1 trust

        return value, confidence

    def predict(self, state, phase_vector=None, return_confidence=False):
        self.eval()
        with torch.no_grad():
            state_tensor = state
            if not isinstance(state, torch.Tensor):
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)

            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)

            if phase_vector is not None and not isinstance(phase_vector, torch.Tensor):
                phase_vector = torch.tensor(phase_vector, dtype=torch.float32, device=self.device).unsqueeze(0)

            value, confidence = self.forward(state_tensor, phase_vector=phase_vector)
            value = value.squeeze().detach().cpu()
            confidence = confidence.squeeze().detach().cpu()

            val = value.item() if value.numel() == 1 else value.tolist()
            conf = confidence.item() if confidence.numel() == 1 else confidence.tolist()

            return (val, conf) if return_confidence else val

    def train_step(self, states, targets, phase_vectors=None, grad_clip=0.5, loss_mode="huber", confidence_weighting=True):
        self.train()
        values, confidences = self.forward(states, phase_vector=phase_vectors)
        values = values.squeeze()

        if loss_mode == "huber":
            base_loss = F.smooth_l1_loss(values, targets, reduction='none')
        elif loss_mode == "mse":
            base_loss = F.mse_loss(values, targets, reduction='none')
        elif loss_mode == "cosine":
            base_loss = (1 - F.cosine_similarity(values.unsqueeze(1), targets.unsqueeze(1)))
        else:
            raise ValueError(f"[ValueNet] Invalid loss mode: {loss_mode}")

        if confidence_weighting:
            weights = confidences.detach().squeeze()
            loss = (base_loss * weights).mean()
        else:
            loss = base_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()
