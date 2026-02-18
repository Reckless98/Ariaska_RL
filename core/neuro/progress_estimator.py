#!/usr/bin/env python3
"""
core/neuro/progress_estimator.py — Phase 16.0: Proprioceptive Progress Estimation

Answers the question: "How close am I to foothold / root?"
Two continuous signals in [0, 1]:
  - foothold_progress: estimated proximity to first shell
  - root_progress: estimated proximity to root-level access

Architecture:
  - ProgressMLP: 512 → 128 → 64 → 2 (~40K params, <100ms CPU)
  - ProgressDataset: circular buffer (5000 entries), JSONL persistence
  - GPT labels the *past* (accurate retroactive labeling)
  - MLP predicts the *present* (fast inference, no API call)
  - Confidence gating: output is weighted by min(1.0, dataset_size / 500)
  - Autonomy schedule: as confidence grows, GPT labeling frequency decreases

This is the proprioception/interoception layer — the agent's self-awareness
of how much progress it has actually made toward its objectives.

Feature-flag gated: FF_PROGRESS_ESTIMATOR (default OFF).

Author: Phase 16.0 Contract
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ariaska.neuro.progress_estimator")

# ── Constants ───────────────────────────────────────────────────────────────

_DATASET_MAX = 5000       # circular buffer capacity
_MIN_TRAIN_SAMPLES = 20   # minimum samples before MLP training
_CONFIDENCE_RAMP = 500    # dataset_size / _CONFIDENCE_RAMP → confidence [0, 1]
_MLP_EPOCHS = 5           # training epochs per call
_MLP_LR = 1e-3            # MLP learning rate
_MLP_BATCH = 32           # MLP minibatch size
_AUTONOMY_THRESHOLD_1 = 0.7   # confidence > this → label every 3rd episode
_AUTONOMY_THRESHOLD_2 = 0.9   # confidence > this → label every 10th episode
_MOMENTUM_ALPHA = 0.3     # EMA for progress momentum


# ── Schemas ─────────────────────────────────────────────────────────────────

@dataclass
class ProgressLabel:
    """Ground-truth label for a state vector (from GPT or heuristic)."""
    foothold_progress: float = 0.0   # [0, 1]
    root_progress: float = 0.0       # [0, 1]
    source: str = "heuristic"        # "gpt", "heuristic"
    episode_id: str = ""
    step: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProgressLabel":
        return cls(
            foothold_progress=float(d.get("foothold_progress", 0.0)),
            root_progress=float(d.get("root_progress", 0.0)),
            source=str(d.get("source", "heuristic")),
            episode_id=str(d.get("episode_id", "")),
            step=int(d.get("step", 0)),
            timestamp=float(d.get("timestamp", 0.0)),
        )


@dataclass
class ProgressEstimate:
    """Output of progress estimation (MLP or heuristic fallback)."""
    foothold_progress: float = 0.0   # [0, 1]
    root_progress: float = 0.0       # [0, 1]
    confidence: float = 0.0          # [0, 1] — min(1.0, dataset_size / 500)
    delta: float = 0.0               # change from previous estimate
    momentum: float = 0.0            # EMA-smoothed delta
    source: str = "heuristic"        # "mlp", "heuristic"

    @property
    def combined(self) -> float:
        """Weighted combined progress: 60% foothold + 40% root."""
        return 0.6 * self.foothold_progress + 0.4 * self.root_progress

    def to_dict(self) -> Dict[str, Any]:
        return {
            "foothold_progress": round(self.foothold_progress, 4),
            "root_progress": round(self.root_progress, 4),
            "confidence": round(self.confidence, 4),
            "delta": round(self.delta, 4),
            "momentum": round(self.momentum, 4),
            "combined": round(self.combined, 4),
            "source": self.source,
        }


# ── Dataset ─────────────────────────────────────────────────────────────────

class ProgressDataset:
    """
    Circular buffer of (state_vector, label) pairs with JSONL persistence.

    Stores flattened 512-dim vectors as lists alongside ProgressLabel dicts.
    Capacity: 5000 entries (oldest evicted on overflow).
    """

    def __init__(self, capacity: int = _DATASET_MAX) -> None:
        self._capacity = capacity
        self._entries: List[Dict[str, Any]] = []  # [{state: [...], label: {...}}, ...]

    @property
    def size(self) -> int:
        return len(self._entries)

    def add(self, state_vector: List[float], label: ProgressLabel) -> None:
        """Add a (state, label) pair, evicting oldest if at capacity."""
        entry = {
            "state": state_vector[:512],  # ensure bounded
            "label": label.to_dict(),
        }
        self._entries.append(entry)
        if len(self._entries) > self._capacity:
            self._entries = self._entries[-self._capacity:]

    def get_training_data(self) -> Tuple[List[List[float]], List[Tuple[float, float]]]:
        """Return (X, Y) lists for MLP training."""
        X: List[List[float]] = []
        Y: List[Tuple[float, float]] = []
        for entry in self._entries:
            X.append(entry["state"])
            lbl = entry["label"]
            Y.append((
                float(lbl.get("foothold_progress", 0.0)),
                float(lbl.get("root_progress", 0.0)),
            ))
        return X, Y

    def save(self, path: Path) -> None:
        """Persist dataset to JSONL file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for entry in self._entries:
                    f.write(json.dumps(entry, default=str) + "\n")
            logger.debug(f"[P16] Saved {self.size} progress labels to {path}")
        except Exception as e:
            logger.warning(f"[P16] Failed to save progress dataset: {e}")

    def load(self, path: Path) -> None:
        """Load dataset from JSONL file."""
        if not path.exists():
            return
        try:
            entries: List[Dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
            # Keep only most recent entries up to capacity
            self._entries = entries[-self._capacity:]
            logger.info(f"[P16] Loaded {self.size} progress labels from {path}")
        except Exception as e:
            logger.warning(f"[P16] Failed to load progress dataset: {e}")

    def clear(self) -> None:
        self._entries.clear()


# ── MLP ─────────────────────────────────────────────────────────────────────

class ProgressMLP:
    """
    Lightweight MLP: 512 → 128 → 64 → 2 (sigmoid).

    ~40K parameters, <100ms inference on CPU.
    Outputs: [foothold_progress, root_progress] in [0, 1].
    """

    def __init__(self, input_dim: int = 512) -> None:
        self._input_dim = input_dim
        self._model: Any = None  # torch.nn.Sequential, lazy-built
        self._optimizer: Any = None
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    def _ensure_model(self) -> None:
        """Lazy-build the model on first use."""
        if self._model is not None:
            return
        try:
            import torch
            import torch.nn as nn
            self._model = nn.Sequential(
                nn.Linear(self._input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.Sigmoid(),
            )
            self._optimizer = torch.optim.Adam(
                self._model.parameters(), lr=_MLP_LR
            )
            # Count params for logging
            n_params = sum(p.numel() for p in self._model.parameters())
            logger.info(f"[P16] ProgressMLP built: {n_params} parameters")
        except ImportError:
            logger.warning("[P16] PyTorch not available — MLP disabled")

    def predict(self, state_vector: List[float]) -> Tuple[float, float]:
        """
        Predict (foothold_progress, root_progress) from a state vector.

        Returns (0.0, 0.0) if model is not trained.
        """
        if not self._trained or self._model is None:
            return (0.0, 0.0)
        try:
            import torch
            self._model.eval()
            with torch.no_grad():
                x = torch.tensor(state_vector[:self._input_dim], dtype=torch.float32)
                out = self._model(x.unsqueeze(0))
                return (float(out[0, 0]), float(out[0, 1]))
        except Exception as e:
            logger.debug(f"[P16] MLP predict failed: {e}")
            return (0.0, 0.0)

    def train_on_dataset(
        self,
        X: List[List[float]],
        Y: List[Tuple[float, float]],
        epochs: int = _MLP_EPOCHS,
        batch_size: int = _MLP_BATCH,
    ) -> Dict[str, float]:
        """
        Train MLP on labelled dataset.

        Returns training metrics: {loss, epochs, samples}.
        """
        if len(X) < _MIN_TRAIN_SAMPLES:
            return {"loss": -1.0, "epochs": 0, "samples": len(X)}

        self._ensure_model()
        if self._model is None:
            return {"loss": -1.0, "epochs": 0, "samples": len(X)}

        try:
            import torch
            import torch.nn as nn

            self._model.train()
            criterion = nn.MSELoss()

            X_t = torch.tensor(X, dtype=torch.float32)
            Y_t = torch.tensor(Y, dtype=torch.float32)

            total_loss = 0.0
            n_batches = 0

            for epoch in range(epochs):
                # Shuffle
                perm = torch.randperm(len(X_t))
                X_t = X_t[perm]
                Y_t = Y_t[perm]

                for i in range(0, len(X_t), batch_size):
                    x_batch = X_t[i:i + batch_size]
                    y_batch = Y_t[i:i + batch_size]

                    self._optimizer.zero_grad()
                    pred = self._model(x_batch)
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    self._optimizer.step()

                    total_loss += loss.item()
                    n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            self._trained = True
            logger.info(
                f"[P16] MLP trained: loss={avg_loss:.4f}, "
                f"epochs={epochs}, samples={len(X)}"
            )
            return {"loss": avg_loss, "epochs": epochs, "samples": len(X)}

        except Exception as e:
            logger.warning(f"[P16] MLP training failed: {e}")
            return {"loss": -1.0, "epochs": 0, "samples": len(X)}


# ── Main Estimator ──────────────────────────────────────────────────────────

class ProgressEstimator:
    """
    Proprioceptive progress estimation — the agent's self-awareness.

    Combines:
      1. Heuristic progress estimation (always available, offline-safe)
      2. GPT retroactive labeling (accurate, budget-gated)
      3. MLP prediction (fast, no API call, trained from GPT labels)

    Chicken-and-egg resolution:
      - Start with heuristic labels → train MLP
      - Add GPT labels when available → MLP improves
      - Confidence ramp gates MLP output
      - Autonomy schedule reduces GPT labeling as MLP matures

    Usage in training loop:
      1. Per-step:  estimate(state_vector, discovery_board) → ProgressEstimate
      2. End-of-ep: label_episode_retroactively(run_trace, gpt_manager) → labels
      3. End-of-ep: train_mlp() → update MLP weights
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        dataset_capacity: int = _DATASET_MAX,
    ) -> None:
        self._dataset = ProgressDataset(capacity=dataset_capacity)
        self._mlp = ProgressMLP()
        self._persist_dir = Path(persist_dir) if persist_dir else Path("models/progress")
        self._prev_estimate: Optional[ProgressEstimate] = None
        self._momentum: float = 0.0
        self._episode_count: int = 0
        self._total_labels: int = 0

        # Load persisted dataset if it exists
        self._dataset.load(self._persist_dir / "progress_labels.jsonl")

        # Train MLP on loaded data if sufficient
        if self._dataset.size >= _MIN_TRAIN_SAMPLES:
            X, Y = self._dataset.get_training_data()
            self._mlp.train_on_dataset(X, Y)

    @property
    def confidence(self) -> float:
        """Confidence in MLP predictions: min(1.0, dataset_size / 500)."""
        return min(1.0, self._dataset.size / _CONFIDENCE_RAMP)

    @property
    def dataset_size(self) -> int:
        return self._dataset.size

    def estimate(
        self,
        state_vector: List[float],
        discovery_board: Optional[Dict[str, Any]] = None,
    ) -> ProgressEstimate:
        """
        Estimate current progress toward foothold and root.

        Uses MLP if trained + confident, otherwise falls back to heuristic.
        Always computes delta and momentum relative to previous estimate.

        Args:
            state_vector: 512-dim state encoding
            discovery_board: Current discovery board dict

        Returns:
            ProgressEstimate with foothold, root, confidence, delta, momentum
        """
        conf = self.confidence

        # Try MLP first (if trained and confidence > 0.3)
        if self._mlp.is_trained and conf > 0.3:
            fp, rp = self._mlp.predict(state_vector)
            source = "mlp"
        else:
            # Heuristic fallback
            fp, rp = self._heuristic_estimate(discovery_board or {})
            source = "heuristic"

        # Compute delta from previous estimate
        prev_combined = self._prev_estimate.combined if self._prev_estimate else 0.0
        current_combined = 0.6 * fp + 0.4 * rp
        delta = current_combined - prev_combined

        # EMA momentum
        self._momentum = (
            (1.0 - _MOMENTUM_ALPHA) * self._momentum
            + _MOMENTUM_ALPHA * delta
        )

        estimate = ProgressEstimate(
            foothold_progress=fp,
            root_progress=rp,
            confidence=conf,
            delta=delta,
            momentum=self._momentum,
            source=source,
        )

        self._prev_estimate = estimate
        return estimate

    def _heuristic_estimate(
        self, discovery_board: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Compute progress from discovery board signals.

        Foothold = weighted sum of recon milestones:
          ports(0.1) + services(0.15) + credentials(0.3) + vulns(0.15) +
          web_paths(0.05) + users(0.05) + shells(0.2)

        Root = conditional on shell:
          no shell → root_progress = foothold * 0.1
          user shell → 0.4 + credential/vuln bonuses
          root shell → 1.0
        """
        if not discovery_board:
            return (0.0, 0.0)

        # Extract counts safely
        def _count(key: str) -> int:
            v = discovery_board.get(key, set())
            if isinstance(v, (set, list)):
                return len(v)
            return 0

        n_ports = _count("ports")
        n_services = _count("services")
        n_creds = _count("credentials")
        n_vulns = _count("vulns")
        n_shells = _count("shells")
        n_users = _count("users")
        n_web = _count("web_paths")
        flags = discovery_board.get("flags_set", set())
        n_flags = len(flags) if isinstance(flags, (set, list)) else 0

        # Foothold progress
        fp = 0.0
        fp += min(n_ports / 5.0, 1.0) * 0.10         # 5+ ports = full credit
        fp += min(n_services / 3.0, 1.0) * 0.15       # 3+ services
        fp += min(n_creds / 1.0, 1.0) * 0.30          # 1+ credential = full
        fp += min(n_vulns / 2.0, 1.0) * 0.15          # 2+ vulns
        fp += min(n_web / 2.0, 1.0) * 0.05            # 2+ web paths
        fp += min(n_users / 2.0, 1.0) * 0.05          # 2+ users
        fp += min(n_shells / 1.0, 1.0) * 0.20         # 1+ shell = full
        fp = min(fp, 1.0)

        # Root progress
        has_root = bool(discovery_board.get("root_shell", False))
        if has_root or n_flags > 0:
            rp = 1.0
        elif n_shells > 0:
            # User shell obtained — root depends on privesc findings
            rp = 0.4
            rp += min(n_creds / 3.0, 1.0) * 0.2       # more creds help privesc
            rp += min(n_vulns / 3.0, 1.0) * 0.2        # more vulns help privesc
            rp += min(n_users / 3.0, 1.0) * 0.1        # more users = lateral
            rp = min(rp, 0.95)  # cap below 1.0 (not root yet)
        else:
            # No shell — root is far
            rp = fp * 0.1
            rp = min(rp, 0.15)

        return (round(fp, 4), round(rp, 4))

    def label_episode_heuristic(
        self,
        episode_states: List[List[float]],
        discovery_boards: List[Dict[str, Any]],
        episode_id: str = "",
    ) -> List[ProgressLabel]:
        """
        Label an entire episode using heuristic estimation.

        Used for offline bootstrap and as a fallback when GPT is unavailable.
        Each step gets a label based on the cumulative discovery board at that step.

        Args:
            episode_states: List of 512-dim state vectors (one per step)
            discovery_boards: List of discovery board dicts (one per step)
            episode_id: Episode identifier

        Returns:
            List of ProgressLabel, one per step
        """
        labels: List[ProgressLabel] = []
        for i, (sv, db) in enumerate(zip(episode_states, discovery_boards)):
            fp, rp = self._heuristic_estimate(db)
            label = ProgressLabel(
                foothold_progress=fp,
                root_progress=rp,
                source="heuristic",
                episode_id=episode_id,
                step=i,
            )
            labels.append(label)
            self._dataset.add(sv, label)
            self._total_labels += 1
        return labels

    def label_episode_retroactively(
        self,
        run_trace: Dict[str, Any],
        gpt_manager: Any = None,
    ) -> List[ProgressLabel]:
        """
        Label an episode retroactively using GPT analysis.

        GPT sees the full episode outcome and labels each step's actual
        progress. This produces high-quality ground-truth that the MLP
        learns to predict in real-time.

        Falls back to heuristic labeling if:
          - gpt_manager is None
          - GPT call fails
          - budget exhausted

        Args:
            run_trace: Episode run trace dict with discoveries, phases, etc.
            gpt_manager: GPTManager instance for API calls

        Returns:
            List of ProgressLabel for the episode
        """
        episode_id = str(run_trace.get("run_id", ""))

        # Extract episode outcome
        discoveries = run_trace.get("discoveries", {})
        phase_progression = run_trace.get("phase_progression", [])
        total_reward = float(run_trace.get("total_reward", 0.0))
        success_rate = float(run_trace.get("success_rate", 0.0))

        # Build discovery board from run trace
        db = {
            "ports": set(discoveries.get("open_port", [])),
            "services": set(discoveries.get("service", [])),
            "credentials": set(discoveries.get("credential", [])),
            "vulns": set(discoveries.get("vulnerability", [])),
            "shells": set(discoveries.get("shell", [])),
            "users": set(discoveries.get("user", [])),
            "web_paths": set(discoveries.get("web_path", [])),
            "flags_set": set(discoveries.get("flag", [])),
            "root_shell": success_rate >= 0.8,
        }

        # Determine final progress (ground truth from outcome)
        final_fp, final_rp = self._heuristic_estimate(db)

        # If GPT available and budget permits, get refined labels
        if gpt_manager is not None:
            try:
                if hasattr(gpt_manager, 'can_make_request') and gpt_manager.can_make_request():
                    gpt_labels = self._gpt_label(
                        run_trace, final_fp, final_rp, gpt_manager
                    )
                    if gpt_labels:
                        self._episode_count += 1
                        return gpt_labels
            except Exception as e:
                logger.debug(f"[P16] GPT labeling failed, using heuristic: {e}")

        # Heuristic fallback: linear ramp to final values
        n_steps = max(int(run_trace.get("total_steps", 1)), 1)
        labels: List[ProgressLabel] = []
        for step in range(n_steps):
            t = (step + 1) / n_steps
            label = ProgressLabel(
                foothold_progress=round(final_fp * t, 4),
                root_progress=round(final_rp * t, 4),
                source="heuristic_retro",
                episode_id=episode_id,
                step=step,
            )
            labels.append(label)
            self._total_labels += 1

        self._episode_count += 1
        return labels

    def _gpt_label(
        self,
        run_trace: Dict[str, Any],
        final_fp: float,
        final_rp: float,
        gpt_manager: Any,
    ) -> Optional[List[ProgressLabel]]:
        """
        Use GPT to produce per-step progress labels.

        Sends a compact episode summary and asks for step-level progress
        estimates. Returns None on failure (caller falls back to heuristic).
        """
        episode_id = str(run_trace.get("run_id", ""))
        n_steps = max(int(run_trace.get("total_steps", 1)), 1)

        # Build compact prompt
        prompt = (
            "You are a penetration testing progress assessor. "
            "An episode had these results:\n"
            f"- Total reward: {run_trace.get('total_reward', 0):.1f}\n"
            f"- Phases reached: {run_trace.get('phase_progression', [])}\n"
            f"- Discoveries: {json.dumps({k: list(v) if isinstance(v, set) else v for k, v in run_trace.get('discoveries', {}).items()}, default=str)[:500]}\n"
            f"- Final progress: foothold={final_fp:.2f}, root={final_rp:.2f}\n"
            f"- Total steps: {n_steps}\n\n"
            f"For each of the {n_steps} steps, estimate the foothold_progress "
            f"and root_progress as floats in [0,1]. Consider that progress is "
            f"gradual during recon, jumps on credential/vuln discoveries, and "
            f"spikes on shell obtainment.\n\n"
            f"Reply ONLY with a JSON array of {n_steps} objects, each with "
            f"'fp' and 'rp' keys. Example: [{{'fp': 0.1, 'rp': 0.0}}, ...]"
        )

        try:
            response = gpt_manager.gpt_request(
                prompt,
                task_type="postmortem",
                agent_id="ProgressEstimator",
            )
            if not response:
                return None

            # Parse JSON array from response
            resp_text = str(response)
            # Find JSON array in response
            start = resp_text.find("[")
            end = resp_text.rfind("]") + 1
            if start < 0 or end <= start:
                return None

            arr = json.loads(resp_text[start:end])
            if not isinstance(arr, list) or len(arr) < 1:
                return None

            labels: List[ProgressLabel] = []
            for i in range(min(n_steps, len(arr))):
                item = arr[i] if i < len(arr) else arr[-1]
                fp = max(0.0, min(1.0, float(item.get("fp", 0.0))))
                rp = max(0.0, min(1.0, float(item.get("rp", 0.0))))
                label = ProgressLabel(
                    foothold_progress=fp,
                    root_progress=rp,
                    source="gpt",
                    episode_id=episode_id,
                    step=i,
                )
                labels.append(label)
                self._total_labels += 1

            # Pad if GPT returned fewer entries than steps
            while len(labels) < n_steps:
                labels.append(ProgressLabel(
                    foothold_progress=final_fp,
                    root_progress=final_rp,
                    source="gpt_padded",
                    episode_id=episode_id,
                    step=len(labels),
                ))
                self._total_labels += 1

            logger.info(f"[P16] GPT labeled {len(labels)} steps for ep {episode_id}")
            return labels

        except Exception as e:
            logger.debug(f"[P16] GPT label parsing failed: {e}")
            return None

    def add_labels_to_dataset(
        self,
        states: List[List[float]],
        labels: List[ProgressLabel],
    ) -> int:
        """
        Add labelled (state, label) pairs to the dataset.

        Args:
            states: List of 512-dim state vectors
            labels: Corresponding ProgressLabel objects

        Returns:
            Number of entries added
        """
        added = 0
        for sv, label in zip(states, labels):
            self._dataset.add(sv, label)
            added += 1
        return added

    def train_mlp(self) -> Dict[str, float]:
        """
        Train the MLP on the current dataset.

        Returns training metrics (loss, epochs, samples, confidence).
        Safe to call even with insufficient data (returns early).
        """
        X, Y = self._dataset.get_training_data()
        metrics = self._mlp.train_on_dataset(X, Y)
        metrics["confidence"] = self.confidence
        metrics["dataset_size"] = self.dataset_size
        return metrics

    def should_gpt_label(self, episode_number: int) -> bool:
        """
        Autonomy schedule: should we request GPT labeling this episode?

        - confidence < 0.7 → label every episode (bootstrap)
        - confidence ∈ [0.7, 0.9) → label every 3rd episode
        - confidence ≥ 0.9 → label every 10th episode
        - Always label episode 1

        This gradually reduces GPT budget consumption as the MLP matures.
        """
        if episode_number <= 1:
            return True

        conf = self.confidence
        if conf < _AUTONOMY_THRESHOLD_1:
            return True
        elif conf < _AUTONOMY_THRESHOLD_2:
            return episode_number % 3 == 0
        else:
            return episode_number % 10 == 0

    def get_autonomy_level(self) -> str:
        """
        Return a human-readable autonomy level.

        Returns one of: "bootstrap", "learning", "autonomous"
        """
        conf = self.confidence
        if conf < _AUTONOMY_THRESHOLD_1:
            return "bootstrap"
        elif conf < _AUTONOMY_THRESHOLD_2:
            return "learning"
        else:
            return "autonomous"

    def save(self) -> None:
        """Persist dataset to disk."""
        self._dataset.save(self._persist_dir / "progress_labels.jsonl")

    def reset_episode(self) -> None:
        """Reset per-episode state (prev estimate and momentum)."""
        self._prev_estimate = None
        self._momentum = 0.0

    def get_metrics(self) -> Dict[str, Any]:
        """Return summary metrics for logging."""
        return {
            "dataset_size": self.dataset_size,
            "confidence": round(self.confidence, 4),
            "mlp_trained": self._mlp.is_trained,
            "total_labels": self._total_labels,
            "episode_count": self._episode_count,
            "autonomy_level": self.get_autonomy_level(),
            "momentum": round(self._momentum, 4),
        }

    def __repr__(self) -> str:
        return (
            f"ProgressEstimator(dataset={self.dataset_size}, "
            f"confidence={self.confidence:.2f}, "
            f"autonomy={self.get_autonomy_level()})"
        )
