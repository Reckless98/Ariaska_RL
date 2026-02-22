"""core/algorithms/transfer_learning.py — Phase 48: Transfer Learning for PPO.

Train on easy targets → freeze backbone → fine-tune heads on hard targets.
Implements progressive unfreezing, domain adaptation, and skill transfer
across different attack scenarios.

Training modes:
  1. PRETRAIN: Train full network on easy scenarios (ms2_easy, generic_linux)
  2. FREEZE: Freeze backbone, only train actor/critic heads on new target
  3. PROGRESSIVE: Gradually unfreeze layers from heads → backbone → input
  4. FINE_TUNE: Fine-tune full network with reduced LR on final target

Layer groups (from PPOActorCritic):
  - Group 0: input_proj + input_norm (input embedding)
  - Group 1: shared_backbone (feature extraction)
  - Group 2: adv_residual / residual (attention/residual)
  - Group 3: actor head + phase_gates (policy)
  - Group 4: critic head + phase_predictor (value)

Progressive unfreezing order: 4 → 3 → 2 → 1 → 0
(most specific heads first, input last)
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("ariaska.algorithms.transfer_learning")

# Layer group definitions (matching PPOActorCritic structure)
LAYER_GROUPS: Dict[str, List[str]] = {
    "input": ["input_proj", "input_norm"],
    "backbone": ["shared_backbone"],
    "residual": ["adv_residual", "residual"],
    "actor": ["actor", "phase_gates"],
    "critic": ["critic", "phase_predictor"],
}

# Progressive unfreezing order (heads first, input last)
UNFREEZE_ORDER: List[str] = ["critic", "actor", "residual", "backbone", "input"]

# Difficulty tiers for scenario-based transfer
DIFFICULTY_TIERS: Dict[str, List[str]] = {
    "easy": ["ms2_easy", "generic_linux", "htb_web_easy"],
    "medium": ["ms3_medium", "htb_web_medium", "ctf_web", "htb_privesc_linux"],
    "hard": [
        "htb_web_hard", "htb_ad_windows", "htb_privesc_windows",
        "lateral_network", "cloud_aws",
    ],
}


@dataclass
class TransferConfig:
    """Configuration for transfer learning.

    Args:
        enabled: Master switch.
        mode: Transfer mode — 'pretrain', 'freeze', 'progressive', 'fine_tune'.
        source_scenarios: Scenarios to pretrain on.
        target_scenarios: Target scenarios to transfer to.
        freeze_groups: Layer groups to freeze (only for 'freeze' mode).
        unfreeze_schedule: Steps at which to unfreeze each group.
        backbone_lr_scale: LR multiplier for backbone during fine-tuning.
        head_lr_scale: LR multiplier for heads during fine-tuning.
        warmup_steps: Steps to warm up after unfreezing a group.
        pretrain_steps: Steps for pretraining phase.
        max_frozen_ratio: Max ratio of parameters that can be frozen.
        checkpoint_dir: Directory for transfer checkpoints.
    """
    enabled: bool = True
    mode: str = "progressive"  # pretrain | freeze | progressive | fine_tune
    source_scenarios: List[str] = field(default_factory=lambda: DIFFICULTY_TIERS["easy"])
    target_scenarios: List[str] = field(default_factory=lambda: DIFFICULTY_TIERS["hard"])
    freeze_groups: List[str] = field(default_factory=lambda: ["input", "backbone"])
    unfreeze_schedule: Dict[str, int] = field(default_factory=lambda: {
        "critic": 0,       # Immediately trainable
        "actor": 0,        # Immediately trainable
        "residual": 100,   # Unfreeze after 100 steps
        "backbone": 300,   # Unfreeze after 300 steps
        "input": 500,      # Unfreeze after 500 steps
    })
    backbone_lr_scale: float = 0.1    # Backbone learns 10x slower
    head_lr_scale: float = 1.0        # Heads learn at full LR
    warmup_steps: int = 20
    pretrain_steps: int = 500
    max_frozen_ratio: float = 0.8     # Never freeze > 80% of params
    checkpoint_dir: str = "models/transfer"


class LayerFreezer:
    """Manages freezing/unfreezing of neural network layer groups.

    Provides fine-grained control over which parts of the network
    are trainable, enabling transfer learning patterns.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._frozen_groups: set = set()
        self._param_counts: Dict[str, int] = {}
        self._total_params = sum(p.numel() for p in model.parameters())

        # Map layer groups to actual module names
        self._group_modules: Dict[str, List[nn.Module]] = {}
        for group_name, patterns in LAYER_GROUPS.items():
            modules: List[nn.Module] = []
            for pattern in patterns:
                if hasattr(model, pattern):
                    mod = getattr(model, pattern)
                    if isinstance(mod, nn.Module):
                        modules.append(mod)
            self._group_modules[group_name] = modules
            self._param_counts[group_name] = sum(
                p.numel() for m in modules for p in m.parameters()
            )

    def freeze_group(self, group_name: str) -> int:
        """Freeze all parameters in a layer group.

        Args:
            group_name: One of 'input', 'backbone', 'residual', 'actor', 'critic'.

        Returns:
            Number of parameters frozen.
        """
        if group_name not in self._group_modules:
            logger.warning("Unknown layer group: %s", group_name)
            return 0

        count = 0
        for module in self._group_modules[group_name]:
            for param in module.parameters():
                param.requires_grad = False
                count += param.numel()

        self._frozen_groups.add(group_name)
        logger.info("Froze layer group '%s' (%d params)", group_name, count)
        return count

    def unfreeze_group(self, group_name: str) -> int:
        """Unfreeze all parameters in a layer group.

        Args:
            group_name: Layer group to unfreeze.

        Returns:
            Number of parameters unfrozen.
        """
        if group_name not in self._group_modules:
            logger.warning("Unknown layer group: %s", group_name)
            return 0

        count = 0
        for module in self._group_modules[group_name]:
            for param in module.parameters():
                param.requires_grad = True
                count += param.numel()

        self._frozen_groups.discard(group_name)
        logger.info("Unfroze layer group '%s' (%d params)", group_name, count)
        return count

    def freeze_all_except(self, groups: List[str]) -> int:
        """Freeze everything except the specified groups.

        Args:
            groups: Layer groups to keep trainable.

        Returns:
            Total number of frozen parameters.
        """
        total_frozen = 0
        for group_name in LAYER_GROUPS:
            if group_name not in groups:
                total_frozen += self.freeze_group(group_name)
        return total_frozen

    def unfreeze_all(self) -> int:
        """Unfreeze all parameters."""
        count = 0
        for group_name in list(self._frozen_groups):
            count += self.unfreeze_group(group_name)
        # Also handle any params not in named groups
        for param in self.model.parameters():
            if not param.requires_grad:
                param.requires_grad = True
                count += param.numel()
        return count

    @property
    def frozen_ratio(self) -> float:
        """Ratio of frozen parameters to total parameters."""
        if self._total_params == 0:
            return 0.0
        frozen = sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad
        )
        return frozen / self._total_params

    @property
    def frozen_groups(self) -> set:
        """Set of currently frozen group names."""
        return set(self._frozen_groups)

    def get_param_groups_for_optimizer(
        self,
        base_lr: float,
        backbone_scale: float = 0.1,
        head_scale: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Create parameter groups with per-group learning rates.

        Useful for fine-tuning: backbone gets lower LR, heads get higher LR.

        Args:
            base_lr: Base learning rate.
            backbone_scale: LR multiplier for backbone/input groups.
            head_scale: LR multiplier for actor/critic groups.

        Returns:
            List of param group dicts for torch.optim.
        """
        param_groups = []
        seen_params: set = set()

        lr_scales: Dict[str, float] = {
            "input": backbone_scale,
            "backbone": backbone_scale,
            "residual": backbone_scale * 2,  # Slightly higher than pure backbone
            "actor": head_scale,
            "critic": head_scale,
        }

        for group_name, modules in self._group_modules.items():
            params = []
            for module in modules:
                for p in module.parameters():
                    if p.requires_grad and id(p) not in seen_params:
                        params.append(p)
                        seen_params.add(id(p))
            if params:
                scale = lr_scales.get(group_name, 1.0)
                param_groups.append({
                    "params": params,
                    "lr": base_lr * scale,
                    "name": group_name,
                })

        # Catch any remaining parameters
        remaining = [
            p for p in self.model.parameters()
            if p.requires_grad and id(p) not in seen_params
        ]
        if remaining:
            param_groups.append({
                "params": remaining,
                "lr": base_lr,
                "name": "other",
            })

        return param_groups

    def get_stats(self) -> Dict[str, Any]:
        """Return freezer statistics."""
        return {
            "total_params": self._total_params,
            "frozen_ratio": self.frozen_ratio,
            "frozen_groups": list(self._frozen_groups),
            "trainable_groups": [
                g for g in LAYER_GROUPS if g not in self._frozen_groups
            ],
            "param_counts": self._param_counts,
        }


class TransferLearning:
    """Transfer learning manager for PPO agents.

    Coordinates the full transfer pipeline:
    1. Pretrain on easy scenarios
    2. Freeze backbone
    3. Progressively unfreeze while training on hard scenarios
    4. Fine-tune with differential learning rates

    Usage::

        transfer = TransferLearning(config)
        transfer.setup(model)

        # During training:
        if transfer.should_unfreeze(step):
            transfer.step_unfreeze(step)

        # Get optimizer params with differential LR:
        param_groups = transfer.get_param_groups(base_lr=3e-4)
        optimizer = torch.optim.Adam(param_groups)
    """

    def __init__(self, config: Optional[TransferConfig] = None) -> None:
        self.config = config or TransferConfig()
        self._freezer: Optional[LayerFreezer] = None
        self._step: int = 0
        self._phase: str = "init"  # init, pretrain, freeze, progressive, fine_tune
        self._unfreeze_log: List[Dict[str, Any]] = []

    def setup(self, model: nn.Module) -> None:
        """Initialize transfer learning for a model.

        Call this once before training begins.

        Args:
            model: PPOActorCritic or compatible nn.Module.
        """
        self._freezer = LayerFreezer(model)

        if self.config.mode == "pretrain":
            # Full network trainable for pretraining
            self._freezer.unfreeze_all()
            self._phase = "pretrain"

        elif self.config.mode == "freeze":
            # Freeze specified groups
            for group in self.config.freeze_groups:
                self._freezer.freeze_group(group)
            self._phase = "freeze"

        elif self.config.mode == "progressive":
            # Start with everything frozen except heads
            self._freezer.freeze_all_except(["actor", "critic"])
            self._phase = "progressive"

        elif self.config.mode == "fine_tune":
            # Full network trainable with differential LR
            self._freezer.unfreeze_all()
            self._phase = "fine_tune"

        logger.info(
            "Transfer learning setup: mode=%s, frozen=%.1f%%",
            self.config.mode, self._freezer.frozen_ratio * 100,
        )

    def should_unfreeze(self, step: int) -> bool:
        """Check if any groups should be unfrozen at this step."""
        if self._phase != "progressive" or self._freezer is None:
            return False

        for group_name, unfreeze_step in self.config.unfreeze_schedule.items():
            if (group_name in self._freezer.frozen_groups
                    and step >= unfreeze_step):
                return True
        return False

    def step_unfreeze(self, step: int) -> List[str]:
        """Unfreeze groups according to schedule.

        Args:
            step: Current training step.

        Returns:
            List of group names that were unfrozen.
        """
        if self._freezer is None:
            return []

        unfrozen: List[str] = []
        for group_name in UNFREEZE_ORDER:
            unfreeze_step = self.config.unfreeze_schedule.get(group_name, float("inf"))
            if (group_name in self._freezer.frozen_groups
                    and step >= unfreeze_step):
                self._freezer.unfreeze_group(group_name)
                unfrozen.append(group_name)
                self._unfreeze_log.append({
                    "step": step,
                    "group": group_name,
                    "frozen_ratio": self._freezer.frozen_ratio,
                })

        if unfrozen:
            logger.info(
                "Step %d: unfroze groups %s (frozen=%.1f%%)",
                step, unfrozen, self._freezer.frozen_ratio * 100,
            )

        # If everything is unfrozen, switch to fine_tune phase
        if self._freezer.frozen_ratio == 0.0:
            self._phase = "fine_tune"

        return unfrozen

    def get_param_groups(self, base_lr: float) -> List[Dict[str, Any]]:
        """Get parameter groups with differential learning rates.

        Args:
            base_lr: Base learning rate.

        Returns:
            Parameter groups for optimizer.
        """
        if self._freezer is None:
            return [{"params": [], "lr": base_lr}]

        return self._freezer.get_param_groups_for_optimizer(
            base_lr=base_lr,
            backbone_scale=self.config.backbone_lr_scale,
            head_scale=self.config.head_lr_scale,
        )

    def save_checkpoint(
        self,
        model: nn.Module,
        step: int,
        scenario: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save transfer learning checkpoint.

        Args:
            model: Model to save.
            step: Current step.
            scenario: Scenario name for labeling.
            extra: Additional metadata.

        Returns:
            Path to saved checkpoint.
        """
        from pathlib import Path
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        filename = f"transfer_{self.config.mode}_step{step}"
        if scenario:
            filename += f"_{scenario}"
        filename += ".pt"

        path = ckpt_dir / filename

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "transfer_config": self.config.__dict__,
            "step": step,
            "phase": self._phase,
            "frozen_groups": list(self._freezer.frozen_groups) if self._freezer else [],
            "frozen_ratio": self._freezer.frozen_ratio if self._freezer else 0.0,
        }
        if extra:
            checkpoint["extra"] = extra

        torch.save(checkpoint, path)
        logger.info("Transfer checkpoint saved: %s", path)
        return str(path)

    def load_pretrained(
        self,
        model: nn.Module,
        checkpoint_path: str,
        strict: bool = False,
    ) -> Dict[str, Any]:
        """Load pretrained weights into model for transfer.

        Args:
            model: Target model.
            checkpoint_path: Path to pretrained checkpoint.
            strict: Whether to require exact key match.

        Returns:
            Checkpoint metadata.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Filter to matching keys (for cross-architecture transfer)
        model_keys = set(model.state_dict().keys())
        filtered = {
            k: v for k, v in state_dict.items()
            if k in model_keys and v.shape == model.state_dict()[k].shape
        }

        missing = model_keys - set(filtered.keys())
        if missing:
            logger.info("Transfer: %d keys not loaded (missing/shape mismatch)", len(missing))

        model.load_state_dict(filtered, strict=False)
        logger.info(
            "Loaded pretrained weights: %d/%d keys transferred",
            len(filtered), len(model_keys),
        )

        return {
            "keys_transferred": len(filtered),
            "keys_missing": len(missing),
            "source_step": checkpoint.get("step", -1),
            "source_phase": checkpoint.get("phase", "unknown"),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return transfer learning statistics."""
        stats: Dict[str, Any] = {
            "mode": self.config.mode,
            "phase": self._phase,
            "unfreeze_log": self._unfreeze_log[-5:],
        }
        if self._freezer:
            stats["freezer"] = self._freezer.get_stats()
        return stats
