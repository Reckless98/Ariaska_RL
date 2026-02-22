"""scripts/hyperparam_sweep.py — Phase 48: Optuna Hyperparameter Optimization.

Automated sweep over PPO hyperparameters using Optuna's TPE sampler
and median pruner. Runs short training trials and selects the best
configuration based on reward-invariant metrics.

Usage::

    # From project root:
    python scripts/hyperparam_sweep.py --trials 50 --steps-per-trial 200
    python scripts/hyperparam_sweep.py --resume --study-name ariaska_ppo

Sweeps over:
    - PPO clip_epsilon: [0.1, 0.3]
    - Learning rate: [1e-5, 1e-3] (log scale)
    - GAE lambda: [0.9, 0.99]
    - Entropy coef: [0.001, 0.05]
    - BC loss coef: [0.0, 0.3]
    - KL teacher coef: [0.0, 0.2]
    - Dropout: [0.0, 0.2]
    - Minibatch size: {8, 16, 32}
    - Hidden dims: {[256,256], [512,256], [512,512,256]}

Objective: maximize (unique_commands * discovery_rate + phase_progress)
           i.e. reward-invariant metrics, not raw reward.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ariaska.scripts.hyperparam_sweep")

# Optional: import optuna lazily
_optuna = None


def _ensure_optuna():
    """Lazy-import optuna, installing if needed."""
    global _optuna
    if _optuna is not None:
        return _optuna
    try:
        import optuna
        _optuna = optuna
        return optuna
    except ImportError:
        raise ImportError(
            "optuna not installed. Run: pip install optuna optuna-dashboard"
        )


@dataclass
class SweepConfig:
    """Configuration for the hyperparameter sweep.

    Args:
        study_name: Optuna study name (for persistence).
        n_trials: Number of trials to run.
        steps_per_trial: Training steps per trial.
        timeout_seconds: Max time per trial (seconds).
        storage: Optuna storage URL (None = in-memory).
        direction: Optimization direction.
        sampler: TPE or Random.
        pruner: Median or Hyperband.
        seed: Random seed.
        output_dir: Directory for sweep results.
    """
    study_name: str = "ariaska_ppo_sweep"
    n_trials: int = 50
    steps_per_trial: int = 200
    timeout_seconds: int = 600
    storage: Optional[str] = None
    direction: str = "maximize"
    sampler: str = "tpe"
    pruner: str = "median"
    seed: int = 42
    output_dir: str = "results/sweeps"


@dataclass
class TrialConfig:
    """Hyperparameter configuration for a single trial."""
    # PPO core
    clip_epsilon: float = 0.2
    learning_rate: float = 3e-4
    gae_lambda: float = 0.97
    gamma: float = 0.99
    entropy_coef: float = 0.01

    # Distillation
    bc_loss_coef: float = 0.1
    kl_teacher_coef: float = 0.15

    # Architecture
    hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 256])
    dropout_rate: float = 0.1

    # Training
    minibatch_size: int = 16
    epochs_per_update: int = 4
    max_grad_norm: float = 0.5

    # Advanced
    use_attention: bool = False
    use_phase_gates: bool = True
    sil_coef: float = 0.25
    value_reg_coef: float = 0.10


def suggest_trial_config(trial: Any) -> TrialConfig:
    """Sample hyperparameters for a trial using Optuna.

    Args:
        trial: Optuna Trial object.

    Returns:
        TrialConfig with sampled parameters.
    """
    # Hidden dims choice
    hidden_choice = trial.suggest_categorical(
        "hidden_dims", ["256_256", "512_256", "512_512_256"]
    )
    hidden_map = {
        "256_256": [256, 256],
        "512_256": [512, 256],
        "512_512_256": [512, 512, 256],
    }

    return TrialConfig(
        clip_epsilon=trial.suggest_float("clip_epsilon", 0.1, 0.3),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        gae_lambda=trial.suggest_float("gae_lambda", 0.9, 0.99),
        gamma=trial.suggest_float("gamma", 0.95, 0.999),
        entropy_coef=trial.suggest_float("entropy_coef", 0.001, 0.05, log=True),
        bc_loss_coef=trial.suggest_float("bc_loss_coef", 0.0, 0.3),
        kl_teacher_coef=trial.suggest_float("kl_teacher_coef", 0.0, 0.2),
        hidden_dims=hidden_map[hidden_choice],
        dropout_rate=trial.suggest_float("dropout_rate", 0.0, 0.2),
        minibatch_size=trial.suggest_categorical("minibatch_size", [8, 16, 32]),
        epochs_per_update=trial.suggest_int("epochs_per_update", 2, 6),
        max_grad_norm=trial.suggest_float("max_grad_norm", 0.3, 1.0),
        use_attention=trial.suggest_categorical("use_attention", [True, False]),
        use_phase_gates=trial.suggest_categorical("use_phase_gates", [True, False]),
        sil_coef=trial.suggest_float("sil_coef", 0.0, 0.5),
        value_reg_coef=trial.suggest_float("value_reg_coef", 0.0, 0.2),
    )


def _apply_trial_to_ppo_config(
    trial_config: TrialConfig,
    base_config: Any,
) -> Any:
    """Apply trial hyperparameters to a PPOConfig instance.

    Args:
        trial_config: Sampled hyperparameters.
        base_config: Base PPOConfig to modify.

    Returns:
        Modified PPOConfig copy.
    """
    config = copy.deepcopy(base_config)
    config.clip_epsilon = trial_config.clip_epsilon
    config.learning_rate = trial_config.learning_rate
    config.gae_lambda = trial_config.gae_lambda
    config.gamma = trial_config.gamma
    config.entropy_coef = trial_config.entropy_coef
    config.bc_loss_coef = trial_config.bc_loss_coef
    config.kl_teacher_coef = trial_config.kl_teacher_coef
    config.hidden_dims = trial_config.hidden_dims
    config.dropout_rate = trial_config.dropout_rate
    config.minibatch_size = trial_config.minibatch_size
    config.epochs_per_update = trial_config.epochs_per_update
    config.max_grad_norm = trial_config.max_grad_norm
    config.use_attention = trial_config.use_attention
    config.use_phase_gates = trial_config.use_phase_gates
    config.sil_coef = trial_config.sil_coef
    config.value_reg_coef = trial_config.value_reg_coef
    return config


def compute_objective(metrics: Dict[str, Any]) -> float:
    """Compute objective value from reward-invariant metrics.

    Prioritizes diversity and discovery over raw reward.

    Score = unique_commands * diversity_ratio * (1 + discovery_count)
            + phase_progress_bonus

    Args:
        metrics: Training metrics dict with reward-invariant fields.

    Returns:
        Scalar objective value to maximize.
    """
    unique_cmds = metrics.get("unique_commands", 0)
    diversity = metrics.get("diversity_ratio", 0.0)
    discoveries = metrics.get("total_discoveries", 0)
    phase_progress = metrics.get("max_phase_reached", 0)

    # Phase bonus (reaching exploitation = 15, privesc = 30, etc.)
    phase_bonus = phase_progress * 5.0

    # Core objective: diversity × discovery synergy
    core = unique_cmds * diversity * (1.0 + discoveries)

    return core + phase_bonus


class HyperparamSweep:
    """Optuna-based hyperparameter optimization for PPO.

    Usage::

        sweep = HyperparamSweep(config)
        sweep.run(train_fn=my_training_function)
        best = sweep.get_best_config()
    """

    def __init__(self, config: Optional[SweepConfig] = None) -> None:
        self.config = config or SweepConfig()
        self._study: Any = None
        self._best_config: Optional[TrialConfig] = None
        self._results: List[Dict[str, Any]] = []

    def _create_study(self) -> Any:
        """Create or load Optuna study."""
        optuna = _ensure_optuna()

        if self.config.sampler == "tpe":
            sampler = optuna.samplers.TPESampler(seed=self.config.seed)
        else:
            sampler = optuna.samplers.RandomSampler(seed=self.config.seed)

        if self.config.pruner == "median":
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
            )
        else:
            pruner = optuna.pruners.HyperbandPruner()

        study = optuna.create_study(
            study_name=self.config.study_name,
            direction=self.config.direction,
            storage=self.config.storage,
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner,
        )
        return study

    def run(
        self,
        train_fn: Callable[[TrialConfig, int], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run the hyperparameter sweep.

        Args:
            train_fn: Callable(trial_config, steps) → metrics_dict.
                     Should create a PPOAgent with the given config,
                     train for `steps` steps, and return metrics.

        Returns:
            Dict with best trial info and all results.
        """
        optuna = _ensure_optuna()
        study = self._create_study()
        self._study = study

        def objective(trial: Any) -> float:
            # Sample hyperparameters
            trial_config = suggest_trial_config(trial)

            # Run training
            try:
                metrics = train_fn(trial_config, self.config.steps_per_trial)
            except Exception as e:
                logger.warning("Trial %d failed: %s", trial.number, e)
                raise optuna.TrialPruned()

            # Compute objective
            score = compute_objective(metrics)

            # Report intermediate values for pruning
            trial.report(score, step=self.config.steps_per_trial)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Store result
            self._results.append({
                "trial": trial.number,
                "config": trial_config.__dict__,
                "metrics": metrics,
                "score": score,
            })

            return score

        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds * self.config.n_trials,
        )

        # Extract best
        best_trial = study.best_trial
        self._best_config = suggest_trial_config(best_trial)

        # Save results
        self._save_results()

        return {
            "best_trial": best_trial.number,
            "best_score": best_trial.value,
            "best_params": best_trial.params,
            "n_trials_completed": len(study.trials),
        }

    def get_best_config(self) -> Optional[TrialConfig]:
        """Return the best TrialConfig found, or None if sweep hasn't run."""
        return self._best_config

    def get_results(self) -> List[Dict[str, Any]]:
        """Return all trial results."""
        return self._results

    def _save_results(self) -> None:
        """Save sweep results to JSON."""
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"sweep_{self.config.study_name}_{timestamp}.json"

        data = {
            "study_name": self.config.study_name,
            "n_trials": len(self._results),
            "best_config": self._best_config.__dict__ if self._best_config else None,
            "results": self._results,
        }

        # Sanitize for JSON (convert non-serializable types)
        def _sanitize(obj: Any) -> Any:
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize(v) for v in obj]
            return str(obj)

        with open(path, "w") as f:
            json.dump(_sanitize(data), f, indent=2)

        logger.info("Sweep results saved to %s", path)


def main() -> None:
    """CLI entry point for hyperparameter sweep."""
    parser = argparse.ArgumentParser(
        description="Ariaska PPO Hyperparameter Sweep (Optuna)"
    )
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--steps-per-trial", type=int, default=200, help="Steps per trial")
    parser.add_argument("--study-name", type=str, default="ariaska_ppo_sweep")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per trial (s)")
    parser.add_argument("--resume", action="store_true", help="Resume existing study")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/sweeps")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = SweepConfig(
        study_name=args.study_name,
        n_trials=args.trials,
        steps_per_trial=args.steps_per_trial,
        timeout_seconds=args.timeout,
        storage=args.storage,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    sweep = HyperparamSweep(config)

    # Dummy training function for validation
    def _dummy_train(trial_config: TrialConfig, steps: int) -> Dict[str, Any]:
        """Placeholder — replace with actual PPO training loop."""
        import random
        random.seed(trial_config.learning_rate * 1e6)  # deterministic seed from config
        return {
            "unique_commands": random.randint(5, 50),
            "diversity_ratio": random.random(),
            "total_discoveries": random.randint(0, 20),
            "max_phase_reached": random.randint(0, 7),
            "mean_reward": random.uniform(-5, 50),
        }

    from rich.console import Console
    console = Console()
    console.print("[bold green]Starting hyperparameter sweep...[/bold green]")
    console.print(f"  Trials: {config.n_trials}")
    console.print(f"  Steps/trial: {config.steps_per_trial}")
    console.print(f"  Study: {config.study_name}")

    result = sweep.run(train_fn=_dummy_train)

    console.print(f"\n[bold green]Sweep complete![/bold green]")
    console.print(f"  Best trial: {result['best_trial']}")
    console.print(f"  Best score: {result['best_score']:.4f}")
    console.print(f"  Best params: {result['best_params']}")


if __name__ == "__main__":
    main()
