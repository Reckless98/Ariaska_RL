"""core/algorithms/pbt.py — Phase 49: Population-Based Training.

Evolves hyperparameters during training via tournament selection.
Spawns a population of PPO configs, evaluates fitness, clones winners
and mutates losers.

Design:
  - Population of N agents with different hyperparameters.
  - Periodic evaluation → tournament selection → exploit + explore.
  - Exploit: copy weights from top performer.
  - Explore: perturb hyperparameters of copied agent.
  - Focused on PPO-relevant params: lr, clip_epsilon, entropy_coef, etc.

Integration:
  - Used by SmartOrchestrator in multi-episode training sweeps.
  - Each population member is a PPOConfig + fitness score.
  - Best config used for production agent.

Reference: Jaderberg et al. "Population Based Training of Neural Networks" 2017.
"""
from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ariaska.algorithms.pbt")


@dataclass
class PBTConfig:
    """Configuration for population-based training.

    Args:
        population_size: Number of agents in the population.
        eval_interval: Steps between evaluations.
        exploit_top_frac: Fraction of population considered 'top'.
        explore_perturb_factor: Magnitude of perturbation (0.0-1.0).
        max_generations: Maximum number of PBT generations.
        seed: Random seed.
    """
    population_size: int = 8
    eval_interval: int = 50
    exploit_top_frac: float = 0.2
    explore_perturb_factor: float = 0.2
    max_generations: int = 100
    seed: int = 42


@dataclass
class HyperConfig:
    """Hyperparameter configuration for a single PBT member.

    Focused on PPO-tunable parameters.
    """
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.97
    minibatch_size: int = 16
    epochs_per_update: int = 4
    max_grad_norm: float = 0.5
    sil_coef: float = 0.25
    value_loss_coef: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "clip_epsilon": self.clip_epsilon,
            "entropy_coef": self.entropy_coef,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "minibatch_size": self.minibatch_size,
            "epochs_per_update": self.epochs_per_update,
            "max_grad_norm": self.max_grad_norm,
            "sil_coef": self.sil_coef,
            "value_loss_coef": self.value_loss_coef,
        }


# Ranges for perturbation
HYPERPARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "learning_rate": (1e-5, 1e-3),
    "clip_epsilon": (0.1, 0.3),
    "entropy_coef": (0.001, 0.05),
    "gamma": (0.95, 0.999),
    "gae_lambda": (0.9, 0.99),
    "max_grad_norm": (0.3, 1.0),
    "sil_coef": (0.05, 0.5),
    "value_loss_coef": (0.25, 1.0),
}


@dataclass
class PopulationMember:
    """A single member of the PBT population.

    Args:
        member_id: Unique identifier.
        hyperparams: Current hyperparameter configuration.
        fitness: Current fitness score (higher = better).
        generation: How many times this member has been evolved.
        weights_snapshot: Optional serialised model weights.
        history: Fitness history over generations.
    """
    member_id: int
    hyperparams: HyperConfig
    fitness: float = 0.0
    generation: int = 0
    weights_snapshot: Optional[Dict[str, Any]] = None
    history: List[float] = field(default_factory=list)

    def record_fitness(self, fitness: float) -> None:
        """Record a fitness evaluation."""
        self.fitness = fitness
        self.history.append(fitness)


class PBTManager:
    """Population-Based Training manager.

    Manages a population of hyperparameter configurations,
    performs tournament selection, and evolves the population.

    Usage::

        pbt = PBTManager(PBTConfig(population_size=8))
        population = pbt.initialize_population()

        for gen in range(max_gens):
            for member in population:
                # Train agent with member.hyperparams
                fitness = evaluate_agent(member)
                member.record_fitness(fitness)

            population = pbt.evolve(population)
            best = pbt.get_best(population)
    """

    def __init__(self, config: Optional[PBTConfig] = None) -> None:
        self.config = config or PBTConfig()
        self._rng = random.Random(self.config.seed)
        self._generation: int = 0
        self._best_ever_fitness: float = float("-inf")
        self._best_ever_config: Optional[HyperConfig] = None
        logger.info(
            f"PBTManager initialised: pop={self.config.population_size}, "
            f"perturb={self.config.explore_perturb_factor}"
        )

    def initialize_population(self) -> List[PopulationMember]:
        """Create initial population with diverse hyperparameters.

        Returns:
            List of PopulationMember instances.
        """
        population: List[PopulationMember] = []
        for i in range(self.config.population_size):
            hp = self._random_hyperparams() if i > 0 else HyperConfig()
            member = PopulationMember(member_id=i, hyperparams=hp)
            population.append(member)

        logger.info(f"Initialised population with {len(population)} members")
        return population

    def _random_hyperparams(self) -> HyperConfig:
        """Generate random hyperparameters within ranges."""
        hp = HyperConfig()
        for param_name, (lo, hi) in HYPERPARAM_RANGES.items():
            if param_name == "learning_rate":
                # Log-uniform for learning rate
                import math
                val = math.exp(
                    self._rng.uniform(math.log(lo), math.log(hi))
                )
            else:
                val = self._rng.uniform(lo, hi)
            setattr(hp, param_name, val)
        return hp

    def evolve(
        self,
        population: List[PopulationMember],
    ) -> List[PopulationMember]:
        """Evolve the population via exploit + explore.

        Bottom performers copy top performers and perturb hyperparams.

        Args:
            population: Current population.

        Returns:
            Evolved population.
        """
        self._generation += 1

        # Sort by fitness (descending)
        sorted_pop = sorted(population, key=lambda m: m.fitness, reverse=True)

        n = len(sorted_pop)
        top_k = max(1, int(n * self.config.exploit_top_frac))
        bottom_k = max(1, int(n * self.config.exploit_top_frac))

        # Track best ever
        if sorted_pop[0].fitness > self._best_ever_fitness:
            self._best_ever_fitness = sorted_pop[0].fitness
            self._best_ever_config = copy.deepcopy(sorted_pop[0].hyperparams)

        # Exploit + Explore for bottom performers
        for i in range(n - bottom_k, n):
            member = sorted_pop[i]
            # Pick a random top performer to exploit
            parent = self._rng.choice(sorted_pop[:top_k])

            # Exploit: copy hyperparams (and weights if available)
            member.hyperparams = copy.deepcopy(parent.hyperparams)
            if parent.weights_snapshot is not None:
                member.weights_snapshot = copy.deepcopy(parent.weights_snapshot)

            # Explore: perturb hyperparams
            member.hyperparams = self._perturb(member.hyperparams)
            member.generation = self._generation

            logger.debug(
                f"Member {member.member_id} cloned from {parent.member_id} "
                f"(fitness {parent.fitness:.3f} → perturbed)"
            )

        return sorted_pop

    def _perturb(self, hp: HyperConfig) -> HyperConfig:
        """Perturb hyperparameters by the explore factor.

        Each parameter is multiplied by a random factor in
        [1-perturb, 1+perturb], then clamped to valid range.
        """
        import math

        hp = copy.deepcopy(hp)
        factor = self.config.explore_perturb_factor

        for param_name, (lo, hi) in HYPERPARAM_RANGES.items():
            current = getattr(hp, param_name)
            if param_name == "learning_rate":
                # Perturb in log space
                log_val = math.log(max(current, 1e-10))
                log_val += self._rng.gauss(0, factor)
                new_val = math.exp(log_val)
            else:
                multiplier = 1.0 + self._rng.uniform(-factor, factor)
                new_val = current * multiplier
            # Clamp
            new_val = max(lo, min(hi, new_val))
            setattr(hp, param_name, new_val)

        return hp

    def get_best(
        self,
        population: List[PopulationMember],
    ) -> PopulationMember:
        """Return the best-performing member."""
        return max(population, key=lambda m: m.fitness)

    def get_best_ever(self) -> Tuple[float, Optional[HyperConfig]]:
        """Return the best fitness and config ever seen."""
        return self._best_ever_fitness, self._best_ever_config

    def get_diversity(
        self,
        population: List[PopulationMember],
    ) -> Dict[str, float]:
        """Compute hyperparameter diversity across population.

        Returns:
            Dict mapping param name to coefficient of variation.
        """
        import math

        diversity: Dict[str, float] = {}
        for param_name in HYPERPARAM_RANGES:
            values = [getattr(m.hyperparams, param_name) for m in population]
            mean_val = sum(values) / len(values)
            if mean_val == 0:
                diversity[param_name] = 0.0
                continue
            variance = sum((v - mean_val) ** 2 for v in values) / len(values)
            std = math.sqrt(variance)
            diversity[param_name] = std / abs(mean_val)  # CV

        return diversity

    def get_stats(self) -> Dict[str, Any]:
        """Return PBT statistics."""
        return {
            "generation": self._generation,
            "population_size": self.config.population_size,
            "best_ever_fitness": self._best_ever_fitness,
            "best_ever_config": (
                self._best_ever_config.to_dict()
                if self._best_ever_config
                else None
            ),
        }
