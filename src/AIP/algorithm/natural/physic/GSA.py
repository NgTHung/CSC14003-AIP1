"""Gravitational Search Algorithm (GSA) - physics-inspired optimization."""

from dataclasses import dataclass
from typing import override
import numpy as np
from AIP.problems.base_problem import Problem, DiscreteProblem
from AIP.problems.continuous.continuous import ContinuousProblem
from AIP.algorithm.base_algorithm import Algorithm


@dataclass
class GravitationalSearchParameter:
    """
    Configuration parameters for Gravitational Search
    pop_size : int
        Population size (number of agents).
    max_iterations : int
        Maximum number of iterations.
    G0 : float
        Initial gravitational constant.
    alpha : float
        Gravitational constant reduction coefficient.
    """

    iteration: int
    G0: float
    alpha: float
    pop_size: int


class GravitationalSearchAlgorithm(
    Algorithm[Problem, np.ndarray, float | None, GravitationalSearchParameter]
):
    """
    Gravitational Search Algorithm (GSA).

    GSA is based on the law of gravity and mass interactions. Agents (masses)
    attract each other with gravitational force proportional to their masses
    (fitness) and inversely proportional to distance.

    Works with both continuous and discrete (binary) problems.
    For :class:`DiscreteProblem` instances positions are discretized to
    binary values after each update via a sigmoid transfer function.

    Attributes
    ----------
    name : str
        Algorithm name.
    """

    name = "Gravitational Search Algorithm"
    population_history: list[np.ndarray]
    _is_discrete: bool
    _is_permutation: bool

    def __init__(self, configuration: GravitationalSearchParameter, problem: Problem):
        """
        Initialize GSA algorithm.

        Parameters
        ----------
        configuration : Gravitational Search parameter
        problem : Problem
            The optimization problem instance.
        """
        super().__init__(configuration, problem)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Element-wise sigmoid for transfer function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _discretize(self, positions: np.ndarray) -> np.ndarray:
        """Convert continuous positions to binary using sigmoid transfer."""
        probs = self._sigmoid(positions)
        return (np.random.rand(*probs.shape) < probs).astype(float)

    def _to_permutation(self, positions: np.ndarray) -> np.ndarray:
        """Convert continuous positions to valid permutations via random-key encoding.

        Each row is converted by sorting: the rank order of the continuous
        values becomes the permutation.  This preserves the relative
        ordering information stored in the continuous representation.
        """
        return np.argsort(positions, axis=1).astype(float)

    def _calculate_mass(self, fitness: np.ndarray) -> np.ndarray:
        """
        Calculate mass of each agent based on fitness.

        Parameters
        ----------
        fitness : np.ndarray
            Fitness values of all agents.

        Returns
        -------
        np.ndarray
            Mass values normalized to [0, 1].
        """
        best = np.min(fitness)
        worst = np.max(fitness)

        if worst == best:
            return np.ones(len(fitness)) / len(fitness)

        # Normalized mass (better fitness = larger mass)
        m = (fitness - worst) / (best - worst)
        M = m / np.sum(m)

        return M

    def _clamp_positions(self, positions: np.ndarray) -> np.ndarray:
        """Clamp positions to the problem bounds (continuous problems only)."""
        if isinstance(self.problem, ContinuousProblem):
            lb = self.problem._bounds[:, 0]
            ub = self.problem._bounds[:, 1]
            return np.clip(positions, lb, ub)
        return positions

    @override
    def reset(self):
        self._is_discrete = isinstance(self.problem, DiscreteProblem)
        self._is_permutation = (
            self._is_discrete
            and getattr(self.problem, "solution_type", None) == "permutation"
        )
        self.best_fitness = None
        self.best_solution = np.array([])

    @override
    def run(self) -> np.ndarray:
        """
        Execute GSA algorithm.

        Saves structured history for plotting:
        - history: list of dicts with 'iteration', 'best_fitness'
        - best_fitness: best fitness value found

        Returns
        -------
        np.ndarray
            Best solution found.
        """
        self.reset()
        # Initialize population
        positions = self.problem.sample(self.conf.pop_size)
        velocities = np.zeros_like(positions)

        # Evaluate initial population
        fitness = np.array(
            [self.problem.eval(positions[i]) for i in range(self.conf.pop_size)]
        )

        # Track best
        best_idx = np.argmin(fitness)
        best_solution = positions[best_idx].copy()
        best_fitness = fitness[best_idx]

        self.history = []
        self.population_history = [positions.copy()]

        # Main loop
        for iteration in range(self.conf.iteration):
            # Update gravitational constant
            G = self.conf.G0 * np.exp(-self.conf.alpha * iteration / self.conf.iteration)

            # Calculate mass
            M = self._calculate_mass(fitness)

            # Kbest: only the K best agents exert force, K decreases linearly
            K = max(1, int(self.conf.pop_size * (1 - iteration / self.conf.iteration)))
            kbest_indices = np.argsort(fitness)[:K]

            # Calculate forces and accelerations
            dim = positions.shape[1]
            acceleration = np.zeros_like(positions)

            for i in range(self.conf.pop_size):
                force = np.zeros(dim)

                for j in kbest_indices:
                    if i != j:
                        # Calculate Euclidean distance
                        R = np.linalg.norm(positions[i] - positions[j]) + 1e-10

                        # Calculate gravitational force
                        force += (
                            np.random.rand() * M[j] * (positions[j] - positions[i]) / R
                        )

                # Calculate acceleration (F = ma => a = F/m)
                if M[i] > 0:
                    acceleration[i] = G * force / M[i]

            # Update velocities and positions
            velocities = np.random.rand(self.conf.pop_size, dim) * velocities + acceleration
            positions = positions + velocities

            # Clamp positions to problem bounds
            positions = self._clamp_positions(positions)

            # Map continuous positions to valid discrete solutions
            if self._is_permutation:
                positions = self._to_permutation(positions)
            elif self._is_discrete:
                positions = self._discretize(positions)

            # Evaluate new population
            fitness = np.array(
                [self.problem.eval(positions[i]) for i in range(self.conf.pop_size)]
            )

            # Update best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_solution = positions[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]

            # Track history
            self.history.append(best_solution)
            self.population_history.append(positions.copy())

        self.best_solution = best_solution
        self.best_fitness = float(best_fitness)
        return best_solution
