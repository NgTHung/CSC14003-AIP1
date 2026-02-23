"""Gravitational Search Algorithm (GSA) - physics-inspired optimization."""

import numpy as np
from problems.base_problem import Problem
from algorithm.base_model import Model


class GravitationalSearchAlgorithm(Model[Problem, np.ndarray, np.ndarray, dict]):
    """
    Gravitational Search Algorithm (GSA).

    GSA is based on the law of gravity and mass interactions. Agents (masses)
    attract each other with gravitational force proportional to their masses
    (fitness) and inversely proportional to distance.

    Attributes
    ----------
    name : str
        Algorithm name.
    pop_size : int
        Population size (number of agents).
    max_iterations : int
        Maximum number of iterations.
    G0 : float
        Initial gravitational constant.
    alpha : float
        Gravitational constant reduction coefficient.
    """

    name = "Gravitational Search Algorithm"

    def __init__(self, configuration: dict, problem: Problem):
        """
        Initialize GSA algorithm.

        Parameters
        ----------
        configuration : dict
            Configuration with keys:
            - 'pop_size': Population size (default: 30)
            - 'max_iterations': Max iterations (default: 100)
            - 'G0': Initial gravitational constant (default: 100.0)
            - 'alpha': G reduction coefficient (default: 20.0)
        problem : Problem
            The optimization problem instance.
        """
        super().__init__(configuration, problem)
        self.pop_size = configuration.get('pop_size', 30)
        self.max_iterations = configuration.get('max_iterations', 100)
        self.G0 = configuration.get('G0', 100.0)
        self.alpha = configuration.get('alpha', 20.0)

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

    def run(self) -> np.ndarray:
        """
        Execute GSA algorithm.

        Returns
        -------
        np.ndarray
            Best solution found.
        """
        # Initialize population
        positions = self.problem.sample(self.pop_size)
        velocities = np.zeros_like(positions)

        # Evaluate initial population
        fitness = np.array([self.problem.eval(positions[i]) for i in range(self.pop_size)])

        # Track best
        best_idx = np.argmin(fitness)
        best_solution = positions[best_idx].copy()
        best_fitness = fitness[best_idx]

        self.history = [positions.copy()]
        self.best_fitness = [best_fitness]

        # Main loop
        for iteration in range(self.max_iterations):
            # Update gravitational constant
            G = self.G0 * np.exp(-self.alpha * iteration / self.max_iterations)

            # Calculate mass
            M = self._calculate_mass(fitness)

            # Calculate forces and accelerations
            dim = positions.shape[1]
            acceleration = np.zeros_like(positions)

            for i in range(self.pop_size):
                force = np.zeros(dim)

                for j in range(self.pop_size):
                    if i != j:
                        # Calculate Euclidean distance
                        R = np.linalg.norm(positions[i] - positions[j]) + 1e-10

                        # Calculate gravitational force
                        force += np.random.rand() * M[j] * (positions[j] - positions[i]) / R

                # Calculate acceleration (F = ma => a = F/m)
                if M[i] > 0:
                    acceleration[i] = G * force / M[i]

            # Update velocities and positions
            velocities = np.random.rand(self.pop_size, dim) * velocities + acceleration
            positions = positions + velocities

            # Evaluate new population
            fitness = np.array([self.problem.eval(positions[i]) for i in range(self.pop_size)])

            # Update best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_solution = positions[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]

            # Track history
            self.history.append(positions.copy())
            self.best_fitness.append(best_fitness)

        self.best_solution = best_solution
        return best_solution
