"""Simulated Annealing (SA) algorithm - physics-inspired optimization."""

import numpy as np
from dataclasses import dataclass
from AIP.problems.base_problem import Problem, DiscreteProblem
from AIP.problems.continuous.continuous import ContinuousProblem
from AIP.algorithm.base_model import Model


@dataclass
class SimulatedAnnealingParameter:
    """
    Configuration parameters for Simulated Annealing.

    initial_temperature : float
        Starting temperature for the annealing process.
    cooling_rate : float
        Rate at which temperature decreases (typically 0.85-0.99).
    min_temperature : float
        Minimum temperature (stopping criterion).
    max_iterations : int
        Maximum number of iterations.
    step_size : float
        Perturbation step size for continuous problems.
    n_flips : int
        Number of bits to flip per step for discrete problems.
    """
    initial_temperature: float = 100.0
    cooling_rate: float = 0.95
    min_temperature: float = 0.01
    max_iterations: int = 1000
    step_size: float = 0.1
    n_flips: int = 1


class SimulatedAnnealing(Model[Problem, np.ndarray, float | None, SimulatedAnnealingParameter]):
    """
    Simulated Annealing algorithm.

    SA is inspired by the annealing process in metallurgy. It probabilistically
    accepts worse solutions to escape local optima, with the probability
    decreasing over time (as temperature cools).

    Works with both continuous and discrete (binary) problems.
    For :class:`DiscreteProblem` instances the perturbation flips random bits
    instead of adding Gaussian noise.

    Attributes
    ----------
    name : str
        Algorithm name.
    initial_temperature : float
        Starting temperature for the annealing process.
    cooling_rate : float
        Rate at which temperature decreases (typically 0.85-0.99).
    min_temperature : float
        Minimum temperature (stopping criterion).
    max_iterations : int
        Maximum number of iterations.
    """

    name = "Simulated Annealing"

    def __init__(self, configuration: SimulatedAnnealingParameter, problem: Problem):
        """
        Initialize Simulated Annealing algorithm.

        Parameters
        ----------
        configuration : SimulatedAnnealingParameter
            Dataclass with SA hyperparameters.
        problem : Problem
            The optimization problem instance.
        """
        super().__init__(configuration, problem)
        self.initial_temperature = configuration.initial_temperature
        self.cooling_rate = configuration.cooling_rate
        self.min_temperature = configuration.min_temperature
        self.max_iterations = configuration.max_iterations
        self.step_size = configuration.step_size
        self.n_flips = configuration.n_flips
        self._is_discrete = isinstance(problem, DiscreteProblem)

    def _perturb(self, state: np.ndarray) -> np.ndarray:
        """
        Generate a neighbor by perturbing the current state.

        For discrete problems the perturbation is delegated to
        ``problem.perturb()`` (which handles both binary-vector and
        permutation representations correctly).  For continuous problems
        Gaussian noise is added.

        Parameters
        ----------
        state : np.ndarray
            Current state.

        Returns
        -------
        np.ndarray
            Perturbed state.
        """
        if self._is_discrete:
            return self.problem.perturb(state, n_flips=self.n_flips)
        perturbation = np.random.randn(len(state)) * self.step_size
        new_state = state + perturbation
        if isinstance(self.problem, ContinuousProblem):
            lb = self.problem._bounds[:, 0]
            ub = self.problem._bounds[:, 1]
            new_state = np.clip(new_state, lb, ub)
        return new_state

    def run(self) -> np.ndarray:
        """
        Execute Simulated Annealing algorithm.

        Saves structured history for plotting:
        - history: list of dicts with 'iteration', 'state', 'energy',
          'best_energy', 'temperature', 'accepted'
        - best_fitness: best energy value found

        Returns
        -------
        np.ndarray
            Best solution found.
        """
        # Initialize
        current_state = self.problem.sample(1).flatten()
        current_energy = self.problem.eval(current_state)

        best_state = current_state.copy()
        best_energy = current_energy

        self.history = []
        self.trajectory = [current_state.copy()]

        temperature = self.initial_temperature
        iteration = 0

        # Main loop
        while temperature > self.min_temperature and iteration < self.max_iterations:
            # Generate neighbor
            new_state = self._perturb(current_state)
            new_energy = self.problem.eval(new_state)

            # Calculate energy difference
            delta_energy = new_energy - current_energy

            # Accept or reject
            accepted = False
            if delta_energy < 0:
                # Better solution - always accept
                current_state = new_state
                current_energy = new_energy
                accepted = True
            else:
                # Worse solution - accept with probability
                acceptance_probability = np.exp(-delta_energy / temperature)
                if np.random.rand() < acceptance_probability:
                    current_state = new_state
                    current_energy = new_energy
                    accepted = True

            # Update best
            if current_energy < best_energy:
                best_state = current_state.copy()
                best_energy = current_energy

            # Cool down
            temperature *= self.cooling_rate
            iteration += 1

            # Track history
            self.history.append(best_state)
            self.trajectory.append(current_state.copy())

        self.best_solution = best_state
        self.best_fitness = float(best_energy)
        return best_state
