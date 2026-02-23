"""Simulated Annealing (SA) algorithm - physics-inspired optimization."""

import numpy as np
import random
import math
from problems.base_problem import Problem
from algorithm.base_model import Model


class SimulatedAnnealing(Model[Problem, np.ndarray, np.ndarray, dict]):
    """
    Simulated Annealing algorithm.

    SA is inspired by the annealing process in metallurgy. It probabilistically
    accepts worse solutions to escape local optima, with the probability
    decreasing over time (as temperature cools).

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

    def __init__(self, configuration: dict, problem: Problem):
        """
        Initialize Simulated Annealing algorithm.

        Parameters
        ----------
        configuration : dict
            Configuration with keys:
            - 'initial_temperature': Starting temperature (default: 100.0)
            - 'cooling_rate': Temperature reduction rate (default: 0.95)
            - 'min_temperature': Stopping temperature (default: 0.01)
            - 'max_iterations': Max iterations (default: 1000)
            - 'step_size': Perturbation step size (default: 0.1)
        problem : Problem
            The optimization problem instance.
        """
        super().__init__(configuration, problem)
        self.initial_temperature = configuration.get('initial_temperature', 100.0)
        self.cooling_rate = configuration.get('cooling_rate', 0.95)
        self.min_temperature = configuration.get('min_temperature', 0.01)
        self.max_iterations = configuration.get('max_iterations', 1000)
        self.step_size = configuration.get('step_size', 0.1)

    def _perturb(self, state: np.ndarray) -> np.ndarray:
        """
        Generate a neighbor by perturbing the current state.

        Parameters
        ----------
        state : np.ndarray
            Current state.

        Returns
        -------
        np.ndarray
            Perturbed state.
        """
        perturbation = np.random.randn(len(state)) * self.step_size
        return state + perturbation

    def run(self) -> np.ndarray:
        """
        Execute Simulated Annealing algorithm.

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

        self.history = [current_state.copy()]
        self.best_fitness = [best_energy]

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
            if delta_energy < 0:
                # Better solution - always accept
                current_state = new_state
                current_energy = new_energy
            else:
                # Worse solution - accept with probability
                acceptance_probability = math.exp(-delta_energy / temperature)
                if random.random() < acceptance_probability:
                    current_state = new_state
                    current_energy = new_energy

            # Update best
            if current_energy < best_energy:
                best_state = current_state.copy()
                best_energy = current_energy

            # Cool down
            temperature *= self.cooling_rate
            iteration += 1

            # Track history
            self.history.append(current_state.copy())
            self.best_fitness.append(best_energy)

        self.best_solution = best_state
        return best_state
