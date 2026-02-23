"""Harmony Search (HS) algorithm - music-inspired optimization."""

import numpy as np
from problems.base_problem import Problem
from algorithm.base_model import Model


class HarmonySearch(Model[Problem, np.ndarray, np.ndarray, dict]):
    """
    Harmony Search algorithm.

    HS is inspired by the musical improvisation process. It maintains a
    "harmony memory" of good solutions and generates new harmonies by:
    1. Selecting values from memory (memory consideration)
    2. Slightly adjusting selected values (pitch adjustment)
    3. Generating random values (randomization)

    Attributes
    ----------
    name : str
        Algorithm name.
    hms : int
        Harmony Memory Size.
    hmcr : float
        Harmony Memory Consideration Rate (0-1).
    par : float
        Pitch Adjustment Rate (0-1).
    bw : float
        Bandwidth for pitch adjustment.
    max_iterations : int
        Maximum number of iterations.
    """

    name = "Harmony Search"

    def __init__(self, configuration: dict, problem: Problem):
        """
        Initialize Harmony Search algorithm.

        Parameters
        ----------
        configuration : dict
            Configuration with keys:
            - 'hms': Harmony Memory Size (default: 20)
            - 'hmcr': Harmony Memory Consideration Rate (default: 0.9)
            - 'par': Pitch Adjustment Rate (default: 0.3)
            - 'bw': Bandwidth for adjustment (default: 0.1)
            - 'max_iterations': Max iterations (default: 1000)
        problem : Problem
            The optimization problem instance.
        """
        super().__init__(configuration, problem)
        self.hms = configuration.get('hms', 20)
        self.hmcr = configuration.get('hmcr', 0.9)
        self.par = configuration.get('par', 0.3)
        self.bw = configuration.get('bw', 0.1)
        self.max_iterations = configuration.get('max_iterations', 1000)

    def _improvise_harmony(self, harmony_memory: np.ndarray) -> np.ndarray:
        """
        Improvise a new harmony from the harmony memory.

        Parameters
        ----------
        harmony_memory : np.ndarray
            Current harmony memory (hms x dimensions).

        Returns
        -------
        np.ndarray
            New harmony vector.
        """
        dim = harmony_memory.shape[1]
        new_harmony = np.zeros(dim)

        for i in range(dim):
            if np.random.rand() < self.hmcr:
                # Memory consideration: pick value from harmony memory
                idx = np.random.randint(0, self.hms)
                new_harmony[i] = harmony_memory[idx, i]

                # Pitch adjustment
                if np.random.rand() < self.par:
                    new_harmony[i] += self.bw * (np.random.rand() - 0.5) * 2
            else:
                # Randomization: generate random value
                sample = self.problem.sample(1).flatten()
                new_harmony[i] = sample[i]

        return new_harmony

    def run(self) -> np.ndarray:
        """
        Execute Harmony Search algorithm.

        Returns
        -------
        np.ndarray
            Best solution found.
        """
        # Initialize harmony memory
        harmony_memory = self.problem.sample(self.hms)

        # Evaluate harmony memory
        fitness = np.array([self.problem.eval(harmony_memory[i]) for i in range(self.hms)])

        # Track best
        best_idx = np.argmin(fitness)
        best_solution = harmony_memory[best_idx].copy()
        best_fitness = fitness[best_idx]

        self.history = [harmony_memory.copy()]
        self.best_fitness = [best_fitness]

        # Main loop
        for iteration in range(self.max_iterations):
            # Improvise new harmony
            new_harmony = self._improvise_harmony(harmony_memory)
            new_fitness = self.problem.eval(new_harmony)

            # Update harmony memory if new harmony is better than worst
            worst_idx = np.argmax(fitness)
            if new_fitness < fitness[worst_idx]:
                harmony_memory[worst_idx] = new_harmony
                fitness[worst_idx] = new_fitness

                # Update best if needed
                if new_fitness < best_fitness:
                    best_solution = new_harmony.copy()
                    best_fitness = new_fitness

            # Track history
            self.history.append(harmony_memory.copy())
            self.best_fitness.append(best_fitness)

        self.best_solution = best_solution
        return best_solution
