"""Harmony Search (HS) algorithm - music-inspired optimization."""

from dataclasses import dataclass
import numpy as np
from AIP.problems.base_problem import Problem, DiscreteProblem
from AIP.problems.continuous.continuous import ContinuousProblem
from AIP.algorithm.base_algorithm import Algorithm


@dataclass
class HarmonySearchParameter:
    """Configuration parameters for Harmony Search.

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

    hms: int = 20
    hmcr: float = 0.9
    par: float = 0.3
    bw: float = 0.1
    max_iterations: int = 1000


class HarmonySearch(
    Algorithm[Problem, np.ndarray, float | None, HarmonySearchParameter]
):
    """
    Harmony Search algorithm.

    HS is inspired by the musical improvisation process. It maintains a
    "harmony memory" of good solutions and generates new harmonies by:
    1. Selecting values from memory (memory consideration)
    2. Slightly adjusting selected values (pitch adjustment)
    3. Generating random values (randomization)

    Works with both continuous and discrete (binary) problems.
    For :class:`DiscreteProblem` instances the pitch adjustment flips bits
    with probability ``par`` instead of adding continuous bandwidth noise.

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
    harmony_memory_history: list[np.ndarray]
    _is_discrete: bool
    _is_permutation: bool

    def __init__(
        self,
        configuration: HarmonySearchParameter,
        problem: Problem,
    ):
        """
        Initialize Harmony Search algorithm.

        Parameters
        ----------
        configuration : HarmonySearchParameter or dict
            Dataclass with HS hyperparameters. A dict is also accepted and
            will be converted for backward compatibility.
        problem : Problem
            The optimization problem instance.
        """
        super().__init__(configuration, problem)

    def reset(self) -> None:
        self._is_discrete = isinstance(self.problem, DiscreteProblem)
        self._is_permutation = (
            self._is_discrete
            and getattr(self.problem, "solution_type", None) == "permutation"
        )
        self.history = []
        self.harmony_memory_history = []
        self.best_fitness = None
        self.best_solution = np.array([])

    def _improvise_harmony(self, harmony_memory: np.ndarray) -> np.ndarray:
        """
        Improvise a new harmony from the harmony memory.

        For permutation problems (e.g. TSP) the new harmony is built by
        picking a whole solution from memory and optionally applying a
        2-opt perturbation, since per-dimension mixing would break the
        permutation constraint.

        For other discrete problems pitch adjustment flips the bit.
        For continuous problems Gaussian bandwidth noise is added.

        Parameters
        ----------
        harmony_memory : np.ndarray
            Current harmony memory (hms x dimensions).

        Returns
        -------
        np.ndarray
            New harmony vector.
        """
        # Permutation problems need whole-solution operations
        if self._is_permutation:
            return self._improvise_permutation(harmony_memory)

        dim = harmony_memory.shape[1]
        new_harmony = np.zeros(dim)

        for i in range(dim):
            if np.random.rand() < self.conf.hmcr:
                # Memory consideration: pick value from harmony memory
                idx = np.random.randint(0, self.conf.hms)
                new_harmony[i] = harmony_memory[idx, i]

                # Pitch adjustment
                if np.random.rand() < self.conf.par:
                    if self._is_discrete:
                        # Delegate to problem's perturb for correct handling
                        # of non-binary discrete (e.g. graph coloring)
                        assert isinstance(self.problem, DiscreteProblem)
                        tmp = new_harmony.copy()
                        tmp = self.problem.perturb(tmp)
                        new_harmony[i] = tmp[i]
                    else:
                        new_harmony[i] += self.conf.bw * (np.random.rand() - 0.5) * 2
                        if isinstance(self.problem, ContinuousProblem):
                            lb = self.problem._bounds[i, 0]
                            ub = self.problem._bounds[i, 1]
                            new_harmony[i] = np.clip(new_harmony[i], lb, ub)
            else:
                # Randomization: generate random value
                sample = self.problem.sample(1).flatten()
                new_harmony[i] = sample[i]

        return new_harmony

    def _improvise_permutation(self, harmony_memory: np.ndarray) -> np.ndarray:
        """Improvise a new harmony for permutation problems.

        Instead of per-dimension mixing (which breaks permutations),
        pick a whole solution from memory and optionally perturb it.
        """
        assert isinstance(self.problem, DiscreteProblem)
        if np.random.rand() < self.conf.hmcr:
            # Memory consideration: pick a full solution from memory
            idx = np.random.randint(0, self.conf.hms)
            new_harmony = harmony_memory[idx].copy()
            # Pitch adjustment: apply 2-opt swap(s)
            if np.random.rand() < self.conf.par:
                new_harmony = self.problem.perturb(new_harmony)
        else:
            # Randomization: generate a fresh random permutation
            new_harmony = self.problem.sample(1).flatten()
        return new_harmony

    def run(self) -> np.ndarray:
        """
        Execute Harmony Search algorithm.

        Saves structured history for plotting:
        - history: list of dicts with 'iteration', 'best_fitness'
        - best_fitness: best fitness value found

        Returns
        -------
        np.ndarray
            Best solution found.
        """
        self.reset()

        # Initialize harmony memory
        harmony_memory = self.problem.sample(self.conf.hms)

        # Evaluate harmony memory
        fitness = np.array(
            [self.problem.eval(harmony_memory[i]) for i in range(self.conf.hms)]
        )

        # Track best
        best_idx = np.argmin(fitness)
        best_solution = harmony_memory[best_idx].copy()
        best_fitness = fitness[best_idx]

        self.harmony_memory_history = [harmony_memory.copy()]

        # Main loop
        for _ in range(self.conf.max_iterations):
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
            self.history.append(best_solution)
            self.harmony_memory_history.append(harmony_memory.copy())

        self.best_solution = best_solution
        self.best_fitness = float(best_fitness)
        return best_solution
