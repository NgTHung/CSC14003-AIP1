"""Cultural Algorithm (CA) implementation.

Cultural Algorithms are a class of evolutionary computation algorithms that
evolve both a population space and a belief space simultaneously. The belief
space stores extracted knowledge used to guide population evolution.

References
----------
Reynolds, R. G. (1994). An Introduction to Cultural Algorithms.
Proceedings of the Third Annual Conference on Evolutionary Programming.
"""

from dataclasses import dataclass
from typing import cast, override
import numpy as np

from AIP.algorithm.base_algorithm import Algorithm
from AIP.problems.continuous.continuous import ContinuousProblem


@dataclass
class CAConfig:
    """Configuration parameters for the Cultural Algorithm.

    Attributes
    ----------
    pop_size : int
        Number of individuals in the population space. Default 50.
    iterations : int
        Number of evolutionary iterations to run. Default 100.
    minimization : bool
        If True the algorithm minimises the objective; if False it maximises.
        Default True.
    accepted_ratio : float
        Fraction of the population (sorted by fitness) used to update the
        belief space each iteration. E.g. 0.2 means the top 20%.
        Default 0.2.
    exploit_ratio : float
        Fraction of the new candidates generated *inside* the normative bounds
        (exploitation). The complement is generated via Gaussian mutation
        (exploration). Default 0.8.
    explore_sigma : float
        Standard deviation multiplier for the Gaussian exploration step,
        expressed as a fraction of the normative interval width. Default 0.1.
    """

    pop_size: int = 50
    iterations: int = 100
    minimization: bool = True
    accepted_ratio: float = 0.2
    exploit_ratio: float = 0.8
    explore_sigma: float = 0.1


class CA(Algorithm[ContinuousProblem, np.ndarray, float, CAConfig]):
    """Cultural Algorithm (CA) optimizer.

    Implements the dual-inheritance framework of Cultural Algorithms:

    * **Population Space** - a standard population of real-valued agents that
      are evaluated against the objective function.
    * **Belief Space** - two knowledge sources updated each iteration:
        - *Situational Knowledge*: the global best solution found so far.
        - *Normative Knowledge*: a per-dimension interval ``[normative_L,
          normative_U]`` representing the "good region" derived from the
          accepted individuals.

    The Influence protocol steers new candidate positions towards the
    normative interval (exploitation) or explores near the current
    position (exploration), followed by greedy selection.

    Parameters
    ----------
    configuration : CAConfig
        Algorithm hyper-parameters.
    problem : ContinuousProblem
        The continuous optimisation problem to solve.

    Attributes
    ----------
    name : str
        Human-readable algorithm name.
    population : np.ndarray, shape (pop_size, n_dim)
        Current population matrix.
    fitness : np.ndarray, shape (pop_size,)
        Fitness values for the current population.
    best_solution : np.ndarray, shape (n_dim,)
        Best solution found across all iterations.
    best_fitness : float
        Fitness value of ``best_solution``.
    history : list[float]
        Best fitness value recorded at the end of each iteration.
    normative_L : np.ndarray, shape (n_dim,)
        Lower bound of the normative knowledge interval (per dimension).
    normative_U : np.ndarray, shape (n_dim,)
        Upper bound of the normative knowledge interval (per dimension).
    """

    name: str = "Cultural Algorithm (CA)"
    population_history: list
    stat: bool
    population: np.ndarray
    fitness: np.ndarray
    normative_l: np.ndarray
    normative_u: np.ndarray

    def __init__(
        self, configuration: CAConfig, problem: ContinuousProblem, stat: bool = False
    ):
        """Initialize Cultural Algorithm.

        Parameters
        ----------
        configuration : CAConfig
            Algorithm hyperparameters.
        problem : ContinuousProblem
            Continuous optimization problem to solve.
        stat : bool, optional
            If True, record full population snapshots each iteration
            into ``population_history`` for visualisation. Default False.
        """
        super().__init__(configuration, problem)
        self.stat = stat

    @override
    def reset(self):
        pop_size = self.conf.pop_size
        lb = self.problem._bounds[:, 0]
        ub = self.problem._bounds[:, 1]
        self.population = self.problem.sample(pop_size)

        self.fitness = cast(np.ndarray, self.problem.eval(self.population))

        self._init_global_best()

        self.normative_l = lb.copy()  # (n_dim,)
        self.normative_u = ub.copy()  # (n_dim,)

        # Record history
        self.history = []
        if self.stat:
            self.population_history: list = []

    @override
    def run(self) -> np.ndarray:
        """Execute the Cultural Algorithm and return the best solution found.

        Returns
        -------
        np.ndarray, shape (n_dim,)
            The best solution vector discovered during the run.
        """
        # Cache frequently used references
        lb = self.problem._bounds[:, 0]  # global lower bounds (n_dim,)
        ub = self.problem._bounds[:, 1]  # global upper bounds (n_dim,)
        n_dim = self.problem._n_dim
        pop_size = self.conf.pop_size
        self.reset()
        # Number of accepted individuals (at least 1)
        k_accept = max(1, int(pop_size * self.conf.accepted_ratio))

        n_exploit = int(pop_size * self.conf.exploit_ratio)
        n_explore = pop_size - n_exploit

        for _it in range(self.conf.iterations):
            self._update_belief_space(lb, ub, k_accept)

            r_exploit = np.random.rand(n_exploit, n_dim)
            norm_width = self.normative_u - self.normative_l  # (n_dim,)
            candidates_exploit = self.normative_l + r_exploit * norm_width

            sigma = self.conf.explore_sigma * norm_width  # (n_dim,)
            noise = np.random.randn(n_explore, n_dim) * sigma  # broadcast
            candidates_explore = self.population[-n_explore:] + noise

            candidates = np.vstack([candidates_exploit, candidates_explore])

            candidates = np.clip(candidates, lb, ub)

            new_fitness = self.problem.eval(candidates)  # (pop_size,)
            self._greedy_update(candidates, new_fitness)

            self._update_global_best()
            self.history.append(self.best_solution.copy())
            if self.stat:
                self.population_history.append(self.population.copy())

        return self.best_solution

    def _is_better(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return a boolean mask where *a* is strictly better than *b*.

        Parameters
        ----------
        a, b : np.ndarray, shape (pop_size,)
            Fitness arrays to compare element-wise.

        Returns
        -------
        np.ndarray of bool, shape (pop_size,)
        """
        if self.conf.minimization:
            return a < b
        return a > b

    def _best_index(self) -> int:
        """Return the index of the current best individual in ``self.fitness``."""
        if self.conf.minimization:
            return int(np.argmin(self.fitness))
        return int(np.argmax(self.fitness))

    def _init_global_best(self) -> None:
        """Initialise ``best_solution`` and ``best_fitness`` from the current population."""
        idx = self._best_index()
        self.best_fitness: float = float(self.fitness[idx])
        self.best_solution: np.ndarray = self.population[idx].copy()

    def _update_global_best(self) -> None:
        """Update ``best_solution`` / ``best_fitness`` if the current population contains
        a better individual.
        """
        idx = self._best_index()
        candidate_fit = float(self.fitness[idx])

        if self._is_better(np.array([candidate_fit]), np.array([self.best_fitness]))[0]:
            self.best_fitness = candidate_fit
            self.best_solution = self.population[idx].copy()

    def _greedy_update(
        self,
        candidates: np.ndarray,
        candidate_fitness: np.ndarray,
    ) -> None:
        """Apply greedy selection in-place over the entire population.

        For each agent *i*, replace ``population[i]`` with ``candidates[i]``
        only when ``candidate_fitness[i]`` is strictly better than
        ``self.fitness[i]``.

        Parameters
        ----------
        candidates : np.ndarray, shape (pop_size, n_dim)
            Proposed new positions.
        candidate_fitness : np.ndarray, shape (pop_size,)
            Fitness values of ``candidates``.
        """
        improved = self._is_better(candidate_fitness, self.fitness)  # (pop_size,)
        improved_2d = improved[:, np.newaxis]  # (pop_size, 1)

        self.population = np.where(improved_2d, candidates, self.population)
        self.fitness = np.where(improved, candidate_fitness, self.fitness)

    def _update_belief_space(
        self,
        global_lb: np.ndarray,
        global_ub: np.ndarray,
        k_accept: int,
    ) -> None:
        """Update both knowledge sources in the belief space.

        Acceptance protocol
        -------------------
        1. Sort the population by fitness and take the top *k_accept*
           individuals as the *accepted set*.
        2. **Situational knowledge**: update the global best if the best
           accepted individual improves on the stored ``best_fitness``.
        3. **Normative knowledge**: compute the per-dimension bounding box
           of the accepted set.  The interval only *shrinks* (by taking the
           element-wise max of the current lower bound with the new min, and
           the element-wise min with the new max) to gradually focus the
           search.  Any dimension whose interval collapses below ``1e-6`` is
           reset to the full global range to prevent stagnation.

        Parameters
        ----------
        global_lb : np.ndarray, shape (n_dim,)
            Global lower bounds of the problem.
        global_ub : np.ndarray, shape (n_dim,)
            Global upper bounds of the problem.
        k_accept : int
            Number of individuals in the accepted set.
        """
        if self.conf.minimization:
            sorted_indices = np.argsort(self.fitness)  # ascending
        else:
            sorted_indices = np.argsort(self.fitness)[::-1]  # descending

        accepted_idx = sorted_indices[:k_accept]  # (k_accept,)
        accepted_pop = self.population[accepted_idx]  # (k_accept, n_dim)

        accepted_min = np.min(accepted_pop, axis=0)  # (n_dim,)
        accepted_max = np.max(accepted_pop, axis=0)  # (n_dim,)

        self.normative_l = np.maximum(self.normative_l, accepted_min)
        self.normative_u = np.minimum(self.normative_u, accepted_max)

        degenerate = (self.normative_u - self.normative_l) < 1e-6  # (n_dim,) bool
        self.normative_l = np.where(degenerate, global_lb, self.normative_l)
        self.normative_u = np.where(degenerate, global_ub, self.normative_u)
