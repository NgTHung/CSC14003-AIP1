"""Ant System (AS) algorithm for combinatorial optimization.

Implements the classic ACO algorithm by Dorigo et al., using pheromone trails
and probabilistic decision-making.

Supports two solution types through ``DiscreteProblem.solution_type``:

* **Permutation** (e.g. TSP): ants build solutions by visiting each node
  exactly once; pheromone is on edges ``tau[from_node][to_node]``.
* **Assignment** (e.g. Knapsack, Graph Coloring): ants assign a value from
  a finite domain to each position; pheromone is on position–value pairs
  ``tau[position][value]``.
"""

from dataclasses import dataclass
from typing import cast, override

import numpy as np
from AIP.problems import DiscreteProblem
from AIP.algorithm import Model


@dataclass
class AntSystemParameter:
    """Configuration parameters for Ant System.

    Attributes
    ----------
    rho : float
        Evaporation rate (0 < rho < 1). Typical: 0.1-0.5.
    m : int
        Number of ants in the colony.
    q : float
        Pheromone deposit factor.
    alpha : float
        Pheromone importance (α). Typical: 1.0.
    beta : float
        Heuristic importance (β). Typical: 2.0-5.0.
    cycle : int
        Number of iterations.
    """
    rho: float      # evaporation rate (0 < rho < 1)
    m: int          # number of ants
    q: float        # pheromone deposit factor
    alpha: float    # pheromone importance
    beta: float     # heuristic importance
    cycle: int      # running cycle


class AntSystem(Model[DiscreteProblem, np.ndarray | None, float, AntSystemParameter]):
    """Ant System algorithm for discrete optimization.

    Constructs solutions probabilistically using pheromone trails and
    heuristics.  Automatically adapts to permutation or assignment problems
    based on ``problem.solution_type``.
    """

    tau: np.ndarray         # pheromone matrix
    eta: np.ndarray         # heuristic information
    _is_permutation: bool
    _combined: np.ndarray   # pre-computed tau^alpha * eta^beta

    def __init__(self, configuration: AntSystemParameter, problem: DiscreteProblem):
        """Initialize Ant System with configuration and problem.

        Parameters
        ----------
        configuration : AntSystemParameter
            Algorithm hyperparameters.
        problem : DiscreteProblem
            Discrete optimization problem to solve.
        """
        super().__init__(configuration, problem)
        n = problem.n_dims
        self._is_permutation = problem.solution_type == "permutation"

        # Load problem-specific heuristic (e.g. 1/distance for TSP,
        # value/weight for Knapsack).  Falls back to all-ones when the
        # problem does not supply one.
        heuristic = problem.aco_heuristic()

        if self._is_permutation:
            self.tau = np.full((n, n), 0.1)
            self.eta = heuristic if heuristic is not None else np.ones((n, n))
        else:
            d = problem.domain_size
            self.tau = np.full((n, d), 0.1)
            self.eta = heuristic if heuristic is not None else np.ones((n, d))

        # Pre-compute eta^beta (constant across iterations)
        self._eta_beta = self.eta ** self.conf.beta
        self._update_combined()

        self.best_solution = None
        self.best_fitness = float("inf")
        self.history = []

    def _update_combined(self):
        """Recompute the combined score matrix tau^alpha * eta^beta."""
        self._combined = self.tau ** self.conf.alpha * self._eta_beta

    # ------------------------------------------------------------------
    # Solution construction
    # ------------------------------------------------------------------

    def construct_solution(self) -> list[tuple[np.ndarray, float]]:
        """Construct solutions for all ants.

        Returns
        -------
        list[tuple[np.ndarray, float]]
            (solution, fitness) pairs for each ant.
        """
        # Recompute combined scores once per iteration
        self._update_combined()

        solutions: list[tuple[np.ndarray, float]] = []

        for _ in range(self.conf.m):
            solution = self._construct_ant_solution()
            fitness = cast(float, self.problem.eval(solution))
            solutions.append((solution, fitness))

            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = solution.copy()

        return solutions

    def _construct_ant_solution(self) -> np.ndarray:
        """Construct a single ant's solution.

        Dispatches to permutation or assignment construction based on
        ``problem.solution_type``.
        """
        if self._is_permutation:
            return self._construct_permutation()
        return self._construct_assignment()

    def _construct_permutation(self) -> np.ndarray:
        """Build a permutation by visiting each node exactly once.

        Uses a boolean mask instead of a set for O(1) numpy indexing.
        """
        n = self.problem.n_dims
        visited = np.zeros(n, dtype=bool)
        solution = np.empty(n, dtype=np.intp)

        current = np.random.randint(0, n)
        solution[0] = current
        visited[current] = True

        for step in range(1, n):
            # Scores for all nodes, zeroing visited ones
            scores = self._combined[current].copy()
            scores[visited] = 0.0

            total = scores.sum()
            if total == 0.0:
                # Fallback: pick uniformly among unvisited
                candidates = np.flatnonzero(~visited)
                current = candidates[np.random.randint(len(candidates))]
            else:
                scores /= total
                current = np.random.choice(n, p=scores)

            solution[step] = current
            visited[current] = True

        return solution.astype(float)

    def _construct_assignment(self) -> np.ndarray:
        """Build an assignment by choosing a value for each position.

        Vectorised: computes cumulative probabilities for all positions
        at once and samples via uniform random draws.
        """
        # _combined shape: (n, d)
        scores = self._combined.copy()
        totals = scores.sum(axis=1, keepdims=True)

        # Positions with zero total get uniform probability
        zero_mask = (totals.ravel() == 0)
        if zero_mask.any():
            scores[zero_mask] = 1.0
            totals[zero_mask] = scores.shape[1]

        # Cumulative probabilities along the domain axis
        cumprobs = np.cumsum(scores / totals, axis=1)

        # Draw one uniform random per position and pick via searchsorted
        rands = np.random.rand(scores.shape[0])
        solution = np.array([
            np.searchsorted(cumprobs[pos], rands[pos])
            for pos in range(scores.shape[0])
        ], dtype=float)

        return solution

    # ------------------------------------------------------------------
    # Pheromone update
    # ------------------------------------------------------------------

    def pheromone_update(self, solutions: list[tuple[np.ndarray, float]]):
        """Update pheromone matrix using vectorised deposits.

        Parameters
        ----------
        solutions : list[tuple[np.ndarray, float]]
            (solution, fitness) pairs from all ants.
        """
        self.tau *= 1 - self.conf.rho

        for solution, fitness in solutions:
            deposit = self.conf.q / fitness if fitness > 0 else self.conf.q

            if self._is_permutation:
                sol = solution.astype(np.intp)
                frm = sol
                to = np.roll(sol, -1)
                self.tau[frm, to] += deposit
                self.tau[to, frm] += deposit
            else:
                positions = np.arange(self.problem.n_dims)
                vals = solution.astype(np.intp)
                self.tau[positions, vals] += deposit

        self.tau = np.clip(self.tau, 0.01, 100.0)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    @override
    def run(self) -> np.ndarray:
        for _ in range(self.conf.cycle):
            solutions = self.construct_solution()
            self.pheromone_update(solutions)

            if self.best_solution is not None:
                self.history.append(self.best_solution.copy())

        return cast(np.ndarray, self.best_solution)
