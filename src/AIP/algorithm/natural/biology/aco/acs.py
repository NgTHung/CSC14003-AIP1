"""Ant Colony System (ACS) algorithm for combinatorial optimization.

Extends Ant System with pseudorandom-proportional selection, local pheromone
updates during construction, and global pheromone updates using only the
best-so-far solution.

Supports two solution types through ``DiscreteProblem.solution_type``:

* **Permutation** (e.g. TSP): edge-based pheromone ``tau[from][to]``.
* **Assignment** (e.g. Knapsack, Graph Coloring): position-value pheromone
  ``tau[position][value]``.

Reference: Dorigo, M. & Gambardella, L.M. (1997). Ant colony system: a
cooperative learning approach to the traveling salesman problem.
"""

from dataclasses import dataclass
from typing import cast, override

import numpy as np
from AIP.problems import DiscreteProblem
from AIP.algorithm import Algorithm


@dataclass
class ACSParameter:
    """Configuration parameters for ACS.

    Attributes
    ----------
    rho : float
        Global evaporation rate (0 < rho < 1). Typical: 0.1.
    xi : float
        Local evaporation rate (0 < xi < 1). Typical: 0.1.
    m : int
        Number of ants in the colony.
    q0 : float
        Exploitation vs exploration threshold (0 <= q0 <= 1). Typical: 0.9.
    alpha : float
        Pheromone importance (α). Typical: 1.0.
    beta : float
        Heuristic importance (β). Typical: 2.0-5.0.
    cycle : int
        Number of iterations.
    """

    rho: float
    xi: float
    m: int
    q0: float
    alpha: float
    beta: float
    cycle: int


class ACS(Algorithm[DiscreteProblem, np.ndarray | None, float, ACSParameter]):
    """Ant Colony System for discrete optimization.

    Key differences from Ant System:
    - Pseudorandom-proportional rule: with probability q0 the ant greedily
      picks the best option, otherwise uses roulette-wheel selection.
    - Local pheromone update: pheromone on chosen edges/values is reduced
      during construction to encourage exploration.
    - Global pheromone update: only the best-so-far ant deposits pheromone.

    Automatically adapts to permutation or assignment problems based on
    ``problem.solution_type``.
    """

    tau: np.ndarray
    eta: np.ndarray
    tau0: float
    _is_permutation: bool
    _eta_beta: np.ndarray  # pre-computed eta^beta (constant)

    def __init__(self, configuration: ACSParameter, problem: DiscreteProblem):
        """Initialize ACS.

        Parameters
        ----------
        configuration : ACSParameter
            Algorithm hyperparameters.
        problem : DiscreteProblem
            Discrete optimization problem to solve.
        """
        super().__init__(configuration, problem)

    def _select_next_permutation(self, current: int, visited: np.ndarray) -> int:
        """Pseudorandom-proportional rule for permutation problems.

        With probability q0 pick the best unvisited node (exploitation),
        otherwise roulette-wheel (exploration).

        Per Dorigo & Gambardella (1997), the exploitation (greedy) step
        uses tau * eta^beta without raising tau to alpha.

        Uses vectorised numpy operations on the full row with a visited
        mask instead of building Python lists per unvisited node.
        """
        n = self.problem.n_dims
        eb = self._eta_beta[current]  # eta^beta row (cached)

        if np.random.random() <= self.conf.q0:
            # Exploitation: argmax_j { tau_ij * eta_ij^beta }  (no alpha)
            greedy_scores = self.tau[current] * eb
            greedy_scores[visited] = -1.0
            return int(np.argmax(greedy_scores)) # type: ignore

        # Exploration: roulette-wheel with tau^alpha * eta^beta
        scores = self.tau[current] ** self.conf.alpha * eb
        scores[visited] = 0.0

        total = scores.sum()
        if total == 0.0:
            candidates = np.flatnonzero(~visited)
            return candidates[np.random.randint(len(candidates))]

        scores /= total
        return int(np.random.choice(n, p=scores))

    def _select_value_assignment(self, pos: int) -> int:
        """Pseudorandom-proportional rule for assignment problems.

        With probability q0 pick the best value (exploitation),
        otherwise roulette-wheel (exploration).

        Per Dorigo & Gambardella (1997), the exploitation (greedy) step
        uses tau * eta^beta without raising tau to alpha.
        """
        d = self.problem.domain_size
        eb = self._eta_beta[pos]

        if np.random.random() <= self.conf.q0:
            # Exploitation: argmax { tau * eta^beta }  (no alpha)
            return int(np.argmax(self.tau[pos] * eb))

        # Exploration: roulette-wheel with tau^alpha * eta^beta
        scores = self.tau[pos] ** self.conf.alpha * eb
        total = scores.sum()
        if total == 0:
            return np.random.randint(d)
        scores /= total
        return int(np.random.choice(d, p=scores))

    def _local_pheromone_update_edge(self, i: int, j: int):
        """Reduce pheromone on edge (i, j) during construction."""
        new_val = (1 - self.conf.xi) * self.tau[i, j] + self.conf.xi * self.tau0
        self.tau[i, j] = new_val
        self.tau[j, i] = new_val

    def _local_pheromone_update_value(self, pos: int, val: int):
        """Reduce pheromone on (position, value) during construction."""
        self.tau[pos, val] = (
            (1 - self.conf.xi) * self.tau[pos, val]
            + self.conf.xi * self.tau0
        )

    def _construct_ant_solution(self) -> np.ndarray:
        """Construct a single ant's solution with local pheromone updates."""
        if self._is_permutation:
            return self._construct_permutation()
        return self._construct_assignment()

    def _construct_permutation(self) -> np.ndarray:
        """Build a permutation visiting each node exactly once.

        Uses a boolean mask instead of a set for O(1) numpy indexing.
        """
        n = self.problem.n_dims
        visited = np.zeros(n, dtype=bool)
        solution = np.empty(n, dtype=np.intp)

        current = np.random.randint(0, n)
        solution[0] = current
        visited[current] = True

        for step in range(1, n):
            next_node = self._select_next_permutation(current, visited)
            solution[step] = next_node
            self._local_pheromone_update_edge(current, next_node)
            visited[next_node] = True
            current = next_node

        return solution.astype(float)

    def _construct_assignment(self) -> np.ndarray:
        """Build an assignment choosing a value for each position."""
        n = self.problem.n_dims
        solution = np.zeros(n, dtype=float)

        for pos in range(n):
            val = self._select_value_assignment(pos)
            solution[pos] = float(val)
            self._local_pheromone_update_value(pos, val)

        return solution

    def construct_solution(self) -> list[tuple[np.ndarray, float]]:
        """Construct solutions for all ants.

        Returns
        -------
        list[tuple[np.ndarray, float]]
            (solution, fitness) pairs for each ant.
        """
        solutions: list[tuple[np.ndarray, float]] = []

        for _ in range(self.conf.m):
            solution = self._construct_ant_solution()
            fitness = cast(float, self.problem.eval(solution))
            solutions.append((solution, fitness))

            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = solution.copy()

        return solutions

    def global_pheromone_update(self):
        """Global pheromone update using only the best-so-far solution.

        Only the globally best ant deposits pheromone, reinforcing the
        best solution found across all iterations.
        """
        if self.best_solution is None:
            return

        # Evaporation
        self.tau *= (1 - self.conf.rho)

        # Deposit
        deposit = (
            self.conf.rho / self.best_fitness
            if self.best_fitness > 0
            else self.conf.rho
        )

        if self._is_permutation:
            sol = self.best_solution.astype(np.intp)
            frm = sol
            to = np.roll(sol, -1)
            self.tau[frm, to] += deposit
            self.tau[to, frm] += deposit
        else:
            positions = np.arange(self.problem.n_dims)
            vals = self.best_solution.astype(np.intp)
            self.tau[positions, vals] += deposit

    @override
    def reset(self):
        n = self.problem.n_dims
        self._is_permutation = self.problem.solution_type == "permutation"

        # Estimate tau0 from a random solution cost
        sample = self.problem.sample(1)[0]
        nn_cost = abs(cast(float, self.problem.eval(sample)))
        self.tau0 = 1.0 / (n * nn_cost) if nn_cost > 0 else 0.1

        # Load problem-specific heuristic (e.g. 1/distance for TSP,
        # value/weight for Knapsack).  Falls back to all-ones when the
        # problem does not supply one.
        heuristic = self.problem.aco_heuristic()

        if self._is_permutation:
            self.tau = np.full((n, n), self.tau0)
            self.eta = heuristic if heuristic is not None else np.ones((n, n))
        else:
            d = self.problem.domain_size
            self.tau = np.full((n, d), self.tau0)
            self.eta = heuristic if heuristic is not None else np.ones((n, d))

        # Pre-compute eta^beta (constant across iterations)
        self._eta_beta = self.eta ** self.conf.beta

        self.best_solution = None
        self.best_fitness = float("inf")
        self.history = []

    @override
    def run(self) -> np.ndarray:
        """Execute the ACS algorithm.

        Returns
        -------
        np.ndarray
            Best solution found.
        """
        self.reset()
        for _ in range(self.conf.cycle):
            self.construct_solution()
            self.global_pheromone_update()

            if self.best_solution is not None:
                self.history.append(self.best_solution.copy())

        return cast(np.ndarray, self.best_solution)
