"""Ant Colony System (ACS) algorithm for combinatorial optimization.

Extends Ant System with pseudorandom-proportional selection, local pheromone
updates during construction, and global pheromone updates using only the
best-so-far solution.

Supports two solution types through ``DiscreteProblem.solution_type``:

* **Permutation** (e.g. TSP): edge-based pheromone ``tau[from][to]``.
* **Assignment** (e.g. Knapsack, Graph Coloring): position–value pheromone
  ``tau[position][value]``.

Reference: Dorigo, M. & Gambardella, L.M. (1997). Ant colony system: a
cooperative learning approach to the traveling salesman problem.
"""

from dataclasses import dataclass
from typing import cast, override

import numpy as np
from problems import DiscreteProblem
from algorithm import Model


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


class ACS(Model[DiscreteProblem, np.ndarray | None, float, ACSParameter]):
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
        n = problem.n_dims
        self._is_permutation = problem.solution_type == "permutation"

        # Estimate tau0 from a random solution cost
        sample = problem.sample(1)[0]
        nn_cost = abs(cast(float, problem.eval(sample)))
        self.tau0 = 1.0 / (n * nn_cost) if nn_cost > 0 else 0.1

        if self._is_permutation:
            self.tau = np.full((n, n), self.tau0)
            self.eta = np.ones((n, n))
        else:
            d = problem.domain_size
            self.tau = np.full((n, d), self.tau0)
            self.eta = np.ones((n, d))

        self.best_solution = None
        self.best_fitness = float("inf")
        self.history = []

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def _select_next_permutation(self, current: int, visited: set[int]) -> int:
        """Pseudorandom-proportional rule for permutation problems.

        With probability q0 pick the best unvisited node (exploitation),
        otherwise roulette-wheel (exploration).
        """
        unvisited = [i for i in range(self.problem.n_dims) if i not in visited]
        if not unvisited:
            return current

        scores = np.array([
            self.tau[current][j] ** self.conf.alpha
            * self.eta[current][j] ** self.conf.beta
            for j in unvisited
        ])

        if np.random.random() <= self.conf.q0:
            return unvisited[int(np.argmax(scores))]

        total = scores.sum()
        if total == 0:
            return np.random.choice(unvisited)
        probs = scores / total
        return np.random.choice(unvisited, p=probs)

    def _select_value_assignment(self, pos: int) -> int:
        """Pseudorandom-proportional rule for assignment problems.

        With probability q0 pick the best value (exploitation),
        otherwise roulette-wheel (exploration).
        """
        d = self.problem.domain_size
        scores = (
            self.tau[pos] ** self.conf.alpha
            * self.eta[pos] ** self.conf.beta
        )

        if np.random.random() <= self.conf.q0:
            return int(np.argmax(scores))

        total = scores.sum()
        if total == 0:
            return np.random.randint(d)
        probs = scores / total
        return int(np.random.choice(d, p=probs))

    # ------------------------------------------------------------------
    # Local pheromone update
    # ------------------------------------------------------------------

    def _local_pheromone_update_edge(self, i: int, j: int):
        """Reduce pheromone on edge (i, j) during construction."""
        self.tau[i][j] = (1 - self.conf.xi) * self.tau[i][j] + self.conf.xi * self.tau0
        self.tau[j][i] = self.tau[i][j]

    def _local_pheromone_update_value(self, pos: int, val: int):
        """Reduce pheromone on (position, value) during construction."""
        self.tau[pos][val] = (
            (1 - self.conf.xi) * self.tau[pos][val]
            + self.conf.xi * self.tau0
        )

    # ------------------------------------------------------------------
    # Solution construction
    # ------------------------------------------------------------------

    def _construct_ant_solution(self) -> np.ndarray:
        """Construct a single ant's solution with local pheromone updates."""
        if self._is_permutation:
            return self._construct_permutation()
        return self._construct_assignment()

    def _construct_permutation(self) -> np.ndarray:
        """Build a permutation visiting each node exactly once."""
        n = self.problem.n_dims
        current = np.random.randint(0, n)
        solution = [current]
        visited = {current}

        while len(solution) < n:
            next_node = self._select_next_permutation(current, visited)
            solution.append(next_node)
            self._local_pheromone_update_edge(current, next_node)
            visited.add(next_node)
            current = next_node

        return np.array(solution, dtype=float)

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

    # ------------------------------------------------------------------
    # Global pheromone update
    # ------------------------------------------------------------------

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
            sol = self.best_solution.astype(int)
            for i in range(len(sol)):
                a, b = sol[i], sol[(i + 1) % len(sol)]
                self.tau[a][b] += deposit
                self.tau[b][a] += deposit
        else:
            for pos in range(self.problem.n_dims):
                val = int(self.best_solution[pos])
                self.tau[pos][val] += deposit

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    @override
    def run(self) -> np.ndarray:
        """Execute the ACS algorithm.

        Returns
        -------
        np.ndarray
            Best solution found.
        """
        for _ in range(self.conf.cycle):
            self.construct_solution()
            self.global_pheromone_update()

            if self.best_solution is not None:
                self.history.append(self.best_solution.copy())

        return cast(np.ndarray, self.best_solution)
