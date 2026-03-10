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
    delta_tau: np.ndarray   # pheromone changes per iteration
    _is_permutation: bool

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

        if self._is_permutation:
            # Edge-based pheromone: tau[from_node][to_node]
            self.tau = np.full((n, n), 0.1)
            self.eta = np.ones((n, n))
            self.delta_tau = np.zeros((n, n))
        else:
            # Position–value pheromone: tau[position][value]
            d = problem.domain_size
            self.tau = np.full((n, d), 0.1)
            self.eta = np.ones((n, d))
            self.delta_tau = np.zeros((n, d))

        self.best_solution = None
        self.best_fitness = float("inf")
        self.history = []

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
        """Build a permutation by visiting each node exactly once."""
        n = self.problem.n_dims
        current = np.random.randint(0, n)
        solution = [current]
        visited = {current}

        while len(solution) < n:
            next_node = self._select_next_permutation(current, visited)
            solution.append(next_node)
            visited.add(next_node)
            current = next_node

        return np.array(solution, dtype=float)

    def _construct_assignment(self) -> np.ndarray:
        """Build an assignment by choosing a value for each position."""
        n = self.problem.n_dims
        d = self.problem.domain_size
        solution = np.zeros(n, dtype=float)

        for pos in range(n):
            scores = (
                self.tau[pos] ** self.conf.alpha
                * self.eta[pos] ** self.conf.beta
            )
            total = scores.sum()
            if total == 0:
                solution[pos] = float(np.random.randint(d))
            else:
                probs = scores / total
                solution[pos] = float(np.random.choice(d, p=probs))

        return solution

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def _select_next_permutation(self, current: int, visited: set[int]) -> int:
        """Roulette-wheel selection for the next unvisited node.

        Parameters
        ----------
        current : int
            Current node.
        visited : set[int]
            Already visited nodes.

        Returns
        -------
        int
            Next node to visit.
        """
        unvisited = [i for i in range(self.problem.n_dims) if i not in visited]
        if not unvisited:
            return current

        scores = np.array([
            self.tau[current][j] ** self.conf.alpha
            * self.eta[current][j] ** self.conf.beta
            for j in unvisited
        ])

        total = scores.sum()
        if total == 0:
            return np.random.choice(unvisited)

        probs = scores / total
        return np.random.choice(unvisited, p=probs)

    # ------------------------------------------------------------------
    # Pheromone update
    # ------------------------------------------------------------------

    def pheromone_update(self, solutions: list[tuple[np.ndarray, float]]):
        """Update pheromone matrix.

        Parameters
        ----------
        solutions : list[tuple[np.ndarray, float]]
            (solution, fitness) pairs from all ants.
        """
        self.tau *= 1 - self.conf.rho
        self.delta_tau.fill(0)

        for solution, fitness in solutions:
            deposit = self.conf.q / fitness if fitness > 0 else self.conf.q

            if self._is_permutation:
                sol = solution.astype(int)
                for i in range(len(sol)):
                    a, b = sol[i], sol[(i + 1) % len(sol)]
                    self.delta_tau[a][b] += deposit
                    self.delta_tau[b][a] += deposit
            else:
                for pos in range(self.problem.n_dims):
                    val = int(solution[pos])
                    self.delta_tau[pos][val] += deposit

        self.tau += self.delta_tau
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
