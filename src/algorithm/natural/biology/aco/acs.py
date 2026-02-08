"""Ant Colony System (ACS) algorithm for combinatorial optimization.

Extends Ant System with pseudorandom-proportional selection, local pheromone
updates during construction, and global pheromone updates using only the
best-so-far solution.

Reference: Dorigo, M. & Gambardella, L.M. (1997). Ant colony system: a
cooperative learning approach to the traveling salesman problem.
"""

from dataclasses import dataclass
from typing import cast, override

import numpy as np
from problems import Problem
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


class ACS(Model[Problem, np.ndarray | None, float, ACSParameter]):
    """Ant Colony System for discrete optimization.

    Key differences from Ant System:
    - Pseudorandom-proportional rule: with probability q0 the ant greedily
      picks the best next node, otherwise uses roulette-wheel selection.
    - Local pheromone update: pheromone on visited edges is reduced during
      construction to encourage exploration.
    - Global pheromone update: only the best-so-far ant deposits pheromone.
    """

    tau: np.ndarray
    eta: np.ndarray
    tau0: float
    n_nodes: int

    def __init__(self, configuration: ACSParameter, problem: Problem):
        """Initialize ACS.

        Parameters
        ----------
        configuration : ACSParameter
            Algorithm hyperparameters.
        problem : Problem
            Optimization problem to solve.
        """
        super().__init__(configuration, problem)
        sample = problem.sample(1)[0]
        self.n_nodes = len(sample)

        self.eta = np.ones((self.n_nodes, self.n_nodes))

        # Initial pheromone: tau0 = 1 / (n * L_nn)
        # where L_nn is the cost of a nearest-neighbor heuristic tour
        nn_cost = cast(float, problem.eval(sample))
        self.tau0 = 1.0 / (self.n_nodes * nn_cost) if nn_cost > 0 else 0.1
        self.tau = np.full((self.n_nodes, self.n_nodes), self.tau0)

        self.best_solution = None
        self.best_fitness = float("inf")
        self.history = []

    def _select_next_node(self, current_node: int, visited: set[int]) -> int:
        """Pseudorandom-proportional selection rule.

        With probability q0 choose the node maximizing tau * eta^beta
        (exploitation). Otherwise, use roulette-wheel selection (exploration).

        Parameters
        ----------
        current_node : int
            Current node position.
        visited : set[int]
            Already visited nodes.

        Returns
        -------
        int
            Next node to visit.
        """
        unvisited = [i for i in range(self.n_nodes) if i not in visited]
        if not unvisited:
            return current_node

        scores = np.array([
            self.tau[current_node][j] ** self.conf.alpha
            * self.eta[current_node][j] ** self.conf.beta
            for j in unvisited
        ])

        q = np.random.random()
        if q <= self.conf.q0:
            # Exploitation: greedy choice
            return unvisited[int(np.argmax(scores))]

        # Exploration: roulette-wheel
        total = np.sum(scores)
        if total == 0:
            return np.random.choice(unvisited)
        probs = scores / total
        return np.random.choice(unvisited, p=probs)

    def _local_pheromone_update(self, i: int, j: int):
        """Apply local pheromone update on edge (i, j).

        Reduces pheromone on visited edges to encourage diversity.

        Parameters
        ----------
        i : int
            From node.
        j : int
            To node.
        """
        self.tau[i][j] = (1 - self.conf.xi) * self.tau[i][j] + self.conf.xi * self.tau0
        self.tau[j][i] = self.tau[i][j]

    def _construct_ant_solution(self) -> np.ndarray:
        """Construct a single ant's solution with local pheromone updates.

        Returns
        -------
        np.ndarray
            Complete solution.
        """
        current = np.random.randint(0, self.n_nodes)
        solution = [current]
        visited = {current}

        while len(solution) < self.n_nodes:
            next_node = self._select_next_node(current, visited)
            solution.append(next_node)
            if len(solution) == self.n_nodes:
                self._local_pheromone_update(current, next_node)
            visited.add(next_node)
            current = next_node

        return np.array(solution)

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
        best tour found across all iterations.
        """
        if self.best_solution is None:
            return

        # Evaporation on all edges
        self.tau *= (1 - self.conf.rho)

        # Deposit only on best-so-far tour edges
        deposit = self.conf.rho / self.best_fitness if self.best_fitness > 0 else self.conf.rho

        for i, a in enumerate(self.best_solution):
            b = self.best_solution[(i + 1) % len(self.best_solution)]
            self.tau[a][b] += deposit
            self.tau[b][a] += deposit

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
