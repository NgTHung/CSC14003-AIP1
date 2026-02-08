"""Ant System (AS) algorithm for combinatorial optimization.

Implements the classic ACO algorithm by Dorigo et al., using pheromone trails
and probabilistic decision-making for problems like TSP, VRP, and scheduling.
"""

from dataclasses import dataclass
from typing import cast, override

import numpy as np
from problems import Problem
from algorithm import Model


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


class AntSystem(Model[Problem, np.ndarray | None, float, AntSystemParameter]):
    """Ant System algorithm for discrete optimization.

    Constructs solutions probabilistically using pheromone trails and heuristics.
    """

    tau: np.ndarray         # pheromone matrix
    eta: np.ndarray         # heuristic information (1/distance)
    n_nodes: int            # number of nodes/cities
    delta_tau: np.ndarray   # pheromone changes

    def __init__(self, configuration: AntSystemParameter, problem: Problem):
        """Initialize Ant System with configuration and problem.

        Parameters
        ----------
        configuration : AntSystemParameter
            Algorithm hyperparameters.
        problem : DiscreteProblem
            Discrete optimization problem to solve.
        """
        super().__init__(configuration, problem)
        sample_solution = problem.sample(1)[0]
        self.n_nodes = len(sample_solution)

        self.tau = np.ones((self.n_nodes, self.n_nodes)) * 0.1

        self.eta = np.ones((self.n_nodes, self.n_nodes))

        self.best_solution = None
        self.best_fitness = float("inf")
        self.history = []

        # Delta tau for this iteration
        self.delta_tau = np.zeros((self.n_nodes, self.n_nodes))

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

            # Update best solution
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = solution.copy()

        return solutions

    def _construct_ant_solution(self) -> np.ndarray:
        """Construct a single ant's solution.

        Returns
        -------
        np.ndarray
            Complete solution (e.g., tour for TSP).
        """
        # Start from a random node
        current_node = np.random.randint(0, self.n_nodes)
        solution = [current_node]
        visited = set([current_node])

        # Build solution by visiting all nodes
        while len(solution) < self.n_nodes:
            next_node = self._select_next_node(current_node, visited)
            solution.append(next_node)
            visited.add(next_node)
            current_node = next_node

        return np.array(solution)

    def _select_next_node(self, current_node: int, visited: set[int]) -> int:
        """Select next node probabilistically.

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
        # Get unvisited nodes
        unvisited = [i for i in range(self.n_nodes) if i not in visited]

        if not unvisited:
            return current_node

        # Calculate probabilities for each unvisited node
        probabilities: list[float] = []
        for node in unvisited:
            # Probability = (tau^alpha) * (eta^beta)
            tau_val: float = self.tau[current_node][node] ** self.conf.alpha
            eta_val: float = self.eta[current_node][node] ** self.conf.beta
            probabilities.append(tau_val * eta_val)

        # Normalize probabilities
        np_probabilities = np.array(probabilities)
        total = np.sum(np_probabilities)

        if total == 0:
            return np.random.choice(unvisited)

        np_probabilities = np_probabilities / total

        next_node = np.random.choice(unvisited, p=np_probabilities)
        return next_node

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

            for i, node in enumerate(solution):
                to_node = solution[(i + 1) % len(solution)]

                self.delta_tau[node][to_node] += deposit
                self.delta_tau[to_node][node] += deposit

        self.tau += self.delta_tau

        self.tau = np.clip(self.tau, 0.01, 100.0)

    @override
    def run(self) -> np.ndarray:
        current_count = 0

        while current_count < self.conf.cycle:
            current_count += 1

            # All ants construct solutions
            solutions = self.construct_solution()

            # Update pheromone trails
            self.pheromone_update(solutions)

            # Store best solution in history
            if self.best_solution is not None:
                self.history.append(self.best_solution.copy())

        return cast(np.ndarray, self.best_solution)
