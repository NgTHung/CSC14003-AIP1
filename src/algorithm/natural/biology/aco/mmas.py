"""MAX-MIN Ant System (MMAS) for combinatorial optimization.

Extends Ant System with bounded pheromone trails [tau_min, tau_max] and
best-ant-only pheromone updates.  Prevents premature convergence by
limiting the pheromone range and optionally re-initializing trails.

Reference: Stützle, T. & Hoos, H.H. (2000). MAX–MIN Ant System. Future
Generation Computer Systems, 16(8), 889-914.
"""

from dataclasses import dataclass
from typing import cast, override

import numpy as np
from problems import Problem
from algorithm import Model


@dataclass
class MMASParameter:
    """Configuration parameters for MMAS.

    Attributes
    ----------
    rho : float
        Evaporation rate (0 < rho < 1). Typical: 0.02-0.2.
    m : int
        Number of ants in the colony.
    alpha : float
        Pheromone importance (α). Typical: 1.0.
    beta : float
        Heuristic importance (β). Typical: 2.0-5.0.
    cycle : int
        Number of iterations.
    p_best : float
        Probability that the best solution is constructed when
        the system has converged. Used to compute tau_min.
        Typical: 0.05.
    use_iteration_best : bool
        If True, alternate between iteration-best and global-best
        for pheromone deposits. If False, always use global-best.
    """

    rho: float
    m: int
    alpha: float
    beta: float
    cycle: int
    p_best: float = 0.05
    use_iteration_best: bool = True


class MMAS(Model[Problem, np.ndarray | None, float, MMASParameter]):
    """MAX-MIN Ant System for discrete optimization.

    Key differences from Ant System:
    - Pheromone trails are bounded within [tau_min, tau_max].
    - Only the best ant (iteration-best or global-best) deposits pheromone.
    - Trails are initialized to tau_max for maximum exploration.
    - Pheromone re-initialization when stagnation is detected.
    """

    tau: np.ndarray
    eta: np.ndarray
    n_nodes: int
    tau_max: float
    tau_min: float

    def __init__(self, configuration: MMASParameter, problem: Problem):
        """Initialize MMAS.

        Parameters
        ----------
        configuration : MMASParameter
            Algorithm hyperparameters.
        problem : Problem
            Optimization problem to solve.
        """
        super().__init__(configuration, problem)
        sample = problem.sample(1)[0]
        self.n_nodes = len(sample)

        self.eta = np.ones((self.n_nodes, self.n_nodes))

        # Initial bounds (will be updated once best solution is found)
        self.tau_max = 1.0
        self.tau_min = self.tau_max / (2.0 * self.n_nodes)

        # Initialize pheromone to tau_max for maximum exploration
        self.tau = np.full((self.n_nodes, self.n_nodes), self.tau_max)

        self.best_solution = None
        self.best_fitness = float("inf")
        self.history = []

        self._stagnation_counter = 0
        self._last_improvement = 0

    def _update_bounds(self):
        """Recompute tau_max and tau_min based on best-so-far cost.

        tau_max = 1 / (rho * L_best)
        tau_min = tau_max * (1 - p_best^(1/n)) / ((n/2 - 1) * p_best^(1/n))
        """
        if self.best_fitness == float("inf") or self.best_fitness <= 0:
            return

        self.tau_max = 1.0 / (self.conf.rho * self.best_fitness)

        n = self.n_nodes
        p_root = self.conf.p_best ** (1.0 / n)
        denominator = (n / 2.0 - 1.0) * p_root
        if denominator > 0:
            self.tau_min = self.tau_max * (1.0 - p_root) / denominator
        else:
            self.tau_min = self.tau_max / (2.0 * n)

        # Ensure tau_min < tau_max
        self.tau_min = min(self.tau_min, self.tau_max)

    def _select_next_node(self, current_node: int, visited: set[int]) -> int:
        """Select next node using roulette-wheel selection.

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

        total = np.sum(scores)
        if total == 0:
            return np.random.choice(unvisited)

        probs = scores / total
        return np.random.choice(unvisited, p=probs)

    def _construct_ant_solution(self) -> np.ndarray:
        """Construct a single ant's solution.

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

        return solutions

    def pheromone_update(
        self,
        solutions: list[tuple[np.ndarray, float]],
        iteration: int,
    ):
        """Update pheromone using best-ant-only deposit and clamping.

        Parameters
        ----------
        solutions : list[tuple[np.ndarray, float]]
            (solution, fitness) pairs from all ants.
        iteration : int
            Current iteration number (for alternating update strategy).
        """
        # Find iteration-best
        iter_best_sol, iter_best_fit = min(solutions, key=lambda x: x[1])
        improved = False

        # Update global best
        if iter_best_fit < self.best_fitness:
            self.best_fitness = iter_best_fit
            self.best_solution = iter_best_sol.copy()
            self._update_bounds()
            self._last_improvement = iteration
            improved = True

        # Choose which ant deposits pheromone
        if self.conf.use_iteration_best and iteration % 2 == 0:
            deposit_sol = iter_best_sol
            deposit_fit = iter_best_fit
        else:
            deposit_sol = cast(np.ndarray, self.best_solution)
            deposit_fit = self.best_fitness

        # Evaporation
        self.tau *= (1 - self.conf.rho)

        # Deposit
        deposit = 1.0 / deposit_fit if deposit_fit > 0 else 1.0
        for i, a in enumerate(deposit_sol):
            b = deposit_sol[(i + 1) % len(deposit_sol)]
            self.tau[a][b] += deposit
            self.tau[b][a] += deposit

        # Clamp to [tau_min, tau_max]
        self.tau = np.clip(self.tau, self.tau_min, self.tau_max)

        # Re-initialize pheromone on stagnation
        if not improved:
            self._stagnation_counter += 1
        else:
            self._stagnation_counter = 0

        if self._stagnation_counter > self.n_nodes:
            self.tau = np.full((self.n_nodes, self.n_nodes), self.tau_max)
            self._stagnation_counter = 0

    @override
    def run(self) -> np.ndarray:
        """Execute the MMAS algorithm.

        Returns
        -------
        np.ndarray
            Best solution found.
        """
        for iteration in range(self.conf.cycle):
            solutions = self.construct_solution()
            self.pheromone_update(solutions, iteration)

            if self.best_solution is not None:
                self.history.append(self.best_solution.copy())

        return cast(np.ndarray, self.best_solution)
