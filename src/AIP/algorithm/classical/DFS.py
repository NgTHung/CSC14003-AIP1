"""Depth-First Search (DFS) algorithm for graph search problems."""

from typing import override

from AIP.problems.base_problem import DiscreteProblem
from AIP.algorithm.base_algorithm import Algorithm


class DepthFirstSearch(Algorithm[DiscreteProblem, list, float | None, dict]):
    """
    Depth-First Search algorithm.

    DFS explores a graph by going as deep as possible along each branch
    before backtracking. It uses a stack (LIFO) to manage the frontier.

    Attributes
    ----------
    name : str
        Algorithm name. 
    """

    name = "Depth First Search"
    explored_count: int

    def __init__(self, configuration: dict, problem: DiscreteProblem):
        """
        Initialize DFS algorithm.

        Parameters
        ----------
        configuration : dict
            Configuration dictionary (can be empty for DFS).
        problem : DiscreteProblem
            The problem instance (must expose graph-search interface).
        """
        super().__init__(configuration, problem)
        self.reset()

    def _get_path(self, parent: dict, state) -> list:
        """
        Reconstruct path from initial state to goal state.

        Parameters
        ----------
        parent : dict   
            Dictionary mapping each state to its parent state.
        state : Any
            The goal state to trace back from.

        Returns
        -------
        list
            Path from initial state to goal state.
        """
        path = [state]
        while state in parent:
            state = parent[state]
            path.append(state)
        path.reverse()
        return path

    @override
    def reset(self):
        self.history = []
        self.explored_count = 0
        self.best_solution = []
        self.best_fitness = None

    @override
    def run(self) -> list | None:
        """
        Execute DFS algorithm.

        Saves structured history for plotting:
        - history: list of dicts with 'state', 'frontier_size', 'explored_count'
        - best_fitness: total path cost (sum of edge costs) if path found

        Returns
        -------
        list or None
            Path from initial state to goal state, or None if no path found.
        """
        self.reset()
        problem = self.problem

        frontier = [problem.initial_state]
        explored = set()
        parent = {}

        while frontier:
            current_state = frontier.pop()

            if current_state in explored:
                continue

            explored.add(current_state)
            self.explored_count += 1

            for action in problem.actions(current_state):
                next_state = problem.result(current_state, action)
                if next_state not in explored:
                    parent[next_state] = current_state
                    if problem.is_goal(next_state):
                        self.best_solution = self._get_path(parent, next_state)
                        # Calculate total path cost
                        path = self.best_solution
                        total_cost = sum(
                            problem.cost(path[i], path[i + 1], path[i + 1])
                            for i in range(len(path) - 1)
                        )
                        self.best_fitness = total_cost
                        return self.best_solution
                    frontier.append(next_state)

        return None
