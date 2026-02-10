"""Greedy Best-First Search algorithm for graph search problems."""

import heapq
from problems.base_problem import GraphSearchProblem
from algorithm.base_model import SearchGraphAlgorithm


class GreedyBestFirstSearch(SearchGraphAlgorithm[list, None, dict]):
    """
    Greedy Best-First Search algorithm.

    Greedy Best-First expands the node that appears to be closest to the goal
    according to a heuristic function h(n). It uses a priority queue ordered by
    the heuristic value.

    Note: This algorithm is not guaranteed to find the optimal path, but it's
    often faster than uninformed search algorithms.

    Attributes
    ----------
    name : str
        Algorithm name.
    """

    name = "Greedy Best-First Search"

    def __init__(self, configuration: dict, problem: GraphSearchProblem):
        """
        Initialize Greedy Best-First Search algorithm.

        Parameters
        ----------
        configuration : dict
            Configuration dictionary (can be empty for Greedy Best-First).
        problem : GraphSearchProblem
            The graph search problem instance to solve.
        """
        super().__init__(configuration, problem)

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

    def run(self) -> list | None:
        """
        Execute Greedy Best-First Search algorithm.

        Returns
        -------
        list or None
            Path from initial state to goal state, or None if no path found.
        """
        problem = self.problem
        self.history = []
        self.best_solution = None
        self.best_fitness = None

        # Priority queue with (heuristic_value, state)
        frontier = []
        h_value = problem.heuristic(problem.initial_state)
        heapq.heappush(frontier, (h_value, problem.initial_state))

        explored = set()
        parent = {}

        while frontier:
            _, current_state = heapq.heappop(frontier)
            self.history.append(current_state)

            if current_state in explored:
                continue

            explored.add(current_state)

            if problem.is_goal(current_state):
                self.best_solution = self._get_path(parent, current_state)
                return self.best_solution

            for action in problem.actions(current_state):
                next_state = problem.result(current_state, action)

                if next_state not in explored:
                    if next_state not in parent:
                        parent[next_state] = current_state
                        h_value = problem.heuristic(next_state)
                        heapq.heappush(frontier, (h_value, next_state))

        return None
