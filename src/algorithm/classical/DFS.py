"""Depth-First Search (DFS) algorithm for graph search problems."""

from problems.base_problem import GraphSearchProblem
from algorithm.base_model import SearchGraphAlgorithm


class DepthFirstSearch(SearchGraphAlgorithm[list, None, dict]):
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

    def __init__(self, configuration: dict, problem: GraphSearchProblem):
        """
        Initialize DFS algorithm.

        Parameters
        ----------
        configuration : dict
            Configuration dictionary (can be empty for DFS).
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
        Execute DFS algorithm.

        Returns
        -------
        list or None
            Path from initial state to goal state, or None if no path found.
        """
        problem = self.problem
        self.history = []
        self.best_solution = None
        self.best_fitness = None

        frontier = [problem.initial_state]
        explored = set()
        parent = {}

        while frontier:
            current_state = frontier.pop()
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
                    parent[next_state] = current_state
                    frontier.append(next_state)

        return None
