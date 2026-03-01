"""Breadth-First Search (BFS) algorithm for graph search problems."""

from collections import deque
from problems.base_problem import GraphSearchProblem
from algorithm.base_model import Model


class BreadthFirstSearch(Model[GraphSearchProblem, list, float | None, dict]):
    """
    Breadth-First Search algorithm.

    BFS explores a graph level by level, visiting all neighbors
    of a node before moving to the next level. It uses a queue (FIFO)
    to manage the frontier, guaranteeing the shortest path in unweighted graphs.

    Attributes
    ----------
    name : str
        Algorithm name.
    """

    name = "Breadth First Search"

    def __init__(self, configuration: dict, problem: GraphSearchProblem):
        """
        Initialize BFS algorithm.

        Parameters
        ----------
        configuration : dict
            Configuration dictionary (can be empty for BFS).
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
        Execute BFS algorithm.

        Saves structured history for plotting:
        - history: list of dicts with 'state', 'frontier_size', 'explored_count'
        - best_fitness: total path cost (sum of edge costs) if path found

        Returns
        -------
        list or None
            Path from initial state to goal state, or None if no path found.
        """
        problem = self.problem
        self.history = []
        self.best_solution = None
        self.best_fitness = None

        frontier = deque([problem.initial_state])
        explored = set()
        parent = {}

        while frontier:
            current_state = frontier.popleft()

            if current_state in explored:
                continue

            explored.add(current_state)
            self.history.append({
                'state': current_state,
                'frontier_size': len(frontier),
                'explored_count': len(explored),
            })

            if problem.is_goal(current_state):
                self.best_solution = self._get_path(parent, current_state)
                # Calculate total path cost
                path = self.best_solution
                total_cost = sum(
                    problem.cost(path[i], path[i + 1], path[i + 1])
                    for i in range(len(path) - 1)
                )
                self.best_fitness = total_cost
                return self.best_solution

            for action in problem.actions(current_state):
                next_state = problem.result(current_state, action)
                if next_state not in explored and next_state not in [node for node in frontier]:
                    parent[next_state] = current_state
                    frontier.append(next_state)

        return None
