"""A* Search algorithm for graph search problems."""

import heapq
from typing import override
from AIP.problems.base_problem import DiscreteProblem
from AIP.algorithm.base_algorithm import Algorithm


class AStarSearch(Algorithm[DiscreteProblem, list, float | None, dict]):
    """
    A* (A-star) Search algorithm.

    A* is an informed search algorithm that finds the optimal path by using
    the evaluation function f(n) = g(n) + h(n), where:
    - g(n) is the cost from the start to node n
    - h(n) is the heuristic estimate from n to the goal
    - f(n) is the estimated total cost of the path through n

    If the heuristic is admissible (never overestimates), A* is guaranteed
    to find the optimal path.

    Attributes
    ----------
    name : str
        Algorithm name.
    """

    name = "A* Search"
    explored_count: int

    def __init__(self, configuration: dict, problem: DiscreteProblem):
        """
        Initialize A* Search algorithm.

        Parameters
        ----------
        configuration : dict
            Configuration dictionary (can be empty for A*).
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
        Execute A* Search algorithm.

        Saves structured history for plotting:
        - history: list of dicts with 'state', 'g', 'h', 'f', 'frontier_size', 'explored_count'
        - best_fitness: total path cost (g-score) if path found

        Returns
        -------
        list or None
            Path from initial state to goal state, or None if no path found.
        """
        self.reset()
        problem = self.problem

        # Priority queue with (f_value, state)
        # f = g + h
        frontier = []
        g_initial = 0
        h_initial = problem.heuristic(problem.initial_state)
        f_initial = g_initial + h_initial
        heapq.heappush(frontier, (f_initial, problem.initial_state))

        explored = set()
        parent = {}
        g_score = {problem.initial_state: 0}

        while frontier:
            _, current_state = heapq.heappop(frontier)

            if current_state in explored:
                continue

            explored.add(current_state)
            current_g = g_score[current_state]
            self.explored_count += 1

            if problem.is_goal(current_state):
                self.best_solution = self._get_path(parent, current_state)
                self.best_fitness = current_g
                return self.best_solution

            for action in problem.actions(current_state):
                next_state = problem.result(current_state, action)

                if next_state not in explored:
                    # Calculate tentative g score
                    tentative_g = current_g + problem.cost(current_state, action, next_state)

                    # Only update if we found a better path
                    if next_state not in g_score or tentative_g < g_score[next_state]:
                        parent[next_state] = current_state
                        g_score.update({next_state:tentative_g})
                        h_value = problem.heuristic(next_state)
                        f_value = tentative_g + h_value
                        heapq.heappush(frontier, (f_value, next_state))

        return None
