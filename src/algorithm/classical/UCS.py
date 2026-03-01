"""Uniform Cost Search (UCS) algorithm for graph search problems."""

import heapq
from problems.base_problem import GraphSearchProblem
from algorithm.base_model import Model


class UniformCostSearch(Model[GraphSearchProblem, list, float | None, dict]):
    """
    Uniform Cost Search algorithm.

    UCS explores paths in order of their total cost from the initial state,
    guaranteeing the optimal (lowest cost) path to the goal. It uses a
    priority queue ordered by path cost.

    Attributes
    ----------
    name : str
        Algorithm name.
    """

    name = "Uniform Cost Search"

    def __init__(self, configuration: dict, problem: GraphSearchProblem):
        """
        Initialize UCS algorithm.

        Parameters
        ----------
        configuration : dict
            Configuration dictionary (can be empty for UCS).
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
        Execute UCS algorithm.

        Saves structured history for plotting:
        - history: list of dicts with 'state', 'cost', 'frontier_size', 'explored_count'
        - best_fitness: total path cost if path found

        Returns
        -------
        list or None
            Path from initial state to goal state, or None if no path found.
        """
        problem = self.problem
        self.history = []
        self.best_solution = None
        self.best_fitness = None

        frontier = []
        heapq.heappush(frontier, (0, problem.initial_state))

        explored = set()
        parent = {}
        cost_so_far = {problem.initial_state: 0}

        while frontier:
            current_cost, current_state = heapq.heappop(frontier)

            if current_state in explored:
                continue

            explored.add(current_state)
            self.history.append({
                'state': current_state,
                'cost': current_cost,
                'frontier_size': len(frontier),
                'explored_count': len(explored),
            })

            if problem.is_goal(current_state):
                self.best_solution = self._get_path(parent, current_state)
                self.best_fitness = current_cost
                return self.best_solution

            for action in problem.actions(current_state):
                next_state = problem.result(current_state, action)
                new_cost = current_cost + problem.cost(current_state, action, next_state)

                if next_state not in explored and (
                    next_state not in cost_so_far or new_cost < cost_so_far[next_state]
                ):
                    cost_so_far[next_state] = new_cost
                    parent[next_state] = current_state
                    heapq.heappush(frontier, (new_cost, next_state))

        return None
