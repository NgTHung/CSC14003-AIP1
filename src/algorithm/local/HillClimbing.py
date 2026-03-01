"""Hill Climbing algorithm for local search problems."""

from typing import Any

from problems.base_problem import LocalSearchProblem
from algorithm.base_model import Model


class HillClimbing(Model[LocalSearchProblem, Any, float, dict]):
    """
    Simple Hill Climbing algorithm (first-choice variant).

    Hill Climbing is a local search algorithm that continually moves to a
    better neighboring state. This variant accepts the first neighbor that
    improves the current state.

    Attributes
    ----------
    name : str
        Algorithm name.
    problem : LocalSearchProblem
        The problem instance to solve.
    current_state : Any
        Current state during search.
    current_value : float
        Value of current state.
    history : list
        History of states visited during search.
    value_history : list
        History of values during search.
    """

    name = "Hill Climbing (First-Choice)"

    def __init__(
        self,
        configuration: dict,
        problem: LocalSearchProblem,
    ):
        """
        Initialize Hill Climbing algorithm.

        Parameters
        ----------
        configuration : dict
            Configuration with keys:
            - 'max_iterations': Maximum iterations (default: 1000).
        problem : LocalSearchProblem
            The local search problem to solve.
        """
        super().__init__(configuration, problem)
        self.max_iterations = configuration.get('max_iterations', 1000)
        self.current_state = None
        self.current_value = None
        self.history = []
        self.value_history = []

    def run(self, initial_state=None):
        """
        Execute Hill Climbing algorithm.

        Saves structured history for plotting:
        - history: list of dicts with 'iteration', 'state', 'value', 'improved'
        - best_fitness: best value found

        Parameters
        ----------
        initial_state : Any, optional
            Starting state. If None, generates a random state.

        Returns
        -------
        tuple
            (best_state, best_value) - the best state found and its value.
        """
        # Initialize
        if initial_state is None:
            self.current_state = self.problem.random_state()
        else:
            self.current_state = initial_state

        self.current_value = self.problem.value(self.current_state)
        self.history = [{
            'iteration': 0,
            'state': self.current_state,
            'value': self.current_value,
            'improved': True,
        }]

        # Main loop
        for iteration in range(self.max_iterations):
            # Get all neighbors
            neighbors = self.problem.neighbors(self.current_state)

            if not neighbors:
                break

            # Find first improving neighbor
            found_better = False
            for neighbor in neighbors:
                neighbor_value = self.problem.value(neighbor)

                # Check if neighbor is better
                if self.problem.is_better(neighbor_value, self.current_value):
                    self.current_state = neighbor
                    self.current_value = neighbor_value
                    self.history.append({
                        'iteration': iteration + 1,
                        'state': self.current_state,
                        'value': self.current_value,
                        'improved': True,
                    })
                    found_better = True
                    break

            # Local optimum reached
            if not found_better:
                break

        self.best_solution = self.current_state
        self.best_fitness = self.current_value
        return self.current_state, self.current_value

    def get_statistics(self) -> dict:
        """
        Get statistics about the search.

        Returns
        -------
        dict
            Dictionary with search statistics.
        """
        initial_value = self.history[0]['value'] if self.history else None
        return {
            'iterations': len(self.history),
            'final_value': self.current_value,
            'initial_value': initial_value,
            'improvement': abs(initial_value - self.current_value) if initial_value is not None else 0
        }
