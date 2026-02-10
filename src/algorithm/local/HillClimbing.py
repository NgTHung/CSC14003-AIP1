"""Hill Climbing algorithm for local search problems."""

from problems.base_problem import LocalSearchProblem


class HillClimbing:
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
    max_iterations : int
        Maximum number of iterations to prevent infinite loops.
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

    def __init__(self, problem: LocalSearchProblem, max_iterations: int = 1000):
        """
        Initialize Hill Climbing algorithm.

        Parameters
        ----------
        problem : LocalSearchProblem
            The local search problem to solve.
        max_iterations : int, optional
            Maximum iterations to prevent infinite loops (default: 1000).
        """
        self.problem = problem
        self.max_iterations = max_iterations
        self.current_state = None
        self.current_value = None
        self.history = []
        self.value_history = []

    def run(self, initial_state=None):
        """
        Execute Hill Climbing algorithm.

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
        self.history = [self.current_state]
        self.value_history = [self.current_value]

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
                    self.history.append(self.current_state)
                    self.value_history.append(self.current_value)
                    found_better = True
                    break

            # Local optimum reached
            if not found_better:
                break

        return self.current_state, self.current_value

    def get_statistics(self) -> dict:
        """
        Get statistics about the search.

        Returns
        -------
        dict
            Dictionary with search statistics.
        """
        return {
            'iterations': len(self.history),
            'final_value': self.current_value,
            'initial_value': self.value_history[0] if self.value_history else None,
            'improvement': abs(self.value_history[0] - self.current_value) if len(self.value_history) > 0 else 0
        }
