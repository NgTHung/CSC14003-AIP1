"""Hill Climbing algorithm for local search problems."""

from typing import Any, cast, override
from dataclasses import dataclass

import numpy as np
from AIP.problems.base_problem import Problem, DiscreteProblem
from AIP.problems.continuous import ContinuousProblem
from AIP.algorithm.base_algorithm import Algorithm


@dataclass
class HillClimbingParameter:
    iteration: int


class HillClimbing(Algorithm[Problem, Any, float, HillClimbingParameter]):
    """
    Simple Hill Climbing algorithm (first-choice variant).

    Hill Climbing is a local search algorithm that continually moves to a
    better neighboring state. This variant accepts the first neighbor that
    improves the current state.

    Works with any problem exposing ``random_state()``, ``neighbors(state)``,
    ``value(state)`` and ``is_better(v1, v2)`` — i.e.
    :class:`DiscreteProblem` and :class:`ContinuousProblem`.

    Attributes
    ----------
    name : str
        Algorithm name.
    problem : DiscreteProblem
        The problem instance (DiscreteProblem or ContinuousProblem).
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
    current_value: float
    current_state: np.ndarray
    def __init__(
        self,
        configuration: HillClimbingParameter,
        problem: Problem,
    ):
        """
        Initialize Hill Climbing algorithm.

        Parameters
        ----------
        configuration : dict
            Configuration with keys:
            - 'max_iterations': Maximum iterations (default: 1000).
        problem : DiscreteProblem
            A problem with random_state/neighbors/value/is_better methods.
        """
        self.max_iterations = configuration.iteration
        self.value_history = []
        super().__init__(configuration, problem)
        self.reset()


    @override
    def reset(self, initial_state=None):
        if initial_state is None:
            self.current_state = self.problem.sample(1).flatten()
        else:
            self.current_state = initial_state

        self.current_value = cast(float,self.problem.eval(self.current_state))
        self.history = [self.current_value]
        self.value_history = []
        self.best_fitness = float("inf")
        self.best_solution = []

    @override
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
        self.reset(initial_state)
        assert self.current_state is not None
        assert self.current_value is not None
        # Main loop
        for _ in range(self.max_iterations):
            # Get all neighbors

            neighbors = None
            if isinstance(self.problem, DiscreteProblem):
                neighbors = self.problem.neighbors(self.current_state)
            elif isinstance(self.problem, ContinuousProblem):
                neighbors = self.problem.neighbors(self.current_state)

            if not neighbors:
                break

            # Find first improving neighbor
            found_better = False
            for neighbor in neighbors:
                neighbor_value = cast(float,self.problem.eval(neighbor))

                # Check if neighbor is better
                if (
                    isinstance(self.problem, DiscreteProblem)
                    or isinstance(self.problem, ContinuousProblem)
                ) and self.problem.is_better(
                    float(neighbor_value), float(self.current_value)
                ):
                    self.current_state = neighbor
                    self.current_value = neighbor_value
                    found_better = True

            # Track current state once per iteration
            self.history.append(self.current_value)

            # Local optimum reached
            if not found_better:
                break

        self.best_solution = self.current_state
        self.best_fitness = float(self.current_value)
        return self.current_state

    def get_statistics(self) -> dict:
        """
        Get statistics about the search.

        Returns
        -------
        dict
            Dictionary with search statistics.
        """
        return {
            "history fitness": self.history,
            "best fitness": self.best_fitness,
            "best solution": self.best_solution,
        }
