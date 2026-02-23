"""Base model abstractions for optimization algorithms.

Defines a generic `Model` interface that stores configuration, problem
instances, and solution history for algorithm implementations.
"""

from problems import Problem
from problems.base_problem import GraphSearchProblem


class Model[Prob: Problem, T, Tr, Opt]:
    """Generic optimization model interface.

    Type parameters
    -------
    Prob: A `Problem` subclass describing the optimization task.\\
    T: The solution representation type.\\
    TR: The fitness/score type returned by the problem.\\
    Opt: The configuration/options type for the algorithm.

    Attributes
    -------
    history: Collected solutions or states during execution.\\
    best_solution: The best solution found so far.\\
    conf: Algorithm configuration/options.\\
    bestFitness: Best fitness value observed.\\
    problem: Problem instance to solve.\\
    name: Human-readable name of the algorithm.
    """

    history: list[T]
    best_solution: T
    conf: Opt
    best_fitness: Tr
    problem: Prob
    name: str = "Generic Model"

    def __init__(self, configuration: Opt, problem: Prob):
        """
        Initialize the base model with configuration and problem instance.

        Parameters
        -------
        configuration (Opt): The optimization configuration object containing
            algorithm parameters and settings.
        problem (Prob): The problem instance to be solved by the algorithm.
        """
        self.conf = configuration
        self.problem = problem

    def run(self) -> T:
        """
        Execute the algorithm.

        This method must be implemented by subclasses to define the specific
        algorithm's execution logic.

        Raises
        -------
        NotImplementedError: This method must be overridden by subclasses.
        """
        raise NotImplementedError

    def set_problem(self, problem: Prob):
        """Update the problem instance.

        Parameters
        ----------
        problem : Prob
            New problem instance to solve.
        """
        self.problem = problem

    def set_config(self, config: Opt):
        """Update the algorithm configuration.

        Parameters
        ----------
        config : Opt
            New configuration object with algorithm parameters.
        """
        self.conf = config


class SearchGraphAlgorithm[T, Tr, Opt]:
    """Generic graph search algorithm interface.

    Type parameters
    -------
    T: The solution representation type (typically list for path).
    Tr: The fitness/score type (typically None or dict for search stats).
    Opt: The configuration/options type for the algorithm.

    Attributes
    -------
    history: Collected states visited during execution.
    best_solution: The best solution (path) found.
    conf: Algorithm configuration/options.
    best_fitness: Best fitness value (typically None for search).
    problem: GraphSearchProblem instance to solve.
    name: Human-readable name of the algorithm.
    """

    history: list[T]
    best_solution: T | None
    conf: Opt
    best_fitness: Tr
    problem: GraphSearchProblem
    name: str = "Generic Search Algorithm"

    def __init__(self, configuration: Opt, problem: GraphSearchProblem):
        """
        Initialize the search algorithm with configuration and problem.

        Parameters
        -------
        configuration (Opt): The algorithm configuration object.
        problem (GraphSearchProblem): The graph search problem instance.
        """
        self.conf = configuration
        self.problem = problem

    def run(self) -> T | None:
        """
        Execute the search algorithm.

        Returns
        -------
        T | None
            The path from initial state to goal state, or None if no path found.
        """
        raise NotImplementedError

    def set_problem(self, problem: GraphSearchProblem):
        """Update the problem instance.

        Parameters
        ----------
        problem : GraphSearchProblem
            New problem instance to solve.
        """
        self.problem = problem

    def set_config(self, config: Opt):
        """Update the algorithm configuration.

        Parameters
        ----------
        config : Opt
            New configuration object with algorithm parameters.
        """
        self.conf = config
