"""Base model abstractions for optimization algorithms.

Defines a generic `Model` interface that stores configuration, problem
instances, and solution history for algorithm implementations.
"""

from abc import ABC, abstractmethod

from AIP.problems import Problem


class Algorithm[Prob: Problem, T, Tr, Opt](ABC):
    """Generic optimization model interface.

    Type parameters
    -------
    Prob: A `Problem` subclass describing the optimization task.\\
    T: The solution representation type.\\
    Tr: The fitness/score type returned by the problem.\\
    Opt: The configuration/options type for the algorithm.

    Attributes
    -------
    history: Collected step records during execution (list of dicts).\\
    best_solution: The best solution found so far.\\
    conf: Algorithm configuration/options.\\
    best_fitness: Best fitness value observed.\\
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
    @abstractmethod
    def run(self) -> T | None:
        """
        Execute the algorithm.

        This method must be implemented by subclasses to define the specific
        algorithm's execution logic.

        Returns
        -------
        T or None
            The solution found, or None if no solution was found.

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

    @abstractmethod
    def reset(self):
        """Reset internal algorithm state before a fresh run."""
        raise NotImplementedError