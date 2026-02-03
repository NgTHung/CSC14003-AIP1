"""Abstract base class definitions for optimization problems."""

from abc import ABC, abstractmethod
import numpy as np


class Problem(ABC):
    """
    Abstract base class for optimization problems.

    This class defines the interface that all specific problem implementations
    must handle. It supports both minimization and maximization problems.

    Attributes
    ----------
    name : str
        The name of the problem (e.g., 'Sphere', 'TSP').
    minimize : bool
        If True, the objective is to minimize the fitness function.
        If False, the objective is to maximize it.
    """
    name: str = "Base Problem"
    minimize: bool = True

    def __init__(self, name: str, is_minimize_problem: bool):
        """
        Initialize the Problem instance.

        Parameters
        ----------
        name : str
            The name of the problem.
        is_minimize_problem : bool
            Flag indicating the optimization goal. True for minimization,
            False for maximization.
        """
        self.name = name
        self.minimize = is_minimize_problem

    @abstractmethod
    def sample(self, pop_size: int = 1) -> np.ndarray:
        """
        Generate random valid solutions for this problem.

        This method is crucial for the initialization phase of optimization
        algorithms, providing a starting population.

        Parameters
        ----------
        pop_size : int, optional
            The number of solutions to generate. Default is 1.

        Returns
        -------
        np.ndarray
            An array of randomly generated solutions.
            The shape depends on the implementation,
            typically (pop_size, n_dims).
        """
        raise NotImplementedError

    @abstractmethod
    def eval(self, values: np.ndarray) -> np.float128 | np.ndarray:
        """
        Calculate the fitness or cost of solution(s).

        Parameters
        ----------
        values : np.ndarray
            Input solution or population of solutions to evaluate.
            Shape can be (n_dims,) for a single solution or
            (pop_size, n_dims) for a batch of solutions.

        Returns
        -------
        np.float128 or np.ndarray
            The calculated fitness or cost value(s). Returns a scalar if a
            single solution is provided, or an array if a batch is provided.
        """
        raise NotImplementedError

    @abstractmethod
    def is_valid(self, x: np.ndarray) -> bool:
        """
        Check if a solution satisfies the problem's constraints.

        Parameters
        ----------
        x : np.ndarray
            The solution vector to check.

        Returns
        -------
        bool
            True if the solution satisfies all constraints, False otherwise.
        """
        raise NotImplementedError
