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
    _name: str = "Base Problem"

    def __init__(self, name: str):
        """
        Initialize the Problem instance.

        Parameters
        ----------
        name : str
            The name of the problem.
        """
        self._name = name

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
    def eval(self, values: np.ndarray) -> float | np.ndarray:
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
        float or np.ndarray
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


class DiscreteProblem(Problem):
    """
    Abstract base class for discrete optimization problems.

    Provides a unified interface so that a single problem class can be solved
    by **all** algorithm families:

    * **Population-based / physics-inspired** — via ``sample()``, ``eval()``,
      ``is_valid()`` inherited from :class:`Problem`.
    * **Local search** — via ``random_state()``, ``neighbors()``, ``value()``,
      ``is_better()``.
    * **Classical graph search** — via ``actions()``, ``result()``, ``cost()``,
      ``is_goal()``, ``heuristic()``.

    Sub-classes **must** implement the abstract methods from :class:`Problem`
    (``sample``, ``eval``, ``is_valid``) **and** the local-search methods
    (``random_state``, ``neighbors``, ``value``).

    The graph-search methods have default implementations that raise
    ``NotImplementedError``; override them if you want to use classical
    search on the problem.

    Attributes
    ----------
    minimize : bool
        If True, we are minimizing ``eval``/``value``; if False, maximizing.
    n_dims : int
        Dimensionality of the solution vector (number of decision variables).
    """

    def __init__(self, n_dims: int, minimize: bool = True,
                 name: str = "Discrete Problem"):
        """
        Parameters
        ----------
        n_dims : int
            Number of decision variables in the solution vector.
        minimize : bool, optional
            Optimization direction (default: True = minimize).
        name : str, optional
            Human-readable problem name.
        """
        super().__init__(name)
        self.n_dims = n_dims
        self.minimize = minimize

    # ------------------------------------------------------------------
    # Local-search interface
    # ------------------------------------------------------------------

    @abstractmethod
    def neighbors(self, state: np.ndarray) -> list[np.ndarray]:
        """Return neighboring solutions of *state*.

        Parameters
        ----------
        state : np.ndarray
            Current solution vector.

        Returns
        -------
        list[np.ndarray]
            List of neighbor solution vectors.
        """
        raise NotImplementedError

    def is_better(self, value1: float, value2: float) -> bool:
        """Compare two objective values respecting the optimization direction.

        Parameters
        ----------
        value1 : float
            First value.
        value2 : float
            Second value.

        Returns
        -------
        bool
            True if *value1* is strictly better than *value2*.
        """
        return value1 < value2 if self.minimize else value1 > value2

    # ------------------------------------------------------------------
    # Graph-search interface  (optional — override in subclass)
    # ------------------------------------------------------------------

    @property
    def initial_state(self):
        """Starting state for graph search.  Override in subclass."""
        raise NotImplementedError(
            "initial_state is not defined — override in subclass if you need "
            "classical graph search."
        )

    def actions(self, state) -> list:
        """Available actions from *state*.  Override for graph search."""
        raise NotImplementedError("actions() not implemented for this problem.")

    def result(self, state, action):
        """State resulting from *action*.  Override for graph search."""
        raise NotImplementedError("result() not implemented for this problem.")

    def cost(self, state, action, next_state) -> float:
        """Step cost.  Override for graph search."""
        raise NotImplementedError("cost() not implemented for this problem.")

    def is_goal(self, state) -> bool:
        """Goal test.  Override for graph search."""
        raise NotImplementedError("is_goal() not implemented for this problem.")

    def heuristic(self, state) -> float:
        """Admissible heuristic.  Override for informed graph search."""
        return 0.0



