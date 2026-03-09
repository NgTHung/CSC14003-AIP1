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
    * **Local search** — via ``random_state()``, ``neighbors()``,
      ``random_neighbor()``, ``perturb()``, ``value()``, ``is_better()``.
    * **Classical graph search** — via ``actions()``, ``result()``, ``cost()``,
      ``is_goal()``, ``heuristic()``.

    Sub-classes **must** implement the abstract methods from :class:`Problem`
    (``sample``, ``eval``, ``is_valid``) **and** the local-search method
    ``neighbors``.

    Sub-classes **should** override ``random_neighbor`` for efficient
    single-neighbour sampling (the default picks a random element from
    ``neighbors()``, which can be expensive for large neighbourhoods).

    The graph-search methods have default implementations that raise
    ``NotImplementedError``; override them if you want to use classical
    search on the problem.

    Attributes
    ----------
    minimize : bool
        If True, we are minimizing ``eval``/``value``; if False, maximizing.
    n_dims : int
        Dimensionality of the solution vector (number of decision variables).
    solution_type : str
        ``"permutation"`` or ``"assignment"`` — controls how ACO
        algorithms construct solutions and lay pheromone.
    domain_size : int
        Number of possible values per dimension (assignment problems).
    """

    def __init__(self, n_dims: int, minimize: bool = True,
                 name: str = "Discrete Problem",
                 solution_type: str = "assignment",
                 domain_size: int = 2):
        """
        Parameters
        ----------
        n_dims : int
            Number of decision variables in the solution vector.
        minimize : bool, optional
            Optimization direction (default: True = minimize).
        name : str, optional
            Human-readable problem name.
        solution_type : str, optional
            ``"permutation"`` if solutions are permutations of
            ``[0, n_dims)`` (e.g. TSP).  ``"assignment"`` if each
            position independently takes a value from a finite
            domain (e.g. Knapsack, Graph Coloring).
            Default: ``"assignment"``.
        domain_size : int, optional
            Number of possible values per position for assignment
            problems (ignored for permutation problems).  Default: 2
            (binary).
        """
        super().__init__(name)
        self.n_dims = n_dims
        self.minimize = minimize
        self.solution_type = solution_type
        self.domain_size = domain_size

    # ------------------------------------------------------------------
    # Local-search interface
    # ------------------------------------------------------------------

    def random_state(self) -> np.ndarray:
        """Generate a single random valid solution.

        Convenience wrapper around ``sample(1)`` for local-search
        algorithms that need a random starting state.

        Returns
        -------
        np.ndarray
            A random solution vector of shape ``(n_dims,)``.
        """
        return self.sample(1).flatten()

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

    def random_neighbor(self, state: np.ndarray) -> np.ndarray:
        """Return a single random neighbour of *state*.

        Override this in subclasses to avoid generating the full
        neighbourhood when only one neighbour is needed (e.g. for
        SA, ABC, CS, FA).

        The default implementation calls ``neighbors()`` and picks
        one uniformly at random.

        Parameters
        ----------
        state : np.ndarray
            Current solution vector.

        Returns
        -------
        np.ndarray
            A neighbouring solution vector.
        """
        nbrs = self.neighbors(state)
        return nbrs[np.random.randint(len(nbrs))]

    def perturb(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """Generate a single neighbor by perturbing *state*.

        Default implementation delegates to ``random_neighbor()``.
        Subclasses may override for specialised perturbation logic
        (e.g. controlling the number of flips or the swap radius).

        Parameters
        ----------
        state : np.ndarray
            Current solution vector.
        **kwargs
            Subclass-specific options (e.g. ``n_flips`` for binary
            problems).

        Returns
        -------
        np.ndarray
            Perturbed solution vector.
        """
        return self.random_neighbor(state)

    def value(self, state: np.ndarray) -> float:
        """Evaluate the objective value of a single solution.

        Convenience wrapper around ``eval()`` for local-search
        algorithms.  Returns a Python float.

        Parameters
        ----------
        state : np.ndarray
            Solution vector of shape ``(n_dims,)``.

        Returns
        -------
        float
            Objective value.
        """
        return float(self.eval(state))

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



