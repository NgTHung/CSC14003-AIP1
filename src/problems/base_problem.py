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


class GraphSearchProblem(ABC):
    """
    Abstract base class for graph search problems.

    This class defines the interface for graph-based search problems
    where we need to find a path from an initial state to a goal state.

    Attributes
    ----------
    initial_state : Any
        The starting state of the search.
    goal_state : Any
        The target state to reach.
    """

    def __init__(self, initial_state, goal_state):
        """
        Initialize the GraphSearchProblem instance.

        Parameters
        ----------
        initial_state : Any
            The starting state of the search.
        goal_state : Any
            The target state to reach.
        """
        self.initial_state = initial_state
        self.goal_state = goal_state

    @abstractmethod
    def actions(self, state) -> list:
        """
        Return available actions from a given state.

        Parameters
        ----------
        state : Any
            The current state.

        Returns
        -------
        list
            List of available actions from this state.
        """
        raise NotImplementedError

    @abstractmethod
    def result(self, state, action):
        """
        Return the state that results from executing an action.

        Parameters
        ----------
        state : Any
            The current state.
        action : Any
            The action to execute.

        Returns
        -------
        Any
            The resulting state after executing the action.
        """
        raise NotImplementedError

    @abstractmethod
    def cost(self, state, action, next_state) -> float:
        """
        Return the cost of executing an action from state to next_state.

        Parameters
        ----------
        state : Any
            The current state.
        action : Any
            The action to execute.
        next_state : Any
            The resulting state.

        Returns
        -------
        float
            The cost of the action.
        """
        raise NotImplementedError

    def is_goal(self, state) -> bool:
        """
        Check if a state is the goal state.

        Parameters
        ----------
        state : Any
            The state to check.

        Returns
        -------
        bool
            True if the state is the goal state, False otherwise.
        """
        return state == self.goal_state

    def heuristic(self, state) -> float:
        """
        Return the estimated cost from state to the goal state.

        This is used by informed search algorithms like A* and Greedy Best-First.
        Default implementation returns 0 (uninformed search).

        Parameters
        ----------
        state : Any
            The state to evaluate.

        Returns
        -------
        float
            Estimated cost to reach the goal from this state.
        """
        return 0.0


class LocalSearchProblem(ABC):
    """
    Abstract base class for local search problems.

    This class defines the interface for optimization problems that can be
    solved using local search algorithms like Hill Climbing, Simulated Annealing, etc.

    Attributes
    ----------
    minimize : bool
        If True, we're minimizing the objective function; if False, maximizing.
    """

    def __init__(self, minimize: bool = True):
        """
        Initialize the LocalSearchProblem instance.

        Parameters
        ----------
        minimize : bool, optional
            If True, minimize the objective function; if False, maximize (default: True).
        """
        self.minimize = minimize

    @abstractmethod
    def random_state(self):
        """
        Generate a random state/solution.

        Returns
        -------
        Any
            A random state in the problem's state space.
        """
        raise NotImplementedError

    @abstractmethod
    def neighbors(self, state) -> list:
        """
        Return all neighboring states of the given state.

        Parameters
        ----------
        state : Any
            The current state.

        Returns
        -------
        list
            List of neighboring states.
        """
        raise NotImplementedError

    @abstractmethod
    def value(self, state) -> float:
        """
        Evaluate the objective function for a state.

        Parameters
        ----------
        state : Any
            The state to evaluate.

        Returns
        -------
        float
            The objective function value (lower is better for minimization,
            higher is better for maximization).
        """
        raise NotImplementedError

    def is_better(self, value1: float, value2: float) -> bool:
        """
        Check if value1 is better than value2 according to the optimization direction.

        Parameters
        ----------
        value1 : float
            First value to compare.
        value2 : float
            Second value to compare.

        Returns
        -------
        bool
            True if value1 is better than value2.
        """
        if self.minimize:
            return value1 < value2
        else:
            return value1 > value2
