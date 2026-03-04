"""Travelling Salesman Problem (TSP) — unified discrete formulation.

Supports **all** algorithm families through a single class:

* **Classical graph search** (BFS, DFS, UCS, A*, Greedy) — via state
  ``(current_city, frozenset(visited))`` with ``actions / result / cost /
  is_goal / heuristic``.
* **Local search** (Hill Climbing, Steepest-Ascent) — via permutation-based
  neighbors (2-opt swaps) with ``neighbors``.
* **Population / physics-inspired** (SA, GA, PSO, …) — via ``sample / eval``.

Solution representation for local / population algorithms is a **permutation
vector** ``x`` of length ``n_cities`` representing the visit order.  The
``eval`` function returns the **total tour distance** (round-trip) so that
standard minimization algorithms find the shortest tour.
"""

from __future__ import annotations

from typing import override

import numpy as np

from problems.base_problem import DiscreteProblem


class TSP(DiscreteProblem):
    """Travelling Salesman Problem.

    Attributes
    ----------
    n_cities : int
        Number of cities.
    dist_matrix : np.ndarray
        Symmetric distance matrix of shape ``(n_cities, n_cities)``.
    city_names : list[str] | None
        Optional human-readable city labels.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray | list,
        city_names: list[str] | None = None,
    ):
        """
        Parameters
        ----------
        dist_matrix : array-like
            Square symmetric distance matrix.  ``dist_matrix[i][j]`` is the
            distance from city *i* to city *j*.
        city_names : list[str], optional
            Human-readable city names.  Defaults to ``["0", "1", …]``.

        Raises
        ------
        ValueError
            If the matrix is not square.
        """
        self.dist_matrix = np.asarray(dist_matrix, dtype=float)
        if self.dist_matrix.ndim != 2 or self.dist_matrix.shape[0] != self.dist_matrix.shape[1]:
            raise ValueError("dist_matrix must be a square 2-D array.")

        self.n_cities = self.dist_matrix.shape[0]
        self.city_names = city_names or [str(i) for i in range(self.n_cities)]

        # minimize total tour distance
        super().__init__(n_dims=self.n_cities, minimize=True, name="TSP")

        # For graph search: start at city 0
        self._initial_state = (0, frozenset([0]))

    # ==================================================================
    # Problem interface  (population-based / physics-inspired)
    # ==================================================================

    @override
    def sample(self, pop_size: int = 1) -> np.ndarray:
        """Generate random permutation solutions of shape ``(pop_size, n_cities)``.

        Each row is a permutation of ``[0, 1, …, n_cities-1]`` representing
        the visit order.
        """
        pop = np.zeros((pop_size, self.n_cities), dtype=float)
        for i in range(pop_size):
            pop[i] = np.random.permutation(self.n_cities).astype(float)
        return pop

    @override
    def eval(self, values: np.ndarray) -> float | np.ndarray:
        """Evaluate tour distance(s).

        Parameters
        ----------
        values : np.ndarray
            Shape ``(n_cities,)`` for one tour or ``(pop, n_cities)``
            for a batch.  Each row is a permutation of city indices.

        Returns
        -------
        float or np.ndarray
            Total round-trip tour distance(s).
        """
        if values.ndim == 1:
            return self._tour_distance(values)

        return np.array([self._tour_distance(row) for row in values])

    def _tour_distance(self, perm: np.ndarray) -> float:
        """Compute the round-trip tour distance for a single permutation."""
        order = perm.astype(int)
        dist = 0.0
        for i in range(len(order) - 1):
            dist += self.dist_matrix[order[i], order[i + 1]]
        # Return to start
        dist += self.dist_matrix[order[-1], order[0]]
        return float(dist)

    @override
    def is_valid(self, x: np.ndarray) -> bool:
        """Check if *x* is a valid permutation of ``[0, …, n_cities-1]``."""
        if x.shape[0] != self.n_cities:
            return False
        order = np.sort(x.astype(int))
        return np.array_equal(order, np.arange(self.n_cities))

    # ==================================================================
    # Local-search interface  (2-opt neighborhood)
    # ==================================================================

    @override
    def perturb(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """Random 2-opt swap: reverse a random sub-tour segment.

        This preserves the permutation structure, unlike the default
        bit-flip perturbation in :class:`DiscreteProblem`.
        """
        new_state = state.copy()
        n = len(new_state)
        i, j = sorted(np.random.choice(n, size=2, replace=False))
        new_state[i:j + 1] = new_state[i:j + 1][::-1]
        return new_state

    @override
    def neighbors(self, state: np.ndarray) -> list[np.ndarray]:
        """2-opt swap neighborhood.

        For each pair ``(i, j)`` with ``i < j``, reverse the sub-tour
        between positions *i* and *j* (inclusive).
        """
        nbrs: list[np.ndarray] = []
        n = len(state)
        for i in range(n - 1):
            for j in range(i + 1, n):
                nbr = state.copy()
                nbr[i:j + 1] = nbr[i:j + 1][::-1]
                nbrs.append(nbr)
        return nbrs

    # ==================================================================
    # Graph-search interface  (classical algorithms)
    # ==================================================================

    @property  # type: ignore[override]
    @override
    def initial_state(self):
        """Starting state ``(current_city=0, visited={0})``."""
        return self._initial_state

    @override
    def actions(self, state) -> list[int]:
        """Return list of unvisited cities (or city 0 if all visited)."""
        current, visited = state
        if len(visited) == self.n_cities:
            # All cities visited — only action is to return home
            return [0]
        return [c for c in range(self.n_cities) if c not in visited]

    @override
    def result(self, state, action) -> tuple[int, frozenset]:
        """Move to *action* city."""
        current, visited = state
        return (action, visited | frozenset([action]))

    @override
    def cost(self, state, action, next_state) -> float:
        """Travel distance from current city to *action* city.

        Handles both calling conventions:
        - ``cost(state, action_city, next_state)`` — from UCS / A*
        - ``cost(state, next_state, next_state)`` — from DFS / BFS / Greedy
        """
        current, _ = state
        # action may be an int (city) or a state tuple (city, visited)
        target = action[0] if isinstance(action, tuple) else int(action)
        return self.dist_matrix[current, target]

    @override
    def is_goal(self, state) -> bool:
        """All cities visited **and** returned to start."""
        current, visited = state
        return len(visited) == self.n_cities and current == 0

    @override
    def heuristic(self, state) -> float:
        """Minimum spanning tree lower-bound on remaining tour cost.

        Uses a simple nearest-neighbor lower bound for remaining unvisited
        cities — admissible for A*.
        """
        current, visited = state
        unvisited = [c for c in range(self.n_cities) if c not in visited]
        if not unvisited:
            # Only need to return to start
            return self.dist_matrix[current, 0]

        # Lower bound: for each unvisited city + current + home, sum the
        # minimum edge out of each.
        nodes = [current] + unvisited + [0]
        h = 0.0
        for node in [current] + unvisited:
            candidates = [c for c in nodes if c != node]
            if candidates:
                h += min(self.dist_matrix[node, c] for c in candidates)
        return float(h)

    # ==================================================================
    # Utility helpers
    # ==================================================================

    def decode_permutation(self, perm: np.ndarray) -> dict:
        """Extract tour info from a permutation solution.

        Parameters
        ----------
        perm : np.ndarray
            Permutation of ``[0, …, n_cities-1]``.

        Returns
        -------
        dict
            ``{tour, tour_names, total_distance}``
        """
        order = perm.astype(int).tolist()
        round_trip = order + [order[0]]
        return {
            "tour": round_trip,
            "tour_names": [self.city_names[i] for i in round_trip],
            "total_distance": self._tour_distance(perm),
        }

    def decode_path(self, path: list) -> dict:
        """Extract tour info from a graph-search path.

        Parameters
        ----------
        path : list[tuple]
            Sequence of ``(city, visited_set)`` states from graph search.

        Returns
        -------
        dict
            ``{tour, tour_names, total_distance}``
        """
        tour = [s[0] for s in path]
        total_dist = 0.0
        for i in range(len(tour) - 1):
            total_dist += self.dist_matrix[tour[i], tour[i + 1]]
        return {
            "tour": tour,
            "tour_names": [self.city_names[c] for c in tour],
            "total_distance": total_dist,
        }

    # ==================================================================
    # Factory methods
    # ==================================================================

    @staticmethod
    def create_small() -> TSP:
        """4-city symmetric instance."""
        dist = [
            [0, 10, 15, 20],
            [10,  0, 35, 25],
            [15, 35,  0, 30],
            [20, 25, 30,  0],
        ]
        return TSP(dist, city_names=["A", "B", "C", "D"])

    @staticmethod
    def create_medium() -> TSP:
        """6-city symmetric instance."""
        dist = [
            [ 0, 12, 10, 19, 8,  14],
            [12,  0,  3, 7,  6,  20],
            [10,  3,  0, 2,  20, 11],
            [19,  7,  2, 0,  4,  15],
            [ 8,  6, 20, 4,  0,  17],
            [14, 20, 11, 15, 17,  0],
        ]
        return TSP(dist, city_names=["A", "B", "C", "D", "E", "F"])

    @staticmethod
    def random(
        n_cities: int,
        max_dist: float = 100.0,
        seed: int | None = None,
    ) -> TSP:
        """Random symmetric TSP instance.

        Generates cities with random coordinates in ``[0, max_dist]²``
        and computes Euclidean distances.
        """
        rng = np.random.default_rng(seed)
        coords = rng.uniform(0, max_dist, size=(n_cities, 2))
        # Euclidean distance matrix
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))
        names = [str(i) for i in range(n_cities)]
        return TSP(dist_matrix, city_names=names)
