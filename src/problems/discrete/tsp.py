"""Traveling Salesman Problem (TSP) for discrete optimization.

Given a set of cities and distances between them, find the shortest possible
tour that visits each city exactly once and returns to the starting city.

NP-hard combinatorial optimization problem with n!/2n distinct tours for n cities.
"""

from typing import cast, override

import numpy as np

from .. import Problem


class TSP(Problem):
    """Symmetric Traveling Salesman Problem.

    A solution is represented as a permutation of city indices.
    The objective is to minimize the total tour distance.

    Attributes
    ----------
    n_cities : int
        Number of cities.
    distance_matrix : np.ndarray
        Symmetric matrix of pairwise distances (n_cities, n_cities).
    """

    _n_cities: int
    _distance_matrix: np.ndarray

    def __init__(self, distance_matrix: np.ndarray):
        """Initialize TSP from a distance matrix.

        Parameters
        ----------
        distance_matrix : np.ndarray
            Symmetric matrix of shape (n_cities, n_cities) where entry
            (i, j) is the distance between city i and city j.

        Raises
        ------
        ValueError
            If the matrix is not square.
        """
        if (
            distance_matrix.ndim != 2
            or distance_matrix.shape[0] != distance_matrix.shape[1]
        ):
            raise ValueError(
                f"Distance matrix must be square, got shape {distance_matrix.shape}"
            )
        self._n_cities = distance_matrix.shape[0]
        self._distance_matrix = distance_matrix
        super().__init__("TSP")

    @staticmethod
    def from_coordinates(coords: np.ndarray) -> "TSP":
        """Create a TSP instance from 2D city coordinates.

        Parameters
        ----------
        coords : np.ndarray
            City positions of shape (n_cities, 2).

        Returns
        -------
        TSP
            A TSP instance with Euclidean distances.
        """
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        distance_matrix = np.sqrt(np.sum(diff**2, axis=-1))
        return TSP(distance_matrix)

    @staticmethod
    def random(n_cities: int, seed: int | None = None) -> "TSP":
        """Generate a random TSP instance with cities in [0, 100]^2.

        Parameters
        ----------
        n_cities : int
            Number of cities.
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        TSP
            A randomly generated TSP instance.
        """
        rng = np.random.default_rng(seed)
        coords = rng.uniform(0, 100, size=(n_cities, 2))
        return TSP.from_coordinates(coords)

    @property
    def n_cities(self) -> int:
        """Number of cities."""
        return self._n_cities

    @property
    def distance_matrix(self) -> np.ndarray:
        """Distance matrix (n_cities, n_cities)."""
        return self._distance_matrix

    def tour_distance(self, tour: np.ndarray) -> float:
        """Calculate total distance of a tour.

        Parameters
        ----------
        tour : np.ndarray
            Permutation of city indices, shape (n_cities,).

        Returns
        -------
        float
            Total tour distance including return to start.
        """
        total = 0.0
        for i, ti in enumerate(tour):
            total += self._distance_matrix[ti][tour[(i + 1) % len(tour)]]
        return cast(float, total)

    @override
    def eval(self, values: np.ndarray) -> float | np.ndarray:
        """Evaluate tour distance(s).

        Parameters
        ----------
        values : np.ndarray
            Shape (n_cities,) for a single tour or
            (pop_size, n_cities) for a batch.

        Returns
        -------
        float or np.ndarray
            Total tour distance(s).
        """
        if values.ndim == 1:
            return self.tour_distance(values)
        return np.array([self.tour_distance(tour) for tour in values])

    @override
    def sample(self, pop_size: int = 1) -> np.ndarray:
        """Generate random tours (permutations).

        Parameters
        ----------
        pop_size : int, optional
            Number of tours to generate. Default is 1.

        Returns
        -------
        np.ndarray
            Random permutations of shape (pop_size, n_cities).
        """
        tours = np.array(
            [np.random.permutation(self._n_cities) for _ in range(pop_size)]
        )
        return tours

    @override
    def is_valid(self, x: np.ndarray) -> bool:
        """Check if a solution is a valid tour.

        A valid tour visits each city exactly once.

        Parameters
        ----------
        x : np.ndarray
            Solution vector of shape (n_cities,).

        Returns
        -------
        bool
            True if x is a valid permutation of city indices.
        """
        if x.shape[0] != self._n_cities:
            return False
        return len(set(x)) == self._n_cities and all(0 <= c < self._n_cities for c in x)
