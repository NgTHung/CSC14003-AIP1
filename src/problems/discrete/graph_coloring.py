"""Graph Coloring Problem — unified discrete formulation.

Supports **all** algorithm families through a single class:

* **Classical graph search** (BFS, DFS, UCS, A*, Greedy) — via state
  ``tuple(color_assignment)`` (partial assignment, -1 = uncolored) with
  ``actions / result / cost / is_goal / heuristic``.
* **Local search** (Hill Climbing, Steepest-Ascent) — via single-vertex
  re-coloring neighborhood with ``neighbors``.
* **Population / physics-inspired** (SA, GA, PSO, …) — via ``sample / eval``.

Solution representation for local / population algorithms is an **integer
vector** ``x`` of length ``n_vertices`` where ``x[i] ∈ {0, …, n_colors-1}``
is the color assigned to vertex *i*.  The ``eval`` function returns the
**number of constraint violations** (edges whose endpoints share a color)
so that standard minimization algorithms seek a legal coloring.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import override

import numpy as np

from problems.base_problem import DiscreteProblem

_DATA_DIR = Path(__file__).resolve().parent / "data"


class GraphColoring(DiscreteProblem):
    """Graph Coloring Problem.

    Given an undirected graph and *k* available colors, assign a color to
    every vertex so that no two adjacent vertices share the same color.

    Attributes
    ----------
    n_vertices : int
        Number of vertices (nodes) in the graph.
    n_colors : int
        Number of available colors.
    adj_matrix : np.ndarray
        Symmetric adjacency matrix of shape ``(n_vertices, n_vertices)``.
        ``adj_matrix[i][j] = 1`` means there is an edge between *i* and *j*.
    edges : list[tuple[int, int]]
        List of edges ``(u, v)`` with ``u < v``.
    """

    def __init__(
        self,
        adj_matrix: np.ndarray | list,
        n_colors: int,
        vertex_names: list[str] | None = None,
    ):
        """
        Parameters
        ----------
        adj_matrix : array-like
            Square symmetric adjacency matrix (0/1).
        n_colors : int
            Number of available colors.
        vertex_names : list[str], optional
            Human-readable vertex labels.  Defaults to ``["0", "1", …]``.

        Raises
        ------
        ValueError
            If the matrix is not square or *n_colors* < 1.
        """
        self.adj_matrix = np.asarray(adj_matrix, dtype=float)
        if self.adj_matrix.ndim != 2 or self.adj_matrix.shape[0] != self.adj_matrix.shape[1]:
            raise ValueError("adj_matrix must be a square 2-D array.")
        if n_colors < 1:
            raise ValueError("n_colors must be >= 1.")

        self.n_vertices = self.adj_matrix.shape[0]
        self.n_colors = n_colors
        self.vertex_names = vertex_names or [str(i) for i in range(self.n_vertices)]

        # Pre-compute edge list (each edge once)
        self.edges: list[tuple[int, int]] = []
        for i in range(self.n_vertices):
            for j in range(i + 1, self.n_vertices):
                if self.adj_matrix[i, j] != 0:
                    self.edges.append((i, j))

        # minimize number of conflicts (0 = legal coloring)
        super().__init__(n_dims=self.n_vertices, minimize=True,
                         name="Graph Coloring",
                         solution_type="assignment", domain_size=n_colors)

        # Graph-search initial state: all vertices uncolored (-1)
        self._initial_state = tuple([-1] * self.n_vertices)

    # ==================================================================
    # Problem interface  (population-based / physics-inspired)
    # ==================================================================

    @override
    def sample(self, pop_size: int = 1) -> np.ndarray:
        """Generate random color assignments of shape ``(pop_size, n_vertices)``.

        Each entry is an integer in ``[0, n_colors)``.
        """
        return np.random.randint(
            0, self.n_colors, size=(pop_size, self.n_vertices)
        ).astype(float)

    @override
    def eval(self, values: np.ndarray) -> float | np.ndarray:
        """Count conflict(s) — edges whose endpoints share a color.

        Parameters
        ----------
        values : np.ndarray
            Shape ``(n_vertices,)`` for one solution or ``(pop, n_vertices)``
            for a batch.

        Returns
        -------
        float or np.ndarray
            Number of conflicting edges.
        """
        if values.ndim == 1:
            return self._count_conflicts(values)
        return np.array([self._count_conflicts(row) for row in values])

    def _count_conflicts(self, coloring: np.ndarray) -> float:
        """Count the number of edges whose endpoints share the same color."""
        c = coloring.astype(int)
        conflicts = 0
        for u, v in self.edges:
            if c[u] == c[v]:
                conflicts += 1
        return float(conflicts)

    @override
    def is_valid(self, x: np.ndarray) -> bool:
        """Check if *x* is a valid, conflict-free coloring."""
        if x.shape[0] != self.n_vertices:
            return False
        c = x.astype(int)
        if np.any(c < 0) or np.any(c >= self.n_colors):
            return False
        return self._count_conflicts(x) == 0

    # ==================================================================
    # Local-search interface  (single-vertex re-coloring)
    # ==================================================================

    @override
    def random_neighbor(self, state: np.ndarray) -> np.ndarray:
        """Recolor one random vertex to a different random color.

        Efficient O(1) neighbour sampling without generating the full
        O(n * k) neighbourhood.
        """
        new_state = state.copy()
        v = np.random.randint(self.n_vertices)
        current_color = int(new_state[v])
        new_color = current_color
        while new_color == current_color:
            new_color = np.random.randint(self.n_colors)
        new_state[v] = float(new_color)
        return new_state

    @override
    def perturb(self, state: np.ndarray, **kwargs) -> np.ndarray:
        """Randomly recolor one vertex to a different color.

        Unlike the default in :class:`DiscreteProblem`, this
        preserves the color-domain constraint ``[0, n_colors)``.
        """
        return self.random_neighbor(state)

    @override
    def neighbors(self, state: np.ndarray) -> list[np.ndarray]:
        """Single-vertex re-coloring neighborhood.

        For each vertex, try every *other* color.
        """
        nbrs: list[np.ndarray] = []
        c = state.astype(int)
        for v in range(self.n_vertices):
            for color in range(self.n_colors):
                if color != c[v]:
                    nbr = state.copy()
                    nbr[v] = float(color)
                    nbrs.append(nbr)
        return nbrs

    # ==================================================================
    # Graph-search interface  (classical algorithms)
    # ==================================================================

    @property  # type: ignore[override]
    @override
    def initial_state(self):
        """Starting state: all vertices uncolored ``(-1, -1, …)``."""
        return self._initial_state

    @override
    def actions(self, state) -> list[int]:
        """Return available colors for the first uncolored vertex.

        The search colors vertices in order ``0, 1, …, n_vertices-1``.
        Returns ``[]`` if all vertices are colored (goal state).
        """
        # Find first uncolored vertex
        for v in range(self.n_vertices):
            if state[v] == -1:
                return list(range(self.n_colors))
        return []  # all colored

    @override
    def result(self, state, action) -> tuple:
        """Assign color *action* to the first uncolored vertex."""
        state_list = list(state)
        for v in range(self.n_vertices):
            if state_list[v] == -1:
                state_list[v] = action
                return tuple(state_list)
        return tuple(state_list)  # should not happen

    @override
    def cost(self, state, action, next_state) -> float:
        """Step cost: number of NEW conflicts introduced by this assignment.

        Handles both calling conventions:
        - ``cost(state, color_int, next_state)`` — from UCS / A*
        - ``cost(state, next_state_tuple, next_state)`` — from DFS / BFS / Greedy
        """
        # Determine the actual next-state tuple
        ns = next_state if not isinstance(next_state, tuple) else next_state
        if isinstance(action, tuple):
            ns = action  # DFS/BFS pass next_state as action
        # Find the vertex that was just colored
        for v in range(self.n_vertices):
            if state[v] == -1 and ns[v] != -1:
                color = ns[v]
                conflicts = 0
                for u in range(self.n_vertices):
                    if u != v and self.adj_matrix[v, u] != 0 and ns[u] == color:
                        conflicts += 1
                return float(conflicts)
        return 0.0

    @override
    def is_goal(self, state) -> bool:
        """All vertices colored."""
        return all(c != -1 for c in state)

    @override
    def heuristic(self, state) -> float:
        """Admissible heuristic: always 0 (trivially admissible).

        A tighter heuristic would require analyzing the remaining sub-graph,
        but 0 keeps things simple and correct for A*.
        """
        return 0.0

    # ==================================================================
    # Utility helpers
    # ==================================================================

    def decode_coloring(self, x: np.ndarray) -> dict:
        """Extract coloring info from a solution vector.

        Parameters
        ----------
        x : np.ndarray
            Color assignment of length ``n_vertices``.

        Returns
        -------
        dict
            ``{coloring, coloring_named, n_conflicts, is_legal}``
        """
        c = x.astype(int).tolist()
        conflicts = int(self._count_conflicts(x))
        return {
            "coloring": c,
            "coloring_named": {self.vertex_names[i]: c[i] for i in range(self.n_vertices)},
            "n_conflicts": conflicts,
            "is_legal": conflicts == 0,
        }

    def decode_path(self, path: list) -> dict:
        """Extract coloring info from a graph-search path.

        Parameters
        ----------
        path : list[tuple]
            Sequence of states (partial color assignments) from graph search.

        Returns
        -------
        dict
            ``{coloring, coloring_named, n_conflicts, is_legal}``
        """
        final = path[-1] if path else self._initial_state
        x = np.array(final, dtype=float)
        return self.decode_coloring(x)

    # ==================================================================
    # Factory methods
    # ==================================================================

    @staticmethod
    def from_file(filepath: str | Path) -> GraphColoring:
        """Load a Graph Coloring instance from a text file.

        File format (lines starting with ``#`` are ignored)::

            <n_vertices> <n_colors>
            <vertex_name_1> <vertex_name_2> ... <vertex_name_n>
            <row 0 of adjacency matrix>
            <row 1 of adjacency matrix>
            ...

        Parameters
        ----------
        filepath : str or Path
            Path to the data file.

        Returns
        -------
        GraphColoring
            A new GraphColoring instance built from the file data.
        """
        lines = [l.strip() for l in Path(filepath).read_text().splitlines()
                 if l.strip() and not l.strip().startswith("#")]
        parts = lines[0].split()
        n_vertices, n_colors = int(parts[0]), int(parts[1])
        vertex_names = lines[1].split()
        adj = []
        for i in range(n_vertices):
            adj.append([int(x) for x in lines[2 + i].split()])
        return GraphColoring(adj, n_colors=n_colors, vertex_names=vertex_names)

    @staticmethod
    def create_tiny() -> GraphColoring:
        """3-vertex triangle loaded from ``data/gc_tiny.txt``."""
        return GraphColoring.from_file(_DATA_DIR / "gc_tiny.txt")

    @staticmethod
    def create_small() -> GraphColoring:
        """4-vertex cycle loaded from ``data/gc_small.txt``."""
        return GraphColoring.from_file(_DATA_DIR / "gc_small.txt")

    @staticmethod
    def create_medium() -> GraphColoring:
        """Petersen graph loaded from ``data/gc_medium.txt``."""
        return GraphColoring.from_file(_DATA_DIR / "gc_medium.txt")

    @staticmethod
    def create_large() -> GraphColoring:
        """20-vertex, 65-edge graph loaded from ``data/gc_large.txt``."""
        return GraphColoring.from_file(_DATA_DIR / "gc_large.txt")

    @staticmethod
    def create_complete(n: int, n_colors: int | None = None) -> GraphColoring:
        """Complete graph K_n.  Chromatic number = n."""
        adj = np.ones((n, n), dtype=int) - np.eye(n, dtype=int)
        if n_colors is None:
            n_colors = n
        return GraphColoring(adj, n_colors=n_colors)

    @staticmethod
    def random(
        n_vertices: int,
        edge_prob: float = 0.3,
        n_colors: int = 3,
        seed: int | None = None,
    ) -> GraphColoring:
        """Random Erdos-Renyi graph ``G(n, p)``.

        Parameters
        ----------
        n_vertices : int
            Number of vertices.
        edge_prob : float
            Probability that each edge exists (default 0.3).
        n_colors : int
            Number of available colors (default 3).
        seed : int, optional
            Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        adj = np.zeros((n_vertices, n_vertices), dtype=int)
        for i in range(n_vertices):
            for j in range(i + 1, n_vertices):
                if rng.random() < edge_prob:
                    adj[i, j] = 1
                    adj[j, i] = 1
        return GraphColoring(adj, n_colors=n_colors)
