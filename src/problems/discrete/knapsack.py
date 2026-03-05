"""0/1 Knapsack Problem — unified discrete formulation.

Supports **all** algorithm families through a single class:

* **Classical graph search** (BFS, DFS, UCS, A*, Greedy) — via tree-search
  state ``(item_index, remaining_capacity)`` with ``actions / result / cost /
  is_goal / heuristic``.
* **Local search** (Hill Climbing, Steepest-Ascent) — via binary-vector
  neighbors (single bit flips) with ``random_state / neighbors / value``.
* **Population / physics-inspired** (SA, GSA, HS) — via ``sample / eval``.

Solution representation for local / population algorithms is a **binary
vector** ``x`` of length ``n_items`` where ``x[i] = 1`` means item *i* is
selected.  The ``eval`` function **minimizes** the negative total value
(with a weight-violation penalty) so that standard minimization algorithms
maximize profit.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast, override

import numpy as np

from problems.base_problem import DiscreteProblem

_DATA_DIR = Path(__file__).resolve().parent / "data"


class Knapsack(DiscreteProblem):
    """0/1 Knapsack Problem.

    Attributes
    ----------
    n_items : int
        Number of available items.
    weights : np.ndarray
        Weight of each item, shape ``(n_items,)``.
    values : np.ndarray
        Value of each item, shape ``(n_items,)``.
    capacity : float
        Maximum weight the knapsack can hold.
    penalty_coeff : float
        Coefficient for the weight-violation penalty added to ``eval``.
    """

    def __init__(
        self,
        weights: np.ndarray | list,
        values: np.ndarray | list,
        capacity: float,
        penalty_coeff: float = 100.0,
    ):
        """
        Parameters
        ----------
        weights : array-like
            Weight of each item.
        values : array-like
            Value (profit) of each item.
        capacity : float
            Maximum total weight allowed.
        penalty_coeff : float, optional
            Multiplier for the weight-violation penalty used in ``eval``
            (default 100).

        Raises
        ------
        ValueError
            If *weights* and *values* differ in length or any weight < 0.
        """
        self.weights = np.asarray(weights, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.capacity = float(capacity)
        self.penalty_coeff = penalty_coeff

        if self.weights.shape != self.values.shape:
            raise ValueError(
                f"weights and values must have the same length, "
                f"got {self.weights.shape} and {self.values.shape}"
            )
        if np.any(self.weights < 0):
            raise ValueError("Item weights must be non-negative.")

        self.n_items = len(self.weights)

        # minimize=True because eval returns -value + penalty
        super().__init__(n_dims=self.n_items, minimize=True, name="0/1 Knapsack")

        # Pre-compute for graph search
        self._initial_state = (0, self.capacity)

    # ==================================================================
    # Problem interface  (population-based / physics-inspired)
    # ==================================================================

    @override
    def sample(self, pop_size: int = 1) -> np.ndarray:
        """Generate random binary solutions of shape ``(pop_size, n_items)``."""
        return np.random.randint(0, 2, size=(pop_size, self.n_items)).astype(float)

    @override
    def eval(self, values: np.ndarray) -> float | np.ndarray:
        """Evaluate solution(s).

        Returns ``-total_value + penalty`` so that **minimizing** this
        quantity **maximizes** profit while respecting the capacity.

        Parameters
        ----------
        values : np.ndarray
            Shape ``(n_items,)`` for one solution or ``(pop, n_items)``
            for a batch.
        """
        if values.ndim == 1:
            total_val = float(np.dot(values, self.values))
            total_wt = float(np.dot(values, self.weights))
            violation = max(0.0, total_wt - self.capacity)
            return -total_val + self.penalty_coeff * violation

        total_vals = values @ self.values
        total_wts = values @ self.weights
        violations = np.maximum(0.0, total_wts - self.capacity)
        return -total_vals + self.penalty_coeff * violations

    @override
    def is_valid(self, x: np.ndarray) -> bool:
        """Check if *x* is a feasible binary solution."""
        if x.shape[0] != self.n_items:
            return False
        if not np.all((x == 0) | (x == 1)):
            return False
        return float(np.dot(x, self.weights)) <= self.capacity

    # ==================================================================
    # Local-search interface
    # ==================================================================



    @override
    def neighbors(self, state: np.ndarray) -> list[np.ndarray]:
        """Single-bit-flip neighborhood."""
        nbrs: list[np.ndarray] = []
        for i in range(self.n_items):
            nbr = state.copy()
            nbr[i] = 1.0 - nbr[i]
            nbrs.append(nbr)
        return nbrs

    # ==================================================================
    # Graph-search interface  (classical algorithms)
    # ==================================================================

    @property  # type: ignore[override]
    @override
    def initial_state(self):
        """Starting state ``(0, capacity)``."""
        return self._initial_state

    @override
    def actions(self, state) -> list[str]:
        """Return ``["take", "skip"]`` or ``["skip"]``."""
        idx, remaining = state
        if idx >= self.n_items:
            return []
        acts: list[str] = ["skip"]
        if self.weights[idx] <= remaining:
            acts.insert(0, "take")
        return acts

    @override
    def result(self, state, action) -> tuple[int, float]:
        """Next state after *action*."""
        idx, remaining = state
        if action == "take":
            return (idx + 1, remaining - self.weights[idx])
        return (idx + 1, remaining)

    @override
    def cost(self, state, action, next_state) -> float:
        """Negative item value for 'take', 0 for 'skip'."""
        if action == "take":
            idx, _ = state
            return cast(float, -self.values[idx])
        return 0.0

    @override
    def is_goal(self, state) -> bool:
        """All items considered."""
        idx, _ = state
        return idx == self.n_items

    @override
    def heuristic(self, state) -> float:
        """Fractional-knapsack relaxation (admissible, negated)."""
        idx, remaining = state
        if idx >= self.n_items:
            return 0.0

        remaining_items = [
            (self.values[i], self.weights[i])
            for i in range(idx, self.n_items)
            if self.weights[i] > 0
        ]
        free_value = sum(
            self.values[i]
            for i in range(idx, self.n_items)
            if self.weights[i] == 0 and self.values[i] > 0
        )
        remaining_items.sort(key=lambda vw: vw[0] / vw[1], reverse=True)

        cap = remaining
        bound = free_value
        for v, w in remaining_items:
            if w <= cap:
                bound += v
                cap -= w
            else:
                bound += v * (cap / w)
                break
        return -bound

    # ==================================================================
    # Utility helpers
    # ==================================================================

    def decode_path(self, path: list) -> dict:
        """Extract selection from a graph-search path.

        Parameters
        ----------
        path : list[tuple[int, float]]
            Sequence of ``(item_index, remaining_capacity)`` states.

        Returns
        -------
        dict
            ``{selected_items, total_value, total_weight}``
        """
        selected: list[int] = []
        for i in range(len(path) - 1):
            _idx_cur, cap_cur = path[i]
            _idx_next, cap_next = path[i + 1]
            if cap_next < cap_cur:
                selected.append(_idx_cur)
        total_value = float(self.values[selected].sum()) if selected else 0.0
        total_weight = float(self.weights[selected].sum()) if selected else 0.0
        return {
            "selected_items": selected,
            "total_value": total_value,
            "total_weight": total_weight,
        }

    def decode_binary(self, x: np.ndarray) -> dict:
        """Extract selection from a binary-vector solution.

        Parameters
        ----------
        x : np.ndarray
            Binary vector of length ``n_items``.

        Returns
        -------
        dict
            ``{selected_items, total_value, total_weight}``
        """
        selected = [int(i) for i in range(self.n_items) if x[i] >= 0.5]
        total_value = float(self.values[selected].sum()) if selected else 0.0
        total_weight = float(self.weights[selected].sum()) if selected else 0.0
        return {
            "selected_items": selected,
            "total_value": total_value,
            "total_weight": total_weight,
        }

    # ==================================================================
    # Factory methods
    # ==================================================================

    @staticmethod
    def from_file(filepath: str | Path) -> Knapsack:
        """Load a Knapsack instance from a text file.

        File format (lines starting with ``#`` are ignored)::

            <n_items> <capacity>
            <weight_1> <weight_2> ... <weight_n>
            <value_1>  <value_2>  ... <value_n>

        Parameters
        ----------
        filepath : str or Path
            Path to the data file.

        Returns
        -------
        Knapsack
            A new Knapsack instance built from the file data.
        """
        lines = [l.strip() for l in Path(filepath).read_text().splitlines()
                 if l.strip() and not l.strip().startswith("#")]
        parts = lines[0].split()
        n_items, capacity = int(parts[0]), float(parts[1])
        weights = [float(x) for x in lines[1].split()]
        values = [float(x) for x in lines[2].split()]
        return Knapsack(weights=weights, values=values, capacity=capacity)

    @staticmethod
    def create_tiny() -> Knapsack:
        """3-item instance loaded from ``data/knapsack_tiny.txt``."""
        return Knapsack.from_file(_DATA_DIR / "knapsack_tiny.txt")

    @staticmethod
    def create_small() -> Knapsack:
        """4-item instance loaded from ``data/knapsack_small.txt``."""
        return Knapsack.from_file(_DATA_DIR / "knapsack_small.txt")

    @staticmethod
    def create_medium() -> Knapsack:
        """10-item benchmark loaded from ``data/knapsack_medium.txt``."""
        return Knapsack.from_file(_DATA_DIR / "knapsack_medium.txt")

    @staticmethod
    def create_large() -> Knapsack:
        """30-item instance loaded from ``data/knapsack_large.txt``."""
        return Knapsack.from_file(_DATA_DIR / "knapsack_large.txt")

    @staticmethod
    def random(
        n_items: int,
        capacity: float | None = None,
        seed: int | None = None,
    ) -> Knapsack:
        """Random instance.  Capacity defaults to ~50 % of total weight."""
        rng = np.random.default_rng(seed)
        weights = rng.integers(1, 50, size=n_items).astype(float)
        values = rng.integers(1, 100, size=n_items).astype(float)
        if capacity is None:
            capacity = float(weights.sum()) * 0.5
        return Knapsack(weights, values, capacity)
