"""Differential Evolution (DE) algorithm for continuous optimization.

A stochastic, population-based optimizer that perturbs solution vectors using
**weighted differences** of randomly selected population members rather than
the probability-distribution-based mutation of traditional Genetic Algorithms.

The algorithm follows a simple loop per generation:
1. **Mutation** — For each target vector, create a *donor* vector using a
   weighted combination of other population members (strategy-dependent).
2. **Crossover** — Mix the donor with the target to form a *trial* vector
   (binomial or exponential).
3. **Selection** — Replace the target with the trial if the trial is at
   least as good (greedy: ``f(trial) ≤ f(target)``).

Supported mutation strategies (``DE/base/num_diffs``):
- ``rand/1``  — classic, robust exploration.
- ``best/1``  — fast convergence, exploitation-heavy.
- ``target-to-best/1`` — balanced exploitation + exploration.
- ``best/2``  — stronger perturbation to escape local optima.

Supported crossover types:
- ``bin``  — binomial (component-wise independent coin-flip).
- ``exp``  — exponential (contiguous segment from the donor).

Reference: Storn, R. & Price, K. (1997). Differential Evolution — A Simple
and Efficient Heuristic for Global Optimization over Continuous Spaces.
Journal of Global Optimization, 11(4), 341-359.
"""

from dataclasses import dataclass
from enum import Enum
from typing import cast, override

import numpy as np
from problems import ContinuousProblem
from algorithm import Model


# ======================================================================
# Configuration
# ======================================================================


class MutationStrategy(Enum):
    """Available DE mutation strategies (``DE/x/y``)."""

    RAND_1 = "rand/1"
    BEST_1 = "best/1"
    TARGET_TO_BEST_1 = "target-to-best/1"
    BEST_2 = "best/2"


class DECrossoverType(Enum):
    """Available DE crossover mechanisms."""

    BIN = "bin"
    EXP = "exp"


class VariableType(Enum):
    """Decision-variable interpretation during evaluation."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


@dataclass
class DEParameter:
    """Configuration parameters for Differential Evolution.

    Attributes
    ----------
    pop_size : int
        Number of individuals in the population (``NP``).
        Recommendation: between ``5 * D`` and ``10 * D`` where ``D`` is the
        dimensionality.  Minimum of 4 is required for the mutation operators.
    F : float
        Scaling (amplification) factor for the difference vector(s).
        Controls exploration vs. exploitation.  Typical: 0.4-1.0,
        good default: 0.5.
    Cr : float
        Crossover rate (0 ≤ Cr ≤ 1).  Probability that a trial-vector
        component is inherited from the *donor* rather than the *target*.
        Use low values (0.1-0.2) for separable problems, high values
        (≈ 0.9) for non-separable problems.
    cycle : int
        Maximum number of generations.
    strategy : MutationStrategy
        Mutation strategy.  Default: ``rand/1`` (classic, most popular).
    crossover_type : DECrossoverType
        Crossover mechanism.  Default: ``bin`` (binomial).
    variable_type : VariableType
        How decision variables are interpreted during fitness evaluation.
        ``continuous`` passes values as-is; ``discrete`` rounds them to
        the nearest integer before calling the objective function (the
        internal population retains full float precision).
        Default: ``continuous``.
    """

    pop_size: int
    F: float
    Cr: float
    cycle: int
    strategy: MutationStrategy = MutationStrategy.RAND_1
    crossover_type: DECrossoverType = DECrossoverType.BIN
    variable_type: VariableType = VariableType.CONTINUOUS


# ======================================================================
# Differential Evolution
# ======================================================================


class DifferentialEvolution(
    Model[ContinuousProblem, np.ndarray | None, float, DEParameter]
):
    """Differential Evolution for continuous (or discretised) optimization.

    Algorithm outline per generation:
    1. **Mutation** — For each target vector *i*, build a donor vector *V*
       using the configured strategy (``rand/1``, ``best/1``, etc.).
    2. **Crossover** — Combine the donor *V* with the target *X_i* to
       produce a trial vector *U* (binomial or exponential).  At least
       one component is guaranteed to come from the donor (``j_rand``).
    3. **Boundary handling** — Components that violate bounds are
       randomly re-initialised within the feasible region.
    4. **Selection** — Greedy: the trial replaces the target if
       ``f(U) ≤ f(X_i)`` (≤ allows traversal of flat landscapes).
    """

    population: np.ndarray   # shape (pop_size, n_dim)
    fitness: np.ndarray      # shape (pop_size,)
    n_dim: int

    def __init__(
        self, configuration: DEParameter, problem: ContinuousProblem
    ):
        """Initialize Differential Evolution.

        Parameters
        ----------
        configuration : DEParameter
            Algorithm hyperparameters.
        problem : ContinuousProblem
            Continuous optimization problem to solve.

        Raises
        ------
        ValueError
            If ``pop_size`` is less than 4 (required by mutation operators)
            or if ``pop_size`` is less than 5 when using ``best/2``.
        """
        min_pop = 5 if configuration.strategy == MutationStrategy.BEST_2 else 4
        if configuration.pop_size < min_pop:
            raise ValueError(
                f"pop_size must be ≥ {min_pop} for strategy "
                f"'{configuration.strategy.value}', got {configuration.pop_size}"
            )

        super().__init__(configuration, problem)
        self.name = "Differential Evolution"
        self.n_dim = problem.n_dim

        # Initialise population uniformly within bounds
        self.population = self._initialize_population()
        self.fitness = self._evaluate_fitness(self.population)

        # Track global best
        best_idx = int(np.argmin(self.fitness))
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = float(self.fitness[best_idx])
        self.history = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize_population(self) -> np.ndarray:
        """Generate random individuals uniformly within the problem bounds.

        Delegates to the problem's ``sample`` method, which guarantees
        feasibility.

        Returns
        -------
        np.ndarray
            Population matrix of shape ``(pop_size, n_dim)``.
        """
        return self.problem.sample(self.conf.pop_size)

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------

    def _evaluate_fitness(self, vectors: np.ndarray) -> np.ndarray:
        """Evaluate objective values for one or more solution vectors.

        When ``variable_type`` is ``discrete``, the vectors are rounded to
        the nearest integer **only** for the objective-function call; the
        population retains full float precision.

        Parameters
        ----------
        vectors : np.ndarray
            Solution(s) of shape ``(n_dim,)`` or ``(pop_size, n_dim)``.

        Returns
        -------
        np.ndarray
            Objective values.  Scalar wrapped in an array for a single
            vector, or shape ``(pop_size,)`` for a batch.
        """
        eval_vectors = vectors
        if self.conf.variable_type == VariableType.DISCRETE:
            eval_vectors = np.rint(vectors)

        result = self.problem.eval(eval_vectors)
        return np.atleast_1d(np.asarray(result, dtype=float))

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _pick_random_indices(
        self, target_idx: int, count: int
    ) -> list[int]:
        """Select *count* mutually-exclusive random population indices.

        All returned indices are distinct from each other **and** from
        *target_idx*.

        Parameters
        ----------
        target_idx : int
            Index of the current target vector (excluded).
        count : int
            Number of indices to draw.

        Returns
        -------
        list[int]
            List of unique random indices.
        """
        candidates = list(range(self.conf.pop_size))
        candidates.remove(target_idx)
        chosen = np.random.choice(candidates, size=count, replace=False)
        return chosen.tolist()

    def _mutate(self, target_idx: int) -> np.ndarray:
        """Create a donor (mutant) vector for the given target.

        The mutation formula depends on ``self.conf.strategy``:

        - **rand/1**: :math:`V = X_{r1} + F (X_{r2} - X_{r3})`
        - **best/1**: :math:`V = X_{best} + F (X_{r1} - X_{r2})`
        - **target-to-best/1**:
          :math:`V = X_i + F (X_{best} - X_i) + F (X_{r1} - X_{r2})`
        - **best/2**:
          :math:`V = X_{best} + F (X_{r1} - X_{r2}) + F (X_{r3} - X_{r4})`

        Parameters
        ----------
        target_idx : int
            Index of the current target vector in the population.

        Returns
        -------
        np.ndarray
            Donor vector of shape ``(n_dim,)``.
        """
        F = self.conf.F
        strategy = self.conf.strategy

        if strategy == MutationStrategy.RAND_1:
            r1, r2, r3 = self._pick_random_indices(target_idx, 3)
            donor = (
                self.population[r1]
                + F * (self.population[r2] - self.population[r3])
            )

        elif strategy == MutationStrategy.BEST_1:
            r1, r2 = self._pick_random_indices(target_idx, 2)
            assert self.best_solution is not None
            donor = (
                self.best_solution
                + F * (self.population[r1] - self.population[r2])
            )

        elif strategy == MutationStrategy.TARGET_TO_BEST_1:
            r1, r2 = self._pick_random_indices(target_idx, 2)
            assert self.best_solution is not None
            donor = (
                self.population[target_idx]
                + F * (self.best_solution - self.population[target_idx])
                + F * (self.population[r1] - self.population[r2])
            )

        elif strategy == MutationStrategy.BEST_2:
            r1, r2, r3, r4 = self._pick_random_indices(target_idx, 4)
            assert self.best_solution is not None
            donor = (
                self.best_solution
                + F * (self.population[r1] - self.population[r2])
                + F * (self.population[r3] - self.population[r4])
            )

        else:
            raise ValueError(f"Unknown mutation strategy: {strategy}")

        return donor

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------

    def _crossover(
        self, target: np.ndarray, donor: np.ndarray
    ) -> np.ndarray:
        """Produce a trial vector by mixing the target and donor.

        A randomly chosen dimension ``j_rand`` is **always** inherited from
        the donor to ensure the trial differs from the target by at least
        one component.

        Parameters
        ----------
        target : np.ndarray
            Current target vector, shape ``(n_dim,)``.
        donor : np.ndarray
            Donor (mutant) vector, shape ``(n_dim,)``.

        Returns
        -------
        np.ndarray
            Trial vector of shape ``(n_dim,)``.
        """
        if self.conf.crossover_type == DECrossoverType.BIN:
            return self._binomial_crossover(target, donor)
        else:
            return self._exponential_crossover(target, donor)

    def _binomial_crossover(
        self, target: np.ndarray, donor: np.ndarray
    ) -> np.ndarray:
        """Binomial (component-wise) crossover.

        Each component is independently inherited from the donor with
        probability ``Cr``, except for one guaranteed dimension ``j_rand``.

        Parameters
        ----------
        target : np.ndarray
            Target vector, shape ``(n_dim,)``.
        donor : np.ndarray
            Donor vector, shape ``(n_dim,)``.

        Returns
        -------
        np.ndarray
            Trial vector.
        """
        trial = target.copy()
        j_rand = np.random.randint(0, self.n_dim)

        for j in range(self.n_dim):
            if np.random.random() < self.conf.Cr or j == j_rand:
                trial[j] = donor[j]

        return trial

    def _exponential_crossover(
        self, target: np.ndarray, donor: np.ndarray
    ) -> np.ndarray:
        """Exponential (contiguous-segment) crossover.

        Starting from a random position ``j_rand``, consecutive components
        are inherited from the donor as long as independent random draws
        are below ``Cr``.  The segment wraps around the vector.

        Parameters
        ----------
        target : np.ndarray
            Target vector, shape ``(n_dim,)``.
        donor : np.ndarray
            Donor vector, shape ``(n_dim,)``.

        Returns
        -------
        np.ndarray
            Trial vector.
        """
        trial = target.copy()
        j_rand = np.random.randint(0, self.n_dim)

        j = j_rand
        L = 0  # length of the contiguous donor segment
        while True:
            trial[j] = donor[j]
            L += 1
            j = (j + 1) % self.n_dim
            if np.random.random() >= self.conf.Cr or L >= self.n_dim:
                break

        return trial

    # ------------------------------------------------------------------
    # Boundary handling
    # ------------------------------------------------------------------

    def _handle_boundaries(self, vector: np.ndarray) -> np.ndarray:
        """Repair components that exceed the problem bounds.

        Out-of-bounds components are **randomly re-initialised** within
        ``[lower, upper]`` for the offending dimension, preserving the
        stochastic character of the search.

        Parameters
        ----------
        vector : np.ndarray
            Solution vector of shape ``(n_dim,)``.

        Returns
        -------
        np.ndarray
            Repaired vector guaranteed to lie within bounds.
        """
        lower = self.problem.bounds[:, 0]
        upper = self.problem.bounds[:, 1]

        below = vector < lower
        above = vector > upper
        violated = below | above

        if np.any(violated):
            vector = vector.copy()
            rand_vals = np.random.uniform(lower, upper)
            vector[violated] = rand_vals[violated]

        return vector

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_best(self):
        """Update the global best from the current population."""
        best_idx = int(np.argmin(self.fitness))
        if self.fitness[best_idx] < self.best_fitness:
            self.best_fitness = float(self.fitness[best_idx])
            self.best_solution = self.population[best_idx].copy()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    @override
    def run(self) -> np.ndarray:
        """Execute the Differential Evolution algorithm.

        For each generation, every individual in the population is used
        once as the *target*.  A donor is created via mutation, combined
        with the target via crossover to form a trial, and the trial
        replaces the target under greedy selection (``f(trial) ≤ f(target)``).

        Returns
        -------
        np.ndarray
            Best solution found after all generations.
        """
        for _ in range(self.conf.cycle):
            for i in range(self.conf.pop_size):
                # 1. Mutation → donor vector
                donor = self._mutate(i)

                # 2. Crossover → trial vector
                trial = self._crossover(self.population[i], donor)

                # 3. Boundary handling
                trial = self._handle_boundaries(trial)

                # 4. Greedy selection (≤ to traverse flat landscapes)
                trial_fitness = float(self._evaluate_fitness(trial)[0])
                if trial_fitness <= self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

            # End-of-generation bookkeeping
            self._update_best()

            if self.best_solution is not None:
                self.history.append(self.best_solution.copy())

        return cast(np.ndarray, self.best_solution)
