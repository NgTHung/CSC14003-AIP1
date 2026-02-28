"""Canonical Genetic Algorithm (GA) for continuous optimization.

Simulates the process of natural evolution. A population of candidate solutions
(chromosomes) evolves over generations through selection, crossover, and
mutation. Solutions are encoded as binary strings (genotype) and decoded into
real-valued vectors (phenotype) for fitness evaluation.

The three core operators are:
1. **Reproduction (Selection)** — favours fitter individuals for the mating
   pool using roulette-wheel or stochastic-remainder selection.
2. **Crossover** — recombines pairs of parent chromosomes (one-site or
   two-site) to produce offspring that inherit traits from both parents.
3. **Mutation** — randomly flips bits with a small probability to maintain
   diversity and prevent premature convergence.

Reference: Holland, J.H. (1975). Adaptation in Natural and Artificial
Systems. University of Michigan Press.
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


class SelectionMethod(Enum):
    """Available parent-selection strategies."""

    ROULETTE_WHEEL = "roulette_wheel"
    STOCHASTIC_REMAINDER = "stochastic_remainder"


class CrossoverMethod(Enum):
    """Available crossover strategies."""

    ONE_SITE = "one_site"
    TWO_SITE = "two_site"


@dataclass
class GAParameter:
    """Configuration parameters for the Genetic Algorithm.

    Attributes
    ----------
    pop_size : int
        Number of individuals in the population. Typical: 20-100.
    n_bits : int
        Number of bits used to encode **each** decision variable.
        Higher values give finer precision within the variable bounds.
        The precision per variable is ``(upper - lower) / (2^n_bits - 1)``.
        Typical: 10-20.
    pc : float
        Crossover probability (0 ≤ pc ≤ 1). Approximately ``100 * pc %``
        of the population participates in crossover each generation.
        Typical: 0.6-0.9.
    pm : float
        Mutation probability per bit (0 ≤ pm ≤ 1). Should be small to
        act as a background operator. Typical: 0.001-0.05.
    cycle : int
        Number of generations to run.
    selection : SelectionMethod
        Parent selection strategy. Default: stochastic remainder.
    crossover : CrossoverMethod
        Crossover strategy. Default: two-site crossover.
    """

    pop_size: int
    n_bits: int
    pc: float
    pm: float
    cycle: int
    selection: SelectionMethod = SelectionMethod.STOCHASTIC_REMAINDER
    crossover: CrossoverMethod = CrossoverMethod.TWO_SITE


# ======================================================================
# Genetic Algorithm
# ======================================================================


class GeneticAlgorithm(
    Model[ContinuousProblem, np.ndarray | None, float, GAParameter]
):
    """Canonical Genetic Algorithm for continuous optimization.

    Algorithm outline per generation:
    1. **Evaluate** — Decode the binary population to real-valued phenotypes
       and compute fitness for every individual.
    2. **Selection** — Build a mating pool by selecting individuals with
       probability proportional to fitness (roulette-wheel) or via
       stochastic remainder selection.
    3. **Crossover** — Pair individuals and recombine their binary
       chromosomes with probability *pc* (one-site or two-site).
    4. **Mutation** — Independently flip each bit with probability *pm*.
    5. **Elitism** — The best individual found so far is preserved.
    """

    population: np.ndarray      # shape (pop_size, total_bits), dtype int8
    fitness: np.ndarray         # shape (pop_size,)
    n_dim: int                  # number of decision variables
    total_bits: int             # n_bits * n_dim

    def __init__(
        self, configuration: GAParameter, problem: ContinuousProblem
    ):
        """Initialize the Genetic Algorithm.

        Parameters
        ----------
        configuration : GAParameter
            Algorithm hyperparameters.
        problem : ContinuousProblem
            Continuous optimization problem to solve.
        """
        super().__init__(configuration, problem)
        self.name = "Genetic Algorithm"
        self.n_dim = problem.n_dim
        self.total_bits = configuration.n_bits * self.n_dim

        # Random binary population
        self.population = np.random.randint(
            0, 2, size=(configuration.pop_size, self.total_bits), dtype=np.int8
        )

        # Decode, evaluate, and track the best
        phenotypes = self._decode_population(self.population)
        self.fitness = self._evaluate_fitness(phenotypes)

        best_idx = int(np.argmin(self.fitness))
        self.best_solution = phenotypes[best_idx].copy()
        self.best_fitness = float(self.fitness[best_idx])
        self.history = []

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def _decode_variable(self, bits: np.ndarray, lower: float, upper: float) -> float:
        """Decode a binary sub-string into a real value.

        Uses the linear mapping rule from the canonical GA:

        .. math::
            x_i = x^l_i + \\frac{x^u_i - x^l_i}{2^\\beta - 1}
                  \\sum_{j=0}^{\\beta-1} \\gamma_j \\, 2^j

        Parameters
        ----------
        bits : np.ndarray
            Binary array of shape ``(n_bits,)`` with values in {0, 1}.
        lower : float
            Lower bound of the variable.
        upper : float
            Upper bound of the variable.

        Returns
        -------
        float
            Decoded real value within ``[lower, upper]``.
        """
        # Convert binary string to its decimal value
        # bits[0] is MSB, bits[-1] is LSB
        powers = 2 ** np.arange(len(bits) - 1, -1, -1)
        decimal_value = int(np.dot(bits, powers))
        max_value = 2 ** len(bits) - 1

        return lower + (upper - lower) / max_value * decimal_value

    def _decode_individual(self, chromosome: np.ndarray) -> np.ndarray:
        """Decode a full chromosome into a real-valued solution vector.

        The chromosome is split into ``n_dim`` sub-strings of ``n_bits``
        each.  Each sub-string is decoded independently using the bounds
        for its corresponding decision variable.

        Parameters
        ----------
        chromosome : np.ndarray
            Binary array of shape ``(total_bits,)``.

        Returns
        -------
        np.ndarray
            Real-valued solution vector of shape ``(n_dim,)``.
        """
        phenotype = np.empty(self.n_dim)
        n_bits = self.conf.n_bits

        for i in range(self.n_dim):
            bits = chromosome[i * n_bits : (i + 1) * n_bits]
            lower = self.problem.bounds[i, 0]
            upper = self.problem.bounds[i, 1]
            phenotype[i] = self._decode_variable(bits, lower, upper)

        return phenotype

    def _decode_population(self, population: np.ndarray) -> np.ndarray:
        """Decode the entire binary population into real-valued phenotypes.

        Parameters
        ----------
        population : np.ndarray
            Binary population of shape ``(pop_size, total_bits)``.

        Returns
        -------
        np.ndarray
            Phenotype matrix of shape ``(pop_size, n_dim)``.
        """
        return np.array([self._decode_individual(chrom) for chrom in population])

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------

    def _evaluate_fitness(self, phenotypes: np.ndarray) -> np.ndarray:
        """Evaluate the objective function for decoded phenotypes.

        Parameters
        ----------
        phenotypes : np.ndarray
            Real-valued solutions of shape ``(pop_size, n_dim)``.

        Returns
        -------
        np.ndarray
            Raw objective values of shape ``(pop_size,)``.
        """
        return cast(np.ndarray, self.problem.eval(phenotypes))

    @staticmethod
    def _fitness_to_selection_prob(objective_values: np.ndarray) -> np.ndarray:
        """Convert objective values (minimisation) to selection probabilities.

        Uses the transformation ``F(x) = 1 / (1 + f(x))`` so that lower
        objective values receive higher selection probability, then
        normalises to a proper probability distribution.

        Parameters
        ----------
        objective_values : np.ndarray
            Raw objective/cost values, shape ``(n,)``.

        Returns
        -------
        np.ndarray
            Selection probabilities summing to 1, shape ``(n,)``.
        """
        transformed = np.where(
            objective_values >= 0,
            1.0 / (1.0 + objective_values),
            1.0 + np.abs(objective_values),
        )
        total = np.sum(transformed)
        if total == 0:
            return np.ones(len(objective_values)) / len(objective_values)
        return transformed / total

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _roulette_wheel_selection(self) -> np.ndarray:
        """Select individuals using roulette-wheel (fitness-proportionate).

        Each individual is selected with probability proportional to its
        transformed fitness.  The wheel is "spun" ``pop_size`` times.

        Returns
        -------
        np.ndarray
            Indices of selected individuals, shape ``(pop_size,)``.
        """
        probs = self._fitness_to_selection_prob(self.fitness)
        return np.random.choice(
            self.conf.pop_size, size=self.conf.pop_size, p=probs
        )

    def _stochastic_remainder_selection(self) -> np.ndarray:
        """Select individuals using stochastic remainder selection.

        1. Compute expected copies ``e_i = p_i * pop_size``.
        2. Each individual deterministically receives ``floor(e_i)`` copies.
        3. Remaining slots are filled by Bernoulli trials on the fractional
           parts.

        Returns
        -------
        np.ndarray
            Indices of selected individuals, shape ``(pop_size,)``.
        """
        probs = self._fitness_to_selection_prob(self.fitness)
        expected = probs * self.conf.pop_size

        selected: list[int] = []

        # Integer parts → guaranteed copies
        for i, e in enumerate(expected):
            n_copies = int(np.floor(e))
            selected.extend([i] * n_copies)

        # Fractional parts → Bernoulli trials to fill remaining slots
        remaining = self.conf.pop_size - len(selected)
        if remaining > 0:
            fractions = expected - np.floor(expected)
            total_frac = np.sum(fractions)
            if total_frac > 0:
                frac_probs = fractions / total_frac
            else:
                frac_probs = np.ones(self.conf.pop_size) / self.conf.pop_size

            extras = np.random.choice(
                self.conf.pop_size, size=remaining, p=frac_probs
            )
            selected.extend(extras.tolist())

        # If we somehow over-filled, truncate
        selected = selected[: self.conf.pop_size]

        return np.array(selected)

    def _select(self) -> np.ndarray:
        """Build a mating pool using the configured selection method.

        Returns
        -------
        np.ndarray
            Selected population of shape ``(pop_size, total_bits)``.
        """
        if self.conf.selection == SelectionMethod.ROULETTE_WHEEL:
            indices = self._roulette_wheel_selection()
        else:
            indices = self._stochastic_remainder_selection()

        return self.population[indices].copy()

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------

    def _one_site_crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform one-site (single-point) crossover.

        A random crossover site is chosen; bits to the right of the site
        are swapped between the two parents.

        Parameters
        ----------
        parent1 : np.ndarray
            First parent chromosome, shape ``(total_bits,)``.
        parent2 : np.ndarray
            Second parent chromosome, shape ``(total_bits,)``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Two offspring chromosomes.
        """
        site = np.random.randint(1, self.total_bits)
        child1 = np.concatenate([parent1[:site], parent2[site:]])
        child2 = np.concatenate([parent2[:site], parent1[site:]])
        return child1, child2

    def _two_site_crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform two-site (two-point) crossover.

        Two crossover sites are chosen; bits **between** the two sites
        are exchanged between the parents.

        Parameters
        ----------
        parent1 : np.ndarray
            First parent chromosome, shape ``(total_bits,)``.
        parent2 : np.ndarray
            Second parent chromosome, shape ``(total_bits,)``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Two offspring chromosomes.
        """
        sites = sorted(np.random.choice(range(1, self.total_bits), size=2, replace=False))
        s1, s2 = sites[0], sites[1]

        child1 = parent1.copy()
        child2 = parent2.copy()
        child1[s1:s2] = parent2[s1:s2]
        child2[s1:s2] = parent1[s1:s2]
        return child1, child2

    def _crossover(self, mating_pool: np.ndarray) -> np.ndarray:
        """Apply crossover to the mating pool.

        Pairs are formed sequentially.  Each pair undergoes crossover with
        probability ``pc``; otherwise the parents pass through unchanged.

        Parameters
        ----------
        mating_pool : np.ndarray
            Selected population of shape ``(pop_size, total_bits)``.

        Returns
        -------
        np.ndarray
            Offspring population of shape ``(pop_size, total_bits)``.
        """
        offspring = mating_pool.copy()

        # Shuffle to randomise pairing
        indices = np.random.permutation(self.conf.pop_size)
        offspring = offspring[indices]

        crossover_fn = (
            self._one_site_crossover
            if self.conf.crossover == CrossoverMethod.ONE_SITE
            else self._two_site_crossover
        )

        for i in range(0, self.conf.pop_size - 1, 2):
            if np.random.random() < self.conf.pc:
                offspring[i], offspring[i + 1] = crossover_fn(
                    offspring[i], offspring[i + 1]
                )

        return offspring

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _mutate(self, population: np.ndarray) -> np.ndarray:
        """Apply bit-flip mutation to the population.

        Each bit in every chromosome is independently flipped (0↔1) with
        probability ``pm``.

        Parameters
        ----------
        population : np.ndarray
            Binary population of shape ``(pop_size, total_bits)``.

        Returns
        -------
        np.ndarray
            Mutated population.
        """
        mutation_mask = np.random.random(population.shape) < self.conf.pm
        population[mutation_mask] = 1 - population[mutation_mask]
        return population

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_best(self, phenotypes: np.ndarray):
        """Update the global best solution from the current generation.

        Parameters
        ----------
        phenotypes : np.ndarray
            Decoded real-valued population, shape ``(pop_size, n_dim)``.
        """
        best_idx = int(np.argmin(self.fitness))
        if self.fitness[best_idx] < self.best_fitness:
            self.best_fitness = float(self.fitness[best_idx])
            self.best_solution = phenotypes[best_idx].copy()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    @override
    def run(self) -> np.ndarray:
        """Execute the Genetic Algorithm.

        Returns
        -------
        np.ndarray
            Best solution (phenotype) found after all generations.
        """
        for _ in range(self.conf.cycle):
            # 1. Selection — build mating pool
            mating_pool = self._select()

            # 2. Crossover — recombine parents
            offspring = self._crossover(mating_pool)

            # 3. Mutation — bit-flip perturbation
            offspring = self._mutate(offspring)

            # 4. Replace population with offspring
            self.population = offspring

            # 5. Evaluate new generation
            phenotypes = self._decode_population(self.population)
            self.fitness = self._evaluate_fitness(phenotypes)

            # 6. Track best
            self._update_best(phenotypes)

            if self.best_solution is not None:
                self.history.append(self.best_solution.copy())

        return cast(np.ndarray, self.best_solution)
