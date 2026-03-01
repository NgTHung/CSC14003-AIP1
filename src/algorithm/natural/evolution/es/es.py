"""Evolution Strategies — unified entry point.

This module provides a single ``EvolutionStrategy`` class that acts as a
façade over all four ES variants.  Callers choose a variant through the
``ESVariant`` enum and supply the appropriate parameter dataclass; the
façade delegates everything to the concrete implementation.

Usage
-----
>>> from algorithm.natural.evolution.es.es import EvolutionStrategy, ESVariant
>>> from algorithm.natural.evolution.es import CMAESParameter
>>> from problems import Sphere
>>>
>>> problem = Sphere(n_dim=10)
>>> es = EvolutionStrategy(
...     variant=ESVariant.CMA_ES,
...     configuration=CMAESParameter(sigma=1.0, cycle=500),
...     problem=problem,
... )
>>> best = es.run()
"""

from enum import Enum
from typing import cast, override

import numpy as np
from problems import ContinuousProblem
from algorithm import Model

from .one_plus_one_es import OnePlusOneES, OnePlusOneESParameter
from .self_adaptive_es import SelfAdaptiveES, SelfAdaptiveESParameter
from .cma_es import CMAES, CMAESParameter
from .mu_rho_plus_lambda_es import MuRhoPlusLambdaES, MuRhoPlusLambdaESParameter


class ESVariant(Enum):
    """Supported Evolution Strategy variants."""

    ONE_PLUS_ONE = "(1+1)-ES"
    SELF_ADAPTIVE = "Self-Adaptive (μ,λ)-ES"
    CMA_ES = "CMA-ES"
    MU_RHO_PLUS_LAMBDA = "(μ/ρ+λ)-ES"


# Map each variant to its expected parameter type for validation.
_VARIANT_PARAM_MAP: dict[ESVariant, type] = {
    ESVariant.ONE_PLUS_ONE: OnePlusOneESParameter,
    ESVariant.SELF_ADAPTIVE: SelfAdaptiveESParameter,
    ESVariant.CMA_ES: CMAESParameter,
    ESVariant.MU_RHO_PLUS_LAMBDA: MuRhoPlusLambdaESParameter,
}

# Type alias for all valid ES parameter types.
ESParameter = (
    OnePlusOneESParameter
    | SelfAdaptiveESParameter
    | CMAESParameter
    | MuRhoPlusLambdaESParameter
)


class EvolutionStrategy(
    Model[ContinuousProblem, np.ndarray | None, float, ESParameter]
):
    """Façade that delegates to the chosen ES variant.

    Parameters
    ----------
    variant : ESVariant
        Which ES algorithm to use.
    configuration : ESParameter
        Parameter dataclass matching the chosen variant.
    problem : ContinuousProblem
        The optimisation problem instance.

    Raises
    ------
    TypeError
        If *configuration* does not match the expected type for *variant*.
    """

    _inner: (
        OnePlusOneES | SelfAdaptiveES | CMAES | MuRhoPlusLambdaES
    )

    def __init__(
        self,
        variant: ESVariant,
        configuration: ESParameter,
        problem: ContinuousProblem,
    ):
        expected = _VARIANT_PARAM_MAP[variant]
        if not isinstance(configuration, expected):
            raise TypeError(
                f"Variant {variant.value} expects {expected.__name__}, "
                f"got {type(configuration).__name__}"
            )

        super().__init__(configuration, problem)
        self.name = f"ES ({variant.value})"

        # Instantiate the concrete algorithm.
        match variant:
            case ESVariant.ONE_PLUS_ONE:
                self._inner = OnePlusOneES(
                    cast(OnePlusOneESParameter, configuration), problem
                )
            case ESVariant.SELF_ADAPTIVE:
                self._inner = SelfAdaptiveES(
                    cast(SelfAdaptiveESParameter, configuration), problem
                )
            case ESVariant.CMA_ES:
                self._inner = CMAES(
                    cast(CMAESParameter, configuration), problem
                )
            case ESVariant.MU_RHO_PLUS_LAMBDA:
                self._inner = MuRhoPlusLambdaES(
                    cast(MuRhoPlusLambdaESParameter, configuration), problem
                )

    @override
    def run(self) -> np.ndarray:
        """Run the chosen ES variant and sync results back to the façade."""
        result = self._inner.run()

        # Mirror inner state to the façade so callers see consistent data.
        self.best_solution = self._inner.best_solution
        self.best_fitness = self._inner.best_fitness
        self.history = self._inner.history

        return result
