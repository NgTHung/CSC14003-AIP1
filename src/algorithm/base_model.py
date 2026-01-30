from typing import Generic, TypeVar
import numpy as np
from problems import Problem,OptimizationProblem

T = TypeVar("T",default=np.ndarray)
TR = TypeVar("TR",default=np.ndarray)
Opt = TypeVar("Opt", default=dict[str, np.float128])
Prob = TypeVar("Prob", Problem, OptimizationProblem)

class Model(Generic[Prob,T,TR,Opt]):
    history: list[T]
    best_solution: T
    conf: Opt
    bestFitness: TR
    problem: Prob
    
    name: str = "Generic Model"
    
    def __init__(self, configuration: Opt, problem: Prob):
        self.conf = configuration
        self.problem = problem
    def run(self):
        raise NotImplementedError
    
class Optimizer(Model[OptimizationProblem, T, TR,Opt]):
    name = "Base Optimizer"
    seed: int | str | float | None = None
    def __init__(self,problem: OptimizationProblem, configuration: Opt, seed: int | str | float | None = None):
        self.seed = seed
        super().__init__(configuration, problem)

