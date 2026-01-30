from typing import Generic, TypeVar
import numpy as np
from problems import Problem,OptimizationProblem

T = TypeVar("T",default=np.ndarray)
TR = TypeVar("TR",default=np.ndarray)
Opt = TypeVar("Opt", default=np.ndarray)
Prob = TypeVar("Problem", Problem, OptimizationProblem)

class Model(Generic[T,Opt,Prob,TR]):
    history: list[T]
    best_solution: T
    conf: Opt
    bestFitness: TR
    problem: Prob
    
    name: str = "Generic Model"
    
    def __init__(self, configuration: Opt, problem: Prob):
        self.conf = configuration
        self.problem = problem
    def run():
        raise NotImplementedError
    
class Optimizer(Model):
    name = "Base Optimizer"
    seed: int | str | float | None = None
    def __init__(self, configuration: Opt, problem: Prob, seed: int | str | float | None = None):
        self.seed = seed
        super().__init__(configuration, problem)