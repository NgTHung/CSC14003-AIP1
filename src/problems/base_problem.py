import numpy as np
class Problem:
    name: str = "Base Problem"
    def eval(self,values: np.ndarray)->np.float128:
        raise NotImplementedError

class OptimizationProblem(Problem):
    name: str = "Optimzation Problem"
    dim: int
    bound: np.ndarray