import numpy as np
from src.algorithm import AntSystem, AntSystemParameter
from src.problems import TSP

ASP = AntSystemParameter(0.15, 5, 0.4, 1.0, 2.5, 200)
tsp = TSP(
    np.array(
        [
            [1, 2, 5, 3, 2],
            [2, 1, 4, 2, 1],
            [3, 5, 3, 1, 5],
            [2, 4, 5, 1, 4],
            [2, 5, 2, 6, 8],
        ]
    )
)
AS = AntSystem(ASP, tsp)

print(AS.run())
