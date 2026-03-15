# Team Project 01 - Search & Nature-Inspired Algorithms

> CSC14003 - Artificial Intelligence Principles

## 1. Project Title & Overview

This repository contains a modular, object-oriented optimization framework developed for a university team project.
The core objective is to implement a taxonomy of search and nature-inspired algorithms from scratch using only Python and NumPy for algorithmic computation.

The framework is designed for two domains:

- Continuous optimization (benchmark functions such as Sphere, Rastrigin, Ackley, Rosenbrock, Griewank)
- Discrete optimization and state-space search (TSP, Knapsack, Graph Coloring)

## 2. Features & Supported Algorithms

### Core Features

- NumPy-first implementation with no SciPy/scikit-learn dependency in algorithm logic
- Unified abstractions (`Problem`, `ContinuousProblem`, `DiscreteProblem`, `Algorithm`)
- Dual-domain support: continuous vector optimization and discrete combinatorial/search problems
- Configurable algorithms via typed dataclasses (for reproducible experiments)
- Built-in history tracking (`history`, `best_solution`, `best_fitness`)
- Ready-to-run demos and comparison scripts in `tests/` and `src/comparision/`

### Implemented Algorithm Families

- Classical graph search:
    - BFS (Breadth-First Search)
    - DFS (Depth-First Search)
    - UCS (Uniform-Cost Search)
    - Greedy Best-First Search
    - A* Search
- Local search:
    - Hill Climbing
- Nature-inspired (Biology):
    - GA (Genetic Algorithm)
    - PSO (Particle Swarm Optimization)
    - ABC (Artificial Bee Colony)
    - FA (Firefly Algorithm)
    - CS (Cuckoo Search)
    - ACO variants: Ant System, ACS, MMAS
- Nature-inspired (Evolution):
    - Differential Evolution (DE)
    - Evolution Strategies (ES variants including 1+1 ES, Self-Adaptive ES, CMA-ES, mu/rho + lambda ES)
- Nature-inspired (Physics/Harmony/Annealing style):
    - Simulated Annealing (SA)
    - Harmony Search (HS)
    - Gravitational Search Algorithm (GSA)
- Human-inspired:
    - TLBO
    - SFO
    - CA

### Implemented Problem Domains

- Continuous:
    - Sphere
    - Rastrigin
    - Ackley
    - Rosenbrock
    - Griewank
- Discrete:
    - TSP
    - 0/1 Knapsack
    - Graph Coloring

## 3. Project Structure

```text
CSC14003-AIP1/
|- README.md
|- pyproject.toml
|- docs/                          # Report, equations, and visualization notebooks
|- src/
|  |- AIP/
|  |  |- __init__.py
|  |  |- main.py
|  |  |- benchmark.py
|  |  |- algorithm/
|  |  |  |- base_algorithm.py     # Generic Algorithm base class
|  |  |  |- classical/            # BFS, DFS, UCS, Greedy, A*
|  |  |  |- local/                # Hill Climbing
|  |  |  |- natural/
|  |  |  |  |- biology/           # GA, PSO, ABC, FA, CS, ACO variants
|  |  |  |  |- evolution/         # DE, ES variants, CMA-ES
|  |  |  |  |- human/             # TLBO, SFO, CA
|  |  |  |  |- physic/            # SA, HS, GSA
|  |  |- problems/
|  |  |  |- base_problem.py       # Problem + DiscreteProblem abstractions
|  |  |  |- continuous/           # Sphere, Rastrigin, Ackley, Rosenbrock, Griewank
|  |  |  |- discrete/             # TSP, Knapsack, Graph Coloring
|  |- comparision/                # Benchmarking/comparison scripts and figures
|- tests/                         # Demo scripts by algorithm/problem family
```

## 4. Installation & Setup

### Prerequisites

- Python 3.11+
- pip (or uv)

### Method A: Clone + Local Install

```bash
git clone https://github.com/NgTHung/CSC14003-AIP1.git
cd CSC14003-AIP1

# Create virtual environment
python -m venv .venv

# Activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate

# Install package and dependencies from pyproject.toml
pip install -e .
```

### Method B: Direct Install from GitHub (No Clone)

```bash
python -m venv .venv

# Activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate

# Install directly from GitHub
pip install "git+https://github.com/NgTHung/CSC14003-AIP1.git"
```

## 5. Usage Examples

### Example 1: Continuous Optimization (Sphere + Firefly Algorithm)

This example shows how to configure an algorithm using a dataclass (`FireflyParameter`) and enable tracking via `stat=True`.

```python
import numpy as np

from AIP.problems.continuous.sphere import Sphere
from AIP.algorithm.natural.biology.fa import FireflyAlgorithm, FireflyParameter

# Reproducibility
np.random.seed(42)

# 30-dimensional Sphere minimization
problem = Sphere(n_dim=30)

# Dataclass-based hyperparameter configuration
params = FireflyParameter(
        n_fireflies=40,
        alpha=0.35,
        beta0=1.0,
        gamma=1.0,
        alpha_decay=0.98,
        cycle=300,
)

# Enable detailed state tracking for analysis/visualization
optimizer = FireflyAlgorithm(configuration=params, problem=problem, stat=True)
best_solution = optimizer.run()

print("Best fitness:", optimizer.best_fitness)
print("Best solution (first 5 dims):", best_solution[:5])
print("History length:", len(optimizer.history))
print("Tracked populations:", len(optimizer.firefly_pos_history))
```

### Example 2: Discrete Optimization (TSP + Cuckoo Search)

The same framework supports discrete combinatorial problems. Here we solve a TSP instance with dataclass parameters and `stat=True`.

```python
import numpy as np

from AIP.problems.discrete.tsp import TSP
from AIP.algorithm.natural.biology.cs import CuckooSearch, CuckooSearchParameter

np.random.seed(42)

# Built-in benchmark instance (8 cities)
problem = TSP.create_medium()

params = CuckooSearchParameter(
        n_nests=30,
        pa=0.25,
        alpha=0.01,
        beta=1.5,
        iteration=500,
)

optimizer = CuckooSearch(configuration=params, problem=problem, stat=True)
best_perm = optimizer.run()

decoded = problem.decode_permutation(best_perm)
print("Best tour:", decoded["tour_names"])
print("Total distance:", decoded["total_distance"])
print("Best fitness:", optimizer.best_fitness)
print("History length:", len(optimizer.history))
print("Tracked populations:", len(optimizer.nests_history))
```

## Running Demos

After installation, you can run ready-made demos in `tests/`, for example:

```bash
python tests/demo_tsp.py
python tests/demo_physics_algorithms.py
python tests/demo_knapsack.py
python tests/demo_evolution.py
```

## License

This project is distributed under the terms of the license in `LICENSE`.