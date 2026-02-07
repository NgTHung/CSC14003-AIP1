# CSC14003-AIP1 — Metaheuristic Optimization Framework

> **CSC14003 — Artificial Intelligence Principles** | Team Project 1

An extensible Python framework for implementing and evaluating metaheuristic optimization algorithms on continuous benchmark problems.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithms](#algorithms)
- [Benchmark Functions](#benchmark-functions)
- [Architecture](#architecture)

---

## Overview

This project provides an extensible framework to:

- Implement metaheuristic optimization algorithms (nature-inspired, classical, …)
- Evaluate performance on standard benchmark functions (Sphere, Rastrigin, Ackley, …)
- Visualize the algorithm’s convergence process

## Project Structure

```
CSC14003-AIP1/
├── pyproject.toml              # Project configuration & dependencies (uv/pip)
├── README.md
├── src/
│   ├── main.py                 # Entry point — run algorithms & plot convergence curves
│   ├── benchmark.py            # Benchmark runner (under development)
│   ├── algorithm/
│   │   ├── base_model.py       # Abstract Model[Prob, T, Tr, Opt] class
│   │   ├── classical/          # Classical algorithms (under development)
│   │   └── natural/            # Nature-inspired algorithms
│   │       ├── biology/        #   Biology-inspired (under development)
│   │       ├── evolution/      #   Evolutionary (under development)
│   │       ├── human/
│   │       │   └── tlbo.py     #   Teaching-Learning-Based Optimization (TLBO)
│   │       └── physic/         #   Physics-inspired (under development)
│   ├── problems/
│   │   ├── base_problem.py     # Abstract Problem base class (ABC)
│   │   ├── continuous/
│   │   │   ├── continuous.py   # Abstract ContinuousProblem class
│   │   │   ├── sphere.py       # Sphere function
│   │   │   ├── rastrigin.py    # Rastrigin function
│   │   │   └── ackley.py       # Ackley function
│   │   └── discrete/           # Discrete problems (under development)
│   └── utils/                  # Shared utilities
└── tests/                      # Unit tests
```

## System Requirements

- **Python** ≥ 3.14
- **Package manager**: [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/NgTHung/CSC14003-AIP1.git
cd CSC14003-AIP1

# Install dependencies
uv sync
```

### Using pip

```bash
git clone https://github.com/NgTHung/CSC14003-AIP1.git
cd CSC14003-AIP1

pip install -e .
```

## Usage

### Quick Run

```bash
# Using uv
uv run src/main.py

# Or run Python directly (after pip install)
python src/main.py
```

### Custom Parameters

```bash
# Set random seed
uv run src/main.py --seed 123
```

### Programmatic Usage

```python
import numpy as np
from problems import Ackley
from algorithm.natural.human.tlbo import TLBO, TLBOConfig

# Initialize the problem — 10D Ackley
problem = Ackley(n_dim=10)

# Configure the algorithm
config = TLBOConfig(
    pop_size=100,       # Population size
    iterations=5000,    # Number of iterations
    minimization=True   # Minimize
)

# Run optimization
optimizer = TLBO(configuration=config, problem=problem)
best_solution = optimizer.run()

print(f"Best Fitness: {optimizer.bestFitness:.10f}")
print(f"Best Solution: {best_solution[:5]}")  # First 5 dimensions
```

## Algorithms

### Implemented

| Algorithm | Type | Module |
|---|---|---|
| **TLBO** (Teaching-Learning-Based Optimization) | Nature-inspired / Human | `algorithm.natural.human.tlbo` |

### Planned Categories (to be extended)

```
algorithm/
├── classical/          # Hill Climbing, Simulated Annealing, ...
└── natural/
    ├── biology/        # Genetic Algorithm, PSO, ...
    ├── evolution/      # Differential Evolution, ...
    ├── human/          # TLBO ✅
    └── physic/         # Gravitational Search, ...
```

### TLBO — Details

The TLBO algorithm models the teaching–learning process in a classroom and consists of two phases:

1. **Teacher Phase** — Learners learn from the teacher (the best solution), moving toward the teacher with a Teaching Factor $TF \in \{1, 2\}$.
2. **Learner Phase** — Learners interact with each other and move toward the better solution between two individuals.

**Configuration parameters (`TLBOConfig`):**

| Parameter | Default | Description |
|---|---|---|
| `pop_size` | 50 | Population size |
| `iterations` | 100 | Maximum number of iterations |
| `minimization` | `True` | `True` = minimize, `False` = maximize |

## Benchmark Functions

| Function | Formula | Search Space | Global Minimum |
|---|---|---|---|
| **Sphere** | $f(\mathbf{x}) = \sum_{i=1}^{n} x_i^2$ | $[-5.12,\ 5.12]^n$ | $f(\mathbf{0}) = 0$ |
| **Rastrigin** | $f(\mathbf{x}) = An + \sum_{i=1}^{n} [x_i^2 - A\cos(2\pi x_i)]$ | $[-5.12,\ 5.12]^n$ | $f(\mathbf{0}) = 0$ |
| **Ackley** | $f(\mathbf{x}) = -a \cdot e^{-b\sqrt{\frac{1}{n}\sum x_i^2}} - e^{\frac{1}{n}\sum \cos(c \cdot x_i)} + a + e$ | $[-32.768,\ 32.768]^n$ | $f(\mathbf{0}) = 0$ |

## Architecture

### Class Diagram

```
Problem (ABC)
└── ContinuousProblem
    ├── Sphere
    ├── Rastrigin
    └── Ackley

Model[Prob, T, Tr, Opt]
└── TLBO
```

### Adding a New Algorithm

1. Create a file in the appropriate directory (e.g., `algorithm/natural/biology/pso.py`)
2. Inherit from `Model[ContinuousProblem, np.ndarray, float, YourConfig]`
3. Implement `run()` → return the best solution
4. Update `history` and `bestFitness` during execution

### Adding a New Benchmark Function

1. Create a file under `problems/continuous/` (or `problems/discrete/`)
2. Inherit from `ContinuousProblem` and implement `eval()`
3. Export it in `__init__.py`

---

## License

See the [LICENSE](LICENSE) file for details.