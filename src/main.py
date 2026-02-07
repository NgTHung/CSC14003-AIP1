import numpy as np
import matplotlib.pyplot as plt
import argparse
import random

from problems import Sphere, Rastrigin, Ackley

from algorithm.natural.human.tlbo import TLBO, TLBOConfig

s = Sphere()
print(s.eval(s.sample(5)))
r = Rastrigin()
print(r.eval(r.sample(5)))
a = Ackley()
print(a.eval(a.sample(5)))

def set_seed(seed):
    """Set random seed for reproducibility"""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        print(f"Random seed set to: {seed}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run optimization algorithms on benchmark problems')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup problem
    # Sphere problem in 10 dimensions
    # problem = Sphere(n_dim=10)

    # Rastrigin problem in 10 dimensions
    # problem = Rastrigin(n_dim=10)

    problem = Ackley(n_dim=10)

    print(f"Problem: {problem._name}, Dimensions: {problem._n_dim}")
    print(f"Bounds: {problem._bounds[0]}")

    # Setup TLBO algorithm with configuration
    config = TLBOConfig(
        pop_size=100,
        iterations= 5000,
        minimization=True
    )

    # Initialize Algorithm
    optimizer = TLBO(configuration=config, problem=problem)

    # Run Optimization
    print(f"\nRunning {optimizer.name}...")
    best_solution = optimizer.run()

    # Report Results
    print("\n=== RESULTS ===")
    print(f"Best Fitness: {optimizer.bestFitness:.10f}")
    print(f"Best Solution (First 5 coords): {best_solution[:5]}")
    
    # Visualization (Simple Convergence Plot)
    plt.figure(figsize=(10, 6))
    plt.plot(optimizer.history, label='TLBO', linewidth=2)
    plt.title(f'Convergence on {problem._name} Function')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.yscale('log') # Log scale to show convergence better
    plt.grid(True, alpha=0.3, which='both', linestyle='--')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

