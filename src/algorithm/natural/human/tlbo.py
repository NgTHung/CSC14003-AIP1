"""
Teaching-Learning-Based Optimization implementation.
"""
from dataclasses import dataclass
import numpy as np
from algorithm.base_model import Model
from problems.continuous.continuous import ContinuousProblem

@dataclass
class TLBOConfig:
    """Configuration parameters for TLBO algorithm."""
    pop_size: int = 50
    iterations: int = 100
    # True if the problem is minimization (default for loss functions), False if maximization
    minimization: bool = True 

class TLBO(Model[ContinuousProblem, np.ndarray, float, TLBOConfig]):
    """
    Teaching-Learning-Based Optimization (TLBO) algorithm.
    """
    name: str = "TLBO"

    def run(self):
        """Execute the optimization process."""
        # 1. Initialize population from Problem (including sampling logic within bounds)
        self.population = self.problem.sample(self.conf.pop_size)
        
        # 2. Evaluate initial fitness
        self.fitness = self.problem.eval(self.population)
        
        # Initialize history and best solution
        self.history = []
        self._update_global_best()
        
        # Get bounds for clipping during movement
        # Note: Accessing protected member _bounds as per ContinuousProblem definition
        lower_bound = self.problem._bounds[:, 0]
        upper_bound = self.problem._bounds[:, 1]
        
        dim = self.problem._n_dim

        for it in range(self.conf.iterations):
            # ==========================
            # TEACHER PHASE
            # ==========================
            # Find Teacher
            teacher_idx = self._get_best_index()
            teacher = self.population[teacher_idx]
            
            # Calculate Mean
            mean_pop = np.mean(self.population, axis=0)
            
            # Create new candidates
            TF = np.random.randint(1, 3) # Teaching Factor: 1 or 2
            r = np.random.rand(self.conf.pop_size, dim)
            
            new_pop_teacher = self.population + r * (teacher - TF * mean_pop)
            new_pop_teacher = np.clip(new_pop_teacher, lower_bound, upper_bound)
            
            # Evaluate and update (Greedy Selection)
            new_fit_teacher = self.problem.eval(new_pop_teacher)
            self._greedy_update(new_pop_teacher, new_fit_teacher)

            # ==========================
            # LEARNER PHASE
            # ==========================
            # Create random permutation to select Partner
            partner_indices = np.arange(self.conf.pop_size)
            np.random.shuffle(partner_indices)
            partners = self.population[partner_indices]
            partners_fitness = self.fitness[partner_indices]
            
            r = np.random.rand(self.conf.pop_size, dim)
            
            # Mask to determine who is better
            if self.conf.minimization:
                better_mask = (self.fitness < partners_fitness).reshape(-1, 1)
            else:
                better_mask = (self.fitness > partners_fitness).reshape(-1, 1)
            
            # Vectorization logic: Move towards the better individual
            # If we are better: step = X - Partner
            # If partner is better: step = Partner - X = -(X - Partner)
            step = self.population - partners
            direction = np.where(better_mask, 1, -1)
            
            new_pop_learner = self.population + r * (step * direction)
            new_pop_learner = np.clip(new_pop_learner, lower_bound, upper_bound)
            
            # Evaluate and update (Greedy Selection)
            new_fit_learner = self.problem.eval(new_pop_learner)
            self._greedy_update(new_pop_learner, new_fit_learner)
            
            # ==========================
            # RECORDING
            # ==========================
            self._update_global_best()
            # Record the best fitness of the current iteration into history
            self.history.append(self.bestFitness)
            
        return self.best_solution

    def _get_best_index(self):
        """Helper to get index of best solution."""
        if self.conf.minimization:
            return np.argmin(self.fitness)
        return np.argmax(self.fitness)

    def _update_global_best(self):
        """Update the global best solution attribute."""
        idx = self._get_best_index()
        current_best_fit = self.fitness[idx]
        
        # If there is no best solution yet or a better one is found
        if not hasattr(self, 'bestFitness') or \
           (self.conf.minimization and current_best_fit < self.bestFitness) or \
           (not self.conf.minimization and current_best_fit > self.bestFitness):
            self.bestFitness = current_best_fit
            self.best_solution = self.population[idx].copy()

    def _greedy_update(self, new_pop, new_fitness):
        """Update population only where new solutions are better."""
        if self.conf.minimization:
            improved = new_fitness < self.fitness
        else:
            improved = new_fitness > self.fitness
            
        self.population[improved] = new_pop[improved]
        self.fitness[improved] = new_fitness[improved]