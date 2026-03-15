# Evolution Strategies

## I. Introduction to ES

### 1. Overview and Core Principles

Evolution Strategies (ES) are stochastic optimization algorithms inspired by biological evolution, developed in the 1960s to address black-box optimization problems in continuous domains,. Unlike Genetic Algorithms, which often use binary representation, ES operates directly on real-valued vectors. The main design principles of ES are unbiasedness and the adaptive control of strategy parameters (such as mutation step-size).

### 2. Notation and Standard Workflow

The structure of ES is described using the notation $(\mu/\rho +, \lambda)$-ES,:

- **$\mu$:** The number of parents in the population.
- **$\lambda$:** The number of offspring generated in each generation.
- **$\rho$:** The number of parents involved in creating one offspring (recombination).
- **$+$ or $,$ :** The selection mechanism. In plus selection $(\mu + \lambda)$, parents and offspring compete for survival (elitism). In comma selection $(\mu, \lambda)$, parents are discarded, and selection is made only from offspring (requires $\lambda > \mu$),.

The basic loop proceeds as follows: 

Parent Population $\rightarrow$ Recombination $\rightarrow$ Mutation $\rightarrow$ Evaluation $\rightarrow$ Selection $\rightarrow$ New Population,.

### 3. Key Operators

- **Recombination:** ES typically creates one offspring from $\rho$ parents. "Intermediate recombination" (averaging parent vectors) is particularly important due to the **"Genetic Repair"** effect. Averaging cancels out harmful noise components while preserving the beneficial direction, allowing the algorithm to use larger mutation steps,,.
- **Mutation:** This is the primary source of variation, usually implemented by adding a normally distributed (Gaussian) random vector to the individual: $y_{new} = y + \sigma \cdot N(0, C)$. The normal distribution is chosen because it ensures maximum entropy and rotational symmetry.
- **Selection:** ES employs deterministic truncation selection. Only the $\mu$ best individuals are retained based on objective function ranking, without using probabilistic methods like roulette wheel selection,.

### 4. Programming Logic of Popular Algorithms

Below is the programming logic (pseudo-code) for the three most important variants described in the documents.

### A. The (1+1)-ES with 1/5th Rule

This is the simplest algorithm, using one parent and one offspring. The step-size $(\sigma)$ is adjusted based on the success rate,.

1. **Initialize:** Choose $x$ (solution), $\sigma$ (step-size).
2. **Loop:**
    - Mutate:  $x_{offspring} = x + \sigma \cdot N(0, I)$.
    - Evaluate: If $f(x_{offspring})$ is better than $f(x)$:
        - Replace parent: $x = x_{offspring}$.
        - Record success.
    - **Adjust $\sigma$ (1/5th Rule):**
        - If success rate > 1/5 (too easy): Increase $\sigma$ (divide by 0.85 or multiply by constant       $a > 1$).
        - If success rate < 1/5 (too hard): Decrease $\sigma$ (multiply by 0.85 or divide by $a$).
3. **Terminate** when stopping criteria are met.

### B. Self-Adaptive $(\mu, \lambda)$-ES

This algorithm encodes the step-size $\sigma$ into the individual's genome and lets it evolve. This is the standard approach for problems with unknown scales,,.

1. **Initialize:** Population $P$ of $\mu$ individuals, each containing $(x, \sigma)$.
2. **Loop:**
    - **Recombination:** Generate $\lambda$ offspring. For each, select $\rho$ parents and average their $x$ and $\sigma$.
    - **Mutation:** For each offspring $k$:
        - Mutate $\sigma$ first (log-normal): $\sigma_k = \sigma_{recomb} \cdot \exp(\tau \cdot N(0,1))$.
        - Use new $\sigma_k$ to mutate $x$: $x_k = x_{recomb} + \sigma_k \cdot N(0, I)$.
    - **Evaluation:** Calculate $f(x_k)$ for all $\lambda$ offspring.
    - **Selection:** Select the best $\mu$ individuals from the $\lambda$ offspring (discarding old parents) to form the new population $P$.
3. **Terminate** when stopping criteria are met.

### C. CMA-ES

This is the most powerful modern variant, using "evolution paths" to update the covariance matrix $C$, allowing it to solve problems where variables are interdependent (non-separable),,.

1. **Initialize:** Mean $x$, step-size $\sigma$, matrix $C=I$, paths $p_\sigma = 0, p_c = 0$.
2. **Loop:**
    - **Sampling:** Generate $\lambda$ offspring: $x_k = x + \sigma \cdot C^{1/2} \cdot z_k$ (with $z_k \sim N(0,I)$).
    - **Selection & Recombination:** Update mean $x_{new}$ by taking the weighted average of the best $\mu$ offspring: $x_{new} = \sum w_i x_{i:\lambda}$.
    - **Update Evolution Paths ($p_\sigma$ and $p_c$):** Accumulate information about the population's movement direction over generations (using smoothing constants).
    - **Update Step-size $\sigma$:** Based on the length of path $p_\sigma$. If the path is longer than expected (steps in same direction), increase $\sigma$. If shorter (steps cancel out), decrease $\sigma$.
    - **Update Matrix $C$:** Based on path $p_c$ and the variance of successful mutation steps (rank-$\mu$ update).
3. **Terminate** when stopping criteria are met.

### D. $(\mu/\rho + \lambda)$-ES (Plus Strategy)

This is an **elitist** strategy. Unlike the "comma" strategy $(\mu, \lambda)$ where parents are discarded immediately after reproduction, the "plus" strategy keeps parents to compete directly with their offspring. This ensures that the best individual found so far always survives, but it may cause the algorithm to adapt more slowly to environmental changes (such as misadjusted step-sizes $\sigma$),. This strategy is often preferred in discrete or combinatorial search spaces,.

1. **Initialize:** Population $P$ of $\mu$ parent individuals $(x, \sigma)$.
2. **Loop:**
    - **Recombination & Mutation:** Generate $\lambda$ offspring. For each, select $\rho$ parents from $P$ for recombination, then apply mutation to $x$ and $\sigma$.
    - **Evaluation:** Calculate objective function values $f(x)$ for all $\lambda$ offspring.
    - **Selection (Plus):** Create a selection pool consisting of the **$\mu$ old parents + $\lambda$ new offspring**.
    - **Truncation:** Select the best $\mu$ individuals from this combined pool $(\mu + \lambda)$ to form the population $P$ for the next generation.
3. **Terminate** when stopping criteria are met.

## II. Testing Functions

### 1. Sphere Model

This is the most fundamental and famous function in ES theory, often used as a baseline to evaluate convergence speed.

- **Definition:** $f(x) = |x|^2 = \sum_{i=1}^n x_i^2$.
- **Characteristics:** This function is convex, unimodal, and perfectly rotationally symmetric. The function value depends only on the distance $R$ from the current point to the optimum,.
- **Role:** Rechenberg used this function (along with the corridor function) to derive the **1/5th success rule**. The optimal normalized progress rate of the (1+1)-ES on this function is approximately 0.202. Additionally, the **Noisy Sphere** version $(f(x) + \text{noise})$ is used to demonstrate the superiority of using a population $(\mu, \lambda)$ over a single individual $(1+1)$ in noisy environments.

### 2. Corridor Model

A simple linear model with inequality constraints, studied by Rechenberg in the early days.

- **Definition:** $f(x) = x_1$ if $|x_i| \le 1$ for all $i = 2, \dots, n$; otherwise infinity $(\infty)$.
- **Characteristics:** The goal is to minimize $x_1$ while keeping other variables within a narrow corridor.
- **Role:** Together with the sphere function, this function helped define the evolution window and reinforce the step-size adjustment rule based on a success probability of approximately 1/5 (about 0.184 for this function).

### 3. Cigar Function
This function represents ill-conditioned problems, where variables have vastly different scales of influence on the objective function.

- **Definition:** $f(x) = x_1^2 + \xi \sum_{i=2}^n x_i^2$ with $\xi \ge 1$.
- **Characteristics:** $\xi$ is the condition number. When $\xi$ is large, the function takes the shape of a long, narrow valley resembling a cigar.
- **Role:** It tests the algorithm's ability to adapt different step sizes for different directions. Without covariance matrix adaptation mechanisms (like CMA-ES), standard algorithms often struggle significantly with this function.

### 4. Ridge Functions

This class of functions tests the algorithm's ability to move along a specific direction (the ridge axis) while balancing to avoid sliding away from that axis.

- **Definition (Parabolic Ridge):** $f(x) = x_1 + \xi (\sum_{i=2}^n x_i^2)^{\alpha/2}$.
- **Characteristics:** Progress is made by moving along the $x_1$ axis (decreasing indefinitely), but deviating into other axes is heavily penalized.
- **Role:** It poses a conflict between short-term goals (keeping distance $R$ to the axis small by reducing step size) and long-term goals (moving far along the axis by maintaining a large step size). This is used to test the step-size adaptation mechanism's ability to avoid stagnation.

### 5. Rastrigin

This is a typical representative of multimodal functions, used to test global search capabilities.

- **Definition:** $F_{Ras}(y) = \sum_{k=1}^N [y_k^2 + B(1 - \cos(2\pi y_k))]$.
- **Characteristics:** The overall structure is parabolic, but the surface is covered with countless local optima (ripples).
- **Role:** It tests whether the ES gets trapped in local optima or can overcome them to reach the global optimum region. Convergence plots on this function often show phases of linear convergence alternating with slower phases as the algorithm navigates through the ripples.

### 6. Jump Function - Binary

In binary space, this function is used to demonstrate the benefits of recombination.

- **Characteristics:** A bimodal sphere function with a large "gap" that simple mutation finds very difficult to cross.
- **Role:** Jansen and Wegener used this function to prove that while the $(1+1)$-ES requires superpolynomial time to solve it, the recombinative $(\mu/2+1)$-ES requires only polynomial time, thereby confirming the role of recombination in escaping local optima.