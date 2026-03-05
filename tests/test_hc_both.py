"""Quick smoke test: HillClimbing on discrete + continuous problems."""

import sys
sys.path.append("src")

from problems.discrete.knapsack import Knapsack
from problems.continuous.sphere import Sphere
from algorithm.local.HillClimbing import HillClimbing
from algorithm.local.SteepestAscentHC import SteepestAscentHillClimbing

# --- Discrete: Knapsack ---
ks = Knapsack.create_small()
hc = HillClimbing({"max_iterations": 500}, ks)
state, value = hc.run()
info = ks.decode_binary(state)
print("=== Hill Climbing on Knapsack (discrete) ===")
print(f"  Selected: {info['selected_items']}  Value: {info['total_value']}  Weight: {info['total_weight']}")
print(f"  Objective (minimised): {value}")
print(f"  Iterations: {len(hc.history)}")

# --- Continuous: Sphere ---
sp = Sphere(n_dim=5)
hc2 = HillClimbing({"max_iterations": 2000}, sp)
state2, value2 = hc2.run()
print()
print("=== Hill Climbing on Sphere (continuous) ===")
print(f"  Best solution: {state2.round(4)}")
print(f"  Best value   : {value2:.6f}")
print(f"  Iterations   : {len(hc2.history)}")

# --- SteepestAscent on Sphere ---
sa = SteepestAscentHillClimbing(sp, max_iterations=2000)
state3, value3 = sa.run()
print()
print("=== Steepest Ascent HC on Sphere (continuous) ===")
print(f"  Best solution: {state3.round(4)}")
print(f"  Best value   : {value3:.6f}")
print(f"  Iterations   : {len(sa.history)}")

# --- SteepestAscent on Knapsack ---
sa2 = SteepestAscentHillClimbing(ks, max_iterations=500)
state4, value4 = sa2.run()
info2 = ks.decode_binary(state4)
print()
print("=== Steepest Ascent HC on Knapsack (discrete) ===")
print(f"  Selected: {info2['selected_items']}  Value: {info2['total_value']}  Weight: {info2['total_weight']}")
print(f"  Objective (minimised): {value4}")
print(f"  Iterations: {len(sa2.history)}")
