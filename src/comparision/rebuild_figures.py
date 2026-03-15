"""Rebuild comparison figures from exported JSON files.

Reads the JSON files produced by ``export_results_json`` and regenerates
publication-quality figures identical to those created during live runs.

Automatically detects whether the JSON is from a **continuous** or
**discrete** benchmark based on its structure.

Usage
-----
    cd src

    # Rebuild a single JSON file (displays interactively):
    python -m comparision.rebuild_figures figures/compare_sphere_dim_2.json

    # Save PNGs instead of showing:
    python -m comparision.rebuild_figures figures/compare_sphere_dim_2.json --save

    # Rebuild every JSON in the figures/ directory:
    python -m comparision.rebuild_figures --all [--save]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt

# ── Colour / marker palettes (kept in sync with live utilities) ──────
_COLORS_CONTINUOUS = [
    "#D32F2F", "#1976D2", "#388E3C", "#F57C00", "#7B1FA2",
    "#00838F", "#C2185B", "#5D4037", "#455A64", "#AFB42B",
    "#0097A7", "#E64A19", "#512DA8", "#00695C", "#AD1457",
    "#283593", "#827717",
]

_COLORS_DISCRETE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", "#AEC7E8",
]

_MARKERS = [
    "o", "s", "^", "D", "v", "P", "X", "*", "p", "h",
    "<", ">", "d", "H", "8", "+", "x",
]

# Classical graph-search algorithms (single-point, no convergence curve)
_CLASSICAL_ALGOS = {"DFS", "BFS", "UCS", "Greedy", "A*"}


# =====================================================================
# Helpers
# =====================================================================

def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pick_colors(n: int, palette: list[str]) -> list[str]:
    return (palette * ((n // len(palette)) + 1))[:n]


def _is_discrete(data: dict) -> bool:
    """Heuristic: discrete JSONs contain ``best_solution`` or lack ``dim``."""
    algos = data.get("algorithms", {})
    if not algos:
        return False
    first = next(iter(algos.values()))
    if "best_solution" in first:
        return True
    if "dim" not in data and ("size" in data or "n_cities" in data
                              or "n_items" in data or "n_vertices" in data):
        return True
    return False


def _title_suffix(data: dict) -> str:
    if "dim" in data:
        return f" (dim = {data['dim']})"
    if "size" in data:
        return f" ({data['size']})"
    return ""


def _out_paths(json_path: str, save: bool):
    """Derive per-subplot save paths from the JSON path."""
    if not save:
        return {}
    base = os.path.splitext(json_path)[0]
    return {
        "convergence": f"{base}_convergence.png",
        "quality":     f"{base}_quality.png",
        "time":        f"{base}_time.png",
        "robustness":  f"{base}_robustness.png",
        "fitness":     f"{base}_fitness.png",
    }


def _save_fig(fig, path: str | None):
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")


# =====================================================================
# Continuous figures (4 separate windows)
# =====================================================================

def rebuild_continuous(data: dict, json_path: str, save: bool) -> None:
    algos = data["algorithms"]
    algo_names = list(algos.keys())
    n = len(algo_names)
    colors = _pick_colors(n, _COLORS_CONTINUOUS)
    problem = data.get("problem", "")
    suffix = _title_suffix(data)
    paths = _out_paths(json_path, save)

    # ── 1. Convergence Speed ─────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    fig1.suptitle(f"Convergence Speed — {problem}{suffix}",
                  fontsize=16, fontweight="bold", y=0.98)
    for idx, name in enumerate(algo_names):
        runs = algos[name]["runs"]
        curves = [r["fitness_curve"] for r in runs]
        max_len = max(len(c) for c in curves)
        arr = np.full((len(curves), max_len), np.nan)
        for i, c in enumerate(curves):
            arr[i, :len(c)] = c
        mean_c = np.nanmean(arr, axis=0)
        std_c = np.nanstd(arr, axis=0)
        x = np.arange(1, max_len + 1)
        marker = _MARKERS[idx % len(_MARKERS)]
        mark_every = max(1, max_len // 10)
        ax1.plot(x, mean_c, label=name, color=colors[idx],
                 linewidth=1.6, marker=marker, markersize=4,
                 markevery=mark_every)
        ax1.fill_between(x, mean_c - std_c, mean_c + std_c,
                         color=colors[idx], alpha=0.08)
    ax1.set_yscale("log")
    ax1.set_xlabel("Iteration", fontsize=11)
    ax1.set_ylabel("Best Fitness (log)", fontsize=11)
    ax1.set_title("Convergence Speed", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=7, ncol=2, loc="upper right")
    ax1.grid(True, alpha=0.3, which="both", linestyle="--")
    fig1.tight_layout(rect=(0, 0, 1, 0.95))
    _save_fig(fig1, paths.get("convergence"))

    # ── 2. Solution Quality (log scale) ──────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    fig2.suptitle(f"Solution Quality — {problem}{suffix}",
                  fontsize=16, fontweight="bold", y=0.98)
    box_data = [
        [r["best_fitness"] for r in algos[name]["runs"]]
        for name in algo_names
    ]
    bp = ax2.boxplot(box_data, tick_labels=algo_names,
                     patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax2.set_yscale("log")
    ax2.set_ylabel("Best Fitness (log)", fontsize=11)
    ax2.set_title("Solution Quality", fontsize=13, fontweight="bold")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3, axis="y", linestyle="--", which="both")
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    _save_fig(fig2, paths.get("quality"))

    # ── 3. Computational Time ────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    fig3.suptitle(f"Computational Time — {problem}{suffix}",
                  fontsize=16, fontweight="bold", y=0.98)
    means_t = [algos[n]["mean_time_ms"] for n in algo_names]
    stds_t = [algos[n]["std_time_ms"] for n in algo_names]
    bars = ax3.bar(algo_names, means_t, yerr=stds_t, capsize=4,
                   color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax3.set_ylabel("Time (ms)", fontsize=11)
    ax3.set_title("Computational Time", fontsize=13, fontweight="bold")
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, alpha=0.3, axis="y", linestyle="--")
    for bar, m in zip(bars, means_t):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{m:.0f}", ha="center", va="bottom", fontsize=7)
    fig3.tight_layout(rect=(0, 0, 1, 0.95))
    _save_fig(fig3, paths.get("time"))

    # ── 4. Robustness ────────────────────────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(10, 7))
    fig4.suptitle(f"Robustness — {problem}{suffix}",
                  fontsize=16, fontweight="bold", y=0.98)
    stds_f = [algos[n]["std_fitness"] for n in algo_names]
    x_pos = np.arange(n)
    width = 0.4
    ax4.bar(x_pos - width / 2, stds_f, width, label="Std Dev (fitness)",
            color=colors, alpha=0.6, edgecolor="black", linewidth=0.5)
    ax4.set_ylabel("Std Dev of Best Fitness", fontsize=11)
    ax4.set_title("Robustness", fontsize=13, fontweight="bold")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(algo_names, rotation=45, fontsize=9)
    ax4.grid(True, alpha=0.3, axis="y", linestyle="--")

    ax4r = ax4.twinx()
    means_f = [algos[n]["mean_fitness"] for n in algo_names]
    cv = np.array([s / abs(m) if abs(m) > 1e-30 else 0.0
                   for s, m in zip(stds_f, means_f)])
    ax4r.plot(x_pos, cv, "D-", color="red", markersize=5, linewidth=1.2,
              label="CV (Std/Mean)")
    ax4r.set_ylabel("Coefficient of Variation", fontsize=10, color="red")
    ax4r.tick_params(axis="y", labelcolor="red")

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4r.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    fig4.tight_layout(rect=(0, 0, 1, 0.95))
    _save_fig(fig4, paths.get("robustness"))

    if not save:
        plt.show()


# =====================================================================
# Discrete figures (fitness + time + convergence)
# =====================================================================

def rebuild_discrete(data: dict, json_path: str, save: bool) -> None:
    algos = data["algorithms"]
    algo_names = list(algos.keys())
    n = len(algo_names)
    colors = _pick_colors(n, _COLORS_DISCRETE)
    problem = data.get("problem", "")
    suffix = _title_suffix(data)
    paths = _out_paths(json_path, save)

    # Detect fitness label from problem name
    prob_lower = problem.lower()
    if "knapsack" in prob_lower:
        fitness_label = "Profit"
    elif "tsp" in prob_lower:
        fitness_label = "Distance"
    elif "coloring" in prob_lower or "graph" in prob_lower:
        fitness_label = "Conflicts"
    else:
        fitness_label = "Best Fitness"

    # ── 1. Best Fitness ──────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    fig1.suptitle(f"{fitness_label} — {problem}{suffix}",
                  fontsize=16, fontweight="bold", y=0.98)
    means_f = [algos[name]["mean_fitness"] for name in algo_names]
    stds_f = [algos[name]["std_fitness"] for name in algo_names]
    bars = ax1.bar(algo_names, means_f, yerr=stds_f, capsize=4,
                   color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel(fitness_label, fontsize=11)
    ax1.set_title(fitness_label, fontsize=13, fontweight="bold")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3, axis="y", linestyle="--")
    for bar, m in zip(bars, means_f):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{m:.2f}", ha="center", va="bottom", fontsize=7)
    fig1.tight_layout(rect=(0, 0, 1, 0.95))
    _save_fig(fig1, paths.get("fitness"))

    # ── 2. Computational Time ────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    fig2.suptitle(f"Computational Time — {problem}{suffix}",
                  fontsize=16, fontweight="bold", y=0.98)
    means_t = [algos[name]["mean_time_ms"] for name in algo_names]
    stds_t = [algos[name]["std_time_ms"] for name in algo_names]
    bars = ax2.bar(algo_names, means_t, yerr=stds_t, capsize=4,
                   color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Time (ms)", fontsize=11)
    ax2.set_title("Computational Time", fontsize=13, fontweight="bold")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3, axis="y", linestyle="--")
    for bar, m in zip(bars, means_t):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{m:.0f}", ha="center", va="bottom", fontsize=7)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    _save_fig(fig2, paths.get("time"))

    # ── 3. Convergence ───────────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    fig3.suptitle(f"Convergence Speed — {problem}{suffix}",
                  fontsize=16, fontweight="bold", y=0.98)

    max_iter = 1
    for name in algo_names:
        if name not in _CLASSICAL_ALGOS:
            for r in algos[name]["runs"]:
                if len(r["fitness_curve"]) > max_iter:
                    max_iter = len(r["fitness_curve"])

    for idx, name in enumerate(algo_names):
        runs = algos[name]["runs"]
        curves = [r["fitness_curve"] for r in runs]
        curve_len = max(len(c) for c in curves)

        if name in _CLASSICAL_ALGOS:
            val = curves[0][0]
            ax3.axhline(y=val, label=name, color=colors[idx],
                        linewidth=1.4, linestyle="--", alpha=0.7)
        else:
            arr = np.full((len(curves), curve_len), np.nan)
            for i, c in enumerate(curves):
                arr[i, :len(c)] = c
            mean_c = np.nanmean(arr, axis=0)
            x = np.arange(1, curve_len + 1)
            ax3.plot(x, mean_c, label=name, color=colors[idx], linewidth=1.6)

    ax3.set_xlabel("Iteration", fontsize=12)
    ax3.set_ylabel(fitness_label, fontsize=12)
    ax3.set_title("All Algorithms", fontsize=13)
    ax3.legend(fontsize=9, loc="best")
    ax3.grid(True, alpha=0.3, linestyle="--")
    fig3.tight_layout(rect=(0, 0, 1, 0.95))

    convergence_path = None
    if save:
        convergence_path = os.path.splitext(json_path)[0] + "_convergence.png"
    _save_fig(fig3, convergence_path)

    if not save:
        plt.show()


# =====================================================================
# Entry point
# =====================================================================

def rebuild_one(json_path: str, save: bool) -> None:
    data = _load_json(json_path)
    print(f"\nRebuilding: {json_path}  "
          f"[{data.get('problem', '?')}]")
    if _is_discrete(data):
        rebuild_discrete(data, json_path, save)
    else:
        rebuild_continuous(data, json_path, save)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild comparison figures from exported JSON")
    parser.add_argument("json", nargs="?", default=None,
                        help="Path to a single JSON file to rebuild")
    parser.add_argument("--all", action="store_true",
                        help="Rebuild all JSON files in figures/")
    parser.add_argument("--save", action="store_true",
                        help="Save PNGs instead of showing interactively")
    args = parser.parse_args()

    if not args.json and not args.all:
        parser.error("Provide a JSON path or use --all")

    if args.all:
        fig_dir = os.path.join(os.path.dirname(__file__), "figures")
        paths = sorted(glob.glob(os.path.join(fig_dir, "*.json")))
        if not paths:
            print(f"No JSON files found in {fig_dir}")
            return
        print(f"Found {len(paths)} JSON file(s) in {fig_dir}")
        for p in paths:
            rebuild_one(p, save=args.save)
    else:
        rebuild_one(args.json, save=args.save)


if __name__ == "__main__":
    main()
