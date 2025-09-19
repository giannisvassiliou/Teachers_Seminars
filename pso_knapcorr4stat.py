
"""
Particle Swarm Optimization (PSO) add-on for the Teacher Seminar Selection problem.

This module plugs into the existing code in knapcorr4stat.py and uses the same
data structures and fitness function. You can:
  - import and call `run_pso_single(...)` to get per-trial metrics shaped like the GA block,
  - or instantiate `TeacherSeminarPSO` directly and call `.run()` for full details.

Requirements: numpy
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import random

# Import the problem definitions & fitness from your existing file
from knapcorr4stat import (
    Teacher, SeminarConfig, calculate_fitness_detailed,
    create_100_teachers_dataset, create_100_teacher_config
)

@dataclass
class PSOParams:
    """Particle Swarm Optimization hyperparameters"""
    swarm_size: int = 120
    iterations: int = 200
    inertia: float = 0.72            # ω
    cognitive: float = 1.49          # c1
    social: float = 1.49             # c2
    vmin: float = -4.0
    vmax: float = 4.0
    penalty_weight: float = 150
    repair_positions: bool = True    # softly enforce total_positions

class TeacherSeminarPSO:
    """Binary PSO for Teacher Seminar Selection.

    - Velocity updated in R^n, position realized via sigmoid probability -> {0,1}.
    - Optional repair to respect the total_positions limit by keeping highest
      value-density choices when over-selected.
    - Fitness uses the same `calculate_fitness_detailed` as the GA.
    """
    def __init__(self, teachers: List[Teacher], config: SeminarConfig, params: PSOParams):
        self.teachers = teachers
        self.config = config
        self.params = params
        self.dim = len(teachers)
        self.rng = np.random.default_rng()

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def _repair_positions_limit(self, pos_bin: np.ndarray):
        if not self.params.repair_positions:
            return pos_bin
        selected_idx = np.where(pos_bin == 1)[0]
        excess = len(selected_idx) - self.config.total_positions
        if excess <= 0:
            return pos_bin
        # Keep top by adjusted_value / cost (density)
        scores = []
        for idx in selected_idx:
            t = self.teachers[idx]
            adjusted = t.base_benefit * self.config.category_priorities[t.category]
            density = adjusted / max(1e-9, t.cost)
            scores.append((idx, density))
        scores.sort(key=lambda x: x[1], reverse=True)
        keep_set = set(i for i, _ in scores[:self.config.total_positions])
        # drop the lowest densities
        for i, _ in scores[self.config.total_positions:]:
            pos_bin[i] = 0
        return pos_bin

    def _fitness(self, pos_bin: np.ndarray) -> float:
        individual = [bool(x) for x in pos_bin.astype(int).tolist()]
        fit, _ = calculate_fitness_detailed(
            self.teachers, individual, self.config, self.params.penalty_weight, debug=False
        )
        return fit

    def run(self, verbose: bool = False) -> Dict:
        n, d = self.params.swarm_size, self.dim

        # Initialize velocities and positions
        V = self.rng.uniform(self.params.vmin, self.params.vmax, size=(n, d))
        Xprob = self._sigmoid(V)
        X = (self.rng.random((n, d)) < Xprob).astype(int)
        # Optional repair for positions limit
        for i in range(n):
            X[i] = self._repair_positions_limit(X[i])

        # Personal & global bests
        pbest_pos = X.copy()
        pbest_fit = np.array([self._fitness(x) for x in pbest_pos])
        g_idx = int(np.argmax(pbest_fit))
        gbest_pos = pbest_pos[g_idx].copy()
        gbest_fit = float(pbest_fit[g_idx])

        # Main loop
        for it in range(self.params.iterations):
            r1 = self.rng.random((n, d))
            r2 = self.rng.random((n, d))
            # Velocity update
            V = (self.params.inertia * V +
                 self.params.cognitive * r1 * (pbest_pos - X) +
                 self.params.social * r2 * (gbest_pos - X))
            V = np.clip(V, self.params.vmin, self.params.vmax)

            # Position update via sigmoid sampling
            Xprob = self._sigmoid(V)
            X = (self.rng.random((n, d)) < Xprob).astype(int)
            # Repair
            for i in range(n):
                X[i] = self._repair_positions_limit(X[i])

            # Evaluate and update pbest/gbest
            fitnesses = np.array([self._fitness(x) for x in X])

            improved = fitnesses > pbest_fit
            pbest_fit[improved] = fitnesses[improved]
            pbest_pos[improved] = X[improved]

            g_idx = int(np.argmax(pbest_fit))
            if pbest_fit[g_idx] > gbest_fit:
                gbest_fit = float(pbest_fit[g_idx])
                gbest_pos = pbest_pos[g_idx].copy()

            if verbose and (it % max(1, self.params.iterations // 10) == 0):
                print(f"PSO iter {it:4d} | best={gbest_fit:.3f}")

        # Prepare detailed results similar to GA's structure
        best_individual = [bool(x) for x in gbest_pos.tolist()]
        _, details = calculate_fitness_detailed(
            self.teachers, best_individual, self.config, self.params.penalty_weight, debug=False
        )

        selected_teachers = [{
            'id': t.id,
            'name': t.name,
            'subject': t.subject,
            'category': t.category,
            'base_benefit': t.base_benefit,
            'cost': t.cost,
            'adjusted_benefit': (t.base_benefit * self.config.category_priorities[t.category])
        } for t in details['selected_teachers']]

        return {
            'best_fitness': details['fitness'],
            'best_solution': {
                'selected_teachers': selected_teachers,
                'total_cost': details['total_cost'],
                'total_benefit': details['total_benefit'],
                'category_distribution': details['category_counts'],
                'constraint_violations': details['violations'],
                'utilization_rates': {
                    'budget': (details['total_cost'] / self.config.budget_capacity) * 100,
                    'positions': (details['selected_count'] / self.config.total_positions) * 100
                }
            }
        }

def run_pso_single(teachers: List[Teacher], config: SeminarConfig, pso_params: PSOParams, verbose: bool = False) -> Dict:
    """Helper to run PSO once and return a compact metrics dict (like run_single_trial uses)."""
    pso = TeacherSeminarPSO(teachers, config, pso_params)
    result = pso.run(verbose=verbose)
    return {
        'fitness': result['best_fitness'],
        'cost': result['best_solution']['total_cost'],
        'benefit': result['best_solution']['total_benefit'],
        'violations': len(result['best_solution']['constraint_violations']),
        'budget_utilization': result['best_solution']['utilization_rates']['budget'],
        'position_utilization': result['best_solution']['utilization_rates']['positions'],
        'category_distribution': result['best_solution']['category_distribution']
    }

if __name__ == "__main__":
    # Quick demo run so you can `python pso_knapcorr4stat.py` directly.
    random.seed(42)
    np.random.seed(42)
    teachers = create_100_teachers_dataset()
    config = create_100_teacher_config()
    params = PSOParams(
        swarm_size=120,
        iterations=180,
        inertia=0.72,
        cognitive=1.49,
        social=1.49,
        vmin=-4.0,
        vmax=4.0,
        penalty_weight=150,
        repair_positions=True
    )
    print("Running PSO on Teacher Seminar Selection...")
    summary = run_pso_single(teachers, config, params, verbose=True)
    print("\nPSO Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
