
"""
Tabu Search (binary) for the Teacher Seminar Selection problem.

This module plugs into knapcorr4stat.py, reusing its data structures and fitness function.
Provides:
  - TabuParams dataclass
  - TeacherSeminarTabu class with .run()
  - run_tabu_single() helper for one-off trial
"""
from dataclasses import dataclass
from typing import List, Dict
import random
import numpy as np

from knapcorr4stat import (
    Teacher, SeminarConfig, calculate_fitness_detailed,
    create_100_teachers_dataset, create_100_teacher_config
)

@dataclass
class TabuParams:
    iterations: int = 600
    tenure: int = 20
    candidate_pool: int = 60
    neighborhood: str = "both"  # 'flip' | 'swap' | 'both'
    penalty_weight: float = 150.0
    diversify_every: int = 120
    diversify_strength: int = 4
    respect_constraints: bool = True
    init: str = "greedy"  # 'greedy' | 'random' | 'half-feasible'

class TeacherSeminarTabu:
    def __init__(self, teachers: List[Teacher], config: SeminarConfig, params: TabuParams):
        self.teachers = teachers
        self.config = config
        self.params = params
        self.n = len(teachers)
        self.rng = random.Random()

    def _value(self, idx: int) -> float:
        t = self.teachers[idx]
        return t.base_benefit * self.config.category_priorities[t.category]

    def _repair_positions(self, x: np.ndarray) -> np.ndarray:
        sel = np.where(x == 1)[0]
        excess = len(sel) - self.config.total_positions
        if excess <= 0:
            return x
        scored = [(i, self._value(i) / max(1e-9, self.teachers[i].cost)) for i in sel]
        scored.sort(key=lambda z: z[1], reverse=True)
        keep = set(i for i, _ in scored[:self.config.total_positions])
        for i in sel:
            if i not in keep:
                x[i] = 0
        return x

    def _repair_budget(self, x: np.ndarray) -> np.ndarray:
        total_cost = sum(self.teachers[i].cost for i in np.where(x == 1)[0])
        while total_cost > self.config.budget_capacity:
            selected = [i for i in np.where(x == 1)[0]]
            if not selected:
                break
            eff = [(i, self._value(i) / max(1e-9, self.teachers[i].cost)) for i in selected]
            i_drop = min(eff, key=lambda z: z[1])[0]
            x[i_drop] = 0
            total_cost -= self.teachers[i_drop].cost
        return x

    def _repair(self, x: np.ndarray) -> np.ndarray:
        if not self.params.respect_constraints:
            return x
        x = self._repair_positions(x.copy())
        x = self._repair_budget(x)
        return x

    def _fitness(self, x: np.ndarray) -> float:
        individual = [bool(v) for v in x.astype(int).tolist()]
        fit, _ = calculate_fitness_detailed(
            self.teachers, individual, self.config, self.params.penalty_weight, debug=False
        )
        return fit

    def _init_solution(self) -> np.ndarray:
        if self.params.init == "random":
            p = min(0.9, self.config.total_positions / max(1, self.n) * 1.2)
            x = (np.random.rand(self.n) < p).astype(int)
            return self._repair(x)
        else:  # greedy density
            density = [(i, self._value(i) / max(1e-9, self.teachers[i].cost)) for i in range(self.n)]
            density.sort(key=lambda z: z[1], reverse=True)
            x = np.zeros(self.n, dtype=int)
            for i, _ in density:
                x[i] = 1
                x = self._repair_positions(x)
                if sum(self.teachers[k].cost for k in np.where(x == 1)[0]) > self.config.budget_capacity:
                    x[i] = 0
            return x

    def _neighbors(self, x: np.ndarray):
        nbh = []
        N = self.n
        cand = self.params.candidate_pool
        if self.params.neighborhood in ("flip", "both"):
            idxs = self.rng.sample(range(N), k=min(cand, N))
            for i in idxs:
                y = x.copy()
                y[i] = 1 - y[i]
                nbh.append((('flip', (i,)), self._repair(y)))
        return nbh

    def run(self, verbose: bool = False) -> Dict:
        x = self._init_solution()
        fx = self._fitness(x)
        best_x, best_f = x.copy(), fx
        tabu = {}

        def is_tabu(move):
            return tabu.get(move, 0) > 0

        for it in range(1, self.params.iterations + 1):
            tabu = {m: t-1 for m, t in tabu.items() if t-1 > 0}
            best_move, best_candidate, best_candidate_f = None, None, -1e18
            for move, y in self._neighbors(x):
                fy = self._fitness(y)
                if is_tabu(move) and fy <= best_f:
                    continue
                if fy > best_candidate_f:
                    best_candidate_f, best_candidate, best_move = fy, y, move

            if best_candidate is None:
                j = self.rng.randrange(self.n)
                x[j] = 1 - x[j]
                x = self._repair(x)
                fx = self._fitness(x)
                continue

            x, fx = best_candidate, best_candidate_f
            tabu[best_move] = self.params.tenure
            if fx > best_f:
                best_f, best_x = fx, x.copy()

            if verbose and it % max(1, self.params.iterations // 10) == 0:
                print(f"[TS] iter {it:4d} | current={fx:.2f} | best={best_f:.2f}")

        best_individual = [bool(v) for v in best_x.astype(int).tolist()]
        _, details = calculate_fitness_detailed(
            self.teachers, best_individual, self.config, self.params.penalty_weight, debug=False
        )
        return {
            'best_fitness': details['fitness'],
            'best_solution': {
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

def run_tabu_single(teachers: List[Teacher], config: SeminarConfig, tabu_params: TabuParams, verbose: bool = False) -> Dict:
    ts = TeacherSeminarTabu(teachers, config, tabu_params)
    result = ts.run(verbose=verbose)
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
    random.seed(123)
    np.random.seed(123)
    teachers = create_100_teachers_dataset()
    config = create_100_teacher_config()
    params = TabuParams()
    print("Running Tabu Search on Teacher Seminar Selection...")
    metrics = run_tabu_single(teachers, config, params, verbose=True)
    print("\nTabu Summary:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
