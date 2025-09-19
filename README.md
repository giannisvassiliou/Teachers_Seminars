# Teachers_Seminars

Educational institutions face complex challenges when allocating limited teach- 1
ing resources to specialized seminars, where budget, capacity, and balanced disciplinary 2
representation must all be satisfied simultaneously. We address this for the first time 3
in the educational domain by formulating the teacher seminar selection problem as a 4
multi-dimensional knapsack variant with category-specific benefit multipliers. To solve 5
it, we design a constraint-aware genetic algorithm that incorporates smart initialization, 6
category-sensitive operators, adaptive penalties, and targeted repair mechanisms. In ex- 7
periments on a realistic dataset representing multiple academic categories, our method 8
achieved an 11.5% improvement in solution quality compared to the best constraint-aware 9
greedy baseline while maintaining perfect constraint satisfaction (100% feasibility) versus 10
0–30% for baseline methods. Statistical tests confirmed significant and practically meaning- 11
ful advantages. For comprehensive benchmarking, we also implemented binary Particle 12
Swarm Optimization (PSO) and Tabu Search (TS) solvers with standard parameteriza- 13
tions. While PSO consistently produced feasible solutions with high budget utilization, its 14
optimization quality was substantially lower than that of the GA. Notably, Tabu Search 15
achieved the highest performance with a mean fitness of 1557.3 compared to GA’s 1533.2, 16
demonstrating that memory-based local search can be highly competitive for this problem 17
structure. These findings show that metaheuristic approaches, particularly those integrat- 18
ing constraint-awareness into evolutionary or memory-based search, provide effective, 19
scalable decision-support frameworks for complex, multi-constraint educational resource 20
allocation


knapcorr4stat.py - GA Teacher Seminar Selection
pso_knapcorr4stat.py - Particle Swarm Optimization - Teacher Seminar Selection
tabu_knapcorr4stat.py - Tabu Search -Teacher Seminar Selection
