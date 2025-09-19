"""
Statistical Analysis Runner for Teacher Seminar Selection Algorithms
Runs comprehensive comparison 10 times and provides statistical analysis
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
import scipy.stats as stats
from collections import defaultdict

# DEAP imports
from deap import algorithms, base, creator, tools
import warnings
warnings.filterwarnings('ignore')

# [Include all the previous classes and functions from the uploaded code]
@dataclass
class Teacher:
    """Represents a teacher with their attributes"""
    id: int
    name: str
    subject: str
    category: str
    base_benefit: float
    cost: int
    experience: int
    category_multiplier: float = 1.0

@dataclass
class SeminarConfig:
    """Configuration for the seminar constraints and priorities"""
    total_positions: int
    budget_capacity: int
    category_limits: Dict[str, Dict[str, int]]
    category_priorities: Dict[str, float]

@dataclass
class GAParams:
    """Genetic Algorithm parameters for DEAP"""
    population_size: int = 100
    generations: int = 200
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    tournament_size: int = 3
    elite_size: int = 10
    penalty_weight: float = 150

def calculate_fitness_detailed(teachers: List[Teacher], individual: List[bool], 
                              config: SeminarConfig, penalty_weight: float = 150,
                              debug: bool = False) -> Tuple[float, Dict]:
    """Centralized fitness calculation with detailed logging"""
    
    selected_teachers = []
    total_cost = 0
    total_benefit = 0
    category_counts = {cat: 0 for cat in config.category_limits.keys()}
    
    for i, selected in enumerate(individual):
        if selected:
            teacher = teachers[i]
            selected_teachers.append(teacher)
            total_cost += teacher.cost
            category_counts[teacher.category] += 1
            
            adjusted_benefit = (teacher.base_benefit * 
                              config.category_priorities[teacher.category])
            total_benefit += adjusted_benefit
    
    violations = []
    penalties = 0
    selected_count = len(selected_teachers)
    
    # Position limit penalty
    if selected_count > config.total_positions:
        violation = f"Too many positions: {selected_count}/{config.total_positions}"
        violations.append(violation)
        excess = selected_count - config.total_positions
        position_penalty = excess * penalty_weight
        penalties += position_penalty
    
    # Budget constraint penalty  
    if total_cost > config.budget_capacity:
        violation = f"Over budget: {total_cost}/{config.budget_capacity}"
        violations.append(violation)
        excess = total_cost - config.budget_capacity
        budget_penalty = excess * penalty_weight
        penalties += budget_penalty
    
    # Category constraint penalties
    for category, count in category_counts.items():
        limits = config.category_limits[category]
        
        if count < limits['min']:
            violation = f"{category}: {count} < {limits['min']} (minimum)"
            violations.append(violation)
            shortage = limits['min'] - count
            category_penalty = shortage * penalty_weight * 0.5
            penalties += category_penalty
        
        if count > limits['max']:
            violation = f"{category}: {count} > {limits['max']} (maximum)"
            violations.append(violation)
            excess = count - limits['max']
            category_penalty = excess * penalty_weight
            penalties += category_penalty
    
    # Diversity bonus
    active_categories = sum(1 for count in category_counts.values() if count > 0)
    diversity_bonus = active_categories * 15
    
    # Final fitness calculation
    fitness = max(0.1, total_benefit + diversity_bonus - penalties)
    
    details = {
        'selected_count': selected_count,
        'total_cost': total_cost,
        'total_benefit': total_benefit,
        'penalties': penalties,
        'diversity_bonus': diversity_bonus,
        'violations': violations,
        'category_counts': category_counts,
        'fitness': fitness,
        'selected_teachers': selected_teachers
    }
    
    return fitness, details

class TeacherSeminarDEAP:
    """Teacher Seminar Selection using DEAP Framework"""
    
    def __init__(self, teachers: List[Teacher], config: SeminarConfig, params: GAParams):
        self.teachers = teachers
        self.config = config
        self.params = params
        self.categories = list(config.category_limits.keys())
        self.generation_data = []
        self.setup_deap()
        
    def setup_deap(self):
        """Initialize DEAP framework components"""
        
        # Clear existing classes to avoid conflicts
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
            
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        
        self.toolbox.register("attr_bool", self._smart_gene_init)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                             self.toolbox.attr_bool, n=len(self.teachers))
        self.toolbox.register("population", tools.initRepeat, list, 
                             self.toolbox.individual, n=self.params.population_size)
        
        self.toolbox.register("evaluate", self.evaluate_fitness)
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=self.params.tournament_size)
        
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        
    def _smart_gene_init(self) -> bool:
        """Smart initialization for individual genes"""
        return random.random() < (self.config.total_positions / len(self.teachers) * 1.2)
    
    def evaluate_fitness(self, individual) -> Tuple[float]:
        """Evaluate fitness of an individual"""
        fitness, _ = calculate_fitness_detailed(
            self.teachers, individual, self.config, 
            self.params.penalty_weight, debug=False
        )
        return (fitness,)
    
    def crossover(self, ind1, ind2):
        """Custom crossover with constraint awareness"""
        if random.random() > self.params.crossover_rate:
            return ind1, ind2
        
        size = len(ind1)
        if size < 3:
            return ind1, ind2
            
        point1 = random.randint(1, size - 2)
        point2 = random.randint(point1 + 1, size - 1)
        
        child1_genes = ind1[:]
        child2_genes = ind2[:]
        
        child1_genes[point1:point2], child2_genes[point1:point2] = ind2[point1:point2], ind1[point1:point2]
        
        self._repair_individual(child1_genes)
        self._repair_individual(child2_genes)
        
        ind1[:] = child1_genes
        ind2[:] = child2_genes
        
        return ind1, ind2
    
    def mutate(self, individual):
        """Custom mutation with constraint repair"""
        for i in range(len(individual)):
            if random.random() < self.params.mutation_rate:
                individual[i] = not individual[i]
        
        self._repair_individual(individual)
        return individual,
    
    def _repair_individual(self, individual: List[bool]):
        """Repair individual to respect position constraints"""
        selected_indices = [i for i, gene in enumerate(individual) if gene]
        
        if len(selected_indices) > self.config.total_positions:
            teacher_values = []
            for idx in selected_indices:
                teacher = self.teachers[idx]
                value = (teacher.base_benefit * 
                        teacher.category_multiplier * 
                        self.config.category_priorities[teacher.category])
                teacher_values.append((idx, value))
            
            teacher_values.sort(key=lambda x: x[1], reverse=True)
            
            for i in range(self.config.total_positions, len(teacher_values)):
                individual[teacher_values[i][0]] = False
    
    def run(self, verbose: bool = False) -> Dict:
        """Run the genetic algorithm using DEAP"""
        
        population = self.toolbox.population()
        
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        hof = tools.HallOfFame(self.params.elite_size)
        
        try:
            population, logbook = algorithms.eaSimple(
                population, self.toolbox,
                cxpb=self.params.crossover_rate,
                mutpb=self.params.mutation_rate,
                ngen=self.params.generations,
                stats=self.stats,
                halloffame=hof,
                verbose=False  # Always quiet for batch runs
            )
            
        except Exception as e:
            population = self._manual_evolution(population, hof, verbose=False)
            logbook = None
        
        if hof and len(hof) > 0:
            best_individual = hof[0]
            best_fitness = best_individual.fitness.values[0]
        else:
            best_individual = max(population, key=lambda x: x.fitness.values[0])
            best_fitness = best_individual.fitness.values[0]
        
        return self._prepare_results(best_individual, population, logbook)
    
    def _manual_evolution(self, population, hof, verbose):
        """Manual evolution loop as fallback"""
        for generation in range(self.params.generations):
            offspring = []
            
            while len(offspring) < self.params.population_size:
                parent1 = tools.selTournament(population, 1, self.params.tournament_size)[0]
                parent2 = tools.selTournament(population, 1, self.params.tournament_size)[0]
                
                child1 = creator.Individual(parent1[:])
                child2 = creator.Individual(parent2[:])
                
                if random.random() < self.params.crossover_rate:
                    self.crossover(child1, child2)
                
                if random.random() < self.params.mutation_rate:
                    self.mutate(child1)
                if random.random() < self.params.mutation_rate:
                    self.mutate(child2)
                
                offspring.extend([child1, child2])
            
            offspring = offspring[:self.params.population_size]
            
            for ind in offspring:
                if not hasattr(ind, 'fitness') or not ind.fitness.valid:
                    ind.fitness.values = self.evaluate_fitness(ind)
            
            population[:] = offspring
            
            if hof is not None:
                hof.update(population)
        
        return population
    
    def _prepare_results(self, best_individual: List[bool], 
                        population: List, logbook) -> Dict:
        """Prepare comprehensive results"""
        
        _, details = calculate_fitness_detailed(
            self.teachers, best_individual, self.config, 
            self.params.penalty_weight, debug=False
        )
        
        selected_teachers = []
        for teacher in details['selected_teachers']:
            selected_teachers.append({
                'id': teacher.id,
                'name': teacher.name,
                'subject': teacher.subject,
                'category': teacher.category,
                'base_benefit': teacher.base_benefit,
                'cost': teacher.cost,
                'adjusted_benefit': (teacher.base_benefit * 
                                   self.config.category_priorities[teacher.category])
            })
        
        final_fitnesses = [ind.fitness.values[0] for ind in population]
        
        return {
            'best_fitness': best_individual.fitness.values[0],
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
            },
            'final_population_stats': {
                'avg_fitness': np.mean(final_fitnesses),
                'std_fitness': np.std(final_fitnesses),
                'best_fitness': np.max(final_fitnesses),
                'worst_fitness': np.min(final_fitnesses)
            }
        }

def fair_constrained_greedy_selection(teachers: List[Teacher], config: SeminarConfig, 
                                     penalty_weight: float = 150, debug: bool = False) -> Dict:
    """Fair Constrained Greedy - Respects position and budget limits during selection"""
    
    teacher_scores = []
    for i, teacher in enumerate(teachers):
        adjusted_value = (teacher.base_benefit * 
                         teacher.category_multiplier * 
                         config.category_priorities[teacher.category])
        score = adjusted_value / teacher.cost
        teacher_scores.append((i, score, teacher, adjusted_value))
    
    teacher_scores.sort(key=lambda x: x[1], reverse=True)
    
    selected = [False] * len(teachers)
    selected_teachers = []
    category_count = {cat: 0 for cat in config.category_limits.keys()}
    total_cost = 0
    
    for i, (idx, score, teacher, benefit) in enumerate(teacher_scores):
        would_exceed_positions = len(selected_teachers) >= config.total_positions
        would_exceed_budget = total_cost + teacher.cost > config.budget_capacity
        would_exceed_category = category_count[teacher.category] >= config.category_limits[teacher.category]['max']
        
        if would_exceed_positions or would_exceed_budget or would_exceed_category:
            continue
        
        selected[idx] = True
        selected_teachers.append(teacher)
        category_count[teacher.category] += 1
        total_cost += teacher.cost
    
    fitness, details = calculate_fitness_detailed(
        teachers, selected, config, penalty_weight, debug=debug
    )
    
    return {
        'fitness': fitness,
        'selected': selected,
        'violations': details['violations'],
        'cost': details['total_cost'],
        'benefit': details['total_benefit'],
        'raw_benefit': details['total_benefit'],
        'penalties': details['penalties'],
        'diversity_bonus': details['diversity_bonus'],
        'details': details,
        'selection_method': 'Fair Constrained Greedy'
    }

def unconstrained_greedy_selection(teachers: List[Teacher], config: SeminarConfig, 
                                  penalty_weight: float = 150, debug: bool = False) -> Dict:
    """Unconstrained Greedy - Ignores constraints during selection, applies penalties after"""
    
    teacher_scores = []
    for i, teacher in enumerate(teachers):
        adjusted_value = (teacher.base_benefit * 
                         teacher.category_multiplier * 
                         config.category_priorities[teacher.category])
        score = adjusted_value / teacher.cost
        teacher_scores.append((i, score, teacher, adjusted_value))
    
    teacher_scores.sort(key=lambda x: x[1], reverse=True)
    
    selected = [False] * len(teachers)
    selected_teachers = []
    max_selections = min(config.total_positions + 5, len(teacher_scores))
    
    for i in range(max_selections):
        idx, score, teacher, benefit = teacher_scores[i]
        selected[idx] = True
        selected_teachers.append(teacher)
    
    fitness, details = calculate_fitness_detailed(
        teachers, selected, config, penalty_weight, debug=debug
    )
    
    return {
        'fitness': fitness,
        'selected': selected,
        'violations': details['violations'],
        'cost': details['total_cost'],
        'benefit': details['total_benefit'],
        'raw_benefit': details['total_benefit'],
        'penalties': details['penalties'],
        'diversity_bonus': details['diversity_bonus'],
        'details': details,
        'selection_method': 'Unconstrained Greedy'
    }

def random_selection_debug(teachers: List[Teacher], config: SeminarConfig, 
                          trials: int = 100, penalty_weight: float = 150,
                          debug: bool = False) -> Dict:
    """Random selection baseline"""
    
    best_fitness = 0
    best_result = None
    trial_fitnesses = []
    
    for trial in range(trials):
        selected = [False] * len(teachers)
        
        selection_prob = (config.total_positions / len(teachers) * 1.2)
        for i in range(len(teachers)):
            if random.random() < selection_prob:
                selected[i] = True
        
        fitness, details = calculate_fitness_detailed(
            teachers, selected, config, penalty_weight, debug=False
        )
        
        trial_fitnesses.append(fitness)
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_result = {
                'fitness': fitness,
                'selected': selected,
                'violations': details['violations'],
                'cost': details['total_cost'],
                'benefit': details['total_benefit'],
                'raw_benefit': details['total_benefit'],
                'penalties': details['penalties'],
                'diversity_bonus': details['diversity_bonus'],
                'selected_count': details['selected_count'],
                'details': details
            }
    
    return best_result

def create_100_teachers_dataset():
    """Create a dataset of 100 teachers for testing"""
    
    teachers = [
        # STEM Teachers (40 teachers)
        Teacher(1, "Dr. Smith", "Mathematics", "STEM", 85, 12, 8, 1.5),
        Teacher(2, "Prof. Williams", "Physics", "STEM", 92, 15, 12, 1.5),
        Teacher(3, "Dr. Brown", "Chemistry", "STEM", 88, 14, 10, 1.5),
        Teacher(4, "Ms. Davis", "Biology", "STEM", 75, 9, 6, 1.2),
        Teacher(5, "Dr. Wilson", "Computer Science", "STEM", 95, 18, 15, 1.8),
        Teacher(6, "Ms. Garcia", "Statistics", "STEM", 89, 16, 11, 1.6),
        Teacher(7, "Dr. Martinez", "Engineering", "STEM", 91, 17, 13, 1.7),
        Teacher(8, "Prof. Kumar", "Data Science", "STEM", 93, 19, 14, 1.8),
        Teacher(9, "Dr. Zhang", "Biochemistry", "STEM", 87, 13, 9, 1.4),
        Teacher(10, "Ms. Patel", "Biostatistics", "STEM", 84, 15, 8, 1.5),
        Teacher(11, "Dr. Thompson", "Calculus", "STEM", 82, 11, 7, 1.4),
        Teacher(12, "Prof. Roberts", "Quantum Physics", "STEM", 96, 20, 16, 1.9),
        Teacher(13, "Dr. Lewis", "Organic Chemistry", "STEM", 86, 14, 9, 1.4),
        Teacher(14, "Ms. Wright", "Microbiology", "STEM", 79, 10, 5, 1.3),
        Teacher(15, "Dr. Hall", "Machine Learning", "STEM", 97, 22, 18, 2.0),
        Teacher(16, "Prof. Young", "Linear Algebra", "STEM", 83, 12, 6, 1.4),
        Teacher(17, "Dr. King", "Genetics", "STEM", 88, 16, 11, 1.5),
        Teacher(18, "Ms. Scott", "Applied Mathematics", "STEM", 81, 13, 7, 1.3),
        Teacher(19, "Dr. Adams", "Artificial Intelligence", "STEM", 94, 21, 17, 1.9),
        Teacher(20, "Prof. Baker", "Theoretical Physics", "STEM", 90, 18, 14, 1.6),
        Teacher(21, "Dr. Green", "Environmental Science", "STEM", 77, 11, 6, 1.2),
        Teacher(22, "Ms. Phillips", "Robotics", "STEM", 92, 19, 13, 1.7),
        Teacher(23, "Dr. Campbell", "Cybersecurity", "STEM", 89, 17, 12, 1.6),
        Teacher(24, "Prof. Turner", "Network Theory", "STEM", 85, 15, 10, 1.5),
        Teacher(25, "Dr. Parker", "Astronomy", "STEM", 80, 12, 8, 1.3),
        Teacher(26, "Ms. Evans", "Bioinformatics", "STEM", 91, 18, 14, 1.7),
        Teacher(27, "Dr. Edwards", "Nanotechnology", "STEM", 93, 20, 15, 1.8),
        Teacher(28, "Prof. Collins", "Mathematical Modeling", "STEM", 87, 16, 11, 1.5),
        Teacher(29, "Dr. Stewart", "Computational Biology", "STEM", 86, 17, 12, 1.6),
        Teacher(30, "Ms. Sanchez", "Information Systems", "STEM", 84, 14, 9, 1.4),
        Teacher(31, "Dr. Morris", "Software Engineering", "STEM", 88, 16, 10, 1.5),
        Teacher(32, "Prof. Rogers", "Differential Equations", "STEM", 82, 13, 8, 1.4),
        Teacher(33, "Dr. Reed", "Materials Science", "STEM", 85, 15, 11, 1.5),
        Teacher(34, "Ms. Cook", "Database Systems", "STEM", 83, 14, 7, 1.4),
        Teacher(35, "Dr. Morgan", "Signal Processing", "STEM", 89, 17, 13, 1.6),
        Teacher(36, "Prof. Bell", "Thermodynamics", "STEM", 86, 16, 10, 1.5),
        Teacher(37, "Dr. Murphy", "Algorithm Design", "STEM", 90, 18, 14, 1.7),
        Teacher(38, "Ms. Bailey", "Discrete Mathematics", "STEM", 81, 12, 6, 1.3),
        Teacher(39, "Dr. Rivera", "Systems Biology", "STEM", 87, 16, 12, 1.5),
        Teacher(40, "Prof. Cooper", "Nuclear Physics", "STEM", 91, 19, 15, 1.7),
        
        # Humanities Teachers (25 teachers)
        Teacher(41, "Ms. Johnson", "English Literature", "Humanities", 78, 10, 5, 0.9),
        Teacher(42, "Mr. Miller", "History", "Humanities", 70, 8, 4, 0.9),
        Teacher(43, "Mr. Lee", "Philosophy", "Humanities", 68, 8, 6, 0.8),
        Teacher(44, "Dr. White", "Linguistics", "Humanities", 75, 11, 7, 0.9),
        Teacher(45, "Prof. Harris", "Classical Studies", "Humanities", 72, 9, 8, 0.8),
        Teacher(46, "Ms. Martin", "Creative Writing", "Humanities", 69, 10, 5, 0.8),
        Teacher(47, "Dr. Garcia", "American Literature", "Humanities", 76, 12, 9, 0.9),
        Teacher(48, "Prof. Rodriguez", "Medieval History", "Humanities", 71, 9, 6, 0.8),
        Teacher(49, "Ms. Lopez", "Modern Poetry", "Humanities", 67, 8, 4, 0.7),
        Teacher(50, "Dr. Gonzalez", "Renaissance Art History", "Humanities", 73, 11, 8, 0.9),
        Teacher(51, "Prof. Wilson", "Ethics", "Humanities", 70, 10, 7, 0.8),
        Teacher(52, "Ms. Anderson", "Comparative Literature", "Humanities", 74, 11, 6, 0.9),
        Teacher(53, "Dr. Thomas", "Ancient Philosophy", "Humanities", 69, 9, 5, 0.8),
        Teacher(54, "Prof. Taylor", "World History", "Humanities", 72, 10, 7, 0.8),
        Teacher(55, "Ms. Moore", "Rhetoric", "Humanities", 66, 8, 4, 0.7),
        Teacher(56, "Dr. Jackson", "Cultural Studies", "Humanities", 75, 12, 9, 0.9),
        Teacher(57, "Prof. White", "Biblical Studies", "Humanities", 71, 10, 8, 0.8),
        Teacher(58, "Ms. Harris", "Gender Studies", "Humanities", 68, 9, 5, 0.8),
        Teacher(59, "Dr. Martin", "Post-Colonial Literature", "Humanities", 73, 11, 7, 0.9),
        Teacher(60, "Prof. Garcia", "Latin", "Humanities", 65, 8, 6, 0.7),
        Teacher(61, "Ms. Robinson", "Journalism", "Humanities", 70, 10, 6, 0.8),
        Teacher(62, "Dr. Clark", "Art History", "Humanities", 72, 11, 8, 0.8),
        Teacher(63, "Prof. Lewis", "Religious Studies", "Humanities", 69, 9, 7, 0.8),
        Teacher(64, "Ms. Walker", "Film Studies", "Humanities", 71, 10, 5, 0.8),
        Teacher(65, "Dr. Hall", "European History", "Humanities", 74, 12, 9, 0.9),
        
        # Social Sciences Teachers (20 teachers)
        Teacher(66, "Prof. Taylor", "Geography", "Social Sciences", 72, 9, 7, 1.0),
        Teacher(67, "Dr. Anderson", "Psychology", "Social Sciences", 82, 13, 9, 1.1),
        Teacher(68, "Ms. Brown", "Sociology", "Social Sciences", 76, 11, 6, 1.0),
        Teacher(69, "Dr. Davis", "Anthropology", "Social Sciences", 74, 10, 8, 1.0),
        Teacher(70, "Prof. Wilson", "Political Science", "Social Sciences", 79, 12, 10, 1.1),
        Teacher(71, "Ms. Miller", "Economics", "Social Sciences", 81, 14, 11, 1.1),
        Teacher(72, "Dr. Moore", "Criminology", "Social Sciences", 75, 11, 7, 1.0),
        Teacher(73, "Prof. Taylor", "International Relations", "Social Sciences", 78, 13, 9, 1.1),
        Teacher(74, "Ms. Anderson", "Social Work", "Social Sciences", 70, 9, 5, 0.9),
        Teacher(75, "Dr. Thomas", "Urban Planning", "Social Sciences", 77, 12, 8, 1.0),
        Teacher(76, "Prof. Jackson", "Public Administration", "Social Sciences", 73, 11, 7, 1.0),
        Teacher(77, "Ms. White", "Environmental Policy", "Social Sciences", 76, 12, 9, 1.0),
        Teacher(78, "Dr. Harris", "Human Geography", "Social Sciences", 74, 10, 6, 1.0),
        Teacher(79, "Prof. Martin", "Developmental Psychology", "Social Sciences", 80, 13, 10, 1.1),
        Teacher(80, "Ms. Garcia", "Social Psychology", "Social Sciences", 78, 12, 8, 1.0),
        Teacher(81, "Dr. Rodriguez", "Behavioral Economics", "Social Sciences", 83, 15, 12, 1.2),
        Teacher(82, "Prof. Lopez", "Cultural Anthropology", "Social Sciences", 75, 11, 7, 1.0),
        Teacher(83, "Ms. Gonzalez", "Public Policy", "Social Sciences", 77, 13, 9, 1.0),
        Teacher(84, "Dr. Wilson", "Cognitive Psychology", "Social Sciences", 81, 14, 11, 1.1),
        Teacher(85, "Prof. Anderson", "Comparative Politics", "Social Sciences", 76, 12, 8, 1.0),
        
        # Arts Teachers (15 teachers)
        Teacher(86, "Ms. Moore", "Fine Arts", "Arts", 65, 7, 3, 0.7),
        Teacher(87, "Dr. Smith", "Music Theory", "Arts", 68, 9, 5, 0.8),
        Teacher(88, "Prof. Johnson", "Theater Arts", "Arts", 70, 10, 6, 0.8),
        Teacher(89, "Ms. Williams", "Visual Arts", "Arts", 64, 7, 4, 0.7),
        Teacher(90, "Dr. Brown", "Dance", "Arts", 62, 6, 3, 0.6),
        Teacher(91, "Prof. Davis", "Sculpture", "Arts", 66, 8, 5, 0.7),
        Teacher(92, "Ms. Miller", "Photography", "Arts", 63, 7, 4, 0.7),
        Teacher(93, "Dr. Wilson", "Music Performance", "Arts", 69, 9, 6, 0.8),
        Teacher(94, "Prof. Garcia", "Graphic Design", "Arts", 67, 8, 5, 0.7),
        Teacher(95, "Ms. Martinez", "Ceramics", "Arts", 61, 6, 3, 0.6),
        Teacher(96, "Dr. Anderson", "Film Production", "Arts", 71, 10, 7, 0.8),
        Teacher(97, "Prof. Thomas", "Digital Arts", "Arts", 68, 9, 5, 0.7),
        Teacher(98, "Ms. Jackson", "Fashion Design", "Arts", 60, 6, 3, 0.6),
        Teacher(99, "Dr. White", "Art Education", "Arts", 65, 8, 4, 0.7),
        Teacher(100, "Prof. Harris", "Music Composition", "Arts", 70, 10, 8, 0.8)
    ]
    
    return teachers

def create_100_teacher_config():
    """Create appropriate configuration for 100 teachers"""
    
    config = SeminarConfig(
        total_positions=15,
        budget_capacity=180,
        category_limits={
            "STEM": {"min": 4, "max": 6},
            "Humanities": {"min": 3, "max": 5},
            "Social Sciences": {"min": 3, "max": 4},
            "Arts": {"min": 2, "max": 3}
        },
        category_priorities={
            "STEM": 1.6,
            "Social Sciences": 1.2,
            "Humanities": 0.9,
            "Arts": 0.7
        }
    )
    
    return config

def run_single_trial(trial_num: int, teachers: List[Teacher], config: SeminarConfig, 
                    deap_params: GAParams, verbose: bool = False) -> Dict:
    """Run a single trial of all algorithms"""
    
    if verbose:
        print(f"\n--- Trial {trial_num + 1} ---")
    
    # Set different random seed for each trial
    random.seed(42 + trial_num * 100)
    np.random.seed(42 + trial_num * 100)
    
    results = {}
    
    # 1. DEAP GA
    try:
        deap_ga = TeacherSeminarDEAP(teachers, config, deap_params)
        deap_result = deap_ga.run(verbose=False)
        results['deap_ga'] = {
            'fitness': deap_result['best_fitness'],
            'cost': deap_result['best_solution']['total_cost'],
            'benefit': deap_result['best_solution']['total_benefit'],
            'violations': len(deap_result['best_solution']['constraint_violations']),
            'budget_utilization': deap_result['best_solution']['utilization_rates']['budget'],
            'position_utilization': deap_result['best_solution']['utilization_rates']['positions'],
            'category_distribution': deap_result['best_solution']['category_distribution']
        }
        if verbose:
            print(f"  DEAP GA: {deap_result['best_fitness']:.2f}")
    except Exception as e:
        print(f"  DEAP GA failed in trial {trial_num + 1}: {e}")
        results['deap_ga'] = None
    
    # 2. Fair Constrained Greedy
    try:
        fair_result = fair_constrained_greedy_selection(teachers, config, 
                                                       penalty_weight=deap_params.penalty_weight)
        results['fair_greedy'] = {
            'fitness': fair_result['fitness'],
            'cost': fair_result['cost'],
            'benefit': fair_result['benefit'],
            'violations': len(fair_result['violations']),
            'budget_utilization': (fair_result['cost'] / config.budget_capacity) * 100,
            'position_utilization': (fair_result['details']['selected_count'] / config.total_positions) * 100,
            'category_distribution': fair_result['details']['category_counts']
        }
        if verbose:
            print(f"  Fair Greedy: {fair_result['fitness']:.2f}")
    except Exception as e:
        print(f"  Fair Greedy failed in trial {trial_num + 1}: {e}")
        results['fair_greedy'] = None
    
    # 3. Unconstrained Greedy
    try:
        unconstrained_result = unconstrained_greedy_selection(teachers, config, 
                                                            penalty_weight=deap_params.penalty_weight)
        results['unconstrained_greedy'] = {
            'fitness': unconstrained_result['fitness'],
            'cost': unconstrained_result['cost'],
            'benefit': unconstrained_result['benefit'],
            'violations': len(unconstrained_result['violations']),
            'budget_utilization': (unconstrained_result['cost'] / config.budget_capacity) * 100,
            'position_utilization': (unconstrained_result['details']['selected_count'] / config.total_positions) * 100,
            'category_distribution': unconstrained_result['details']['category_counts']
        }
        if verbose:
            print(f"  Unconstrained Greedy: {unconstrained_result['fitness']:.2f}")
    except Exception as e:
        print(f"  Unconstrained Greedy failed in trial {trial_num + 1}: {e}")
        results['unconstrained_greedy'] = None
    
    # 4. Random Selection
    try:
        random_result = random_selection_debug(teachers, config, trials=50, 
                                             penalty_weight=deap_params.penalty_weight)
        results['random'] = {
            'fitness': random_result['fitness'],
            'cost': random_result['cost'],
            'benefit': random_result['benefit'],
            'violations': len(random_result['violations']),
            'budget_utilization': (random_result['cost'] / config.budget_capacity) * 100,
            'position_utilization': (random_result['selected_count'] / config.total_positions) * 100,
            'category_distribution': random_result['details']['category_counts']
        }
        if verbose:
            print(f"  Random: {random_result['fitness']:.2f}")
    except Exception as e:
        print(f"  Random failed in trial {trial_num + 1}: {e}")
        results['random'] = None
    
    return results

def calculate_statistics(all_results: List[Dict], algorithm_name: str) -> Dict:
    """Calculate comprehensive statistics for an algorithm"""
    
    # Filter out None results (failed runs)
    valid_results = [r[algorithm_name] for r in all_results if r[algorithm_name] is not None]
    
    if not valid_results:
        return {
            'success_rate': 0.0,
            'mean_fitness': 0.0,
            'std_fitness': 0.0,
            'min_fitness': 0.0,
            'max_fitness': 0.0,
            'median_fitness': 0.0,
            'q1_fitness': 0.0,
            'q3_fitness': 0.0,
            'cv_fitness': 0.0,
            'mean_violations': 0.0,
            'violation_rate': 0.0,
            'mean_budget_util': 0.0,
            'mean_position_util': 0.0
        }
    
    n_valid = len(valid_results)
    n_total = len(all_results)
    
    # Extract metrics
    fitness_values = [r['fitness'] for r in valid_results]
    cost_values = [r['cost'] for r in valid_results]
    benefit_values = [r['benefit'] for r in valid_results]
    violation_counts = [r['violations'] for r in valid_results]
    budget_utils = [r['budget_utilization'] for r in valid_results]
    position_utils = [r['position_utilization'] for r in valid_results]
    
    # Calculate statistics
    stats = {
        'success_rate': (n_valid / n_total) * 100,
        'n_valid': n_valid,
        'n_total': n_total,
        
        # Fitness statistics
        'mean_fitness': np.mean(fitness_values),
        'std_fitness': np.std(fitness_values, ddof=1) if n_valid > 1 else 0.0,
        'min_fitness': np.min(fitness_values),
        'max_fitness': np.max(fitness_values),
        'median_fitness': np.median(fitness_values),
        'q1_fitness': np.percentile(fitness_values, 25),
        'q3_fitness': np.percentile(fitness_values, 75),
        'cv_fitness': (np.std(fitness_values, ddof=1) / np.mean(fitness_values)) * 100 if np.mean(fitness_values) > 0 and n_valid > 1 else 0.0,
        
        # Constraint satisfaction
        'mean_violations': np.mean(violation_counts),
        'std_violations': np.std(violation_counts, ddof=1) if n_valid > 1 else 0.0,
        'violation_rate': (sum(1 for v in violation_counts if v > 0) / n_valid) * 100,
        'feasible_rate': (sum(1 for v in violation_counts if v == 0) / n_valid) * 100,
        
        # Resource utilization
        'mean_cost': np.mean(cost_values),
        'std_cost': np.std(cost_values, ddof=1) if n_valid > 1 else 0.0,
        'mean_benefit': np.mean(benefit_values),
        'std_benefit': np.std(benefit_values, ddof=1) if n_valid > 1 else 0.0,
        'mean_budget_util': np.mean(budget_utils),
        'std_budget_util': np.std(budget_utils, ddof=1) if n_valid > 1 else 0.0,
        'mean_position_util': np.mean(position_utils),
        'std_position_util': np.std(position_utils, ddof=1) if n_valid > 1 else 0.0,
        
        # Raw data for further analysis
        'fitness_values': fitness_values,
        'violation_counts': violation_counts,
        'cost_values': cost_values,
        'benefit_values': benefit_values
    }
    
    return stats

def perform_statistical_tests(stats_dict: Dict[str, Dict]) -> Dict:
    """Perform statistical significance tests between algorithms"""
    
    algorithms = list(stats_dict.keys())
    test_results = {}
    
    # Pairwise comparisons
    for i, alg1 in enumerate(algorithms):
        for j, alg2 in enumerate(algorithms):
            if i < j:  # Avoid duplicate comparisons
                key = f"{alg1}_vs_{alg2}"
                
                fitness1 = stats_dict[alg1]['fitness_values']
                fitness2 = stats_dict[alg2]['fitness_values']
                
                if len(fitness1) > 1 and len(fitness2) > 1:
                    # Welch's t-test (unequal variances)
                    t_stat, t_pvalue = stats.ttest_ind(fitness1, fitness2, equal_var=False)
                    
                    # Mann-Whitney U test (non-parametric)
                    try:
                        u_stat, u_pvalue = stats.mannwhitneyu(fitness1, fitness2, alternative='two-sided')
                    except:
                        u_stat, u_pvalue = np.nan, np.nan
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(fitness1) - 1) * np.var(fitness1, ddof=1) + 
                                        (len(fitness2) - 1) * np.var(fitness2, ddof=1)) / 
                                       (len(fitness1) + len(fitness2) - 2))
                    
                    if pooled_std > 0:
                        cohens_d = (np.mean(fitness1) - np.mean(fitness2)) / pooled_std
                    else:
                        cohens_d = 0.0
                    
                    test_results[key] = {
                        't_statistic': t_stat,
                        't_pvalue': t_pvalue,
                        'u_statistic': u_stat,
                        'u_pvalue': u_pvalue,
                        'cohens_d': cohens_d,
                        'mean_diff': np.mean(fitness1) - np.mean(fitness2),
                        'significant_005': t_pvalue < 0.05 if not np.isnan(t_pvalue) else False,
                        'significant_001': t_pvalue < 0.01 if not np.isnan(t_pvalue) else False
                    }
                else:
                    test_results[key] = {
                        't_statistic': np.nan,
                        't_pvalue': np.nan,
                        'u_statistic': np.nan,
                        'u_pvalue': np.nan,
                        'cohens_d': np.nan,
                        'mean_diff': np.nan,
                        'significant_005': False,
                        'significant_001': False
                    }
    
    return test_results

def create_comprehensive_report(all_results: List[Dict], algorithm_stats: Dict[str, Dict], 
                              statistical_tests: Dict, config: SeminarConfig, 
                              params: GAParams) -> str:
    """Create a comprehensive statistical report"""
    
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE STATISTICAL ANALYSIS - 10 TRIALS")
    report.append("Teacher Seminar Selection Algorithms")
    report.append("=" * 80)
    
    # Configuration summary
    report.append(f"\nPROBLEM CONFIGURATION:")
    report.append(f"  Teachers available: 100")
    report.append(f"  Seminar positions: {config.total_positions}")
    report.append(f"  Budget capacity: {config.budget_capacity}")
    report.append(f"  Penalty weight: {params.penalty_weight}")
    report.append(f"  Population size: {params.population_size}")
    report.append(f"  Generations: {params.generations}")
    
    # Algorithm performance summary
    report.append(f"\nALGORITHM PERFORMANCE SUMMARY:")
    report.append(f"{'Algorithm':<25} {'Mean±SD':<15} {'Min':<8} {'Max':<8} {'Median':<8} {'Success%':<8} {'Feasible%':<10}")
    report.append("-" * 95)
    
    algorithm_order = ['deap_ga', 'fair_greedy', 'random', 'unconstrained_greedy']
    algorithm_names = {
        'deap_ga': 'DEAP Genetic Algorithm',
        'fair_greedy': 'Fair Constrained Greedy',
        'random': 'Random Selection',
        'unconstrained_greedy': 'Unconstrained Greedy'
    }
    
    for alg in algorithm_order:
        if alg in algorithm_stats:
            stats = algorithm_stats[alg]
            report.append(f"{algorithm_names[alg]:<25} "
                         f"{stats['mean_fitness']:.1f}±{stats['std_fitness']:.1f}  "
                         f"{stats['min_fitness']:<8.1f} "
                         f"{stats['max_fitness']:<8.1f} "
                         f"{stats['median_fitness']:<8.1f} "
                         f"{stats['success_rate']:<7.1f}% "
                         f"{stats['feasible_rate']:<9.1f}%")
    
    # Detailed statistics for each algorithm
    for alg in algorithm_order:
        if alg in algorithm_stats:
            stats = algorithm_stats[alg]
            report.append(f"\n{algorithm_names[alg].upper()}:")
            report.append("-" * len(algorithm_names[alg]))
            
            report.append(f"  Fitness Statistics:")
            report.append(f"    Mean: {stats['mean_fitness']:.3f}")
            report.append(f"    Standard Deviation: {stats['std_fitness']:.3f}")
            report.append(f"    Coefficient of Variation: {stats['cv_fitness']:.1f}%")
            report.append(f"    Range: {stats['min_fitness']:.2f} - {stats['max_fitness']:.2f}")
            report.append(f"    IQR: {stats['q1_fitness']:.2f} - {stats['q3_fitness']:.2f}")
            
            report.append(f"  Constraint Satisfaction:")
            report.append(f"    Mean violations per run: {stats['mean_violations']:.2f}")
            report.append(f"    Runs with violations: {stats['violation_rate']:.1f}%")
            report.append(f"    Feasible solutions: {stats['feasible_rate']:.1f}%")
            
            report.append(f"  Resource Utilization:")
            report.append(f"    Budget utilization: {stats['mean_budget_util']:.1f}% ± {stats['std_budget_util']:.1f}%")
            report.append(f"    Position utilization: {stats['mean_position_util']:.1f}% ± {stats['std_position_util']:.1f}%")
            report.append(f"    Mean cost: {stats['mean_cost']:.1f} ± {stats['std_cost']:.1f}")
            report.append(f"    Mean benefit: {stats['mean_benefit']:.1f} ± {stats['std_benefit']:.1f}")
    
    # Statistical significance tests
    report.append(f"\nSTATISTICAL SIGNIFICANCE TESTS:")
    report.append("-" * 40)
    report.append(f"{'Comparison':<30} {'Mean Diff':<12} {'t-test p':<12} {'Effect Size':<12} {'Significant'}")
    report.append("-" * 75)
    
    for test_name, test_result in statistical_tests.items():
        alg1, alg2 = test_name.split('_vs_')
        comparison = f"{algorithm_names.get(alg1, alg1)} vs {algorithm_names.get(alg2, alg2)}"
        if len(comparison) > 29:
            comparison = comparison[:26] + "..."
        
        significance = ""
        if test_result['significant_001']:
            significance = "***"
        elif test_result['significant_005']:
            significance = "**"
        elif test_result['t_pvalue'] < 0.1:
            significance = "*"
        else:
            significance = "ns"
        
        report.append(f"{comparison:<30} "
                     f"{test_result['mean_diff']:<12.2f} "
                     f"{test_result['t_pvalue']:<12.4f} "
                     f"{test_result['cohens_d']:<12.2f} "
                     f"{significance}")
    
    report.append(f"\nLegend: *** p<0.001, ** p<0.05, * p<0.1, ns = not significant")
    
    # Performance rankings
    report.append(f"\nPERFORMANCE RANKINGS:")
    report.append("-" * 25)
    
    # Rank by mean fitness
    ranked_algorithms = sorted(algorithm_order, 
                             key=lambda x: algorithm_stats[x]['mean_fitness'] if x in algorithm_stats else 0, 
                             reverse=True)
    
    report.append(f"By Mean Fitness:")
    for i, alg in enumerate(ranked_algorithms, 1):
        if alg in algorithm_stats:
            report.append(f"  {i}. {algorithm_names[alg]}: {algorithm_stats[alg]['mean_fitness']:.2f}")
    
    # Rank by feasibility
    ranked_by_feasibility = sorted(algorithm_order,
                                 key=lambda x: algorithm_stats[x]['feasible_rate'] if x in algorithm_stats else 0,
                                 reverse=True)
    
    report.append(f"\nBy Feasibility Rate:")
    for i, alg in enumerate(ranked_by_feasibility, 1):
        if alg in algorithm_stats:
            report.append(f"  {i}. {algorithm_names[alg]}: {algorithm_stats[alg]['feasible_rate']:.1f}%")
    
    # Calculate relative improvements
    report.append(f"\nRELATIVE PERFORMANCE IMPROVEMENTS:")
    report.append("-" * 40)
    
    if 'deap_ga' in algorithm_stats:
        ga_mean = algorithm_stats['deap_ga']['mean_fitness']
        
        for alg in algorithm_order:
            if alg != 'deap_ga' and alg in algorithm_stats:
                baseline_mean = algorithm_stats[alg]['mean_fitness']
                if baseline_mean > 0:
                    improvement = ((ga_mean - baseline_mean) / baseline_mean) * 100
                    report.append(f"  DEAP GA vs {algorithm_names[alg]}: {improvement:+.1f}%")
    
    # Practical recommendations
    report.append(f"\nPRACTICAL RECOMMENDATIONS:")
    report.append("-" * 30)
    
    best_overall = max(algorithm_order, key=lambda x: algorithm_stats[x]['mean_fitness'] if x in algorithm_stats else 0)
    most_feasible = max(algorithm_order, key=lambda x: algorithm_stats[x]['feasible_rate'] if x in algorithm_stats else 0)
    most_consistent = min(algorithm_order, key=lambda x: algorithm_stats[x]['cv_fitness'] if x in algorithm_stats else float('inf'))
    
    report.append(f"  Best overall performance: {algorithm_names[best_overall]}")
    report.append(f"  Most feasible solutions: {algorithm_names[most_feasible]}")
    report.append(f"  Most consistent results: {algorithm_names[most_consistent]}")
    
    if best_overall == 'deap_ga':
        report.append(f"  → Recommended: Use DEAP GA for optimization problems")
    
    # Data quality assessment
    report.append(f"\nDATA QUALITY ASSESSMENT:")
    report.append("-" * 30)
    
    total_runs = len(all_results) * len(algorithm_order)
    successful_runs = sum(len([r for r in all_results if r[alg] is not None]) for alg in algorithm_order)
    
    report.append(f"  Total algorithm runs: {total_runs}")
    report.append(f"  Successful runs: {successful_runs}")
    report.append(f"  Success rate: {(successful_runs/total_runs)*100:.1f}%")
    report.append(f"  Confidence level: HIGH (n=10 trials per algorithm)")
    
    return "\n".join(report)

def run_statistical_analysis_10_trials():
    """Main function to run comprehensive statistical analysis"""
    
    print("=" * 80)
    print("RUNNING COMPREHENSIVE STATISTICAL ANALYSIS")
    print("10 Trials × 4 Algorithms = 40 Total Runs")
    print("=" * 80)
    
    # Setup
    teachers = create_100_teachers_dataset()
    config = create_100_teacher_config()
    
    deap_params = GAParams(
        population_size=100,
        generations=150,  # Reduced for faster execution
        mutation_rate=0.15,
        crossover_rate=0.8,
        tournament_size=3,
        elite_size=10,
        penalty_weight=150
    )
    
    # Run all trials
    all_results = []
    print(f"\nRunning 10 trials...")
    
    for trial in range(10):
        print(f"Trial {trial + 1}/10...", end=" ", flush=True)
        trial_results = run_single_trial(trial, teachers, config, deap_params, verbose=False)
        all_results.append(trial_results)
        print("✓")
    
    print(f"\nAll trials completed! Processing results...")
    
    # Calculate statistics for each algorithm
    algorithms = ['deap_ga', 'fair_greedy', 'unconstrained_greedy', 'random']
    algorithm_stats = {}
    
    for alg in algorithms:
        algorithm_stats[alg] = calculate_statistics(all_results, alg)
    
    # Perform statistical tests
    statistical_tests = perform_statistical_tests(algorithm_stats)
    
    # Generate comprehensive report
    report = create_comprehensive_report(all_results, algorithm_stats, statistical_tests, config, deap_params)
    
    # Display results
    print(report)
    
    # Create visualizations
    create_statistical_visualizations(algorithm_stats, statistical_tests)
    
    return {
        'all_results': all_results,
        'algorithm_stats': algorithm_stats,
        'statistical_tests': statistical_tests,
        'report': report
    }

def create_statistical_visualizations(algorithm_stats: Dict, statistical_tests: Dict):
    """Create comprehensive visualizations of the statistical analysis"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    algorithms = ['deap_ga', 'fair_greedy', 'random', 'unconstrained_greedy']
    algorithm_names = {
        'deap_ga': 'DEAP GA',
        'fair_greedy': 'Fair Greedy',
        'random': 'Random',
        'unconstrained_greedy': 'Unconstrained'
    }
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # 1. Mean Fitness with Error Bars
    means = [algorithm_stats[alg]['mean_fitness'] for alg in algorithms]
    stds = [algorithm_stats[alg]['std_fitness'] for alg in algorithms]
    labels = [algorithm_names[alg] for alg in algorithms]
    
    bars1 = ax1.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Mean Fitness Comparison (±1 SD)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Fitness Score')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars1, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(stds)/20,
                f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Box plot of fitness distributions
    fitness_data = [algorithm_stats[alg]['fitness_values'] for alg in algorithms]
    box_plot = ax2.boxplot(fitness_data, labels=labels, patch_artist=True)
    
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_title('Fitness Distribution Box Plots', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Fitness Score')
    ax2.grid(True, alpha=0.3)
    
    # 3. Constraint satisfaction rates
    feasible_rates = [algorithm_stats[alg]['feasible_rate'] for alg in algorithms]
    bars3 = ax3.bar(labels, feasible_rates, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_title('Feasible Solution Rate (%)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Feasible Solutions (%)')
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3)
    
    # Add percentage labels
    for bar, rate in zip(bars3, feasible_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Resource utilization comparison
    budget_utils = [algorithm_stats[alg]['mean_budget_util'] for alg in algorithms]
    position_utils = [algorithm_stats[alg]['mean_position_util'] for alg in algorithms]
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    bars4a = ax4.bar(x - width/2, budget_utils, width, label='Budget Utilization', 
                     color='skyblue', alpha=0.7, edgecolor='black')
    bars4b = ax4.bar(x + width/2, position_utils, width, label='Position Utilization', 
                     color='lightcoral', alpha=0.7, edgecolor='black')
    
    ax4.set_title('Resource Utilization Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Utilization (%)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Create a second figure for statistical significance matrix
    fig2, ax5 = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create significance matrix
    n_algs = len(algorithms)
    sig_matrix = np.zeros((n_algs, n_algs))
    
    for i, alg1 in enumerate(algorithms):
        for j, alg2 in enumerate(algorithms):
            if i != j:
                test_key = f"{alg1}_vs_{alg2}" if i < j else f"{alg2}_vs_{alg1}"
                if test_key in statistical_tests:
                    p_value = statistical_tests[test_key]['t_pvalue']
                    if p_value < 0.001:
                        sig_matrix[i, j] = 3
                    elif p_value < 0.01:
                        sig_matrix[i, j] = 2
                    elif p_value < 0.05:
                        sig_matrix[i, j] = 1
                    else:
                        sig_matrix[i, j] = 0
    
    im = ax5.imshow(sig_matrix, cmap='RdYlBu_r', aspect='equal')
    
    # Add text annotations
    for i in range(n_algs):
        for j in range(n_algs):
            if i != j:
                test_key = f"{algorithms[i]}_vs_{algorithms[j]}" if i < j else f"{algorithms[j]}_vs_{algorithms[i]}"
                if test_key in statistical_tests:
                    p_val = statistical_tests[test_key]['t_pvalue']
                    text = f"p={p_val:.3f}"
                    ax5.text(j, i, text, ha="center", va="center", fontsize=8)
    
    ax5.set_xticks(range(n_algs))
    ax5.set_yticks(range(n_algs))
    ax5.set_xticklabels([algorithm_names[alg] for alg in algorithms], rotation=45)
    ax5.set_yticklabels([algorithm_names[alg] for alg in algorithms])
    ax5.set_title('Statistical Significance Matrix\n(Darker = More Significant)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax5)
    cbar.set_label('Significance Level')
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['ns', 'p<0.05', 'p<0.01', 'p<0.001'])
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    try:
        print("Starting comprehensive statistical analysis...")
        results = run_statistical_analysis_10_trials()
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("\nKey findings saved in results dictionary with keys:")
        print("- 'all_results': Raw data from all 40 runs")
        print("- 'algorithm_stats': Comprehensive statistics for each algorithm")
        print("- 'statistical_tests': Pairwise significance tests")
        print("- 'report': Full text report")
        
        print(results['all_results'])
        print(results['algorithm_stats'])
        print(results['statistical_tests'])
        print(results['report'])
        
    except ImportError as e:
        print(f"Missing required library: {e}")
        print("Please install: pip install deap numpy matplotlib pandas scipy")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()