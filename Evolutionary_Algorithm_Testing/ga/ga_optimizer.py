import pygad
import numpy as np
from Evolutionary_Algorithm_Testing.ea_optimizer import EvolutionaryOptimizer


class _CostConvergenceTracker:
    """Track one cost-based patience rule shared by all optimizers."""

    def __init__(self, patience, tolerance, penalty_cost):
        self.patience = patience
        self.tolerance = tolerance
        self.penalty_cost = penalty_cost
        self.counter = 0
        self.best_cost = float('inf')
        self.history = []

    def update(self, current_best):
        if not np.isfinite(current_best):
            current_best = self.penalty_cost

        if current_best < (self.best_cost - self.tolerance):
            self.best_cost = current_best
            self.counter = 0
        else:
            self.counter += 1

        self.history.append(self.best_cost)
        return self.counter >= self.patience


class GAOptimizer(EvolutionaryOptimizer):
    def __init__(self, config, tf_params):
        super().__init__(config, tf_params)
        self.num_parents_mating = config.get('num_parents_mating', 10)
        self.parent_selection_type = config.get('parent_selection_type', "rank")
        self.keep_elitism = config.get('keep_elitism', 2)
        self.crossover_type = config.get('crossover_type', "scattered")
        self.crossover_probability = 0.90

    def optimize_round(self, round_num):
        best_sol_tracker = {'x': None, 'cost': float('inf'), 'raw': None}

        def fitness_wrapper(ga_instance, solution, solution_idx):
            Kp, Ki = solution[0], solution[1]
            raw_costs = self.calculate_cost(Kp, Ki)

            if self.is_penalty_costs(raw_costs):
                return -self.scalar_penalty

            weighted_cost = self.weighted_cost(raw_costs)

            if weighted_cost < best_sol_tracker['cost']:
                best_sol_tracker['cost'] = weighted_cost
                best_sol_tracker['x'] = (Kp, Ki)
                best_sol_tracker['raw'] = raw_costs

            return 1.0 / (weighted_cost + 1e-6)

        tracker = _CostConvergenceTracker(
            self.patience, self.tol, self.scalar_penalty
        )

        def convergence_callback(ga_instance):
            # Use the directly evaluated scalar cost. Inverting reciprocal
            # fitness can change strict comparisons at the tolerance boundary.
            if tracker.update(best_sol_tracker['cost']):
                return "stop"
        
        # Applying boundaries extracted from config in base class
        bounds = [{'low': self.min_kp, 'high': self.max_kp}, {'low': self.min_ki, 'high': self.max_ki}]

        ga_instance = pygad.GA(
            num_generations=self.max_iters,
            num_parents_mating=self.num_parents_mating,
            fitness_func=fitness_wrapper,
            sol_per_pop=self.pop_size,
            num_genes=2,
            gene_space=bounds,
            parent_selection_type=self.parent_selection_type,
            keep_elitism=self.keep_elitism,
            crossover_type=self.crossover_type,
            crossover_probability=self.crossover_probability,
            on_generation=convergence_callback,
            suppress_warnings=True
        )

        ga_instance.run()
        iterations_run = ga_instance.generations_completed
        
        if best_sol_tracker['x'] is None:
            final_Kp, final_Ki = 0.0, 0.0
            best_sol_tracker['cost'] = self.scalar_penalty
            best_sol_tracker['raw'] = self.penalty_costs
        else:
            final_Kp, final_Ki = best_sol_tracker['x']
            
        return (final_Kp, final_Ki, best_sol_tracker['cost'], best_sol_tracker['raw']), iterations_run, tracker.history
