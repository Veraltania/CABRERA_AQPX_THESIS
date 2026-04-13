import pygad
import numpy as np
from Evolutionary_Algorithm_Testing.ea_optimizer import EvolutionaryOptimizer

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
            
            if raw_costs[0] >= 1e8:
                return -1e9 

            weighted_cost = sum(w * c for w, c in zip(self.weights, raw_costs))

            if weighted_cost < best_sol_tracker['cost']:
                best_sol_tracker['cost'] = weighted_cost
                best_sol_tracker['x'] = (Kp, Ki)
                best_sol_tracker['raw'] = raw_costs

            return 1.0 / (weighted_cost + 1e-6)

        class Tracker:
            def __init__(self, patience, tol):
                self.patience = patience
                self.tol = tol
                self.counter = 0
                self.best_fitness = -float('inf')
                self.history = []

            def callback(self, ga_instance):
                best_fitness = ga_instance.best_solution()[1]
                if best_fitness > (self.best_fitness + self.tol):
                    self.best_fitness = best_fitness
                    self.counter = 0
                else:
                    self.counter += 1
                
                if self.best_fitness <= 0:
                    self.history.append(1e9)
                else:
                    self.history.append((1.0 / self.best_fitness) - 1e-6)

                if self.counter >= self.patience:
                    return "stop"

        tracker = Tracker(self.patience, self.tol)
        
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
            on_generation=tracker.callback,
            suppress_warnings=True
        )

        ga_instance.run()
        iterations_run = ga_instance.generations_completed
        
        if best_sol_tracker['x'] is None:
            final_Kp, final_Ki = 0.0, 0.0
            best_sol_tracker['cost'] = 1e9
            best_sol_tracker['raw'] = (1e9, 1e9, 1e9, 1e9)
        else:
            final_Kp, final_Ki = best_sol_tracker['x']
            
        return (final_Kp, final_Ki, best_sol_tracker['cost'], best_sol_tracker['raw']), iterations_run, tracker.history