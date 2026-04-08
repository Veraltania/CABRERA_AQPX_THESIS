import os
import sys
import numpy as np
import pygad
from ga_optimizer import GAOptimizer
from Evolutionary_Algorithm_Testing.optimization_sweeper import OptimizationSweeper

class BaseMutationSweepGAOptimizer(GAOptimizer):
    def __init__(self, config, tf_params):
        pop_size = config.get('population_size', 50)
        
        # Convert num_parents_mating if it's a percentage
        raw_parents = config.get('num_parents_mating')
        if isinstance(raw_parents, float) and 0.0 < raw_parents <= 1.0:
            config['num_parents_mating'] = int(raw_parents * pop_size)
            
        # Convert keep_elitism if it's a percentage
        raw_elitism = config.get('keep_elitism')
        if isinstance(raw_elitism, float) and 0.0 < raw_elitism <= 1.0:
            # Calculate percentage, but ensure it's at least 1 so elitism isn't completely lost
            config['keep_elitism'] = max(1, int(raw_elitism * pop_size))
            
        # Initialize the parent class
        super().__init__(config, tf_params)
        
        # Extract the dynamic base mutation value injected by the sweeper
        self.base_mutation = config.get('base_mutation', 0.1)

    # OVERRIDE optimize_round to inject the custom base mutation into the decay formula
    def optimize_round(self, round_num):
        def cost_wrapper(solution):
            return self.calculate_cost(solution[0], solution[1])

        class Tracker:
            def __init__(self, patience, tol, cost_func, base_mutation):
                self.patience = patience
                self.tol = tol
                self.cost_func = cost_func
                self.counter = 0
                self.best_cost = float('inf')
                self.history = []
                self.base_mutation = base_mutation

            def on_generation(self, ga_instance):
                # Apply dynamic Mutation factor using the SWEPT base mutation
                n = ga_instance.generations_completed
                P = ga_instance.num_generations
                
                # Formula: Base - 0.01 * (n/P). Bounded at 0.0 to prevent negative probabilities crashing PyGAD
                ga_instance.mutation_probability = max(0.0, self.base_mutation - 0.01 * (n / P))

                # Evaluate best solution
                solution, _, _ = ga_instance.best_solution()
                cost = self.cost_func(solution)
                self.history.append(cost)

                if cost < 1e8:
                    if cost < (self.best_cost - self.tol):
                        self.best_cost = cost
                        self.counter = 0
                    else:
                        self.counter += 1

                if self.counter >= self.patience:
                    return "stop"

        def fitness_wrapper(ga_instance, solution, solution_idx):
            cost = cost_wrapper(solution)
            return float(1.0 / (cost + 1e-8))

        # Pass our custom base_mutation into the new Tracker
        tracker = Tracker(self.patience, self.tol, cost_wrapper, self.base_mutation)

        if self.max_kp is None:
            safe_limit = -100.0 if self.is_reverse_acting else 100.0
        else:
            safe_limit = float(self.max_kp)

        if self.is_reverse_acting:
            min_kp, max_kp = safe_limit, -0.001
            min_ki, max_ki = -0.01, -1e-6
        else:
            min_kp, max_kp = 0.001, safe_limit
            min_ki, max_ki = 1e-6, 0.01

        bounds = [{'low': min_kp, 'high': max_kp}, {'low': min_ki, 'high': max_ki}]

        ga_kwargs = {
            "num_generations": self.max_iters,
            "num_parents_mating": self.num_parents_mating,
            "fitness_func": fitness_wrapper,
            "sol_per_pop": self.pop_size,
            "num_genes": 2,
            "gene_space": bounds,
            "parent_selection_type": self.parent_selection_type,
            "keep_elitism": self.keep_elitism,
            "crossover_type": self.crossover_type,
            "crossover_probability": self.crossover_probability,  
            "mutation_type": "random",  
            "mutation_probability": self.base_mutation,  # Set initial Pm for Generation 0 based on sweep
            "on_generation": tracker.on_generation,
            "suppress_warnings": True
        }

        ga_instance = pygad.GA(**ga_kwargs)
        ga_instance.run()

        solution, _, _ = ga_instance.best_solution()
        final_cost = cost_wrapper(solution)

        return solution[0], solution[1], final_cost, len(tracker.history), tracker.history


if __name__ == '__main__':
    sweep_type = "ga_sweep_mutation"
    SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

    # --- 2. TRANSFER FUNCTIONS ---
    transfer_functions = {
        f"{sweep_type}_do_feb5_daytime": {
            'tf_num': [1.346], 'tf_den': [1551.955, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 'max_kp': 100.0
        },
        f"{sweep_type}_do_feb7_daytime": {
            'tf_num': [1.133], 'tf_den': [2833.82, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 'max_kp': 100.0
        },
        f"{sweep_type}_do_feb25_daytime": {
            'tf_num': [2.287], 'tf_den': [3010.296, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 'max_kp': 100.0
        },
        f"{sweep_type}_do_feb26_daytime": {
            'tf_num': [2.430], 'tf_den': [3492.589, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 'max_kp': 100.0
        },
        f"{sweep_type}_do_feb5_nighttime": {
            'tf_num': [2.355], 'tf_den': [3083.590, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 'max_kp': 100.0
        },
        f"{sweep_type}_do_feb7_nighttime": {
            'tf_num': [2.049], 'tf_den': [4499.996, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 'max_kp': 100.0
        },
        f"{sweep_type}_do_feb25_nighttime": {
            'tf_num': [3.923], 'tf_den': [3012.232, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 'max_kp': 100.0
        },
        f"{sweep_type}_do_feb26_nighttime": {
            'tf_num': [3.132], 'tf_den': [2530.052, 1], 'tf_delay': 0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 'max_kp': 100.0
        },
        f"{sweep_type}_tds_feb09_10": {
            'tf_num': [-21.082], 'tf_den': [71160.91, 1], 'tf_delay': 0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': True, 'max_kp': -100.0
        },
        f"{sweep_type}_tds_feb10_11": {
            'tf_num': [-15.519], 'tf_den': [40156.08, 1], 'tf_delay': 0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': True, 'max_kp': -100.0
        },
        f"{sweep_type}_tds_feb11_12": {
            'tf_num': [-12.458], 'tf_den': [16825.29, 1], 'tf_delay': 0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': True, 'max_kp': -100.0
        }
    }

    # --- 3. BASE CONFIGURATION ---
    base_config = {
        'max_iters': 100,           
        'runs': 10,                 
        'patience': 20,             
        'tol': 1e-3,                
        'num_parents_mating': 0.5,  
        'parent_selection_type': "rank",
        'keep_elitism': 0.05,    # percentage      
        'crossover_type': "scattered",
        'crossover_probability': 0.90 # Locking this to a static value while we sweep mutation
    }

    # Mutation is usually lower than crossover. We will sweep from 0.0 to 0.5 in steps of 0.05
    start_value = 0.0
    end_value = 0.5
    num_bins = 11  
    
    dynamic_values = np.linspace(start_value, end_value, num_bins).tolist()
    
    print(f"Generated Sweep Values: {dynamic_values}")

    sweep_config = {
        'label': 'Base_Mutation',
        'keys': ['base_mutation'],
        'values': dynamic_values, 
        'pop_sizes': [50]  
    }

    output_dir = os.path.join(SCRIPT_DIR, 'GA_SWEEP_BASE_MUTATION')

    # --- 5. INITIALIZE AND RUN SWEEPER ---
    sweeper = OptimizationSweeper(
        optimizer_class=BaseMutationSweepGAOptimizer, 
        sweep_config=sweep_config,
        transfer_functions=transfer_functions,
        base_config=base_config,
        output_dir=output_dir
    )

    sweeper.run_sweep()