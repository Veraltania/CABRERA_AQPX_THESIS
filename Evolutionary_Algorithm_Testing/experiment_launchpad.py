from Evolutionary_Algorithm_Testing.de.de_optimizer import DEOptimizer
from Evolutionary_Algorithm_Testing.ga.ga_optimizer import GAOptimizer
from Evolutionary_Algorithm_Testing.pso.pso_optimizer import PSOOptimizer

if __name__ == "__main__":
    # --- 1. GLOBAL CONFIGURATION ---
    tf_params = {
        'tf_num': [44.93],
        'tf_den': [1474.32, 1],
        'tf_delay': 343.93,
        'tf_n_pade': 2
    }

    # Settings shared across ALL algorithms
    base_config = {
        'population_size': 100,
        'patience_limit': 25,
        'max_iters': 200,
        'improvement_tol': 0.01, # small tolerance since we are using log transforms
        'n_rounds': 50
    }

    # --- 2. ALGORITHM-SPECIFIC OVERRIDES ---
    algo_specific_configs = {
        "GA": {
            "num_parents_mating": 0.55,
            "keep_elitism": 0.05,
            "mutation_type": "adaptive",
            "crossover_type": "scattered",
        },
        "DE": {
            "mutation": (0.5, 1.0),
            "recombination": 0.7
        },
        "PSO": {
            "phi1": 2.5,
            "phi2": 2.5
        }
    }

    # --- 3. EXPERIMENT EXECUTION ---

    # Map string names directly to the classes
    ALGO_MAP = {
        'DE': DEOptimizer,
        'GA': GAOptimizer,
        'PSO': PSOOptimizer
    }

    algorithms_to_run = [
        "GA",
        # "DE",
        # "PSO"
    ]

    for algo_name in algorithms_to_run:
        print(f"\n{'#' * 50}\nINITIALIZING {algo_name} EXPERIMENT\n{'#' * 50}")

        # 1. Create a fresh copy of the base config
        run_config = base_config.copy()

        # 2. Inject specific overrides
        run_config.update(algo_specific_configs.get(algo_name, {}))

        # 3. Instantiate and run (Output folders auto-generate in the base class!)
        optimizer_class = ALGO_MAP[algo_name]
        optimizer = optimizer_class(run_config, tf_params)
        optimizer.run_experiment()