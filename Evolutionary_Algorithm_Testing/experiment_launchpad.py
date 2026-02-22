from Evolutionary_Algorithm_Testing.de.de_optimizer import DEOptimizer
from Evolutionary_Algorithm_Testing.ga.ga_optimizer import GAOptimizer
from Evolutionary_Algorithm_Testing.pso.pso_optimizer import PSOOptimizer

if __name__ == "__main__":
    # Global Configuration
    tf_params = {'tf_num': [44.93], 'tf_den': [1474.32, 1], 'tf_delay': 343.93, 'tf_n_pade': 2}
    base_config = {'population_size': 100, 'patience_limit': 25, 'max_iters': 200, 'improvement_tol': 1.0,
                   'n_rounds': 50}

    # Run DE Experiment
    # de_config = base_config.copy()
    # de_config['output_folder'] = "experiment_images_de_population_100"
    #
    # de_exp = DEOptimizer(de_config, tf_params)
    # de_exp.run_experiment()

    # Run GA Experiment
    ga_config = base_config.copy()
    ga_config['output_folder'] = "experiment_images_ga_population_100"

    ga_exp = GAOptimizer(ga_config, tf_params)
    ga_exp.run_experiment()
    #
    # # Run PSO Experiment
    # pso_config = base_config.copy()
    # pso_config['output_folder'] = "experiment_images_pso_population_100"
    #
    # pso_exp = PSOOptimizer(pso_config, tf_params)
    # pso_exp.run_experiment()