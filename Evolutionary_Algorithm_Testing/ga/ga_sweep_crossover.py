import os
import sys
import numpy as np
from ga_optimizer import GAOptimizer
from Evolutionary_Algorithm_Testing.optimization_sweeper import OptimizationSweeper

class CrossoverSweepGAOptimizer(GAOptimizer):
    def __init__(self, config, tf_params):
        pop_size = config.get('population_size', 50)
        
        raw_parents = config.get('num_parents_mating')
        if isinstance(raw_parents, float) and 0.0 < raw_parents <= 1.0:
            config['num_parents_mating'] = int(raw_parents * pop_size)
            
        super().__init__(config, tf_params)
        
        if 'crossover_probability' in config:
            self.crossover_probability = config['crossover_probability']

if __name__ == '__main__':
    sweep_type = "ga_sweep_crossover"
    SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

    transfer_functions = {
        f"{sweep_type}_do_feb5_daytime": {
            'tf_num': [1.346], 'tf_den': [1551.955, 1], 'tf_delay': 0.0, 'tf_n_pade': 2, 'computed_delay': 0.05, 
            'is_reverse_acting': False, 'min_kp': 0.001, 'max_kp': 100.0, 'min_ki': 1e-6, 'max_ki': 0.05
        },
        f"{sweep_type}_do_feb7_daytime": {
            'tf_num': [1.133], 'tf_den': [2833.82, 1], 'tf_delay': 0.0, 'tf_n_pade': 2, 'computed_delay': 0.05, 
            'is_reverse_acting': False, 'min_kp': 0.001, 'max_kp': 100.0, 'min_ki': 1e-6, 'max_ki': 0.05
        },
        f"{sweep_type}_do_feb25_daytime": {
            'tf_num': [2.287], 'tf_den': [3010.296, 1], 'tf_delay': 0.0, 'tf_n_pade': 2, 'computed_delay': 0.05, 
            'is_reverse_acting': False, 'min_kp': 0.001, 'max_kp': 100.0, 'min_ki': 1e-6, 'max_ki': 0.05
        },
        f"{sweep_type}_do_feb26_daytime": {
            'tf_num': [2.430], 'tf_den': [3492.589, 1], 'tf_delay': 0.0, 'tf_n_pade': 2, 'computed_delay': 0.05, 
            'is_reverse_acting': False, 'min_kp': 0.001, 'max_kp': 100.0, 'min_ki': 1e-6, 'max_ki': 0.05
        },
        f"{sweep_type}_do_feb5_nighttime": {
            'tf_num': [2.355], 'tf_den': [3083.590, 1], 'tf_delay': 0.0, 'tf_n_pade': 2, 'computed_delay': 0.05, 
            'is_reverse_acting': False, 'min_kp': 0.001, 'max_kp': 100.0, 'min_ki': 1e-6, 'max_ki': 0.05
        },
        f"{sweep_type}_do_feb7_nighttime": {
            'tf_num': [2.049], 'tf_den': [4499.996, 1], 'tf_delay': 0.0, 'tf_n_pade': 2, 'computed_delay': 0.05, 
            'is_reverse_acting': False, 'min_kp': 0.001, 'max_kp': 100.0, 'min_ki': 1e-6, 'max_ki': 0.05
        },
        f"{sweep_type}_do_feb25_nighttime": {
            'tf_num': [3.923], 'tf_den': [3012.232, 1], 'tf_delay': 0.0, 'tf_n_pade': 2, 'computed_delay': 0.05, 
            'is_reverse_acting': False, 'min_kp': 0.001, 'max_kp': 100.0, 'min_ki': 1e-6, 'max_ki': 0.05
        },
        f"{sweep_type}_do_feb26_nighttime": {
            'tf_num': [3.132], 'tf_den': [2530.052, 1], 'tf_delay': 0, 'tf_n_pade': 2, 'computed_delay': 0.05, 
            'is_reverse_acting': False, 'min_kp': 0.001, 'max_kp': 100.0, 'min_ki': 1e-6, 'max_ki': 0.05
        },
        f"{sweep_type}_tds_feb09_10": {
            'tf_num': [-21.082], 'tf_den': [71160.91, 1], 'tf_delay': 0, 'tf_n_pade': 2, 'computed_delay': 0.05, 
            'is_reverse_acting': True, 'min_kp': -100.0, 'max_kp': -0.001, 'min_ki': -0.05, 'max_ki': -1e-6
        },
        f"{sweep_type}_tds_feb10_11": {
            'tf_num': [-15.519], 'tf_den': [40156.08, 1], 'tf_delay': 0, 'tf_n_pade': 2, 'computed_delay': 0.05, 
            'is_reverse_acting': True, 'min_kp': -100.0, 'max_kp': -0.001, 'min_ki': -0.05, 'max_ki': -1e-6
        },
        f"{sweep_type}_tds_feb11_12": {
            'tf_num': [-12.458], 'tf_den': [16825.29, 1], 'tf_delay': 0, 'tf_n_pade': 2, 'computed_delay': 0.05, 
            'is_reverse_acting': True, 'min_kp': -100.0, 'max_kp': -0.001, 'min_ki': -0.05, 'max_ki': -1e-6
        }
    }

    base_config = {
        'max_iters': 100,           
        'runs': 10,                 
        'patience': 20,             
        'tol': 1e-3,                
        'num_parents_mating': 0.5,  
        'parent_selection_type': "rank",
        'keep_elitism': 2,          
        'crossover_type': "scattered",
        'weights': [1.0, 1.0, 1.0, 1.0] # Custom fitness weights
    }

    start_value = 0.0
    end_value = 1.0
    num_bins = 11  
    
    dynamic_values = np.linspace(start_value, end_value, num_bins).tolist()
    print(f"Generated Sweep Values: {dynamic_values}")

    sweep_config = {
        'label': 'Crossover_Probability',
        'keys': ['crossover_probability'],
        'values': dynamic_values, 
        'pop_sizes': [50]  
    }

    output_dir = os.path.join(SCRIPT_DIR, 'GA_SWEEP_CROSSOVER')

    sweeper = OptimizationSweeper(
        optimizer_class=CrossoverSweepGAOptimizer, 
        sweep_config=sweep_config,
        transfer_functions=transfer_functions,
        base_config=base_config,
        output_dir=output_dir
    )

    sweeper.run_sweep()