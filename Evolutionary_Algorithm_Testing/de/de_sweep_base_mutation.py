import os
import sys
import numpy as np
from de_optimizer import DEOptimizer
from Evolutionary_Algorithm_Testing.optimization_sweeper import OptimizationSweeper

if __name__ == '__main__':
    sweep_type = "de_sweep_mutation"
    SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

    # --- 1. TRANSFER FUNCTIONS ---
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

    # --- 2. BASE CONFIGURATION ---
    base_config = {
        'max_iters': 100,           
        'runs': 10,                 
        'patience': 20,             
        'tol': 1e-3,                
        'strategy': 'best1bin',
        'recombination': 0.75  # Set as requested for DE
    }

    # --- 3. SWEEP CONFIGURATION ---
    # Mutation for DE typically ranges from 0 to 2. 
    start_value = 0.1
    end_value = 2.0
    num_bins = 20
    
    dynamic_values = np.linspace(start_value, end_value, num_bins).tolist()
    
    print(f"Generated Sweep Values: {dynamic_values}")

    sweep_config = {
        'label': 'Mutation_Factor',
        'keys': ['mutation'],  # Maps directly to DEOptimizer's self.mutation extraction
        'values': dynamic_values, 
        'pop_sizes': [50]  
    }

    output_dir = os.path.join(SCRIPT_DIR, 'DE_SWEEP_MUTATION')

    # --- 4. INITIALIZE AND RUN SWEEPER ---
    sweeper = OptimizationSweeper(
        optimizer_class=DEOptimizer, # DEOptimizer natively takes float mutations via the config dict
        sweep_config=sweep_config,
        transfer_functions=transfer_functions,
        base_config=base_config,
        output_dir=output_dir
    )

    sweeper.run_sweep()