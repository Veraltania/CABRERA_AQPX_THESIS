import os
import sys
import numpy as np
from de_optimizer import DEOptimizer
from Evolutionary_Algorithm_Testing.optimization_sweeper import OptimizationSweeper

# --- 1. THE WRAPPER CLASS ---
class RecombinationSweepDEOptimizer(DEOptimizer):
    def __init__(self, config, tf_params):
        # Initialize the parent class
        super().__init__(config, tf_params)
        
        # The base DEOptimizer already extracts config.get('recombination', 0.745)
        # This explicit override ensures the sweep value is strictly enforced.
        if 'recombination' in config:
            self.recombination = config['recombination']


if __name__ == '__main__':
    sweep_type = "de_sweep_recombination"
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
        # DE Specific Parameters
        'mutation': (0.5, 1.0),     # Standard DE dithering mutation bounds
        'strategy': 'best1bin',     # Standard robust DE strategy
        'recombination': 0.75       # Set base to 0.75 as requested (will be overwritten by the sweep bins)
    }

    # --- 4. CONFIGURE SWEEP PARAMETERS ---
    # Recombination (CR) scales from 0.0 to 1.0. 
    # 11 bins will generate steps of 0.1: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    start_value = 0.0
    end_value = 1.0
    num_bins = 11  
    
    dynamic_values = np.linspace(start_value, end_value, num_bins).tolist()
    
    print(f"Generated Sweep Values: {dynamic_values}")

    sweep_config = {
        'label': 'Recombination',
        'keys': ['recombination'],
        'values': dynamic_values, 
        'pop_sizes': [50]  
    }

    output_dir = os.path.join(SCRIPT_DIR, 'DE_SWEEP_RECOMBINATION')

    # --- 5. INITIALIZE AND RUN SWEEPER ---
    sweeper = OptimizationSweeper(
        optimizer_class=RecombinationSweepDEOptimizer, 
        sweep_config=sweep_config,
        transfer_functions=transfer_functions,
        base_config=base_config,
        output_dir=output_dir
    )

    sweeper.run_sweep()