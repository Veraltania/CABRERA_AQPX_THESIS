import os
import numpy as np
from Evolutionary_Algorithm_Testing.optimization_sweeper import OptimizationSweeper
from pso_optimizer import PSOOptimizer
import sys

if __name__ == "__main__":
    SWEEP_START = 0
    SWEEP_END = 1.0
    SWEEP_BINS = 11

    SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

    sweep_type = "pso_sweep_vmax"
    label = "Maximum Velocity"
    keys = ['v_max']
    folder_name = "PSO_SWEEP_VMAX"

    # --- DEFINING TRANSFER FUNCTIONS WITH SPECIFIC Kp & Ki BOUNDS ---
    max_kp_do = 3.0
    min_kp_do = 0
    max_ki_do = 0.05
    min_ki_do = 0

    # Adjusted limits for massive Tp (71,160s) to prevent integral windup
    max_kp_tds = 0 
    min_kp_tds = -1.5
    max_ki_tds = 0
    min_ki_tds = -0.0005 # Dropped from -0.05 to give the solver a chance

    transfer_functions = {
        f"{sweep_type}_do_feb5_daytime": {
            'tf_num': [1.346], 'tf_den': [1551.955, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 
            'min_kp': min_kp_do, 'max_kp': max_kp_do, 'min_ki': min_ki_do, 'max_ki': max_ki_do
        },
        f"{sweep_type}_do_feb7_daytime": {
            'tf_num': [1.133], 'tf_den': [2833.82, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 
            'min_kp': min_kp_do, 'max_kp': max_kp_do, 'min_ki': min_ki_do, 'max_ki': max_ki_do
        },
        f"{sweep_type}_do_feb25_daytime": {
            'tf_num': [2.287], 'tf_den': [3010.296, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 
            'min_kp': min_kp_do, 'max_kp': max_kp_do, 'min_ki': min_ki_do, 'max_ki': max_ki_do
        },
        f"{sweep_type}_do_feb26_daytime": {
            'tf_num': [2.430], 'tf_den': [3492.589, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 
            'min_kp': min_kp_do, 'max_kp': max_kp_do, 'min_ki': min_ki_do, 'max_ki': max_ki_do
        },
        f"{sweep_type}_do_feb5_nighttime": {
            'tf_num': [2.355], 'tf_den': [3083.590, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 
            'min_kp': min_kp_do, 'max_kp': max_kp_do, 'min_ki': min_ki_do, 'max_ki': max_ki_do
        },
        f"{sweep_type}_do_feb7_nighttime": {
            'tf_num': [2.049], 'tf_den': [4499.996, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 
            'min_kp': min_kp_do, 'max_kp': max_kp_do, 'min_ki': min_ki_do, 'max_ki': max_ki_do
        },
        f"{sweep_type}_do_feb25_nighttime": {
            'tf_num': [3.923], 'tf_den': [3012.232, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 
            'min_kp': min_kp_do, 'max_kp': max_kp_do, 'min_ki': min_ki_do, 'max_ki': max_ki_do
        },
        f"{sweep_type}_do_feb26_nighttime": {
            'tf_num': [3.132], 'tf_den': [2530.052, 1], 'tf_delay': 0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 
            'min_kp': min_kp_do, 'max_kp': max_kp_do, 'min_ki': min_ki_do, 'max_ki': max_ki_do
        },
        # Reverse-acting Systems:
        f"{sweep_type}_tds_feb09_10": {
            'tf_num': [-21.082], 'tf_den': [71160.91, 1], 'tf_delay': 0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': True, 
            'min_kp': min_kp_tds, 'max_kp': max_kp_tds, 'min_ki': min_ki_tds, 'max_ki': max_ki_tds
        },
        f"{sweep_type}_tds_feb10_11": {
            'tf_num': [-15.519], 'tf_den': [40156.08, 1], 'tf_delay': 0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': True, 
            'min_kp': min_kp_tds, 'max_kp': max_kp_tds, 'min_ki': min_ki_tds, 'max_ki': max_ki_tds
        },
        f"{sweep_type}_tds_feb11_12": {
            'tf_num': [-12.458], 'tf_den': [16825.29, 1], 'tf_delay': 0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': True, 
            'min_kp': min_kp_tds, 'max_kp': max_kp_tds, 'min_ki': min_ki_tds, 'max_ki': max_ki_tds
        }
    }
    
    # --- 3. BASE CONFIGURATION ---
    base_config = {
        'patience_limit': 25,
        'max_iters': 100,
        'improvement_tol': 0.01,
        'n_rounds': 50
    }

    pso_sweep_config = {
        'label': label,
        'keys': keys,
        'values': np.linspace(SWEEP_START, SWEEP_END, SWEEP_BINS),
        'pop_sizes': [50]
    }

    pso_sweeper = OptimizationSweeper(
        optimizer_class=PSOOptimizer,
        sweep_config=pso_sweep_config,
        transfer_functions=transfer_functions,
        base_config=base_config,
        output_dir=os.path.join(SCRIPT_DIR, folder_name)
    )
    pso_sweeper.run_sweep()