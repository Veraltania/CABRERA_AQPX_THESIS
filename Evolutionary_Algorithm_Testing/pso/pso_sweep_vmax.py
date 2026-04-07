import os
import csv
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Evolutionary_Algorithm_Testing.pso.pso_optimizer import PSOOptimizer

# --- WORKER FUNCTION FOR MULTIPROCESSING ---
# Must be defined at the top level so it can be pickled across processes
def worker_run_pso_vmax(args):
    vmax_val, vmax_label, pop_size, tf_params, base_config, master_sweep_dir = args

    run_config = base_config.copy()
    run_config['population_size'] = pop_size
    
    # --- PSO SPECIFIC HYPERPARAMETERS ---
    run_config['v_min'] = -1.0
    run_config['v_max'] = vmax_val
    run_config['w'] = 0.9
    run_config['c1'] = 2.0
    run_config['c2'] = 2.0

    bin_folder_name = os.path.join(master_sweep_dir, f"bin_vmax_{vmax_label}")
    run_config['output_folder'] = bin_folder_name

    # Instantiate and run the optimized code
    optimizer = PSOOptimizer(run_config, tf_params)
    optimizer.run_experiment()

    # Extract the 50 trial costs AND iterations from this bin's history
    bin_costs = optimizer.agg_history['costs']
    bin_iters = optimizer.agg_history['iterations']

    # --- RAW FITNESS HISTORY EXTRACTION & PADDING ---
    # Retrieve the raw fitness histories (5th returned element from optimize_round)
    raw_histories = optimizer.agg_history.get('histories', [])
    avg_convergence_curve = None
    
    if raw_histories:
        padded_histories = []
        max_len = run_config['max_iters']
        
        for h in raw_histories:
            h_list = list(h)
            if len(h_list) == 0:
                continue
                
            # Pad early-stopped trials with their last known best cost
            if len(h_list) < max_len:
                h_list.extend([h_list[-1]] * (max_len - len(h_list)))
            # Truncate if it somehow exceeds max_iters
            elif len(h_list) > max_len:
                h_list = h_list[:max_len]
                
            padded_histories.append(h_list)
            
        if padded_histories:
            # Average the curves across the 50 trials for this specific Vmax bin
            avg_convergence_curve = np.mean(padded_histories, axis=0)

    # --- COST STATISTICS ---
    min_cost = np.min(bin_costs)
    max_cost = np.max(bin_costs)
    mean_cost = np.mean(bin_costs)
    std_cost = np.std(bin_costs)

    # --- ITERATION STATISTICS ---
    min_iter = np.min(bin_iters)
    max_iter = np.max(bin_iters)
    mean_iter = np.mean(bin_iters)
    std_iter = np.std(bin_iters)

    # Dictionaries representing this specific bin's statistics
    bin_stats_cost = {
        'Population': pop_size,
        'Vmax_Value': vmax_label,
        'Min_Cost': min_cost,
        'Max_Cost': max_cost,
        'Avg_Cost': mean_cost,
        'Std_Cost': std_cost
    }

    bin_stats_iter = {
        'Population': pop_size,
        'Vmax_Value': vmax_label,
        'Min_Iter': min_iter,
        'Max_Iter': max_iter,
        'Avg_Iter': mean_iter,
        'Std_Iter': std_iter
    }
    
    return {
        'vmax_label': vmax_label,
        'bin_costs': bin_costs,
        'bin_iters': bin_iters,
        'bin_stats_cost': bin_stats_cost,
        'bin_stats_iter': bin_stats_iter,
        'avg_convergence_curve': avg_convergence_curve
    }

if __name__ == "__main__":
    # --- 0. MULTIPROCESSING SETUP ---
    total_cores = multiprocessing.cpu_count()
    use_cores = max(1, int(total_cores * 0.75))
    
    # --- 1. PATH RESOLUTION & CONFIGURATION ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    transfer_functions = {
        "pso_sweep_vmax_do_feb5_daytime": {
            'tf_num': [1.346], 'tf_den': [1551.955, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 'max_kp': 100.0
        },
        "pso_sweep_vmax_do_feb7_daytime": {
            'tf_num': [1.133], 'tf_den': [2833.82, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 'max_kp': 100.0
        },
        "pso_sweep_vmax_do_feb25_daytime": {
            'tf_num': [2.287], 'tf_den': [3010.296, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 'max_kp': 100.0
        },
        "pso_sweep_vmax_do_feb26_daytime": {
            'tf_num': [2.430], 'tf_den': [3492.589, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 'max_kp': 100.0
        },
        "pso_sweep_vmax_do_feb5_nighttime": {
            'tf_num': [2.355], 'tf_den': [3083.590, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 'max_kp': 100.0
        },
        "pso_sweep_vmax_do_feb7_nighttime": {
            'tf_num': [2.049], 'tf_den': [4499.996, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 'max_kp': 100.0
        },
        "pso_sweep_vmax_do_feb25_nighttime": {
            'tf_num': [3.923], 'tf_den': [3012.232, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 'max_kp': 100.0
        },
        "pso_sweep_vmax_do_feb26_nighttime": {
            'tf_num': [3.132], 'tf_den': [2530.052, 1], 'tf_delay': 0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 'max_kp': 100.0
        },
        "pso_sweep_vmax_tds_feb09_10": {
            'tf_num': [-21.082], 'tf_den': [71160.91, 1], 'tf_delay': 0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': True, 'max_kp': -100.0
        },
        "pso_sweep_vmax_tds_feb10_11": {
            'tf_num': [-15.519], 'tf_den': [40156.08, 1], 'tf_delay': 0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': True, 'max_kp': -100.0
        },
        "pso_sweep_vmax_tds_feb11_12": {
            'tf_num': [-12.458], 'tf_den': [16825.29, 1], 'tf_delay': 0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': True, 'max_kp': -100.0
        }
    }

    base_config = {
        'patience_limit': 25,
        'max_iters': 100,
        'improvement_tol': 0.01,
        'n_rounds': 50  
    }

    population_sizes = [50]

    # --- 2. SWEEP CONFIGURATION (VMAX PARAMETERS) ---
    start_vmax = -1.00
    end_vmax = 1.00
    num_bins = 11

    OUTPUT_DIRECTORY = "PSO_SWEEP_VMAX"
    vmax_bins = np.linspace(start_vmax, end_vmax, num_bins)

    # --- 3. TOP-LEVEL MASTER DIRECTORY ---
    GLOBAL_ROOT_DIR = os.path.join(SCRIPT_DIR, OUTPUT_DIRECTORY)
    os.makedirs(GLOBAL_ROOT_DIR, exist_ok=True)

    print("STARTING 1D PSO VMAX SWEEP (MULTI-TF)")
    print(f"Global Directory         : {GLOBAL_ROOT_DIR}")
    print(f"Transfer Functions Queued: {list(transfer_functions.keys())}")
    print(f"Populations to test      : {population_sizes}")
    print(f"Vmax Ranges (Vmin = -1.0): [{start_vmax} to {end_vmax}] (11 bins)")
    print(f"Multiprocessing          : Using {use_cores} of {total_cores} available cores.")
    print("\n")

    # Initialize a Pool with 75% of available cores
    with multiprocessing.Pool(processes=use_cores) as pool:

        # --- 4. EXPERIMENT EXECUTION (OUTERMOST LOOP: TRANSFER FUNCTION) ---
        for tf_name, tf_params in transfer_functions.items():
            print(f"\n\n{'=' * 80}")
            print(f"INITIATING SWEEP FOR TRANSFER FUNCTION: {tf_name}")
            print(f"{'=' * 80}")

            TF_LEVEL_DIR = os.path.join(GLOBAL_ROOT_DIR, f"results_{tf_name}")
            os.makedirs(TF_LEVEL_DIR, exist_ok=True)

            final_report_cost_data = []
            final_report_iter_data = []

            # --- OUTER LOOP: POPULATION ---
            for pop_size in population_sizes:
                print(f"\n{'-' * 70}")
                print(f"SWEEPING POPULATION SIZE: {pop_size} [{tf_name}] ---")
                print(f"{'-' * 70}")

                MASTER_SWEEP_DIR = os.path.join(TF_LEVEL_DIR, f"pso_sweep_vmax_pop-{pop_size}")
                os.makedirs(MASTER_SWEEP_DIR, exist_ok=True)

                all_bins_costs = []
                avg_costs = []
                all_bins_iters = []
                avg_iters = []
                vmax_labels = []
                pop_specific_cost_data = []
                pop_specific_iter_data = []
                all_convergence_curves = []

                # Compile arguments for the multiprocessed pool mapping
                tasks = [
                    (vmax_val, f"{vmax_val:.2f}", pop_size, tf_params, base_config, MASTER_SWEEP_DIR)
                    for vmax_val in vmax_bins
                ]

                # --- INNER LOOP: MULTIPROCESSING EXECUTION ---
                # `imap` preserves the order of completion matching the input order
                results = list(tqdm(
                    pool.imap(worker_run_pso_vmax, tasks), 
                    total=len(tasks), 
                    desc=f"Running PSO (Pop: {pop_size})", 
                    unit="bin"
                ))

                # Unpack sequential results
                for res in results:
                    vmax_labels.append(res['vmax_label'])
                    
                    all_bins_costs.append(res['bin_costs'])
                    avg_costs.append(res['bin_stats_cost']['Avg_Cost'])
                    
                    all_bins_iters.append(res['bin_iters'])
                    avg_iters.append(res['bin_stats_iter']['Avg_Iter'])

                    pop_specific_cost_data.append(res['bin_stats_cost'])
                    pop_specific_iter_data.append(res['bin_stats_iter'])
                    
                    final_report_cost_data.append(res['bin_stats_cost'])
                    final_report_iter_data.append(res['bin_stats_iter'])
                    
                    all_convergence_curves.append(res['avg_convergence_curve'])


                # --- 5. VISUALIZATIONS & LOCAL REPORTS PER POPULATION SIZE ---
                print(f"\nGenerating Sweep Visualizations & Local Reports for Pop {pop_size} [{tf_name}]...")

                # ---------------------------
                #    COST VISUALIZATIONS
                # ---------------------------
                plt.figure(figsize=(10, 6))
                plt.plot(vmax_labels, avg_costs, marker='o', linestyle='-', color='b', linewidth=2)
                plt.title(f'Average Cost vs. PSO Vmax ({tf_name} | Pop: {pop_size})')
                plt.xlabel('Maximum Velocity (Vmax)')
                plt.ylabel('Average Cost')
                plt.ylim(bottom=0)
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'average_cost_line_plot.png'))
                plt.close()

                plt.figure(figsize=(12, 6))
                plt.boxplot(all_bins_costs, tick_labels=vmax_labels, showfliers=True)
                plt.title(f'Cost Distribution WITH Outliers ({tf_name} | Pop: {pop_size})')
                plt.xlabel('Maximum Velocity (Vmax)')
                plt.ylabel('Cost')
                plt.ylim(bottom=0)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'cost_distribution_boxplot_with_outliers.png'))
                plt.close()

                plt.figure(figsize=(12, 6))
                plt.boxplot(all_bins_costs, tick_labels=vmax_labels, showfliers=False)
                plt.title(f'Cost Distribution NO Outliers ({tf_name} | Pop: {pop_size})')
                plt.xlabel('Maximum Velocity (Vmax)')
                plt.ylabel('Cost')
                plt.ylim(bottom=0)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'cost_distribution_boxplot_no_outliers.png'))
                plt.close()

                # ---------------------------
                #  ITERATION VISUALIZATIONS
                # ---------------------------
                plt.figure(figsize=(10, 6))
                plt.plot(vmax_labels, avg_iters, marker='s', linestyle='-', color='g', linewidth=2)
                plt.title(f'Average Iterations vs. PSO Vmax ({tf_name} | Pop: {pop_size})')
                plt.xlabel('Maximum Velocity (Vmax)')
                plt.ylabel('Average Iterations')
                plt.ylim(bottom=0)
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'average_iterations_line_plot.png'))
                plt.close()

                plt.figure(figsize=(12, 6))
                plt.boxplot(all_bins_iters, tick_labels=vmax_labels, showfliers=True)
                plt.title(f'Iteration Distribution WITH Outliers ({tf_name} | Pop: {pop_size})')
                plt.xlabel('Maximum Velocity (Vmax)')
                plt.ylabel('Iterations to Converge')
                plt.ylim(bottom=0)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'iteration_distribution_boxplot_with_outliers.png'))
                plt.close()

                plt.figure(figsize=(12, 6))
                plt.boxplot(all_bins_iters, tick_labels=vmax_labels, showfliers=False)
                plt.title(f'Iteration Distribution NO Outliers ({tf_name} | Pop: {pop_size})')
                plt.xlabel('Maximum Velocity (Vmax)')
                plt.ylabel('Iterations to Converge')
                plt.ylim(bottom=0)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'iteration_distribution_boxplot_no_outliers.png'))
                plt.close()

                # ---------------------------
                #  CONVERGENCE VISUALIZATION (COST VS. ITERATION)
                # ---------------------------
                # Check if we successfully captured history data
                if any(c is not None for c in all_convergence_curves):
                    plt.figure(figsize=(12, 8))
                    
                    # Colormap to cleanly separate the lines
                    colors = plt.cm.viridis(np.linspace(0, 1, len(vmax_labels)))
                    
                    for i, (vmax_label, curve) in enumerate(zip(vmax_labels, all_convergence_curves)):
                        if curve is not None:
                            plt.plot(range(len(curve)), curve, label=f'Vmax = {vmax_label}', 
                                     color=colors[i], linewidth=2, alpha=0.8)

                    plt.title(f'Cost vs. Iterations by Vmax Value ({tf_name} | Pop: {pop_size})', fontsize=14)
                    plt.xlabel('Iteration', fontsize=12)
                    plt.ylabel('Average Best Cost', fontsize=12)
                    
                    # Log scale provides the best view of EA exponential cost dropoffs
                    plt.yscale('log') 
                    plt.grid(True, which="both", linestyle='--', alpha=0.6)
                    
                    # Legend outside the plot area
                    plt.legend(title='Vmax', bbox_to_anchor=(1.02, 1), 
                               loc='upper left', borderaxespad=0.)
                    plt.tight_layout()
                    plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'cost_vs_iteration_comparison.png'))
                    plt.close()

                # --- GENERATE LOCAL REPORTS FOR THIS POPULATION ---
                local_cost_path = os.path.join(MASTER_SWEEP_DIR, f"report_costs_pop_{pop_size}.csv")
                with open(local_cost_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        ['Population_Size', 'Vmax_Value', 'Lowest_Cost', 'Highest_Cost',
                         'Average_Cost', 'Std_Dev'])
                    for data in pop_specific_cost_data:
                        writer.writerow(
                            [data['Population'], data['Vmax_Value'], data['Min_Cost'], data['Max_Cost'], data['Avg_Cost'],
                             data['Std_Cost']])

                local_iter_path = os.path.join(MASTER_SWEEP_DIR, f"report_iterations_pop_{pop_size}.csv")
                with open(local_iter_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        ['Population_Size', 'Vmax_Value', 'Least_Iterations', 'Most_Iterations', 'Average_Iterations',
                         'Std_Dev_Iterations'])
                    for data in pop_specific_iter_data:
                        writer.writerow(
                            [data['Population'], data['Vmax_Value'], data['Min_Iter'], data['Max_Iter'], data['Avg_Iter'],
                             data['Std_Iter']])

                print(f"-> Local COST report saved to: {local_cost_path}")
                print(f"-> Local ITERATIONS report saved to: {local_iter_path}")

            # --- 6. FINAL MASTER REPORT GENERATION FOR THIS TF ---
            master_cost_report_csv_path = os.path.join(TF_LEVEL_DIR, f"master_pso_vmax_cost_report_{tf_name}.csv")
            master_iter_report_csv_path = os.path.join(TF_LEVEL_DIR, f"master_pso_vmax_iterations_report_{tf_name}.csv")

            print(
                "\n===========================================================================================================")
            print(f"                    FINAL REPORTS GENERATED FOR: {tf_name}")
            print(
                "===========================================================================================================")

            # Write Master Cost CSV
            with open(master_cost_report_csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Population_Size', 'Vmax_Value', 'Lowest_Cost', 'Highest_Cost',
                                 'Average_Cost', 'Std_Dev'])
                for data in final_report_cost_data:
                    writer.writerow(
                        [data['Population'], data['Vmax_Value'], data['Min_Cost'], data['Max_Cost'], data['Avg_Cost'],
                         data['Std_Cost']])

            # Write Master Iterations CSV
            with open(master_iter_report_csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    ['Population_Size', 'Vmax_Value', 'Least_Iterations', 'Most_Iterations', 'Average_Iterations',
                     'Std_Dev_Iterations'])
                for data in final_report_iter_data:
                    writer.writerow(
                        [data['Population'], data['Vmax_Value'], data['Min_Iter'], data['Max_Iter'], data['Avg_Iter'],
                         data['Std_Iter']])

            print(f"Master COST report saved to: {master_cost_report_csv_path}")
            print(f"Master ITERATIONS report saved to: {master_iter_report_csv_path}")