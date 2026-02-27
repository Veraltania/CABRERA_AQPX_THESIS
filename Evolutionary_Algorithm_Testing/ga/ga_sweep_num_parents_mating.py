import os
import csv
import numpy as np
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- MATPLOTLIB OPTIMIZATION ---
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for faster, thread-safe background plotting
import matplotlib.pyplot as plt

from Evolutionary_Algorithm_Testing.ga.ga_optimizer import GAOptimizer

# --- IMPORT STABILITY TOOLS FOR PRE-COMPUTATION ---
from Transfer_Function_Analysis.analyze_transfer_func_stability import define_transfer_func, define_guardrail_gain


# --- WORKER FUNCTION FOR MULTIPROCESSING ---
# Note: This must be defined at the top level so it can be pickled by multiprocessing
def run_single_ga_experiment(params):
    pop_size = params['pop_size']
    pct = params['pct']
    num_parents = params['num_parents']
    pct_label = params['pct_label']
    base_config = params['base_config']
    tf_params = params['tf_params']
    master_sweep_dir = params['master_sweep_dir']

    # Dynamic static config based on current pop_size
    ga_static_config = {
        "keep_elitism": int(pop_size * 0.05),
        "mutation_type": "adaptive",
        "crossover_type": "scattered",
    }

    run_config = base_config.copy()
    run_config.update(ga_static_config)
    run_config['population_size'] = pop_size
    run_config['num_parents_mating'] = num_parents

    bin_folder_name = os.path.join(master_sweep_dir, f"bin_mating_{pct_label}pct")
    os.makedirs(bin_folder_name, exist_ok=True)
    run_config['output_folder'] = bin_folder_name

    # Capture start time for this specific bin
    bin_start_time = time.time()

    # Instantiate and run the GA Optimizer
    optimizer = GAOptimizer(run_config, tf_params)
    optimizer.run_experiment()

    # Capture end time
    bin_end_time = time.time()
    elapsed_time = bin_end_time - bin_start_time

    # Extract the history
    bin_costs = optimizer.agg_history['costs']
    bin_iters = optimizer.agg_history['iterations']

    # Return a comprehensive dictionary of all necessary data
    return {
        'pop_size': pop_size,
        'pct': pct,
        'pct_label': f"{pct_label}%",
        'min_cost': np.min(bin_costs),
        'max_cost': np.max(bin_costs),
        'mean_cost': np.mean(bin_costs),
        'std_cost': np.std(bin_costs),
        'min_iter': np.min(bin_iters),
        'max_iter': np.max(bin_iters),
        'mean_iter': np.mean(bin_iters),
        'std_iter': np.std(bin_iters),
        'elapsed_time': elapsed_time,
        'raw_costs': bin_costs,
        'raw_iters': bin_iters
    }


if __name__ == "__main__":
    # --- Capture total start time ---
    start_time_sec = time.time()
    start_datetime = datetime.now()
    print(f"\n--- EXECUTION STARTED: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')} ---")

    # --- 1. GLOBAL CONFIGURATION ---
    tf_params = {
        'tf_num': [-24.44],
        'tf_den': [84487.79, 1],
        'tf_delay': 0.50,
        'tf_n_pade': 2
    }

    # --- PRE-COMPUTE STABILITY ONCE ---
    print("\n--- PRE-COMPUTING SYSTEM STABILITY ---")
    extracted_delay = tf_params.get('tf_delay', 0.0)
    computed_delay = 1.0 if extracted_delay == 0.0 else extracted_delay
    is_reverse_acting = tf_params['tf_num'][0] < 0

    temp_plant = define_transfer_func(
        tf_params['tf_num'],
        tf_params['tf_den'],
        computed_delay,
        tf_params.get('tf_n_pade', 2)
    )
    max_kp = define_guardrail_gain(temp_plant, find_negative_gain=is_reverse_acting)

    # Append to params so workers don't have to calculate it inside the loop
    tf_params['computed_delay'] = computed_delay
    tf_params['is_reverse_acting'] = is_reverse_acting
    tf_params['max_kp'] = float(max_kp) if max_kp is not None else None
    print(f"Computed Bounds -> Max Kp: {tf_params['max_kp']} | Reverse Acting: {is_reverse_acting}\n")

    base_config = {
        'patience_limit': 25,
        'max_iters': 200,
        'improvement_tol': 0.01,
        'n_rounds': 50
    }

    population_sizes = [20, 40, 60, 80, 100]
    start_pct = 0.1
    end_pct = 1.00
    num_bins = 20
    parent_mating_pct_bins = np.linspace(start_pct, end_pct, num_bins)

    TOP_LEVEL_DIR = "ga_sweep_results_mating-percentage_tf3_tds_test"
    os.makedirs(TOP_LEVEL_DIR, exist_ok=True)

    print("==========================================================")
    print("                 STARTING 2D MATING SWEEP")
    print(f"Transfer Function Folder : {TOP_LEVEL_DIR}")
    print(f"Populations to test      : {population_sizes}")
    print(f"Mating Percentages       : {[f'{int(p * 100)}%' for p in parent_mating_pct_bins]}")
    print("==========================================================\n")

    # --- 2. BUILD TASK LIST FOR MULTIPROCESSING ---
    tasks = []
    for pop_size in population_sizes:
        MASTER_SWEEP_DIR = os.path.join(TOP_LEVEL_DIR, f"ga_sweep_mating_pct_pop-{pop_size}")
        os.makedirs(MASTER_SWEEP_DIR, exist_ok=True)

        for pct in parent_mating_pct_bins:
            num_parents = max(2, int(pop_size * pct))
            tasks.append({
                'pop_size': pop_size,
                'pct': pct,
                'num_parents': num_parents,
                'pct_label': int(pct * 100),
                'base_config': base_config,
                'tf_params': tf_params,
                'master_sweep_dir': MASTER_SWEEP_DIR
            })

    # --- 3. EXECUTE TASKS CONCURRENTLY ---
    all_results = []
    total_tasks = len(tasks)

    print(f"Starting multiprocessing pool with {total_tasks} total configurations...")

    # ProcessPoolExecutor automatically uses all available CPU cores
    with ProcessPoolExecutor() as executor:
        # Submit all tasks
        futures = {executor.submit(run_single_ga_experiment, task): task for task in tasks}

        # Gather results as they complete
        completed = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
                completed += 1
                print(
                    f"Progress: [{completed}/{total_tasks}] Completed Pop={result['pop_size']}, Mating={result['pct_label']}")
            except Exception as exc:
                print(f"A worker generated an exception: {exc}")

    # Sort results to ensure ordered processing for plots and reports
    all_results.sort(key=lambda x: (x['pop_size'], x['pct']))

    # --- 4. DATA AGGREGATION & BULK DISK I/O ---
    print("\nAll GA experiments finished! Generating plots and saving CSV reports...")

    final_report_cost_data = []
    final_report_iter_data = []
    worker_timing_data = []

    # Group results by population size for local reporting/plotting
    for pop_size in population_sizes:
        MASTER_SWEEP_DIR = os.path.join(TOP_LEVEL_DIR, f"ga_sweep_mating_pct_pop-{pop_size}")

        # Filter results for this specific population size
        pop_results = [r for r in all_results if r['pop_size'] == pop_size]

        pct_labels = [r['pct_label'] for r in pop_results]
        all_bins_costs = [r['raw_costs'] for r in pop_results]
        avg_costs = [r['mean_cost'] for r in pop_results]

        all_bins_iters = [r['raw_iters'] for r in pop_results]
        avg_iters = [r['mean_iter'] for r in pop_results]

        pop_cost_rows = []
        pop_iter_rows = []

        for r in pop_results:
            # Build data for global and local reports
            cost_row = [r['pop_size'], r['pct_label'], r['min_cost'], r['max_cost'], r['mean_cost'], r['std_cost']]
            iter_row = [r['pop_size'], r['pct_label'], r['min_iter'], r['max_iter'], r['mean_iter'], r['std_iter']]

            pop_cost_rows.append(cost_row)
            pop_iter_rows.append(iter_row)
            final_report_cost_data.append(cost_row)
            final_report_iter_data.append(iter_row)

            worker_timing_data.append([r['pop_size'], r['pct_label'], r['elapsed_time']])

        # --- LOCAL CSV REPORTS (Bulk Write) ---
        local_cost_path = os.path.join(MASTER_SWEEP_DIR, f"report_costs_pop_{pop_size}.csv")
        with open(local_cost_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ['Population_Size', 'Mating_Percentage', 'Lowest_Cost_Log10_ITAE', 'Highest_Cost_Log10_ITAE',
                 'Average_Cost_Log10_ITAE', 'Std_Dev_Log10_ITAE'])
            writer.writerows(pop_cost_rows)

        local_iter_path = os.path.join(MASTER_SWEEP_DIR, f"report_iterations_pop_{pop_size}.csv")
        with open(local_iter_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ['Population_Size', 'Mating_Percentage', 'Least_Iterations', 'Most_Iterations', 'Average_Iterations',
                 'Std_Dev_Iterations'])
            writer.writerows(pop_iter_rows)

        # --- LOCAL PLOTS ---
        # Plot 1: Average Cost Line
        plt.figure(figsize=(10, 6))
        plt.plot(pct_labels, avg_costs, marker='o', linestyle='-', color='b', linewidth=2)
        plt.title(f'Average ITAE Cost vs. Population Mating Pct (Pop: {pop_size})')
        plt.xlabel('Percentage of Population Mating')
        plt.ylabel('Average Log10(ITAE) Cost')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'average_cost_line_plot.png'))
        plt.close()

        # Plot 2: Cost Boxplot (Outliers)
        plt.figure(figsize=(12, 6))
        plt.boxplot(all_bins_costs, tick_labels=pct_labels, showfliers=True)
        plt.title(f'Cost Distribution WITH Outliers (Pop: {pop_size}, {base_config["n_rounds"]} Trials/Bin)')
        plt.xlabel('Percentage of Population Mating')
        plt.ylabel('Log10(ITAE) Cost')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'cost_distribution_boxplot_with_outliers.png'))
        plt.close()

        # Plot 3: Cost Boxplot (No Outliers)
        plt.figure(figsize=(12, 6))
        plt.boxplot(all_bins_costs, tick_labels=pct_labels, showfliers=False)
        plt.title(f'Cost Distribution NO Outliers (Pop: {pop_size}, {base_config["n_rounds"]} Trials/Bin)')
        plt.xlabel('Percentage of Population Mating')
        plt.ylabel('Log10(ITAE) Cost')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'cost_distribution_boxplot_no_outliers.png'))
        plt.close()

        # Plot 4: Average Iterations Line
        plt.figure(figsize=(10, 6))
        plt.plot(pct_labels, avg_iters, marker='s', linestyle='-', color='g', linewidth=2)
        plt.title(f'Average Iterations vs. Population Mating Pct (Pop: {pop_size})')
        plt.xlabel('Percentage of Population Mating')
        plt.ylabel('Average Iterations')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'average_iterations_line_plot.png'))
        plt.close()

        # Plot 5: Iterations Boxplot (Outliers)
        plt.figure(figsize=(12, 6))
        plt.boxplot(all_bins_iters, tick_labels=pct_labels, showfliers=True)
        plt.title(f'Iteration Distribution WITH Outliers (Pop: {pop_size}, {base_config["n_rounds"]} Trials/Bin)')
        plt.xlabel('Percentage of Population Mating')
        plt.ylabel('Iterations to Converge')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'iteration_distribution_boxplot_with_outliers.png'))
        plt.close()

        # Plot 6: Iterations Boxplot (No Outliers)
        plt.figure(figsize=(12, 6))
        plt.boxplot(all_bins_iters, tick_labels=pct_labels, showfliers=False)
        plt.title(f'Iteration Distribution NO Outliers (Pop: {pop_size}, {base_config["n_rounds"]} Trials/Bin)')
        plt.xlabel('Percentage of Population Mating')
        plt.ylabel('Iterations to Converge')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'iteration_distribution_boxplot_no_outliers.png'))
        plt.close()

    # --- 5. GLOBAL MASTER CSV REPORTS (Bulk Write) ---
    print(
        "\n===========================================================================================================")
    print("                                FINAL MATING 2D SWEEP REPORTS")
    print("===========================================================================================================")

    master_cost_report_csv_path = os.path.join(TOP_LEVEL_DIR, "master_mating_cost_report.csv")
    with open(master_cost_report_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Population_Size', 'Mating_Percentage', 'Lowest_Cost_Log10_ITAE', 'Highest_Cost_Log10_ITAE',
                         'Average_Cost_Log10_ITAE', 'Std_Dev_Log10_ITAE'])
        writer.writerows(final_report_cost_data)

    master_iter_report_csv_path = os.path.join(TOP_LEVEL_DIR, "master_mating_iterations_report.csv")
    with open(master_iter_report_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['Population_Size', 'Mating_Percentage', 'Least_Iterations', 'Most_Iterations', 'Average_Iterations',
             'Std_Dev_Iterations'])
        writer.writerows(final_report_iter_data)

    print(f"Global master COST report saved to: ./{master_cost_report_csv_path}")
    print(f"Global master ITERATIONS report saved to: ./{master_iter_report_csv_path}")

    # --- 6. TIMING REPORT GENERATION (Bulk Write) ---
    end_time_sec = time.time()
    end_datetime = datetime.now()
    elapsed_seconds = end_time_sec - start_time_sec

    m, s = divmod(elapsed_seconds, 60)
    h, m = divmod(m, 60)
    elapsed_formatted = f"{int(h):02d}:{int(m):02d}:{s:05.2f}"

    timestamp_str = start_datetime.strftime('%Y%m%d_%H%M%S')

    total_timing_filename = os.path.join(TOP_LEVEL_DIR, f"execution_timing_total_{timestamp_str}.csv")
    with open(total_timing_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Start Time", "End Time", "Elapsed Time (HH:MM:SS)", "Elapsed Time (Seconds)"])
        writer.writerow([start_datetime.strftime('%Y-%m-%d %H:%M:%S'), end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                         elapsed_formatted, round(elapsed_seconds, 2)])

    bin_timing_filename = os.path.join(TOP_LEVEL_DIR, f"execution_timing_bins_{timestamp_str}.csv")
    with open(bin_timing_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Population Size", "Mating Percentage", "Elapsed Time (HH:MM:SS)", "Elapsed Time (Seconds)"])

        # Format individual bin timings
        formatted_timing_rows = []
        for row in worker_timing_data:
            w_sec = row[2]
            wm, ws = divmod(w_sec, 60)
            wh, wm = divmod(wm, 60)
            w_formatted = f"{int(wh):02d}:{int(wm):02d}:{ws:05.2f}"
            formatted_timing_rows.append([row[0], row[1], w_formatted, round(w_sec, 2)])

        writer.writerows(formatted_timing_rows)

    print(f"\n--- EXECUTION FINISHED: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')} ---")
    print(f"Total Time Elapsed: {elapsed_formatted} ({elapsed_seconds:.2f} pure seconds)\n")