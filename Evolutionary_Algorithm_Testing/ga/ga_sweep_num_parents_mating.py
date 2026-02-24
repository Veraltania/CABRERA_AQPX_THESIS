import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from Evolutionary_Algorithm_Testing.ga.ga_optimizer import GAOptimizer

if __name__ == "__main__":
    # --- 1. GLOBAL CONFIGURATION ---
    tf_params = {
        'tf_num': [92.76],
        'tf_den': [2629.13, 1],
        'tf_delay': 54.00,
        'tf_n_pade': 2
    }

    base_config = {
        'patience_limit': 25,
        'max_iters': 200,
        'improvement_tol': 0.01,
        'n_rounds': 50  # Keep at 50 for the final high-fidelity run!
    }

    # Array of population sizes to sweep across
    population_sizes = [20, 40, 60, 80, 100]

    # --- 2. SWEEP CONFIGURATION (PERCENTAGES) ---
    start_pct = 0.05  # 5%
    end_pct = 1.00  # 100%
    num_bins = 20

    # Create an array of percentages: [0.05, 0.1, ..., 1.0]
    parent_mating_pct_bins = np.linspace(start_pct, end_pct, num_bins)

    # --- TOP-LEVEL MASTER DIRECTORY ---
    TOP_LEVEL_DIR = "ga_sweep_results_mating-percentage_tf3_tds_test"
    os.makedirs(TOP_LEVEL_DIR, exist_ok=True)

    # Lists to track every single bin's stats for the comprehensive master reports
    final_report_cost_data = []
    final_report_iter_data = []

    print("==========================================================")
    print("                 STARTING 2D MATING SWEEP")
    print(f"Transfer Function Folder : {TOP_LEVEL_DIR}")
    print(f"Populations to test      : {population_sizes}")
    print(f"Mating Percentages       : {[f'{int(p * 100)}%' for p in parent_mating_pct_bins]}")
    print("==========================================================\n")

    # --- 3. EXPERIMENT EXECUTION (OUTER LOOP: POPULATION) ---
    for pop_size in population_sizes:
        print(f"\n\n{'*' * 70}")
        print(f"--- SWEEPING POPULATION SIZE: {pop_size} ---")
        print(f"{'*' * 70}")

        MASTER_SWEEP_DIR = os.path.join(TOP_LEVEL_DIR, f"ga_sweep_mating_pct_pop-{pop_size}")
        os.makedirs(MASTER_SWEEP_DIR, exist_ok=True)

        # Dynamic static config based on current pop_size
        ga_static_config = {
            "keep_elitism": int(pop_size * 0.05),
            "mutation_type": "adaptive",
            "crossover_type": "scattered",
        }

        # Arrays for Cost Plotting
        all_bins_costs = []
        avg_costs = []

        # Arrays for Iteration Plotting
        all_bins_iters = []
        avg_iters = []

        pct_labels = []

        # Lists to track just THIS population's stats for its local reports
        pop_specific_cost_data = []
        pop_specific_iter_data = []

        # --- INNER LOOP: MATING PERCENTAGE ---
        for pct in parent_mating_pct_bins:
            num_parents = int(pop_size * pct)
            num_parents = max(2, num_parents)  # Guardrail: At least 2 parents needed

            pct_label = int(pct * 100)
            pct_labels.append(f"{pct_label}%")

            print(f"\n{'#' * 60}")
            print(f"GA EXPERIMENT: Pop={pop_size} | Mating={pct_label}% (N={num_parents})")
            print(f"{'#' * 60}")

            run_config = base_config.copy()
            run_config.update(ga_static_config)
            run_config['population_size'] = pop_size
            run_config['num_parents_mating'] = num_parents

            bin_folder_name = os.path.join(MASTER_SWEEP_DIR, f"bin_mating_{pct_label}pct")
            run_config['output_folder'] = bin_folder_name

            # Instantiate and run the optimized code
            optimizer = GAOptimizer(run_config, tf_params)
            optimizer.run_experiment()

            # Extract the 50 trial costs AND iterations from this bin's history
            bin_costs = optimizer.agg_history['costs']
            bin_iters = optimizer.agg_history['iterations']

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

            # Store for plotting (Cost)
            all_bins_costs.append(bin_costs)
            avg_costs.append(mean_cost)

            # Store for plotting (Iterations)
            all_bins_iters.append(bin_iters)
            avg_iters.append(mean_iter)

            # Dictionaries representing this specific bin's statistics
            bin_stats_cost = {
                'Population': pop_size,
                'Mating_Pct': f"{pct_label}%",
                'Min_Cost': min_cost,
                'Max_Cost': max_cost,
                'Avg_Cost': mean_cost,
                'Std_Cost': std_cost
            }

            bin_stats_iter = {
                'Population': pop_size,
                'Mating_Pct': f"{pct_label}%",
                'Min_Iter': min_iter,
                'Max_Iter': max_iter,
                'Avg_Iter': mean_iter,
                'Std_Iter': std_iter
            }

            # Store for the local population reports
            pop_specific_cost_data.append(bin_stats_cost)
            pop_specific_iter_data.append(bin_stats_iter)

            # Store for the global master reports
            final_report_cost_data.append(bin_stats_cost)
            final_report_iter_data.append(bin_stats_iter)

        # --- 4. VISUALIZATIONS & LOCAL REPORT PER POPULATION SIZE ---
        print(f"\nGenerating Sweep Visualizations & Local Reports for Population {pop_size}...")

        # ---------------------------
        #    COST VISUALIZATIONS
        # ---------------------------
        # Plot 1: Line Graph (Average Cost per Bin)
        plt.figure(figsize=(10, 6))
        plt.plot(pct_labels, avg_costs, marker='o', linestyle='-', color='b', linewidth=2)
        plt.title(f'Average ITAE Cost vs. Population Mating Pct (Pop: {pop_size})')
        plt.xlabel('Percentage of Population Mating')
        plt.ylabel('Average Log10(ITAE) Cost')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'average_cost_line_plot.png'))
        plt.close()

        # Plot 2: Box Plot WITH Outliers (Cost)
        plt.figure(figsize=(12, 6))
        plt.boxplot(all_bins_costs, tick_labels=pct_labels, showfliers=True)
        plt.title(f'Cost Distribution WITH Outliers (Pop: {pop_size}, {base_config["n_rounds"]} Trials/Bin)')
        plt.xlabel('Percentage of Population Mating')
        plt.ylabel('Log10(ITAE) Cost')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'cost_distribution_boxplot_with_outliers.png'))
        plt.close()

        # Plot 3: Box Plot WITHOUT Outliers (Cost)
        plt.figure(figsize=(12, 6))
        plt.boxplot(all_bins_costs, tick_labels=pct_labels, showfliers=False)
        plt.title(f'Cost Distribution NO Outliers (Pop: {pop_size}, {base_config["n_rounds"]} Trials/Bin)')
        plt.xlabel('Percentage of Population Mating')
        plt.ylabel('Log10(ITAE) Cost')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'cost_distribution_boxplot_no_outliers.png'))
        plt.close()

        # ---------------------------
        #  ITERATION VISUALIZATIONS
        # ---------------------------
        # Plot 4: Line Graph (Average Iterations per Bin)
        plt.figure(figsize=(10, 6))
        plt.plot(pct_labels, avg_iters, marker='s', linestyle='-', color='g', linewidth=2)
        plt.title(f'Average Iterations vs. Population Mating Pct (Pop: {pop_size})')
        plt.xlabel('Percentage of Population Mating')
        plt.ylabel('Average Iterations')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'average_iterations_line_plot.png'))
        plt.close()

        # Plot 5: Box Plot WITH Outliers (Iterations)
        plt.figure(figsize=(12, 6))
        plt.boxplot(all_bins_iters, tick_labels=pct_labels, showfliers=True)
        plt.title(f'Iteration Distribution WITH Outliers (Pop: {pop_size}, {base_config["n_rounds"]} Trials/Bin)')
        plt.xlabel('Percentage of Population Mating')
        plt.ylabel('Iterations to Converge')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'iteration_distribution_boxplot_with_outliers.png'))
        plt.close()

        # Plot 6: Box Plot WITHOUT Outliers (Iterations)
        plt.figure(figsize=(12, 6))
        plt.boxplot(all_bins_iters, tick_labels=pct_labels, showfliers=False)
        plt.title(f'Iteration Distribution NO Outliers (Pop: {pop_size}, {base_config["n_rounds"]} Trials/Bin)')
        plt.xlabel('Percentage of Population Mating')
        plt.ylabel('Iterations to Converge')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'iteration_distribution_boxplot_no_outliers.png'))
        plt.close()

        # --- GENERATE LOCAL REPORTS FOR THIS POPULATION ---

        # 1. Local Cost Report
        local_cost_path = os.path.join(MASTER_SWEEP_DIR, f"report_costs_pop_{pop_size}.csv")
        with open(local_cost_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ['Population_Size', 'Mating_Percentage', 'Lowest_Cost_Log10_ITAE', 'Highest_Cost_Log10_ITAE',
                 'Average_Cost_Log10_ITAE', 'Std_Dev_Log10_ITAE'])
            for data in pop_specific_cost_data:
                writer.writerow(
                    [data['Population'], data['Mating_Pct'], data['Min_Cost'], data['Max_Cost'], data['Avg_Cost'],
                     data['Std_Cost']])

        # 2. Local Iterations Report
        local_iter_path = os.path.join(MASTER_SWEEP_DIR, f"report_iterations_pop_{pop_size}.csv")
        with open(local_iter_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ['Population_Size', 'Mating_Percentage', 'Least_Iterations', 'Most_Iterations', 'Average_Iterations',
                 'Std_Dev_Iterations'])
            for data in pop_specific_iter_data:
                writer.writerow(
                    [data['Population'], data['Mating_Pct'], data['Min_Iter'], data['Max_Iter'], data['Avg_Iter'],
                     data['Std_Iter']])

        print(f"-> Local COST report saved to: {local_cost_path}")
        print(f"-> Local ITERATIONS report saved to: {local_iter_path}")

    # --- 5. FINAL MASTER REPORT GENERATION ---
    master_cost_report_csv_path = os.path.join(TOP_LEVEL_DIR, "master_mating_cost_report.csv")
    master_iter_report_csv_path = os.path.join(TOP_LEVEL_DIR, "master_mating_iterations_report.csv")

    print(
        "\n===========================================================================================================")
    print("                                FINAL MATING 2D SWEEP REPORTS")
    print("===========================================================================================================")

    # Write Master Cost CSV
    with open(master_cost_report_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Population_Size', 'Mating_Percentage', 'Lowest_Cost_Log10_ITAE', 'Highest_Cost_Log10_ITAE',
                         'Average_Cost_Log10_ITAE', 'Std_Dev_Log10_ITAE'])

        for data in final_report_cost_data:
            writer.writerow(
                [data['Population'], data['Mating_Pct'], data['Min_Cost'], data['Max_Cost'], data['Avg_Cost'],
                 data['Std_Cost']])

    # Write Master Iterations CSV
    with open(master_iter_report_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['Population_Size', 'Mating_Percentage', 'Least_Iterations', 'Most_Iterations', 'Average_Iterations',
             'Std_Dev_Iterations'])
        for data in final_report_iter_data:
            writer.writerow(
                [data['Population'], data['Mating_Pct'], data['Min_Iter'], data['Max_Iter'], data['Avg_Iter'],
                 data['Std_Iter']])

    print(f"Global master COST report saved to: ./{master_cost_report_csv_path}")
    print(f"Global master ITERATIONS report saved to: ./{master_iter_report_csv_path}")
    print(
        "===========================================================================================================\n")