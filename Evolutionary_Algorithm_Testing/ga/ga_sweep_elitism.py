import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from Evolutionary_Algorithm_Testing.ga.ga_optimizer import GAOptimizer

if __name__ == "__main__":
    # --- 1. GLOBAL CONFIGURATION ---
    tf_params = {
        'tf_num': [51.39],
        'tf_den': [1550.18, 1],
        'tf_delay': 71.78,
        'tf_n_pade': 2
    }

    base_config = {
        'patience_limit': 25,
        'max_iters': 200,
        'improvement_tol': 0.01,
        'n_rounds': 50  # 50 trials for statistical significance
    }

    # Array of population sizes to sweep across
    population_sizes = [20, 40, 60, 80, 100]

    # --- 2. ELITISM SWEEP CONFIGURATION ---

    # -> INPUT YOUR OPTIMAL MATING PERCENTAGE HERE <-
    OPTIMAL_MATING_PCT = 0.55

    # Sweeping Elitism from 0% to 38% (High elitism halts exploration)
    start_elitism_pct = 0.00  # 0%
    end_elitism_pct = 0.38  # 38%
    num_bins = 20  # Creates exactly 2% increments (0%, 2%, 4%...)

    # Create an array of percentages
    elitism_pct_bins = np.linspace(start_elitism_pct, end_elitism_pct, num_bins)

    # --- 3. TOP-LEVEL MASTER DIRECTORY ---
    TOP_LEVEL_DIR = "ga_sweep_elitism_tf1"
    os.makedirs(TOP_LEVEL_DIR, exist_ok=True)

    # Lists to track every single bin's stats for the comprehensive master reports
    final_report_cost_data = []
    final_report_iter_data = []

    print("==========================================================")
    print("                 STARTING 2D ELITISM SWEEP")
    print(f"Transfer Function Folder : {TOP_LEVEL_DIR}")
    print(f"Fixed Mating Percentage  : {int(OPTIMAL_MATING_PCT * 100)}%")
    print(f"Populations to test      : {population_sizes}")
    print(f"Elitism Percentages      : {[f'{int(p * 100)}%' for p in elitism_pct_bins]}")
    print("==========================================================\n")

    # --- 4. EXPERIMENT EXECUTION (OUTER LOOP: POPULATION) ---
    for pop_size in population_sizes:
        print(f"\n\n{'*' * 70}")
        print(f"--- SWEEPING POPULATION SIZE: {pop_size} ---")
        print(f"{'*' * 70}")

        MASTER_SWEEP_DIR = os.path.join(TOP_LEVEL_DIR, f"ga_sweep_elitism_pop-{pop_size}")
        os.makedirs(MASTER_SWEEP_DIR, exist_ok=True)

        # Dynamic static config applying your known optimal mating amount
        optimal_num_parents = max(2, int(pop_size * OPTIMAL_MATING_PCT))

        ga_static_config = {
            "num_parents_mating": optimal_num_parents,
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

        # --- INNER LOOP: ELITISM PERCENTAGE ---
        for pct in elitism_pct_bins:
            # Calculate integer number of elites to keep
            num_elites = int(pop_size * pct)

            pct_label = int(pct * 100)
            pct_labels.append(f"{pct_label}%")

            print(f"\n{'#' * 60}")
            print(f"GA EXPERIMENT: Pop={pop_size} | Elitism={pct_label}% (N={num_elites})")
            print(f"{'#' * 60}")

            run_config = base_config.copy()
            run_config.update(ga_static_config)
            run_config['population_size'] = pop_size
            run_config['keep_elitism'] = num_elites

            # Nest the bin folder inside this population's master directory
            bin_folder_name = os.path.join(MASTER_SWEEP_DIR, f"bin_elitism_{pct_label}pct")
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
                'Elitism_Pct': f"{pct_label}%",
                'Min_Cost': min_cost,
                'Max_Cost': max_cost,
                'Avg_Cost': mean_cost,
                'Std_Cost': std_cost
            }

            bin_stats_iter = {
                'Population': pop_size,
                'Elitism_Pct': f"{pct_label}%",
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

        # --- 5. VISUALIZATIONS & LOCAL REPORTS PER POPULATION SIZE ---
        print(f"\nGenerating Sweep Visualizations & Local Reports for Population {pop_size}...")

        # ---------------------------
        #    COST VISUALIZATIONS
        # ---------------------------
        # Plot 1: Line Graph (Average Cost per Bin)
        plt.figure(figsize=(10, 6))
        plt.plot(pct_labels, avg_costs, marker='o', linestyle='-', color='r', linewidth=2)
        plt.title(f'Average ITAE Cost vs. Population Elitism Pct (Pop: {pop_size})')
        plt.xlabel('Percentage of Population Kept as Elites')
        plt.ylabel('Average Log10(ITAE) Cost')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'average_cost_line_plot.png'))
        plt.close()

        # Plot 2: Box Plot WITH Outliers (Cost)
        plt.figure(figsize=(12, 6))
        plt.boxplot(all_bins_costs, tick_labels=pct_labels, showfliers=True)
        plt.title(f'Cost Distribution WITH Outliers (Pop: {pop_size}, {base_config["n_rounds"]} Trials/Bin)')
        plt.xlabel('Percentage of Population Kept as Elites')
        plt.ylabel('Log10(ITAE) Cost')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'cost_distribution_boxplot_with_outliers.png'))
        plt.close()

        # Plot 3: Box Plot WITHOUT Outliers (Cost)
        plt.figure(figsize=(12, 6))
        plt.boxplot(all_bins_costs, tick_labels=pct_labels, showfliers=False)
        plt.title(f'Cost Distribution NO Outliers (Pop: {pop_size}, {base_config["n_rounds"]} Trials/Bin)')
        plt.xlabel('Percentage of Population Kept as Elites')
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
        plt.title(f'Average Iterations vs. Population Elitism Pct (Pop: {pop_size})')
        plt.xlabel('Percentage of Population Kept as Elites')
        plt.ylabel('Average Iterations')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'average_iterations_line_plot.png'))
        plt.close()

        # Plot 5: Box Plot WITH Outliers (Iterations)
        plt.figure(figsize=(12, 6))
        plt.boxplot(all_bins_iters, tick_labels=pct_labels, showfliers=True)
        plt.title(f'Iteration Distribution WITH Outliers (Pop: {pop_size}, {base_config["n_rounds"]} Trials/Bin)')
        plt.xlabel('Percentage of Population Kept as Elites')
        plt.ylabel('Iterations to Converge')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'iteration_distribution_boxplot_with_outliers.png'))
        plt.close()

        # Plot 6: Box Plot WITHOUT Outliers (Iterations)
        plt.figure(figsize=(12, 6))
        plt.boxplot(all_bins_iters, tick_labels=pct_labels, showfliers=False)
        plt.title(f'Iteration Distribution NO Outliers (Pop: {pop_size}, {base_config["n_rounds"]} Trials/Bin)')
        plt.xlabel('Percentage of Population Kept as Elites')
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
                ['Population_Size', 'Elitism_Percentage', 'Lowest_Cost_Log10_ITAE', 'Highest_Cost_Log10_ITAE',
                 'Average_Cost_Log10_ITAE', 'Std_Dev_Log10_ITAE'])
            for data in pop_specific_cost_data:
                writer.writerow(
                    [data['Population'], data['Elitism_Pct'], data['Min_Cost'], data['Max_Cost'], data['Avg_Cost'],
                     data['Std_Cost']])

        # 2. Local Iterations Report
        local_iter_path = os.path.join(MASTER_SWEEP_DIR, f"report_iterations_pop_{pop_size}.csv")
        with open(local_iter_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ['Population_Size', 'Elitism_Percentage', 'Least_Iterations', 'Most_Iterations', 'Average_Iterations',
                 'Std_Dev_Iterations'])
            for data in pop_specific_iter_data:
                writer.writerow(
                    [data['Population'], data['Elitism_Pct'], data['Min_Iter'], data['Max_Iter'], data['Avg_Iter'],
                     data['Std_Iter']])

        print(f"-> Local COST report saved to: {local_cost_path}")
        print(f"-> Local ITERATIONS report saved to: {local_iter_path}")

    # --- 6. FINAL MASTER REPORT GENERATION ---
    master_cost_report_csv_path = os.path.join(TOP_LEVEL_DIR, "master_elitism_cost_report.csv")
    master_iter_report_csv_path = os.path.join(TOP_LEVEL_DIR, "master_elitism_iterations_report.csv")

    print(
        "\n===========================================================================================================")
    print("                                FINAL ELITISM 2D SWEEP REPORTS")
    print("===========================================================================================================")

    # Write Master Cost CSV
    with open(master_cost_report_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Population_Size', 'Elitism_Percentage', 'Lowest_Cost_Log10_ITAE', 'Highest_Cost_Log10_ITAE',
                         'Average_Cost_Log10_ITAE', 'Std_Dev_Log10_ITAE'])
        for data in final_report_cost_data:
            writer.writerow(
                [data['Population'], data['Elitism_Pct'], data['Min_Cost'], data['Max_Cost'], data['Avg_Cost'],
                 data['Std_Cost']])

    # Write Master Iterations CSV
    with open(master_iter_report_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['Population_Size', 'Elitism_Percentage', 'Least_Iterations', 'Most_Iterations', 'Average_Iterations',
             'Std_Dev_Iterations'])
        for data in final_report_iter_data:
            writer.writerow(
                [data['Population'], data['Elitism_Pct'], data['Min_Iter'], data['Max_Iter'], data['Avg_Iter'],
                 data['Std_Iter']])

    print(f"Global master COST report saved to: ./{master_cost_report_csv_path}")
    print(f"Global master ITERATIONS report saved to: ./{master_iter_report_csv_path}")
    print(
        "===========================================================================================================\n")