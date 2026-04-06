import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Added tqdm import
from Evolutionary_Algorithm_Testing.pso.pso_optimizer import PSOOptimizer

if __name__ == "__main__":
    # --- 0. PATH RESOLUTION ---
    # Dynamically grab the directory where this specific python script lives
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # --- 1. GLOBAL CONFIGURATION ---

    # EDITABLE DICTIONARY OF TRANSFER FUNCTIONS
    # Format -> "Folder_Name": {tf parameters dict}
    transfer_functions = {
        "tf3_do_daytime": {
            'tf_num': [45.52],
            'tf_den': [2654.54, 1],
            'tf_delay': 0.00,
            'tf_n_pade': 2
        },
        "tf_example_2": {
            'tf_num': [10.5],
            'tf_den': [1500.0, 1],
            'tf_delay': 0.10,
            'tf_n_pade': 2
        }
        # Add more transfer functions here as needed
    }

    base_config = {
        'patience_limit': 25,
        'max_iters': 100,
        'improvement_tol': 0.01,
        'n_rounds': 50  # Keep at 50 for the final high-fidelity run!
    }

    # Array of population sizes to sweep across
    population_sizes = [50]

    # --- 2. SWEEP CONFIGURATION (PHI PARAMETERS) ---
    start_phi = 2.05
    end_phi = 4.00
    num_bins = 20

    OUTPUT_DIRECTORY = "PSO_SWEEP_PHI"
    # Create an array of phi values to sweep
    phi_bins = np.linspace(start_phi, end_phi, num_bins)

    # --- 3. TOP-LEVEL MASTER DIRECTORY ---
    # Binds the output folder directly to the script's directory
    GLOBAL_ROOT_DIR = os.path.join(SCRIPT_DIR, OUTPUT_DIRECTORY)
    os.makedirs(GLOBAL_ROOT_DIR, exist_ok=True)

    print("STARTING 2D PSO PHI SWEEP (MULTI-TF)")
    print(f"Global Directory         : {GLOBAL_ROOT_DIR}")
    print(f"Transfer Functions Queued: {list(transfer_functions.keys())}")
    print(f"Populations to test      : {population_sizes}")
    print(f"Phi Ranges (phi1=phi2)   : [{start_phi} to {end_phi}] (20 bins)")
    print("\n")

    # --- 4. EXPERIMENT EXECUTION (OUTERMOST LOOP: TRANSFER FUNCTION) ---
    for tf_name, tf_params in transfer_functions.items():
        print(f"\n\n{'=' * 80}")
        print(f"INITIATING SWEEP FOR TRANSFER FUNCTION: {tf_name}")
        print(f"{'=' * 80}")

        # Directory specific to this Transfer Function
        TF_LEVEL_DIR = os.path.join(GLOBAL_ROOT_DIR, f"results_{tf_name}")
        os.makedirs(TF_LEVEL_DIR, exist_ok=True)

        # Lists to track every single bin's stats for this TF's master reports
        final_report_cost_data = []
        final_report_iter_data = []

        # --- OUTER LOOP: POPULATION ---
        for pop_size in population_sizes:
            print(f"\n{'-' * 70}")
            print(f"SWEEPING POPULATION SIZE: {pop_size} [{tf_name}] ---")
            print(f"{'-' * 70}")

            MASTER_SWEEP_DIR = os.path.join(TF_LEVEL_DIR, f"pso_sweep_phi_pop-{pop_size}")
            os.makedirs(MASTER_SWEEP_DIR, exist_ok=True)

            # Arrays for Cost Plotting
            all_bins_costs = []
            avg_costs = []

            # Arrays for Iteration Plotting
            all_bins_iters = []
            avg_iters = []

            phi_labels = []

            # Lists to track just THIS population's stats for its local reports
            pop_specific_cost_data = []
            pop_specific_iter_data = []

            # --- INNER LOOP: PHI VALUE WITH TQDM ---
            # Set up the progress bar here
            pbar = tqdm(phi_bins, desc=f"Running PSO (Pop: {pop_size})", unit="bin")

            for phi_val in pbar:
                phi_label = f"{phi_val:.2f}"
                phi_labels.append(phi_label)

                # Update progress bar with the current phi parameter being processed
                pbar.set_postfix({'Phi': phi_label})

                run_config = base_config.copy()
                run_config['population_size'] = pop_size
                run_config['phi1'] = phi_val
                run_config['phi2'] = phi_val

                bin_folder_name = os.path.join(MASTER_SWEEP_DIR, f"bin_phi_{phi_label}")
                run_config['output_folder'] = bin_folder_name

                # Instantiate and run the optimized code
                optimizer = PSOOptimizer(run_config, tf_params)
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
                    'Phi_Value': phi_label,
                    'Min_Cost': min_cost,
                    'Max_Cost': max_cost,
                    'Avg_Cost': mean_cost,
                    'Std_Cost': std_cost
                }

                bin_stats_iter = {
                    'Population': pop_size,
                    'Phi_Value': phi_label,
                    'Min_Iter': min_iter,
                    'Max_Iter': max_iter,
                    'Avg_Iter': mean_iter,
                    'Std_Iter': std_iter
                }

                # Store for the local population reports
                pop_specific_cost_data.append(bin_stats_cost)
                pop_specific_iter_data.append(bin_stats_iter)

                # Store for the global master reports for this TF
                final_report_cost_data.append(bin_stats_cost)
                final_report_iter_data.append(bin_stats_iter)

            # --- 5. VISUALIZATIONS & LOCAL REPORTS PER POPULATION SIZE ---
            print(f"\nGenerating Sweep Visualizations & Local Reports for Pop {pop_size} [{tf_name}]...")

            # ---------------------------
            #    COST VISUALIZATIONS
            # ---------------------------
            plt.figure(figsize=(10, 6))
            plt.plot(phi_labels, avg_costs, marker='o', linestyle='-', color='b', linewidth=2)
            plt.title(f'Average Cost vs. PSO Phi ({tf_name} | Pop: {pop_size})')
            plt.xlabel('Phi Coefficient (phi1 = phi2)')
            plt.ylabel('Average Cost')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'average_cost_line_plot.png'))
            plt.close()

            plt.figure(figsize=(12, 6))
            plt.boxplot(all_bins_costs, tick_labels=phi_labels, showfliers=True)
            plt.title(f'Cost Distribution WITH Outliers ({tf_name} | Pop: {pop_size})')
            plt.xlabel('Phi Coefficient (phi1 = phi2)')
            plt.ylabel('Cost')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'cost_distribution_boxplot_with_outliers.png'))
            plt.close()

            plt.figure(figsize=(12, 6))
            plt.boxplot(all_bins_costs, tick_labels=phi_labels, showfliers=False)
            plt.title(f'Cost Distribution NO Outliers ({tf_name} | Pop: {pop_size})')
            plt.xlabel('Phi Coefficient (phi1 = phi2)')
            plt.ylabel('Cost')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'cost_distribution_boxplot_no_outliers.png'))
            plt.close()

            # ---------------------------
            #  ITERATION VISUALIZATIONS
            # ---------------------------
            plt.figure(figsize=(10, 6))
            plt.plot(phi_labels, avg_iters, marker='s', linestyle='-', color='g', linewidth=2)
            plt.title(f'Average Iterations vs. PSO Phi ({tf_name} | Pop: {pop_size})')
            plt.xlabel('Phi Coefficient (phi1 = phi2)')
            plt.ylabel('Average Iterations')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'average_iterations_line_plot.png'))
            plt.close()

            plt.figure(figsize=(12, 6))
            plt.boxplot(all_bins_iters, tick_labels=phi_labels, showfliers=True)
            plt.title(f'Iteration Distribution WITH Outliers ({tf_name} | Pop: {pop_size})')
            plt.xlabel('Phi Coefficient (phi1 = phi2)')
            plt.ylabel('Iterations to Converge')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'iteration_distribution_boxplot_with_outliers.png'))
            plt.close()

            plt.figure(figsize=(12, 6))
            plt.boxplot(all_bins_iters, tick_labels=phi_labels, showfliers=False)
            plt.title(f'Iteration Distribution NO Outliers ({tf_name} | Pop: {pop_size})')
            plt.xlabel('Phi Coefficient (phi1 = phi2)')
            plt.ylabel('Iterations to Converge')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(MASTER_SWEEP_DIR, 'iteration_distribution_boxplot_no_outliers.png'))
            plt.close()

            # --- GENERATE LOCAL REPORTS FOR THIS POPULATION ---
            local_cost_path = os.path.join(MASTER_SWEEP_DIR, f"report_costs_pop_{pop_size}.csv")
            with open(local_cost_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    ['Population_Size', 'Phi_Value', 'Lowest_Cost', 'Highest_Cost',
                     'Average_Cost', 'Std_Dev'])
                for data in pop_specific_cost_data:
                    writer.writerow(
                        [data['Population'], data['Phi_Value'], data['Min_Cost'], data['Max_Cost'], data['Avg_Cost'],
                         data['Std_Cost']])

            local_iter_path = os.path.join(MASTER_SWEEP_DIR, f"report_iterations_pop_{pop_size}.csv")
            with open(local_iter_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    ['Population_Size', 'Phi_Value', 'Least_Iterations', 'Most_Iterations', 'Average_Iterations',
                     'Std_Dev_Iterations'])
                for data in pop_specific_iter_data:
                    writer.writerow(
                        [data['Population'], data['Phi_Value'], data['Min_Iter'], data['Max_Iter'], data['Avg_Iter'],
                         data['Std_Iter']])

            print(f"-> Local COST report saved to: {local_cost_path}")
            print(f"-> Local ITERATIONS report saved to: {local_iter_path}")

        # --- 6. FINAL MASTER REPORT GENERATION FOR THIS TF ---
        master_cost_report_csv_path = os.path.join(TF_LEVEL_DIR, f"master_pso_phi_cost_report_{tf_name}.csv")
        master_iter_report_csv_path = os.path.join(TF_LEVEL_DIR, f"master_pso_phi_iterations_report_{tf_name}.csv")

        print(
            "\n===========================================================================================================")
        print(f"                    FINAL REPORTS GENERATED FOR: {tf_name}")
        print(
            "===========================================================================================================")

        # Write Master Cost CSV
        with open(master_cost_report_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Population_Size', 'Phi_Value', 'Lowest_Cost', 'Highest_Cost',
                             'Average_Cost', 'Std_Dev'])
            for data in final_report_cost_data:
                writer.writerow(
                    [data['Population'], data['Phi_Value'], data['Min_Cost'], data['Max_Cost'], data['Avg_Cost'],
                     data['Std_Cost']])

        # Write Master Iterations CSV
        with open(master_iter_report_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ['Population_Size', 'Phi_Value', 'Least_Iterations', 'Most_Iterations', 'Average_Iterations',
                 'Std_Dev_Iterations'])
            for data in final_report_iter_data:
                writer.writerow(
                    [data['Population'], data['Phi_Value'], data['Min_Iter'], data['Max_Iter'], data['Avg_Iter'],
                     data['Std_Iter']])

        print(f"Master COST report saved to: {master_cost_report_csv_path}")
        print(f"Master ITERATIONS report saved to: {master_iter_report_csv_path}")