import os
import numpy as np
import matplotlib.pyplot as plt
from Evolutionary_Algorithm_Testing.ga.ga_optimizer import GAOptimizer

if __name__ == "__main__":
    # --- 1. GLOBAL CONFIGURATION ---
    tf_params = {
        'tf_num': [44.93],
        'tf_den': [1474.32, 1],
        'tf_delay': 343.93,
        'tf_n_pade': 2
    }

    base_config = {
        'population_size': 100,
        'patience_limit': 25,
        'max_iters': 200,
        'improvement_tol': 0.01,
        'n_rounds': 10  # <-- Changed to 10 for faster prototyping
    }

    ga_static_config = {
        "keep_elitism": int(base_config['population_size'] * 0.05),
        "mutation_type": "adaptive",
        "crossover_type": "scattered",
    }

    # --- 2. SWEEP CONFIGURATION (PERCENTAGES) ---
    start_pct = 0.10  # 10%
    end_pct = 1.00  # 100%
    num_bins = 10

    # Create an array of percentages: [0.1, 0.2, 0.3, ..., 1.0]
    parent_mating_pct_bins = np.linspace(start_pct, end_pct, num_bins)

    # Define the single master directory for EVERYTHING
    MASTER_SWEEP_DIR = f"ga_sweep_mating_pct_pop-{base_config['population_size']}"
    os.makedirs(MASTER_SWEEP_DIR, exist_ok=True)

    print(f"Starting GA Sweep for mating percentages: {[f'{int(p * 100)}%' for p in parent_mating_pct_bins]}")
    print(f"All outputs will be saved inside: ./{MASTER_SWEEP_DIR}/")

    # Data structures to hold our sweep results
    all_bins_costs = []
    avg_costs = []
    pct_labels = []  # For clean plotting labels

    # --- 3. EXPERIMENT EXECUTION ---
    for pct in parent_mating_pct_bins:
        # Calculate absolute number of parents and ensure it's an integer >= 2
        num_parents = int(base_config['population_size'] * pct)
        num_parents = max(2, num_parents)

        pct_label = int(pct * 100)
        pct_labels.append(f"{pct_label}%")

        print(f"\n{'#' * 60}")
        print(f"INITIALIZING GA EXPERIMENT: Mating = {pct_label}% of Population (N={num_parents})")
        print(f"{'#' * 60}")

        run_config = base_config.copy()
        run_config.update(ga_static_config)
        run_config['num_parents_mating'] = num_parents

        # Nest the bin folder inside the master directory using the percentage
        bin_folder_name = os.path.join(MASTER_SWEEP_DIR, f"bin_mating_{pct_label}pct")
        run_config['output_folder'] = bin_folder_name

        # Instantiate and run
        optimizer = GAOptimizer(run_config, tf_params)
        optimizer.run_experiment()

        # Extract the 10 trial costs from this bin's history
        bin_costs = optimizer.agg_history['costs']

        # Store for plotting
        all_bins_costs.append(bin_costs)
        avg_costs.append(np.mean(bin_costs))

    # --- 4. SWEEP VISUALIZATION ---
    print("\nGenerating Sweep Visualizations...")

    # Plot 1: Line Graph (Average Cost per Bin)
    plt.figure(figsize=(10, 6))
    plt.plot(pct_labels, avg_costs, marker='o', linestyle='-', color='b', linewidth=2)
    plt.title('Average ITAE Cost vs. Population Mating Percentage')
    plt.xlabel('Percentage of Population Mating')
    plt.ylabel('Average Log10(ITAE) Cost')
    plt.grid(True)
    plt.tight_layout()

    # Save directly to the master folder
    line_plot_path = os.path.join(MASTER_SWEEP_DIR, 'average_cost_line_plot.png')
    plt.savefig(line_plot_path)
    plt.close()
    print(f"Saved Line Plot: {line_plot_path}")

    # Plot 2: Box Plot (Distribution of Trials per Bin)
    plt.figure(figsize=(12, 6))
    plt.boxplot(all_bins_costs, tick_labels=pct_labels)
    plt.title(f'Distribution of ITAE Costs ({base_config["n_rounds"]} Trials/Bin)')
    plt.xlabel('Percentage of Population Mating')
    plt.ylabel('Log10(ITAE) Cost')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save directly to the master folder
    box_plot_path = os.path.join(MASTER_SWEEP_DIR, 'cost_distribution_boxplot.png')
    plt.savefig(box_plot_path)
    plt.close()
    print(f"Saved Box Plot: {box_plot_path}")

    print("\nSweep Complete!")