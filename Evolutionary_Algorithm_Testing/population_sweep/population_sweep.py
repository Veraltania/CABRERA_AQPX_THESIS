import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from datetime import datetime
from tqdm import tqdm
import csv

try:
    from Evolutionary_Algorithm_Testing.de.de_optimizer import DEOptimizer
    from Evolutionary_Algorithm_Testing.ga.ga_optimizer import GAOptimizer
    from Evolutionary_Algorithm_Testing.pso.pso_optimizer import PSOOptimizer
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import algorithm modules.\n{e}")
    exit(1)

ALGO_MAP = {'DE': DEOptimizer, 'GA': GAOptimizer, 'PSO': PSOOptimizer}


# --- 1. WORKER FUNCTION ---
def worker(task):
    # Capture the start time for this specific task
    worker_start_time = time.time()

    algo_name, pop_size, base_config, tf_params, base_output_folder, algo_specific_config = task

    # Use exact algo_name to get uppercase folders (GA, DE, PSO)
    output_folder = os.path.join(base_output_folder, algo_name, f"pop_{pop_size}")

    run_config = base_config.copy()
    run_config.update(algo_specific_config)

    # Ensure both naming conventions are satisfied for base class vs subclass overrides
    run_config['pop_size'] = pop_size
    run_config['population_size'] = pop_size
    run_config['output_folder'] = output_folder

    # Dynamic scaling for GA parameters to prevent crashes on low population iterations
    if algo_name == "GA":
        if "mating_ratio" in run_config:
            run_config["num_parents_mating"] = max(2, int(pop_size * run_config.pop("mating_ratio")))
        if "elitism_ratio" in run_config:
            run_config["keep_elitism"] = max(1, int(pop_size * run_config.pop("elitism_ratio")))

    try:
        optimizer_class = ALGO_MAP[algo_name]
        optimizer = optimizer_class(run_config, tf_params)
        optimizer.run_experiment()
    except Exception as e:
        print(f"\n[CRASH] {algo_name} Pop {pop_size}: {e}")
        worker_end_time = time.time()
        return {
            "algo": algo_name,
            "pop_size": pop_size,
            "cost": [],
            "kp": [],
            "ki": [],
            "iters": [],
            "elapsed_time": worker_end_time - worker_start_time
        }

    time.sleep(1.0)

    cost_results = []
    kp_results = []
    ki_results = []
    iter_results = []

    search_pattern = os.path.join(output_folder, "*.csv")
    found_files = glob.glob(search_pattern)

    if found_files:
        try:
            latest_file = max(found_files, key=os.path.getmtime)
            # ea_optimizer natively saves: [current_round, iterations_run, cost, best_Kp, best_Ki]
            df_log = pd.read_csv(latest_file, header=None)

            # Extract data using exact positional column indices matching your base class
            iter_results = df_log.iloc[:, 1].tolist()
            cost_results = df_log.iloc[:, 2].tolist()
            kp_results = df_log.iloc[:, 3].tolist()
            ki_results = df_log.iloc[:, 4].tolist()

        except Exception as e:
            print(f"Error reading CSV for {algo_name} Pop {pop_size}: {e}")

    # Capture the end time and calculate elapsed time for this task
    worker_end_time = time.time()
    return {
        "algo": algo_name,
        "pop_size": pop_size,
        "cost": cost_results,
        "kp": kp_results,
        "ki": ki_results,
        "iters": iter_results,
        "elapsed_time": worker_end_time - worker_start_time
    }


# --- 2. DATA PROCESSING ---
def save_checkpoint(all_data, base_output_dir):
    rows = []
    for entry in all_data:
        if entry['cost']:
            for i in range(len(entry['cost'])):
                rows.append({
                    "Algorithm": entry['algo'],
                    "Population Size": entry['pop_size'],
                    "Trial_Number": i,
                    "Final_Cost": entry['cost'][i],
                    "Kp": entry['kp'][i],
                    "Ki": entry['ki'][i],
                    "Iterations": entry['iters'][i]
                })

    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Save checkpoint into the base output directory
    filename = os.path.join(base_output_dir, f"checkpoint_sweep_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
    df.to_csv(filename, index=False)
    print(f"Checkpoint saved to: {filename}")
    return df


# --- 3. PLOTTING FUNCTIONS ---
def generate_standardized_plot(df, base_output_dir):
    if df.empty: return
    print("Generating Clean Grid Plot...")

    g = sns.FacetGrid(
        df,
        col="Population Size",
        hue="Algorithm",
        col_wrap=3,
        height=3.5,
        aspect=1.5,
        sharey=True,
        sharex=True
    )

    g.map(sns.lineplot, "Trial_Number", "Final_Cost",
          marker="o", markersize=4, linewidth=1.5, alpha=0.8)

    for ax in g.axes.flat:
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    g.add_legend(title="Algorithm")
    g.set_axis_labels("Trial Number", "Final Cost - Log Scale")

    plt.tight_layout()
    # Save plot into the base output directory
    filename = os.path.join(base_output_dir, "line_graphs_population_sweep.png")
    plt.savefig(filename, dpi=300)
    print(f"[SUCCESS] Plot saved as '{filename}'")
    plt.close()  # Close to prevent overlapping with next plots


def generate_iteration_plots(df, base_output_dir):
    """Generates a side-by-side Boxplot and Lineplot for Iteration analysis."""
    if df.empty: return
    print("Generating Iteration Analysis Plots...")

    # Set up a 1x2 grid for the plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Boxplot (Distribution & Variance) ---
    sns.boxplot(
        data=df,
        x="Population Size",
        y="Iterations",
        hue="Algorithm",
        ax=axes[0],
        palette="Set2"
    )
    axes[0].set_title("Iteration Distribution per Population Size")
    axes[0].set_ylabel("Number of Iterations")
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # --- Plot 2: Line Graph (Average Iterations) ---
    # sns.lineplot automatically calculates the mean and draws a confidence interval
    sns.lineplot(
        data=df,
        x="Population Size",
        y="Iterations",
        hue="Algorithm",
        marker="o",
        linewidth=2,
        ax=axes[1],
        palette="Set2"
    )
    axes[1].set_title("Average Iterations vs. Population Size")
    axes[1].set_ylabel("Average Iterations")
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # Ensure x-axis only shows integers for population sizes
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    filename = os.path.join(base_output_dir, "iteration_analysis_sweep.png")
    plt.savefig(filename, dpi=300)
    print(f"[SUCCESS] Iteration analysis plot saved as '{filename}'")
    plt.close()


# --- 4. EXECUTION ---
if __name__ == "__main__":
    # 1. Capture total start time
    start_time_sec = time.time()
    start_datetime = datetime.now()
    timestamp_str = start_datetime.strftime('%Y%m%d_%H%M%S')
    print(f"\n--- EXECUTION STARTED: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')} ---")

    START_POP = 10
    END_POP = 100
    STEP_SIZE = 10

    # Easily editable base output directory name
    BASE_OUTPUT_DIR = "results_population_sweep_tds_tf1"

    # Create the root folder if it doesn't already exist to prevent PathNotFoundError
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    shared_config = {
        "patience_limit": 25,
        "max_iters": 200,
        "tol": 1.0,
        "improvement_tol": 1.0,
        "n_rounds": 50
    }

    tf_params = {
        'tf_num': [45.52],
        'tf_den': [2654.54, 1],
        'tf_delay': 0.50,
        'tf_n_pade': 2,
        'computed_delay': 0.50,
        'is_reverse_acting': False,
        'max_kp': 100.0
    }

    # Algorithm-specific configuration overrides
    algo_specific_configs = {
        "GA": {
            "mating_ratio": 0.55,
            "elitism_ratio": 0.05,
            "parent_selection_type": "rank",
            "crossover_type": "scattered",
        },
        "DE": {
            "mutation": (0.5, 1.0),
            "recombination": 0.745,
            "strategy": "best1bin"
        },
        "PSO": {
            "c1": 2.0,
            "c2": 2.0,
            "w": 0.6,
            "v_min": -1.0,
            "v_max": 1.0
        }
    }

    pop_sizes = list(range(START_POP, END_POP + 1, STEP_SIZE))
    if pop_sizes[-1] != END_POP: pop_sizes.append(END_POP)

    tasks = [(algo, size, shared_config, tf_params, BASE_OUTPUT_DIR, algo_specific_configs.get(algo, {}))
             for algo in ALGO_MAP.keys()
             for size in pop_sizes]

    print(f"--- STARTING SWEEP (Output Directory: {BASE_OUTPUT_DIR}) ---")
    print("Running SEQUENTIALLY to prevent Raspberry Pi memory overflow...")

    # ==========================================
    # --- SEQUENTIAL EXECUTION BLOCK ---
    # ==========================================
    raw_results = []
    for task in tqdm(tasks, total=len(tasks), desc="Executing Sweep"):
        result = worker(task)
        raw_results.append(result)

    # Pass base directory to saving functions
    df_results = save_checkpoint(raw_results, BASE_OUTPUT_DIR)

    if not df_results.empty:
        generate_standardized_plot(df_results, BASE_OUTPUT_DIR)
        generate_iteration_plots(df_results, BASE_OUTPUT_DIR)
    else:
        print("No data collected.")

    # 2. Capture total end time and calculate elapsed
    end_time_sec = time.time()
    end_datetime = datetime.now()
    elapsed_seconds = end_time_sec - start_time_sec

    # 3. Format total time to HH:MM:SS
    m, s = divmod(elapsed_seconds, 60)
    h, m = divmod(m, 60)
    elapsed_formatted = f"{int(h):02d}:{int(m):02d}:{s:05.2f}"

    # 4. Print total timing to console
    print(f"\n--- EXECUTION FINISHED: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')} ---")
    print(f"Total Time Elapsed: {elapsed_formatted} ({elapsed_seconds:.2f} pure seconds)")

    # 5. Save TOTAL timing to CSV in the base directory
    total_timing_filename = os.path.join(BASE_OUTPUT_DIR, f"execution_timing_total_{timestamp_str}.csv")

    with open(total_timing_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Start Time", "End Time", "Elapsed Time (HH:MM:SS)", "Elapsed Time (Seconds)"])
        writer.writerow([
            start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            elapsed_formatted,
            round(elapsed_seconds, 2)
        ])
    print(f"Total timing details saved to: {total_timing_filename}")

    # 6. Save INDIVIDUAL WORKER timings to CSV in the base directory
    worker_timing_filename = os.path.join(BASE_OUTPUT_DIR, f"execution_timing_workers_{timestamp_str}.csv")
    with open(worker_timing_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Algorithm", "Population Size", "Elapsed Time (HH:MM:SS)", "Elapsed Time (Seconds)"])

        # Sort results neatly by Algorithm, then Population Size
        sorted_results = sorted(raw_results, key=lambda x: (x['algo'], x['pop_size']))

        for res in sorted_results:
            w_sec = res['elapsed_time']
            wm, ws = divmod(w_sec, 60)
            wh, wm = divmod(wm, 60)
            w_formatted = f"{int(wh):02d}:{int(wm):02d}:{ws:05.2f}"
            writer.writerow([res['algo'], res['pop_size'], w_formatted, round(w_sec, 2)])

    print(f"Individual worker timings saved to: {worker_timing_filename}\n")