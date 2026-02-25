import os
import glob
import multiprocessing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from datetime import datetime
from tqdm import tqdm

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
    algo_name, pop_size, base_config, tf_params, base_output_folder, algo_specific_config = task

    output_folder = os.path.join(base_output_folder, algo_name.lower(), f"pop_{pop_size}")

    run_config = base_config.copy()
    run_config.update(algo_specific_config)
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
        return {"algo": algo_name, "pop_size": pop_size, "fitness": []}

    time.sleep(1.0)

    fitness_results = []
    search_pattern = os.path.join(output_folder, "*.csv")
    found_files = glob.glob(search_pattern)

    if found_files:
        try:
            latest_file = max(found_files, key=os.path.getmtime)
            df_log = pd.read_csv(latest_file)

            if 'Final_Cost_ITAE' in df_log.columns:
                fitness_results = df_log['Final_Cost_ITAE'].tolist()
            elif 'Cost' in df_log.columns:
                fitness_results = df_log['Cost'].tolist()
            else:
                fitness_results = df_log.iloc[:, 2].tolist()
        except Exception as e:
            print(f"Error reading CSV for {algo_name} Pop {pop_size}: {e}")

    return {"algo": algo_name, "pop_size": pop_size, "fitness": fitness_results}

# --- 2. DATA PROCESSING ---
def save_checkpoint(all_data):
    rows = []
    for entry in all_data:
        if entry['fitness']:
            for i, val in enumerate(entry['fitness']):
                rows.append({
                    "Algorithm": entry['algo'],
                    "Population Size": entry['pop_size'],
                    "Trial_Number": i,
                    "Best Fitness": val
                })

    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    filename = f"checkpoint_sweep_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(filename, index=False)
    print(f"Checkpoint saved to: {filename}")
    return df

# --- 3. THE CLEAN PLOT ---
def generate_standardized_plot(df):
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

    g.map(sns.lineplot, "Trial_Number", "Best Fitness",
          marker="o", markersize=4, linewidth=1.5, alpha=0.8)

    for ax in g.axes.flat:
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    g.add_legend(title="Algorithm")
    g.set_axis_labels("Trial Number", "Cost (ITAE) - Log Scale")

    plt.tight_layout()
    filename = "line_graphs_population_sweep.png"
    plt.savefig(filename, dpi=300)
    print(f"[SUCCESS] Plot saved as '{filename}'")
    plt.show()

# --- 4. EXECUTION ---
if __name__ == "__main__":
    START_POP = 10
    END_POP = 100
    STEP_SIZE = 10
    BASE_OUTPUT_DIR = "results_population_sweep_tds_tf2"

    shared_config = {
        "patience_limit": 25,
        "max_iters": 200,
        "improvement_tol": 1.0,
        "n_rounds": 50
    }

    tf_params = {
        'tf_num': [-24.44],
        'tf_den': [84487.79, 1],
        'tf_delay': 0.50,
        'tf_n_pade': 2
    }

    # Algorithm-specific configuration overrides
    algo_specific_configs = {
        "GA": {
            "mating_ratio": 0.55,   # Converted to a ratio for dynamic scaling!
            "elitism_ratio": 0.05,  # Converted to a ratio for dynamic scaling!
            "mutation_type": "adaptive",
            "crossover_type": "scattered",
        },
        "DE": {
            "mutation": (0.5, 1.0),
            "recombination": 0.7
        },
        "PSO": {
            "phi1": 2.5,
            "phi2": 2.5
        }
    }

    pop_sizes = list(range(START_POP, END_POP + 1, STEP_SIZE))
    if pop_sizes[-1] != END_POP: pop_sizes.append(END_POP)

    # Pass the algo-specific config dictionary into the task tuple
    tasks = [(algo, size, shared_config, tf_params, BASE_OUTPUT_DIR, algo_specific_configs.get(algo, {}))
             for algo in ALGO_MAP.keys()
             for size in pop_sizes]

    num_cores = multiprocessing.cpu_count() - 1 or 1
    multiprocessing.set_start_method('spawn', force=True)

    print(f"--- STARTING SWEEP (Output: {BASE_OUTPUT_DIR}) ---")
    print(f"Running on {num_cores} cores...")

    with multiprocessing.Pool(processes=num_cores) as pool:
        raw_results = list(tqdm(pool.imap(worker, tasks), total=len(tasks)))

    df_results = save_checkpoint(raw_results)

    if not df_results.empty:
        generate_standardized_plot(df_results)
    else:
        print("No data collected.")