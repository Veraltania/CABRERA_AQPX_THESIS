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

# --- IMPORTS ---
try:
    from Evolutionary_Algorithm_Testing.de.test_de import run_de_experiment
    from Evolutionary_Algorithm_Testing.ga.test_ga import run_ga_experiment
    from Evolutionary_Algorithm_Testing.pso.test_pso import run_pso_experiment
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import algorithm modules.\n{e}")
    exit(1)

ALGO_MAP = {'DE': run_de_experiment, 'GA': run_ga_experiment, 'PSO': run_pso_experiment}


# --- 1. WORKER FUNCTION ---
def worker(task):
    algo_name, pop_size, config = task
    output_folder = f"results/{algo_name.lower()}/pop_{pop_size}"
    os.makedirs(output_folder, exist_ok=True)

    try:
        ALGO_MAP[algo_name](population_size=pop_size, output_folder=output_folder, **config)
    except Exception as e:
        print(f"\n[CRASH] {algo_name} Pop {pop_size}: {e}")
        return {"algo": algo_name, "pop_size": pop_size, "fitness": []}

    time.sleep(1.0)  # Safety buffer

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
        except:
            pass

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
    df.to_csv(f"checkpoint_sweep_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", index=False)
    return df


# --- 3. THE CLEAN PLOT (No Title) ---
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
        sharey=True,  # Keeps Y-axis locked/standardized
        sharex=True
    )

    g.map(sns.lineplot, "Trial_Number", "Best Fitness",
          marker="o", markersize=4, linewidth=1.5, alpha=0.8)

    # --- STYLING ---
    for ax in g.axes.flat:
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    g.add_legend(title="Algorithm")
    g.set_axis_labels("Trial Number", "Cost (ITAE) - Log Scale")

    # Removed g.fig.suptitle() to prevent overlapping mess
    plt.tight_layout()

    filename = "line_graphs_population_sweep.png"
    plt.savefig(filename, dpi=300)
    print(f"[SUCCESS] Plot saved as '{filename}'")
    plt.show()


# --- 4. EXECUTION ---
if __name__ == "__main__":
    # --- CONFIGURATION ---
    START_POP = 10
    END_POP = 100  # Change to 100 for full run
    STEP_SIZE = 10

    shared_config = {
        "patience_limit": 25,
        "max_iters": 200,
        "improvement_tol": 1.0,
        "n_rounds": 50,  # Change to 50 for full run
        "tf_num": [44.93], "tf_den": [1474.32, 1], "tf_delay": 343.93, "tf_n_pade": 2
    }

    pop_sizes = list(range(START_POP, END_POP + 1, STEP_SIZE))
    if pop_sizes[-1] != END_POP: pop_sizes.append(END_POP)

    tasks = [(algo, size, shared_config) for algo in ALGO_MAP.keys() for size in pop_sizes]

    num_cores = 2
    multiprocessing.set_start_method('spawn', force=True)

    print(f"--- STARTING SWEEP ---")
    with multiprocessing.Pool(processes=num_cores) as pool:
        raw_results = list(tqdm(pool.imap(worker, tasks), total=len(tasks)))

    df_results = save_checkpoint(raw_results)

    if not df_results.empty:
        generate_standardized_plot(df_results)
    else:
        print("No data collected.")