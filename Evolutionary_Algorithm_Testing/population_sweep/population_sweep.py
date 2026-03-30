import os
import glob
import multiprocessing
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm
import csv
from pathlib import Path
import math
import matplotlib.pyplot as plt

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
    worker_start_time = time.time()
    algo_name, pop_size, base_config, tf_params, base_output_folder, algo_specific_config = task
    output_folder = os.path.join(base_output_folder, algo_name, f"pop_{pop_size}")

    # --- MICRO-LEVEL RESUME CHECK ---
    # If the CSV already exists, skip computation and just read the data
    search_pattern = os.path.join(output_folder, "*.csv")
    found_files = glob.glob(search_pattern)

    if found_files:
        try:
            latest_file = max(found_files, key=os.path.getmtime)
            df_log = pd.read_csv(latest_file, header=None)
            iter_results = df_log.iloc[:, 1].tolist()
            cost_results = df_log.iloc[:, 2].tolist()
            kp_results = df_log.iloc[:, 3].tolist()
            ki_results = df_log.iloc[:, 4].tolist()
        except Exception as e:
            print(f"Error reading existing CSV for {algo_name} Pop {pop_size}: {e}")
            cost_results, kp_results, ki_results, iter_results = [], [], [], []
            
        return {
            "algo": algo_name,
            "pop_size": pop_size,
            "cost": cost_results,
            "kp": kp_results,
            "ki": ki_results,
            "iters": iter_results,
            "elapsed_time": 0.0  # 0 seconds spent computing this run
        }
    # --------------------------------

    # --- RUN OPTIMIZER IF NO DATA EXISTS ---
    run_config = base_config.copy()
    run_config.update(algo_specific_config)
    run_config['pop_size'] = pop_size
    run_config['population_size'] = pop_size
    run_config['output_folder'] = output_folder

    # Dynamic scaling for GA parameters
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

    cost_results = []
    kp_results = []
    ki_results = []
    iter_results = []

    found_files = glob.glob(search_pattern)

    if found_files:
        try:
            latest_file = max(found_files, key=os.path.getmtime)
            df_log = pd.read_csv(latest_file, header=None)
            iter_results = df_log.iloc[:, 1].tolist()
            cost_results = df_log.iloc[:, 2].tolist()
            kp_results = df_log.iloc[:, 3].tolist()
            ki_results = df_log.iloc[:, 4].tolist()
        except Exception as e:
            print(f"Error reading CSV for {algo_name} Pop {pop_size}: {e}")

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
    filename = os.path.join(base_output_dir, f"checkpoint_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(filename, index=False)
    print(f"Checkpoint saved to: {filename}")
    return df


# --- 3. EXECUTION ---
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    total_cores = multiprocessing.cpu_count()

    # Calculate 50% and ensure it's at least 1 core
    num_cores = max(1, math.floor(total_cores * 0.5))

    START_POP = 10
    END_POP = 100
    STEP_SIZE = 10

    shared_config = {
        "patience_limit": 25,
        "max_iters": 200,
        "tol": 1.0,
        "improvement_tol": 1.0,
        "n_rounds": 50
    }

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

    transfer_functions_to_run = [
        {
            "base_dir": "results_population_sweep_do_feb5_daytime",
            "tf_params": {
                'tf_num': [1.359],
                'tf_den': [1745.481, 1],
                'tf_delay': 0.0,
                'tf_n_pade': 2,
                'computed_delay': 0.05,
                'is_reverse_acting': False,
                'max_kp': 100.0
            }
        },
        {
            "base_dir": "results_population_sweep_do_feb5_nighttime",
            "tf_params": {
                'tf_num': [2.336],
                'tf_den': [3086.933, 1],
                'tf_delay': 0.0,
                'tf_n_pade': 2,
                'computed_delay': 0.05,
                'is_reverse_acting': False,
                'max_kp': 100.0
            }
        },
    ]

    total_global_start_time = time.time()
    script_dir = Path(__file__).parent.resolve()
    batch_dir = "BATCH_TEST"

    with multiprocessing.Pool(processes=num_cores) as pool:
        for idx, tf_config in enumerate(transfer_functions_to_run):
            BASE_OUTPUT_DIR = script_dir / batch_dir / tf_config["base_dir"]
            
            print(f"\n{'='*60}")
            print(f"PROCESSING TRANSFER FUNCTION {idx + 1} OF {len(transfer_functions_to_run)}")
            print(f"Output Directory: {BASE_OUTPUT_DIR}")
            print(f"{'='*60}")

            os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

            # --- MACRO-LEVEL RESUME CHECK ---
            existing_checkpoints = glob.glob(os.path.join(BASE_OUTPUT_DIR, "checkpoint_sweep_*.csv"))
            if existing_checkpoints:
                print(f"[RESUME] Skipping TF {idx + 1} ({tf_config['base_dir']}): Master checkpoint already exists.")
                continue
            # --------------------------------

            start_time_sec = time.time()
            start_datetime = datetime.now()
            timestamp_str = start_datetime.strftime('%Y%m%d_%H%M%S')
            print(f"--- SWEEP STARTED: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')} ---")
            print(f"Running on {num_cores} cores to prevent Memory Overflows...")

            tasks = [(algo, size, shared_config, tf_config["tf_params"], BASE_OUTPUT_DIR, algo_specific_configs.get(algo, {}))
                     for algo in ALGO_MAP.keys()
                     for size in pop_sizes]

            raw_results = list(tqdm(pool.imap(worker, tasks), total=len(tasks)))

            df_results = save_checkpoint(raw_results, BASE_OUTPUT_DIR)

            if df_results.empty:
                print("No data collected for this transfer function.")

            print("\nGenerating combined cost history graphs for each population size...")
            target_round = shared_config['n_rounds']  # Usually 50

            for pop_size in pop_sizes:
                plt.figure(figsize=(10, 6))
                lines_plotted = 0

                # Colors mapped for consistency across graphs
                color_map = {'DE': '#1f77b4', 'GA': '#ff7f0e', 'PSO': '#2ca02c'}

                for algo in ALGO_MAP.keys():
                    # Construct path to the saved raw history
                    algo_dir = BASE_OUTPUT_DIR / algo / f"pop_{pop_size}"
                    history_file = algo_dir / f"raw_cost_history_round_{target_round:03d}.csv"

                    if history_file.exists():
                        try:
                            df_hist = pd.read_csv(history_file)
                            plt.plot(
                                df_hist['Iteration'],
                                df_hist['Cost'],
                                linewidth=2.5,
                                color=color_map.get(algo, 'black'),
                                label=f"{algo} (Final Cost: {df_hist['Cost'].iloc[-1]:.4f})"
                            )
                            lines_plotted += 1
                        except Exception as e:
                            print(f"Failed to plot {algo} pop {pop_size}: {e}")

                if lines_plotted > 0:
                    plt.title(f'Algorithm Comparison: Cost Convergence - Pop {pop_size} (Round {target_round})',
                              fontsize=14, fontweight='bold')
                    plt.ylabel('ITAE Cost (log10)', fontsize=12)
                    plt.xlabel('Iteration', fontsize=12)
                    plt.grid(True, which='both', linestyle=':', linewidth=0.7)
                    plt.legend(loc='upper right', fontsize=11)

                    plot_path = BASE_OUTPUT_DIR / f'combined_cost_history_pop_{pop_size:03d}.png'
                    plt.tight_layout()
                    plt.savefig(plot_path, dpi=300)

                plt.close()

            end_time_sec = time.time()
            end_datetime = datetime.now()
            elapsed_seconds = end_time_sec - start_time_sec

            m, s = divmod(elapsed_seconds, 60)
            h, m = divmod(m, 60)
            elapsed_formatted = f"{int(h):02d}:{int(m):02d}:{s:05.2f}"

            print(f"\n--- SWEEP FINISHED: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')} ---")
            print(f"Time Elapsed (Current TF): {elapsed_formatted} ({elapsed_seconds:.2f} pure seconds)")

            # Save Execution Timing
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

            # Save Worker Timings
            worker_timing_filename = os.path.join(BASE_OUTPUT_DIR, f"execution_timing_workers_{timestamp_str}.csv")
            with open(worker_timing_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Algorithm", "Population Size", "Elapsed Time (HH:MM:SS)", "Elapsed Time (Seconds)"])

                sorted_results = sorted(raw_results, key=lambda x: (x['algo'], x['pop_size']))
                for res in sorted_results:
                    w_sec = res['elapsed_time']
                    wm, ws = divmod(w_sec, 60)
                    wh, wm = divmod(wm, 60)
                    w_formatted = f"{int(wh):02d}:{int(wm):02d}:{ws:05.2f}"
                    writer.writerow([res['algo'], res['pop_size'], w_formatted, round(w_sec, 2)])

    # Final overall execution log
    global_elapsed = time.time() - total_global_start_time
    gm, gs = divmod(global_elapsed, 60)
    gh, gm = divmod(gm, 60)
    print(f"\n{'='*60}")
    print(f"ALL TRANSFER FUNCTIONS PROCESSED.")
    print(f"Total Global Execution Time: {int(gh):02d}:{int(gm):02d}:{gs:05.2f}")
    print(f"{'='*60}\n")