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
import numpy as np

try:
    from Evolutionary_Algorithm_Testing.de.de_optimizer import DEOptimizer
    from Evolutionary_Algorithm_Testing.ga.ga_optimizer import GAOptimizer
    from Evolutionary_Algorithm_Testing.pso.pso_optimizer import PSOOptimizer
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import algorithm modules.\n{e}")
    exit(1)

ALGO_MAP = {'DE': DEOptimizer, 'GA': GAOptimizer, 'PSO': PSOOptimizer}
SUMMARY_COLUMNS = [
    "Algorithm",
    "Population Size",
    "Mean Final Cost",
    "Standard Deviation",
    "Mean Convergence Iteration"
]
TRANSFER_FUNCTION_SUMMARY_COLUMNS = [
    "Transfer Function",
    "Algorithm",
    "Mean Final Cost",
    "Standard Deviation",
    "Mean Convergence Iteration"
]
PENALTY_COST_THRESHOLD = 1e8


def read_detailed_log(log_path):
    df_log = pd.read_csv(log_path)
    result_columns = {
        "Iterations": "iters",
        "Best_Cost": "cost",
        "Kp": "kp",
        "Ki": "ki"
    }

    missing_columns = set(result_columns) - set(df_log.columns)
    if missing_columns:
        raise ValueError(
            f"Detailed log is missing columns: {sorted(missing_columns)}"
        )

    clean_log = df_log[list(result_columns)].copy()
    for column in result_columns:
        clean_log[column] = pd.to_numeric(clean_log[column], errors="coerce")
    clean_log = clean_log.dropna()

    return {
        output_name: clean_log[column].tolist()
        for column, output_name in result_columns.items()
    }

# --- 1. WORKER FUNCTION ---
def worker(task):
    worker_start_time = time.time()
    algo_name, pop_size, base_config, tf_params, base_output_folder, algo_specific_config = task
    output_folder = os.path.join(base_output_folder, algo_name, f"pop_{pop_size}")

    # --- MICRO-LEVEL RESUME CHECK ---
    search_pattern = os.path.join(output_folder, "*_detailed_log.csv")
    found_files = glob.glob(search_pattern)

    if found_files:
        try:
            latest_file = max(found_files, key=os.path.getmtime)
            log_results = read_detailed_log(latest_file)
            iter_results = log_results["iters"]
            cost_results = log_results["cost"]
            kp_results = log_results["kp"]
            ki_results = log_results["ki"]
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
            "elapsed_time": 0.0
        }
    # --------------------------------

    # --- RUN OPTIMIZER IF NO DATA EXISTS ---
    run_config = base_config.copy()
    run_config.update(algo_specific_config)
    run_config['pop_size'] = pop_size
    run_config['population_size'] = pop_size
    run_config['output_folder'] = output_folder

    # --- NEW: Inject custom boundaries and flags per transfer function into the optimizer config ---
    keys_to_inject = ['min_kp', 'max_kp', 'min_ki', 'max_ki', 'is_reverse_acting']
    for config_key in keys_to_inject:
        if config_key in tf_params:
            run_config[config_key] = tf_params[config_key]

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

    cost_results, kp_results, ki_results, iter_results = [], [], [], []
    found_files = glob.glob(search_pattern)

    if found_files:
        try:
            latest_file = max(found_files, key=os.path.getmtime)
            log_results = read_detailed_log(latest_file)
            iter_results = log_results["iters"]
            cost_results = log_results["cost"]
            kp_results = log_results["kp"]
            ki_results = log_results["ki"]
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
def prepare_summary_data(df_results):
    required_columns = [
        "Algorithm", "Population Size", "Final_Cost", "Iterations"
    ]
    if df_results.empty or not set(required_columns).issubset(df_results.columns):
        return pd.DataFrame(columns=required_columns)

    clean_results = df_results[required_columns].copy()
    for column in ["Population Size", "Final_Cost", "Iterations"]:
        clean_results[column] = pd.to_numeric(
            clean_results[column], errors="coerce"
        )

    clean_results = clean_results.dropna(subset=required_columns)
    clean_results = clean_results[
        np.isfinite(clean_results["Final_Cost"])
        & np.isfinite(clean_results["Iterations"])
        & (clean_results["Final_Cost"] < PENALTY_COST_THRESHOLD)
    ]
    return clean_results


def summarize_results(df_results):
    clean_results = prepare_summary_data(df_results)
    if clean_results.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    return (
        clean_results
        .groupby(["Algorithm", "Population Size"], as_index=False)
        .agg(
            **{
                "Mean Final Cost": ("Final_Cost", "mean"),
                "Standard Deviation": ("Final_Cost", "std"),
                "Mean Convergence Iteration": ("Iterations", "mean")
            }
        )
        [SUMMARY_COLUMNS]
    )


def summarize_across_populations(df_results):
    clean_results = prepare_summary_data(df_results)
    if clean_results.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    summary = (
        clean_results
        .groupby("Algorithm", as_index=False)
        .agg(
            **{
                "Mean Final Cost": ("Final_Cost", "mean"),
                "Standard Deviation": ("Final_Cost", "std"),
                "Mean Convergence Iteration": ("Iterations", "mean")
            }
        )
    )
    summary.insert(1, "Population Size", "All Populations")
    return summary[SUMMARY_COLUMNS]


def summarize_transfer_function(df_results, transfer_function):
    summary = summarize_across_populations(df_results)
    if summary.empty:
        return pd.DataFrame(columns=TRANSFER_FUNCTION_SUMMARY_COLUMNS)

    summary = summary.drop(columns="Population Size")
    summary.insert(0, "Transfer Function", transfer_function)
    return summary[TRANSFER_FUNCTION_SUMMARY_COLUMNS]


def save_transfer_function_summary(summaries, batch_output_dir):
    valid_summaries = [summary for summary in summaries if not summary.empty]
    if valid_summaries:
        combined_summary = pd.concat(valid_summaries, ignore_index=True)
    else:
        combined_summary = pd.DataFrame(
            columns=TRANSFER_FUNCTION_SUMMARY_COLUMNS
        )

    batch_output_dir = Path(batch_output_dir)
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        batch_output_dir / "algorithm_summary_by_transfer_function.csv"
    )
    combined_summary.to_csv(summary_path, index=False)
    print(f"Transfer-function summary saved to: {summary_path}")
    return combined_summary


def save_summary_reports(df_results, base_output_dir):
    base_output_dir = Path(base_output_dir)
    population_summary = summarize_results(df_results)

    for pop_size in population_summary["Population Size"].unique():
        pop_summary = population_summary[
            population_summary["Population Size"] == pop_size
        ]
        summary_path = (
            base_output_dir / f"algorithm_summary_pop_{int(pop_size):03d}.csv"
        )
        pop_summary.to_csv(summary_path, index=False)

    overall_summary = summarize_across_populations(df_results)
    combined_summary = pd.concat(
        [population_summary, overall_summary], ignore_index=True
    )
    combined_summary.to_csv(
        base_output_dir / "algorithm_summary_all_populations.csv",
        index=False
    )
    print(f"Summary reports saved to: {base_output_dir}")
    return population_summary


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
    filename = os.path.join(base_output_dir, f"checkpoint_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(filename, index=False)
    print(f"Checkpoint saved to: {filename}")
    return df


# --- 3. EXECUTION ---
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    total_cores = multiprocessing.cpu_count()
    num_cores = max(1, math.floor(total_cores * 0.75))

    START_POP = 10
    END_POP = 100
    STEP_SIZE = 10

    shared_config = {
        "patience_limit": 25,
        "max_iters": 100,
        "tol": 0.01,
        "improvement_tol": 0.01,
        "n_rounds": 50,
        "weights": [1.0, 1.0, 1.0, 1.0]
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
            "recombination": 0.75,
            "strategy": "best1bin"
        },
        "PSO": {
            "c1": 2.0,
            "c2": 2.0,
            "w": 0.8,
            "v_min": -1.0,
            "v_max": 1.0
        }
    }

    pop_sizes = list(range(START_POP, END_POP + 1, STEP_SIZE))
    if pop_sizes[-1] != END_POP: pop_sizes.append(END_POP)

    batch_dir = "BATCH_OPENLOOP_CONTROL_EFFORT_V2"
    sweep_type = "population_sweep"

    # --- DEFINING TRANSFER FUNCTIONS WITH SPECIFIC Kp & Ki BOUNDS ---
    max_kp_do = 1.5
    min_kp_do = 0
    max_ki_do = 0.005
    min_ki_do = 0

    # Adjusted limits for massive Tp (71,160s) to prevent integral windup
    max_kp_tds = 0 
    min_kp_tds = -1
    max_ki_tds = 0
    min_ki_tds = -0.0005 

    transfer_functions = {
        f"{sweep_type}_do_feb5_daytime": {
            'tf_num': [1.346], 'tf_den': [1551.955, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 
            'min_kp': min_kp_do, 'max_kp': max_kp_do, 'min_ki': min_ki_do, 'max_ki': max_ki_do
        },
        f"{sweep_type}_do_feb7_daytime": {
            'tf_num': [1.133], 'tf_den': [2833.82, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 
            'min_kp': min_kp_do, 'max_kp': max_kp_do, 'min_ki': min_ki_do, 'max_ki': max_ki_do
        },
        f"{sweep_type}_do_feb25_daytime": {
            'tf_num': [2.287], 'tf_den': [3010.296, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 
            'min_kp': min_kp_do, 'max_kp': max_kp_do, 'min_ki': min_ki_do, 'max_ki': max_ki_do
        },
        f"{sweep_type}_do_feb26_daytime": {
            'tf_num': [2.430], 'tf_den': [3492.589, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 
            'min_kp': min_kp_do, 'max_kp': max_kp_do, 'min_ki': min_ki_do, 'max_ki': max_ki_do
        },
        f"{sweep_type}_do_feb5_nighttime": {
            'tf_num': [2.355], 'tf_den': [3083.590, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 
            'min_kp': min_kp_do, 'max_kp': max_kp_do, 'min_ki': min_ki_do, 'max_ki': max_ki_do
        },
        f"{sweep_type}_do_feb7_nighttime": {
            'tf_num': [2.049], 'tf_den': [4499.996, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 
            'min_kp': min_kp_do, 'max_kp': max_kp_do, 'min_ki': min_ki_do, 'max_ki': max_ki_do
        },
        f"{sweep_type}_do_feb25_nighttime": {
            'tf_num': [3.923], 'tf_den': [3012.232, 1], 'tf_delay': 0.0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 
            'min_kp': min_kp_do, 'max_kp': max_kp_do, 'min_ki': min_ki_do, 'max_ki': max_ki_do
        },
        f"{sweep_type}_do_feb26_nighttime": {
            'tf_num': [3.132], 'tf_den': [2530.052, 1], 'tf_delay': 0,
            'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 
            'min_kp': min_kp_do, 'max_kp': max_kp_do, 'min_ki': min_ki_do, 'max_ki': max_ki_do
        }
    }


    total_global_start_time = time.time()
    script_dir = Path(__file__).parent.resolve()
    transfer_function_summaries = []

    with multiprocessing.Pool(processes=num_cores) as pool:
        for idx, (base_dir, tf_params) in enumerate(transfer_functions.items()):
            BASE_OUTPUT_DIR = script_dir / batch_dir / base_dir
            
            print(f"\n{'='*60}")
            print(f"PROCESSING TRANSFER FUNCTION {idx + 1} OF {len(transfer_functions)}")
            print(f"Output Directory: {BASE_OUTPUT_DIR}")
            print(f"{'='*60}")

            os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

            start_time_sec = time.time()
            start_datetime = datetime.now()
            timestamp_str = start_datetime.strftime('%Y%m%d_%H%M%S')
            existing_checkpoints = glob.glob(os.path.join(BASE_OUTPUT_DIR, "checkpoint_sweep_*.csv"))
            if existing_checkpoints:
                latest_checkpoint = max(existing_checkpoints, key=os.path.getmtime)
                print(
                    f"[RESUME] Loading checkpoint for TF {idx + 1} "
                    f"({base_dir}): {latest_checkpoint}"
                )
                df_results = pd.read_csv(latest_checkpoint)
                raw_results = []
            else:
                print(f"--- SWEEP STARTED: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')} ---")
                tasks = [
                    (
                        algo, size, shared_config, tf_params, BASE_OUTPUT_DIR,
                        algo_specific_configs.get(algo, {})
                    )
                    for algo in ALGO_MAP.keys()
                    for size in pop_sizes
                ]

                raw_results = list(
                    tqdm(pool.imap(worker, tasks), total=len(tasks))
                )
                df_results = save_checkpoint(raw_results, BASE_OUTPUT_DIR)

            population_summary = save_summary_reports(
                df_results, BASE_OUTPUT_DIR
            )
            transfer_function_summaries.append(
                summarize_transfer_function(df_results, base_dir)
            )
            save_transfer_function_summary(
                transfer_function_summaries, script_dir / batch_dir
            )

            # --- COST HISTORY GRAPHING WITH BROKEN Y-AXIS ---
            print("\nGenerating combined cost history graphs for each population size...")
            target_round = shared_config['n_rounds']  
            target_max_iter = 50  

            for pop_size in pop_sizes:
                color_map = {'DE': '#1f77b4', 'GA': '#ff7f0e', 'PSO': '#2ca02c'}
                loaded_data = {}

                # 1. Load Data
                for algo in ALGO_MAP.keys():
                    algo_dir = BASE_OUTPUT_DIR / algo / f"pop_{pop_size}"
                    history_file = algo_dir / f"raw_cost_history_round_{target_round:03d}.csv"

                    if history_file.exists():
                        try:
                            df_hist = pd.read_csv(history_file)
                            df_hist = df_hist[df_hist['Iteration'] <= target_max_iter]
                            if not df_hist.empty:
                                loaded_data[algo] = df_hist
                        except Exception as e:
                            print(f"Failed to load {algo} pop {pop_size}: {e}")

                if not loaded_data:
                    continue

                # 2. Find valid cost ranges to setup the split zoom
                min_valid_cost = float('inf')
                max_valid_cost = 0.0
                for algo, df_hist in loaded_data.items():
                    valid_costs = df_hist['Cost'][df_hist['Cost'] < 1e8]
                    if not valid_costs.empty:
                        min_valid_cost = min(min_valid_cost, valid_costs.min())
                        max_valid_cost = max(max_valid_cost, valid_costs.max())

                if min_valid_cost == float('inf'):
                    continue

                # Setup broken axis (top: zoom range, bottom: anchors to 0)
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6), gridspec_kw={'height_ratios': [4, 1]})
                fig.subplots_adjust(hspace=0.08)

                # 3. Plot Data on both axes
                for algo, df_hist in loaded_data.items():
                    last_iter = df_hist['Iteration'].iloc[-1]
                    last_cost = df_hist['Cost'].iloc[-1]
                    algo_summary = population_summary[
                        (population_summary["Algorithm"] == algo)
                        & (population_summary["Population Size"] == pop_size)
                    ]

                    if algo_summary.empty:
                        legend_label = f"{algo} (Final Cost: {last_cost:.4f})"
                    else:
                        summary_row = algo_summary.iloc[0]
                        legend_label = (
                            f"{algo} ($\\mu$={summary_row['Mean Final Cost']:.4f}, "
                            f"$\\sigma$={summary_row['Standard Deviation']:.4f}, "
                            f"mean iter={summary_row['Mean Convergence Iteration']:.1f})"
                        )

                    if last_iter < target_max_iter:
                        pad_iters = list(range(int(last_iter) + 1, target_max_iter + 1))
                        pad_df = pd.DataFrame({'Iteration': pad_iters, 'Cost': [last_cost] * len(pad_iters)})
                        df_hist = pd.concat([df_hist, pad_df], ignore_index=True)

                    # Plot on Top
                    ax1.plot(df_hist['Iteration'], df_hist['Cost'], linewidth=2.5, 
                             color=color_map.get(algo, 'black'), label=legend_label)
                    # Plot on Bottom
                    ax2.plot(df_hist['Iteration'], df_hist['Cost'], linewidth=2.5, color=color_map.get(algo, 'black'))

                # Determine the zoomed range for the top graph
                y_margin = (max_valid_cost - min_valid_cost) * 0.1 if max_valid_cost > min_valid_cost else min_valid_cost * 0.1
                if y_margin == 0: y_margin = 0.1
                
                top_min = min_valid_cost - y_margin
                if top_min <= 0:
                    top_min = min_valid_cost * 0.5 # Force a break margin if it naturally drops to 0

                ax1.set_ylim(top_min, max_valid_cost + y_margin)
                ax2.set_ylim(0, top_min * 0.15) # Only show the absolute bottom near 0

                # Hide the spines bridging the gap
                ax1.spines['bottom'].set_visible(False)
                ax2.spines['top'].set_visible(False)
                ax1.xaxis.tick_top()
                ax1.tick_params(labeltop=False, bottom=False) 
                ax2.xaxis.tick_bottom()

                # Add the two red, dotted lines for the break effect
                ax1.axhline(ax1.get_ylim()[0], color='red', linestyle=':', linewidth=2.5)
                ax2.axhline(ax2.get_ylim()[1], color='red', linestyle=':', linewidth=2.5)

                # Styling
                ax1.set_title(f'Algorithm Comparison: Cost Convergence - Pop {pop_size}', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Iteration', fontsize=12)
                fig.text(0.04, 0.5, 'Cost', va='center', rotation='vertical', fontsize=12)
                
                ax1.set_xlim(0, target_max_iter)
                ax2.set_xlim(0, target_max_iter)
                
                ax1.grid(True, which='both', linestyle=':', linewidth=0.7)
                ax2.grid(True, which='both', linestyle=':', linewidth=0.7)
                ax1.legend(loc='upper right', fontsize=9)

                plot_path = BASE_OUTPUT_DIR / f'combined_cost_history_pop_{pop_size:03d}.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()

            end_time_sec = time.time()
            end_datetime = datetime.now()
            elapsed_seconds = end_time_sec - start_time_sec
            m, s = divmod(elapsed_seconds, 60)
            h, m = divmod(m, 60)
            elapsed_formatted = f"{int(h):02d}:{int(m):02d}:{s:05.2f}"

            print(f"\n--- SWEEP FINISHED: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')} ---")
            
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

            worker_timing_filename = os.path.join(BASE_OUTPUT_DIR, f"execution_timing_workers_{timestamp_str}.csv")
            with open(worker_timing_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Algorithm", "Population Size", "Elapsed Time (HH:MM:SS)", "Elapsed Time (Seconds)"])
                for res in sorted(raw_results, key=lambda x: (x['algo'], x['pop_size'])):
                    w_sec = res['elapsed_time']
                    wm, ws = divmod(w_sec, 60)
                    wh, wm = divmod(wm, 60)
                    w_formatted = f"{int(wh):02d}:{int(wm):02d}:{ws:05.2f}"
                    writer.writerow([res['algo'], res['pop_size'], w_formatted, round(w_sec, 2)])

    global_elapsed = time.time() - total_global_start_time
    gm, gs = divmod(global_elapsed, 60)
    gh, gm = divmod(gm, 60)
    print(f"\n{'='*60}\nALL TRANSFER FUNCTIONS PROCESSED.")
    print(f"Total Global Execution Time: {int(gh):02d}:{int(gm):02d}:{gs:05.2f}\n{'='*60}\n")
