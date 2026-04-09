import os
import pandas as pd
import math
import matplotlib.pyplot as plt
from pathlib import Path

def generate_broken_axis_replots():
    # --- Configuration ---
    BASE_DATA_DIR = Path("Evolutionary_Algorithm_Testing/population_sweep/BATCH_DO_OPENLOOP")
    START_POP = 10
    END_POP = 100
    STEP_SIZE = 10
    TARGET_ROUND = 50 # Default from your shared config
    
    ALGO_MAP = ['DE', 'GA', 'PSO']
    color_map = {'DE': '#1f77b4', 'GA': '#ff7f0e', 'PSO': '#2ca02c'}

    if not BASE_DATA_DIR.exists():
        print(f"CRITICAL ERROR: Directory {BASE_DATA_DIR} not found.")
        print("Please run this script in the directory containing BATCH_DO_OPENLOOP.")
        return

    # Find all Transfer Function directories inside BATCH_DO_OPENLOOP
    # Filtering out files to only get directories (e.g., population_sweep_do_feb5_daytime)
    tf_dirs = [d for d in BASE_DATA_DIR.iterdir() if d.is_dir()]
    
    if not tf_dirs:
        print(f"No transfer function directories found inside {BASE_DATA_DIR}.")
        return

    pop_sizes = list(range(START_POP, END_POP + 1, STEP_SIZE))
    if pop_sizes[-1] != END_POP: pop_sizes.append(END_POP)

    for tf_dir in tf_dirs:
        tf_name = tf_dir.name
        print(f"\n{'='*60}")
        print(f"PROCESSING TRANSFER FUNCTION: {tf_name}")
        print(f"{'='*60}")
        
        for pop_size in pop_sizes:
            loaded_data = {}
            max_iter_found = 0
            
            # --- Load Data from raw_cost_history_round_xxx.csv ---
            for algo in ALGO_MAP:
                algo_dir = tf_dir / algo / f"pop_{pop_size}"
                
                # Check for the history files
                history_files = list(algo_dir.glob("raw_cost_history_round_*.csv"))
                
                if history_files:
                    # Get the file for the highest round available (usually 050)
                    history_file = max(history_files, key=lambda p: int(p.stem.split('_')[-1]))
                    try:
                        df_hist = pd.read_csv(history_file)
                        loaded_data[algo] = df_hist
                        current_max = df_hist['Iteration'].max()
                        if current_max > max_iter_found:
                            max_iter_found = current_max
                    except Exception as e:
                        print(f"  -> Failed to load {history_file}: {e}")
            
            if not loaded_data:
                # No data for this population size, skip smoothly
                continue
                
            target_max_iter = max(50, int(max_iter_found))
            
            # --- Pad Early Stopping Data ---
            all_costs = []
            for algo, df_hist in loaded_data.items():
                last_iter = df_hist['Iteration'].iloc[-1]
                last_cost = df_hist['Cost'].iloc[-1]

                if last_iter < target_max_iter:
                    pad_iters = list(range(int(last_iter) + 1, target_max_iter + 1))
                    pad_costs = [last_cost] * len(pad_iters)
                    pad_df = pd.DataFrame({'Iteration': pad_iters, 'Cost': pad_costs})
                    
                    # Concat and update dictionary
                    loaded_data[algo] = pd.concat([df_hist, pad_df], ignore_index=True)
                
                all_costs.extend(loaded_data[algo]['Cost'].tolist())

            # --- Calculate Dynamic Boundaries ---
            global_min_cost = min(all_costs)
            global_max_cost = max(all_costs)
            
            floor_min = math.floor(global_min_cost / 10) * 10
            ceil_max = math.ceil(global_max_cost / 10) * 10
            
            # Catch edge case where min and max are in the same 10-unit bracket
            if floor_min >= ceil_max:
                ceil_max = floor_min + 10
                floor_min -= 10

            # Only break the axis if the minimum cost is significantly above 0
            needs_broken_axis = floor_min > 15
            
            plot_path = tf_dir / f'combined_cost_history_pop_{pop_size:03d}_broken_y.png'
            
            # --- Plotting: Broken Y-Axis (Image Style) ---
            if needs_broken_axis:
                y_top_zoom = (floor_min, ceil_max)
                y_bottom_zoom = (0, 10) # Fixed lower bound starting at 0
                
                fig, (ax1, ax2) = plt.subplots(
                    2, 1,
                    sharex=True,
                    figsize=(10, 6),
                    gridspec_kw={'height_ratios': [4, 1]}
                )
                fig.subplots_adjust(hspace=0.15)
                
                for algo, df_hist in loaded_data.items():
                    last_cost = df_hist['Cost'].iloc[-1]
                    color = color_map.get(algo, 'black')
                    lbl = f"{algo} (Final Cost: {last_cost:.4f})"
                    
                    # Plot on both axes
                    ax1.plot(df_hist['Iteration'], df_hist['Cost'], linewidth=2.5, color=color, label=lbl)
                    ax2.plot(df_hist['Iteration'], df_hist['Cost'], linewidth=2.5, color=color)
                
                # Setup boundaries
                ax1.set_ylim(*y_top_zoom)
                ax2.set_ylim(*y_bottom_zoom)
                ax1.set_xlim(0, target_max_iter)
                ax2.set_xlim(0, target_max_iter)

                # Hide the spines between ax1 and ax2
                ax1.spines['bottom'].set_visible(False)
                ax2.spines['top'].set_visible(False)
                
                # Move ticks
                ax1.xaxis.tick_top()
                ax1.tick_params(axis='both', labeltop=False, labelsize=12)
                ax2.xaxis.tick_bottom()
                ax2.tick_params(axis='both', labelsize=12)

                # Add dashed red lines for the break
                ax1.axhline(y=y_top_zoom[0], color='red', linestyle='--', linewidth=2.0, zorder=10)
                ax2.axhline(y=y_bottom_zoom[1], color='red', linestyle='--', linewidth=2.0, zorder=10)

                # Labels
                ax2.set_xlabel("Iteration", fontsize=12)
                fig.supylabel("Cost", fontsize=12)

                title_str = f'Algorithm Comparison: Cost Convergence - Pop {pop_size}\n(Round {TARGET_ROUND} | {tf_name})'
                ax1.set_title(title_str, fontsize=14, fontweight='bold', pad=15)
                ax1.legend(loc='upper right', fontsize=11)
                
                ax1.grid(True, linestyle=':', alpha=0.6)
                ax2.grid(True, linestyle=':', alpha=0.6)

                plt.tight_layout()
                plt.savefig(plot_path, dpi=300)
                plt.close()
                print(f"  -> Saved broken-axis replot: {plot_path.name}")
                
            # --- Plotting: Standard Plot (If cost reaches ~0 natively) ---
            else:
                plt.figure(figsize=(10, 6))
                for algo, df_hist in loaded_data.items():
                    last_cost = df_hist['Cost'].iloc[-1]
                    color = color_map.get(algo, 'black')
                    lbl = f"{algo} (Final Cost: {last_cost:.4f})"
                    plt.plot(df_hist['Iteration'], df_hist['Cost'], linewidth=2.5, color=color, label=lbl)

                plt.title(f'Algorithm Comparison: Cost Convergence - Pop {pop_size}\n(Round {TARGET_ROUND} | {tf_name})',
                          fontsize=14, fontweight='bold')
                plt.ylabel('Cost', fontsize=12)
                plt.xlabel('Iteration', fontsize=12)
                plt.xlim(0, target_max_iter) 
                plt.grid(True, which='both', linestyle=':', linewidth=0.7)
                plt.legend(loc='upper right', fontsize=11)

                plt.tight_layout()
                plt.savefig(plot_path, dpi=300)
                plt.close()
                print(f"  -> Saved standard replot (data near 0): {plot_path.name}")

if __name__ == "__main__":
    generate_broken_axis_replots()