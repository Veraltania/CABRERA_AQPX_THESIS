import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_pdf_plot(base_dir, pop_size, target_round, target_max_iter):
    """
    Reads algorithm cost history CSVs and generates a broken y-axis plot in PDF format.
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        print(f"Error: Directory '{base_dir}' does not exist.")
        return

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 18
    })

    algorithms = ['DE', 'GA', 'PSO']
    color_map = {'DE': '#1f77b4', 'GA': '#ff7f0e', 'PSO': '#2ca02c'}
    loaded_data = {}

    print(f"Scanning directory: {base_dir}")
    
    # 1. Load Data
    for algo in algorithms:
        # Matches the folder structure from your sweep script
        algo_dir = base_dir / algo / f"pop_{pop_size}"
        history_file = algo_dir / f"raw_cost_history_round_{target_round:03d}.csv"

        if history_file.exists():
            try:
                df_hist = pd.read_csv(history_file)
                # Filter up to target maximum iteration
                df_hist = df_hist[df_hist['Iteration'] <= target_max_iter]
                if not df_hist.empty:
                    loaded_data[algo] = df_hist
                    print(f"  -> Loaded data for {algo}")
            except Exception as e:
                print(f"  -> Failed to load {algo} pop {pop_size}: {e}")
        else:
            print(f"  -> Missing file for {algo}: {history_file}")

    if not loaded_data:
        print("No valid data found. Exiting.")
        return

    # 2. Find valid cost ranges to setup the split zoom
    min_valid_cost = float('inf')
    max_valid_cost = 0.0
    for algo, df_hist in loaded_data.items():
        valid_costs = df_hist['Cost'][df_hist['Cost'] < 1e8]
        if not valid_costs.empty:
            min_valid_cost = min(min_valid_cost, valid_costs.min())
            max_valid_cost = max(max_valid_cost, valid_costs.max())

    if min_valid_cost == float('inf'):
        print("No valid costs (< 1e8) found to plot.")
        return

    # Setup broken axis (top: zoom range, bottom: anchors to 0)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6), gridspec_kw={'height_ratios': [4, 1]})
    fig.subplots_adjust(hspace=0.08)

    # 3. Plot Data on both axes
    for algo, df_hist in loaded_data.items():
        last_iter = df_hist['Iteration'].iloc[-1]
        last_cost = df_hist['Cost'].iloc[-1]

        # Pad iterations if the algorithm converged early
        if last_iter < target_max_iter:
            pad_iters = list(range(int(last_iter) + 1, target_max_iter + 1))
            pad_df = pd.DataFrame({'Iteration': pad_iters, 'Cost': [last_cost] * len(pad_iters)})
            df_hist = pd.concat([df_hist, pad_df], ignore_index=True)

        # Plot on Top
        ax1.plot(df_hist['Iteration'], df_hist['Cost'], linewidth=2.5, 
                 color=color_map.get(algo, 'black'), label=f"{algo} (Final Cost: {last_cost:.4f})")
        # Plot on Bottom
        ax2.plot(df_hist['Iteration'], df_hist['Cost'], linewidth=2.5, color=color_map.get(algo, 'black'))

    # Determine the zoomed range for the top graph
    y_margin = (max_valid_cost - min_valid_cost) * 0.1 if max_valid_cost > min_valid_cost else min_valid_cost * 0.1
    if y_margin == 0: 
        y_margin = 0.1
    
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

    # Styling (Inherits global font size 14, NO BOLD)
    ax1.set_title(f'Cost Convergence Comparison of DE, PSO, and GA')
    ax2.set_xlabel('Iteration')
    fig.text(0.04, 0.5, 'Cost', va='center', rotation='vertical')
    
    ax1.set_xlim(0, target_max_iter)
    ax2.set_xlim(0, target_max_iter)
    
    ax1.grid(True, which='both', linestyle=':', linewidth=0.7)
    ax1.legend(loc='upper right')

    # Save as Vector Graphic (PDF)
    output_filename = base_dir / f'vector_cost_history_pop_{pop_size:03d}.pdf'
    plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSuccess! PDF saved to: {output_filename}")


if __name__ == "__main__":
    # ==========================================
    # --- EDIT YOUR VARIABLES DIRECTLY BELOW ---
    # ==========================================
    
    TARGET_DIR = "Evolutionary_Algorithm_Testing/population_sweep/BATCH_OPENLOOP_CONTROL_EFFORT/population_sweep_do_feb25_nighttime"
    POPULATION_SIZE = 50
    TARGET_ROUND = 50
    MAX_ITERATIONS = 50

    # ==========================================
    
    generate_pdf_plot(
        base_dir=TARGET_DIR, 
        pop_size=POPULATION_SIZE, 
        target_round=TARGET_ROUND, 
        target_max_iter=MAX_ITERATIONS
    )