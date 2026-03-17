import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# --- PLOTTING FUNCTIONS ---
def generate_cost_line_graph(df, output_dir):
    if df.empty: return
    print(f"Generating Cost Line Graph in {output_dir}...")

    # Force numeric to prevent silent Seaborn rendering failures
    df["Population Size"] = pd.to_numeric(df["Population Size"], errors='coerce')
    df["Final_Cost"] = pd.to_numeric(df["Final_Cost"], errors='coerce')

    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=df,
        x="Population Size",
        y="Final_Cost",
        hue="Algorithm",
        marker="o",
        linewidth=2,
        errorbar=None, 
        palette="Set1"
    )

    ax.set_title("Final Cost vs. Population Size")
    ax.set_xlabel("Population Size")
    ax.set_ylabel("Final Cost")
    ax.set_ylim(bottom=0)  
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle='--', alpha=0.5)

    # FORCE THE LEGEND
    ax.legend(title="Algorithm", loc='best')

    plt.tight_layout()

    filename = os.path.join(output_dir, "cost_vs_pop_size_linegraph.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved: {filename}")


def generate_iteration_boxplots(df, output_dir):
    if df.empty: return
    print(f"Generating Iteration Box Plots in {output_dir}...")

    # Force numeric to prevent silent Seaborn rendering failures
    df["Population Size"] = pd.to_numeric(df["Population Size"], errors='coerce')
    df["Iterations"] = pd.to_numeric(df["Iterations"], errors='coerce')

    # sharex=False and sharey=False forces Seaborn to give EVERY plot its own axes
    g = sns.catplot(
        data=df,
        x="Algorithm",
        y="Iterations",
        col="Population Size",
        col_wrap=3,
        hue="Algorithm",
        kind="box",
        height=4,
        aspect=1.2,
        sharex=False, 
        sharey=False, 
        palette="Set2",
        legend=False # Disable auto-legend so we can force it below
    )

    g.set_titles("Pop Size: {col_name}")

    # FORCE X and Y labels on EVERY subplot, and lock y-axis to 0
    for ax in g.axes.flat:
        ax.set_xlabel("Algorithm", fontweight='bold')
        ax.set_ylabel("Number of Iterations", fontweight='bold')
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='--', alpha=0.5)

    # FORCE THE LEGEND ON THE ENTIRE GRID
    g.add_legend(title="Algorithm", bbox_to_anchor=(1.02, 0.5), loc='center left')
    
    # Apply tight layout to ensure labels and legends don't overlap
    g.tight_layout()

    filename = os.path.join(output_dir, "iteration_boxplots_sweep.png")
    g.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f" -> Saved: {filename}")


# --- EXECUTION ---
if __name__ == "__main__":
    # Directories from your main script

    base_dirs = [
        "results_population_sweep_do_tf1_daytime",
        "results_population_sweep_do_tf1_nighttime",
        "results_population_sweep_do_tf3_daytime",
        "results_population_sweep_do_tf3_nighttime"
    ]

    script_dir = Path(__file__).parent.resolve()
    batch_dir = "BATCH_2"

    for folder_name in base_dirs:
        target_dir = script_dir / batch_dir / folder_name
        
        if not target_dir.exists():
            print(f"\n[SKIP] Directory not found: {target_dir}")
            continue
            
        # Find the latest checkpoint CSV in the directory
        search_pattern = os.path.join(target_dir, "checkpoint_sweep_*.csv")
        csv_files = glob.glob(search_pattern)
        
        if not csv_files:
            print(f"\n[SKIP] No checkpoint CSV found in {target_dir}")
            continue
            
        # Always grab the most recent run's data
        latest_csv = max(csv_files, key=os.path.getmtime)
        print(f"\nProcessing data from: {latest_csv}")
        
        # Load data and plot
        try:
            df = pd.read_csv(latest_csv)
            generate_cost_line_graph(df, target_dir)
            generate_iteration_boxplots(df, target_dir)
        except Exception as e:
            print(f"[ERROR] Failed to process {latest_csv}: {e}")