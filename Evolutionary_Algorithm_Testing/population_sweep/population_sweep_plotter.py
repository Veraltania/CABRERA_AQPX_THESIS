import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Configuration - Matches your BASE_OUTPUT_DIR
BASE_DIR = "results_population_sweep_tds_tf1"


def load_and_clean_data(base_dir):
    """Finds the latest checkpoint and cleans out header strings from the data."""
    search_pattern = os.path.join(base_dir, "checkpoint_sweep_*.csv")
    list_of_files = glob.glob(search_pattern)

    if not list_of_files:
        print(f"ERROR: No checkpoint CSV found in {base_dir}. Run your sweep first.")
        return None

    latest_file = max(list_of_files, key=os.path.getmtime)
    print(f"Cleaning and loading: {latest_file}")

    df = pd.read_csv(latest_file)

    # CRITICAL FIX: Convert columns to numeric and drop any rows that were header strings
    # This prevents the "useless" categorical axis you saw in the image.
    cols_to_fix = ['Final_Cost', 'Iterations', 'Population Size']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove rows that couldn't be converted (the 'IDIOT' strings in the data)
    df = df.dropna(subset=['Final_Cost', 'Iterations'])
    return df


def generate_report_plots(df):
    if df is None or df.empty:
        return

    sns.set_theme(style="whitegrid")

    # --- PLOT 1: COST BOXPLOTS ---
    # x-axis = Algorithm, y-axis = Cost, Facet = Population
    print("Plotting Cost Boxplots...")
    g1 = sns.catplot(
        data=df, x="Algorithm", y="Final_Cost", col="Population Size",
        kind="box", col_wrap=3, height=4, aspect=1.2,
        sharey=False,  # Allows each population to be seen clearly
        palette="husl"
    )
    for ax in g1.axes.flat:
        ax.set_ylim(0, None)  # Force Y-axis to start at 0
        ax.set_ylabel("Final Cost (ITAE)")
        ax.tick_params(labelbottom=True)  # Ensure Algo names show on every plot

    g1.fig.subplots_adjust(top=0.9)
    g1.fig.suptitle("Cost Distribution per Algorithm and Population Size", fontsize=16)
    g1.savefig(os.path.join(BASE_DIR, "fixed_cost_boxplots.png"), dpi=300)

    # --- PLOT 2: ITERATION BOXPLOTS ---
    # x-axis = Algorithm, y-axis = Iterations, Facet = Population
    print("Plotting Iteration Boxplots...")
    g2 = sns.catplot(
        data=df, x="Algorithm", y="Iterations", col="Population Size",
        kind="box", col_wrap=3, height=4, aspect=1.2,
        sharey=False,
        palette="Set2"
    )
    for ax in g2.axes.flat:
        ax.set_ylim(0, None)
        ax.set_ylabel("Number of Iterations")
        ax.tick_params(labelbottom=True)

    g2.fig.subplots_adjust(top=0.9)
    g2.fig.suptitle("Iterations to Convergence per Population Size", fontsize=16)
    g2.savefig(os.path.join(BASE_DIR, "fixed_iteration_boxplots.png"), dpi=300)

    # --- PLOT 3: COST VS ITERATIONS (Scatter) ---
    # Shows the trade-off per population size
    print("Plotting Cost vs Iteration Scatter...")
    g3 = sns.relplot(
        data=df, x="Iterations", y="Final_Cost", hue="Algorithm",
        col="Population Size", col_wrap=3, height=4, aspect=1.2,
        kind="scatter", s=70, alpha=0.6, palette="bright"
    )
    for ax in g3.axes.flat:
        ax.set_ylim(0, None)
        ax.set_xlim(0, None)
        ax.grid(True, linestyle='--', alpha=0.5)

    g3.fig.subplots_adjust(top=0.9)
    g3.fig.suptitle("Efficiency: Cost vs Iterations per Population Size", fontsize=16)
    g3.savefig(os.path.join(BASE_DIR, "fixed_cost_vs_iteration.png"), dpi=300)


if __name__ == "__main__":
    results_df = load_and_clean_data(BASE_DIR)
    generate_report_plots(results_df)
    print(f"\nSUCCESS: All plots saved in {BASE_DIR}")