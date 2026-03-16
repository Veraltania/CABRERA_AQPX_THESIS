import os
import glob
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# --- 1. DATA LOADING ---
def load_data(base_dir="results_population_sweep_do_tf3_daytime"):
    print(f"--- Scanning '{base_dir}' for experiment data ---")
    data_frames = []
    target_algos = ['DE', 'GA', 'PSO']

    for algo in target_algos:
        algo_path = os.path.join(base_dir, algo.lower())
        # Try alternate capitalizations if default fails
        if not os.path.exists(algo_path): algo_path = os.path.join(base_dir, algo)
        if not os.path.exists(algo_path): algo_path = os.path.join(base_dir, algo.capitalize())
        if not os.path.exists(algo_path): continue

        pop_folders = glob.glob(os.path.join(algo_path, "pop_*"))
        for folder in pop_folders:
            match = re.search(r'pop_(\d+)', folder)
            if not match: continue
            pop_size = int(match.group(1))

            csv_files = glob.glob(os.path.join(folder, "*.csv"))
            if not csv_files: continue
            latest_file = max(csv_files, key=os.path.getmtime)

            try:
                df = pd.read_csv(latest_file)
                cols = df.columns

                # Extract Cost
                if 'Final_Cost_ITAE' in cols:
                    cost_vals = df['Final_Cost_ITAE']
                elif 'Cost' in cols:
                    cost_vals = df['Cost']
                else:
                    cost_vals = df.iloc[:, 2]  # Fallback based on your log structure

                # Extract Iterations
                if 'Iterations_Run' in cols:
                    iter_vals = df['Iterations_Run']
                else:
                    iter_vals = df.iloc[:, 1]  # Fallback based on your log structure

                # Build a temporary DataFrame for this specific file
                temp_df = pd.DataFrame({
                    "Algorithm": algo,
                    "Population Size": pop_size,
                    "Cost": pd.to_numeric(cost_vals, errors='coerce'),
                    "Iterations": pd.to_numeric(iter_vals, errors='coerce')
                })

                # Drop rows where either metric is NaN
                temp_df.dropna(subset=['Cost', 'Iterations'], inplace=True)
                data_frames.append(temp_df)

            except Exception as e:
                print(f"[ERR] {latest_file}: {e}")

    if not data_frames:
        return pd.DataFrame()

    return pd.concat(data_frames, ignore_index=True)


# --- 2. OUTLIER REMOVAL (IQR Method) ---
def remove_outliers_iqr(df, value_col):
    """
    Removes outliers using the 1.5 * IQR rule, applied independently
    to each Algorithm within each Population Size group, for a specific metric.
    """
    print(f"Data points before outlier removal ({value_col}): {len(df)}")

    cleaned_rows = []
    grouped = df.groupby(['Population Size', 'Algorithm'])

    for (pop, algo), group_df in grouped:
        q1 = group_df[value_col].quantile(0.25)
        q3 = group_df[value_col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Keep only rows within bounds
        mask = (group_df[value_col] >= lower_bound) & (group_df[value_col] <= upper_bound)
        cleaned_rows.append(group_df[mask])

    df_clean = pd.concat(cleaned_rows, ignore_index=True)
    print(f"Data points after outlier removal: {len(df_clean)} (Removed: {len(df) - len(df_clean)})\n")
    return df_clean


# --- 3. THOROUGH PLOTTING ---
def generate_thorough_plots(df, metric_col, y_axis_label, output_dir):
    if df.empty:
        print(f"No data found for {metric_col}.")
        return

    # --- STEP 1: CLEAN OUTLIERS ---
    df_clean = remove_outliers_iqr(df, metric_col)

    if df_clean.empty:
        print(f"Error: All data was removed as outliers for {metric_col}. Cannot plot.")
        return

    # Sort cleaned data
    df_clean.sort_values(by=["Population Size", "Algorithm"], inplace=True)

    # --- STEP 2: CALCULATE TIGHT LIMITS (ON CLEAN DATA) ---
    global_min = df_clean[metric_col].min()
    global_max = df_clean[metric_col].max()

    y_limit_top = global_max * 1.02
    # Adjust bottom buffer if values are heavily negative vs positive
    if global_min > 0:
        y_limit_bottom = global_min * 0.98
    else:
        y_limit_bottom = global_min * 1.02

    sns.set_style("whitegrid")

    # --- 3A. INDIVIDUAL PLOTS ---
    for pop_size, group_data in df_clean.groupby("Population Size"):
        plt.figure(figsize=(6, 5))

        # 1. Box Plot
        sns.boxplot(data=group_data, x="Algorithm", y=metric_col, hue="Algorithm",
                    palette="viridis", dodge=False, width=0.5, fliersize=0)

        # 2. Strip Plot
        sns.stripplot(data=group_data, x="Algorithm", y=metric_col,
                      color=".2", size=4, alpha=0.6, jitter=True)

        plt.title(f"Algorithm Performance - Pop Size: {pop_size}", fontweight='bold')
        plt.xlabel("Evolutionary Algorithm", fontweight='bold')
        plt.ylabel(y_axis_label, fontweight='bold')
        plt.ylim(y_limit_bottom, y_limit_top)

        plt.tight_layout()
        ind_filename = os.path.join(output_dir, f"pop_{pop_size}_{metric_col.lower()}_boxplot.png")
        plt.savefig(ind_filename, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory

    # --- 3B. COMBINED GRID PLOT ---
    g = sns.FacetGrid(
        df_clean,
        col="Population Size",
        col_wrap=3,
        sharex=False,
        sharey=False,
        height=4,
        aspect=1.2
    )

    g.map_dataframe(sns.boxplot, x="Algorithm", y=metric_col, hue="Algorithm",
                    palette="viridis", dodge=False, width=0.5, fliersize=0)
    g.map_dataframe(sns.stripplot, x="Algorithm", y=metric_col,
                    color=".2", size=3, alpha=0.6, jitter=True)

    for ax in g.axes.flat:
        ax.set_ylim(y_limit_bottom, y_limit_top)
        ax.tick_params(labelbottom=True)
        # ADDED: X-Axis Label explicitly set for every subplot
        ax.set_xlabel("Evolutionary Algorithm", fontweight='bold')

    g.set_titles(col_template="Pop Size: {col_name}", fontweight='bold')
    g.set_axis_labels("Evolutionary Algorithm", "")  # Ensure FacetGrid knows the x-axis

    # Set main y-axis label for the entire grid
    g.fig.supylabel(y_axis_label, fontweight='bold')
    g.fig.suptitle(f"Algorithm {metric_col} Distribution", y=1.02, fontsize=14)

    plt.tight_layout()

    comb_filename = os.path.join(output_dir, f"combined_{metric_col.lower()}_boxplot.png")
    plt.savefig(comb_filename, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Saved combined and individual plots for '{metric_col}' to '{output_dir}/'")
    plt.close()


if __name__ == "__main__":
    base_dir = "results_population_sweep_tds_tf2"
    df_results = load_data(base_dir=base_dir)

    if df_results.empty:
        print("Failed to load any data.")
    else:
        # Create an output directory for the generated images
        output_directory = f"population_sweep_plots_{base_dir}"
        os.makedirs(output_directory, exist_ok=True)

        print(f"\n--- Generating Plots for Final Cost (ITAE) ---")
        generate_thorough_plots(
            df=df_results,
            metric_col="Cost",
            y_axis_label="Log10(ITAE Cost) (Outliers Removed)",
            output_dir=output_directory
        )

        print(f"\n--- Generating Plots for Iterations Run ---")
        generate_thorough_plots(
            df=df_results,
            metric_col="Iterations",
            y_axis_label="Number of Iterations (Outliers Removed)",
            output_dir=output_directory
        )