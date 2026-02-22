import os
import glob
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# --- 1. DATA LOADING ---
def load_data(base_dir="results_patience_sweep"):
    print(f"--- Scanning '{base_dir}' for experiment data ---")
    data = []
    target_algos = ['DE', 'GA', 'PSO']

    for algo in target_algos:
        algo_path = os.path.join(base_dir, algo.lower())
        # Try alternate capitalizations if default fails
        if not os.path.exists(algo_path): algo_path = os.path.join(base_dir, algo)
        if not os.path.exists(algo_path): algo_path = os.path.join(base_dir, algo.capitalize())
        if not os.path.exists(algo_path): continue

        # Changed to match 'patience_' folders
        patience_folders = glob.glob(os.path.join(algo_path, "patience_*"))
        for folder in patience_folders:
            match = re.search(r'patience_(\d+)', folder)
            if not match: continue
            patience = int(match.group(1))

            csv_files = glob.glob(os.path.join(folder, "*.csv"))
            if not csv_files: continue
            latest_file = max(csv_files, key=os.path.getmtime)

            try:
                df = pd.read_csv(latest_file)
                cols = df.columns
                fitness_vals = []
                if 'Final_Cost_ITAE' in cols:
                    fitness_vals = df['Final_Cost_ITAE']
                elif 'Cost' in cols:
                    fitness_vals = df['Cost']
                elif len(cols) >= 3:
                    fitness_vals = df.iloc[:, 2]

                # Ensure numeric and drop NaNs right away
                fitness_vals = pd.to_numeric(fitness_vals, errors='coerce').dropna()

                for val in fitness_vals:
                    data.append({
                        "Algorithm": algo,
                        "Patience Limit": patience,  # Changed from Population Size
                        "Fitness": val
                    })
            except Exception as e:
                print(f"[ERR] {latest_file}: {e}")

    return pd.DataFrame(data)


# --- 2. OUTLIER REMOVAL (IQR Method) ---
def remove_outliers_iqr(df):
    """
    Removes outliers using the 1.5 * IQR rule, applied independently
    to each Algorithm within each Patience Limit group.
    """
    print(f"Data points before outlier removal: {len(df)}")

    cleaned_rows = []

    # Group by Patience Limit and Algorithm to calculate IQR specifically for that set
    grouped = df.groupby(['Patience Limit', 'Algorithm'])

    for (pat, algo), group_df in grouped:
        q1 = group_df['Fitness'].quantile(0.25)
        q3 = group_df['Fitness'].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Keep only rows within bounds
        mask = (group_df['Fitness'] >= lower_bound) & (group_df['Fitness'] <= upper_bound)
        cleaned_rows.append(group_df[mask])

    if cleaned_rows:
        df_clean = pd.concat(cleaned_rows, ignore_index=True)
    else:
        df_clean = pd.DataFrame(columns=df.columns)

    print(f"Data points after outlier removal: {len(df_clean)} (Removed: {len(df) - len(df_clean)})")
    return df_clean


# --- 3. THOROUGH PLOTTING ---
def generate_thorough_plots(df):
    if df.empty:
        print("No data found.")
        return

    # --- STEP 1: CLEAN OUTLIERS ---
    df_clean = remove_outliers_iqr(df)

    if df_clean.empty:
        print("Error: All data was removed as outliers. Cannot plot.")
        return

    # Sort cleaned data by Patience Limit instead of Population Size
    df_clean.sort_values(by=["Patience Limit", "Algorithm"], inplace=True)

    # --- STEP 2: CALCULATE TIGHT LIMITS (ON CLEAN DATA) ---
    global_min = df_clean["Fitness"].min()
    global_max = df_clean["Fitness"].max()

    # Add a tiny buffer (e.g., 2%) for visual breathing room around the dots
    y_limit_top = global_max * 1.02
    # Avoid division by zero if min is 0 (though unlikely for ITAE)
    y_limit_bottom = global_min * 0.98 if global_min > 0 else global_min

    print(f"Capping Y-Axis at: {y_limit_top:.2e} (Max Clean Data: {global_max:.2e})")

    sns.set_style("whitegrid")

    # Create Grid using CLEAN data
    g = sns.FacetGrid(
        df_clean,
        col="Patience Limit",  # Update grid layout to partition by Patience Limit
        col_wrap=3,
        sharex=False,
        sharey=False,  # False so we can force our custom limits
        height=4,
        aspect=1.2
    )

    # 1. Box Plot (fliersize=0 is redundant now, but good practice)
    g.map_dataframe(sns.boxplot,
                    x="Algorithm",
                    y="Fitness",
                    hue="Algorithm",
                    palette="viridis",
                    dodge=False,
                    width=0.5,
                    fliersize=0)

    # 2. Strip Plot (Cleaned data points only)
    g.map_dataframe(sns.stripplot,
                    x="Algorithm",
                    y="Fitness",
                    color=".2",
                    size=3,
                    alpha=0.6,
                    jitter=True)

    # --- 3. APPLY TIGHT AXIS LIMITS & LABELS ---
    for ax in g.axes.flat:
        ax.set_yscale("log")
        # Apply limits based on cleaned data range
        ax.set_ylim(y_limit_bottom, y_limit_top)

        # Force X-Axis labels on every plot
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("")

        # Titles and Labels
    g.set_titles(col_template="Patience: {col_name}", fontweight='bold')
    g.set_axis_labels("", "")
    g.fig.supylabel("Cost (ITAE) - Log Scale (Outliers Removed)", fontweight='bold')
    g.fig.suptitle("Algorithm Performance Distribution (Patience Sweep)", y=1.02, fontsize=14)

    plt.tight_layout()

    filename = "box_plots_patience_sweep.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Plot saved to: {filename}")
    plt.show()


if __name__ == "__main__":
    df_results = load_data()
    generate_thorough_plots(df_results)