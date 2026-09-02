import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- CUSTOMIZATION ---
# Define specific orders for the legends
DO_ORDER = [
    "Feb 5, Daytime",
    "Feb 7, Daytime",
    "Feb 25, Daytime",
    "Feb 26, Daytime",
    "Feb 5, Nighttime",
    "Feb 7, Nighttime",
    "Feb 25, Nighttime",
    "Feb 26, Nighttime"
]

TDS_ORDER = [
    "February 9 to 10",
    "February 10 to 11",
    "February 11 to 12"
]

# Define custom color palettes 
# DO: Daytime = Reds/Oranges, Nighttime = Blues/Greens
DO_PALETTE = {
    "Feb 5, Daytime": "#990000",    
    "Feb 7, Daytime": "#FF3333",    
    "Feb 25, Daytime": "#FF8000",   
    "Feb 26, Daytime": "#FFB266",   
    "Feb 5, Nighttime": "#000099",  
    "Feb 7, Nighttime": "#3366FF",  
    "Feb 25, Nighttime": "#6342f5", 
    "Feb 26, Nighttime": "#21a2d1"  
}

# TDS: Standard distinct colors
TDS_PALETTE = {
    "February 9 to 10": "#9467bd",  # Purple
    "February 10 to 11": "#8c564b", # Brown
    "February 11 to 12": "#e377c2"  # Pink
}


# --- HELPER FUNCTION ---
def format_legend_label(tf_name):
    """Formats the transfer function name for the legend."""
    parts = tf_name.lower().split('_')
    
    if 'do' in parts:
        try:
            idx = parts.index('do')
            date_part = parts[idx+1]  # e.g., 'feb5' or 'feb25'
            time_part = parts[idx+2]  # e.g., 'daytime' or 'nighttime'
            
            month = date_part[:3].capitalize()
            day = date_part[3:]
            time_formatted = time_part.capitalize()
            
            return f"{month} {day}, {time_formatted}"
        except IndexError:
            pass
            
    elif 'tds' in parts:
        try:
            idx = parts.index('tds')
            date_part = parts[idx+1]  # e.g., 'feb09'
            end_day_part = parts[idx+2]  # e.g., '10'
            
            month_str = date_part[:3]
            month_map = {
                'jan': 'January', 'feb': 'February', 'mar': 'March', 'apr': 'April',
                'may': 'May', 'jun': 'June', 'jul': 'July', 'aug': 'August',
                'sep': 'September', 'oct': 'October', 'nov': 'November', 'dec': 'December'
            }
            full_month = month_map.get(month_str, month_str.capitalize())
            
            start_day = str(int(date_part[3:])) # Cast to int to remove leading zeros
            end_day = str(int(end_day_part))
            
            return f"{full_month} {start_day} to {end_day}"
        except (IndexError, ValueError):
            pass
            
    return tf_name


# --- PLOTTING FUNCTION ---
def generate_scatterplot(df, x_col, y_col, category, is_average, output_dir):
    """Generates a scatterplot for either Average or Raw Kp vs Ki."""
    if df.empty:
        print(f"No valid data available to plot for {category} ({'Average' if is_average else 'Raw'}).")
        return
        
    plot_type = "Average" if is_average else ""
    print(f"Generating DE {plot_type} Gain Scatterplot for {category} in {output_dir}...")

    # Select the correct order and palette based on the category
    hue_order = DO_ORDER if category == "DO" else TDS_ORDER
    palette = DO_PALETTE if category == "DO" else TDS_PALETTE

    # Filter out any legend labels in our hue_order that don't actually exist in the current DataFrame
    active_labels = df["Transfer Function"].unique()
    valid_hue_order = [label for label in hue_order if label in active_labels]

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 16
    })

    # Set up the plot area with EXACT 7.16 inch width (height is proportionally estimated to 4.5 inches)
    fig, ax = plt.subplots(figsize=(7.16, 4.5))

    # Scaled down marker sizes for smaller canvas
    marker_size = 120 if is_average else 40
    alpha_val = 1.0 if is_average else 0.6

    # Create the scatter plot
    scatter = sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue="Transfer Function",
        hue_order=valid_hue_order, # Applies the custom sorting
        palette=palette,           # Applies the custom colors
        marker="o",
        s=marker_size,
        alpha=alpha_val,
        ax=ax
    )

    # Label the axes (inherited font size 14)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"DE-tuned Gain Distribution", pad=15)

    # Ensure axes start at 0 (Accounting for negative reverse-acting gains in TDS)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    if x_min >= 0:
        ax.set_xlim(left=0)
    elif x_max <= 0:
        ax.set_xlim(right=0)
        
    if y_min >= 0:
        ax.set_ylim(bottom=0)
    elif y_max <= 0:
        ax.set_ylim(top=0)

    # Add a grid
    ax.grid(True, linestyle='--', alpha=0.5)

    # Put the legend outside the graph on the right side
    legend = ax.legend(
        title="Transfer Function", 
        loc='center left', 
        bbox_to_anchor=(1.02, 0.5), # Pushes it outside the right edge
        frameon=True
    )
    legend.get_title().set_fontweight('bold')

    # Dynamic filename with .pdf extension
    suffix = "average" if is_average else "raw"
    filename = f"de_{suffix}_kp_ki_pop50_{category.lower()}_scatter.pdf"
    plot_path = os.path.join(output_dir, filename)
    
    # Save as PDF vector graphic, tight bbox ensures the external legend isn't cut off
    plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f" -> Saved: {plot_path}")


# --- EXECUTION ---
if __name__ == "__main__":
    # Define directories
    script_dir = Path(__file__).parent.resolve()
    batch_dir = script_dir / "BATCH_OPENLOOP_CONTROL_EFFORT_V2"

    if not batch_dir.exists():
        print(f"\n[SKIP] Directory not found: {batch_dir}")
        exit(1)

    # Discover transfer function directories based on population_sweep_* pattern
    search_pattern = os.path.join(batch_dir, "population_sweep_*")
    tf_dirs = [Path(d) for d in glob.glob(search_pattern) if os.path.isdir(d)]

    if not tf_dirs:
        print(f"\n[SKIP] No transfer function directories found in {batch_dir}")
        exit(1)

    # Lists to hold data
    all_tf_averages = []
    all_raw_gains = []

    for tf_dir in tf_dirs:
        print(f"\nProcessing data from: {tf_dir.name}")
        
        # Determine category based on folder name
        if "_do_" in tf_dir.name.lower():
            category = "DO"
        elif "_tds_" in tf_dir.name.lower():
            category = "TDS"
        else:
            print(f" -> [SKIP] Could not determine category (DO/TDS) from {tf_dir.name}")
            continue

        # Find the latest checkpoint CSV directly in the transfer function directory
        csv_pattern = os.path.join(tf_dir, "checkpoint_sweep_*.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            print(f" -> [SKIP] No checkpoint CSV found in {tf_dir}")
            continue
            
        # Select the most recent checkpoint CSV for data extraction
        latest_csv = max(csv_files, key=os.path.getmtime)
        
        # Apply the legend formatting rule
        formatted_tf_name = format_legend_label(tf_dir.name)
        
        try:
            # Load the whole experiment checkpoint data
            df = pd.read_csv(latest_csv)
            
            # Filter for DE and Population Size 50, and use .copy()
            filtered_df = df[(df['Algorithm'] == 'DE') & (df['Population Size'] == 50)].copy()
            
            if filtered_df.empty:
                print(f" -> [SKIP] No DE data for Pop 50 found in {tf_dir.name}")
                continue
            
            # Clean strings (e.g. remove list brackets) and convert to numeric 
            filtered_df['Kp'] = pd.to_numeric(filtered_df['Kp'].astype(str).str.replace(r'[\[\]]', '', regex=True), errors='coerce')
            filtered_df['Ki'] = pd.to_numeric(filtered_df['Ki'].astype(str).str.replace(r'[\[\]]', '', regex=True), errors='coerce')
            
            # Drop any rows where Kp or Ki failed to convert to numbers
            filtered_df = filtered_df.dropna(subset=['Kp', 'Ki'])

            if filtered_df.empty:
                print(f" -> [SKIP] No valid numeric Kp/Ki data after cleaning for {tf_dir.name}")
                continue

            # --- 1. COLLECT RAW DATA ---
            for _, row in filtered_df.iterrows():
                all_raw_gains.append({
                    'Category': category,
                    'Transfer Function': formatted_tf_name,
                    'Raw Kp': row['Kp'],
                    'Raw Ki': row['Ki']
                })

            # --- 2. COLLECT AVERAGE DATA ---
            avg_kp = filtered_df['Kp'].mean()
            avg_ki = filtered_df['Ki'].mean()
            
            all_tf_averages.append({
                'Category': category,
                'Transfer Function': formatted_tf_name,
                'Average Kp': avg_kp,
                'Average Ki': avg_ki
            })

        except Exception as e:
            print(f" -> [ERROR] Failed to process {latest_csv}: {e}")

    # Compile all collected data into DataFrames
    avg_df = pd.DataFrame(all_tf_averages)
    raw_df = pd.DataFrame(all_raw_gains)

    # Plot graphs for each category
    for cat in ["DO", "TDS"]:
        print(f"\n--- Generating Plots for {cat} ---")
        
        # Filter DataFrames for the current category
        cat_avg_df = avg_df[avg_df['Category'] == cat] if not avg_df.empty else pd.DataFrame()
        cat_raw_df = raw_df[raw_df['Category'] == cat] if not raw_df.empty else pd.DataFrame()

        # Generate Average Plot
        generate_scatterplot(
            df=cat_avg_df, 
            x_col="Average Kp", 
            y_col="Average Ki", 
            category=cat, 
            is_average=True, 
            output_dir=batch_dir
        )

        # Generate Raw Points Plot
        generate_scatterplot(
            df=cat_raw_df, 
            x_col="Raw Kp", 
            y_col="Raw Ki", 
            category=cat, 
            is_average=False, 
            output_dir=batch_dir
        )