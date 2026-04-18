import os
import csv
import math
import re
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
CSV_FILE_PATH = "Simulation_Testing/margin_analysis_reports_tds_extended/phase_margin_analysis_deg.csv"
OUTPUT_DIR = "Simulation_Testing/margin_analysis_reports_tds_extended"

LINE_PLOT_TITLE = "Phase margin vs. control effort priority for TDS controllers"
BOX_PLOT_TITLE = LINE_PLOT_TITLE
# ==========================================

def read_margin_csv(filepath):
    """Reads the generated CSV and reconstructs the pm_matrix."""
    ce_pcts = []
    pm_matrix = {}
    
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)  # ['', 'TF1', 'TF2', ...]
        except StopIteration:
            return [], {}
            
        tf_names = headers[1:]
        
        for row in reader:
            if not row or not row[0].strip():
                continue
            
            # Extract the CE percentage from the row label (e.g., "CE Percentage in Cost Function (12%)")
            match = re.search(r'\((\d+(?:\.\d+)?)%\)', row[0])
            if not match:
                continue
                
            ce = float(match.group(1))
            ce_pcts.append(ce)
            pm_matrix[ce] = {}
            
            for idx, val_str in enumerate(row[1:]):
                if idx >= len(tf_names):
                    break
                tf_name = tf_names[idx]
                
                # Handle Inf, NaN, or empty cells
                if val_str.strip().lower() in ['inf', 'nan', '']:
                    val = float('inf')
                else:
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = float('inf')
                        
                pm_matrix[ce][tf_name] = val
                
    return ce_pcts, pm_matrix

def generate_ieee_plots(output_dir, ce_pcts, pm_matrix, line_title, box_title):
    """Generates IEEE-compliant Line and Box plots using the reconstructed data."""
    # IEEE plotting configurations
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 10
    
    # Standard IEEE single column width: 3.5 inches
    fig_width = 5 
    fig_height = 3.5
    
    avg_pms = []
    box_data = []
    valid_ce_pcts = []
    
    # Aggregate and filter data
    for ce in ce_pcts:
        # Filter out Inf/NaN responses to prevent graphing errors
        vals = [v for v in pm_matrix[ce].values() if not (math.isinf(v) or np.isnan(v))]
        if vals:
            avg_pms.append(np.mean(vals))
            box_data.append(vals)
            valid_ce_pcts.append(ce)
    
    if not valid_ce_pcts:
        print("No valid data to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Line Plot of Average Phase Margins ---
    fig_line, ax_line = plt.subplots(figsize=(fig_width, fig_height))
    plt.ylim(0, 100)
    ax_line.plot(valid_ce_pcts, avg_pms, 'k-o', label='Average PM') 
    
    ax_line.set_title(line_title)
    ax_line.set_xlabel('Control Effort Weight (%)')
    ax_line.set_ylabel('Average Phase Margin (deg)')
    ax_line.set_xticks(valid_ce_pcts)
    
    # Rotate labels to prevent overlapping
    ax_line.set_xticklabels([f"{ce:g}" for ce in valid_ce_pcts], rotation=0, ha='right')
    ax_line.grid(True, linestyle=':', alpha=0.7)
    
    fig_line.tight_layout()
    line_path = os.path.join(output_dir, "average_phase_margin.pdf")
    fig_line.savefig(line_path, format='pdf', bbox_inches='tight')
    plt.close(fig_line)
    print(f"Saved: {line_path}")

    # --- 2. Box Plot of Phase Margins ---
    fig_box, ax_box = plt.subplots(figsize=(fig_width, fig_height))
    
    # Strict black and white styling for IEEE conformity
    ax_box.boxplot(box_data, positions=valid_ce_pcts, widths=6, patch_artist=True,
                   boxprops=dict(facecolor='white', color='black'),
                   capprops=dict(color='black'),
                   whiskerprops=dict(color='black', linestyle='--'),
                   flierprops=dict(marker='x', color='black', markersize=4),
                   medianprops=dict(color='black', linewidth=1.2))
    
    ax_box.set_title(box_title)
    ax_box.set_xlabel('Control Effort Weight (%)')
    ax_box.set_ylabel('Phase Margin (deg)')
    plt.ylim(0, 100)
    
    # Adjust axes spacing dynamically to fit the width of box plots
    ax_box.set_xlim(min(valid_ce_pcts) - 10, max(valid_ce_pcts) + 10)
    ax_box.set_xticks(valid_ce_pcts)
    
    # Rotate labels to prevent overlapping
    ax_box.set_xticklabels([f"{ce:g}" for ce in valid_ce_pcts], rotation=0, ha='right')
    ax_box.grid(True, linestyle=':', alpha=0.7)
    
    fig_box.tight_layout()
    box_path = os.path.join(output_dir, "boxplot_phase_margin.pdf")
    fig_box.savefig(box_path, format='pdf', bbox_inches='tight')
    plt.close(fig_box)
    print(f"Saved: {box_path}")

def main():
    if not os.path.exists(CSV_FILE_PATH):
        print(f"Error: Could not find '{CSV_FILE_PATH}'. Check the configuration variables.")
        return

    print(f"Reading data from {CSV_FILE_PATH}...")
    ce_pcts, pm_matrix = read_margin_csv(CSV_FILE_PATH)
    
    if not ce_pcts:
        print("Error: No data found or failed to parse the CSV.")
        return

    print("Generating plots...")
    generate_ieee_plots(OUTPUT_DIR, ce_pcts, pm_matrix, LINE_PLOT_TITLE, BOX_PLOT_TITLE)
    print("Done.")

if __name__ == "__main__":
    main()