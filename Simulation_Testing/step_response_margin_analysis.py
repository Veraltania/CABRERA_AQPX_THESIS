import os
import csv
import math
import numpy as np
import control as ct
import matplotlib.pyplot as plt

from scipy_de_tuner import run_scipy_de_tuner

def create_fopdt_sys(K, tau, delay, pade_order=2):
    """Creates a Transfer Function for FOPDT using Pade approximation for delay."""
    num, den = [K], [tau, 1]
    plant_linear = ct.tf(num, den)
    
    if delay > 0:
        num_delay, den_delay = ct.pade(delay, pade_order)
        delay_tf = ct.tf(num_delay, den_delay)
        return ct.series(delay_tf, plant_linear)
    return plant_linear

def create_pi_controller(kp, ki):
    """Creates a PI controller Transfer Function: C(s) = Kp + Ki/s"""
    return ct.tf([kp, ki], [1, 0])

def safe_float(val):
    if not val or str(val).strip() == '':
        return 0.0
    cleaned_val = str(val).replace('−', '-').replace('–', '-').replace(' ', '').strip()
    try:
        return float(cleaned_val)
    except ValueError:
        return 0.0

def read_tf_parameters(filepath):
    """Reads FOPDT parameters from the specified CSV."""
    tfs = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader) # Skip main header
        next(reader) # Skip sub-header
        for row in reader:
            if not row or not row[0].strip():
                continue
            row = row + [''] * (10 - len(row))
            
            tf_data = {
                'name': row[0].strip(),
                'K': safe_float(row[1]),
                'tau': safe_float(row[2]),
                'delay': safe_float(row[3])
            }
            tfs.append(tf_data)
    return tfs

def write_margin_csv(output_path, tf_names, ce_pcts, data_matrix, is_gm=False):
    """Writes the metrics to a CSV matching the required phase_gain_analysis matrix layout."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header row: Empty first cell, followed by TF names
        writer.writerow([''] + tf_names)
        
        for ce in ce_pcts:
            row_label = f"CE Percentage in Cost Function ({int(ce)}%)"
            row_data = [row_label]
            
            for tf_name in tf_names:
                val = data_matrix[ce].get(tf_name, '')
                if isinstance(val, (float, np.floating)):
                    if math.isinf(val) or np.isnan(val):
                        row_data.append("Inf")
                    else:
                        if is_gm:
                            # Standard practice: convert absolute gain margin to dB
                            val_db = 20 * np.log10(val)
                            row_data.append(f"{val_db:.4f}")
                        else:
                            # Phase margin remains in degrees
                            row_data.append(f"{val:.4f}")
                else:
                    row_data.append(str(val))
                    
            writer.writerow(row_data)

def generate_ieee_plots(output_dir, ce_pcts, pm_matrix):
    """Generates IEEE-compliant Line and Box plots for the Phase Margin data."""
    # IEEE plotting configurations
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'lines.linewidth': 1.0,
        'lines.markersize': 4
    })
    
    # Standard IEEE single column width: 3.5 inches
    fig_width = 3.5 
    fig_height = 2.5
    
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
        print("No valid phase margin data to plot.")
        return

    # --- 1. Line Plot of Average Phase Margins ---
    fig_line, ax_line = plt.subplots(figsize=(fig_width, fig_height))
    plt.ylim(0, 100)
    ax_line.plot(valid_ce_pcts, avg_pms, 'k-o', label='Average PM') 
    
    ax_line.set_xlabel('Control Effort Weight (%)')
    ax_line.set_ylabel('Average Phase Margin (deg)')
    ax_line.set_xticks(valid_ce_pcts)
    ax_line.grid(True, linestyle=':', alpha=0.7)
    

    fig_line.tight_layout()
    line_path = os.path.join(output_dir, "average_phase_margin.pdf")
    fig_line.savefig(line_path, format='pdf', bbox_inches='tight')
    plt.close(fig_line)

    # --- 2. Box Plot of Phase Margins ---
    fig_box, ax_box = plt.subplots(figsize=(fig_width, fig_height))
    
    # Strict black and white styling for IEEE conformity
    ax_box.boxplot(box_data, positions=valid_ce_pcts, widths=6, patch_artist=True,
                   boxprops=dict(facecolor='white', color='black'),
                   capprops=dict(color='black'),
                   whiskerprops=dict(color='black', linestyle='--'),
                   flierprops=dict(marker='x', color='black', markersize=4),
                   medianprops=dict(color='black', linewidth=1.2))
    
    ax_box.set_xlabel('Control Effort Weight (%)')
    ax_box.set_ylabel('Phase Margin (deg)')
    plt.ylim(0, 100)
    
    # Adjust axes spacing dynamically to fit the width of box plots
    ax_box.set_xlim(min(valid_ce_pcts) - 10, max(valid_ce_pcts) + 10)
    ax_box.set_xticks(valid_ce_pcts)
    ax_box.grid(True, linestyle=':', alpha=0.7)
    
    fig_box.tight_layout()
    box_path = os.path.join(output_dir, "boxplot_phase_margin.pdf")
    fig_box.savefig(box_path, format='pdf', bbox_inches='tight')
    plt.close(fig_box)

def main():
    # ==========================================
    # CONFIGURATION
    # ==========================================
    INPUT_CSV_NAME = "tf_parameters_do.csv"
    FOLDER_NAME = "do_extended_margin_analysis_reports"
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(base_dir, INPUT_CSV_NAME)
    output_dir = os.path.join(base_dir, FOLDER_NAME)
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_csv):
        print(f"Error: Could not find '{input_csv}'. Ensure the file is in {base_dir}")
        return

    tf_list = read_tf_parameters(input_csv)
    tf_names = [tf['name'] for tf in tf_list]
    
    ce_pcts_to_test = [0.0, 12.5, 25.0, 37.50, 50.0, 62.5, 75.0, 87.5, 100.0]
    
    pm_matrix = {ce: {} for ce in ce_pcts_to_test}
    gm_matrix = {ce: {} for ce in ce_pcts_to_test}

    for tf_data in tf_list:
        tf_name = tf_data['name']
        print(f"\n--- Processing Transfer Function: {tf_name} ---")
        
        plant_params = {'K': tf_data['K'], 'tau': tf_data['tau'], 'delay': tf_data['delay']} 
        plant = create_fopdt_sys(**plant_params)
        
        target_sp = 1.0 if plant_params['K'] > 0 else -1.0
        
        if plant_params['K'] < 0:
            max_kp, min_kp = 0, -1
            max_ki, min_ki = 0, -0.0005 
        else:
            max_kp, min_kp = 1.5, 0
            max_ki, min_ki = 0.01, 0

        for ce_pct in ce_pcts_to_test:
            print(f"  Tuning & Analyzing CE = {ce_pct}%...")
            
            ce_val = (ce_pct / 100.0) * 4.0 
            perf = (4.0 - ce_val) / 3.0
            de_weights = [perf, ce_val, perf, perf]
            
            de_kp, de_ki = run_scipy_de_tuner(
                f"DE (ce={ce_pct}%)", plant,
                min_kp, max_kp, min_ki, max_ki,
                plant_params['tau'], plant_params['delay'], de_weights,
                target_sp=target_sp
            )
            
            controller = create_pi_controller(de_kp, de_ki)
            open_loop_sys = ct.series(controller, plant)
            
            gm, pm, wg, wp = ct.margin(open_loop_sys)
            
            pm_matrix[ce_pct][tf_name] = pm
            gm_matrix[ce_pct][tf_name] = gm

    pm_csv_path = os.path.join(output_dir, "phase_margin_analysis_deg.csv")
    gm_csv_path = os.path.join(output_dir, "gain_margin_analysis_db.csv")
    
    write_margin_csv(pm_csv_path, tf_names, ce_pcts_to_test, pm_matrix, is_gm=False)
    write_margin_csv(gm_csv_path, tf_names, ce_pcts_to_test, gm_matrix, is_gm=True)
    
    # Generate the requested PDF graphs
    generate_ieee_plots(output_dir, ce_pcts_to_test, pm_matrix)

    print(f"\nAnalysis complete. Cleanly formatted reports and graphs saved to:\n{output_dir}")

if __name__ == "__main__":
    main()