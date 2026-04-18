import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import control as ct

from scipy_de_tuner import run_scipy_de_tuner

# ==========================================
# 0. CONFIGURATION
# ==========================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 14
})

# ==========================================
# 1. PLANT CREATION
# ==========================================

def create_fopdt_sys(K, tau, delay, pade_order=4):
    """Creates a Transfer Function for FOPDT using Pade approximation for delay."""
    num, den = [K], [tau, 1]
    plant_linear = ct.tf(num, den)
    
    if delay > 0:
        num_delay, den_delay = ct.pade(delay, pade_order)
        delay_tf = ct.tf(num_delay, den_delay)
        return ct.series(delay_tf, plant_linear)
    return plant_linear

# ==========================================
# 2. METRIC & FILE HELPERS
# ==========================================

def safe_float(val):
    if not val or str(val).strip() == '':
        return 0.0
    cleaned_val = str(val).replace('−', '-').replace('–', '-').replace(' ', '').strip()
    try:
        return float(cleaned_val)
    except ValueError:
        return 0.0

def read_tf_parameters(filepath):
    tfs = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader) 
        next(reader) 
        for row in reader:
            if not row or not row[0].strip():
                continue
            row = row + [''] * (10 - len(row))
            
            tf_data = {
                'name': row[0].strip(),
                'K': safe_float(row[1]),
                'tau': safe_float(row[2]),
                'delay': safe_float(row[3]),
                'matlab_kp': safe_float(row[6]),
                'matlab_ki': safe_float(row[7]),
                'lambda_kp': safe_float(row[8]),
                'lambda_ki': safe_float(row[9])
            }
            tfs.append(tf_data)
    return tfs

def write_formatted_table(output_path, metric_name, tfs, metrics_dict):
    headers = ['Gain'] + [tf['name'] for tf in tfs]
    tuners = ['DE-tuned', 'Lambda-tuned ($\\lambda$ = 3$\\tau$)', 'MATLAB pidtune(), balanced']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for tuner in tuners:
            row = [tuner]
            for tf in tfs:
                val = metrics_dict[tf['name']].get(tuner, 0.0)
                row.append(f"{val:.4f}")
            writer.writerow(row)

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(base_dir, "tf_parameters_do.csv")
    output_dir = os.path.join(base_dir, "simulation_graphs_disturbance_do")
    os.makedirs(output_dir, exist_ok=True)
    
    aggregated_metrics = {'IAE': {}, 'Control_Effort_AUC': {}}
    tf_list = read_tf_parameters(input_csv)
    
    cf_weight = 1.0
    perf_weight = (4.0 - cf_weight) / 3
    de_weights = [perf_weight, cf_weight, perf_weight, perf_weight]

    s = ct.tf('s')

    for tf_data in tf_list:
        tf_name = tf_data['name']
        print(f"\n--- Processing Transfer Function: {tf_name} ---")
        
        for key in aggregated_metrics:
            aggregated_metrics[key][tf_name] = {}
        
        plant_params = {'K': tf_data['K'], 'tau': tf_data['tau'], 'delay': tf_data['delay']} 
        plant = create_fopdt_sys(**plant_params)
        
        t_final = max(plant_params['tau'], plant_params['delay']) * 12 
        num_points = 5000 
        t_eval = np.linspace(0, t_final, num_points)

        if plant_params['K'] < 0:
            max_kp, min_kp = 0, -1
            max_ki, min_ki = 0, -0.0005 
        else:
            max_kp, min_kp = 1.5, 0
            max_ki, min_ki = 0.01, 0

        target_sp = 1.0 if plant_params['K'] > 0 else -1.0
        de_kp, de_ki = run_scipy_de_tuner(
            f"DE ({tf_name})", plant,
            min_kp, max_kp, min_ki, max_ki,
            plant_params['tau'], plant_params['delay'], de_weights,
            target_sp=target_sp
        )

        configs = [
            {"name": "DE-tuned", "color": "red", "kp": de_kp, "ki": de_ki},
            {"name": "Lambda-tuned ($\\lambda$ = 3$\\tau$)", "color": "green", "kp": tf_data['lambda_kp'], "ki": tf_data['lambda_ki']},
            {"name": "MATLAB pidtune(), balanced", "color": "blue", "kp": tf_data['matlab_kp'], "ki": tf_data['matlab_ki']}
        ]

        # Create two separate figures 
        # Width set to 7.16 inches. Height set to 4.3 to maintain approximate 10:6 aspect ratio
        fig_y, ax_y = plt.subplots(figsize=(7.16, 4.3))
        fig_u, ax_u = plt.subplots(figsize=(7.16, 4.3))
        
        fig_y.canvas.manager.set_window_title(f'Process Output - {tf_name}')
        fig_u.canvas.manager.set_window_title(f'Control Effort - {tf_name}')

        for cfg in configs:
            Gc = cfg["kp"] + cfg["ki"]/s
            
            T_yd = ct.feedback(plant, Gc)
            T_ud = -Gc * T_yd
            
            _, y_dist = ct.step_response(T_yd, T=t_eval)
            _, u_dist = ct.step_response(T_ud, T=t_eval)
            
            iae = np.trapezoid(np.abs(y_dist), t_eval)
            auc_u = np.trapezoid(np.abs(u_dist), t_eval)

            tuner_key = cfg["name"]
            aggregated_metrics['IAE'][tf_name][tuner_key] = iae
            aggregated_metrics['Control_Effort_AUC'][tf_name][tuner_key] = auc_u

            # Plot on respective axes
            ax_y.plot(t_eval, y_dist, color=cfg["color"], linewidth=1.5, label=f'{cfg["name"]}')
            ax_u.plot(t_eval, u_dist, color=cfg["color"], linewidth=1.5, label=f'{cfg["name"]}')

        # --- Figure 1: Load Disturbance Rejection ---
        ax_y.set_title('Load Disturbance Rejection Response')
        ax_y.set_xlabel('Time (s)')
        ax_y.set_ylabel('Process Output (y)')
        ax_y.legend(
            loc='best', 
            fontsize=12, 
            labelspacing=0.3,   # Reduces vertical space between items
            handlelength=1.5,   # Makes the colored line segments shorter
            handletextpad=0.4,  # Reduces space between the line and the text
            borderpad=0.3       # Shrinks the padding around the edges of the box
        )
        ax_y.grid(True, alpha=0.5)
        
        fig_y.tight_layout()
        # Changed to .pdf and explicitly specified the format
        fig_y.savefig(os.path.join(output_dir, f"load_disturbance_y_{tf_name}.pdf"), format='pdf', bbox_inches='tight')
        plt.close(fig_y)

        # --- Figure 2: Control Effort ---
        ax_u.set_title('Control Effort Response')
        ax_u.set_xlabel('Time (s)')
        ax_u.set_ylabel('Control Effort (u)')
        ax_u.legend(
            loc='best', 
            fontsize=8, 
            labelspacing=0.3,   # Reduces vertical space between items
            handlelength=1.5,   # Makes the colored line segments shorter
            handletextpad=0.4,  # Reduces space between the line and the text
            borderpad=0.3       # Shrinks the padding around the edges of the box
        )
        ax_u.grid(True, alpha=0.5)

        fig_u.tight_layout()
        # Changed to .pdf and explicitly specified the format
        fig_u.savefig(os.path.join(output_dir, f"load_disturbance_u_{tf_name}.pdf"), format='pdf', bbox_inches='tight')
        plt.close(fig_u)

    print("\n--- Exporting Formatted Metric Tables ---")
    write_formatted_table(os.path.join(output_dir, "IAE_table.csv"), 'IAE', tf_list, aggregated_metrics['IAE'])
    write_formatted_table(os.path.join(output_dir, "Control_Effort_AUC_table.csv"), 'Control_Effort_AUC', tf_list, aggregated_metrics['Control_Effort_AUC'])
    
    print(f"Simulation Complete. Processed {len(tf_list)} transfer functions.")

if __name__ == "__main__":
    main()