import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import control as ct

# Import your custom modules
from scipy_de_tuner import run_scipy_de_tuner, simulate_saturated_pi

# Apply global font settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

# ==========================================
# 1. PLANT DEFINITIONS
# ==========================================

def create_fopdt_sys(K, tau, delay, pade_order=2):
    """Creates a Transfer Function for FOPDT using Pade approximation for delay."""
    num, den = [K], [tau, 1]
    plant_linear = ct.tf(num, den)
    
    if delay > 0:
        num_delay, den_delay = ct.pade(delay, pade_order)
        delay_tf = ct.tf(num_delay, den_delay)
        return ct.series(delay_tf, plant_linear)
    return plant_linear

# ==========================================
# 2. DATA I/O MODULE
# ==========================================

def load_tf_parameters(filepath):
    """Reads K, tau, and theta from the specific TDS CSV format."""
    params = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip source header
        next(reader) # Skip parameter column header
        
        for row in reader:
            if not row or not row[0].startswith('TF_TDS'):
                continue
            params.append({
                'name': row[0],
                'K': float(row[1]),
                'tau': float(row[2]),
                'delay': float(row[3])
            })
    return params

def export_summary_csv(filepath, data_dict, scenario_names):
    """Exports the metric dictionary to a CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Tuning Strategy'] + scenario_names)
        for tuning, values in data_dict.items():
            formatted_vals = [f"{v:.2f}" for v in values]
            writer.writerow([tuning] + formatted_vals)

# ==========================================
# 3. METRICS ENGINE
# ==========================================

def calculate_metrics(t, y, u, sp):
    """Calculates Integral Absolute Error and Area Under Curve (Control Effort)."""
    error = sp - y
    iae = np.trapezoid(np.abs(error), t)
    u_auc = np.trapezoid(np.abs(u), t) 
    return iae, u_auc

# ==========================================
# 4. PLOTTING MODULE
# ==========================================

def plot_daily_comparison(output_dir, scenario_name, t, sp, results):
    """Generates and saves clean unit step response graphs."""
    # Convert time to hours for cleaner x-axis reading
    t_hours = t / 3600.0
    
    fig_width = 7.16
    fig_height = 4.09
    
    # 1. Step Response Plot
    plt.figure(figsize=(fig_width, fig_height)) 
    plt.plot(t_hours, sp, 'k--', label='Setpoint (0 to -1)', alpha=0.6)

    for res in results:
        plt.plot(t_hours, res["y"], color=res["color"], label=f'{res["name"]}')
        
    plt.title(f'Step Response Comparison ({scenario_name})', pad=10)
    plt.xlabel("Elapsed Time (Hours)")
    plt.ylabel('System Output')
    
    plt.legend(loc='best', fontsize=9, labelspacing=0.3, handlelength=1.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout() 
    plt.savefig(os.path.join(output_dir, f"unit_step_{scenario_name.replace(' ', '_')}.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

    # 2. Control Effort Plot
    plt.figure(figsize=(fig_width, fig_height))
    for res in results:
        plt.plot(t_hours, res["u"], color=res["color"], label=f'{res["name"]} Effort')
        
    plt.title(f'Control Effort Comparison ({scenario_name})', pad=10)
    plt.xlabel("Elapsed Time (Hours)")
    plt.ylabel('Control Signal (u)')
    plt.ylim(-0.05, 1.05) # Explicitly show the 0 to 1 bounds
    
    plt.legend(loc='best', fontsize=9, labelspacing=0.3, handlelength=1.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"unit_effort_{scenario_name.replace(' ', '_')}.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "simulation_graphs_tds_unit_step")
    os.makedirs(output_dir, exist_ok=True)
    
    tds_csv = os.path.join(base_dir, "tf_parameters_tds.csv")
    tds_params_list = load_tf_parameters(tds_csv)
    
    if len(tds_params_list) < 3:
        print("Error: Expected at least 3 days of TDS parameters in the CSV.")
        return
        
    scenario_names = ["Feb 9-10", "Feb 10-11", "Feb 11-12"]
    
    # ---------------------------------------------------------
    # Unit Step Configuration
    # ---------------------------------------------------------
    TARGET_SP = -1.0 
    
    # TDS time constants are large (~70,000s max). 
    # Simulate for ~80 hours (288,000s) to fully capture the step settling
    T_SIMULATION = 288000 
    N_STEPS = 3000
    
    t_eval = np.linspace(0, T_SIMULATION, N_STEPS)
    
    # Create a clean step from 0 to -1 starting a fraction into the simulation
    sp_eval = np.zeros_like(t_eval)
    step_start_idx = int(N_STEPS * 0.02)
    sp_eval[step_start_idx:] = TARGET_SP

    final_iae_data = {"Static (Day 1) Tuning": [], "Daily Retuned": []}
    final_auc_data = {"Static (Day 1) Tuning": [], "Daily Retuned": []}

    # 1. Auto-Tune the Static Baseline Controller on Day 1
    plant_day_1 = tds_params_list[0]
    plant_sys_1 = create_fopdt_sys(plant_day_1['K'], plant_day_1['tau'], plant_day_1['delay'])
    
    print(f"\n================ Establishing Static Baseline on {scenario_names[0]} ================")
    kp_static, ki_static = run_scipy_de_tuner(
        "Static Tuner", plant_sys_1, 
        min_kp=-1.0, max_kp=-0.001,   
        min_ki=-1e-3, max_ki=-1e-8,   
        tau=plant_day_1['tau'], delay=plant_day_1['delay'], 
        weights=(1.0, 1.0, 1.0, 1.0), target_sp=TARGET_SP
    )
    print(f"Static Tuning -> Kp: {kp_static:.4f}, Ki: {ki_static:.6f}")

    # 2. Iterate through each day, testing static vs retuned
    for idx in range(3):
        plant_params = tds_params_list[idx]
        plant_sys = create_fopdt_sys(plant_params['K'], plant_params['tau'], plant_params['delay'])
        name = scenario_names[idx]
        print(f"\n================ Processing {name} ================")
        
        # Daily Retuning
        kp_daily, ki_daily = run_scipy_de_tuner(
            f"Retuned {name}", plant_sys, 
            min_kp=-1.0, max_kp=-0.001,   
            min_ki=-1e-3, max_ki=-1e-8,   
            tau=plant_params['tau'], delay=plant_params['delay'], 
            weights=(1.0, 1.0, 1.0, 1.0), target_sp=TARGET_SP
        )
        
        # Simulate Static Tuning Setup (Strictly clamped 0.0 to 1.0)
        y_static, u_static = simulate_saturated_pi(
            plant_sys, kp_static, ki_static, t_eval, sp_eval, u_min=0.0, u_max=1.0
        )
        iae_static, auc_static = calculate_metrics(t_eval, y_static, u_static, sp_eval)
        
        final_iae_data["Static (Day 1) Tuning"].append(iae_static)
        final_auc_data["Static (Day 1) Tuning"].append(auc_static)

        # Simulate Retuned Setup (Strictly clamped 0.0 to 1.0)
        y_retuned, u_retuned = simulate_saturated_pi(
            plant_sys, kp_daily, ki_daily, t_eval, sp_eval, u_min=0.0, u_max=1.0
        )
        iae_retuned, auc_retuned = calculate_metrics(t_eval, y_retuned, u_retuned, sp_eval)
        
        final_iae_data["Daily Retuned"].append(iae_retuned)
        final_auc_data["Daily Retuned"].append(auc_retuned)

        # Plot & Save Graphs for Scenario
        plot_results = [
            {"name": "Static (Day 1) Tuning", "color": "orange", "y": y_static, "u": u_static, "iae": iae_static},
            {"name": "Daily Retuned", "color": "blue", "y": y_retuned, "u": u_retuned, "iae": iae_retuned}
        ]
        plot_daily_comparison(output_dir, name, t_eval, sp_eval, plot_results)

    # Export Cross-Scenario Summary CSVs
    export_summary_csv(os.path.join(output_dir, "tds_unit_step_iae.csv"), final_iae_data, scenario_names)
    export_summary_csv(os.path.join(output_dir, "tds_unit_step_control_effort.csv"), final_auc_data, scenario_names)
    
    print(f"\nAll scenarios processed. Summary CSVs and unit step graphs saved in: {output_dir}")

if __name__ == "__main__":
    main()