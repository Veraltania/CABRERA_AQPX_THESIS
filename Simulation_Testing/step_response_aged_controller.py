import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy_de_tuner import run_scipy_de_tuner

# Apply global font settings for Times New Roman, Size 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

# ==========================================
# 1. PLANT & CONTROLLER DEFINITIONS
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

def create_pi_controller(kp, ki):
    """Creates a PI Controller Transfer Function: (kp*s + ki) / s"""
    return ct.tf([kp, ki], [1, 0])

def generate_setpoint_array(t, sequence_config):
    """Generates the reference signal array for the simulation time vector."""
    base_sp = sequence_config['base_sp']
    step_sp = sequence_config['step_sp']
    ref = np.full_like(t, base_sp, dtype=float)
    
    current_t = sequence_config['pre_step_delay']
    for _ in range(sequence_config['cycles']):
        ref[t >= current_t] = step_sp
        current_t += sequence_config['step_duration']
        ref[t >= current_t] = base_sp
        current_t += sequence_config['recovery_duration']
        
    return ref

# ==========================================
# 2. DATA I/O MODULE
# ==========================================

def load_tf_parameters_dict(filepath):
    """Reads K, tau, and theta from the CSV into a dictionary keyed by name."""
    params = {}
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip generic header row 1
        next(reader) # Skip parameter header row 2
        
        for row in reader:
            # We are extracting the plant name and its K, tau, delay parameters
            if not row or not row[0].startswith('TF_DO'):
                continue
            name = row[0]
            params[name] = {
                'name': name,
                'K': float(row[1]),
                'tau': float(row[2]),
                'delay': float(row[3])
            }
    return params

def export_summary_csv(filepath, data_dict, scenario_names):
    """Exports the metric dictionary to a CSV matching the required format."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Controller Setup'] + scenario_names)
        for tuning, values in data_dict.items():
            formatted_vals = [f"{v:.2f}" for v in values]
            writer.writerow([tuning] + formatted_vals)

# ==========================================
# 3. SIMULATION & METRICS ENGINE
# ==========================================

def run_de_tuner(name, tf_config, weights=(1.0, 1.0, 1.0, 1.0), min_kp=0.001, max_kp=20.0, min_ki=1e-6, max_ki=0.05):
    """Wrapper function to utilize the extracted SciPy DE Tuner."""
    plant = create_fopdt_sys(tf_config['K'], tf_config['tau'], tf_config['delay'])
    return run_scipy_de_tuner(
        name=name, plant=plant, 
        min_kp=min_kp, max_kp=max_kp, min_ki=min_ki, max_ki=max_ki, 
        tau=tf_config['tau'], delay=tf_config['delay'], 
        weights=weights, target_sp=1.0 
    )

def simulate_period(plant_params, kp, ki, t_eval, setpoints):
    """Simulates using pure control library."""
    plant = create_fopdt_sys(
        K=plant_params['K'], 
        tau=plant_params['tau'], 
        delay=plant_params['delay']
    )
    controller = create_pi_controller(kp, ki)
    
    T_y = ct.feedback(ct.series(controller, plant), 1)
    T_u = ct.feedback(controller, plant)

    base_sp = setpoints[0]
    sp_shifted = setpoints - base_sp
    
    time_res_y = ct.forced_response(T_y, T=t_eval, U=sp_shifted)
    time_res_u = ct.forced_response(T_u, T=t_eval, U=sp_shifted)
    
    u_ss = base_sp / plant_params['K']
    return time_res_y.outputs + base_sp, time_res_u.outputs + u_ss

def calculate_metrics(t, y, u, sp):
    """Calculates Integral Absolute Error and Area Under Curve (Control Effort)."""
    error = sp - y
    iae = np.trapezoid(np.abs(error), t)
    u_auc = np.trapezoid(np.clip(u, 0, 1), t)
    return iae, u_auc

# ==========================================
# 4. PLOTTING MODULE
# ==========================================

def plot_scenario_results(output_dir, scenario_name, t_full_hours, sp_full, shift_hour, results):
    """Generates and saves step response and control effort graphs for a specific scenario in PDF format."""
    
    tick_positions = np.arange(0, 25, 3) 
    tick_labels = ['6 AM', '9 AM', '12 PM', '3 PM', '6 PM', '9 PM', '12 AM', '3 AM', '6 AM']
    
    # 1. Step Response Plot
    # Sized to exactly 7.16 inches wide. Height set to 4.5 inches to maintain a clean aspect ratio.
    plt.figure(figsize=(7.16, 4.5)) 
    plt.plot(t_full_hours, sp_full, 'k--', label='Setpoint', alpha=0.6)
    plt.axvline(shift_hour, color='red', linestyle=':', linewidth=1.5, label='Day/Night Shift (6 PM)')

    for res in results:
        plt.plot(t_full_hours, res["y"], color=res["color"], label=f'{res["name"]} (IAE: {res["iae"]:.0f})')
        
    plt.title(f'Performance of Aged vs Fresh Controllers ({scenario_name})', pad=15)
    plt.xlabel("Time of Day")
    plt.ylabel('DO (mg/l)')
    plt.xticks(tick_positions, tick_labels)
    
    plt.legend(
        loc='best', 
        fontsize=8, 
        labelspacing=0.3,   # Reduces vertical space between items
        handlelength=1.5,   # Makes the colored line segments shorter
        handletextpad=0.4,  # Reduces space between the line and the text
        borderpad=0.3       # Shrinks the padding around the edges of the box
    )
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout() 
    
    # Save as PDF vector graphic
    plt.savefig(os.path.join(output_dir, f"step_response_{scenario_name}.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

    # 2. Control Effort Plot
    plt.figure(figsize=(7.16, 4.5))
    plt.axvline(shift_hour, color='red', linestyle=':', linewidth=1.5, label='Day/Night Shift (6 PM)')

    for res in results:
        plt.plot(t_full_hours, np.clip(res["u"], 0, 1), color=res["color"], label=f'{res["name"]} Effort')
        
    plt.title(f'Control Effort: Aged vs Fresh Controllers ({scenario_name})', pad=15)
    plt.xlabel("Time of Day")
    plt.ylabel('Control Signal')
    plt.xticks(tick_positions, tick_labels)
    plt.ylim(-0.05, 1.1)
    
    plt.legend(
        loc='best', 
        fontsize=8, 
        labelspacing=0.3,
        handlelength=1.5,
        handletextpad=0.4,
        borderpad=0.3
    )
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save as PDF vector graphic
    plt.savefig(os.path.join(output_dir, f"control_effort_{scenario_name}.pdf"), format='pdf', bbox_inches='tight')
    plt.close()
    
# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "simulation_graphs_controller_aging")
    os.makedirs(output_dir, exist_ok=True)
    
    # Input File Path (Combined dated parameters)
    csv_file = os.path.join(base_dir, "tf_parameters_do_dated.csv")
    
    # Load Parameters into a dictionary lookup
    params_dict = load_tf_parameters_dict(csv_file)
    
    # Setup Time and Reference Arrays
    SEC_PER_HOUR = 3600
    TWELVE_HOURS = 12 * SEC_PER_HOUR
    shift_hour = TWELVE_HOURS / SEC_PER_HOUR 
    
    seq_config = {
        'base_sp': 0.5, 'step_sp': 1.5,
        'pre_step_delay': 1800, 'step_duration': 10800,
        'recovery_duration': 7600, 'cycles': 2 
    }

    t_half = np.linspace(0, TWELVE_HOURS, TWELVE_HOURS + 1)
    sp_half = generate_setpoint_array(t_half, seq_config)
    
    t_full = np.concatenate([t_half, t_half + TWELVE_HOURS + 1])
    sp_full = np.concatenate([sp_half, sp_half])
    t_full_hours = t_full / SEC_PER_HOUR

    warmup_seconds = 3 * SEC_PER_HOUR
    sp_warmup = sp_half[-warmup_seconds:] 
    t_night_sim = np.arange(len(sp_half) + warmup_seconds) 
    sp_night_sim = np.concatenate([sp_warmup, sp_half])

    # ====================================================
    # BASELINE: Tune on Feb 5 Daytime Data
    # ====================================================
    print("\n--- Establishing Baseline (Tuning on Feb 5 Daytime) ---")
    baseline_plant = params_dict['TF_DO_FEB_5_DAYTIME']
    kp_aged, ki_aged = run_de_tuner("Feb 5 Baseline", baseline_plant, max_kp=1.5, max_ki=0.002)
    print(f"Feb 5 Daytime Baseline Gains tuned -> Kp: {kp_aged:.4f}, Ki: {ki_aged:.6f}")
    
    # Dates to test the "aged" controller on
    test_dates = ["FEB_25", "FEB_26"]
    
    # Data Tracking for Summary CSVs
    final_iae_data = {"Aged (Feb 5) Tuner": [], "Fresh (Current Day) Tuner": []}
    final_auc_data = {"Aged (Feb 5) Tuner": [], "Fresh (Current Day) Tuner": []}

    # Iterate through test date scenarios
    for date_str in test_dates:
        scenario_name = f"{date_str.replace('_', ' ')}"
        print(f"\n================ Evaluating Controller on {scenario_name} ================")
        
        # Load the physical plant data for the specific testing date
        day_plant = params_dict[f'TF_DO_{date_str}_DAYTIME']
        night_plant = params_dict[f'TF_DO_{date_str}_NIGHTTIME']
        
        # 1. Fresh Tuning: Auto-Tune on the current test day/night data
        kp_fresh_day, ki_fresh_day = run_de_tuner(f"{date_str} Fresh Day", day_plant, max_kp=1.5, max_ki=0.002)
        kp_fresh_night, ki_fresh_night = run_de_tuner(f"{date_str} Fresh Night", night_plant, max_kp=1.5, max_ki=0.002)

        # 2. Run Set-up A: "Aged" Controller (Feb 5 Daytime tuning applied across the whole new date)
        y_day_aged, u_day_aged = simulate_period(day_plant, kp_aged, ki_aged, t_half, sp_half)
        y_night_full_aged, u_night_full_aged = simulate_period(night_plant, kp_aged, ki_aged, t_night_sim, sp_night_sim)
        
        y_aged_full = np.concatenate([y_day_aged, y_night_full_aged[warmup_seconds:]])
        u_aged_full = np.concatenate([u_day_aged, u_night_full_aged[warmup_seconds:]])
        iae_aged, u_auc_aged = calculate_metrics(t_full, y_aged_full, u_aged_full, sp_full)
        
        final_iae_data["Aged (Feb 5) Tuner"].append(iae_aged)
        final_auc_data["Aged (Feb 5) Tuner"].append(u_auc_aged)

        # 3. Run Set-up B: "Fresh" Controller (Fresh Day tuning for daytime, Fresh Night tuning for nighttime)
        y_day_fresh, u_day_fresh = simulate_period(day_plant, kp_fresh_day, ki_fresh_day, t_half, sp_half)
        y_night_full_fresh, u_night_full_fresh = simulate_period(night_plant, kp_fresh_night, ki_fresh_night, t_night_sim, sp_night_sim)
        
        y_fresh_full = np.concatenate([y_day_fresh, y_night_full_fresh[warmup_seconds:]])
        u_fresh_full = np.concatenate([u_day_fresh, u_night_full_fresh[warmup_seconds:]])
        iae_fresh, u_auc_fresh = calculate_metrics(t_full, y_fresh_full, u_fresh_full, sp_full)
        
        final_iae_data["Fresh (Current Day) Tuner"].append(iae_fresh)
        final_auc_data["Fresh (Current Day) Tuner"].append(u_auc_fresh)

        # 4. Plot & Save Graphs for Scenario
        plot_results = [
            {"name": "Aged (Feb 5)", "color": "orange", "y": y_aged_full, "u": u_aged_full, "iae": iae_aged},
            {"name": "Fresh (Day/Night)", "color": "blue", "y": y_fresh_full, "u": u_fresh_full, "iae": iae_fresh}
        ]
        plot_scenario_results(output_dir, scenario_name, t_full_hours, sp_full, shift_hour, plot_results)

    # Export Cross-Scenario Summary CSVs
    csv_headers = [d.replace('_', '-') for d in test_dates]
    export_summary_csv(os.path.join(output_dir, "controller_aging_iae.csv"), final_iae_data, csv_headers)
    export_summary_csv(os.path.join(output_dir, "controller_aging_control_effort.csv"), final_auc_data, csv_headers)
    print(f"\nAll scenarios processed. Summary CSVs and graphs saved in: {output_dir}")

if __name__ == "__main__":
    main()