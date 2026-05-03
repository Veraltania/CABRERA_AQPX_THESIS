import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy_de_tuner import run_scipy_de_tuner

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

def load_tf_parameters(filepath):
    """Reads K, tau, and theta from the specific CSV format."""
    params = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip generic header row 1
        next(reader) # Skip parameter header row 2
        
        for row in reader:
            if not row or not row[0].startswith('TF_DO'):
                continue
            params.append({
                'name': row[0],
                'K': float(row[1]),
                'tau': float(row[2]),
                'delay': float(row[3])
            })
    return params

def export_summary_csv(filepath, data_dict, scenario_names):
    """Exports the metric dictionary to a CSV matching the required format."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Tuning'] + scenario_names)
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
    # EXPLICITLY extract only K, tau, and delay so 'name' is ignored
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
    """Generates and saves step response and control effort graphs for a specific scenario."""
    tick_positions = np.arange(0, 25, 3) 
    tick_labels = ['6 AM', '9 AM', '12 PM', '3 PM', '6 PM', '9 PM', '12 AM', '3 AM', '6 AM']
    
    # Target width: 7.16 inches. Height scaled to match original 14:8 ratio (7.16 * 8/14 = 4.091)
    fig_width = 7.16
    fig_height = 4.09
    
    # 1. Step Response Plot
    plt.figure(figsize=(fig_width, fig_height)) 
    plt.plot(t_full_hours, sp_full, 'k--', label='Setpoint', alpha=0.6)
    plt.axvline(shift_hour, color='red', linestyle=':', linewidth=2, label='Day/Night Shift (6 PM)')

    for res in results:
        plt.plot(t_full_hours, res["y"], color=res["color"], label=f'{res["name"]} (IAE: {res["iae"]:.0f})')
        
    plt.title(f'Day/Night Cycle Step Response ({scenario_name})', pad=10)
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
    # Saved as PDF
    plt.savefig(os.path.join(output_dir, f"step_response_{scenario_name}.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

    # 2. Control Effort Plot
    plt.figure(figsize=(fig_width, fig_height))
    plt.axvline(shift_hour, color='red', linestyle=':', linewidth=2, label='Day/Night Shift (6 PM)')

    for res in results:
        plt.plot(t_full_hours, np.clip(res["u"], 0, 1), color=res["color"], label=f'{res["name"]} Effort')
        
    plt.title(f'Day/Night Cycle Control Effort ({scenario_name})', pad=10)
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
    # Saved as PDF
    plt.savefig(os.path.join(output_dir, f"control_effort_{scenario_name}.pdf"), format='pdf', bbox_inches='tight')
    plt.close()

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "simulation_graphs_day_night_do")
    os.makedirs(output_dir, exist_ok=True)
    
    # Input File Paths
    day_csv = os.path.join(base_dir, "tf_parameters_do_daytime.csv")
    night_csv = os.path.join(base_dir, "tf_parameters_do_nighttime.csv")
    
    # Load Parameters
    day_params_list = load_tf_parameters(day_csv)
    night_params_list = load_tf_parameters(night_csv)
    
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

    # Date labels corresponding to TF_DO_1, 2, 3, 4
    scenario_names = ["5-Feb", "7-Feb", "25-Feb", "26-Feb"]
    
    # Data Tracking for Summary CSVs
    final_iae_data = {"One-shot": [], "Two-shot": []}
    final_auc_data = {"One-shot": [], "Two-shot": []}

    # Iterate through paired scenarios
    for idx, (day_plant, night_plant) in enumerate(zip(day_params_list, night_params_list)):
        scenario_name = scenario_names[idx] if idx < len(scenario_names) else f"Scenario_{idx+1}"
        print(f"\n================ Processing {scenario_name} ================")
        
        # 1. Auto-Tuning
        kp_day, ki_day = run_de_tuner(f"{scenario_name} Day", day_plant, max_kp=1.5, max_ki=0.002)
        kp_night, ki_night = run_de_tuner(f"{scenario_name} Night", night_plant, max_kp=1.5, max_ki=0.002)

        # 2. Run Set-up 1: One-Shot (Day Tuner Only)
        y_day_os, u_day_os = simulate_period(day_plant, kp_day, ki_day, t_half, sp_half)
        y_night_full_os, u_night_full_os = simulate_period(night_plant, kp_day, ki_day, t_night_sim, sp_night_sim)
        
        y_os_full = np.concatenate([y_day_os, y_night_full_os[warmup_seconds:]])
        u_os_full = np.concatenate([u_day_os, u_night_full_os[warmup_seconds:]])
        iae_os, u_auc_os = calculate_metrics(t_full, y_os_full, u_os_full, sp_full)
        
        final_iae_data["One-shot"].append(iae_os)
        final_auc_data["One-shot"].append(u_auc_os)

        # 3. Run Set-up 2: Two-Shot (Day & Night Tuners)
        y_day_ts, u_day_ts = simulate_period(day_plant, kp_day, ki_day, t_half, sp_half)
        y_night_full_ts, u_night_full_ts = simulate_period(night_plant, kp_night, ki_night, t_night_sim, sp_night_sim)
        
        y_ts_full = np.concatenate([y_day_ts, y_night_full_ts[warmup_seconds:]])
        u_ts_full = np.concatenate([u_day_ts, u_night_full_ts[warmup_seconds:]])
        iae_ts, u_auc_ts = calculate_metrics(t_full, y_ts_full, u_ts_full, sp_full)
        
        final_iae_data["Two-shot"].append(iae_ts)
        final_auc_data["Two-shot"].append(u_auc_ts)

        # 4. Plot & Save Graphs for Scenario
        plot_results = [
            {"name": "Day-only", "color": "orange", "y": y_os_full, "u": u_os_full, "iae": iae_os},
            {"name": "Day/night", "color": "blue", "y": y_ts_full, "u": u_ts_full, "iae": iae_ts}
        ]
        plot_scenario_results(output_dir, scenario_name, t_full_hours, sp_full, shift_hour, plot_results)

    # Export Cross-Scenario Summary CSVs
    export_summary_csv(os.path.join(output_dir, "day_night_do_iae.csv"), final_iae_data, scenario_names)
    export_summary_csv(os.path.join(output_dir, "day_night_do_control_effort.csv"), final_auc_data, scenario_names)
    print(f"\nAll scenarios processed. Summary CSVs and graphs saved in: {output_dir}")

if __name__ == "__main__":
    main()