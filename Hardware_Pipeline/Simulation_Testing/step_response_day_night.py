import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import control as ct

# ==========================================
# HARDWARE PIPELINE IMPORTS
# ==========================================
from Evolutionary_Algorithm_Testing.de.de_optimizer import DEOptimizer

# ==========================================
# 1. CONTROL LIBRARY WRAPPERS
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
        # Step Action
        ref[t >= current_t] = step_sp
        current_t += sequence_config['step_duration']
        
        # Recovery Action
        ref[t >= current_t] = base_sp
        current_t += sequence_config['recovery_duration']
        
    return ref

# ==========================================
# 2. AUTO-TUNING & SIMULATION HELPERS
# ==========================================

def run_de_tuner(name, tf_config, weights=(1.0, 1.0, 1.0, 1.0), min_kp=0.001, max_kp=20.0, min_ki=1e-6, max_ki=0.05):
    print(f"\n[Auto-Tuner] Running DE Optimization: {name}")
    
    tf_params = {
        'tf_num': [tf_config['K']], 
        'tf_den': [tf_config['tau'], 1], 
        'tf_delay': tf_config['delay'],
        'tf_n_pade': 2, 
        'computed_delay': tf_config['delay'], 
        'is_reverse_acting': False, 
        'min_kp': min_kp, 'max_kp': max_kp,
        'min_ki': min_ki, 'max_ki': max_ki
    }
    
    config = {
        'population_size': 1000,
        'patience_limit': 20, 
        'improvement_tol': 1e-4, 
        'mutation': (0.5, 1.0), 
        'recombination': 0.745, 
        'strategy': 'best1bin',
        'weights': weights
    }
    
    optimizer = DEOptimizer(config, tf_params)
    best_sol, _, _ = optimizer.optimize_round(round_num=1)
    best_kp, best_ki, best_cost, _ = best_sol
    
    print(f"[Result] {name} -> Kp: {best_kp:.4f}, Ki: {best_ki:.4f} (Cost: {best_cost:.4f})")
    return best_kp, best_ki

def simulate_period(plant_params, kp, ki, t_eval, setpoints):
    """Simulates a block of time, accounting for baseline offsets."""
    plant = create_fopdt_sys(**plant_params)
    controller = create_pi_controller(kp, ki)
    
    T_y = ct.feedback(ct.series(controller, plant), 1)
    T_u = ct.feedback(controller, plant)

    base_sp = setpoints[0]
    sp_shifted = setpoints - base_sp
    
    _, y_out = ct.forced_response(T_y, t_eval, sp_shifted)
    _, u_out = ct.forced_response(T_u, t_eval, sp_shifted)
    
    # Calculate initial steady state control effort required to maintain base_sp
    u_ss = base_sp / plant_params['K']
    
    return y_out + base_sp, u_out + u_ss

# ==========================================
# 3. METRIC CALCULATIONS
# ==========================================

def calculate_rise_time(t, y, sp, base_sp, step_sp):
    """Calculates 10% to 90% rise time for the first step found."""
    step_indices = np.where(sp == step_sp)[0]
    if len(step_indices) == 0: return 0.0
    
    start_idx = step_indices[0]
    end_idx = start_idx
    while end_idx < len(sp) and sp[end_idx] == step_sp:
        end_idx += 1
        
    y_step = y[start_idx:end_idx]
    t_step = t[start_idx:end_idx]
    
    y_10 = base_sp + 0.1 * (step_sp - base_sp)
    y_90 = base_sp + 0.9 * (step_sp - base_sp)
    
    idx_10 = np.where(y_step >= y_10)[0]
    idx_90 = np.where(y_step >= y_90)[0]
    
    if len(idx_10) > 0 and len(idx_90) > 0:
        return t_step[idx_90[0]] - t_step[idx_10[0]]
    return 0.0

def calculate_overshoot(y, sp, base_sp, step_sp):
    """Calculates maximum overshoot percentage across all steps."""
    max_os = 0.0
    in_step = False
    current_max_y = 0.0
    
    for i in range(len(sp)):
        if sp[i] == step_sp:
            if not in_step:
                in_step = True
                current_max_y = y[i]
            else:
                if y[i] > current_max_y:
                    current_max_y = y[i]
        else:
            if in_step:
                os = ((current_max_y - step_sp) / (step_sp - base_sp)) * 100
                if os > max_os: 
                    max_os = os
                in_step = False
    return max(0, max_os)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def main():
    # =======================================================
    # PLOT CONFIGURATION BLOCK (Local Variables)
    # Edit these to easily switch between DO, TDS, pH, etc.
    # =======================================================
    plot_font_size = 18

    # Step Response Labels
    step_title = 'Day/Night Cycle Step Response'
    step_x_label = 'Time (hours)'
    step_y_label = 'DO (mg/l)'  # <-- Change to 'TDS (ppm)' when ready!

    # Control Effort Labels
    effort_title = 'Control Effort Comparison Across Shifts'
    effort_x_label = 'Time (hours)'
    effort_y_label = 'Control Signal'
    # =======================================================

    folder_name = "simulation_graphs_day_night_feb7"
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. System Definition
    plant_params_day = {'K': 1.1334 , 'tau': 2833.8197 , 'delay': 0.05} 
    plant_params_night = {'K': 2.0494, 'tau': 4499.9964, 'delay': 0.05}
    
    seq_config = {
        'base_sp': 0.5, 'step_sp': 1.5,
        'pre_step_delay': 10000, 'step_duration': 10000,
        'recovery_duration': 10000, 'cycles': 2
    }
    
    # Time Constants (Seconds)
    SEC_PER_HOUR = 3600
    TWELVE_HOURS = 12 * SEC_PER_HOUR
    
    # 12-Hour Mirror Config: Sums to 43,200s (12h)
    # 2h baseline, 2 cycles of (2h step + 3h recovery)
    seq_config = {
        'base_sp': 0.5, 'step_sp': 1.5,
        'pre_step_delay': 2 * SEC_PER_HOUR, 
        'step_duration': 2 * SEC_PER_HOUR,
        'recovery_duration': 3 * SEC_PER_HOUR, 
        'cycles': 2 
    }

    # Time Vectors
    t_half = np.linspace(0, TWELVE_HOURS, TWELVE_HOURS + 1)
    sp_half = generate_setpoint_array(t_half, seq_config)
    
    t_full = np.concatenate([t_half, t_half + TWELVE_HOURS + 1])
    sp_full = np.concatenate([sp_half, sp_half])
    t_full_hours = t_full / SEC_PER_HOUR

    # 2. Strategy Executions
    results = []

    print("\n--- Running Set-up 1: One-Shot (Day Tuner Only) ---")
    kp_os, ki_os = run_de_tuner("One-Shot Day Tuner", plant_params_day, max_kp=1.5, max_ki=0.002)
    y_day_os, u_day_os = simulate_period(plant_params_day, kp_os, ki_os, t_half, sp_half)
    y_night_os, u_night_os = simulate_period(plant_params_night, kp_os, ki_os, t_half, sp_half)
    
    results.append({
        "name": "One-Shot", "color": "orange",
        "y_day": y_day_os, "y_night": y_night_os,
        "y": np.concatenate([y_day_os, y_night_os]),
        "u": np.concatenate([u_day_os, u_night_os])
    })

    print("\n--- Running Set-up 2: Two-Shot (Day & Night Tuners) ---")
    kp_ts_day, ki_ts_day = run_de_tuner("Two-Shot Day Tuner", plant_params_day, max_kp=1.5, max_ki=0.002)
    kp_ts_night, ki_ts_night = run_de_tuner("Two-Shot Night Tuner", plant_params_night, max_kp=1.5, max_ki=0.002)
    y_day_ts, u_day_ts = simulate_period(plant_params_day, kp_ts_day, ki_ts_day, t_half, sp_half)
    y_night_ts, u_night_ts = simulate_period(plant_params_night, kp_ts_night, ki_ts_night, t_half, sp_half)
    
    results.append({
        "name": "Two-Shot", "color": "blue",
        "y_day": y_day_ts, "y_night": y_night_ts,
        "y": np.concatenate([y_day_ts, y_night_ts]),
        "u": np.concatenate([u_day_ts, u_night_ts])
    })

    # 3. Compute Metrics
    csv_data = []
    for res in results:
        y_full = res["y"]
        u_full = res["u"]
        error = sp_full - y_full
        
        iae = np.trapezoid(np.abs(error), t_full)
        mae = np.mean(np.abs(error))
        u_auc = np.trapezoid(np.clip(u_full, 0, 1), t_full)
        
        rt_day = calculate_rise_time(t_half, res["y_day"], sp_half, seq_config['base_sp'], seq_config['step_sp'])
        rt_night = calculate_rise_time(t_half, res["y_night"], sp_half, seq_config['base_sp'], seq_config['step_sp'])
        os_day = calculate_overshoot(res["y_day"], sp_half, seq_config['base_sp'], seq_config['step_sp'])
        os_night = calculate_overshoot(res["y_night"], sp_half, seq_config['base_sp'], seq_config['step_sp'])
        
        res.update({"iae": iae, "mae": mae, "u_auc": u_auc, 
                    "rt_day": rt_day, "rt_night": rt_night,
                    "os_day": os_day, "os_night": os_night})
                    
        csv_data.append([
            res["name"], f"{iae:.2f}", f"{mae:.4f}", f"{u_auc:.2f}", 
            f"{rt_day:.2f}", f"{rt_night:.2f}", f"{os_day:.2f}", f"{os_night:.2f}"
        ])

    csv_path = os.path.join(output_dir, "simulation_metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Setup', 'IAE', 'MAE', 'AUC_Effort', 'Rise_Time_Day_s', 'Rise_Time_Night_s', 'Overshoot_Day_pct', 'Overshoot_Night_pct'])
        writer.writerows(csv_data)

    # ==========================================
    # 5. PLOTTING WITH LOCAL CONFIG VARIABLES
    # ==========================================
    
    # ---- Figure 1: Step Response ----
    plt.figure(figsize=(14, 8)) # Slightly taller to make room for top legend
    plt.plot(t_full_hours, sp_full, 'k--', label='Reference Setpoint', alpha=0.6)
    
    # Red dotted line bisecting day and night
    plt.axvline(t_half, color='red', linestyle=':', linewidth=2, label='Day/Night Shift')

    for res in results:
        plt.plot(t_full_hours, res["y"], color=res["color"], label=f'{res["name"]} (IAE: {res["iae"]:.0f})')
        
    # Apply Font Sizes & Labels
    plt.title(step_title, fontsize=plot_font_size, pad=20)
    plt.xlabel(step_x_label, fontsize=plot_font_size)
    plt.ylabel(step_y_label, fontsize=plot_font_size)
    plt.xticks(fontsize=plot_font_size)
    plt.yticks(fontsize=plot_font_size)
 
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, 
           fontsize=plot_font_size, borderaxespad=0.)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout() # Ensures the outside legend isn't cut off
    plt.savefig(os.path.join(output_dir, "step_response_day_night.png"))

    # ---- Figure 2: Control Effort ----
    plt.figure(figsize=(14, 8))
    
    # Red dotted line bisecting day and night
    plt.axvline(time_half_hours, color='red', linestyle=':', linewidth=2, label='Day/Night Shift')

    for res in results:
        plt.plot(t_full_hours, np.clip(res["u"], 0, 1), color=res["color"], label=f'{res["name"]} Effort')
        
    # Apply Font Sizes & Labels
    plt.title(effort_title, fontsize=plot_font_size, pad=20)
    plt.xlabel(effort_x_label, fontsize=plot_font_size)
    plt.ylabel(effort_y_label, fontsize=plot_font_size)
    plt.xticks(fontsize=plot_font_size)
    plt.yticks(fontsize=plot_font_size)
    plt.ylim(-0.05, 1.1)
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, 
           fontsize=plot_font_size, borderaxespad=0.)
               
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "control_effort_day_night.png"))
    
    print(f"\nSimulation Complete! Data and plots saved in: {output_dir}")

if __name__ == "__main__":
    main()