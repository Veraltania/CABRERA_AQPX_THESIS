import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import control as ct

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
    """Wrapper function to utilize the extracted SciPy DE Tuner."""
    plant = create_fopdt_sys(tf_config['K'], tf_config['tau'], tf_config['delay'])
    
    best_kp, best_ki = run_scipy_de_tuner(
        name=name, 
        plant=plant, 
        min_kp=min_kp, 
        max_kp=max_kp, 
        min_ki=min_ki, 
        max_ki=max_ki, 
        tau=tf_config['tau'], 
        delay=tf_config['delay'], 
        weights=weights,
        target_sp=1.0 
    )
    
    return best_kp, best_ki

def simulate_period(plant_params, kp, ki, t_eval, setpoints):
    """Simulates using pure control library."""
    plant = create_fopdt_sys(**plant_params)
    controller = create_pi_controller(kp, ki)
    
    T_y = ct.feedback(ct.series(controller, plant), 1)
    T_u = ct.feedback(controller, plant)

    base_sp = setpoints[0]
    sp_shifted = setpoints - base_sp
    
    # Standard forced response from rest
    time_res_y = ct.forced_response(T_y, T=t_eval, U=sp_shifted)
    time_res_u = ct.forced_response(T_u, T=t_eval, U=sp_shifted)
    
    u_ss = base_sp / plant_params['K']
    
    return time_res_y.outputs + base_sp, time_res_u.outputs + u_ss

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
    plot_font_size = 22

    # Step Response Labels
    step_title = 'Day/Night Cycle Step Response'
    step_x_label = 'Time (hours)'
    step_y_label = 'DO (mg/l)'  # <-- Change to 'TDS (ppm)' when ready!

    # Control Effort Labels
    effort_title = 'Day/Night Cycle Control Effort'
    effort_x_label = 'Time (hours)'
    effort_y_label = 'Control Signal'
    # =======================================================

    folder_name = "simulation_graphs_day_night_feb5"
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. System Definition
    plant_params_day = {'K': 1.346, 'tau': 1551.9555, 'delay': 104.6485} 
    plant_params_night = {'K': 2.355, 'tau': 3083.5899, 'delay': 0.05}
    
    # Time Constants (Seconds)
    SEC_PER_HOUR = 3600
    TWELVE_HOURS = 12 * SEC_PER_HOUR
    shift_hour = TWELVE_HOURS / SEC_PER_HOUR  # Used dynamically for plotting
    
    # 12-Hour Mirror Config: Sums to 43,200s (12h)
    # 2h baseline, 2 cycles of (2h step + 3h recovery)
    seq_config = {
        'base_sp': 0.5, 'step_sp': 1.5,
        'pre_step_delay': 1800, 
        'step_duration': 10800,
        'recovery_duration': 7600, 
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

    print("\n--- Phase 1: Auto-Tuning Controllers ---")
    kp_day, ki_day = run_de_tuner("Day Tuner", plant_params_day, max_kp=1.5, max_ki=0.002)
    kp_night, ki_night = run_de_tuner("Night Tuner", plant_params_night, max_kp=1.5, max_ki=0.002)

    # --- Setup Warm-up arrays for the Night Shift ---
    # We "pre-roll" the night simulation by 3 hours so the Padé states settle correctly
    warmup_seconds = 3 * SEC_PER_HOUR
    sp_warmup = sp_half[-warmup_seconds:] # Grab the last 3 hours of the day's setpoints
    
    # Create an extended time and setpoint array for the night simulation
    t_night_sim = np.arange(len(sp_half) + warmup_seconds) 
    sp_night_sim = np.concatenate([sp_warmup, sp_half])

    print("\n--- Phase 2: Running Set-up 1: One-Shot (Day Tuner Only) ---")
    # 1. Simulate Day normally
    y_day_os, u_day_os = simulate_period(plant_params_day, kp_day, ki_day, t_half, sp_half)
    
    # 2. Simulate Night with warm-up
    y_night_full_os, u_night_full_os = simulate_period(plant_params_night, kp_day, ki_day, t_night_sim, sp_night_sim)
    
    # 3. Slice off the warm-up period to get the exact 12-hour night shift
    y_night_os = y_night_full_os[warmup_seconds:]
    u_night_os = u_night_full_os[warmup_seconds:]
    
    results.append({
        "name": "Day-only", "color": "orange",
        "y_day": y_day_os, "y_night": y_night_os,
        "y": np.concatenate([y_day_os, y_night_os]),
        "u": np.concatenate([u_day_os, u_night_os])
    })

    print("\n--- Phase 3: Running Set-up 2: Two-Shot (Day & Night Tuners) ---")
    # 1. Simulate Day normally
    y_day_ts, u_day_ts = simulate_period(plant_params_day, kp_day, ki_day, t_half, sp_half)
    
    # 2. Simulate Night with warm-up (using Night Tuner gains)
    y_night_full_ts, u_night_full_ts = simulate_period(plant_params_night, kp_night, ki_night, t_night_sim, sp_night_sim)
    
    # 3. Slice off the warm-up
    y_night_ts = y_night_full_ts[warmup_seconds:]
    u_night_ts = u_night_full_ts[warmup_seconds:]
    
    results.append({
        "name": "Day/Night", "color": "blue",
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
    
    # --- Custom X-Axis Time Mapping ---
    # Simulation runs from 0 to 24 hours. We map these to 6 AM -> 6 AM next day.
    # We will place a tick mark every 3 hours.
    tick_positions = np.arange(0, 25, 3) 
    tick_labels = ['6 AM', '9 AM', '12 PM', '3 PM', '6 PM', '9 PM', '12 AM', '3 AM', '6 AM']

    # ---- Figure 1: Step Response ----
    plt.figure(figsize=(14, 8)) 
    plt.plot(t_full_hours, sp_full, 'k--', label='Reference Setpoint', alpha=0.6)
    
    # Red dotted line bisecting day and night
    plt.axvline(shift_hour, color='red', linestyle=':', linewidth=2, label='Day/Night Shift (6 PM)')

    for res in results:
        plt.plot(t_full_hours, res["y"], color=res["color"], label=f'{res["name"]} (IAE: {res["iae"]:.0f})')
        
    # Apply Font Sizes & Labels
    plt.title(step_title, fontsize=plot_font_size, pad=20)
    plt.xlabel("Time of Day", fontsize=plot_font_size) # Changed label
    plt.ylabel(step_y_label, fontsize=plot_font_size)
    
    # Apply custom time labels to the X-axis
    plt.xticks(tick_positions, tick_labels, fontsize=plot_font_size)
    plt.yticks(fontsize=plot_font_size)
 
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, 
           fontsize=plot_font_size, borderaxespad=0.)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout() 
    plt.savefig(os.path.join(output_dir, "step_response_day_night.png"))

    # ---- Figure 2: Control Effort ----
    plt.figure(figsize=(14, 8))
    
    # Red dotted line bisecting day and night
    plt.axvline(shift_hour, color='red', linestyle=':', linewidth=2, label='Day/Night Shift (6 PM)')

    for res in results:
        plt.plot(t_full_hours, np.clip(res["u"], 0, 1), color=res["color"], label=f'{res["name"]} Effort')
        
    # Apply Font Sizes & Labels
    plt.title(effort_title, fontsize=plot_font_size, pad=20)
    plt.xlabel("Time of Day", fontsize=plot_font_size) # Changed label
    plt.ylabel(effort_y_label, fontsize=plot_font_size)
    
    # Apply custom time labels to the X-axis
    plt.xticks(tick_positions, tick_labels, fontsize=plot_font_size)
    plt.yticks(fontsize=plot_font_size)
    plt.ylim(-0.05, 1.1)
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, 
           fontsize=plot_font_size, borderaxespad=0.)
               
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "control_effort_day_night.png"))

if __name__ == "__main__":
    main()