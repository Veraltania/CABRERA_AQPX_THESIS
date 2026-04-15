import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy.optimize import differential_evolution

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
# 2. DISCRETE SIMULATION WITH SATURATION
# ==========================================

def simulate_saturated_pi(plant, kp, ki, t_eval, setpoints, u_min=0.0, u_max=1.0):
    """
    Simulates the closed loop response step-by-step, explicitly clamping 
    the control effort (u) to [u_min, u_max] to model physical actuators like pumps/aerators.
    """
    dt = t_eval[1] - t_eval[0]
    
    sys_d_tf = ct.c2d(plant, dt)
    sys_d_ss = ct.ss(sys_d_tf)  
    A, B, C, D = sys_d_ss.A, sys_d_ss.B, sys_d_ss.C, sys_d_ss.D
    
    n_steps = len(t_eval)
    y_out = np.zeros(n_steps)
    u_out = np.zeros(n_steps)
    
    x = np.zeros((sys_d_ss.nstates, 1))
    integral_error = 0.0
    
    for i in range(n_steps):
        sp = setpoints[i]
        
        current_y = (C @ x)[0, 0]
        y_out[i] = current_y
        
        error = sp - current_y
        integral_error += error * dt
        
        u_raw = (kp * error) + (ki * integral_error)
        
        # Actuator Saturation
        u_clamped = np.clip(u_raw, u_min, u_max)
        u_out[i] = u_clamped
        
        # Anti-windup
        if u_raw != u_clamped and ki != 0:
            integral_error -= error * dt 
            
        if i < n_steps - 1:
            x = A @ x + B * u_clamped
            
    return y_out, u_out

# ==========================================
# 3. METRIC HELPERS
# ==========================================

def calculate_rise_time(t, y, sp, base_sp, step_sp):
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
    
    idx_10 = np.where(np.abs(y_step - base_sp) >= np.abs(y_10 - base_sp))[0]
    idx_90 = np.where(np.abs(y_step - base_sp) >= np.abs(y_90 - base_sp))[0]
    
    if len(idx_10) > 0 and len(idx_90) > 0:
        return t_step[idx_90[0]] - t_step[idx_10[0]]
    return 0.0

def calculate_overshoot(y, sp, base_sp, step_sp):
    max_os = 0.0
    in_step = False
    current_extreme_y = base_sp
    
    for i in range(len(sp)):
        if sp[i] == step_sp:
            if not in_step:
                in_step = True
                current_extreme_y = y[i]
            else:
                if step_sp > base_sp:
                    if y[i] > current_extreme_y: current_extreme_y = y[i]
                else:
                    if y[i] < current_extreme_y: current_extreme_y = y[i]
        else:
            if in_step:
                os = ((current_extreme_y - step_sp) / (step_sp - base_sp)) * 100
                if os > max_os: 
                    max_os = os
                in_step = False
    return max(0, max_os)

# ==========================================
# 4. SCIPY DE TUNER (UPDATED FOR DIRECTION)
# ==========================================
def run_scipy_de_tuner(name, plant, min_kp, max_kp, min_ki, max_ki, tau, delay, weights, 
                       target_sp=1.0, max_overshoot_limit=0.20):
    """Uses SciPy's DE applying the realistic saturated simulation"""
    print(f"\n[Auto-Tuner] Running SciPy DE Optimization: {name}")
    
    T_sim = (tau * 3) + delay
    t_opt = np.linspace(0, T_sim, 1000)
    sp_opt = np.full_like(t_opt, target_sp) 
    
    avg_rise_time = tau * 2.2
    w_iae, w_effort, w_os, w_rt = weights

    def objective(params):
        kp, ki = params
        penalty = 1e20
        
        try:
            y_out, u_out = simulate_saturated_pi(plant, kp, ki, t_opt, sp_opt, u_min=0.0, u_max=1.0)
            
            # Normalize y_out
            y_norm = y_out * np.sign(target_sp)
            
            if np.any(np.isnan(y_norm)) or np.any(np.isinf(y_norm)):
                return penalty
                
            # --- OVERSHOOT CALCULATION ---
            # Peak overshoot is the maximum value minus the setpoint (1.0 in y_norm)
            peak_val = np.max(y_norm)
            actual_overshoot = max(0.0, peak_val - 1.0) 
            
            # 1. Hard Constraint: If overshoot exceeds the limit, kill this candidate
            if actual_overshoot > max_overshoot_limit:
                return penalty

            # 2. Stability Check: Kill candidates that oscillate wildly below zero
            if np.min(y_norm) < -0.1:
                return penalty
            # -----------------------------

            error = 1.0 - y_norm
            int_error = np.trapezoid(np.abs(error), t_opt)
            
            # --- THE PROPER NORMALIZATION: Total Variation ---
            # Levels the playing field between Kp and Ki, but strictly punishes oscillation/chatter
            # Prepend 0.0 to capture the initial jump at t=0
            u_with_initial = np.concatenate(([0.0], u_out))
            norm_effort = np.sum(np.abs(np.diff(u_with_initial)))
            
            # Integral of Overshoot Area (for the soft cost)
            overshoot_array = np.where(error < 0, np.abs(error), 0.0)
            int_overshoot = np.trapezoid(overshoot_array, t_opt)

            crossings_10 = np.where(y_norm >= 0.1)[0]
            crossings_90 = np.where(y_norm >= 0.9)[0]
            
            if len(crossings_10) > 0 and len(crossings_90) > 0:
                rise_time = t_opt[crossings_90[0]] - t_opt[crossings_10[0]]
            else:
                rise_time = T_sim * 10
                
            norm_error = int_error / T_sim
            norm_overshoot = int_overshoot / T_sim
            norm_rise_time = rise_time / avg_rise_time
            
            # Kill candidates that are wildly unstable
            if norm_error > 2.0 or norm_rise_time > 10.0:
                return penalty
                
            cost = (w_iae * norm_error) + (w_effort * norm_effort) + (w_os * norm_overshoot) + (w_rt * norm_rise_time)
            return cost
            
        except Exception as e:
            return penalty

    bounds = [(min_kp, max_kp), (min_ki, max_ki)]
    
    result = differential_evolution(
        objective, 
        bounds, 
        strategy='best1bin', 
        maxiter=30,      
        popsize=20,     
        mutation=(0.5, 1.0), 
        recombination=0.745, 
        tol=1e-4,
        disp=False
    )
    
    best_kp, best_ki = result.x
    print(f"[Result] {name} -> Kp: {best_kp:.4f}, Ki: {best_ki:.4f} (Cost: {result.fun:.4f})")
    return best_kp, best_ki

# ==========================================
# 5. UTILITIES
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
# 6. MAIN EXECUTION
# ==========================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(base_dir, "tf_parameters_do.csv")
    output_dir = os.path.join(base_dir, "simulation_graphs_comparison_do_0_75xcf")
    os.makedirs(output_dir, exist_ok=True)
    
    aggregated_metrics = {'IAE': {}, 'Control_Effort': {}, 'Rise_Time': {}, 'Overshoot': {}}
    tf_list = read_tf_parameters(input_csv)
    
    cf_weight = 0.75
    perf_weight = (4.0 - cf_weight) / 3
    de_weights = [perf_weight, cf_weight, perf_weight, perf_weight]

    for tf_data in tf_list:
        tf_name = tf_data['name']
        print(f"\n--- Processing Transfer Function: {tf_name} ---")
        
        for key in aggregated_metrics:
            aggregated_metrics[key][tf_name] = {}
        
        plant_params = {'K': tf_data['K'], 'tau': tf_data['tau'], 'delay': tf_data['delay']} 
        plant = create_fopdt_sys(**plant_params)
        
        target_sp = 1.0 if plant_params['K'] > 0 else -1.0
        
        time_factor = max(plant_params['tau'], 1000)
        seq_config = {
            'base_sp': 0.0, 
            'step_sp': target_sp,
            'pre_step_delay': max(time_factor, plant_params['delay']) * 2, 
            'step_duration': time_factor * 8,
            'recovery_duration': time_factor * 8, 
            'cycles': 1
        }
        
        total_time = seq_config['pre_step_delay'] + (seq_config['step_duration'] + seq_config['recovery_duration']) * seq_config['cycles']
        num_points = min(int(total_time) + 1, 10000) 
        t_eval = np.linspace(0, total_time, num_points)
        t_eval_hours = t_eval / 3600.0  
        setpoints = generate_setpoint_array(t_eval, seq_config)

        if plant_params['K'] < 0:
            max_kp, min_kp = 0, -1
            max_ki, min_ki = 0, -0.0005 
        else:
            max_kp, min_kp = 1.5, 0
            max_ki, min_ki = 0.01, 0

        # Run Auto-Tuner
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

        fig_y, ax_y = plt.subplots(figsize=(14, 8))
        fig_u, ax_u = plt.subplots(figsize=(14, 8))

        ax_y.plot(t_eval_hours, setpoints, 'k--', label='Reference Setpoint', alpha=0.6)

        for cfg in configs:
            y_out, u_out = simulate_saturated_pi(plant, cfg["kp"], cfg["ki"], t_eval, setpoints, u_min=0.0, u_max=1.0)
            
            error = setpoints - y_out
            iae = np.trapezoid(np.abs(error), t_eval)
            
            # --- UPDATED METRIC EXPORT: Total Variation ---
            u_with_initial = np.concatenate(([0.0], u_out))
            u_aggressiveness = np.sum(np.abs(np.diff(u_with_initial)))
            
            rt = calculate_rise_time(t_eval, y_out, setpoints, seq_config['base_sp'], seq_config['step_sp'])
            os_pct = calculate_overshoot(y_out, setpoints, seq_config['base_sp'], seq_config['step_sp'])

            tuner_key = cfg["name"]
            aggregated_metrics['IAE'][tf_name][tuner_key] = iae
            aggregated_metrics['Control_Effort'][tf_name][tuner_key] = u_aggressiveness
            aggregated_metrics['Rise_Time'][tf_name][tuner_key] = rt
            aggregated_metrics['Overshoot'][tf_name][tuner_key] = os_pct

            ax_y.plot(t_eval_hours, y_out, color=cfg["color"], label=f'{cfg["name"]} (IAE: {iae:.0f})')
            ax_u.plot(t_eval_hours, u_out, color=cfg["color"], label=f'{cfg["name"]} (TV Effort: {u_aggressiveness:.4f})')

        ax_y.set_title(f'Step Response Comparison', fontsize=22, pad=20)
        ax_y.set_xlabel('Time (hours)', fontsize=18)
        ax_y.set_ylabel('System Output', fontsize=18)
        ax_y.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=16, borderaxespad=0.)
        ax_y.grid(True, alpha=0.3)
        fig_y.tight_layout()
        fig_y.savefig(os.path.join(output_dir, f"step_response_{tf_name}.png"))
        plt.close(fig_y)
        
        ax_u.set_title(f'Control Effort Comparison', fontsize=22, pad=20)
        ax_u.set_xlabel('Time (hours)', fontsize=18)
        ax_u.set_ylabel('Control Signal (u)', fontsize=18)
        ax_u.set_ylim(-0.1, 1.1) 
        ax_u.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=16, borderaxespad=0.)
        ax_u.grid(True, alpha=0.3)
        fig_u.tight_layout()
        fig_u.savefig(os.path.join(output_dir, f"control_effort_{tf_name}.png"))
        plt.close(fig_u)

    print("\n--- Exporting Formatted Metric Tables ---")
    write_formatted_table(os.path.join(output_dir, "IAE_table.csv"), 'IAE', tf_list, aggregated_metrics['IAE'])
    write_formatted_table(os.path.join(output_dir, "Control_Effort_table.csv"), 'Control_Effort', tf_list, aggregated_metrics['Control_Effort'])
    write_formatted_table(os.path.join(output_dir, "Rise_Time_table.csv"), 'Rise_Time', tf_list, aggregated_metrics['Rise_Time'])
    write_formatted_table(os.path.join(output_dir, "Overshoot_table.csv"), 'Overshoot', tf_list, aggregated_metrics['Overshoot'])
    
    print(f"Simulation Complete. Processed {len(tf_list)} transfer functions.")

if __name__ == "__main__":
    main()