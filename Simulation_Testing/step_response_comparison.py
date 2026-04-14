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
# 2. METRIC HELPERS (For the Graph Output Tables)
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
# 3. SCIPY DE TUNER WITH NATIVE COST FUNCTION
# ==========================================

def run_scipy_de_tuner(name, plant, min_kp, max_kp, min_ki, max_ki, tau, delay, weights):
    """Uses SciPy's DE applying the exact cost function logic from ea_optimizer.py"""
    print(f"\n[Auto-Tuner] Running SciPy DE Optimization: {name}")
    
    # Match T_plant * 3 + delay simulation boundary from ea_optimizer.py
    T_sim = (tau * 3) + delay
    t_opt = np.linspace(0, T_sim, 1000)
    sp_opt = np.ones_like(t_opt) # Step from 0 to 1
    
    avg_rise_time = tau * 2.2
    w_iae, w_effort, w_os, w_rt = weights

    def objective(params):
        kp, ki = params
        controller = create_pi_controller(kp, ki)
        
        # Closed-loop systems
        T_y = ct.feedback(ct.series(controller, plant), 1)
        T_u = ct.feedback(controller, plant)
        
        penalty = 1e9
        
        try:
            # Simulate
            _, y_out = ct.forced_response(T_y, t_opt, sp_opt)
            _, u_out = ct.forced_response(T_u, t_opt, sp_opt)
            
            if np.any(np.isnan(y_out)) or np.any(np.isinf(y_out)):
                return penalty
                
            # --- EA OPTIMIZER COST FUNCTION REPLICATION ---
            error = 1.0 - y_out
            int_error = np.trapezoid(np.abs(error), t_opt)
            
            # Note u^2 from ea_optimizer.py 
            u_delayed = np.clip(u_out, -1.0, 1.0)
            int_control = np.trapezoid(u_delayed**2, t_opt)
            
            # Rise time calculation exactly mirroring EA
            crossings_10 = np.where(y_out >= 0.1)[0]
            crossings_90 = np.where(y_out >= 0.9)[0]
            
            if len(crossings_10) > 0 and len(crossings_90) > 0:
                rise_time = t_opt[crossings_90[0]] - t_opt[crossings_10[0]]
            else:
                rise_time = T_sim * 10
                
            # Normalization blocks
            norm_error = int_error / T_sim
            norm_effort = int_control / T_sim
            peak_y = np.max(y_out)
            norm_overshoot = max(0.0, peak_y - 1.0) / 0.5
            norm_rise_time = rise_time / avg_rise_time
            
            # Limit checks
            if norm_error > 1.0 or norm_effort > 1.0 or norm_overshoot > 1.0 or norm_rise_time > 1.0:
                return penalty
                
            if np.max(y_out) > 1.3 or np.min(y_out) < -0.1:
                return penalty
                
            cost = (w_iae * norm_error) + (w_effort * norm_effort) + (w_os * norm_overshoot) + (w_rt * norm_rise_time)
            return cost
            
        except Exception:
            return penalty

    bounds = [(min_kp, max_kp), (min_ki, max_ki)]
    
    # SciPy Differential Evolution
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
# 4. UTILITIES
# ==========================================

# ==========================================
# 4. UTILITIES
# ==========================================

def safe_float(val):
    """Safely converts string to float, handling fancy minus signs and empty cells."""
    if not val or str(val).strip() == '':
        return 0.0
    # Replace en-dashes and unicode minus signs with standard ASCII hyphen-minus
    cleaned_val = str(val).replace('−', '-').replace('–', '-').replace(' ', '').strip()
    try:
        return float(cleaned_val)
    except ValueError:
        print(f"[Warning] Could not parse '{val}' to float. Defaulting to 0.0")
        return 0.0

def read_tf_parameters(filepath):
    tfs = []
    with open(filepath, 'r', encoding='utf-8-sig') as f: # utf-8-sig handles potential BOM characters
        reader = csv.reader(f)
        next(reader) # Skip main headers
        next(reader) # Skip sub headers
        for row in reader:
            if not row or not row[0].strip():
                continue
            
            # Ensure row has enough columns to prevent index out of bounds
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
    
    tuners = [
        'DE-tuned', 
        'Lambda-tuned ($\\lambda$ = 3$\\tau$)', 
        'MATLAB pidtune(), balanced'
    ]
    
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
# 5. MAIN EXECUTION
# ==========================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(base_dir, "tf_parameters.csv")
    output_dir = os.path.join(base_dir, "simulation_graphs_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    aggregated_metrics = {
        'IAE': {}, 'Control_Effort': {}, 'Rise_Time': {}, 'Overshoot': {}
    }

    tf_list = read_tf_parameters(input_csv)
    
    # Match the default weights from ea_optimizer.py
    de_weights = [1.0, 1.0, 1.0, 1.0]

    for tf_data in tf_list:
        tf_name = tf_data['name']
        print(f"\n--- Processing Transfer Function: {tf_name} ---")
        
        for key in aggregated_metrics:
            aggregated_metrics[key][tf_name] = {}
        
        plant_params = {'K': tf_data['K'], 'tau': tf_data['tau'], 'delay': tf_data['delay']} 
        plant = create_fopdt_sys(**plant_params)
        
        # Simulation duration for final visualization graphs
        time_factor = max(plant_params['tau'], 1000)
        seq_config = {
            'base_sp': 0.0, 
            'step_sp': 1.0,
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

        # Set search bounds based on process gain direction
        if plant_params['K'] < 0:
            max_kp = 0 
            min_kp = -1
            max_ki = 0
            min_ki = -0.0005 
        else:
            max_kp = 1.5
            min_kp = 0
            max_ki = 0.005
            min_ki = 0

        # 1. Run Scipy DE Auto-tuner using Native Cost Logic
        de_kp, de_ki = run_scipy_de_tuner(
            f"DE ({tf_name})", plant,
            min_kp, max_kp, min_ki, max_ki,
            plant_params['tau'], plant_params['delay'], de_weights
        )

        configs = [
            {"name": "DE-tuned", "color": "red", "kp": de_kp, "ki": de_ki},
            {"name": "Lambda-tuned ($\\lambda$ = 3$\\tau$)", "color": "green", "kp": tf_data['lambda_kp'], "ki": tf_data['lambda_ki']},
            {"name": "MATLAB pidtune(), balanced", "color": "blue", "kp": tf_data['matlab_kp'], "ki": tf_data['matlab_ki']}
        ]

        plt.figure(figsize=(14, 8))
        plt.plot(t_eval_hours, setpoints, 'k--', label='Reference Setpoint', alpha=0.6)

        for cfg in configs:
            controller = create_pi_controller(cfg["kp"], cfg["ki"])
            
            T_y = ct.feedback(ct.series(controller, plant), 1)
            T_u = ct.feedback(controller, plant)

            _, y_out = ct.forced_response(T_y, t_eval, setpoints)
            _, u_out = ct.forced_response(T_u, t_eval, setpoints)
            
            error = setpoints - y_out
            iae = np.trapezoid(np.abs(error), t_eval)
            u_auc = np.trapezoid(np.clip(u_out, -1, 1)**2, t_eval) 
            rt = calculate_rise_time(t_eval, y_out, setpoints, seq_config['base_sp'], seq_config['step_sp'])
            os_pct = calculate_overshoot(y_out, setpoints, seq_config['base_sp'], seq_config['step_sp'])

            tuner_key = cfg["name"]
            aggregated_metrics['IAE'][tf_name][tuner_key] = iae
            aggregated_metrics['Control_Effort'][tf_name][tuner_key] = u_auc
            aggregated_metrics['Rise_Time'][tf_name][tuner_key] = rt
            aggregated_metrics['Overshoot'][tf_name][tuner_key] = os_pct

            plt.plot(t_eval_hours, y_out, color=cfg["color"], label=f'{cfg["name"]} (IAE: {iae:.0f})')
            
        plt.title(f'Step Response Comparison: {tf_name}', fontsize=22, pad=20)
        plt.xlabel('Time (hours)', fontsize=18)
        plt.ylabel('System Output', fontsize=18)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=16, borderaxespad=0.)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f"step_response_{tf_name}.png")
        plt.savefig(plot_path)
        plt.close() 

    print("\n--- Exporting Formatted Metric Tables ---")
    write_formatted_table(os.path.join(output_dir, "IAE_table.csv"), 'IAE', tf_list, aggregated_metrics['IAE'])
    write_formatted_table(os.path.join(output_dir, "Control_Effort_table.csv"), 'Control_Effort', tf_list, aggregated_metrics['Control_Effort'])
    write_formatted_table(os.path.join(output_dir, "Rise_Time_table.csv"), 'Rise_Time', tf_list, aggregated_metrics['Rise_Time'])
    write_formatted_table(os.path.join(output_dir, "Overshoot_table.csv"), 'Overshoot', tf_list, aggregated_metrics['Overshoot'])
    
    print(f"Simulation Complete. Processed {len(tf_list)} transfer functions.")

if __name__ == "__main__":
    main()