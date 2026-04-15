import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import control as ct

from scipy_de_tuner import run_scipy_de_tuner, simulate_saturated_pi

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
# 3. UTILITIES
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
                'delay': safe_float(row[3])
                # Removed MATLAB and Lambda reading logic as they are no longer used
            }
            tfs.append(tf_data)
    return tfs

def write_tradeoff_table(output_path, tf_name, sweep_results):
    """Writes the sweep metrics to a CSV for a specific transfer function."""
    headers = ['ce_weight', 'Kp', 'Ki', 'IAE', 'Total_Variation', 'Rise_Time', 'Overshoot']
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for res in sweep_results:
            writer.writerow([
                f"{res['ce']:.2f}",
                f"{res['kp']:.6f}",
                f"{res['ki']:.6f}",
                f"{res['iae']:.2f}",
                f"{res['tv']:.4f}",
                f"{res['rt']:.2f}",
                f"{res['os']:.2f}"
            ])

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(base_dir, "tf_parameters_do.csv")
    output_dir = os.path.join(base_dir, "simulation_graphs_ce_sweep")
    os.makedirs(output_dir, exist_ok=True)
    
    tf_list = read_tf_parameters(input_csv)
    
    # ------------------------------------------
    # CONFIGURATION: Parameter Sweep Settings
    # ------------------------------------------
    start_ce = 1.0      # Minimum Control Effort Weight
    end_ce = 2.0        # Maximum Control Effort Weight
    num_bins = 11        # Number of evaluations to sweep
    # ------------------------------------------

    fontsize = 18
    
    ce_weights_to_test = np.linspace(start_ce, end_ce, num_bins)
    colors = cm.viridis(np.linspace(0, 0.9, num_bins)) # Colormap for visualization

    for tf_data in tf_list:
        tf_name = tf_data['name']
        print(f"\n--- Processing Transfer Function: {tf_name} ---")
        
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

        sweep_results = []

        # 1. RUN THE PARAMETER SWEEP
        for idx, ce in enumerate(ce_weights_to_test):
            perf = (4.0 - ce) / 3.0
            de_weights = [perf, ce, perf, perf]
            
            print(f"  Tuning for ce_weight = {ce:.2f}...")
            de_kp, de_ki = run_scipy_de_tuner(
                f"DE (ce={ce:.2f})", plant,
                min_kp, max_kp, min_ki, max_ki,
                plant_params['tau'], plant_params['delay'], de_weights,
                target_sp=target_sp
            )
            
            y_out, u_out = simulate_saturated_pi(plant, de_kp, de_ki, t_eval, setpoints, u_min=0.0, u_max=1.0)
            
            error = setpoints - y_out
            iae = np.trapezoid(np.abs(error), t_eval)
            
            u_with_initial = np.concatenate(([0.0], u_out))
            tv_effort = np.sum(np.abs(np.diff(u_with_initial)))
            
            rt = calculate_rise_time(t_eval, y_out, setpoints, seq_config['base_sp'], seq_config['step_sp'])
            os_pct = calculate_overshoot(y_out, setpoints, seq_config['base_sp'], seq_config['step_sp'])
            
            sweep_results.append({
                'ce': ce, 'kp': de_kp, 'ki': de_ki,
                'iae': iae, 'tv': tv_effort, 'rt': rt, 'os': os_pct,
                'y': y_out, 'u': u_out, 'color': colors[idx]
            })

        # 2. GENERATE PLOTS
        # Plot A: Step Response Overlay
        fig_y, ax_y = plt.subplots(figsize=(12, 6))
        ax_y.plot(t_eval_hours, setpoints, 'k--', label='Setpoint', alpha=0.6)
        for res in sweep_results:
            ax_y.plot(t_eval_hours, res['y'], color=res['color'], label=f"ce={res['ce']:.2f}")
            
        ax_y.set_title(f'Step Response Sweep ({tf_name})', fontsize=fontsize)
        ax_y.set_xlabel('Time (hours)', fontsize=fontsize)
        ax_y.set_ylabel('System Output', fontsize=fontsize)
        ax_y.legend(loc='lower right', fontsize=fontsize-2)
        ax_y.grid(True, alpha=0.3)
        fig_y.tight_layout()
        fig_y.savefig(os.path.join(output_dir, f"step_response_sweep_{tf_name}.png"))
        plt.close(fig_y)
        
        # Plot B: Control Effort Overlay
        fig_u, ax_u = plt.subplots(figsize=(12, 6))
        for res in sweep_results:
            ax_u.plot(t_eval_hours, res['u'], color=res['color'], label=f"ce={res['ce']:.2f}")
            
        ax_u.set_title(f'Control Effort Sweep ({tf_name})', fontsize=fontsize)
        ax_u.set_xlabel('Time (hours)', fontsize=fontsize)
        ax_u.set_ylabel('Control Signal (u)', fontsize=fontsize)
        ax_u.set_ylim(-0.1, 1.1) 
        ax_u.legend(loc='upper right', fontsize=fontsize)
        ax_u.grid(True, alpha=0.3)
        fig_u.tight_layout()
        fig_u.savefig(os.path.join(output_dir, f"control_effort_sweep_{tf_name}.png"))
        plt.close(fig_u)

        # Plot C: Pareto Trade-off Curve (IAE vs Total Variation)
        fig_p, ax_p = plt.subplots(figsize=(10, 8))
        iaes = [r['iae'] for r in sweep_results]
        tvs = [r['tv'] for r in sweep_results]
        ces = [r['ce'] for r in sweep_results]
        
        ax_p.plot(tvs, iaes, 'k-', alpha=0.3, zorder=1) # Connecting line
        scatter = ax_p.scatter(tvs, iaes, c=ces, cmap='viridis', s=100, zorder=2)
        
        for i, res in enumerate(sweep_results):
            ax_p.annotate(f"ce={res['ce']:.2f}", (tvs[i], iaes[i]), textcoords="offset points", xytext=(10,5), ha='left')

        ax_p.set_title(f'Trade-off: IAE vs Control Effort ({tf_name})', fontsize=fontsize)
        ax_p.set_xlabel('Total Variation (Control Effort)', fontsize=fontsize)
        ax_p.set_ylabel('Integral Absolute Error (IAE)', fontsize=fontsize)
        ax_p.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax_p)
        cbar.set_label('Control Effort Weight (CEW)', rotation=270, labelpad=15)
        
        fig_p.tight_layout()
        fig_p.savefig(os.path.join(output_dir, f"tradeoff_curve_{tf_name}.png"))
        plt.close(fig_p)

        # 3. EXPORT METRICS TO CSV
        csv_path = os.path.join(output_dir, f"metrics_sweep_{tf_name}.csv")
        write_tradeoff_table(csv_path, tf_name, sweep_results)

    print(f"\nSimulation Complete. Check the '{os.path.basename(output_dir)}' folder for outputs.")

if __name__ == "__main__":
    main()