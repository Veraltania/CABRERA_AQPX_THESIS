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
            }
            tfs.append(tf_data)
    return tfs

def write_tradeoff_table(output_path, tf_name, sweep_results):
    """Writes the sweep metrics to a CSV for a specific transfer function."""
    headers = ['CE_Weight_Pct', 'Kp', 'Ki', 'IAE', 'Total_Variation', 'Rise_Time', 'Overshoot']
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for res in sweep_results:
            writer.writerow([
                f"{res['ce_pct']:.1f}%",
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
    output_dir = os.path.join(base_dir, "simulation_graphs_ce_sweep_do_wide_range")
    os.makedirs(output_dir, exist_ok=True)
    
    tf_list = read_tf_parameters(input_csv)
    
    # ------------------------------------------
    # CONFIGURATION: Parameter Sweep Settings
    # ------------------------------------------
    start_ce = 0.25      # Minimum Control Effort Weight
    end_ce = 3.75        # Maximum Control Effort Weight
    num_bins = 11        # Number of evaluations to sweep
    # ------------------------------------------

    fontsize = 16
    
    ce_weights_to_test = np.linspace(start_ce, end_ce, num_bins)
    
    # Create normalization and colormap mapping specifically for the continuous colorbars
    ce_pcts_global = [(ce / 4.0) * 100.0 for ce in ce_weights_to_test]
    colormap = cm.viridis
    norm = plt.Normalize(vmin=min(ce_pcts_global), vmax=max(ce_pcts_global))
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])

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
            ce_pct = (ce / 4.0) * 100.0 
            perf = (4.0 - ce) / 3.0
            de_weights = [perf, ce, perf, perf]
            
            print(f"  Tuning for CE Weight = {ce_pct:.1f}%...")
            de_kp, de_ki = run_scipy_de_tuner(
                f"DE (ce={ce_pct:.1f}%)", plant,
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
                'ce': ce, 'ce_pct': ce_pct, 'kp': de_kp, 'ki': de_ki,
                'iae': iae, 'tv': tv_effort, 'rt': rt, 'os': os_pct,
                'y': y_out, 'u': u_out, 'color': colormap(norm(ce_pct)) # Map color perfectly to CE%
            })

        # 2. GENERATE TIME SERIES PLOTS
        # Plot A: Step Response Overlay
        fig_y, ax_y = plt.subplots(figsize=(12, 6))
        ax_y.plot(t_eval_hours, setpoints, 'k--', label='Setpoint', alpha=0.6)
        for res in sweep_results:
            ax_y.plot(t_eval_hours, res['y'], color=res['color']) # Removed discrete label
            
        ax_y.set_title(f'Step Response Sweep', fontsize=fontsize)
        ax_y.set_xlabel('Time (hours)', fontsize=fontsize)
        ax_y.set_ylabel('System Output', fontsize=fontsize)
        ax_y.grid(True, alpha=0.3)
        
        # Add colorbar instead of legend
        cbar_y = fig_y.colorbar(sm, ax=ax_y)
        cbar_y.set_label('Control Effort Weight (%)', rotation=270, labelpad=20, fontsize=fontsize-2)
        
        fig_y.tight_layout()
        fig_y.savefig(os.path.join(output_dir, f"step_response_sweep_{tf_name}.png"))
        plt.close(fig_y)
        
        # Plot B: Control Effort Overlay
        fig_u, ax_u = plt.subplots(figsize=(12, 6))
        for res in sweep_results:
            ax_u.plot(t_eval_hours, res['u'], color=res['color']) # Removed discrete label
            
        ax_u.set_title(f'Control Effort Sweep', fontsize=fontsize)
        ax_u.set_xlabel('Time (hours)', fontsize=fontsize)
        ax_u.set_ylabel('Control Signal (u)', fontsize=fontsize)
        ax_u.set_ylim(-0.1, 1.1) 
        ax_u.grid(True, alpha=0.3)
        
        # Add colorbar instead of legend
        cbar_u = fig_u.colorbar(sm, ax=ax_u)
        cbar_u.set_label('Control Effort Weight (%)', rotation=270, labelpad=20, fontsize=fontsize-2)

        fig_u.tight_layout()
        fig_u.savefig(os.path.join(output_dir, f"control_effort_sweep_{tf_name}.png"))
        plt.close(fig_u)

        # 3. GENERATE TRADEOFF PLOTS
        iaes = [r['iae'] for r in sweep_results]
        rts = [r['rt'] for r in sweep_results]
        oss = [r['os'] for r in sweep_results]
        ce_pcts = [r['ce_pct'] for r in sweep_results]
        
        # Define the three tradeoffs to plot
        tradeoff_configs = [
            ('iae', iaes, 'Integral Absolute Error (IAE)', 'IAE'),
            ('rt', rts, 'Rise Time (hours)', 'Rise Time'),
            ('os', oss, 'Overshoot (%)', 'Overshoot')
        ]
        
        for metric_key, y_vals, ylabel, title_name in tradeoff_configs:
            fig_p, ax_p = plt.subplots(figsize=(10, 8))
            
            # Sort for clean connecting lines (Using ce_pcts instead of tvs_pct)
            sort_indices = np.argsort(ce_pcts)
            x_sorted = np.array(ce_pcts)[sort_indices]
            y_sorted = np.array(y_vals)[sort_indices]
            
            # Connecting line
            ax_p.plot(x_sorted, y_sorted, 'k-', alpha=0.3, zorder=1) 
            
            # Linear Fit (Using ce_pcts instead of tvs_pct)
            z = np.polyfit(ce_pcts, y_vals, 1)
            p = np.poly1d(z)

            # Scatter points (Using ce_pcts for x-axis to match labels and colorbars!)
            scatter = ax_p.scatter(ce_pcts, y_vals, c=ce_pcts, cmap=colormap, norm=norm, s=100, zorder=3)
            
            ax_p.set_title(f'Trade-off: {title_name} vs %CE', fontsize=fontsize)
            ax_p.set_xlabel('Control Effort Weight (%)', fontsize=fontsize)
            ax_p.set_ylabel(ylabel, fontsize=fontsize)
            
            # Ensure X-axis always starts at 0
            ax_p.set_xlim(left=0) 
            ax_p.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(scatter, ax=ax_p)
            cbar.set_label('Control Effort Weight (%)', rotation=270, labelpad=20)
            
            fig_p.tight_layout()
            fig_p.savefig(os.path.join(output_dir, f"tradeoff_{metric_key}_{tf_name}.png"))
            plt.close(fig_p)

        # 4. EXPORT METRICS TO CSV
        csv_path = os.path.join(output_dir, f"metrics_sweep_{tf_name}.csv")
        write_tradeoff_table(csv_path, tf_name, sweep_results)

    print(f"\nSimulation Complete. Check the '{os.path.basename(output_dir)}' folder for outputs.")

if __name__ == "__main__":
    main()