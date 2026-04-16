import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import control as ct
import warnings

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

# ==========================================
# 2. METRIC HELPERS
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

def analyze_root_locus_metrics(L_nom):
    """
    Analyzes the nominal open-loop transfer function to find the scale factors (k)
    for marginal stability and the 50% damping ratio (zeta = 0.5).
    """
    k_vect = np.logspace(-5, 6, 8000)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        roots, gains = ct.root_locus(L_nom, kvect=k_vect, plot=False)

    k_marg = None
    k_zeta_05 = None

    # 1. Marginal Stability (Imaginary axis crossing)
    for i, k in enumerate(gains):
        if np.any(np.real(roots[i]) >= 1e-5):
            if i > 0:
                prev_max = np.max(np.real(roots[i-1]))
                curr_max = np.max(np.real(roots[i]))
                if curr_max - prev_max > 0:
                    frac = (0 - prev_max) / (curr_max - prev_max)
                    k_marg = gains[i-1] + frac * (gains[i] - gains[i-1])
                else:
                    k_marg = k
            else:
                k_marg = k
            break

    # 2. Damping Ratio of 50% (zeta = 0.5)
    for branch_idx in range(roots.shape[1]):
        branch_roots = roots[:, branch_idx]
        valid_mask = np.abs(np.imag(branch_roots)) > 1e-5 
        
        zetas = np.zeros_like(branch_roots, dtype=float)
        with np.errstate(invalid='ignore', divide='ignore'):
            zetas[valid_mask] = -np.real(branch_roots[valid_mask]) / np.abs(branch_roots[valid_mask])
        
        for i in range(len(zetas) - 1):
            if valid_mask[i] and valid_mask[i+1]:
                z1, z2 = zetas[i], zetas[i+1]
                if (z1 - 0.5) * (z2 - 0.5) <= 0:
                    w1 = abs(z2 - 0.5)
                    w2 = abs(z1 - 0.5)
                    if w1 + w2 > 0:
                        k_interp = (gains[i]*w1 + gains[i+1]*w2) / (w1 + w2)
                        if k_zeta_05 is None or k_interp < k_zeta_05:
                            k_zeta_05 = k_interp

    return roots, gains, k_marg, k_zeta_05

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
                row.append(f"{val:.4f}" if isinstance(val, float) else val)
            writer.writerow(row)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(base_dir, "tf_parameters_do.csv")
    output_dir = os.path.join(base_dir, "simulation_graphs_comparison_do_1xcf_root_locus")
    os.makedirs(output_dir, exist_ok=True)
    
    aggregated_metrics = {
        'IAE': {}, 'Control_Effort': {}, 'Rise_Time': {}, 'Overshoot': {},
        'RL_Marginal_Gain': {}, 'RL_Zeta05_Gain': {}
    }
    
    tf_list = read_tf_parameters(input_csv)
    
    cf_weight = 1.0
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
        fig_rl, axes_rl = plt.subplots(1, 3, figsize=(20, 6.5))
        fig_rl.suptitle(f'Root Locus Analysis - {tf_name}', fontsize=20, y=0.98)

        ax_y.plot(t_eval_hours, setpoints, 'k--', label='Reference Setpoint', alpha=0.6)

        all_rl_reals = []
        all_rl_imags = []
        all_roots_arrays = []

        for idx, cfg in enumerate(configs):
            tuner_key = cfg["name"]
            
            # ---------------------------
            # 1. Step Response Simulation
            # ---------------------------
            y_out, u_out = simulate_saturated_pi(plant, cfg["kp"], cfg["ki"], t_eval, setpoints, u_min=0.0, u_max=1.0)
            
            error = setpoints - y_out
            iae = np.trapezoid(np.abs(error), t_eval)
            
            u_with_initial = np.concatenate(([0.0], u_out))
            u_aggressiveness = np.sum(np.abs(np.diff(u_with_initial)))
            
            rt = calculate_rise_time(t_eval, y_out, setpoints, seq_config['base_sp'], seq_config['step_sp'])
            os_pct = calculate_overshoot(y_out, setpoints, seq_config['base_sp'], seq_config['step_sp'])

            aggregated_metrics['IAE'][tf_name][tuner_key] = iae
            aggregated_metrics['Control_Effort'][tf_name][tuner_key] = u_aggressiveness
            aggregated_metrics['Rise_Time'][tf_name][tuner_key] = rt
            aggregated_metrics['Overshoot'][tf_name][tuner_key] = os_pct

            ax_y.plot(t_eval_hours, y_out, color=cfg["color"], label=f'{cfg["name"]} (IAE: {iae:.0f})')
            ax_u.plot(t_eval_hours, u_out, color=cfg["color"], label=f'{cfg["name"]} (TV Effort: {u_aggressiveness:.4f})')
            
            # ---------------------------
            # 2. Root Locus Computation
            # ---------------------------
            ax_rl = axes_rl[idx]
            Kp_nom = cfg['kp']
            Ki_nom = cfg['ki']
            
            if abs(Kp_nom) < 1e-6 and abs(Ki_nom) < 1e-6:
                ax_rl.set_title(f"{cfg['name']}\n(No valid tuning)")
                aggregated_metrics['RL_Marginal_Gain'][tf_name][tuner_key] = 0.0
                aggregated_metrics['RL_Zeta05_Gain'][tf_name][tuner_key] = 0.0
                continue
                
            C_nom = ct.tf([Kp_nom, Ki_nom], [1, 0])
            L_nom = ct.series(C_nom, plant)
            
            roots, gains, k_marg, k_zeta_05 = analyze_root_locus_metrics(L_nom)
            all_roots_arrays.append(roots) # Save to extract y-axis bounds later
            
            if abs(Kp_nom) < 1e-6: 
                marg_val = k_marg * Ki_nom if k_marg is not None else 0.0
                zeta_val = k_zeta_05 * Ki_nom if k_zeta_05 is not None else 0.0
                gain_str = "Ki"
            else:
                marg_val = k_marg * Kp_nom if k_marg is not None else 0.0
                zeta_val = k_zeta_05 * Kp_nom if k_zeta_05 is not None else 0.0
                gain_str = "Kp"
                
            aggregated_metrics['RL_Marginal_Gain'][tf_name][tuner_key] = marg_val
            aggregated_metrics['RL_Zeta05_Gain'][tf_name][tuner_key] = zeta_val
            
            for i in range(roots.shape[1]):
                ax_rl.plot(np.real(roots[:, i]), np.imag(roots[:, i]), color=cfg['color'], alpha=0.6)
                
            ol_poles = ct.poles(L_nom)
            ol_zeros = ct.zeros(L_nom)
            cl_poles = ct.poles(ct.feedback(L_nom))
            
            for pts in [ol_poles, ol_zeros, cl_poles]:
                if len(pts) > 0:
                    all_rl_reals.extend(np.real(pts))
                    all_rl_imags.extend(np.imag(pts))

            ax_rl.plot(np.real(cl_poles), np.imag(cl_poles), 'kx', 
                       markersize=10, markeredgewidth=2, 
                       label=f'Nominal Poles ({gain_str}={Kp_nom if gain_str=="Kp" else Ki_nom:.3f})')
            
            title_str = f"{cfg['name']}\n"
            title_str += f"Marginal {gain_str} ≈ {marg_val:.3f}\n" if marg_val else f"Marginal {gain_str}: N/A\n"
            title_str += f"{gain_str} at $\\zeta=0.5$ ≈ {zeta_val:.3f}" if zeta_val else f"{gain_str} at $\\zeta=0.5$: N/A"
            
            ax_rl.set_title(title_str, fontsize=12)
            ax_rl.axhline(0, color='black', lw=0.5, ls='--')
            ax_rl.axvline(0, color='black', lw=0.5, ls='--')
            
            ax_rl.grid(True, alpha=0.3)
            ax_rl.set_xlabel('Real Axis')
            if idx == 0:
                ax_rl.set_ylabel('Imaginary Axis')

        # ---------------------------
        # 3. Apply Universal RL Axes (X and Y)
        # ---------------------------
        if all_rl_reals:
            # Universalize X-Axis based on poles/zeros
            min_x = min(all_rl_reals)
            max_x = max(all_rl_reals)
            x_pad = (max_x - min_x) * 0.15
            if x_pad < 1e-4: x_pad = 0.1  
            
            global_xlim = [min_x - x_pad, max(max_x + x_pad, 0.05)] 
            
            # Universalize Y-Axis by finding the highest locus branch *strictly within our X-bounds*
            max_y = 0.0
            for roots_array in all_roots_arrays:
                # Mask roots that fall outside our visual X window to avoid infinity scaling
                valid_mask = (np.real(roots_array) >= global_xlim[0]) & (np.real(roots_array) <= global_xlim[1])
                if np.any(valid_mask):
                    local_max = np.max(np.abs(np.imag(roots_array[valid_mask])))
                    if local_max > max_y:
                        max_y = local_max
                        
            # Ensure it also encompasses any stray complex poles/zeros if they exist
            if all_rl_imags:
                max_y = max(max_y, max(np.abs(all_rl_imags)))
            
            y_pad = max_y * 0.15
            if max_y < 1e-4: 
                y_pad = (global_xlim[1] - global_xlim[0]) * 0.5 

            global_ylim = [-(max_y + y_pad), (max_y + y_pad)]
            
            for ax_rl in axes_rl:
                ax_rl.set_xlim(global_xlim)
                ax_rl.set_ylim(global_ylim)
                
                # Draw constant zeta=0.5 boundaries spanning to the new bounding box limits
                x_damp = np.linspace(global_xlim[0], 0, 100)
                y_damp = np.sqrt(3) * np.abs(x_damp)
                ax_rl.plot(x_damp, y_damp, 'k:', alpha=0.4, label='$\\zeta=0.5$ boundary')
                ax_rl.plot(x_damp, -y_damp, 'k:', alpha=0.4)
                
                # Remove duplicate legend entries
                handles, labels = ax_rl.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax_rl.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=9)

        # Save Step Response figure
        ax_y.set_title(f'Step Response Comparison', fontsize=22, pad=20)
        ax_y.set_xlabel('Time (hours)', fontsize=18)
        ax_y.set_ylabel('System Output', fontsize=18)
        ax_y.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=16, borderaxespad=0.)
        ax_y.grid(True, alpha=0.3)
        fig_y.tight_layout()
        fig_y.savefig(os.path.join(output_dir, f"step_response_{tf_name}.png"))
        plt.close(fig_y)
        
        # Save Control Effort figure
        ax_u.set_title(f'Control Effort Comparison', fontsize=22, pad=20)
        ax_u.set_xlabel('Time (hours)', fontsize=18)
        ax_u.set_ylabel('Control Signal (u)', fontsize=18)
        ax_u.set_ylim(-0.1, 1.1) 
        ax_u.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=16, borderaxespad=0.)
        ax_u.grid(True, alpha=0.3)
        fig_u.tight_layout()
        fig_u.savefig(os.path.join(output_dir, f"control_effort_{tf_name}.png"))
        plt.close(fig_u)
        
        # Save Root Locus figure
        fig_rl.tight_layout()
        fig_rl.subplots_adjust(top=0.82) 
        fig_rl.savefig(os.path.join(output_dir, f"root_locus_{tf_name}.png"))
        plt.close(fig_rl)

    print("\n--- Exporting Formatted Metric Tables ---")
    write_formatted_table(os.path.join(output_dir, "IAE_table.csv"), 'IAE', tf_list, aggregated_metrics['IAE'])
    write_formatted_table(os.path.join(output_dir, "Control_Effort_table.csv"), 'Control_Effort', tf_list, aggregated_metrics['Control_Effort'])
    write_formatted_table(os.path.join(output_dir, "Rise_Time_table.csv"), 'Rise_Time', tf_list, aggregated_metrics['Rise_Time'])
    write_formatted_table(os.path.join(output_dir, "Overshoot_table.csv"), 'Overshoot', tf_list, aggregated_metrics['Overshoot'])
    write_formatted_table(os.path.join(output_dir, "RL_Marginal_Gain_table.csv"), 'RL_Marginal_Gain', tf_list, aggregated_metrics['RL_Marginal_Gain'])
    write_formatted_table(os.path.join(output_dir, "RL_Zeta05_Gain_table.csv"), 'RL_Zeta05_Gain', tf_list, aggregated_metrics['RL_Zeta05_Gain'])
    
    print(f"Simulation Complete. Processed {len(tf_list)} transfer functions.")

if __name__ == "__main__":
    main()