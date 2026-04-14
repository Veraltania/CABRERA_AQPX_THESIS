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
        # Pade approximation creates a linear representation of the dead time
        num_delay, den_delay = ct.pade(delay, pade_order)
        delay_tf = ct.tf(num_delay, den_delay)
        return ct.series(delay_tf, plant_linear)
    return plant_linear

def create_pi_controller(kp, ki):
    """Creates a PI Controller Transfer Function: (kp*s + ki) / s"""
    return ct.tf([kp, ki], [1, 0])

def generate_setpoint_array(t, sequence_config):
    """Generates the reference signal array for the simulation time vector."""
    # This replaces the logic in StepSequence for the control library
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
# 2. AUTO-TUNING & METRIC HELPERS
# ==========================================
def run_de_tuner(name, tf_config, weights, min_kp=0.001, max_kp=20.0, min_ki=1e-6, max_ki=0.05):
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
# 3. MAIN EXECUTION
# ==========================================
def main():
    # =======================================================
    # PLOT CONFIGURATION BLOCK
    # =======================================================
    plot_font_size = 22

    # Step Response Labels
    step_title = 'Step Response Comparison'
    step_x_label = 'Time (hours)'
    step_y_label = 'DO (mg/l)'

    # Control Effort Labels
    effort_title = 'Control Effort Comparison'
    effort_x_label = 'Time (hours)'
    effort_y_label = 'Duty Cycle'
    # =======================================================

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation_graphs_control_lib")
    os.makedirs(output_dir, exist_ok=True)

    # 1. System Definition
    plant_params = {'K': 1.3460, 'tau': 1551.955, 'delay': 104.469} 
    plant = create_fopdt_sys(**plant_params)
    
    # 2. Tuner Configurations
    configs = [
        {"name": "MATLAB Tuning", 
         "type": "manual", 
         "color": "blue", 
         "kp": 0, 
         "ki": 0},
        {"name": "Lambda Tuning", 
         "type": "manual", 
         "color": "green"
         "kp": 0.242213, 
         "ki": 0.000551, 
         }
    ]

    seq_config = {
        'base_sp': 1.0, 'step_sp': 2.0,
        'pre_step_delay': 10000, 'step_duration': 10000,
        'recovery_duration': 10000, 'cycles': 2
    }
    
    total_time = seq_config['pre_step_delay'] + (seq_config['step_duration'] + seq_config['recovery_duration']) * seq_config['cycles']
    t_eval = np.linspace(0, total_time, int(total_time) + 1)
    t_eval_hours = t_eval / 3600.0  # Convert to hours for plotting
    setpoints = generate_setpoint_array(t_eval, seq_config)

    results = []

    print("\n--- Running Simulations ---")
    for cfg in configs:
        if cfg["type"] == "de":
            kp, ki = run_de_tuner(cfg["name"], plant_params, cfg["weights"], max_kp=1.5, max_ki=0.002)
        else:
            kp, ki = cfg["kp"], cfg["ki"]

        # Create Controller
        controller = create_pi_controller(kp, ki)
        
        # Closed-loop systems
        T_y = ct.feedback(ct.series(controller, plant), 1)
        T_u = ct.feedback(controller, plant)

        # Simulate
        _, y_out = ct.forced_response(T_y, t_eval, setpoints)
        _, u_out = ct.forced_response(T_u, t_eval, setpoints)
        
        # Metrics Computation
        error = setpoints - y_out
        iae = np.trapezoid(np.abs(error), t_eval)
        itae = np.trapezoid(t_eval * np.abs(error), t_eval)
        mae = np.mean(np.abs(error))
        u_auc = np.trapezoid(np.clip(u_out, 0, 1), t_eval) # Clipping for realistic AUC
        rt = calculate_rise_time(t_eval, y_out, setpoints, seq_config['base_sp'], seq_config['step_sp'])
        os_pct = calculate_overshoot(y_out, setpoints, seq_config['base_sp'], seq_config['step_sp'])

        results.append({
            "name": cfg["name"], "color": cfg["color"], "kp": kp, "ki": ki,
            "t": t_eval, "t_hours": t_eval_hours, "y": y_out, "u": u_out, "sp": setpoints,
            "iae": iae, "itae": itae, "mae": mae, "u_auc": u_auc, "rt": rt, "os": os_pct
        })

    # 3. Export CSV Metrics
    csv_data = []
    for res in results:
        csv_data.append([
            res["name"], f'{res["iae"]:.2f}', f'{res["itae"]:.2f}', f'{res["mae"]:.4f}', 
            f'{res["u_auc"]:.2f}', f'{res["rt"]:.2f}', f'{res["os"]:.2f}'
        ])

    csv_path = os.path.join(output_dir, "simulation_metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Setup', 'IAE', 'ITAE', 'MAE', 'AUC_Effort', 'Rise_Time_s', 'Overshoot_pct'])
        writer.writerows(csv_data)

    # ==========================================
    # 4. PLOTTING
    # ==========================================
    
    # ---- Figure 1: Step Response ----
    plt.figure(figsize=(14, 8))
    plt.plot(t_eval_hours, setpoints, 'k--', label='Reference Setpoint', alpha=0.6)
    for res in results:
        plt.plot(res["t_hours"], res["y"], color=res["color"], label=f'{res["name"]} (IAE:{res["iae"]:.0f})')
        
    plt.title(step_title, fontsize=plot_font_size, pad=20)
    plt.xlabel(step_x_label, fontsize=plot_font_size)
    plt.ylabel(step_y_label, fontsize=plot_font_size)
    plt.xticks(fontsize=plot_font_size)
    plt.yticks(fontsize=plot_font_size)
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, 
               fontsize=plot_font_size, borderaxespad=0.)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "step_response.png"))

    # ---- Figure 2: Control Effort ----
    plt.figure(figsize=(14, 8))
    for res in results:
        plt.plot(res["t_hours"], np.clip(res["u"], 0, 1), color=res["color"], label=f'{res["name"]} Effort')
        
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
    plt.savefig(os.path.join(output_dir, "control_effort.png"))
    
    print(f"\nSimulation Complete. Plots and metrics saved in: {output_dir}")

if __name__ == "__main__":
    main()