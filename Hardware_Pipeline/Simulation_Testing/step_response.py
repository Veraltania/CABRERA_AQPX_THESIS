import os
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
    
    ref = np.full_like(t, base_sp)
    
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
# 2. AUTO-TUNING WRAPPER
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

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation_graphs_control_lib")
    os.makedirs(output_dir, exist_ok=True)

    # 1. System Definition
    plant_params = {'K': 2.43, 'tau': 3492.589, 'delay': 0.05} 
    plant = create_fopdt_sys(**plant_params)
    
    seq_config = {
        'base_sp': 1.0, 'step_sp': 2.0,
        'pre_step_delay': 10000, 'step_duration': 10000,
        'recovery_duration': 10000, 'cycles': 2
    }
    
    total_time = seq_config['pre_step_delay'] + (seq_config['step_duration'] + seq_config['recovery_duration']) * seq_config['cycles']
    t_eval = np.linspace(0, total_time, int(total_time) + 1)
    setpoints = generate_setpoint_array(t_eval, seq_config)

    # 2. Tuner Configurations
    configs = [
        {"name": "DE Baseline", "type": "de", "weights": (1.0, 1.0, 1.0, 1.0), "color": "blue", "kp": 0, "ki": 0},
        {"name": "Lambda Tuning", "type": "manual", "kp": 0.41, "ki": 0.000136, "color": "cyan"}
    ]

    results = []

    for cfg in configs:
        if cfg["type"] == "de":
            kp, ki = run_de_tuner(cfg["name"], plant_params, cfg["weights"], max_kp=1.5, max_ki=0.002)
        else:
            kp, ki = cfg["kp"], cfg["ki"]

        # Create Controller
        controller = create_pi_controller(kp, ki)
        
        # Closed-loop systems
        # T_y: Setpoint -> Output (y)
        # T_u: Setpoint -> Control Effort (u)
        T_y = ct.feedback(ct.series(controller, plant), 1)
        T_u = ct.feedback(controller, plant)

        # Simulate
        # Note: We subtract initial baseline for linear simulation, then add back
        # Since standard linear simulation assumes 0 initial state
        _, y_out = ct.forced_response(T_y, t_eval, setpoints)
        _, u_out = ct.forced_response(T_u, t_eval, setpoints)
        
        # Metrics
        error = setpoints - y_out
        iae = np.trapezoid(np.abs(error), t_eval)
        itae = np.trapezoid(t_eval * np.abs(error), t_eval)
        u_auc = np.trapezoid(np.clip(u_out, 0, 1), t_eval) # Clipping for realistic AUC

        results.append({
            "name": cfg["name"], "color": cfg["color"], "kp": kp, "ki": ki,
            "t": t_eval, "y": y_out, "u": u_out, "sp": setpoints,
            "iae": iae, "itae": itae, "u_auc": u_auc
        })

    # 4. Plotting (Same logic as original)
    plt.figure(figsize=(12, 7))
    plt.plot(t_eval, setpoints, 'k--', label='Reference Setpoint', alpha=0.6)
    for res in results:
        plt.plot(res["t"], res["y"], color=res["color"], label=f'{res["name"]} (IAE:{res["iae"]:.0f})')
    plt.title('Step Response Comparison (Control Library)')
    plt.xlabel('Time (s)')
    plt.ylabel('DO (mg/l)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "step_response.png"))

    plt.figure(figsize=(12, 7))
    for res in results:
        plt.plot(res["t"], np.clip(res["u"], 0, 1), color=res["color"], label=f'{res["name"]} Effort')
    plt.title('Control Effort Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Duty Cycle')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "control_effort.png"))
    
    print(f"Simulation Complete. Plots saved in: {output_dir}")

if __name__ == "__main__":
    main()