import os
import collections
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# HARDWARE PIPELINE IMPORTS
# ==========================================
from Evolutionary_Algorithm_Testing.de.de_optimizer import DEOptimizer

# ==========================================
# 1. MODULAR SYSTEM CLASSES
# ==========================================
class FOPDTPlant:
    """First-Order Plus Dead Time (FOPDT) Virtual Plant"""
    def __init__(self, K, tau, delay, dt=1.0, initial_do=5.0, baseline_do=0.0):
        self.K = K
        self.tau = tau
        self.dt = dt
        self.baseline_do = baseline_do
        self.current_do = initial_do
        self.sim_time = 0.0
        
        delay_steps = int(delay / self.dt)
        self.u_buffer = collections.deque([0.0] * max(1, delay_steps), maxlen=max(1, delay_steps))

    def step(self, u):
        delayed_u = self.u_buffer.popleft()
        self.u_buffer.append(u)
        
        deviation_do = self.current_do - self.baseline_do
        dy = (self.dt / self.tau) * ((self.K * delayed_u) - deviation_do)
        
        self.current_do += dy
        self.sim_time += self.dt
        return self.current_do

class PIController:
    """Standard PI Controller with Conditional Integration (Anti-windup)"""
    def __init__(self, kp, ki, dt=1.0):
        self.kp = kp
        self.ki = ki
        self.dt = dt
        self.integral_sum = 0.0

    def compute(self, measured_y, setpoint):
        error = setpoint - measured_y
        tentative_int = self.integral_sum + (error * self.dt)
        
        pi_out = (self.kp * error) + (self.ki * tentative_int)
        clamped_out = max(0.0, min(1.0, pi_out))
        
        if 0.0 < pi_out < 1.0: 
            self.integral_sum = tentative_int
            
        return clamped_out

class StepSequence:
    """Customizable Step Response Schedule Generator"""
    def __init__(self, base_sp, step_sp, pre_step_delay, step_duration, recovery_duration, cycles=1):
        self.timeline = [(0, base_sp)]
        current_t = pre_step_delay
        
        for _ in range(cycles):
            # Step down (or up)
            self.timeline.append((current_t, step_sp))
            current_t += step_duration
            
            # Recover back to base
            self.timeline.append((current_t, base_sp))
            current_t += recovery_duration
            
        self.total_duration = current_t

    def get_setpoint(self, t):
        active_sp = self.timeline[0][1]
        for change_t, sp in self.timeline:
            if t >= change_t:
                active_sp = sp
            else:
                break
        return active_sp

# ==========================================
# 2. SIMULATION ENGINE
# ==========================================
class Simulator:
    @staticmethod
    def run(plant_config, controller, sequence, dt=1.0):
        # Initialize plant at the sequence's starting setpoint
        initial_sp = sequence.get_setpoint(0)
        plant = FOPDTPlant(**plant_config, dt=dt, initial_do=initial_sp)
        
        t_hist, y_hist, u_hist, sp_hist = [], [], [], []
        current_time = 0.0

        while current_time <= sequence.total_duration:
            setpoint = sequence.get_setpoint(current_time)
            
            u = controller.compute(plant.current_do, setpoint)
            y = plant.step(u)
            
            t_hist.append(current_time)
            y_hist.append(y)
            u_hist.append(u)
            sp_hist.append(setpoint)
            
            current_time += dt
            
        return np.array(t_hist), np.array(y_hist), np.array(u_hist), np.array(sp_hist)

# ==========================================
# 3. AUTO-TUNING WRAPPER
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
# 4. MAIN EXECUTION
# ==========================================
def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation_graphs")
    os.makedirs(output_dir, exist_ok=True)

    # 1. System Definition
    plant_tf = {'K': 2.43, 'tau': 3492.589, 'delay': 0.05} 
    dt_step = 1.0

    test_sequence = StepSequence(
        base_sp=0.0,             
        step_sp=1.0,             
        pre_step_delay=15000,     
        step_duration=15000,      
        recovery_duration=15000,  
        cycles=2                 
    )

    max_kp = 3
    min_kp = 0
    max_ki = 0.05
    min_ki = 0.0001
    
    # 2a. Define DE Tuners [Error, Control Effort, Overshoot, Rise Time]
    de_tuner_configs = [
        {
            "name": "DE Baseline", 
            "weights": (1.0, 10.0, 1.0, 1.0), 
            "color": "blue",
            "bounds": {"min_kp": min_kp, "max_kp": max_kp, "min_ki": min_ki, "max_ki": max_ki}
        },
    ]

    # 2b. Define Hardcoded / External Tuners (MATLAB, Lambda, etc.)
    manual_configs = [
        {
            "name": "Lambda Tuning", 
            "kp": 0.41,   # Replace with your Lambda Kp
            "ki": 0.0029,  # Replace with your Lambda Ki
            "color": "cyan"
        }
    ]

    results = []

    # 3a. Tune and Simulate DE Configurations
    for cfg in de_tuner_configs:
        bounds = cfg.get("bounds", {})
        kp, ki = run_de_tuner(cfg["name"], plant_tf, cfg["weights"], **bounds)
        
        controller = PIController(kp=kp, ki=ki, dt=dt_step)
        t, y, u, sp = Simulator.run(plant_tf, controller, sequence=test_sequence, dt=dt_step)
        
        results.append({
            "name": cfg["name"], "color": cfg["color"],
            "t": t, "y": y, "u": u, "sp": sp, "kp": kp, "ki": ki
        })

    # 3b. Simulate Hardcoded / Manual Configurations
    print("\n[Simulator] Running Manual/External Configurations...")
    for cfg in manual_configs:
        kp, ki = cfg["kp"], cfg["ki"]
        
        controller = PIController(kp=kp, ki=ki, dt=dt_step)
        t, y, u, sp = Simulator.run(plant_tf, controller, sequence=test_sequence, dt=dt_step)
        
        print(f"[Result] {cfg['name']} -> Simulated with Kp: {kp:.4f}, Ki: {ki:.4f}")
        results.append({
            "name": cfg["name"], "color": cfg["color"],
            "t": t, "y": y, "u": u, "sp": sp, "kp": kp, "ki": ki
        })

    # 4. Plotting
    print("\n--- Generating Plots ---")
    
    # Plot 1: Step Response Comparison
    plt.figure(figsize=(12, 7))
    plt.plot(results[0]["t"], results[0]["sp"], color='black', linewidth=1.5, label='Reference Setpoint', linestyle='--')
    
    for res in results:
        label = f'{res["name"]} (Kp:{res["kp"]:.2f}, Ki:{res["ki"]:.4f})'
        plt.plot(res["t"], res["y"], color=res["color"], linewidth=1.5, label=label)

    plt.title('Dissolved Oxygen Step Response Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Dissolved Oxygen (mg l$^{-1}$)')
    plt.xlim(0, test_sequence.total_duration)
    plt.ylim(0.0, 3.0) 
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "step_response_comparison.png"), dpi=300)
    plt.close()

    # Plot 2: Control Effort Comparison
    plt.figure(figsize=(12, 7))
    for res in results:
        auc = np.trapezoid(res["u"], res["t"])
        plt.plot(res["t"], res["u"], color=res["color"], linewidth=1.2, alpha=0.8, 
                 label=f'{res["name"]} Effort (AUC: {auc:.0f})')

    plt.title('Control Effort (Duty Cycle) Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Duty Cycle (0.0 to 1.0)')
    plt.xlim(0, test_sequence.total_duration)
    plt.ylim(0, 1.05)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "control_effort_comparison.png"), dpi=300)
    plt.close()
    
    print(f"Done! Total simulated duration: {test_sequence.total_duration}s. Graphs saved to {output_dir}")

if __name__ == "__main__":
    main()