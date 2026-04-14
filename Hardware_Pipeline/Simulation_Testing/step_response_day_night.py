import os
import collections
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# HARDWARE PIPELINE IMPORTS
# ==========================================
# Make sure this module is accessible in your environment
from Evolutionary_Algorithm_Testing.de.de_optimizer import DEOptimizer

# ==========================================
# 1. MODULAR SYSTEM CLASSES
# ==========================================
class DynamicFOPDTPlant:
    """First-Order Plus Dead Time (FOPDT) Virtual Plant with Day/Night Switching"""
    def __init__(self, day_config, night_config, switch_time_s, dt=1.0, initial_do=0.0, baseline_do=0.0):
        self.day_config = day_config
        self.night_config = night_config
        self.switch_time_s = switch_time_s
        self.dt = dt
        self.baseline_do = baseline_do
        self.current_do = initial_do
        self.sim_time = 0.0
        
        # Buffer sized to handle the maximum possible delay between the two profiles
        max_delay = max(self.day_config['delay'], self.night_config['delay'])
        self.max_delay_steps = max(1, int(max_delay / self.dt))
        self.u_buffer = collections.deque([0.0] * self.max_delay_steps, maxlen=self.max_delay_steps)

    def step(self, u):
        # Determine active plant dynamics
        if self.sim_time < self.switch_time_s:
            active_K = self.day_config['K']
            active_tau = self.day_config['tau']
            active_delay = self.day_config['delay']
        else:
            active_K = self.night_config['K']
            active_tau = self.night_config['tau']
            active_delay = self.night_config['delay']

        # Delay Handling
        delay_steps = int(active_delay / self.dt)
        if delay_steps == 0:
            delayed_u = u
        else:
            # Look back 'delay_steps' into the buffer
            delayed_u = self.u_buffer[-delay_steps]
            
        self.u_buffer.append(u)
        
        # FOPDT difference equation
        deviation_do = self.current_do - self.baseline_do
        dy = (self.dt / active_tau) * ((active_K * delayed_u) - deviation_do)
        
        self.current_do += dy
        self.sim_time += self.dt
        return self.current_do

class ScheduledPIController:
    """PI Controller capable of switching gains for Day/Night schedules"""
    def __init__(self, kp_day, ki_day, kp_night, ki_night, switch_time_s, dt=1.0):
        self.kp_day = kp_day
        self.ki_day = ki_day
        self.kp_night = kp_night
        self.ki_night = ki_night
        self.switch_time_s = switch_time_s
        self.dt = dt
        self.integral_sum = 0.0
        self.current_time = 0.0

    def compute(self, measured_y, setpoint):
        # Determine active gains
        if self.current_time < self.switch_time_s:
            kp, ki = self.kp_day, self.ki_day
        else:
            kp, ki = self.kp_night, self.ki_night

        error = setpoint - measured_y
        tentative_int = self.integral_sum + (error * self.dt)
        
        pi_out = (kp * error) + (ki * tentative_int)
        clamped_out = max(0.0, min(1.0, pi_out))
        
        # Anti-windup conditional integration
        if 0.0 < pi_out < 1.0: 
            self.integral_sum = tentative_int
            
        self.current_time += self.dt
        return clamped_out

class DayNightStepSequence:
    """Step Response Schedule with identical setpoints for Day and Night"""
    def __init__(self, base_sp, step_sp, half_duration):
        self.timeline = []
        self.half_duration = half_duration
        self.total_duration = half_duration * 2
        self.switch_time = half_duration
        
        # --- DAY SCHEDULE ---
        self.timeline.append((0, base_sp))
        self.timeline.append((half_duration * 0.25, step_sp)) # Step up
        self.timeline.append((half_duration * 0.75, base_sp)) # Recover

        # --- NIGHT SCHEDULE (Exact same values) ---
        self.timeline.append((half_duration, base_sp))
        self.timeline.append((half_duration * 1.25, step_sp)) # Step up
        self.timeline.append((half_duration * 1.75, base_sp)) # Recover

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
    def run(day_config, night_config, controller, sequence, dt=1.0):
        initial_sp = sequence.get_setpoint(0)
        
        plant = DynamicFOPDTPlant(
            day_config=day_config,
            night_config=night_config, 
            switch_time_s=sequence.switch_time, 
            dt=dt, 
            initial_do=initial_sp
        )
        
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

# ==========================================
# 4. METRIC COMPUTATION HELPER
# ==========================================
def compute_metrics(t, y, u, sp):
    """Computes IAE, ITAE, and Control Effort AUC from simulation arrays."""
    error = sp - y
    abs_error = np.abs(error)
    
    iae = np.trapezoid(abs_error, t)
    itae = np.trapezoid(t * abs_error, t)
    u_auc = np.trapezoid(u, t)
    
    return iae, itae, u_auc

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation_graphs")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------
    # A. SYSTEM SPECIFICATIONS
    # ------------------------------------------
    day_tf_config = {'K': 1.346 , 'tau': 1551.955, 'delay': 0.05} 
    night_tf_config = {'K': 2.36, 'tau': 3083.59, 'delay': 104.469} 
    dt_step = 1.0

    half_dur = 86400/2
    
    test_sequence = DayNightStepSequence(
        base_sp=0.0,             
        step_sp=1.0, 
        half_duration=half_dur         
    )

    max_kp, min_kp = 1.5, 0.001
    max_ki, min_ki = 0.002, 1e-6
    
    # ------------------------------------------
    # B. TUNE SETUPS 
    # ------------------------------------------
    print("\n--- Tuning Phase ---")
    # Setup 1: One-Shot Tuner (Based on Day only)
    kp_os, ki_os = run_de_tuner("Setup 1 (Day-Only Reference)", day_tf_config, 
                                min_kp=min_kp, max_kp=max_kp, min_ki=min_ki, max_ki=max_ki)
    
    # Setup 2: Two-Shot Tuner (Based on Day, then Night)
    kp_ts_day, ki_ts_day = run_de_tuner("Setup 2 (Day Phase)", day_tf_config, 
                                        min_kp=min_kp, max_kp=max_kp, min_ki=min_ki, max_ki=max_ki)
    kp_ts_night, ki_ts_night = run_de_tuner("Setup 2 (Night Phase)", night_tf_config, 
                                            min_kp=min_kp, max_kp=max_kp, min_ki=min_ki, max_ki=max_ki)

    # ------------------------------------------
    # C. SIMULATE SETUPS
    # ------------------------------------------
    results = []

    # 1. Simulate Setup 1 (One-Shot)
    controller_os = ScheduledPIController(
        kp_day=kp_os, ki_day=ki_os, 
        kp_night=kp_os, ki_night=ki_os, # Night uses same as Day
        switch_time_s=test_sequence.switch_time, dt=dt_step
    )
    t, y, u, sp = Simulator.run(day_tf_config, night_tf_config, controller_os, test_sequence, dt_step)
    iae, itae, u_auc = compute_metrics(t, y, u, sp)
    results.append({
        "name": "Setup 1 (One-Shot)", "color": "blue",
        "t": t, "y": y, "u": u, "sp": sp, "iae": iae, "itae": itae, "u_auc": u_auc
    })
    print(f"\n[Metrics] Setup 1 (One-Shot) -> IAE: {iae:.2f} | ITAE: {itae:.2f} | u_AUC: {u_auc:.2f}")

    # 2. Simulate Setup 2 (Two-Shot)
    controller_ts = ScheduledPIController(
        kp_day=kp_ts_day, ki_day=ki_ts_day, 
        kp_night=kp_ts_night, ki_night=ki_ts_night, 
        switch_time_s=test_sequence.switch_time, dt=dt_step
    )
    t, y, u, sp = Simulator.run(day_tf_config, night_tf_config, controller_ts, test_sequence, dt_step)
    iae, itae, u_auc = compute_metrics(t, y, u, sp)
    results.append({
        "name": "Setup 2 (Two-Shot)", "color": "orange",
        "t": t, "y": y, "u": u, "sp": sp, "iae": iae, "itae": itae, "u_auc": u_auc
    })
    print(f"[Metrics] Setup 2 (Two-Shot) -> IAE: {iae:.2f} | ITAE: {itae:.2f} | u_AUC: {u_auc:.2f}")

    # ------------------------------------------
    # D. PLOTTING
    # ------------------------------------------
    print("\n--- Generating Plots ---")
    
    # Plot 1: Step Response Comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Shaded regions for Day / Night
    ax.axvspan(0, half_dur, facecolor='yellow', alpha=0.1, label='Daytime Dynamics')
    ax.axvspan(half_dur, test_sequence.total_duration, facecolor='navy', alpha=0.08, label='Nighttime Dynamics')

    ax.plot(results[0]["t"], results[0]["sp"], color='black', linewidth=1.5, label='Reference Setpoint', linestyle='--')
    
    for res in results:
        label = f'{res["name"]} (IAE:{res["iae"]:.0f} | ITAE:{res["itae"]:.2e})'
        ax.plot(res["t"], res["y"], color=res["color"], linewidth=1.5, label=label)

    ax.set_title('Dissolved Oxygen Step Response Comparison (Day vs Night)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Dissolved Oxygen (mg l$^{-1}$)')
    ax.set_xlim(0, test_sequence.total_duration)
    ax.set_ylim(0.0, 3.0) 
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "day_night_response_comparison.png"), dpi=300)
    plt.close()

    # Plot 2: Control Effort Comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.axvspan(0, half_dur, facecolor='yellow', alpha=0.1)
    ax.axvspan(half_dur, test_sequence.total_duration, facecolor='navy', alpha=0.08)

    for res in results:
        ax.plot(res["t"], res["u"], color=res["color"], linewidth=1.2, alpha=0.8, 
                 label=f'{res["name"]} Effort (AUC: {res["u_auc"]:.0f})')

    ax.set_title('Control Effort (Duty Cycle) Comparison (Day vs Night)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Duty Cycle (0.0 to 1.0)')
    ax.set_xlim(0, test_sequence.total_duration)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "day_night_effort_comparison.png"), dpi=300)
    plt.close()
    
    print(f"Done! Total simulated duration: {test_sequence.total_duration}s. Graphs saved to {output_dir}")

if __name__ == "__main__":
    main()