import sys
import os
import collections
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch

# ==========================================
# HARDWARE PIPELINE IMPORTS
# ==========================================
from Evolutionary_Algorithm_Testing.de.de_optimizer import DEOptimizer
from Hardware_Pipeline.controllers import DOController
from Hardware_Pipeline.tuning_strategies import StaticTuningStrategy

# ==========================================
# 1. VIRTUAL HARDWARE CLASSES
# ==========================================
class VirtualAquaculturePlant:
    def __init__(self, day_tf, night_tf, dt=5.0, disturbances=None):
        self.day_tf = day_tf
        self.night_tf = night_tf
        self.dt = dt
        self.baseline_do = 1.0  
        self.current_do = 1.0
        self.active_tf = self.day_tf
        delay_steps = int(self.active_tf['delay'] / self.dt)
        self.u_buffer = collections.deque([0.0] * max(1, delay_steps), maxlen=max(1, delay_steps))
        self.sim_time = 0.0
        self.disturbances = disturbances if disturbances else {}
        self.applied_disturbances = set()

    def switch_to_day(self): 
        self.active_tf = self.day_tf
        
    def switch_to_night(self): 
        self.active_tf = self.night_tf

    def step(self, u):
        K = self.active_tf['K']
        tau = self.active_tf['tau']
        delayed_u = self.u_buffer.popleft()
        self.u_buffer.append(u)
        deviation_do = self.current_do - self.baseline_do
        dy = (self.dt / tau) * ((K * delayed_u) - deviation_do)
        self.current_do += dy
        self.sim_time += self.dt
        for d_time, d_val in self.disturbances.items():
            if self.sim_time >= d_time and d_time not in self.applied_disturbances:
                self.current_do += d_val
                self.applied_disturbances.add(d_time)
                print(f"[Plant] ⚠️ Sudden Disturbance of {d_val} DO applied at {self.sim_time/3600:.2f} hrs")
        return self.current_do

class VirtualActuator:
    def __init__(self): 
        self.duty_cycle = 0.0
        
    def set_duty_cycle(self, duty_cycle): 
        self.duty_cycle = max(0.0, min(1.0, duty_cycle))

class VirtualStrategyManager:
    def __init__(self):
        # Strictly Non-Adaptive Static Tuning Strategy
        self.strategy = StaticTuningStrategy()

    def get_active_strategy(self): 
        return self.strategy

class VirtualClock:
    def __init__(self, start_time=0.0): 
        self.current_time = start_time
        
    def tick(self, dt): 
        self.current_time += dt
        
    def get_time(self): 
        return self.current_time

# ==========================================
# 2. AUTO-TUNING (COST FUNCTION BASED)
# ==========================================
def auto_tune_gains(tf_name, tf_config, weights=(1.0, 1.0, 1.0, 1.0)):
    print(f"\n[Auto-Tuner] Running DE Optimization for {tf_name} Phase...")
    tf_params = {
        'tf_num': [tf_config['K']], 'tf_den': [tf_config['tau'], 1], 'tf_delay': tf_config['delay'],
        'tf_n_pade': 2, 'computed_delay': tf_config['delay'], 'is_reverse_acting': False, 'max_kp': 20.0
    }
    config = {
        'patience': 20, 
        'tol': 1e-4, 
        'mutation': (0.5, 1.0), 
        'recombination': 0.745, 
        'strategy': 'best1bin',
        'weights': weights
    }
    optimizer = DEOptimizer(config, tf_params)
    
    best_sol, iterations = optimizer.optimize_round(round_num=1)
    best_kp, best_ki, best_cost, raw_costs = best_sol
    
    print(f"[Auto-Tuner] {tf_name} Optimal Gains Found -> Kp: {best_kp:.4f}, Ki: {best_ki:.4f} (Cost: {best_cost:.4f})")
    return best_kp, best_ki

# ==========================================
# 3. PURE MATLAB BASELINE SIMULATION
# ==========================================
def run_matlab_baseline_simulation(matlab_tf, kp, ki, target_setpoint, sim_duration=86400, dt=5.0, 
                                   add_sensor_noise=False, sensor_noise_std=0.05,
                                   add_process_noise=False, process_noise_std=0.005, disturbances=None):
    plant = VirtualAquaculturePlant(day_tf=matlab_tf, night_tf=matlab_tf, dt=dt, disturbances=disturbances)
    integral_sum = 0.0
    current_time = 0.0
    time_history, do_history, u_history = [], [], []
    print(f"\n--- Starting MATLAB Baseline (Hardcoded) Simulation ---")

    while current_time < sim_duration:
        current_do = plant.current_do
        if add_process_noise: current_do += random.gauss(0, process_noise_std)
        plant.current_do = current_do
        measured_do = current_do
        if add_sensor_noise: measured_do += random.gauss(0, sensor_noise_std)
        
        error = target_setpoint - measured_do
        tentative_int = integral_sum + (error * dt)
        pi_out = (kp * error) + (ki * tentative_int)
        clamped_out = max(0.0, min(1.0, pi_out))
        if 0.0 < pi_out < 1.0: integral_sum = tentative_int
        
        plant.step(clamped_out)
        time_history.append(current_time / 3600.0)
        do_history.append(plant.current_do)
        u_history.append(clamped_out)
        current_time += dt
        
    return time_history, do_history, u_history

# ==========================================
# 4. HARDWARE PIPELINE SIMULATION (NON-ADAPTIVE)
# ==========================================
def run_simulation(day_tf, night_tf, day_gains, night_gains, target_setpoint, 
                   sim_duration=86400, dt=5.0, add_sensor_noise=False, sensor_noise_std=0.05,
                   add_process_noise=False, process_noise_std=0.005, disturbances=None):
    
    plant = VirtualAquaculturePlant(day_tf, night_tf, dt=dt, disturbances=disturbances)
    actuator = VirtualActuator()
    manager = VirtualStrategyManager()
    v_clock = VirtualClock()
    
    controller = DOController(name="Sim-DO", strategy_manager=manager, actuator=actuator)
    controller.setpoint = target_setpoint
    controller.kp = day_gains[0]
    controller.ki = day_gains[1]
    
    time_history, do_history, u_history = [], [], []
    night_switched = False
    
    print(f"\n--- Starting Hardware Pipeline NON-ADAPTIVE Simulation ---")

    # Patch time to sync the hardware controller with the virtual clock
    with patch('Hardware_Pipeline.controllers.time.time', side_effect=v_clock.get_time):
        while v_clock.get_time() < sim_duration:
            t = v_clock.get_time()
            if t >= 43200 and not night_switched:
                plant.switch_to_night()
                night_switched = True

            current_do = plant.current_do
            if add_process_noise: current_do += random.gauss(0, process_noise_std)
            plant.current_do = current_do
            
            measured_do = current_do
            if add_sensor_noise: measured_do += random.gauss(0, sensor_noise_std)

            fake_payload = {'mcp_wq': {'do': measured_do}}
            controller.process(fake_payload)
            plant.step(actuator.duty_cycle)
            
            time_history.append(v_clock.get_time() / 3600.0) 
            do_history.append(current_do)           
            u_history.append(actuator.duty_cycle)
            
            v_clock.tick(dt)

    return time_history, do_history, u_history

# ==========================================
# 5. MAIN EXECUTION & PLOTTING 
# ==========================================
def main():
    output_dir_name = "simulation_graphs"  
    target_do_setpoint = 1.5  
    sim_duration = 86400 
    dt_step = 5.0
    pwm_window_minutes = 30 

    add_sensor_noise = False
    sensor_noise_std = 0.05 
    add_process_noise = False
    process_noise_std = 0.005 
    seed_value = 42

    day_tf = {'K': 1.133, 'tau': 2833.82, 'delay': 0.05}
    night_tf = {'K': 2.049, 'tau': 4499.996, 'delay': 0.05}
    
    weights = (1.0, 1.0, 1.0, 1.0)
    matlab_plant = day_tf  
    matlab_kp = 0.92
    matlab_ki = 0.000974
    
    scheduled_disturbances = {
        14400.0: -0.5,
        28800.0: +0.5,
        43200.0: -0.5,
        57600.0: +0.5
    }
    
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    # Run MATLAB Baseline
    t_matlab, do_matlab, u_matlab = run_matlab_baseline_simulation(
        matlab_tf=matlab_plant, kp=matlab_kp, ki=matlab_ki, 
        target_setpoint=target_do_setpoint, sim_duration=sim_duration, dt=dt_step,
        add_sensor_noise=add_sensor_noise, sensor_noise_std=sensor_noise_std,
        add_process_noise=add_process_noise, process_noise_std=process_noise_std,
        disturbances=scheduled_disturbances
    )

    # Run Hardware Optimizer (Cost Function Based)
    day_kp, day_ki = auto_tune_gains("Daytime", day_tf, weights=weights)
    night_kp, night_ki = auto_tune_gains("Nighttime", night_tf, weights=weights)

    # Run Hardware Non-Adaptive Simulation
    random.seed(seed_value) 
    t_non_adaptive, do_non_adaptive, u_non_adaptive = run_simulation(
        day_tf=day_tf, night_tf=night_tf, 
        day_gains=(day_kp, day_ki), night_gains=(night_kp, night_ki),
        target_setpoint=target_do_setpoint, sim_duration=sim_duration, dt=dt_step,
        add_sensor_noise=add_sensor_noise, sensor_noise_std=sensor_noise_std,
        add_process_noise=add_process_noise, process_noise_std=process_noise_std,
        disturbances=scheduled_disturbances
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # PLOT 1: DO Comparison (MATLAB vs HW Tuned)
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    # ----> SHADE DISTURBANCES
    added_dist_label = False
    for d_time in scheduled_disturbances.keys():
        d_time_h = d_time / 3600.0
        plt.axvspan(d_time_h, d_time_h + 0.5, color='salmon', alpha=0.3, 
                    label='Disturbance Applied' if not added_dist_label else "")
        added_dist_label = True

    plt.plot(t_matlab, do_matlab, label='MATLAB Baseline (Hardcoded LTI)', color='black', linestyle=':', linewidth=2, alpha=0.9)
    plt.plot(t_non_adaptive, do_non_adaptive, label='Hardware (Cost-Function Tuned)', color='red', linestyle='-', linewidth=2, alpha=0.8)
    
    plt.axhline(y=target_do_setpoint, color='green', linestyle='-', label=f'Setpoint ({target_do_setpoint})')
    
    plt.axvline(x=12.0, color='gray', linestyle='-', alpha=0.5)
    
    ax.text(0.25, 1.01, 'DAYTIME', transform=ax.transAxes, fontsize=12, fontweight='bold', alpha=0.6, ha='center', va='bottom')
    ax.text(0.75, 1.01, 'NIGHTTIME', transform=ax.transAxes, fontsize=12, fontweight='bold', alpha=0.6, ha='center', va='bottom')
    
    plt.title('Simulated DO Control: MATLAB Baseline vs Hardware Pipeline', y=1.05)
    plt.xlabel('Time (Hours)')
    plt.ylabel('Dissolved Oxygen (mg/L)')
    plt.ylim(bottom=0) 
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "do_comparison_matlab_vs_hw.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 2: Continuous Control Signal & Control Effort (AUC)
    # ---------------------------------------------------------
    # Calculate Area Under the Curve (Control Effort) using Trapezoidal Rule
    # Note: Time is in hours, so AUC is in (Duty Cycle * Hours)
    auc_matlab = np.trapezoid(u_matlab, t_matlab)
    auc_non_adaptive = np.trapezoid(u_non_adaptive, t_non_adaptive)

    plt.figure(figsize=(12, 7))
    
    # --- MATLAB Subplot ---
    plt.subplot(2, 1, 1)
    added_dist_label = False
    for d_time in scheduled_disturbances.keys():
        d_time_h = d_time / 3600.0
        plt.axvspan(d_time_h, d_time_h + 0.5, color='salmon', alpha=0.3, 
                    label='Disturbance Applied' if not added_dist_label else "")
        added_dist_label = True

    plt.plot(t_matlab, u_matlab, color='black', linewidth=1.5, alpha=0.9)
    plt.fill_between(t_matlab, 0, u_matlab, color='black', alpha=0.3, 
                     label=f'MATLAB Control Effort (AUC: {auc_matlab:.2f})')
    plt.title('MATLAB LTI Continuous Control Signal')
    plt.ylabel('Duty Cycle (0.0 to 1.0)')
    plt.ylim(0, 1.05)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # --- Hardware Subplot ---
    plt.subplot(2, 1, 2)
    added_dist_label = False
    for d_time in scheduled_disturbances.keys():
        d_time_h = d_time / 3600.0
        plt.axvspan(d_time_h, d_time_h + 0.5, color='salmon', alpha=0.3, 
                    label='Disturbance Applied' if not added_dist_label else "")
        added_dist_label = True

    plt.plot(t_non_adaptive, u_non_adaptive, color='red', linewidth=1.5, alpha=0.9)
    plt.fill_between(t_non_adaptive, 0, u_non_adaptive, color='red', alpha=0.3, 
                     label=f'Hardware Tuned Control Effort (AUC: {auc_non_adaptive:.2f})')
    plt.axvline(x=12.0, color='gray', linestyle='-', alpha=0.5)
    plt.title('Hardware Tuned Continuous Control Signal')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Duty Cycle (0.0 to 1.0)')
    plt.ylim(0, 1.05)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "continuous_control_signal_matlab_vs_hw.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    main()