import sys
import os
import collections
import time
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from unittest.mock import patch
import numpy as np
from scipy.optimize import minimize

# ==========================================
# HARDWARE PIPELINE IMPORTS
# ==========================================
from Evolutionary_Algorithm_Testing.de.de_optimizer import DEOptimizer
from Hardware_Pipeline.controllers import DOController
from Hardware_Pipeline.tuning_strategies import AdaptiveTuningStrategy, StaticTuningStrategy

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
    def __init__(self, is_adaptive):
        if is_adaptive:
            # 1-Hour window for calculating Mean Absolute Error
            self.strategy = AdaptiveTuningStrategy(window_duration=3600)
        else:
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
# 2. AUTO-TUNING & CUSTOM FITTER
# ==========================================
def auto_tune_gains(tf_name, tf_config):
    print(f"\n[Auto-Tuner] Running DE Optimization for {tf_name} Phase...")
    tf_params = {
        'tf_num': [tf_config['K']], 'tf_den': [tf_config['tau'], 1], 'tf_delay': tf_config['delay'],
        'tf_n_pade': 2, 'computed_delay': tf_config['delay'], 'is_reverse_acting': False, 'max_kp': 2.0
    }
    config = {'patience': 20, 'tol': 1e-4, 'mutation': (0.5, 1.0), 'recombination': 0.745, 'strategy': 'best1bin'}
    optimizer = DEOptimizer(config, tf_params)
    
    # Corrected unpacking based on updated DEOptimizer signature
    best_sol, iterations = optimizer.optimize_round(round_num=1)
    best_kp, best_ki, best_cost, raw_costs = best_sol
    
    print(f"[Auto-Tuner] {tf_name} Optimal Gains Found -> Kp: {best_kp:.4f}, Ki: {best_ki:.4f} (Cost: {best_cost:.4f})")
    return best_kp, best_ki

def fit_closed_loop_fopdt(t_arr, u_arr, y_arr):
    t_arr, u_arr, y_arr = np.array(t_arr), np.array(u_arr), np.array(y_arr)
    dt = t_arr[1] - t_arr[0]
    u0, y0 = u_arr[0], y_arr[0]
    du = u_arr - u0
    dy = y_arr - y0
    
    def simulate_fopdt(K, tau, delay):
        y_sim = np.zeros_like(t_arr)
        delay_steps = int(max(0.0, delay) / dt)
        du_delayed = np.zeros_like(du)
        if delay_steps < len(du): 
            du_delayed[delay_steps:] = du[:-delay_steps]
            
        for i in range(1, len(t_arr)):
            derivative = (dt / max(1.0, tau)) * (K * du_delayed[i-1] - y_sim[i-1])
            y_sim[i] = y_sim[i-1] + derivative
        return y_sim
        
    def objective_function(scaled_params):
        K = scaled_params[0]
        tau = scaled_params[1] * 1000.0   
        delay = scaled_params[2] * 10.0   
        
        y_sim = simulate_fopdt(K, tau, delay)
        ise = np.sum((dy - y_sim)**2) * dt
        return ise
        
    bnds = ((0.1, 10.0), (0.01, 10.0), (0.0, 50.0)) 
    initial_guess = [1.5, 1.5, 1.0] 
    
    res = minimize(objective_function, initial_guess, bounds=bnds, method='L-BFGS-B')
    return res.x[0], res.x[1] * 1000.0, res.x[2] * 10.0

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
    return time_history, do_history, u_history, []

# ==========================================
# 4. HARDWARE PIPELINE SIMULATION LOOP
# ==========================================
def run_simulation(is_adaptive, day_tf, night_tf, day_gains, night_gains, target_setpoint, 
                   sim_duration=86400, dt=5.0, add_sensor_noise=False, sensor_noise_std=0.05,
                   add_process_noise=False, process_noise_std=0.005, disturbances=None,
                   adaptive_weights=(1.0, 1.0, 1.0, 1.0)):
    
    plant = VirtualAquaculturePlant(day_tf, night_tf, dt=dt, disturbances=disturbances)
    actuator = VirtualActuator()
    manager = VirtualStrategyManager(is_adaptive)
    v_clock = VirtualClock()
    
    controller = DOController(name="Sim-DO", strategy_manager=manager, actuator=actuator)
    controller.setpoint = target_setpoint
    controller.kp = day_gains[0]
    controller.ki = day_gains[1]
    
    time_history, do_history, u_history = [], [], []
    retune_intervals = [] 
    night_switched = False
    
    print(f"\n--- Starting Hardware Pipeline {'ADAPTIVE' if is_adaptive else 'NON-ADAPTIVE'} Simulation ---")

    with patch('Hardware_Pipeline.controllers.time.time', side_effect=v_clock.get_time), \
         patch('Hardware_Pipeline.tuning_strategies.time.time', side_effect=v_clock.get_time), \
         patch('Hardware_Pipeline.controllers.datetime') as mock_dt:
        
        mock_dt.now.side_effect = lambda: datetime(2025, 1, 1) + timedelta(seconds=v_clock.get_time())
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

        def sim_retune_full():
            if controller.retune_thread_active: return
            controller.retune_thread_active = True
            
            print(f"\n[MIL Sim] 🛑 Adaptive Retune Triggered natively by MAE. Waiting for stability...")
            
            # ==================================================
            # 1. PRE-TEST STABILIZATION PHASE
            # ==================================================
            stability_window = int(1800 / dt)  
            do_buffer = collections.deque(maxlen=stability_window)
            is_stable = False
            max_wait_steps = int(7200 / dt)   # Max wait 1 hour
            
            old_setpoint = controller.setpoint
            
            for _ in range(max_wait_steps):
                v_clock.tick(dt)
                curr_do = plant.current_do
                meas_do = curr_do + (random.gauss(0, sensor_noise_std) if add_sensor_noise else 0)
                if add_process_noise: curr_do += random.gauss(0, process_noise_std)
                plant.current_do = curr_do
                
                # Normal PI control to maintain old setpoint while waiting
                error = old_setpoint - meas_do
                tentative_int = controller.integral_sum + (error * dt)
                pi_out = (controller.kp * error) + (controller.ki * tentative_int)
                pi_out = max(controller.min_out, min(controller.max_out, pi_out))
                if controller.min_out < pi_out < controller.max_out:
                    controller.integral_sum = tentative_int
                
                actuator.set_duty_cycle(pi_out)
                plant.step(actuator.duty_cycle)
                
                time_history.append(v_clock.get_time() / 3600.0)
                do_history.append(plant.current_do)
                u_history.append(actuator.duty_cycle)
                
                do_buffer.append(meas_do)
                
                # Check if buffer is full and variance is low enough
                if len(do_buffer) == stability_window:
                    if np.std(do_buffer) <= (sensor_noise_std * 1.2): 
                        is_stable = True
                        break
                        
            if not is_stable:
                print("[MIL Sim] Retune aborted. System could not stabilize at the setpoint.")
                controller.retune_thread_active = False
                return
                
            print(f"[MIL Sim] System stabilized. Executing closed-loop step test...")

            # ==================================================
            # 2. EXCITATION STEP TEST (FIXED DURATION)
            # ==================================================
            start_hour = v_clock.get_time() / 3600.0
            new_setpoint = old_setpoint * 1.2
            
            controller.start_retuning_session(controller.target_column)
            controller.setpoint = new_setpoint
            
            bump_t, bump_u, bump_y = [], [], []
            
            bump_t.append(v_clock.get_time())
            bump_u.append(actuator.duty_cycle)
            bump_y.append(plant.current_do)
            
            test_duration_seconds = 3600 
            total_steps = int(test_duration_seconds / dt)
            print(f"[MIL Sim] Running fixed-duration closed-loop step test for {test_duration_seconds / 3600:.2f} hours...")
            
            for _ in range(total_steps):
                v_clock.tick(dt)
                curr_do = plant.current_do
                meas_do = curr_do + (random.gauss(0, sensor_noise_std) if add_sensor_noise else 0)
                if add_process_noise: curr_do += random.gauss(0, process_noise_std)
                plant.current_do = curr_do
                
                error = new_setpoint - meas_do
                tentative_int = controller.integral_sum + (error * dt)
                pi_out = (controller.kp * error) + (controller.ki * tentative_int)
                pi_out = max(controller.min_out, min(controller.max_out, pi_out))
                if controller.min_out < pi_out < controller.max_out:
                    controller.integral_sum = tentative_int
                
                controller._record_retuning_data(meas_do)
                actuator.set_duty_cycle(pi_out)
                plant.step(actuator.duty_cycle)
                
                time_history.append(v_clock.get_time() / 3600.0)
                do_history.append(plant.current_do)
                u_history.append(actuator.duty_cycle)
                bump_t.append(v_clock.get_time())
                bump_u.append(actuator.duty_cycle)
                bump_y.append(plant.current_do)
            
            controller.stop_retuning_session()
            
            # ==================================================
            # 3. IDENTIFICATION & OPTIMIZATION
            # ==================================================
            try:
                ex_K, ex_tau, ex_delay = fit_closed_loop_fopdt(bump_t, bump_u, bump_y)
                safe_delay = max(0.05, ex_delay) 
                safe_tau = max(1.0, ex_tau) 
                
                tf_params = {
                    'tf_num': [ex_K], 'tf_den': [safe_tau, 1], 'tf_delay': safe_delay,        
                    'tf_n_pade': 2, 'computed_delay': safe_delay, 'is_reverse_acting': ex_K < 0, 'max_kp': 2.0
                }
                de_config = {
                    'population_size': 50, 'max_iters': 100, 
                    'patience_limit': 5,
                    'mutation': (0.5, 1.0), 
                    'recombination': 0.745, 
                    'strategy': 'best1bin', 
                    'n_rounds': 1,
                    'weights': adaptive_weights
                }

                print(f"Weights: {adaptive_weights}")
                optimizer = DEOptimizer(config=de_config, tf_params=tf_params)
                
                # Corrected unpacking for inner simulation retune
                best_sol, iterations = optimizer.optimize_round(round_num=1)
                best_Kp, best_Ki, best_cost, raw_costs = best_sol
                
                controller.update_tuning_parameters(best_Kp, best_Ki, ex_K, safe_tau, safe_delay)
            except Exception as e:
                print(f"[MIL Sim] Hardware retuning pipeline failed: {e}")
                
            controller.setpoint = old_setpoint
            controller.retune_thread_active = False
            end_hour = v_clock.get_time() / 3600.0
            retune_intervals.append((start_hour, end_hour))

        controller.retune = sim_retune_full

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

    return time_history, do_history, u_history, retune_intervals

# ==========================================
# 5. MAIN EXECUTION & PLOTTING 
# ==========================================
def main():
    # Defining localized execution variables to eliminate global scope contamination
    output_dir_name = "simulation_graphs"  
    target_do_setpoint = 1.5  
    sim_duration = 86400 
    dt_step = 5.0

    add_sensor_noise = False
    sensor_noise_std = 0.05 
    add_process_noise = False
    process_noise_std = 0.005 
    seed_value = 42

    day_tf = {'K': 1.133, 'tau': 2833.82, 'delay': 0.05}
    night_tf = {'K': 2.049, 'tau': 4499.996, 'delay': 0.05}
    
    weights = (0.0, 0.0, 0.0, 0.0)
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
    
    t_matlab, do_matlab, u_matlab, _ = run_matlab_baseline_simulation(
        matlab_tf=matlab_plant, kp=matlab_kp, ki=matlab_ki, 
        target_setpoint=target_do_setpoint, sim_duration=sim_duration, dt=dt_step,
        add_sensor_noise=add_sensor_noise, sensor_noise_std=sensor_noise_std,
        add_process_noise=add_process_noise, process_noise_std=process_noise_std,
        disturbances=scheduled_disturbances
    )

    day_kp, day_ki = auto_tune_gains("Daytime", day_tf)
    night_kp, night_ki = auto_tune_gains("Nighttime", night_tf)

    random.seed(seed_value) 
    t_non_adaptive, do_non_adaptive, u_non_adaptive, _ = run_simulation(
        is_adaptive=False, day_tf=day_tf, night_tf=night_tf, 
        day_gains=(day_kp, day_ki), night_gains=(night_kp, night_ki),
        target_setpoint=target_do_setpoint, sim_duration=sim_duration, dt=dt_step,
        add_sensor_noise=add_sensor_noise, sensor_noise_std=sensor_noise_std,
        add_process_noise=add_process_noise, process_noise_std=process_noise_std,
        disturbances=scheduled_disturbances, adaptive_weights=weights
    )

    random.seed(seed_value) 
    t_adaptive, do_adaptive, u_adaptive, adaptive_intervals = run_simulation(
        is_adaptive=True, day_tf=day_tf, night_tf=night_tf, 
        day_gains=(day_kp, day_ki), night_gains=(night_kp, night_ki),
        target_setpoint=target_do_setpoint, sim_duration=sim_duration, dt=dt_step,
        add_sensor_noise=add_sensor_noise, sensor_noise_std=sensor_noise_std,
        add_process_noise=add_process_noise, process_noise_std=process_noise_std,
        disturbances=scheduled_disturbances, adaptive_weights=weights
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # PLOT 1: DO Comparison
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    # ----> SHADE RETUNING REGIONS
    added_label = False
    for (start_h, end_h) in adaptive_intervals:
        plt.axvspan(start_h, end_h, color='gold', alpha=0.3, 
                    label='Adaptive Retuning Phase' if not added_label else "")
        added_label = True

    # ----> SHADE DISTURBANCES
    added_dist_label = False
    for d_time in scheduled_disturbances.keys():
        d_time_h = d_time / 3600.0
        plt.axvspan(d_time_h, d_time_h + 0.5, color='salmon', alpha=0.3, 
                    label='Disturbance Applied' if not added_dist_label else "")
        added_dist_label = True

    plt.plot(t_matlab, do_matlab, label='MATLAB Baseline (Hardcoded LTI)', color='black', linestyle=':', linewidth=2, alpha=0.9)
    plt.plot(t_non_adaptive, do_non_adaptive, label='Hardware: Non-Adaptive', color='red', linestyle='--', alpha=0.8)
    plt.plot(t_adaptive, do_adaptive, label='Hardware: Adaptive', color='blue', linewidth=2, alpha=0.8)
    
    plt.axhline(y=target_do_setpoint, color='green', linestyle='-', label=f'Base Setpoint ({target_do_setpoint})')
    plt.axhline(y=target_do_setpoint + 0.2, color='green', linestyle=':', alpha=0.6, label=f'Retune Bump Target (2.2)')
    
    plt.axvline(x=12.0, color='gray', linestyle='-', alpha=0.5)
    
    ax.text(0.25, 1.01, 'DAYTIME', transform=ax.transAxes, fontsize=12, fontweight='bold', alpha=0.6, ha='center', va='bottom')
    ax.text(0.75, 1.01, 'NIGHTTIME', transform=ax.transAxes, fontsize=12, fontweight='bold', alpha=0.6, ha='center', va='bottom')
    
    plt.title('Simulated DO Control: MATLAB Baseline vs Hardware Pipelines', y=1.05)
    plt.xlabel('Time (Hours)')
    plt.ylabel('Dissolved Oxygen (mg/L)')
    plt.ylim(bottom=0) 
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "do_comparison_all_3.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 2: Continuous Control Signals
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 10))
    
    # 1) MATLAB
    plt.subplot(3, 1, 1)
    added_dist_label = False
    for d_time in scheduled_disturbances.keys():
        d_time_h = d_time / 3600.0
        plt.axvspan(d_time_h, d_time_h + 0.5, color='salmon', alpha=0.3, 
                    label='Disturbance Applied' if not added_dist_label else "")
        added_dist_label = True

    plt.plot(t_matlab, u_matlab, color='black', alpha=0.8)
    plt.fill_between(t_matlab, 0, u_matlab, color='black', alpha=0.3, label='Continuous Output')
    plt.title('MATLAB LTI Control Signal (Continuous)')
    plt.ylabel('Duty Cycle (0.0 - 1.0)')
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # 2) Non-Adaptive
    plt.subplot(3, 1, 2)
    added_dist_label = False
    for d_time in scheduled_disturbances.keys():
        d_time_h = d_time / 3600.0
        plt.axvspan(d_time_h, d_time_h + 0.5, color='salmon', alpha=0.3, 
                    label='Disturbance Applied' if not added_dist_label else "")
        added_dist_label = True

    plt.plot(t_non_adaptive, u_non_adaptive, color='red', alpha=0.8)
    plt.fill_between(t_non_adaptive, 0, u_non_adaptive, color='red', alpha=0.3, label='Continuous Output')
    plt.axvline(x=12.0, color='gray', linestyle='-', alpha=0.5)
    plt.title('Hardware Non-Adaptive Control Signal (Continuous)')
    plt.ylabel('Duty Cycle (0.0 - 1.0)')
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # 3) Adaptive
    plt.subplot(3, 1, 3)
    added_dist_label = False
    for d_time in scheduled_disturbances.keys():
        d_time_h = d_time / 3600.0
        plt.axvspan(d_time_h, d_time_h + 0.5, color='salmon', alpha=0.3, 
                    label='Disturbance Applied' if not added_dist_label else "")
        added_dist_label = True

    added_label = False
    for (start_h, end_h) in adaptive_intervals:
        plt.axvspan(start_h, end_h, color='gold', alpha=0.3, 
                    label='Adaptive Retuning Phase' if not added_label else "")
        added_label = True

    plt.plot(t_adaptive, u_adaptive, color='blue', alpha=0.8)
    plt.fill_between(t_adaptive, 0, u_adaptive, color='blue', alpha=0.3, label='Continuous Output')
    plt.axvline(x=12.0, color='gray', linestyle='-', alpha=0.5)
    plt.title('Hardware Adaptive Control Signal (Continuous)')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Duty Cycle (0.0 - 1.0)')
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "continuous_control_signal_all_3.png"), dpi=300)
    plt.close()

    print(f"\n[Success] Simulation complete. Both plots saved to: {output_dir}")

if __name__ == "__main__":
    main()