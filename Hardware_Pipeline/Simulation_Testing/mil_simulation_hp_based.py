import sys
import os
import collections
import time
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from unittest.mock import patch
import numpy as np
from scipy.signal import savgol_filter  # Added for pre-filtering noisy bump data

# ==========================================
# HARDWARE PIPELINE IMPORTS
# ==========================================
from Evolutionary_Algorithm_Testing.de.de_optimizer import DEOptimizer
from Transfer_Function_Modeling.closed_loop_fitter import fit_closed_loop_fopdt
from Hardware_Pipeline.controllers import DOController
from Hardware_Pipeline.tuning_strategies import AdaptiveTuningStrategy, StaticTuningStrategy

# ==========================================
# 1. VIRTUAL HARDWARE CLASSES
# ==========================================
class VirtualAquaculturePlant:
    def __init__(self, day_tf, night_tf, dt=5.0):
        self.day_tf = day_tf
        self.night_tf = night_tf
        self.dt = dt
        self.baseline_do = 1.0  
        self.current_do = 1.5
        self.active_tf = self.day_tf
        delay_steps = int(self.active_tf['delay'] / self.dt)
        self.u_buffer = collections.deque([0.0] * max(1, delay_steps), maxlen=max(1, delay_steps))

    def switch_to_day(self): self.active_tf = self.day_tf
    def switch_to_night(self): self.active_tf = self.night_tf

    def step(self, u):
        K = self.active_tf['K']
        tau = self.active_tf['tau']
        delayed_u = self.u_buffer.popleft()
        self.u_buffer.append(u)
        deviation_do = self.current_do - self.baseline_do
        dy = (self.dt / tau) * ((K * delayed_u) - deviation_do)
        self.current_do += dy
        return self.current_do

class VirtualActuator:
    def __init__(self): self.duty_cycle = 0.0
    def set_duty_cycle(self, duty_cycle): self.duty_cycle = max(0.0, min(1.0, duty_cycle))

class VirtualStrategyManager:
    def __init__(self, is_adaptive):
        if is_adaptive: self.strategy = AdaptiveTuningStrategy(window_duration=1800)
        else: self.strategy = StaticTuningStrategy()
    def get_active_strategy(self): return self.strategy

class VirtualClock:
    def __init__(self, start_time=0.0): self.current_time = start_time
    def tick(self, dt): self.current_time += dt
    def get_time(self): return self.current_time

# ==========================================
# 2. AUTO-TUNING
# ==========================================
def auto_tune_gains(tf_name, tf_config):
    print(f"\n[Auto-Tuner] Running DE Optimization for {tf_name} Phase...")
    tf_params = {
        'tf_num': [tf_config['K']], 
        'tf_den': [tf_config['tau'], 1], 
        'tf_delay': tf_config['delay'],
        'tf_n_pade': 2, 
        'computed_delay': tf_config['delay'], 
        'is_reverse_acting': False, 
        'max_kp': 3.0  
    }
    config = {'patience': 20, 'tol': 1e-4, 'mutation': (0.5, 1.0), 'recombination': 0.745, 'strategy': 'best1bin', 'population_size': 30, 'n_rounds': 1}
    optimizer = DEOptimizer(config, tf_params)
    best_kp, best_ki, best_cost, gens, history = optimizer.optimize_round(round_num=1)
    print(f"[Auto-Tuner] {tf_name} Optimal Gains Found -> Kp: {best_kp:.4f}, Ki: {best_ki:.4f}")
    return best_kp, best_ki

# ==========================================
# 3. PURE MATLAB BASELINE SIMULATION
# ==========================================
def run_matlab_baseline_simulation(matlab_tf, kp, ki, target_setpoint, sim_duration=86400, dt=5.0, add_sensor_noise=False, sensor_noise_std=0.05, add_process_noise=False, process_noise_std=0.005):
    plant = VirtualAquaculturePlant(day_tf=matlab_tf, night_tf=matlab_tf, dt=dt)
    integral_sum = 0.0
    current_time = 0.0
    time_history, do_history, u_history = [], [], []
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
# 4. HARDWARE PIPELINE SIMULATION LOOP
# ==========================================
def run_simulation(is_adaptive, day_tf, night_tf, day_gains, night_gains, target_setpoint, sim_duration=86400, dt=5.0, add_sensor_noise=False, sensor_noise_std=0.05, add_process_noise=False, process_noise_std=0.005):
    plant = VirtualAquaculturePlant(day_tf, night_tf, dt=dt)
    actuator = VirtualActuator()
    manager = VirtualStrategyManager(is_adaptive)
    v_clock = VirtualClock()
    
    controller = DOController(name="Sim-DO", strategy_manager=manager, actuator=actuator)
    controller.setpoint = target_setpoint
    controller.kp = day_gains[0]
    controller.ki = day_gains[1]
    
    # meas_do_history added to provide the fitter with true, noisy observed data rather than the oracle DO
    time_history, do_history, meas_do_history, u_history = [], [], [], []
    night_switched = False
    
    print(f"\n--- Starting Hardware Pipeline {'ADAPTIVE' if is_adaptive else 'NON-ADAPTIVE'} Simulation ---")

    with patch('Hardware_Pipeline.controllers.time.time', side_effect=v_clock.get_time), \
         patch('Hardware_Pipeline.controllers.datetime') as mock_dt:
        mock_dt.now.side_effect = lambda: datetime(2025, 1, 1) + timedelta(seconds=v_clock.get_time())
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

        def sim_retune_full():
            if controller.retune_thread_active: return
            controller.retune_thread_active = True
            
            old_setpoint = controller.setpoint
            step_size = 0.5 
            new_setpoint = old_setpoint + step_size
            
            print(f"\n[MIL Sim] 🛑 Adaptive Retune Triggered at {v_clock.get_time()/3600.0:.2f} hrs")
            controller.start_retuning_session(controller.target_column)
            controller.setpoint = new_setpoint
            step_start_t = v_clock.get_time()
            
            # PRE-FILL baseline logic uses the MEASURED data to match real-world sensor conditions
            baseline_samples = 30
            if len(time_history) >= baseline_samples:
                bump_t = [t * 3600.0 for t in time_history[-baseline_samples:]]
                bump_u = u_history[-baseline_samples:]
                bump_y = meas_do_history[-baseline_samples:]
            else:
                bump_t, bump_u, bump_y = [], [], []
            
            tolerance = abs(step_size) * 0.05
            rise_time = 0
            max_steps = int(3600 / dt) 
            reached = False
            
            # 1. Step response phase
            for _ in range(max_steps):
                v_clock.tick(dt)
                curr_do = plant.current_do
                meas_do = curr_do + (random.gauss(0, sensor_noise_std) if add_sensor_noise else 0)
                if add_process_noise: curr_do += random.gauss(0, process_noise_std)
                plant.current_do = curr_do
                
                error = new_setpoint - meas_do
                tentative_int = controller.integral_sum + (error * dt)
                pi_out = (controller.kp * error) + (controller.ki * tentative_int)
                
                # Robustly enforce 0 to 1 bounds, mirroring hardware rather than relying on abstract controller vars
                clamped_out = max(0.0, min(1.0, pi_out))
                if 0.0 < pi_out < 1.0: controller.integral_sum = tentative_int
                
                controller._record_retuning_data(meas_do)
                actuator.set_duty_cycle(clamped_out)
                plant.step(actuator.duty_cycle)
                
                time_history.append(v_clock.get_time() / 3600.0)
                do_history.append(plant.current_do)
                meas_do_history.append(meas_do)
                u_history.append(actuator.duty_cycle)
                
                bump_t.append(v_clock.get_time())
                bump_u.append(actuator.duty_cycle)
                bump_y.append(meas_do)  # Record strictly measured values
                
                if not reached and abs(meas_do - new_setpoint) <= tolerance:
                    rise_time = v_clock.get_time() - step_start_t
                    reached = True
                    break
                    
            # 2. Tail stabilizing phase
            if reached:
                tail_time_secs = 7200 
                for _ in range(int(tail_time_secs / dt)):
                    v_clock.tick(dt)
                    curr_do = plant.current_do
                    meas_do = curr_do + (random.gauss(0, sensor_noise_std) if add_sensor_noise else 0)
                    if add_process_noise: curr_do += random.gauss(0, process_noise_std)
                    plant.current_do = curr_do
                    
                    error = new_setpoint - meas_do
                    tentative_int = controller.integral_sum + (error * dt)
                    pi_out = (controller.kp * error) + (controller.ki * tentative_int)
                    
                    clamped_out = max(0.0, min(1.0, pi_out))
                    if 0.0 < pi_out < 1.0: controller.integral_sum = tentative_int
                    
                    controller._record_retuning_data(meas_do)
                    actuator.set_duty_cycle(clamped_out)
                    plant.step(actuator.duty_cycle)
                    
                    time_history.append(v_clock.get_time() / 3600.0)
                    do_history.append(plant.current_do)
                    meas_do_history.append(meas_do)
                    u_history.append(actuator.duty_cycle)
                    
                    bump_t.append(v_clock.get_time())
                    bump_u.append(actuator.duty_cycle)
                    bump_y.append(meas_do)  # Record strictly measured values
            
            controller.stop_retuning_session()
            
            try:
                # IMPORTANT: FOPDT relies on baseline initialization y[0]. Raw sensor noise can spike y[0] 
                # and skew the whole fit. Smooth the curve purely for the fitter to deduce clean parameter estimates.
                if len(bump_y) > 15:
                    bump_y_fit = savgol_filter(bump_y, window_length=15, polyorder=3)
                else:
                    bump_y_fit = bump_y

                ex_K, ex_tau, ex_delay = fit_closed_loop_fopdt(bump_t, bump_u, bump_y_fit)
                safe_delay = max(0.05, ex_delay) 
                safe_tau = max(1.0, ex_tau) 
                
                tf_params = {
                    'tf_num': [ex_K], 'tf_den': [safe_tau, 1], 'tf_delay': safe_delay,        
                    'tf_n_pade': 2, 'computed_delay': safe_delay,
                    'is_reverse_acting': ex_K < 0, 
                    'max_kp': 3.0  
                }
                
                de_config = {
                    'population_size': 30, 'max_iters': 20, 'patience_limit': 5,
                    'mutation': (0.5, 1.0), 'recombination': 0.745, 'strategy': 'best1bin', 'n_rounds': 1
                }
                
                optimizer = DEOptimizer(config=de_config, tf_params=tf_params)
                best_Kp, best_Ki, cost, _, _ = optimizer.optimize_round(round_num=1)
                controller.update_tuning_parameters(best_Kp, best_Ki, ex_K, safe_tau, safe_delay)
                
            except Exception as e:
                print(f"[MIL Sim] Pipeline failed: {e}")
                
            controller.setpoint = old_setpoint
            controller.retune_thread_active = False

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
            meas_do_history.append(measured_do)
            u_history.append(actuator.duty_cycle)
            
            v_clock.tick(dt)

    return time_history, do_history, u_history


# ==========================================
# 5. MAIN EXECUTION & PLOTTING
# ==========================================
if __name__ == "__main__":
    OUTPUT_DIR_NAME = "simulation_graphs"  
    TARGET_DO_SETPOINT = 1.5
    SIM_DURATION = 86400 
    DT_STEP = 5.0
    PWM_WINDOW_MINUTES = 10 

    ADD_SENSOR_NOISE = True
    SENSOR_NOISE_STD = 0.05 
    ADD_PROCESS_NOISE = True
    PROCESS_NOISE_STD = 0.01
    SEED_VALUE = 42

    day_tf = {'K': 1.133, 'tau': 2833.82, 'delay': 0.05}
    night_tf = {'K': 2.049, 'tau': 4499.996, 'delay': 0.05}

    MATLAB_PLANT = day_tf
    MATLAB_KP = 0.92
    MATLAB_KI = 0.000974

    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    
    t_matlab, do_matlab, u_matlab = run_matlab_baseline_simulation(
        matlab_tf=MATLAB_PLANT, kp=MATLAB_KP, ki=MATLAB_KI, 
        target_setpoint=TARGET_DO_SETPOINT, sim_duration=SIM_DURATION, dt=DT_STEP,
        add_sensor_noise=ADD_SENSOR_NOISE, sensor_noise_std=SENSOR_NOISE_STD,
        add_process_noise=ADD_PROCESS_NOISE, process_noise_std=PROCESS_NOISE_STD
    )

    day_kp, day_ki = auto_tune_gains("Daytime", day_tf)
    night_kp, night_ki = auto_tune_gains("Nighttime", night_tf)

    random.seed(SEED_VALUE) 
    t_non_adaptive, do_non_adaptive, u_non_adaptive = run_simulation(
        is_adaptive=False, day_tf=day_tf, night_tf=night_tf, 
        day_gains=(day_kp, day_ki), night_gains=(night_kp, night_ki),
        target_setpoint=TARGET_DO_SETPOINT, sim_duration=SIM_DURATION, dt=DT_STEP,
        add_sensor_noise=ADD_SENSOR_NOISE, sensor_noise_std=SENSOR_NOISE_STD,
        add_process_noise=ADD_PROCESS_NOISE, process_noise_std=PROCESS_NOISE_STD
    )

    random.seed(SEED_VALUE) 
    t_adaptive, do_adaptive, u_adaptive = run_simulation(
        is_adaptive=True, day_tf=day_tf, night_tf=night_tf, 
        day_gains=(day_kp, day_ki), night_gains=(night_kp, night_ki),
        target_setpoint=TARGET_DO_SETPOINT, sim_duration=SIM_DURATION, dt=DT_STEP,
        add_sensor_noise=ADD_SENSOR_NOISE, sensor_noise_std=SENSOR_NOISE_STD,
        add_process_noise=ADD_PROCESS_NOISE, process_noise_std=PROCESS_NOISE_STD
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, OUTPUT_DIR_NAME)
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(t_matlab, do_matlab, label='MATLAB Baseline', color='black', linestyle=':', linewidth=2, alpha=0.9)
    plt.plot(t_non_adaptive, do_non_adaptive, label='Hardware: Non-Adaptive', color='red', linestyle='--', alpha=0.8)
    plt.plot(t_adaptive, do_adaptive, label='Hardware: Adaptive', color='blue', linewidth=2, alpha=0.8)
    plt.axhline(y=TARGET_DO_SETPOINT, color='green', linestyle='-', label=f'Base Setpoint ({TARGET_DO_SETPOINT})')
    plt.axhline(y=TARGET_DO_SETPOINT + 0.5, color='green', linestyle=':', alpha=0.6, label='Retune Bump Target')
    plt.axvline(x=12.0, color='gray', linestyle='-', alpha=0.5)
    plt.text(5, max(max(do_non_adaptive), max(do_adaptive)) - 0.5, 'DAYTIME', fontsize=12, fontweight='bold', alpha=0.6)
    plt.text(17, max(max(do_non_adaptive), max(do_adaptive)) - 0.5, 'NIGHTTIME', fontsize=12, fontweight='bold', alpha=0.6)
    plt.title('Simulated DO Control: MATLAB Baseline vs Hardware Pipelines')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Dissolved Oxygen (mg/L)')
    plt.ylim(bottom=0) 
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "do_comparison_all_3.png"), dpi=300)
    plt.close()

    print(f"\n[Success] Simulation complete. Plots saved to: {output_dir}")