import sys
import os
import collections
import time
import random
import matplotlib.pyplot as plt
from unittest.mock import patch

# Import your existing pipeline modules
# (Adjust paths if your directory structure differs)
from Evolutionary_Algorithm_Testing.de.de_optimizer import DEOptimizer
from Hardware_Pipeline.controllers import DOController

# ==========================================
# 1. VIRTUAL HARDWARE CLASSES
# ==========================================
class VirtualAquaculturePlant:
    """Simulates the physical DO response using First-Order Plus Dead Time (FOPDT)."""
    def __init__(self, day_tf, night_tf, dt=5.0):
        self.day_tf = day_tf
        self.night_tf = night_tf
        self.dt = dt
        
        self.baseline_do = 1.0  
        self.current_do = 1.0
        self.active_tf = self.day_tf
        
        delay_steps = int(self.active_tf['delay'] / self.dt)
        self.u_buffer = collections.deque([0.0] * max(1, delay_steps), maxlen=max(1, delay_steps))

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
        
        return self.current_do


class VirtualActuator:
    def __init__(self):
        self.duty_cycle = 0.0
        
    def set_duty_cycle(self, duty_cycle):
        self.duty_cycle = max(0.0, min(1.0, duty_cycle))


class MockStrategyManager:
    def get_active_strategy(self):
        return None  


# ==========================================
# 2. AUTO-TUNING INTEGRATION
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
        'max_kp': 100.0
    }

    config = {
        'patience': 20,
        'tol': 1e-4,
        'mutation': (0.5, 1.0),
        'recombination': 0.745,
        'strategy': 'best1bin'
    }

    optimizer = DEOptimizer(config, tf_params)
    best_kp, best_ki, best_cost, gens, history = optimizer.optimize_round(round_num=1)
    
    print(f"[Auto-Tuner] {tf_name} Optimal Gains Found -> Kp: {best_kp:.4f}, Ki: {best_ki:.4f} (Cost: {best_cost:.4f})")
    return best_kp, best_ki


# ==========================================
# 3. CORE SIMULATION LOOP
# ==========================================

def run_simulation(is_adaptive, day_tf, night_tf, day_gains, night_gains, target_setpoint, 
                   sim_duration=86400, dt=5.0, 
                   add_sensor_noise=False, sensor_noise_std=0.05,
                   add_process_noise=False, process_noise_std=0.005):
    
    plant = VirtualAquaculturePlant(day_tf, night_tf, dt=dt)
    actuator = VirtualActuator()
    manager = MockStrategyManager()
    
    controller = DOController(name="Sim-DO", strategy_manager=manager, actuator=actuator)
    controller.setpoint = target_setpoint
    controller.kp = day_gains[0]
    controller.ki = day_gains[1]
    
    time_history = []
    do_history = []       # The TRUE DO of the water
    measured_do_history = [] # The NOISY DO the controller sees
    u_history = []

    virtual_time = 0.0 

    print(f"--- Starting {'ADAPTIVE' if is_adaptive else 'NON-ADAPTIVE'} Simulation ---")
    
    with patch('time.time', side_effect=lambda: virtual_time):
        for t in range(0, sim_duration, int(dt)):
            
            if t == 43200:
                print(f"[Simulation] t={t}s: Sun has set. Switching to Night Transfer Function.")
                plant.switch_to_night()
                if is_adaptive:
                    print(f"[Simulation] Adaptive Mode: Swapping PID gains to Night optimal.")
                    controller.kp = night_gains[0]
                    controller.ki = night_gains[1]

            # 1. Apply Process Noise (Affects actual physical water DO)
            if add_process_noise:
                plant.current_do += random.gauss(0, process_noise_std)

            current_do = plant.current_do
            
            # 2. Apply Sensor Noise (Affects only the reading sent to the controller)
            measured_do = current_do
            if add_sensor_noise:
                measured_do += random.gauss(0, sensor_noise_std)

            fake_payload = {'mcp_wq': {'do': measured_do}}
            
            # Controller acts on the potentially noisy measurement
            controller.process(fake_payload)
            plant.step(actuator.duty_cycle)
            
            # Record keeping
            time_history.append(t / 3600.0) 
            do_history.append(current_do)           # We plot the TRUE DO
            measured_do_history.append(measured_do) # Keep track if you want to plot the noisy sensor later
            u_history.append(actuator.duty_cycle)
            
            virtual_time += dt 

    return time_history, do_history, u_history

def generate_binary_pwm(t_hist_hours, u_hist, window_minutes):
    """Translates the duty cycle (0-1) into an ON/OFF signal using Sample-and-Hold."""
    window_hours = window_minutes / 60.0
    binary_signal = []
    
    current_window_start = 0.0
    # Sample the very first duty cycle to start
    locked_u = u_hist[0] if u_hist else 0.0 
    
    for t, u in zip(t_hist_hours, u_hist):
        # 1. Check if we've crossed into a new window
        if t >= current_window_start + window_hours - 1e-5:
            current_window_start += window_hours
            locked_u = u  # Lock in the new duty cycle
            
        # 2. Calculate time elapsed purely within the CURRENT window
        time_in_window = t - current_window_start
        
        # 3. Compare against the LOCKED duty cycle
        if time_in_window <= (locked_u * window_hours):
            binary_signal.append(1)
        else:
            binary_signal.append(0)
            
    return binary_signal


# ==========================================
# 4. MAIN EXECUTION & PLOTTING
# ==========================================

if __name__ == "__main__":
    OUTPUT_DIR_NAME = "simulation_graphs"  
    TARGET_DO_SETPOINT = 2.0
    SIM_DURATION = 86400 
    DT_STEP = 5.0
    PWM_WINDOW_MINUTES = 30 # For generating the ON/OFF signal

    # --- NOISE CONFIGURATION ---
    ADD_SENSOR_NOISE = True
    SENSOR_NOISE_STD = 0.05
    ADD_PROCESS_NOISE = True
    PROCESS_NOISE_STD = 0.025 
    SEED_VALUE = random.random()

    # --- BASELINE PLANT CONFIGURATION ---
    day_tf = {'K': 1.133, 'tau': 2833.82, 'delay': 0.05}
    night_tf = {'K': 2.049, 'tau': 4499.996, 'delay': 0.05}

    HARDCODED_SCENARIOS = [
        {
            "name": "MATLAB Auto-Tuned (Adaptive)",
            "is_adaptive": True, 
            "day_tf": day_tf,     
            "night_tf": night_tf,
            "day_gains": (0.92, 0.000974),  
            "night_gains": (0.92, 0.000974), 
            "color": "orange",
            "linestyle": "-."
        }
    ]

    # 1. Run DE Auto-Tuner for your AI models
    day_kp, day_ki = auto_tune_gains("Daytime", day_tf)
    night_kp, night_ki = auto_tune_gains("Nighttime", night_tf)

    # 2. Run Non-Adaptive Simulation
    print(f"\nRunning Baseline (Non-Adaptive) Model with Setpoint {TARGET_DO_SETPOINT}...")
    random.seed(SEED_VALUE)  
    t_non_adaptive, do_non_adaptive, u_non_adaptive = run_simulation(
        is_adaptive=False, day_tf=day_tf, night_tf=night_tf, 
        day_gains=(day_kp, day_ki), night_gains=(night_kp, night_ki),
        target_setpoint=TARGET_DO_SETPOINT, sim_duration=SIM_DURATION, dt=DT_STEP,
        add_sensor_noise=ADD_SENSOR_NOISE, sensor_noise_std=SENSOR_NOISE_STD,
        add_process_noise=ADD_PROCESS_NOISE, process_noise_std=PROCESS_NOISE_STD
    )

    # 3. Run Adaptive Simulation
    print(f"\nRunning Proposed (Adaptive) Model with Setpoint {TARGET_DO_SETPOINT}...")
    random.seed(SEED_VALUE)  
    t_adaptive, do_adaptive, u_adaptive = run_simulation(
        is_adaptive=True, day_tf=day_tf, night_tf=night_tf, 
        day_gains=(day_kp, day_ki), night_gains=(night_kp, night_ki),
        target_setpoint=TARGET_DO_SETPOINT, sim_duration=SIM_DURATION, dt=DT_STEP,
        add_sensor_noise=ADD_SENSOR_NOISE, sensor_noise_std=SENSOR_NOISE_STD,
        add_process_noise=ADD_PROCESS_NOISE, process_noise_std=PROCESS_NOISE_STD
    )

    # 4. Run Hardcoded Scenarios (e.g. MATLAB)
    hardcoded_results = []
    for scenario in HARDCODED_SCENARIOS:
        print(f"\nRunning Hardcoded Model: {scenario['name']}...")
        random.seed(SEED_VALUE)
        t_hc, do_hc, u_hc = run_simulation(
            is_adaptive=scenario['is_adaptive'],
            day_tf=scenario['day_tf'],
            night_tf=scenario['night_tf'],
            day_gains=scenario['day_gains'],
            night_gains=scenario['night_gains'],
            target_setpoint=TARGET_DO_SETPOINT, sim_duration=SIM_DURATION, dt=DT_STEP,
            add_sensor_noise=ADD_SENSOR_NOISE, sensor_noise_std=SENSOR_NOISE_STD,
            add_process_noise=ADD_PROCESS_NOISE, process_noise_std=PROCESS_NOISE_STD
        )
        
        # Calculate PWM binary signal and total off hours
        binary_hc = generate_binary_pwm(t_hc, u_hc, PWM_WINDOW_MINUTES)
        off_hours_hc = binary_hc.count(0) * (DT_STEP / 3600.0)
        
        hardcoded_results.append({
            "name": scenario['name'],
            "t": t_hc, "do": do_hc, "binary": binary_hc, "off_hours": off_hours_hc,
            "color": scenario['color'], "linestyle": scenario['linestyle']
        })

    # Setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, OUTPUT_DIR_NAME)
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # PLOT 1: DO Comparison
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(t_non_adaptive, do_non_adaptive, label='DE Non-Adaptive (True DO)', color='red', linestyle='--', alpha=0.8)
    plt.plot(t_adaptive, do_adaptive, label='DE Adaptive (True DO)', color='blue', linewidth=2, alpha=0.8)
    
    # Plot hardcoded scenarios
    for res in hardcoded_results:
        plt.plot(res["t"], res["do"], label=f'{res["name"]} (True DO)', 
                 color=res["color"], linestyle=res["linestyle"], linewidth=2, alpha=0.8)

    plt.axhline(y=TARGET_DO_SETPOINT, color='green', linestyle=':', label=f'Target Setpoint ({TARGET_DO_SETPOINT})')
    plt.axvline(x=12.0, color='gray', linestyle='-', alpha=0.5)
    plt.text(5, max(max(do_non_adaptive), max(do_adaptive)) - 0.5, 'DAYTIME', fontsize=12, fontweight='bold', alpha=0.6)
    plt.text(17, max(max(do_non_adaptive), max(do_adaptive)) - 0.5, 'NIGHTTIME', fontsize=12, fontweight='bold', alpha=0.6)
    
    plt.title('Simulated DO Control: Algorithm Comparison (With Disturbance/Noise)')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Dissolved Oxygen (mg/L)')
    plt.ylim(bottom=0) 
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    do_save_path = os.path.join(output_dir, "do_comparison_all.png")
    plt.savefig(do_save_path, dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 2: Literal ON/OFF Control Signal
    # ---------------------------------------------------------
    binary_non_adaptive = generate_binary_pwm(t_non_adaptive, u_non_adaptive, PWM_WINDOW_MINUTES)
    binary_adaptive = generate_binary_pwm(t_adaptive, u_adaptive, PWM_WINDOW_MINUTES)

    dt_in_hours = DT_STEP / 3600.0
    off_hours_non_adaptive = binary_non_adaptive.count(0) * dt_in_hours
    off_hours_adaptive = binary_adaptive.count(0) * dt_in_hours

    # Dynamically scale plot height based on number of scenarios
    total_subplots = 2 + len(hardcoded_results)
    plt.figure(figsize=(12, 4 * total_subplots))

    # Subplot 1: Non-Adaptive
    plt.subplot(total_subplots, 1, 1)
    label_na = f'DE Non-Adaptive (Total OFF: {off_hours_non_adaptive:.2f} hrs)'
    plt.step(t_non_adaptive, binary_non_adaptive, color='red', where='post', alpha=0.8)
    plt.fill_between(t_non_adaptive, 0, binary_non_adaptive, step='post', color='red', alpha=0.3, label=label_na)
    plt.axvline(x=12.0, color='gray', linestyle='-', alpha=0.5)
    plt.title(f'DE Non-Adaptive ON/OFF Control Signal ({PWM_WINDOW_MINUTES}-min windows)')
    plt.ylabel('State (1=ON, 0=OFF)')
    plt.yticks([0, 1])
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # Subplot 2: Adaptive
    plt.subplot(total_subplots, 1, 2)
    label_a = f'DE Adaptive (Total OFF: {off_hours_adaptive:.2f} hrs)'
    plt.step(t_adaptive, binary_adaptive, color='blue', where='post', alpha=0.8)
    plt.fill_between(t_adaptive, 0, binary_adaptive, step='post', color='blue', alpha=0.3, label=label_a)
    plt.axvline(x=12.0, color='gray', linestyle='-', alpha=0.5)
    plt.title(f'DE Adaptive ON/OFF Control Signal ({PWM_WINDOW_MINUTES}-min windows)')
    plt.ylabel('State (1=ON, 0=OFF)')
    plt.yticks([0, 1])
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # Subplots 3+: Hardcoded Scenarios
    for i, res in enumerate(hardcoded_results):
        plt.subplot(total_subplots, 1, i + 3)
        label_hc = f"{res['name']} (Total OFF: {res['off_hours']:.2f} hrs)"
        plt.step(res["t"], res["binary"], color=res["color"], where='post', alpha=0.8)
        plt.fill_between(res["t"], 0, res["binary"], step='post', color=res["color"], alpha=0.3, label=label_hc)
        plt.axvline(x=12.0, color='gray', linestyle='-', alpha=0.5)
        plt.title(f"{res['name']} ON/OFF Control Signal ({PWM_WINDOW_MINUTES}-min windows)")
        plt.ylabel('State (1=ON, 0=OFF)')
        
        # Only add the X-axis label to the very bottom subplot
        if i == len(hardcoded_results) - 1:
            plt.xlabel('Time (Hours)')
            
        plt.yticks([0, 1])
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    pwm_save_path = os.path.join(output_dir, "pwm_on_off_signal_all.png")
    plt.savefig(pwm_save_path, dpi=300)
    plt.close()

    print(f"\n[Success] Simulation complete. Both plots saved to: {output_dir}")