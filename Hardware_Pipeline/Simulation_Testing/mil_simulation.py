import sys
import os
import collections
import time
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
        self.current_do = 6.0 
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

def run_simulation(is_adaptive, day_tf, night_tf, day_gains, night_gains, target_setpoint, sim_duration=86400, dt=5.0):
    plant = VirtualAquaculturePlant(day_tf, night_tf, dt=dt)
    actuator = VirtualActuator()
    manager = MockStrategyManager()
    
    controller = DOController(name="Sim-DO", strategy_manager=manager, actuator=actuator)
    controller.setpoint = target_setpoint
    controller.kp = day_gains[0]
    controller.ki = day_gains[1]
    
    time_history = []
    do_history = []
    u_history = []

    virtual_time = 0.0 

    print(f"\n--- Starting {'ADAPTIVE' if is_adaptive else 'NON-ADAPTIVE'} Simulation ---")
    
    with patch('time.time', side_effect=lambda: virtual_time):
        for t in range(0, sim_duration, int(dt)):
            
            if t == 43200:
                print(f"[Simulation] t={t}s: Sun has set. Switching to Night Transfer Function.")
                plant.switch_to_night()
                if is_adaptive:
                    print(f"[Simulation] Adaptive Mode: Swapping PID gains to Night optimal.")
                    controller.kp = night_gains[0]
                    controller.ki = night_gains[1]

            current_do = plant.current_do
            fake_payload = {'mcp_wq': {'do': current_do}}
            
            controller.process(fake_payload)
            plant.step(actuator.duty_cycle)
            
            time_history.append(t / 3600.0) 
            do_history.append(current_do)
            u_history.append(actuator.duty_cycle)
            
            virtual_time += dt 

    return time_history, do_history, u_history


# ==========================================
# 4. MAIN EXECUTION & PLOTTING
# ==========================================

if __name__ == "__main__":
    OUTPUT_DIR_NAME = "simulation_graphs"  
    TARGET_DO_SETPOINT = 2.2
    SIM_DURATION = 86400 
    DT_STEP = 5.0
    PWM_WINDOW_MINUTES = 30 # For generating the ON/OFF signal

    day_tf = {'K': 1.346, 'tau': 1551.955, 'delay': 104.469}
    night_tf = {'K': 2.355, 'tau': 3083.590, 'delay': 0.05}

    day_kp, day_ki = auto_tune_gains("Daytime", day_tf)
    night_kp, night_ki = auto_tune_gains("Nighttime", night_tf)

    print(f"\nRunning Baseline (Non-Adaptive) Model with Setpoint {TARGET_DO_SETPOINT}...")
    t_non_adaptive, do_non_adaptive, u_non_adaptive = run_simulation(
        is_adaptive=False, day_tf=day_tf, night_tf=night_tf, 
        day_gains=(day_kp, day_ki), night_gains=(night_kp, night_ki),
        target_setpoint=TARGET_DO_SETPOINT, sim_duration=SIM_DURATION, dt=DT_STEP
    )

    print(f"\nRunning Proposed (Adaptive) Model with Setpoint {TARGET_DO_SETPOINT}...")
    t_adaptive, do_adaptive, u_adaptive = run_simulation(
        is_adaptive=True, day_tf=day_tf, night_tf=night_tf, 
        day_gains=(day_kp, day_ki), night_gains=(night_kp, night_ki),
        target_setpoint=TARGET_DO_SETPOINT, sim_duration=SIM_DURATION, dt=DT_STEP
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, OUTPUT_DIR_NAME)
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # PLOT 1: DO Comparison
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(t_non_adaptive, do_non_adaptive, label='Non-Adaptive', color='red', linestyle='--')
    plt.plot(t_adaptive, do_adaptive, label='Adaptive', color='blue', linewidth=2)
    plt.axhline(y=TARGET_DO_SETPOINT, color='green', linestyle=':', label=f'Target Setpoint ({TARGET_DO_SETPOINT})')
    plt.axvline(x=12.0, color='gray', linestyle='-', alpha=0.5)
    plt.text(5, max(max(do_non_adaptive), max(do_adaptive)) - 0.5, 'DAYTIME', fontsize=12, fontweight='bold', alpha=0.6)
    plt.text(17, max(max(do_non_adaptive), max(do_adaptive)) - 0.5, 'NIGHTTIME', fontsize=12, fontweight='bold', alpha=0.6)
    
    plt.title('Simulated DO Control: Adaptive vs Non-Adaptive PID')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Dissolved Oxygen (mg/L)')
    plt.ylim(bottom=0) 
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    do_save_path = os.path.join(output_dir, "do_adaptive_vs_non_adaptive.png")
    plt.savefig(do_save_path, dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 2: Literal ON/OFF Control Signal
    # ---------------------------------------------------------
    def generate_binary_pwm(t_hist_hours, u_hist, window_minutes):
        """Translates the duty cycle (0-1) into an actual ON (1) / OFF (0) signal."""
        window_hours = window_minutes / 60.0
        binary_signal = []
        for t, u in zip(t_hist_hours, u_hist):
            # Calculate where we are in the current PWM window
            time_in_window = t % window_hours
            
            # If our current time in the window is less than the active ON-duration, aerator is ON
            if time_in_window <= (u * window_hours):
                binary_signal.append(1)
            else:
                binary_signal.append(0)
        return binary_signal

    # Convert continuous duty cycles to discrete 1/0 square waves
    na_on_off = generate_binary_pwm(t_non_adaptive, u_non_adaptive, PWM_WINDOW_MINUTES)
    a_on_off = generate_binary_pwm(t_adaptive, u_adaptive, PWM_WINDOW_MINUTES)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)

    # Subplot 1: Non-Adaptive
    # Using fill_between to create clear visual "blocks" of ON time
    ax1.fill_between(t_non_adaptive, 0, na_on_off, color='red', step='post', alpha=0.7)
    ax1.axvline(x=12.0, color='gray', linestyle='-', alpha=0.8, linewidth=2)
    ax1.set_title(f'Non-Adaptive Aerator State (Literal ON/OFF with {PWM_WINDOW_MINUTES}-min Window)')
    ax1.set_ylabel('Relay State\n(1 = ON, 0 = OFF)')
    ax1.set_yticks([0, 1])
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.text(5, 1.1, 'DAYTIME', fontsize=10, fontweight='bold', alpha=0.6)
    ax1.text(17, 1.1, 'NIGHTTIME', fontsize=10, fontweight='bold', alpha=0.6)

    # Subplot 2: Adaptive
    ax2.fill_between(t_adaptive, 0, a_on_off, color='blue', step='post', alpha=0.7)
    ax2.axvline(x=12.0, color='gray', linestyle='-', alpha=0.8, linewidth=2)
    ax2.set_title(f'Adaptive Aerator State (Literal ON/OFF with {PWM_WINDOW_MINUTES}-min Window)')
    ax2.set_xlabel('Time (Hours)')
    ax2.set_ylabel('Relay State\n(1 = ON, 0 = OFF)')
    ax2.set_yticks([0, 1])
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Force Y-axis to slightly pad the 0 to 1 range so blocks are fully visible
    plt.ylim(-0.1, 1.3)
    plt.tight_layout()
    
    power_save_path = os.path.join(output_dir, "aerator_on_off_state.png")
    plt.savefig(power_save_path, dpi=300)
    plt.close()

    print(f"\n[Success] DO Comparison Plot saved to {do_save_path}")
    print(f"[Success] Literal ON/OFF Plot saved to {power_save_path}")