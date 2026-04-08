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
        
        # The natural DO of the water when the aerator is OFF (0% duty cycle)
        self.baseline_do = 1.0  
        
        self.current_do = 6.0  # Initial starting DO (mg/L)
        self.active_tf = self.day_tf
        
        # Buffer to simulate dead-time (delay) in the physical system
        delay_steps = int(self.active_tf['delay'] / self.dt)
        self.u_buffer = collections.deque([0.0] * max(1, delay_steps), maxlen=max(1, delay_steps))

    def switch_to_day(self):
        self.active_tf = self.day_tf

    def switch_to_night(self):
        self.active_tf = self.night_tf

    def step(self, u):
        """Calculates the next DO value based on control signal u."""
        K = self.active_tf['K']
        tau = self.active_tf['tau']
        
        # Apply the actuator signal after the delay
        delayed_u = self.u_buffer.popleft()
        self.u_buffer.append(u)
        
        # 1. Calculate how far we currently are from the natural baseline
        deviation_do = self.current_do - self.baseline_do
        
        # 2. Apply FOPDT integration to the deviation, NOT the absolute DO
        dy = (self.dt / tau) * ((K * delayed_u) - deviation_do)
        
        # 3. Update the absolute current DO
        self.current_do += dy
        
        return self.current_do


class VirtualActuator:
    """Mocks the hardware relay, simply storing the duty cycle."""
    def __init__(self):
        self.duty_cycle = 0.0
        
    def set_duty_cycle(self, duty_cycle):
        # Clamped to 1.0 to perfectly match the 'max_out = 1.0' in your controllers.py
        self.duty_cycle = max(0.0, min(1.0, duty_cycle))


class MockStrategyManager:
    """Bypasses the complex time-scheduling logic for forced simulation testing."""
    def get_active_strategy(self):
        return None  


# ==========================================
# 2. AUTO-TUNING INTEGRATION
# ==========================================

def auto_tune_gains(tf_name, tf_config):
    """
    Calculates the optimal Kp and Ki using your DEOptimizer.
    """
    print(f"\n[Auto-Tuner] Running DE Optimization for {tf_name} Phase...")
    
    # Format parameters exactly as de_optimizer and ea_optimizer expect
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

    # Initialize the optimizer
    optimizer = DEOptimizer(config, tf_params)
    
    # Unpack the 5-element tuple returned by optimize_round()
    best_kp, best_ki, best_cost, gens, history = optimizer.optimize_round(round_num=1)
    
    print(f"[Auto-Tuner] {tf_name} Optimal Gains Found -> Kp: {best_kp:.4f}, Ki: {best_ki:.4f} (Cost: {best_cost:.4f})")
    return best_kp, best_ki


# ==========================================
# 3. CORE SIMULATION LOOP
# ==========================================

def run_simulation(is_adaptive, day_tf, night_tf, day_gains, night_gains, target_setpoint, sim_duration=86400, dt=5.0):
    """Runs a complete timeline simulation. Returns time and DO history."""
    
    plant = VirtualAquaculturePlant(day_tf, night_tf, dt=dt)
    actuator = VirtualActuator()
    manager = MockStrategyManager()
    
    # Initialize your actual Controller from controllers.py
    controller = DOController(
        name="Sim-DO", 
        strategy_manager=manager, 
        actuator=actuator
    )
    
    # Use the unified target setpoint
    controller.setpoint = target_setpoint
    controller.kp = day_gains[0]
    controller.ki = day_gains[1]
    
    time_history = []
    do_history = []

    # Initialize a virtual clock starting at 0
    virtual_time = 0.0 

    print(f"\n--- Starting {'ADAPTIVE' if is_adaptive else 'NON-ADAPTIVE'} Simulation ---")
    
    # Patch time.time() so dt calculations work instantly
    with patch('time.time', side_effect=lambda: virtual_time):
        
        for t in range(0, sim_duration, int(dt)):
            
            # At 12 hours (43200 seconds), switch the environment to night
            if t == 43200:
                print(f"[Simulation] t={t}s: Sun has set. Switching to Night Transfer Function.")
                plant.switch_to_night()
                
                # If adaptive strategy is enabled, swap the PID gains automatically
                if is_adaptive:
                    print(f"[Simulation] Adaptive Mode: Swapping PID gains to Night optimal.")
                    controller.kp = night_gains[0]
                    controller.ki = night_gains[1]
                else:
                    print(f"[Simulation] Non-Adaptive Mode: Maintaining Day PID gains.")

            # 1. Read sensor (Virtual Plant)
            current_do = plant.current_do
            fake_payload = {'mcp_wq': {'do': current_do}}
            
            # 2. Execute controller logic
            controller.process(fake_payload)
            
            # 3. Apply control signal to Plant
            plant.step(actuator.duty_cycle)
            
            # 4. Log state
            time_history.append(t / 3600.0) # Store as Hours
            do_history.append(current_do)
            
            # Advance virtual clock
            virtual_time += dt 

    return time_history, do_history


# ==========================================
# 4. MAIN EXECUTION & PLOTTING
# ==========================================

if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    # Directory will be created relative to where this python file is physically located
    OUTPUT_DIR_NAME = "simulation_graphs"  
    
    # Set to 2.2 because physical Day maximum is 2.346 (Baseline 1.0 + K 1.346)
    TARGET_DO_SETPOINT = 2.2               
    # ---------------------

    # Define our simulated environment physics (Derived from open-loop tests)
    day_tf = {'K': 1.346, 'tau': 1551.955, 'delay': 104.469}
    night_tf = {'K': 2.355, 'tau': 3083.590, 'delay': 0.05}

    # 1. Auto-compute optimal gains using DEOptimizer
    day_kp, day_ki = auto_tune_gains("Daytime", day_tf)
    night_kp, night_ki = auto_tune_gains("Nighttime", night_tf)

    # 2. Run Simulations
    print(f"\nRunning Baseline (Non-Adaptive) Model with Setpoint {TARGET_DO_SETPOINT}...")
    t_non_adaptive, do_non_adaptive = run_simulation(
        is_adaptive=False, 
        day_tf=day_tf, night_tf=night_tf, 
        day_gains=(day_kp, day_ki), night_gains=(night_kp, night_ki),
        target_setpoint=TARGET_DO_SETPOINT
    )

    print(f"\nRunning Proposed (Adaptive) Model with Setpoint {TARGET_DO_SETPOINT}...")
    t_adaptive, do_adaptive = run_simulation(
        is_adaptive=True, 
        day_tf=day_tf, night_tf=night_tf, 
        day_gains=(day_kp, day_ki), night_gains=(night_kp, night_ki),
        target_setpoint=TARGET_DO_SETPOINT
    )

    # 3. Generate Comparative Plot
    plt.figure(figsize=(12, 6))
    
    # Plot DO responses
    plt.plot(t_non_adaptive, do_non_adaptive, label='Non-Adaptive (Static Day Gains)', color='red', linestyle='--')
    plt.plot(t_adaptive, do_adaptive, label='Adaptive (Day/Night Dynamic Tuning)', color='blue', linewidth=2)
    
    # Plot Setpoint using the global configuration
    plt.axhline(y=TARGET_DO_SETPOINT, color='green', linestyle=':', label=f'Target DO Setpoint ({TARGET_DO_SETPOINT} mg/L)')
    plt.axvline(x=12.0, color='gray', linestyle='-', alpha=0.5)
    
    # Annotations
    plt.text(5, max(do_non_adaptive) - 0.5, 'DAYTIME', fontsize=12, fontweight='bold', alpha=0.6)
    plt.text(17, max(do_non_adaptive) - 0.5, 'NIGHTTIME', fontsize=12, fontweight='bold', alpha=0.6)
    
    plt.title('Simulated DO Control: Adaptive vs Non-Adaptive PID Strategies')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Dissolved Oxygen (mg/L)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 4. Determine Script Directory & Save
    # Get the directory where this script file lives
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, OUTPUT_DIR_NAME)
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "do_adaptive_vs_non_adaptive.png")
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"\n[Success] Plot successfully saved to {save_path}!")