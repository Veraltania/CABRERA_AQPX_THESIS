import collections
import matplotlib.pyplot as plt
from unittest.mock import patch

# ==========================================
# HARDWARE PIPELINE IMPORTS
# ==========================================
from Evolutionary_Algorithm_Testing.de.de_optimizer import DEOptimizer
from Evolutionary_Algorithm_Testing.ga.ga_optimizer import GAOptimizer
from Evolutionary_Algorithm_Testing.pso.pso_optimizer import PSOOptimizer

from Hardware_Pipeline.controllers import DOController
from Hardware_Pipeline.tuning_strategies import AdaptiveTuningStrategy

# ==========================================
# 1. MOCK HARDWARE BRIDGES FOR MIL
# ==========================================
class MockActuator:
    """Mock actuator to catch the duty cycle set by DOController.process()"""
    def __init__(self):
        self.duty_cycle = 0.0
    
    def set_duty_cycle(self, val):
        self.duty_cycle = val

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
        self._update_delay_buffer()

    def switch_to_night(self): 
        self.active_tf = self.night_tf
        self._update_delay_buffer()

    def _update_delay_buffer(self):
        delay_steps = int(self.active_tf['delay'] / self.dt)
        if delay_steps > self.u_buffer.maxlen:
            new_buffer = collections.deque([0.0]*delay_steps, maxlen=delay_steps)
            for val in self.u_buffer:
                new_buffer.append(val)
            self.u_buffer = new_buffer

    def step(self, u_t):
        u_t = max(0.0, min(1.0, float(u_t)))
        self.u_buffer.append(u_t)
        u_delayed = self.u_buffer[0]

        K = self.active_tf['tf_num'][0]
        T = self.active_tf['tf_den'][0]

        dy = (K * u_delayed - (self.current_do - self.baseline_do)) / T
        self.current_do += dy * self.dt

        for d_time, d_val in self.disturbances.items():
            if self.sim_time >= d_time and d_time not in self.applied_disturbances:
                self.current_do += d_val
                self.applied_disturbances.add(d_time)
                print(f"[{self.sim_time/3600:.2f}h] Disturbance applied: {d_val} DO")

        self.current_do = max(0.0, self.current_do)
        self.sim_time += self.dt
        return self.current_do

# ==========================================
# 2. MAIN SIMULATION LOOP
# ==========================================
def run_mil_simulation():
    # --- Local Tuning Configuration ---
    active_optimizer_name = 'DE'  
    cost_weights = [1.0, 1.5, 1.0, 0.5] # [Error, Effort, Overshoot, Rise Time]

    optimizer_config = {
        'population_size': 15,
        'max_iters': 10,
        'patience_limit': 3,
        'improvement_tol': 1e-4,
        'n_rounds': 1,           
        'weights': cost_weights, 
        'output_folder': f"mil_tuning_logs_hp_{active_optimizer_name.lower()}"
    }

    # --- Plant Parameters ---
    day_tf = {'tf_num': [2.5], 'tf_den': [1200.0], 'delay': 60.0}
    night_tf = {'tf_num': [1.8], 'tf_den': [1800.0], 'delay': 120.0}
    
    scheduled_disturbances = {
        4 * 3600: -1.5,
        16 * 3600: -2.0
    }

    dt = 5.0
    plant_static = VirtualAquaculturePlant(day_tf, night_tf, dt=dt, disturbances=scheduled_disturbances.copy())
    plant_adaptive = VirtualAquaculturePlant(day_tf, night_tf, dt=dt, disturbances=scheduled_disturbances.copy())

    # --- Controller Setup ---
    actuator_static = MockActuator()
    actuator_adaptive = MockActuator()

    ctrl_static = DOController(name="DO_Static", strategy_manager=None, actuator=actuator_static)
    ctrl_adaptive = DOController(name="DO_Adaptive", strategy_manager=None, actuator=actuator_adaptive)

    # Override standard Initial Parameters cleanly
    sp_value = 6.0
    initial_kp, initial_ki = 0.5, 0.005
    for ctrl in [ctrl_static, ctrl_adaptive]:
        ctrl.setpoint = sp_value
        ctrl.kp = initial_kp
        ctrl.ki = initial_ki
        ctrl.foptd_gain = day_tf['tf_num'][0]
        ctrl.foptd_tau = day_tf['tf_den'][0]
        ctrl.foptd_delay = day_tf['delay']

    # --- Optimizer & Strategy Setup ---
    tf_params_adaptive = {
        'tf_num': day_tf['tf_num'],
        'tf_den': day_tf['tf_den'],
        'computed_delay': day_tf['delay'],
        'avg_rise_time': day_tf['tf_den'][0] * 1.5, 
        'max_kp': 5.0
    }

    optimizer_map = {'DE': DEOptimizer, 'GA': GAOptimizer, 'PSO': PSOOptimizer}
    SelectedOptimizer = optimizer_map.get(active_optimizer_name, DEOptimizer)
    optimizer_instance = SelectedOptimizer(optimizer_config, tf_params_adaptive)

    strat_adaptive = AdaptiveTuningStrategy(window_duration=3600)

    # --- Simulation Variables ---
    sim_hours = 24.0
    total_steps = int((sim_hours * 3600) / dt)

    t_vals = []
    y_static, u_static_vals = [], []
    y_adaptive, u_adaptive_vals = [], []

    adaptive_intervals = []
    energy_static, energy_adaptive = 0.0, 0.0
    sim_time_sec = 0.0

    print("Starting 24-Hour Simulation Loop...")

    # We MUST patch time.time() so the PI controllers think time is actually advancing!
    with patch('time.time') as mock_time:
        mock_time.return_value = sim_time_sec

        for step in range(total_steps):
            sim_time_sec += dt
            mock_time.return_value = sim_time_sec  # Advance mock time
            current_time_hr = sim_time_sec / 3600.0

            # Day / Night Switching
            if abs(current_time_hr - 12.0) < (dt / 7200.0): # Avoid floating point misses
                print(f"[{current_time_hr:.2f}h] Transition to Night Dynamics")
                plant_static.switch_to_night()
                plant_adaptive.switch_to_night()
                
                # Update optimizer's active plant map dynamically
                optimizer_instance.K_plant = night_tf['tf_num'][0]
                optimizer_instance.T_plant = night_tf['tf_den'][0]
                optimizer_instance.delay = night_tf['delay']
                optimizer_instance.avg_rise_time = night_tf['tf_den'][0] * 1.5

            # 1. Read DO Plant Sensors
            do_s = plant_static.current_do
            do_a = plant_adaptive.current_do

            # 2. Process via Backend Controllers (Expects Payload Dict)
            ctrl_static.process({'mcp_wq': {'do': do_s}})
            ctrl_adaptive.process({'mcp_wq': {'do': do_a}})

            # Retrieve processed control signals
            u_s = actuator_static.duty_cycle
            u_a = actuator_adaptive.duty_cycle

            # 3. Evaluate Strategy Performance & Retune
            error_a = ctrl_adaptive.setpoint - do_a
            
            # evaluate_performance returns True if retuning is required
            needs_tuning = strat_adaptive.evaluate_performance(ctrl_adaptive, error_a, dt)
            
            if needs_tuning:
                print(f"[{current_time_hr:.2f}h] Poor performance detected! Retuning triggered...")
                adaptive_intervals.append([current_time_hr, current_time_hr + 0.1]) # visual marker width
                
                # Run the Optimizer directly
                best_sol, _ = optimizer_instance.optimize_round(1)
                
                # best_sol is a tuple: (Kp, Ki, Total_Cost, Raw_Costs)
                ctrl_adaptive.kp = best_sol[0]
                ctrl_adaptive.ki = best_sol[1]
                
                print(f"[{current_time_hr:.2f}h] Retuning Complete. New Gains: Kp={ctrl_adaptive.kp:.3f}, Ki={ctrl_adaptive.ki:.4f}")
                
                # Reset strategy flags if your backend requires it
                strat_adaptive.abs_error_sum = 0.0
                strat_adaptive.window_timer = 0.0

            # 4. Step Physical Plants
            plant_static.step(u_s)
            plant_adaptive.step(u_a)

            # Logging
            t_vals.append(current_time_hr)
            y_static.append(do_s)
            u_static_vals.append(u_s)
            energy_static += u_s * dt

            y_adaptive.append(do_a)
            u_adaptive_vals.append(u_a)
            energy_adaptive += u_a * dt

    print("Simulation Complete!")
    print(f"Total Energy Index (Static):   {energy_static:.2f}")
    print(f"Total Energy Index (Adaptive): {energy_adaptive:.2f}")

    # ==========================================
    # 3. PLOTTING RESULTS
    # ==========================================
    plt.figure(figsize=(12, 10))

    # Subplot 1: DO Comparison
    plt.subplot(3, 1, 1)
    plt.plot(t_vals, y_static, label='Static DO', color='red', alpha=0.7)
    plt.plot(t_vals, y_adaptive, label='Adaptive DO', color='blue', alpha=0.8)
    plt.axhline(y=sp_value, color='green', linestyle='--', label='Setpoint')
    plt.axvline(x=12.0, color='black', linestyle=':', label='Day/Night Shift')
    plt.title('Dissolved Oxygen Concentration (Continuous Control)')
    plt.ylabel('DO (mg/L)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # Subplot 2: Static Effort
    plt.subplot(3, 1, 2)
    plt.plot(t_vals, u_static_vals, color='red', alpha=0.7)
    plt.fill_between(t_vals, 0, u_static_vals, color='red', alpha=0.2, label=f'Static (Energy: {energy_static:.2f})')
    plt.title('Static Continuous Control Effort')
    plt.ylabel('Aerator Rate (0.0 - 1.0)')
    plt.ylim(0, 1.1)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # Subplot 3: Adaptive Effort
    plt.subplot(3, 1, 3)
    
    # Highlight Disturbances
    for d_time in scheduled_disturbances.keys():
        d_time_h = d_time / 3600.0
        plt.axvspan(d_time_h, d_time_h + 0.5, color='salmon', alpha=0.3, label='Disturbance')

    # Highlight Retuning phases
    for (start_h, end_h) in adaptive_intervals:
        plt.axvspan(start_h, end_h, color='gold', alpha=0.3, label='Adaptive Retuning Phase')

    plt.plot(t_vals, u_adaptive_vals, color='blue', alpha=0.8)
    plt.fill_between(t_vals, 0, u_adaptive_vals, color='blue', alpha=0.2, label=f'Adaptive (Energy: {energy_adaptive:.2f})')
    plt.title('Adaptive Continuous Control Effort')
    plt.ylabel('Aerator Rate (0.0 - 1.0)')
    plt.xlabel('Time (Hours)')
    plt.ylim(0, 1.1)
    
    # Deduplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mil_continuous_results_comparison.png', dpi=300)
    print("Saved plot to 'mil_continuous_results_comparison.png'.")
    plt.show()

if __name__ == "__main__":
    run_mil_simulation()