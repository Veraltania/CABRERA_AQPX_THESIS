import numpy as np
import matplotlib.pyplot as plt
import control as ct
import os

# ==========================================
# 1. SYSTEM CONFIGURATION
# ==========================================
day_cfg = {'K': 1.346, 'tau': 1551.955, 'delay':  104.469}
night_cfg = {'K': 2.36, 'tau': 3083.59, 'delay': 0.05}
half_dur = 86400 / 2
dt = 1.0

def create_fopdt_sys(config):
    """Creates a Transfer Function with Pade approximation for delay."""
    s = ct.TransferFunction.s
    # Base FOPDT: G(s) = K / (tau*s + 1)
    G_base = config['K'] / (config['tau'] * s + 1)
    
    # Delay approximation (2nd order Pade)
    num_delay, den_delay = ct.pade(config['delay'], n=2)
    G_delay = ct.TransferFunction(num_delay, den_delay)
    
    return G_base * G_delay

def create_pi_controller(kp, ki):
    """Creates a PI controller: C(s) = Kp + Ki/s"""
    s = ct.TransferFunction.s
    return kp + (ki / s)

# ==========================================
# 2. SIMULATION LOGIC
# ==========================================
def simulate_phase(plant_cfg, kp, ki, start_time, duration, initial_y=0.0):
    """Simulates a specific phase using control.forced_response"""
    G = create_fopdt_sys(plant_cfg)
    C = create_pi_controller(kp, ki)
    
    # Closed-loop system: T = (C*G) / (1 + C*G)
    sys_cl = ct.feedback(C * G, 1)
    
    # Time vector for this phase
    t_phase = np.arange(0, duration + dt, dt)
    
    # Define setpoint sequence (Step up at 25%, Recover at 75%)
    sp_step = 1.0
    u_ref = np.zeros_like(t_phase)
    u_ref[int(0.25 * len(t_phase)):int(0.75 * len(t_phase))] = sp_step
    
    # Simulate. X0 can be estimated, but for simplicity we use forced_response
    # Note: Non-zero initial conditions in TFs are tricky; for exactness 
    # one usually converts to State Space.
    result = ct.forced_response(sys_cl, T=t_phase, U=u_ref)
    
    return (result.time + start_time), result.outputs, u_ref

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    # Mocked Tuned Values (replace with your DE Optimizer results)
    kp_day, ki_day = 0.5, 0.0005
    kp_night, ki_night = 0.3, 0.0001

    # --- Setup 1: One-Shot (Day gains used for both) ---
    t_d1, y_d1, sp_d1 = simulate_phase(day_cfg, kp_day, ki_day, 0, half_dur)
    t_n1, y_n1, sp_n1 = simulate_phase(night_cfg, kp_day, ki_day, half_dur, half_dur)
    
    # --- Setup 2: Two-Shot (Scheduled gains) ---
    t_d2, y_d2, sp_d2 = simulate_phase(day_cfg, kp_day, ki_day, 0, half_dur)
    t_n2, y_n2, sp_n2 = simulate_phase(night_cfg, kp_night, ki_night, half_dur, half_dur)

    # Combine for plotting
    t_total = np.concatenate([t_d1, t_n1])
    y_os = np.concatenate([y_d1, y_n1])
    y_ts = np.concatenate([y_d2, y_n2])
    sp_total = np.concatenate([sp_d1, sp_n1])

    # --- PLOTTING ---
    plt.figure(figsize=(12, 6))
    plt.axvspan(0, half_dur, color='yellow', alpha=0.1, label='Day')
    plt.axvspan(half_dur, half_dur*2, color='navy', alpha=0.1, label='Night')
    
    plt.plot(t_total, sp_total, 'k--', label='Setpoint', alpha=0.6)
    plt.plot(t_total, y_os, label='One-Shot (Day Gains Only)', color='blue')
    plt.plot(t_total, y_ts, label='Two-Shot (Scheduled)', color='orange')
    
    plt.title("Step Response via Python-Control Library")
    plt.xlabel("Time (s)")
    plt.ylabel("Output")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()