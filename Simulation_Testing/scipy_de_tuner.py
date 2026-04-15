import numpy as np
import control as ct
from scipy.optimize import differential_evolution

def simulate_saturated_pi(plant, kp, ki, t_eval, setpoints, u_min=0.0, u_max=1.0):
    """
    Simulates the closed loop response step-by-step, explicitly clamping 
    the control effort (u) to [u_min, u_max] to model physical actuators like pumps/aerators.
    """
    dt = t_eval[1] - t_eval[0]
    
    sys_d_tf = ct.c2d(plant, dt)
    sys_d_ss = ct.ss(sys_d_tf)  
    A, B, C, D = sys_d_ss.A, sys_d_ss.B, sys_d_ss.C, sys_d_ss.D
    
    n_steps = len(t_eval)
    y_out = np.zeros(n_steps)
    u_out = np.zeros(n_steps)
    
    x = np.zeros((sys_d_ss.nstates, 1))
    integral_error = 0.0
    
    for i in range(n_steps):
        sp = setpoints[i]
        
        current_y = (C @ x)[0, 0]
        y_out[i] = current_y
        
        error = sp - current_y
        integral_error += error * dt
        
        u_raw = (kp * error) + (ki * integral_error)
        
        # Actuator Saturation
        u_clamped = np.clip(u_raw, u_min, u_max)
        u_out[i] = u_clamped
        
        # Anti-windup
        if u_raw != u_clamped and ki != 0:
            integral_error -= error * dt 
            
        if i < n_steps - 1:
            x = A @ x + B * u_clamped
            
    return y_out, u_out

def run_scipy_de_tuner(name, plant, min_kp, max_kp, min_ki, max_ki, tau, delay, weights, 
                       target_sp=1.0, max_overshoot_limit=0.20):
    """Uses SciPy's DE applying the realistic saturated simulation"""
    print(f"\n[Auto-Tuner] Running SciPy DE Optimization: {name}")
    
    T_sim = (tau * 3) + delay
    t_opt = np.linspace(0, T_sim, 1000)
    sp_opt = np.full_like(t_opt, target_sp) 
    
    avg_rise_time = tau * 2.2
    w_iae, w_effort, w_os, w_rt = weights

    def objective(params):
        kp, ki = params
        penalty = 1e20
        
        try:
            y_out, u_out = simulate_saturated_pi(plant, kp, ki, t_opt, sp_opt, u_min=0.0, u_max=1.0)
            
            # Normalize y_out
            y_norm = y_out * np.sign(target_sp)
            
            if np.any(np.isnan(y_norm)) or np.any(np.isinf(y_norm)):
                return penalty
                
            # --- OVERSHOOT CALCULATION ---
            # Peak overshoot is the maximum value minus the setpoint (1.0 in y_norm)
            peak_val = np.max(y_norm)
            actual_overshoot = max(0.0, peak_val - 1.0) 
            
            # 1. Hard Constraint: If overshoot exceeds the limit, kill this candidate
            if actual_overshoot > max_overshoot_limit:
                return penalty

            # 2. Stability Check: Kill candidates that oscillate wildly below zero
            if np.min(y_norm) < -0.1:
                return penalty
            # -----------------------------

            error = 1.0 - y_norm
            int_error = np.trapezoid(np.abs(error), t_opt)
            
            # --- THE PROPER NORMALIZATION: Total Variation ---
            # Levels the playing field between Kp and Ki, but strictly punishes oscillation/chatter
            # Prepend 0.0 to capture the initial jump at t=0
            u_with_initial = np.concatenate(([0.0], u_out))
            norm_effort = np.sum(np.abs(np.diff(u_with_initial)))
            
            # Integral of Overshoot Area (for the soft cost)
            overshoot_array = np.where(error < 0, np.abs(error), 0.0)
            int_overshoot = np.trapezoid(overshoot_array, t_opt)

            crossings_10 = np.where(y_norm >= 0.1)[0]
            crossings_90 = np.where(y_norm >= 0.9)[0]
            
            if len(crossings_10) > 0 and len(crossings_90) > 0:
                rise_time = t_opt[crossings_90[0]] - t_opt[crossings_10[0]]
            else:
                rise_time = T_sim * 10
                
            norm_error = int_error / T_sim
            norm_overshoot = int_overshoot / T_sim
            norm_rise_time = rise_time / avg_rise_time
            
            # Kill candidates that are wildly unstable
            if norm_error > 2.0 or norm_rise_time > 10.0:
                return penalty
                
            cost = (w_iae * norm_error) + (w_effort * norm_effort) + (w_os * norm_overshoot) + (w_rt * norm_rise_time)
            return cost
            
        except Exception as e:
            return penalty

    bounds = [(min_kp, max_kp), (min_ki, max_ki)]
    
    result = differential_evolution(
        objective, 
        bounds, 
        strategy='best1bin', 
        maxiter=30,      
        popsize=20,     
        mutation=(0.5, 1.0), 
        recombination=0.745, 
        tol=1e-4,
        disp=False
    )
    
    best_kp, best_ki = result.x
    print(f"[Result] {name} -> Kp: {best_kp:.4f}, Ki: {best_ki:.4f} (Cost: {result.fun:.4f})")
    return best_kp, best_ki