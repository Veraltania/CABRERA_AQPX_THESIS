import numpy as np
from scipy.optimize import minimize

def simulate_closed_loop_fopdt(X, t, u, y0):
    """
    Numerically integrates the FOPDT model to predict the response y(t) 
    given a dynamic, varying control signal u(t).
    """
    K, tau, delay = X
    
    # Enforce physical constraints internally to prevent math errors
    tau = max(1e-5, tau)
    delay = max(0.0, delay)

    sim_y = np.zeros(len(t))
    sim_y[0] = y0
    
    # Estimate the system's baseline (y_base). 
    # In steady state, y = y_base + K*u. Therefore, y_base = y0 - K*u0.
    y_base = y0 - (K * u[0])
    
    dt_array = np.diff(t)
    dt_mean = np.mean(dt_array) if len(dt_array) > 0 else 1.0
    
    # Apply time delay (theta) to the u(t) array
    delay_steps = int(delay / dt_mean)
    u_delayed = np.roll(u, delay_steps)
    u_delayed[:delay_steps] = u[0]  # Pad the initial delay period with the starting u value

    # Euler Integration (matches the VirtualAquaculturePlant step logic)
    for i in range(1, len(t)):
        dt = dt_array[i-1]
        deviation_y = sim_y[i-1] - y_base
        dy = (dt / tau) * ((K * u_delayed[i-1]) - deviation_y)
        sim_y[i] = sim_y[i-1] + dy
        
    return sim_y

def err(X, t, u, y_actual):
    """
    Objective function: Computes the Integral Absolute Error (IAE) between
    the simulated FOPDT response and the actual closed-loop measurement.
    """
    sim_y = simulate_closed_loop_fopdt(X, t, u, y_actual[0])
    
    dt_mean = np.mean(np.diff(t))
    iae = np.sum(np.abs(sim_y - y_actual)) * dt_mean
    return iae

def fit_closed_loop_fopdt(bump_t, bump_u, bump_y):
    """
    Takes time, control signal, and process variable arrays from a closed-loop 
    bump test and extracts the optimal K, Tau, and Delay parameters.
    """
    t = np.array(bump_t)
    u = np.array(bump_u)
    y = np.array(bump_y)
    
    # Shift time so t[0] = 0
    t = t - t[0]
    
    # Generate intelligent initial guesses to help the solver converge
    delta_y = y[-1] - y[0]
    delta_u = u[-1] - u[0]
    
    # K guess: final change in Y over final change in U (if U changed significantly)
    if abs(delta_u) > 1e-4:
        K_guess = delta_y / delta_u
    else:
        K_guess = 1.0
        
    tau_guess = (t[-1] - t[0]) / 3.0  # Assume settling time is roughly 3-4 time constants
    delay_guess = 0.05
    
    X_initial = [K_guess, max(1.0, tau_guess), max(0.0, delay_guess)]
    
    # Set bounds: K is unbounded (can be reverse acting), Tau > 0, Delay >= 0
    bounds = ((None, None), (1e-3, None), (0.0, None))
    
    # Run bounded optimization minimizing IAE
    res = minimize(
        err, 
        X_initial, 
        args=(t, u, y), 
        bounds=bounds, 
        method='L-BFGS-B'
    )
    
    K_opt, tau_opt, delay_opt = res.x
    
    return K_opt, tau_opt, delay_opt