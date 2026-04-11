import numpy as np
from scipy.optimize import differential_evolution
import csv
from datetime import datetime

def fit_closed_loop_fopdt(t_arr, u_arr, y_arr):
    """
    Minimizes MSE between physical data and an ideal FOPDT response using 
    Differential Evolution to avoid getting trapped in local minima.
    """
    t_arr, u_arr, y_arr = np.array(t_arr), np.array(u_arr), np.array(y_arr)
    
    # Calculate average dt (time step)
    dt = np.mean(np.diff(t_arr))
    if dt <= 0: dt = 1.0 
    
    # Isolate the delta (bump) from the steady-state baseline
    u0, y0 = u_arr[0], y_arr[0]
    du = u_arr - u0
    dy = y_arr - y0
    
    def objective_function(params):
        K, tau, delay = params
        y_sim = np.zeros_like(t_arr)
        
        # Calculate how many array indices represent the time delay
        delay_steps = int(max(0.0, delay) / dt)
        
        # Apply dead-time shift to the control signal
        du_delayed = np.zeros_like(du)
        if delay_steps < len(du):
            du_delayed[delay_steps:] = du[:-delay_steps]
            
        # Numerical integration of a First-Order Process
        for i in range(1, len(t_arr)):
            derivative = (dt / max(1.0, tau)) * (K * du_delayed[i-1] - y_sim[i-1])
            y_sim[i] = y_sim[i-1] + derivative
            
        # Return Mean Squared Error
        return np.mean((dy - y_sim)**2) 
        
    # K bounds: (0.1, 150.0) -> Increased upper bound to support MATLAB baseline
    # Tau bounds: (10.0, 10000.0)
    # Delay bounds: (0.0, 500.0)
    bnds = [(0.1, 150.0), (10.0, 10000.0), (0.0, 500.0)]
    
    # Use Differential Evolution instead of L-BFGS-B
    # popsize=15 is a good balance between speed and exhaustive searching
    res = differential_evolution(objective_function, bounds=bnds, seed=42, popsize=15)
    
    return res.x[0], res.x[1], res.x[2] # Returns optimal K, Tau, Delay


def extract_csv_and_fit(csv_file_path, y_column, u_column='Duty_Cycle', date_col='Date', time_col='Time'):
    """
    Reads the hardware's retuning CSV file, converts timestamps to elapsed seconds, 
    and passes the data arrays to the mathematical fitter.
    """
    t_arr, u_arr, y_arr = [], [], []
    
    with open(csv_file_path, 'r') as f:
        reader = csv.DictReader(f)
        first_time = None
        
        for row in reader:
            dt_str = f"{row[date_col]} {row[time_col]}"
            try:
                current_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                current_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f") 
                
            if first_time is None:
                first_time = current_dt
                
            t_secs = (current_dt - first_time).total_seconds()
            t_arr.append(t_secs)
            
            u_arr.append(float(row.get(u_column, 0.0))) 
            y_arr.append(float(row.get(y_column, 0.0)))
            
    if len(t_arr) < 10:
        raise ValueError("Not enough data points in CSV to perform a reliable curve fit.")
        
    return fit_closed_loop_fopdt(t_arr, u_arr, y_arr)