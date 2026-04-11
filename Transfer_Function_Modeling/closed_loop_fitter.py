import numpy as np
from scipy.optimize import differential_evolution
import csv
from datetime import datetime

def fit_closed_loop_fopdt(t_arr, u_arr, y_arr):
    t_arr, u_arr, y_arr = np.array(t_arr), np.array(u_arr), np.array(y_arr)
    dt = np.mean(np.diff(t_arr))
    if dt <= 0: dt = 1.0 
    
    # FIX 1: Average the first few points to prevent random noise spikes 
    # from permanently skewing the baseline for the whole math operation.
    base_samples = min(5, len(u_arr))
    u0 = np.mean(u_arr[:base_samples])
    y0 = np.mean(y_arr[:base_samples])
    
    du = u_arr - u0
    dy = y_arr - y0
    
    def objective_function(params):
        K, tau, delay = params
        y_sim = np.zeros_like(t_arr)
        
        # FIX 2: Continuous Delay Interpolation
        # This makes the curve smooth, preventing the optimizer from breaking!
        t_shifted = t_arr - delay
        du_delayed = np.interp(t_shifted, t_arr, du, left=0.0)
            
        for i in range(1, len(t_arr)):
            derivative = (dt / max(1.0, tau)) * (K * du_delayed[i-1] - y_sim[i-1])
            y_sim[i] = y_sim[i-1] + derivative
            
        return np.mean((dy - y_sim)**2) 
        
    bnds = [(0.1, 150.0), (10.0, 10000.0), (0.0, 500.0)]
    res = differential_evolution(objective_function, bounds=bnds, seed=42, popsize=15)
    
    return res.x[0], res.x[1], res.x[2]


def extract_csv_and_fit(csv_file_path, y_column, u_column='Duty_Cycle', date_col='Date', time_col='Time'):
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
                
            if first_time is None: first_time = current_dt
                
            t_secs = (current_dt - first_time).total_seconds()
            t_arr.append(t_secs)
            u_arr.append(float(row.get(u_column, 0.0))) 
            y_arr.append(float(row.get(y_column, 0.0)))
            
    if len(t_arr) < 10:
        raise ValueError("Not enough data points in CSV to perform a reliable curve fit.")
        
    return fit_closed_loop_fopdt(t_arr, u_arr, y_arr)