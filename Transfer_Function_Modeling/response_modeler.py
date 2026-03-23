import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path

# Uses FOPTD modeling code by Jeffrey Kantor [1]

def extract_and_process(file_paths, start_time, end_time, target_column, window_seconds=60, time_col='Time'):
    """
    Extracts data from one or multiple CSVs, filters by time, and applies a moving average.
    """
    # Ensure file_paths is a list
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    dfs = []
    for fp in file_paths:
        if os.path.exists(fp):
            dfs.append(pd.read_csv(fp))
        else:
            print(f"Warning: File {fp} not found.")

    if not dfs:
        raise ValueError("No valid data files provided.")

    # Combine and sort data
    df = pd.concat(dfs, ignore_index=True)

    # Combine 'Date' and 'Time' columns into a single Datetime column
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    time_col = 'Datetime'  # Override time_col to use the new accurate column

    df = df.sort_values(by=time_col)

    # Crop to the specified time window
    mask = (df[time_col] >= pd.to_datetime(start_time)) & (df[time_col] <= pd.to_datetime(end_time))
    df_filtered = df.loc[mask].copy()

    # Apply moving average (centered to negate lag)
    df_filtered.set_index(time_col, inplace=True)

    window_str = f'{window_seconds}s'
    smoothed_col = f'{target_column}_smoothed'

    df_filtered[smoothed_col] = df_filtered[target_column].rolling(window=window_str, center=True).mean()

    # Drop NaNs introduced by the centered rolling window
    df_filtered.dropna(subset=[smoothed_col], inplace=True)

    return df_filtered.reset_index()

def foptd(t, K=1, tau=1, tau_d=0):
    """
    Computes the response of a first order system with time delay to a unit step input.
    Assumes the step change happens at t=0.
    """
    tau_d = max(0, tau_d)
    tau = max(1e-8, tau)  # Prevent division by zero
    return np.array([K * (1 - np.exp(-(t_val - tau_d) / tau)) if t_val >= tau_d else 0 for t_val in t])

def err(X, t, y):
    """
    Objective function: Computes the Integral Absolute Error (IAE) between
    the FOPTD model and the scaled experimental data.
    """
    K, tau, tau_d = X
    z = foptd(t, K, tau, tau_d)

    # Calculate Integral Absolute Error (IAE)
    if len(t) > 0:
        iae = sum(abs(z - y)) * (max(t) - min(t)) / len(t)
    else:
        iae = float('inf')

    return iae

def plot_and_save(t, y_raw, y_smooth, y_model, target_column, tf_name, output_base_dir="Transfer_Functions_Outputs"):
    """
    Plots the raw data, smoothed data, and FOPTD fit, then saves the figure.
    """
    # Create unique folder for this specific transfer function
    save_dir = Path(output_base_dir) / tf_name
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    # Plotting routines
    plt.plot(t, y_raw, color='lightgray', label=f"Raw {target_column}", alpha=0.7)
    plt.plot(t, y_smooth, color='blue', label=f"Smoothed Data", linewidth=2)
    plt.plot(t, y_model, color='red', linestyle='--', label=f"FOPTD Fit", linewidth=2)

    plt.title(f"Open Loop Response & FOPTD Fit: {tf_name}")
    plt.xlabel("Time (seconds) from start of dataset")
    plt.ylabel(target_column)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save to the unique directory
    save_path = save_dir / f"{tf_name}_fopdt_response.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot successfully saved to: {save_path}")

def analyze_response(file_paths, start_step, end_step, target_column, window_seconds, tf_name, t_step_time, delta_u):
    """
    Main execution wrapper that ties extraction, Notebook FOPTD fitting, and plot saving together.
    """
    print(f"--- Analyzing {tf_name} ---")

    # 1. Extract and smooth data
    df = extract_and_process(file_paths, start_step, end_step, target_column, window_seconds)

    # Convert Datetime to relative seconds where t=0 is the EXACT moment the step occurs
    step_dt = pd.to_datetime(t_step_time)
    t_rel = (df['Datetime'] - step_dt).dt.total_seconds().values

    y_raw = df[target_column].values
    y_smooth = df[f'{target_column}_smoothed'].values

    # Determine baseline initial steady-state (y0)
    # Uses the average of all available smoothed data *before* the step time.
    pre_step_mask = t_rel <= 0
    if pre_step_mask.any():
        y0 = np.mean(y_smooth[pre_step_mask])
    else:
        y0 = y_smooth[0]

    # 2. Scale data to fit the framework of the notebook FOPTD model
    # Filter only the data that occurs ON or AFTER the step change
    fit_mask = t_rel >= 0
    ts = t_rel[fit_mask]
    y_fit = y_smooth[fit_mask]

    # Scale the response to a unit step input: ys = (y(t) - y0) / delta_u
    ys = (y_fit - y0) / delta_u

    # 3. Fit parameters using scipy.optimize.minimize (minimizing IAE)
    K_guess = ys[-1] if len(ys) > 0 else 1.0
    tau_guess = (ts[-1] - ts[0]) / 3.0 if len(ts) > 0 else 10.0
    tau_d_guess = 0.0
    X_initial = [K_guess, tau_guess, tau_d_guess]

    # Optimize using IAE error function
    res = minimize(err, X_initial, args=(ts, ys))
    K_opt, tau_opt, tau_d_opt = res.x

    # Enforce non-negative physics constraints
    tau_opt = max(0, tau_opt)
    tau_d_opt = max(0, tau_d_opt)

    print("\n[ FOPTD Model Parameters Extracted ]")
    print(f"Process Gain (K):      {K_opt:.4f}")
    print(f"Time Constant (Tau):   {tau_opt:.4f} seconds")
    print(f"Time Delay (Theta):    {tau_d_opt:.4f} seconds")

    # 4. Rescale FOPTD output to reconstruct the prediction over the entire timeline
    y_model = np.ones_like(t_rel) * y0  # Baseline before the step
    z_opt = foptd(ts, K_opt, tau_opt, tau_d_opt)  # Notebook model
    y_model[fit_mask] = y0 + (z_opt * delta_u)  # Scaled output after the step

    # Calculate absolute elapsed time for the x-axis of the plot
    t_seconds = (df['Datetime'] - df['Datetime'].iloc[0]).dt.total_seconds().values

    # 5. Plot and save
    plot_and_save(t_seconds, y_raw, y_smooth, y_model, target_column, tf_name)

    return {'K': K_opt, 'tau': tau_opt, 'theta': tau_d_opt, 'y0': y0}

if __name__ == "__main__":
    do_file = [r'D:\aqpx\Cabrera_Thesis_AQPX\Transfer_Function_Modeling\data\AQPX_data_log_2026-02-07.csv']

    base_data_path = r'D:\aqpx\Cabrera_Thesis_AQPX\Transfer_Function_Modeling\data'
    tds_files = [
        fr'{base_data_path}\AQPX_data_log_2026-02-09.csv',
        fr'{base_data_path}\AQPX_data_log_2026-02-10.csv',
        fr'{base_data_path}\AQPX_data_log_2026-02-11.csv',
        fr'{base_data_path}\AQPX_data_log_2026-02-12.csv'
    ]

    do_params = analyze_response(
        file_paths=do_file,
        start_step='2026-02-07 20:00:00',
        end_step='2026-02-07 23:59:59',
        target_column='MCP_WQ_DO',
        window_seconds=60,
        tf_name='DO_TF3_NIGHTTIME',
        t_step_time='2026-02-07 20:00:00',
        delta_u=1.0  # 1 to denote an ON/OFF state change, or standard magnitude
    )

'''
REFERENCES: 

[1] J. C. Kantor, “Fitting First Order Plus Time Delay to Step Response,”
CBE30338 Chemical Process Control.
https://jckantor.github.io/CBE30338/03.04-Fitting-First-Order-plus-Time-Delay-to-Step-Response.html
(accessed Mar. 18, 2026).
'''
