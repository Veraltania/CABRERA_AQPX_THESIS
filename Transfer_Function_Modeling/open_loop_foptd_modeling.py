import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path


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

    # FIX: Use lowercase 's' instead of 'S' to comply with new Pandas offsets
    window_str = f'{window_seconds}s'
    smoothed_col = f'{target_column}_smoothed'

    df_filtered[smoothed_col] = df_filtered[target_column].rolling(window=window_str, center=True).mean()

    # Drop NaNs introduced by the centered rolling window
    df_filtered.dropna(subset=[smoothed_col], inplace=True)

    return df_filtered.reset_index()


def fopdt_response(t, K, tau, theta, y0, t_step, delta_u):
    """
    Mathematical definition of a First-Order Process with Time Delay (FOPDT)
    response to a step input.
    """
    y = np.ones_like(t) * y0

    # Shift time vector relative to step change and time delay
    delayed_t = t - t_step - theta

    # Compute response only for t >= (t_step + theta)
    mask = delayed_t >= 0
    y[mask] = y0 + K * delta_u * (1 - np.exp(-delayed_t[mask] / tau))

    return y

def fit_fopdt_model(t, y, t_step, delta_u):
    """
    Fits FOPDT parameters (K, tau, theta) to experimental data.
    Returns a dictionary of the optimized parameters.
    """
    y0 = y.iloc[0] if isinstance(y, pd.Series) else y[0]

    # Wrapper function for scipy's curve_fit
    def objective(t_val, K_val, tau_val, theta_val):
        return fopdt_response(t_val, K_val, tau_val, theta_val, y0, t_step, delta_u)

    # Initial Guesses
    K_guess = (y.iloc[-1] - y0) / delta_u if delta_u != 0 else 1.0
    tau_guess = (t.iloc[-1] - t.iloc[0]) / 3.0
    theta_guess = 0.0  # Assume start with zero dead time

    # Optimization bounds:
    # Gain K: [-inf, inf] (allows for inverse response)
    # Tau: [0.001, inf] (must be positive)
    # Theta (delay): [0, max time difference]
    max_theta = max(0, t.iloc[-1] - t_step)
    bounds = ([-np.inf, 0.001, 0.0], [np.inf, np.inf, max_theta])

    try:
        popt, _ = curve_fit(
            objective, t, y,
            p0=[K_guess, tau_guess, theta_guess],
            bounds=bounds,
            maxfev=5000
        )
        K, tau, theta = popt
        return {'K': K, 'tau': tau, 'theta': theta, 'y0': y0}
    except Exception as e:
        print(f"Curve fitting failed: {e}")
        return None


def plot_and_save(t, y_raw, y_smooth, y_model, target_column, tf_name, output_base_dir="Transfer_Functions_Outputs"):
    """
    Plots the raw data, smoothed data, and FOPDT fit, then saves the figure
    into a uniquely specified folder.
    """
    # Create unique folder for this specific transfer function
    save_dir = Path(output_base_dir) / tf_name
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    # Plotting routines
    plt.plot(t, y_raw, color='lightgray', label=f"Raw {target_column}", alpha=0.7)
    plt.plot(t, y_smooth, color='blue', label=f"Smoothed Data", linewidth=2)
    plt.plot(t, y_model, color='red', linestyle='--', label=f"FOPDT Fit", linewidth=2)

    plt.title(f"Open Loop Response & FOPDT Fit: {tf_name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel(target_column)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save to the unique directory
    save_path = save_dir / f"{tf_name}_fopdt_response.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot successfully saved to: {save_path}")


def analyze_open_loop(file_paths, start_step, end_step, target_column, window_seconds, tf_name, t_step_time, delta_u):
    """
    Main execution wrapper that ties extraction, FOPDT fitting, and plot saving together.
    """
    print(f"--- Analyzing {tf_name} ---")

    # 1. Extract and smooth data
    df = extract_and_process(file_paths, start_step, end_step, target_column, window_seconds)

    # Convert timestamps to elapsed seconds for modeling
    t_seconds = (df['Datetime'] - df['Datetime'].iloc[0]).dt.total_seconds()
    y_raw = df[target_column]
    y_smooth = df[f'{target_column}_smoothed']

    # Calculate step time in relative seconds
    t_step_rel = (pd.to_datetime(t_step_time) - df['Datetime'].iloc[0]).total_seconds()

    # 2. Fit FOPDT parameters
    params = fit_fopdt_model(t_seconds, y_smooth, t_step_rel, delta_u)

    if params:
        print("\n[ FOPDT Model Parameters Extracted ]")
        print(f"Process Gain (K):      {params['K']:.4f}")
        print(f"Time Constant (Tau):   {params['tau']:.4f} seconds")
        print(f"Time Delay (Theta):    {params['theta']:.4f} seconds")

        # Generate the model array using the discovered parameters
        y_model = fopdt_response(t_seconds, params['K'], params['tau'], params['theta'],
                                 params['y0'], t_step_rel, delta_u)

        # 3. Plot and save to unique TF directory
        plot_and_save(t_seconds, y_raw, y_smooth, y_model, target_column, tf_name)

        return params
    else:
        print("Failed to extract model parameters.")
        return None

if __name__ == "__main__":
    # ==========================================
    # EXAMPLE USAGE 1: TDS OPEN LOOP (TRIAL 3)
    # ==========================================
    base_data_path = r'/Transfer_Function_Modeling/data'
    # tds_files = [
    #     f'{base_data_path}\\AQPX_data_log_2026-02-09.csv',
    #     f'{base_data_path}\\AQPX_data_log_2026-02-10.csv',
    #     f'{base_data_path}\\AQPX_data_log_2026-02-11.csv',
    #     f'{base_data_path}\\AQPX_data_log_2026-02-12.csv'
    # ]
    #
    # tds_params = analyze_open_loop(
    #     file_paths=tds_files,
    #     start_step='2026-02-10 21:00:00',
    #     end_step='2026-02-11 9:00:00',
    #     target_column='MCP_WQ_TDS',
    #     window_seconds=120,
    #     tf_name='TDS_Nutrient_Pump_Transfer_Function',  # Unique folder will be named this
    #     t_step_time='2026-02-10 21:05:00',  # Time the actuator was switched
    #     delta_u=10.0  # Change in input variable (e.g. pump speed % or voltage change)
    # )

    do_file = [
        r'D:\aqpx\Cabrera_Thesis_AQPX\Transfer_Function_Modeling\data\AQPX_data_log_2026-02-05.csv']

    do_params = analyze_open_loop(
        file_paths=do_file,
        start_step='2026-02-05 20:00:00',
        end_step='2026-02-05 23:59:59',
        target_column='MCP_WQ_DO',
        window_seconds=60,
        tf_name='DO_Aeration_Transfer_Function',  
        t_step_time='2026-02-05 20:00:00',
        delta_u=1.0  # 1 to denote an ON/OFF state change, or standard magnitude
    )