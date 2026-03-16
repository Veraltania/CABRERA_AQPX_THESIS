import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.stattools import acf, ccf

# Import everything needed directly from your module
from open_loop_modeling_tools import (
    extract_and_process,
    simulate_system,
    func_foptd,
    func_soptd,
    func_toptd,
    func_foptdld,
    func_soptdld
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Setup your list of data files and target columns
DATA_FILES = [
    r'D:\aqpx\Cabrera_Thesis_AQPX\Open_Loop_Modeling\data\AQPX_data_log_2026-02-05.csv',
    r'D:\aqpx\Cabrera_Thesis_AQPX\Open_Loop_Modeling\data\AQPX_data_log_2026-02-07.csv'
]

TARGET_COLUMN = 'MCP_WQ_DO'
INPUT_COLUMN = 'Pump Activation'
WINDOW_SECONDS = 60

# Define the list of data segments/trials you want to analyze.
# Each segment has a name and its own specific time range.
DATA_SEGMENTS = [
    {
        "segment_name": "Trial_1_Daytime",
        "start_time": "2026-02-05 10:00:00",
        "end_time": "2026-02-05 15:00:00"
    },
    {
        "segment_name": "Trial_1_Nighttime",
        "start_time": "2026-02-05 20:00:00",
        "end_time": "2026-02-05 23:59:59"
    },
    {
        "segment_name": "Trial_1_Daytime",  # You might want to rename this to Trial_2 or Trial_3
        "start_time": "2026-02-07 10:00:00",
        "end_time": "2026-02-07 15:00:00"
    },
    {
        "segment_name": "Trial_3_Nighttime",
        "start_time": "2026-02-07 20:00:00",
        "end_time": "2026-02-07 23:59:59"
    }
]

# The five models from your tools that will be fitted for EVERY segment
MODELS = ['FOPTD', 'SOPTD', 'TOPTD', 'FOPTDLD', 'SOPTDLD']


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_transfer_function_text(model_name, popt, param_names):
    """Generates the mathematical transfer function format dynamically"""
    p = {k: f"{v:.4f}" for k, v in zip(param_names, popt)}

    if model_name == 'FOPTD':
        tf = f"G(s) = [ {p['K']} / ({p['Tp1']}s + 1) ] * e^(-{p['Td']}s)"
    elif model_name == 'SOPTD':
        tf = f"G(s) = [ {p['K']} / (({p['Tp1']}s + 1)({p['Tp2']}s + 1)) ] * e^(-{p['Td']}s)"
    elif model_name == 'TOPTD':
        tf = f"G(s) = [ {p['K']} / (({p['Tp1']}s + 1)({p['Tp2']}s + 1)({p['Tp3']}s + 1)) ] * e^(-{p['Td']}s)"
    elif model_name == 'FOPTDLD':
        tf = f"G(s) = [ {p['K']} * ({p['Tz']}s + 1) / ({p['Tp1']}s + 1) ] * e^(-{p['Td']}s)"
    elif model_name == 'SOPTDLD':
        tf = f"G(s) = [ {p['K']} * ({p['Tz']}s + 1) / (({p['Tp1']}s + 1)({p['Tp2']}s + 1)) ] * e^(-{p['Td']}s)"
    else:
        tf = "Model Unknown"

    tf += f"\nBaseline / Offset (y0): {p['y0']}"
    return tf


# ==========================================
# 3. AUTOMATION LOGIC
# ==========================================
def run_automation():
    print("==================================================")
    print("Combining data files...")
    # Read and concatenate all CSV files from the list
    df_list = [pd.read_csv(f) for f in DATA_FILES]
    combined_df = pd.concat(df_list, ignore_index=True)

    # Save to a temporary file so extract_and_process can use it as a single file path
    temp_combined_file = 'temp_combined_data.csv'
    combined_df.to_csv(temp_combined_file, index=False)
    print(f"Successfully combined {len(DATA_FILES)} files.\n")

    for segment in DATA_SEGMENTS:
        segment_name = segment["segment_name"]
        start_time = pd.to_datetime(segment["start_time"])
        end_time = pd.to_datetime(segment["end_time"])

        print(f"\n==================================================")
        print(f"Processing Segment: {segment_name} | {start_time} to {end_time}")
        print(f"==================================================")

        # --- 1. Extract data ONCE per segment from the COMBINED file ---
        df = extract_and_process(temp_combined_file, start_time, end_time, TARGET_COLUMN, WINDOW_SECONDS)

        if df is None or df.empty:
            print(f"Skipping {segment_name}: No data returned.")
            continue

        filtered_col = f"{TARGET_COLUMN}_filtered"
        if filtered_col not in df.columns:
            filtered_col = TARGET_COLUMN

        t_abs = df['DateTime']
        t_rel = (t_abs - t_abs.iloc[0]).dt.total_seconds().values
        y_raw = df[TARGET_COLUMN].values
        y_filt = df[filtered_col].values

        if INPUT_COLUMN in df.columns:
            u_input = df[INPUT_COLUMN].values
        else:
            u_input = np.ones_like(y_filt)

        # --- 2. Build fitting initial constraints ---
        y_start = np.mean(y_filt[:5])
        y_end = np.mean(y_filt[-5:])
        K_g = y_end - y_start
        T_total = t_rel[-1]
        Tp_g = T_total / 4
        Td_g = 10.0
        Tz_g = 1.0

        # --- 3. Loop through ALL models for this segment ---
        for model_name in MODELS:
            print(f"\n---> Fitting {model_name} for {segment_name}...")

            if model_name == 'FOPTD':
                func, p0 = func_foptd, [K_g, Tp_g, Td_g, y_start]
                bounds = ([-np.inf, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf])
                param_names = ['K', 'Tp1', 'Td', 'y0']
                eqn_label = r'$G(s) = \frac{K}{T_{p1}s+1}e^{-T_ds}$'
            elif model_name == 'SOPTD':
                func, p0 = func_soptd, [K_g, Tp_g, Tp_g, Td_g, y_start]
                bounds = ([-np.inf, 0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])
                param_names = ['K', 'Tp1', 'Tp2', 'Td', 'y0']
                eqn_label = r'$G(s) = \frac{K}{(T_{p1}s+1)(T_{p2}s+1)}e^{-T_ds}$'
            elif model_name == 'TOPTD':
                func, p0 = func_toptd, [K_g, Tp_g, Tp_g, Tp_g, Td_g, y_start]
                bounds = ([-np.inf, 0, 0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
                param_names = ['K', 'Tp1', 'Tp2', 'Tp3', 'Td', 'y0']
                eqn_label = r'$G(s) = \frac{K}{\Pi(T_{pi}s+1)}e^{-T_ds}$'
            elif model_name == 'FOPTDLD':
                func, p0 = func_foptdld, [K_g, Tp_g, Tz_g, Td_g, y_start]
                bounds = ([-np.inf, 0, -np.inf, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])
                param_names = ['K', 'Tp1', 'Tz', 'Td', 'y0']
                eqn_label = r'$G(s) = \frac{K(T_zs+1)}{T_{p1}s+1}e^{-T_ds}$'
            elif model_name == 'SOPTDLD':
                func, p0 = func_soptdld, [K_g, Tp_g, Tp_g, Tz_g, Td_g, y_start]
                bounds = ([-np.inf, 0, 0, -np.inf, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
                param_names = ['K', 'Tp1', 'Tp2', 'Tz', 'Td', 'y0']
                eqn_label = r'$G(s) = \frac{K(T_zs+1)}{\Pi(T_{pi}s+1)}e^{-T_ds}$'

            try:
                popt, _ = curve_fit(func, t_rel, y_filt, p0=p0, bounds=bounds, maxfev=10000)
            except Exception as e:
                print(f"Fit failed for {model_name}: {e}")
                continue

            # --- FIXED RESIDUAL CALCULATION ---
            # Calculate simulation using the filtered time array, but
            # calculate residuals against the RAW data to avoid moving-average artifacts.
            y_model = simulate_system(t_rel, model_name, popt)
            residuals = y_raw - y_model  # CHANGED from y_filt to y_raw
            r2 = r2_score(y_filt, y_model)
            rmse = np.sqrt(mean_squared_error(y_filt, y_model))

            # --- 4. Generate Output Directory ---
            dir_name = os.path.join(segment_name, model_name)
            os.makedirs(dir_name, exist_ok=True)

            # Save Text File
            tf_text = get_transfer_function_text(model_name, popt, param_names)
            with open(os.path.join(dir_name, '00_transfer_function.txt'), 'w') as f:
                f.write(f"Segment: {segment_name} | Time Range: {start_time} to {end_time}\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"R-Squared Fit: {r2:.4f}\n")
                f.write(f"RMSE: {rmse:.4f}\n" + "-" * 40 + "\n")
                f.write(tf_text + "\n")

            # Save CSV
            with open(os.path.join(dir_name, f"00_{model_name}_parameters.csv"), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Parameter', 'Value'])
                for name, val in zip(param_names, popt):
                    writer.writerow([name, val])

            # --- 5. Generate Figures ---
            # Graph A: Fitting and Simulating
            plt.figure(figsize=(10, 6))
            plt.plot(t_abs, y_raw, color='lightgray', label='Raw Data')
            plt.plot(t_abs, y_filt, color='red', linestyle=':', label='Filtered Data', linewidth=2)
            plt.plot(t_abs, y_model, color='blue', linewidth=2, label=f'{model_name} Simulated (R2={r2:.3f})')

            plt.title(f"{model_name} System ID\nSegment: {segment_name} ({start_time} - {end_time})")
            plt.xlabel("Time")
            plt.ylabel(TARGET_COLUMN)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Info Box
            param_str_lines = [f"{n}: {v:.2f}" for n, v in zip(param_names, popt)]
            info_text = f"{eqn_label}\nRMSE: {rmse:.3f}\n" + "\n".join(param_str_lines)
            props = dict(boxstyle='round', facecolor='white', alpha=0.9)
            plt.gca().text(0.02, 0.95, info_text, transform=plt.gca().transAxes, fontsize=10,
                           verticalalignment='top',
                           bbox=props)

            plt.tight_layout()
            plt.savefig(os.path.join(dir_name, '01_fitting_simulation.png'))
            plt.close()

            # Graph B: Autocorrelation of True (Raw) Residuals
            lag_acf = acf(residuals, nlags=40)
            plt.figure(figsize=(10, 4))
            plt.stem(range(len(lag_acf)), lag_acf, basefmt="b-")
            plt.axhline(y=0, linestyle='--', color='gray')
            plt.axhline(y=-1.96 / np.sqrt(len(residuals)), linestyle='--', color='red', alpha=0.5)
            plt.axhline(y=1.96 / np.sqrt(len(residuals)), linestyle='--', color='red', alpha=0.5)
            plt.title(f"Residual Autocorrelation (ACF) - {segment_name} ({model_name})")
            plt.xlabel("Lags")
            plt.ylabel("ACF")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(dir_name, '02_residuals_acf.png'))
            plt.close()

            # Graph C: Residuals vs Time (Replaces CCF for Step Tests)
            plt.figure(figsize=(10, 4))
            plt.plot(t_abs, residuals, color='purple', linewidth=1)
            plt.axhline(y=0, linestyle='--', color='black', linewidth=1.5)
            plt.title(f"Residuals vs Time - {segment_name} ({model_name})")
            plt.xlabel("Time")
            plt.ylabel("Error (Raw - Model)")
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(dir_name, '03_residuals_vs_time.png'))
            plt.close()

    # Clean up the temporary combined file after everything is done
    if os.path.exists(temp_combined_file):
        os.remove(temp_combined_file)
        print("\nCleaned up temporary combined data file.")


if __name__ == "__main__":
    run_automation()