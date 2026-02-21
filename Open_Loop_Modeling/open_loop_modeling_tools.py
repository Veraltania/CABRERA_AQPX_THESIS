import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import butter, sosfiltfilt
from scipy.optimize import curve_fit
from scipy import signal
from sklearn.metrics import r2_score, mean_squared_error

# ================================
# FILTER DESIGN & APPLICATION
# ================================

def design_butterworth_filter(fs, period_minutes=None, order=4):
    """
    Designs a low-pass Butterworth filter.
    """
    if period_minutes is None:
        raise ValueError("Please specify period_minutes.")

    period_sec = period_minutes * 60

    # Cutoff frequency (Hz)
    # Using a standard definition where cutoff is related to the period
    cutoff = 1.0 / (period_sec * 2)

    sos = butter(N=order, Wn=cutoff, btype="low", fs=fs, output="sos")
    print(f"Filter designed: Preserves variations SLOWER than {period_minutes} minutes.")
    return sos


def apply_butterworth_filter(df, column_name, sos, output_column_name=None):
    """
    Applies the filter to a dataframe column, handling NaNs.
    """
    if output_column_name is None:
        output_column_name = f"{column_name}_filtered"

    data = df[column_name].astype(float).values

    # Handle NaNs by linear interpolation
    if np.any(np.isnan(data)):
        data = pd.Series(data).interpolate(method='linear').values

    df[output_column_name] = sosfiltfilt(sos, data)
    return df


# ================================
# PLOTTING FUNCTION
# ================================

def plot_filtered_data(df, column_name, filtered_column_name=None, title=None):
    """
    Plots the filtered data with the raw data in the background (gray).
    """
    if filtered_column_name is None:
        filtered_column_name = f"{column_name}_filtered"

    if title is None:
        title = f"{column_name} (Filtered vs Raw)"

    plt.figure(figsize=(14, 6))

    # --- RAW DATA (Gray, Background) ---
    plt.plot(df['DateTime'], df[column_name],
             color='gray', alpha=0.3, label='Raw Data', linewidth=1)

    # --- FILTERED DATA (Red, Foreground) ---
    plt.plot(df['DateTime'], df[filtered_column_name],
             color='red', linewidth=2, label='Filtered Data')

    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ================================
# DATA EXTRACTION
# ================================
def extract_and_process(file_paths, start_time, end_time, column_name, filter_minutes=10):
    """
    Handles both a single file path (string) or a list of file paths.
    """
    # --- FLEXIBILITY CHECK ---
    # If the user passed a single string, wrap it in a list so the loop works
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    # 1. Load and stitch all files
    df_list = []
    for path in file_paths:
        temp_df = pd.read_csv(path)
        temp_df['DateTime'] = pd.to_datetime(temp_df['Date'] + ' ' + temp_df['Time'])
        df_list.append(temp_df)

    df_full = pd.concat(df_list, ignore_index=True)

    # 2. Sort and Clean
    df_full = df_full.sort_values('DateTime').drop_duplicates(subset=['DateTime']).reset_index(drop=True)

    # 3. Filter by Time Range
    mask = (df_full['DateTime'] >= start_time) & (df_full['DateTime'] <= end_time)
    df_subset = df_full.loc[mask].copy()

    if df_subset.empty:
        print(f"No data found for range {start_time} to {end_time}")
        return None

    # 4. Estimate Sampling Frequency (fs)
    median_diff = df_subset['DateTime'].diff().dt.total_seconds().median()
    fs = 1.0 / median_diff if (not pd.isna(median_diff) and median_diff > 0) else 0.2

    # 5. Filter & Plot
    sos = design_butterworth_filter(fs, period_minutes=filter_minutes)
    df_subset = apply_butterworth_filter(df_subset, column_name, sos)
    plot_filtered_data(df_subset, column_name, title=f"{column_name}: {start_time} to {end_time}")

    return df_subset


# ================================
# TRANSFER FUNCTION FITTING
# ================================

def simulate_system(t_actual, model_type, params):
    """
    Simulates step response on a dense grid and interpolates to actual time points.
    """
    # Unpack Delay (Td) and Baseline (y0) - always the last two
    Td = params[-2]
    y0 = params[-1]

    # Define Transfer Function G(s) = num / den
    if model_type == 'FOPTD':
        # G(s) = K / (Tp1*s + 1)
        K, Tp1 = params[0], params[1]
        num = [K]
        den = [Tp1, 1]

    elif model_type == 'SOPTD':
        # G(s) = K / ((Tp1*s + 1)(Tp2*s + 1))
        K, Tp1, Tp2 = params[0], params[1], params[2]
        num = [K]
        den = np.convolve([Tp1, 1], [Tp2, 1])

    elif model_type == 'TOPTD':
        # G(s) = K / ((Tp1*s + 1)(Tp2*s + 1)(Tp3*s + 1))
        K, Tp1, Tp2, Tp3 = params[0], params[1], params[2], params[3]
        num = [K]
        den_temp = np.convolve([Tp1, 1], [Tp2, 1])
        den = np.convolve(den_temp, [Tp3, 1])

    elif model_type == 'FOPTDLD':
        # G(s) = K(Tz*s + 1) / (Tp1*s + 1)
        K, Tp1, Tz = params[0], params[1], params[2]
        num = [K * Tz, K]
        den = [Tp1, 1]

    elif model_type == 'SOPTDLD':
        # G(s) = K(Tz*s + 1) / ((Tp1*s + 1)(Tp2*s + 1))
        K, Tp1, Tp2, Tz = params[0], params[1], params[2], params[3]
        num = [K * Tz, K]
        den = np.convolve([Tp1, 1], [Tp2, 1])

    else:
        return np.zeros_like(t_actual)

    # --- SIMULATION & INTERPOLATION ---
    # 1. Create perfect time grid
    t_max = np.max(t_actual)
    t_perfect = np.linspace(0, t_max + Td + 10, 2000)

    # 2. Simulate delay-free response
    system = signal.TransferFunction(num, den)
    _, y_perfect = signal.step(system, T=t_perfect)

    # 3. Shift by Delay (Td) and Interpolate
    t_shifted = t_actual - Td
    y_interpolated = np.interp(t_shifted, t_perfect, y_perfect, left=0)

    # 4. Add Baseline
    return y0 + y_interpolated


# ================================
# WRAPPER FUNCTIONS FOR MODELLING
# ================================
def func_foptd(t, K, Tp1, Td, y0):
    return simulate_system(t, 'FOPTD', [K, Tp1, Td, y0])


def func_soptd(t, K, Tp1, Tp2, Td, y0):
    return simulate_system(t, 'SOPTD', [K, Tp1, Tp2, Td, y0])


def func_toptd(t, K, Tp1, Tp2, Tp3, Td, y0):
    return simulate_system(t, 'TOPTD', [K, Tp1, Tp2, Tp3, Td, y0])


def func_foptdld(t, K, Tp1, Tz, Td, y0):
    return simulate_system(t, 'FOPTDLD', [K, Tp1, Tz, Td, y0])


def func_soptdld(t, K, Tp1, Tp2, Tz, Td, y0):
    return simulate_system(t, 'SOPTDLD', [K, Tp1, Tp2, Tz, Td, y0])


# ================================
# MAIN FITTING FUNCTION
# ================================
def fit_any_model(df, model_name, start_time, end_time,
                  raw_col='MCP_WQ_DO', filtered_col='MCP_WQ_DO_filtered'):
    # --- Data Prep ---
    mask = (df['DateTime'] >= start_time) & (df['DateTime'] <= end_time)
    df_slice = df.loc[mask].copy()

    if df_slice.empty:
        print(f"No data for {model_name}")
        return None

    t_abs = df_slice['DateTime']
    t_rel = (t_abs - t_abs.iloc[0]).dt.total_seconds().values
    y_filt = df_slice[filtered_col].values
    y_raw = df_slice[raw_col].values

    # --- Initial Guesses ---
    y_start = np.mean(y_filt[:5])
    y_end = np.mean(y_filt[-5:])
    K_g = y_end - y_start
    T_total = t_rel[-1]
    Tp_g = T_total / 4
    Td_g = 10.0
    Tz_g = 1.0

    # --- Model Configuration ---
    if model_name == 'FOPTD':
        func = func_foptd
        # Params: K, Tp1, Td, y0
        p0 = [K_g, Tp_g, Td_g, y_start]
        bounds = ([-np.inf, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf])
        eqn_label = r'$G(s) = \frac{K}{T_{p1}s+1}e^{-T_ds}$'
        param_names = ['K', 'Tp1', 'Td', 'y0']

    elif model_name == 'SOPTD':
        func = func_soptd
        # Params: K, Tp1, Tp2, Td, y0
        p0 = [K_g, Tp_g, Tp_g, Td_g, y_start]
        bounds = ([-np.inf, 0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])
        eqn_label = r'$G(s) = \frac{K}{(T_{p1}s+1)(T_{p2}s+1)}e^{-T_ds}$'
        param_names = ['K', 'Tp1', 'Tp2', 'Td', 'y0']

    elif model_name == 'TOPTD':
        func = func_toptd
        # Params: K, Tp1, Tp2, Tp3, Td, y0
        p0 = [K_g, Tp_g, Tp_g, Tp_g, Td_g, y_start]
        bounds = ([-np.inf, 0, 0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        eqn_label = r'$G(s) = \frac{K}{\Pi(T_{pi}s+1)}e^{-T_ds}$'
        param_names = ['K', 'Tp1', 'Tp2', 'Tp3', 'Td', 'y0']

    elif model_name == 'FOPTDLD':
        func = func_foptdld
        # Params: K, Tp1, Tz, Td, y0
        p0 = [K_g, Tp_g, Tz_g, Td_g, y_start]
        bounds = ([-np.inf, 0, -np.inf, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])
        eqn_label = r'$G(s) = \frac{K(T_zs+1)}{T_{p1}s+1}e^{-T_ds}$'
        param_names = ['K', 'Tp1', 'Tz', 'Td', 'y0']

    elif model_name == 'SOPTDLD':
        func = func_soptdld
        # Params: K, Tp1, Tp2, Tz, Td, y0
        p0 = [K_g, Tp_g, Tp_g, Tz_g, Td_g, y_start]
        bounds = ([-np.inf, 0, 0, -np.inf, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        eqn_label = r'$G(s) = \frac{K(T_zs+1)}{\Pi(T_{pi}s+1)}e^{-T_ds}$'
        param_names = ['K', 'Tp1', 'Tp2', 'Tz', 'Td', 'y0']

    else:
        return

    # --- Fitting ---
    try:
        popt, pcov = curve_fit(func, t_rel, y_filt, p0=p0, bounds=bounds, maxfev=10000)
    except Exception as e:
        print(f"Fit failed for {model_name}: {e}")
        return None

    # --- Metrics ---
    y_model = func(t_rel, *popt)
    r2 = r2_score(y_filt, y_model)
    rmse = np.sqrt(mean_squared_error(y_filt, y_model))

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(t_abs, y_raw, color='lightgray', label='Raw Data')
    plt.plot(t_abs, y_filt, color='red', linestyle=':', label='Filtered Data', linewidth=2)
    plt.plot(t_abs, y_model, color='blue', linewidth=2, label=f'{model_name} (R2={r2:.3f})')

    plt.title(f"{model_name} System ID\nRange: {start_time} - {end_time}")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Create Info Box Text
    param_str_lines = [f"{n}: {v:.2f}" for n, v in zip(param_names, popt)]
    info_text = f"{eqn_label}\nRMSE: {rmse:.3f}\n" + "\n".join(param_str_lines)

    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    plt.gca().text(0.02, 0.95, info_text, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.show()

    return popt