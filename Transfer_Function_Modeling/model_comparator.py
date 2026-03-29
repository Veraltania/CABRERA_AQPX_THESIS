import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path

def foptd(t, K, tau, theta):
    tau = max(1e-8, tau)
    theta = max(0, theta)
    return np.array([K * (1 - np.exp(-(t_val - theta) / tau)) if t_val >= theta else 0 for t_val in t])

def soptd(t, K, tau1, tau2, theta):
    tau1, tau2 = max(1e-8, tau1), max(1e-8, tau2)
    theta = max(0, theta)
    if abs(tau1 - tau2) < 1e-6: tau1 += 1e-6
    return np.array([
        K * (1 - (tau1 * np.exp(-(t_val - theta) / tau1) - tau2 * np.exp(-(t_val - theta) / tau2)) / (tau1 - tau2))
        if t_val >= theta else 0 for t_val in t
    ])


def foptdld(t, K, tau_p, tau_z, theta):
    tau_p = max(1e-8, tau_p)
    theta = max(0, theta)
    return np.array([
        K * (1 + (tau_z / tau_p - 1) * np.exp(-(t_val - theta) / tau_p))
        if t_val >= theta else 0 for t_val in t
    ])


def soptdld(t, K, tau1, tau2, tau_z, theta):
    tau1, tau2 = max(1e-8, tau1), max(1e-8, tau2)
    theta = max(0, theta)
    if abs(tau1 - tau2) < 1e-6: tau1 += 1e-6
    return np.array([
        K * (1 + ((tau_z - tau1) * np.exp(-(t_val - theta) / tau1) - (tau_z - tau2) * np.exp(
            -(t_val - theta) / tau2)) / (tau1 - tau2))
        if t_val >= theta else 0 for t_val in t
    ])


import numpy as np


def toptd(t, K, tau1, tau2, tau3, theta):
    # Ensure t is a numpy array
    t = np.asanyarray(t)

    # Clip parameters to physical limits
    tau1, tau2, tau3 = max(1e-8, tau1), max(1e-8, tau2), max(1e-8, tau3)
    theta = max(0, theta)

    # Avoid division by zero with minimal impact on accuracy
    if abs(tau1 - tau2) < 1e-7: tau2 += 2e-7
    if abs(tau1 - tau3) < 1e-7: tau3 += 3e-7
    if abs(tau2 - tau3) < 1e-7: tau3 += 4e-7

    # Calculate time since delay
    t_adj = t - theta

    # Create a mask for values after the time delay
    mask = t_adj > 0
    res = np.zeros_like(t, dtype=float)

    # Vectorized calculation
    t_active = t_adj[mask]
    term1 = (tau1 ** 2 * np.exp(-t_active / tau1)) / ((tau1 - tau2) * (tau1 - tau3))
    term2 = (tau2 ** 2 * np.exp(-t_active / tau2)) / ((tau2 - tau1) * (tau2 - tau3))
    term3 = (tau3 ** 2 * np.exp(-t_active / tau3)) / ((tau3 - tau1) * (tau3 - tau2))

    res[mask] = K * (1 - (term1 + term2 + term3))
    return res

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_tf_latex(name, params):
    """Generates a Mathtext LaTeX string of the transfer function for plotting."""
    if name == 'FOPTD':
        K, tau, theta = params
        return fr"$G(s) = \frac{{{K:.3f}}}{{{tau:.3f}s + 1}} e^{{-{theta:.3f}s}}$"
    elif name == 'SOPTD':
        K, tau1, tau2, theta = params
        return fr"$G(s) = \frac{{{K:.3f}}}{{({tau1:.3f}s + 1)({tau2:.3f}s + 1)}} e^{{-{theta:.3f}s}}$"
    elif name == 'FOPTDLD':
        K, tau_p, tau_z, theta = params
        return fr"$G(s) = {K:.3f} \frac{{{tau_z:.3f}s + 1}}{{{tau_p:.3f}s + 1}} e^{{-{theta:.3f}s}}$"
    elif name == 'SOPTDLD':
        K, tau1, tau2, tau_z, theta = params
        return fr"$G(s) = {K:.3f} \frac{{{tau_z:.3f}s + 1}}{{({tau1:.3f}s + 1)({tau2:.3f}s + 1)}} e^{{-{theta:.3f}s}}$"
    elif name == 'TOPTD':
        K, tau1, tau2, tau3, theta = params
        return fr"$G(s) = \frac{{{K:.3f}}}{{({tau1:.3f}s + 1)({tau2:.3f}s + 1)({tau3:.3f}s + 1)}} e^{{-{theta:.3f}s}}$"
    return ""


def compute_acf_symmetric(residuals, max_lag=200):
    """Computes a two-sided (symmetric) Autocorrelation Function for residuals."""
    n = len(residuals)
    mean_res = np.mean(residuals)
    var_res = np.var(residuals)

    if var_res == 0:
        return np.arange(-max_lag, max_lag + 1), np.zeros(2 * max_lag + 1)

    acf_full = np.correlate(residuals - mean_res, residuals - mean_res, mode='full')

    center = n - 1
    acf_full = acf_full / acf_full[center]

    actual_max_lag = min(max_lag, n - 1)

    acf_symmetric = acf_full[center - actual_max_lag: center + actual_max_lag + 1]
    lags = np.arange(-actual_max_lag, actual_max_lag + 1)

    return lags, acf_symmetric


# ==========================================
# 3. DATA PROCESSING & OPTIMIZATION LOGIC
# ==========================================
def extract_and_process(file_paths, start_time, end_time, target_column):
    """Extracts raw data within the specified timeframe."""
    if isinstance(file_paths, str): file_paths = [file_paths]
    dfs = [pd.read_csv(fp) for fp in file_paths if os.path.exists(fp)]
    if not dfs: raise ValueError("No valid data files provided.")

    df = pd.concat(dfs, ignore_index=True)
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df = df.sort_values(by='Datetime')

    mask = (df['Datetime'] >= pd.to_datetime(start_time)) & (df['Datetime'] <= pd.to_datetime(end_time))
    df_filtered = df.loc[mask].copy()

    df_filtered.set_index('Datetime', inplace=True)
    df_filtered.dropna(subset=[target_column], inplace=True)

    return df_filtered.reset_index()


def err_generic(X, t, y, model_func):
    z = model_func(t, *X)
    if len(t) > 0:
        return sum(abs(z - y)) * (max(t) - min(t)) / len(t)
    return float('inf')


def compare_models(file_paths, start_step, end_step, target_column,
                   tf_name, t_step_time, delta_u, custom_max_lag=200):
    print(f"--- Running Multi-Model Comparison for {tf_name} ---")

    df = extract_and_process(file_paths, start_step, end_step, target_column)
    step_dt = pd.to_datetime(t_step_time)
    t_rel = (df['Datetime'] - step_dt).dt.total_seconds().values

    y_raw = df[target_column].values

    pre_step_mask = t_rel <= 0
    y0 = np.mean(y_raw[pre_step_mask]) if pre_step_mask.any() else y_raw[0]

    fit_mask = t_rel >= 0
    ts = t_rel[fit_mask]
    y_fit = y_raw[fit_mask]
    ys = (y_fit - y0) / delta_u

    K_g = ys[-1] if len(ys) > 0 else 1.0
    tau_g = (ts[-1] - ts[0]) / 3.0 if len(ts) > 0 else 10.0

    models_to_fit = {
        'FOPTD': {'func': foptd, 'guess': [K_g, tau_g, 0.0]},
        'SOPTD': {'func': soptd, 'guess': [K_g, tau_g / 2, tau_g / 2, 0.0]},
        'FOPTDLD': {'func': foptdld, 'guess': [K_g, tau_g, tau_g / 2, 0.0]},
        'SOPTDLD': {'func': soptdld, 'guess': [K_g, tau_g / 2, tau_g / 2, tau_g / 4, 0.0]},
        'TOPTD': {'func': toptd, 'guess': [K_g, tau_g / 3, tau_g / 3, tau_g / 3, 0.0]}
    }

    results = {}
    save_dir = Path("Transfer_Functions_Outputs") / tf_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Main Time Series Comparison Plot Setup
    fig_main, ax_main = plt.subplots(figsize=(14, 8))
    t_seconds = (df['Datetime'] - df['Datetime'].iloc[0]).dt.total_seconds().values
    ax_main.plot(t_seconds, y_raw, color='gray', alpha=0.4, label=f"Raw Data", linewidth=1.5)

    # 2. Main ACF Comparison Plot Setup (Symmetric Lines)
    fig_acf_all, ax_acf_all = plt.subplots(figsize=(12, 5))
    conf_interval = 1.96 / np.sqrt(len(y_raw))  # 95% Confidence Interval

    ax_acf_all.axhline(0, color='black', linewidth=1)
    ax_acf_all.axhline(conf_interval, color='blue', linestyle=':', label='95% Confidence')
    ax_acf_all.axhline(-conf_interval, color='blue', linestyle=':')

    colors = ['red', 'blue', 'm', 'green', 'orange']

    print("\n[ Model Performance ]")
    print(f"{'Model':<10} | {'IAE Score':<12} | {'R2 (Raw)':<10} | {'Optimal Parameters (K, Taus..., Theta)'}")
    print("-" * 85)

    # NEW: List to store data for the CSV export
    summary_data = []

    for i, (name, config) in enumerate(models_to_fit.items()):
        res = minimize(err_generic, config['guess'], args=(ts, ys, config['func']))

        X_opt = list(res.x)
        if name != 'FOPTDLD':
            X_opt = [max(1e-8 if j < len(X_opt) - 1 and j != 0 else 0, val) for j, val in enumerate(res.x)]

        iae_score = res.fun

        # Reconstruct timeline
        y_model = np.ones_like(t_rel) * y0
        z_opt = config['func'](ts, *res.x)
        y_model[fit_mask] = y0 + (z_opt * delta_u)

        # Calculate R2 & Residuals
        residuals = y_raw - y_model
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_raw - np.mean(y_raw)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)

        # Plot onto main Multi-Model Time Series
        ax_main.plot(t_seconds, y_model, color=colors[i], linestyle='--', label=f"{name} (R2: {r2_score:.2f})",
                     linewidth=2.5)

        # --- GENERATE INDIVIDUAL TIME SERIES PLOT ---
        fig_ind, ax_ind = plt.subplots(figsize=(12, 6))
        ax_ind.plot(t_seconds, y_raw, color='gray', alpha=0.4, label="Raw Data", linewidth=1.5)
        ax_ind.plot(t_seconds, y_model, color=colors[i], linestyle='--', label=f"{name} Fit (R2: {r2_score:.3f})",
                    linewidth=2.5)

        tf_equation = get_tf_latex(name, X_opt)
        ax_ind.set_title(f"Individual Model Fit: {name} for {tf_name}\n{tf_equation}", fontsize=14, pad=15)
        ax_ind.set_xlabel("Time (seconds)")
        ax_ind.set_ylabel(target_column)

        # Enforce x and y axes to start at 0
        ax_ind.set_xlim(left=0)
        ax_ind.set_ylim(bottom=0)

        ax_ind.legend()
        ax_ind.grid(True, alpha=0.3)
        fig_ind.savefig(save_dir / f"{tf_name}_{name}_individual_fit.png", dpi=300, bbox_inches='tight')
        plt.close(fig_ind)

        # --- CALCULATE & PROCESS SYMMETRIC ACF ---
        lags, acf_vals = compute_acf_symmetric(residuals, max_lag=custom_max_lag)

        # Add to Multi-Model ACF Plot
        ax_acf_all.plot(lags, acf_vals, color=colors[i], label=f"{name}", linewidth=2)

        # Generate Individual ACF Plot
        fig_acf_ind, ax_acf_ind = plt.subplots(figsize=(10, 4))
        ax_acf_ind.plot(lags, acf_vals, color=colors[i], linewidth=2)

        ax_acf_ind.axhline(0, color='black', linewidth=1)
        ax_acf_ind.axhline(conf_interval, color='blue', linestyle=':', label='95% Confidence')
        ax_acf_ind.axhline(-conf_interval, color='blue', linestyle=':')

        ax_acf_ind.set_title(f"Autocorrelation of residuals for {name}", fontsize=12)
        ax_acf_ind.set_ylim(-0.5, 1.0)
        ax_acf_ind.set_xlim(-custom_max_lag, custom_max_lag)

        fig_acf_ind.savefig(save_dir / f"{tf_name}_{name}_acf_individual.png", dpi=300, bbox_inches='tight')
        plt.close(fig_acf_ind)

        # Save results to dictionary and list for CSV
        results[name] = {'IAE': iae_score, 'R2': r2_score, 'params': X_opt}
        param_str = ", ".join([f"{p:.3f}" for p in X_opt])
        print(f"{name:<10} | {iae_score:<12.3f} | {r2_score:<10.3f} | [{param_str}]")

        # Append to summary data
        summary_data.append({
            'Model': name,
            'IAE_Score': iae_score,
            'R2_Score': r2_score,
            'Optimal_Parameters': [round(p, 4) for p in X_opt]
        })

    # Finalize Multi-Model Time Series
    ax_main.set_title(f"Multi-Model Comparison: {tf_name}")
    ax_main.set_xlabel("Time (seconds)")
    ax_main.set_ylabel(target_column)

    # Enforce x and y axes to start at 0
    ax_main.set_xlim(left=0)
    ax_main.set_ylim(bottom=0)

    ax_main.legend()
    ax_main.grid(True, alpha=0.3)
    fig_main.savefig(save_dir / f"{tf_name}_all_models_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig_main)

    # Finalize Multi-Model ACF
    ax_acf_all.set_title(f"Autocorrelation of residuals for all models")
    ax_acf_all.set_ylim(-0.5, 1.0)
    ax_acf_all.set_xlim(-custom_max_lag, custom_max_lag)
    ax_acf_all.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    fig_acf_all.savefig(save_dir / f"{tf_name}_all_models_acf_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig_acf_all)

    # Export summary to CSV
    summary_df = pd.DataFrame(summary_data)
    csv_path = save_dir / f"{tf_name}_model_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    print(f"\nAll plots and CSV summary saved to: {save_dir}")
    return results

if __name__ == "__main__":
    date = "2026-02-26"
    tf_name = f"DO_DAYTIME_{date}_TEST"
    base_dir = Path(r"D:/aqpx/Cabrera_Thesis_AQPX/Transfer_Function_Modeling/data_calibrated")
    do_file = [base_dir / f"AQPX_data_log_{date}.csv"]
    start_step = "10:00:00"
    end_step = "13:00:00"

    compare_models(
        file_paths=do_file,
        start_step=f'{date} {start_step}',
        end_step=f'{date} {end_step}',
        target_column='MCP_WQ_DO',
        tf_name=tf_name,
        t_step_time=f'{date} {start_step}',
        delta_u=1.0,
        custom_max_lag=300
    )