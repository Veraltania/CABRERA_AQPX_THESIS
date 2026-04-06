import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def plot_tds_experiment_pairs(base_dirs, dates, factor,
                              pump_off_slices, step_response_slices, data_column,
                              data_name, label="TDS", y_top_zoom=(200, 250), y_bottom_zoom=(0, 10)):
    """
    Reads multi-day data, concatenates it, and generates separate plots
    for each paired (Pump Off + Step Response) experiment window.
    Features a broken Y-axis styled with horizontal dashed red lines.
    """
    dfs = []

    # Iterate through the provided dates and base directories
    for base_dir, date_str in zip(base_dirs, dates):
        filename = f"AQPX_data_log_{date_str}.csv"
        csv_path = Path(base_dir) / filename

        if not csv_path.exists():
            print(f"Warning: Could not find {csv_path}. Skipping.")
            continue

        # Load CSV
        df = pd.read_csv(csv_path)

        # Clean Time column
        df["Time"] = df["Time"].astype(str).str.strip()

        # Combine date + time into a proper datetime
        df["Datetime"] = pd.to_datetime(
            date_str + " " + df["Time"],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce"
        )
        dfs.append(df)

    if not dfs:
        print("No valid data files were loaded. Exiting.")
        return

    # Combine all daily dataframes into one continuous timeline
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.dropna(subset=["Datetime"])
    full_df = full_df.sort_values(by="Datetime")

    # Ensure the number of pump off and step response slices match
    if len(pump_off_slices) != len(step_response_slices):
        print("Warning: The number of Pump Off slices does not match Step Response slices.")

    # Iterate through each paired experiment phase
    for idx, (pump_off, step_resp) in enumerate(zip(pump_off_slices, step_response_slices)):

        # Define the start and end of this specific experiment window
        window_start = pd.to_datetime(pump_off[0])
        window_end = pd.to_datetime(step_resp[1])

        # Filter the dataframe for just this window
        mask = (full_df["Datetime"] >= window_start) & (full_df["Datetime"] <= window_end)
        experiment_df = full_df[mask]

        if experiment_df.empty:
            print(f"Warning: No data found for Experiment Pair {idx + 1} ({window_start} to {window_end}).")
            continue

        # Downscale values for this specific slice
        dtime_small = experiment_df["Datetime"].iloc[::factor]
        tds_small = experiment_df[data_column].iloc[::factor]

        # =========================================================
        # Broken Y-Axis Plotting Logic (Image Style)
        # =========================================================
        # Create two subplots sharing the X-axis.
        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            sharex=True,
            figsize=(12, 6),
            gridspec_kw={'height_ratios': [4, 1]}
        )

        # Increase the horizontal space to make the visual gap larger
        fig.subplots_adjust(hspace=0.15)

        # Plot the main data on BOTH axes
        ax1.plot(dtime_small, tds_small, label=label, color="#d95f02", linewidth=2)
        ax2.plot(dtime_small, tds_small, label=label, color="#d95f02", linewidth=2)

        # Plot the shaded regions on BOTH axes
        for ax in (ax1, ax2):
            ax.axvspan(pd.to_datetime(pump_off[0]), pd.to_datetime(pump_off[1]),
                       color='red', alpha=0.2, label='Pump Off')
            ax.axvspan(pd.to_datetime(step_resp[0]), pd.to_datetime(step_resp[1]),
                       color='green', alpha=0.2, label='Step Response')

        # Limit the view to the different portions
        ax1.set_ylim(*y_top_zoom)  # Main data range
        ax2.set_ylim(*y_bottom_zoom)  # Base range

        # Hide the spines between ax1 and ax2
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        # Adjust tick placement so they don't overlap the break and set font size
        ax1.xaxis.tick_top()
        ax1.tick_params(axis='both', labeltop=False, labelsize=16)  # Keep top labels hidden, set y-axis font size
        ax2.xaxis.tick_bottom()
        ax2.tick_params(axis='both', labelsize=16) # Set x and y-axis font size

        # Add the thick, dashed red lines indicating the break
        # Places a line at the bottom of the top plot, and top of the bottom plot
        ax1.axhline(y=y_top_zoom[0], color='red', linestyle='--', linewidth=2.5, zorder=10)
        ax2.axhline(y=y_bottom_zoom[1], color='red', linestyle='--', linewidth=2.5, zorder=10)

        # =========================================================
        # Formatting & Labels
        # =========================================================
        ax2.set_xlabel("Datetime", fontsize=16)
        fig.supylabel(data_name, fontsize=16)  # Shared Y-axis label

        # Title specific to the pair
        start_date_str = window_start.strftime('%b %d')
        ax1.set_title(f"{data_name} vs Time - Phase Pair {idx + 1} ({start_date_str})", fontsize=16, pad=15)

        # Format the X-axis for clearer multi-day viewing (e.g., every 4 hours to reduce clutter)
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))

        # Rotate x-axis labels on the bottom plot
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        # Handle legend to avoid duplicate entries from plotting on two axes
        handles, labels = ax1.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax1.legend(unique_labels.values(), unique_labels.keys(), loc="upper left", fontsize=16)

        # Grid lines (optional: turn off horizontal grids if they clash with the red dashes)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax2.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # List of dates involved in the experiment
    dates = [
        "2026-02-09",
        "2026-02-10",
        "2026-02-11",
        "2026-02-12"
    ]

    # List of base directories (must match the length of `dates`)
    common_dir = Path(r"D:\aqpx\Cabrera_Thesis_AQPX\Transfer_Function_Modeling\data_calibrated")
    base_dirs = [common_dir, common_dir, common_dir, common_dir]

    # Downscale factor
    downscale_factor = 20

    # Time slices for visual separators
    pump_off_times = [
        ("2026-02-09 09:00:00", "2026-02-09 20:59:59"),
        ("2026-02-10 09:00:00", "2026-02-10 20:59:59"),
        ("2026-02-11 09:00:00", "2026-02-11 20:59:59")
    ]

    step_response_times = [
        ("2026-02-09 21:00:00", "2026-02-10 08:59:59"),
        ("2026-02-10 21:00:00", "2026-02-11 08:59:59"),
        ("2026-02-11 21:00:00", "2026-02-12 08:59:59")
    ]

    # Define ranges for the broken y-axis
    top_y_range = (200, 250)
    bottom_y_range = (0, 10)

    plot_tds_experiment_pairs(
        base_dirs=base_dirs,
        dates=dates,
        factor=downscale_factor,
        pump_off_slices=pump_off_times,
        step_response_slices=step_response_times,
        data_name="Total Dissolved Solids",
        data_column="MCP_WQ_TDS",
        y_top_zoom=top_y_range,
        y_bottom_zoom=bottom_y_range
    )