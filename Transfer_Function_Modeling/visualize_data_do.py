import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def plot_do_data(base_dir, date_str, filename, factor, pump_off_slices, step_response_slices, label="DO (mg/L)"):
    """
    Reads the dissolved oxygen data, downscales it, and plots it with visual separators
    for specified time slices (pump off and step response).
    """
    # Load CSV
    csv_path = Path(base_dir) / filename
    df = pd.read_csv(csv_path)

    # Clean Time column
    df["Time"] = df["Time"].astype(str).str.strip()

    # Combine date + time into a proper datetime
    df["Datetime"] = pd.to_datetime(
        date_str + " " + df["Time"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    # FIX 1: Drop invalid times and sort chronologically!
    # This prevents the line from drawing backwards across the graph.
    df = df.dropna(subset=["Datetime"])
    df = df.sort_values(by="Datetime")

    # Downscale values
    dtime_small = df["Datetime"].iloc[::factor]
    do_small = df["MCP_WQ_DO"].iloc[::factor]

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.plot(dtime_small, do_small, label=label)

    # FIX 2: Force the y-axis to start at 0
    plt.ylim(bottom=0)

    # Add visual separators for "Pump Off" time slices
    for idx, (start, end) in enumerate(pump_off_slices):
        start_dt = pd.to_datetime(f"{date_str} {start}")
        end_dt = pd.to_datetime(f"{date_str} {end}")

        # Only add the label to the legend once to avoid duplicates
        label = "Pump Off" if idx == 0 else None

        # Plot a red shaded region for the pump off slice
        plt.axvspan(start_dt, end_dt, color='red', alpha=0.2, label=label)

    # Add visual separators for "Step Response" time slices
    for idx, (start, end) in enumerate(step_response_slices):
        start_dt = pd.to_datetime(f"{date_str} {start}")
        end_dt = pd.to_datetime(f"{date_str} {end}")

        # Only add the label to the legend once
        label = "Step Response" if idx == 0 else None

        # Plot a green shaded region for the step response slice
        plt.axvspan(start_dt, end_dt, color='green', alpha=0.2, label=label)

    plt.xlabel("Time")
    plt.ylabel("DO (MCP_WQ_DO)")
    plt.title(f"Dissolved Oxygen vs Time ({date_str})")

    # Hourly ticks
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # =========================================================
    # INPUT PARAMETERS
    # =========================================================
    date_str = "2026-02-26"
    base_dir = Path(r"D:\aqpx\Cabrera_Thesis_AQPX\Transfer_Function_Modeling\data_calibrated")
    filename = f"AQPX_data_log_{date_str}.csv"


    # Downscale factor
    downscale_factor = 20

    # Time slices for visual separators
    pump_off_times = [
        ("8:00:00", "9:59:59"),
        ("18:00:00", "20:00:00")
    ]

    step_response_times = [
        ("10:00:00", "13:00:00"),
        ("20:00:00", "23:59:59")
    ]
    # =========================================================

    plot_do_data(
        base_dir=base_dir,
        date_str=date_str,
        filename=filename,
        factor=downscale_factor,
        pump_off_slices=pump_off_times,
        step_response_slices=step_response_times
    )