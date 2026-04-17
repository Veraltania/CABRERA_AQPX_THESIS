import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def plot_data(base_dir, date_str, filename, factor,
              pump_off_slices, step_response_slices, data_column,
              data_name, label="DO (mg/L)"):
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

    # Drop invalid times and sort chronologically
    df = df.dropna(subset=["Datetime"])
    df = df.sort_values(by="Datetime")

    # Downscale values
    dtime_small = df["Datetime"].iloc[::factor]
    do_small = df[data_column].iloc[::factor]

    # Plotting - 7.16 inches width for vector format standard
    plt.figure(figsize=(7.16, 4.5))
    plt.plot(dtime_small, do_small, label=label)

    # Standardize the y-axis from 0 to 8
    plt.ylim(0, 8)

    # Add visual separators for "Pump Off" time slices
    for idx, (start, end) in enumerate(pump_off_slices):
        start_dt = pd.to_datetime(f"{date_str} {start}")
        end_dt = pd.to_datetime(f"{date_str} {end}")

        # Only add the label to the legend once to avoid duplicates
        label_text = "Pump Off" if idx == 0 else None

        # Plot a red shaded region for the pump off slice
        plt.axvspan(start_dt, end_dt, color='red', alpha=0.2, label=label_text)

    # Add visual separators for "Step Response" time slices
    for idx, (start, end) in enumerate(step_response_slices):
        start_dt = pd.to_datetime(f"{date_str} {start}")
        end_dt = pd.to_datetime(f"{date_str} {end}")

        # Only add the label to the legend once
        label_text = "Step Response" if idx == 0 else None

        # Plot a green shaded region for the step response slice
        plt.axvspan(start_dt, end_dt, color='green', alpha=0.2, label=label_text)

    plt.xlabel("Time", fontsize=12)
    plt.ylabel(f"{data_name}", fontsize=12)
    plt.title(f"{data_name} vs Time ({date_str})", fontsize=12)

    # Change to every 2 hours to reduce clutter
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.xticks(rotation=45)

    # Set x and y-axis tick font sizes
    plt.tick_params(axis='both', labelsize=12)

    # Legend outside the graph at the bottom
    plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3)
    
    plt.tight_layout()
    
    # Save as PDF vector format
    output_pdf = Path(base_dir) / f"{Path(filename).stem}_DO_plot.pdf"
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    print(f"Saved plot to: {output_pdf}")
    plt.close()


if __name__ == '__main__':
    # =========================================================
    # INPUT PARAMETERS
    # =========================================================
    date_str = "2026-02-05"
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
        ("10:00:00", "14:00:00"),
        ("20:00:00", "23:59:59")
    ]
    # =========================================================

    plot_data(
        base_dir=base_dir,
        date_str=date_str,
        filename=filename,
        factor=downscale_factor,
        pump_off_slices=pump_off_times,
        step_response_slices=step_response_times,
        data_name="Dissolved Oxygen (mg/L)",
        data_column="MCP_WQ_DO"
    )