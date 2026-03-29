import pandas as pd
from pathlib import Path

def compute_average(base_dir, date, column_name, start_time, end_time):
    # Construct file path using the Path object
    file_name = f"AQPX_data_log_{date}.csv"
    full_path = base_dir / file_name

    print(f"Loading data from: {full_path}")

    try:
        df = pd.read_csv(full_path)
    except FileNotFoundError:
        print(f"Error: Could not find the file at {full_path}")
        return

    # Clean the Time column
    df["Time"] = df["Time"].astype(str).str.strip()

    # Combine date + time into a proper datetime column
    df['Datetime'] = pd.to_datetime(
        date + " " + df["Time"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    # Convert start_time and end_time to full datetime objects for comparison
    start_dt = pd.to_datetime(f"{date} {start_time}")
    end_dt = pd.to_datetime(f"{date} {end_time}")

    # Filter the dataframe for the specified interval
    mask = (df['Datetime'] >= start_dt) & (df['Datetime'] <= end_dt)
    filtered_df = df.loc[mask]

    if filtered_df.empty:
        print("Warning: No data found within the specified time interval.")
        return

    # Compute average
    average_val = filtered_df[column_name].mean()

    print(f"\nAverage {column_name} from {start_time} to {end_time} on {date}: {average_val:.4f}")


if __name__ == "__main__":
    base_dir_val = Path(r"D:\aqpx\Cabrera_Thesis_AQPX\Transfer_Function_Modeling\data")

    date_val = "2025-12-19"
    column_val = "MCP_WQ_DO"

    start_val = "11:30:00"  # Format: HH:MM:SS
    end_val = "12:00:00"  # Format: HH:MM:SS
    # ==========================================

    compute_average(
        base_dir=base_dir_val,
        date=date_val,
        column_name=column_val,
        start_time=start_val,
        end_time=end_val
    )