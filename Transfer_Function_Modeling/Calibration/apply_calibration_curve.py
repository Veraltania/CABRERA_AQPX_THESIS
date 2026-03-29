import pandas as pd
from pathlib import Path


def batch_calibrate_do(input_dir: Path, output_dir: Path, slope: float, intercept: float):
    """
    Iterates through CSV files in the input directory, calibrates the
    MCP_WQ_DO column, and saves the new files to the output directory.
    """
    # Create the output folder if it doesn't exist yet (exist_ok=True prevents errors)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory ready: {output_dir}")

    # Find all CSV files in the input directory using pathlib's built-in glob
    csv_files = list(input_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {input_dir}. Check your path!")
        return

    print(f"Found {len(csv_files)} files. Starting calibration...\n")

    for file_path in csv_files:
        # .name extracts just the filename (e.g., 'AQPX_data_log_2025-11-17.csv')
        file_name = file_path.name

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Check if the target column exists to avoid crashing
            if 'MCP_WQ_DO' in df.columns:
                # Apply the calibration equation: y = mx + b
                df['MCP_WQ_DO'] = (df['MCP_WQ_DO'] * slope) + intercept
                # Round to 2 decimal places to keep data clean
                df['MCP_WQ_DO'] = df['MCP_WQ_DO'].round(2)

                # Construct the new save path using the `/` operator
                output_path = output_dir / file_name

                # Save the modified data back to a new CSV
                df.to_csv(output_path, index=False)
                print(f"Success: Calibrated and saved -> {file_name}")
            else:
                print(f"Skipped: 'MCP_WQ_DO' column not found in -> {file_name}")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    print("\nCalibration complete! Check your output folder.")


if __name__ == "__main__":
    # ==========================================
    # SETUP YOUR EQUATION AND FOLDERS HERE
    # ==========================================

    # Your calibration equation variables: DO = (m * Raw) + b
    SLOPE_M = 0.027  # Replace with your chosen slope
    INTERCEPT_B = 2.26  # Replace with your chosen intercept

    # Define your directories using pathlib.Path
    # You can keep the 'r' prefix for Windows paths, Path will parse it correctly for any OS
    INPUT_DIR = Path(r"D:\aqpx\Cabrera_Thesis_AQPX\Transfer_Function_Modeling\data")

    # You can easily construct relative paths using Path as well
    # For example, this puts the calibrated folder right next to the D3 folder
    OUTPUT_DIR = INPUT_DIR.parent / "data_calibrated"

    # Execute the function with our configuration
    batch_calibrate_do(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        slope=SLOPE_M,
        intercept=INTERCEPT_B
    )