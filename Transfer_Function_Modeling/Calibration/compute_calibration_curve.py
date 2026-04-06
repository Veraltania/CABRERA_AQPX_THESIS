import numpy as np
import matplotlib.pyplot as plt


def create_calibration_curve(raw_values, actual_values):
    """
    Takes raw analog values and manually recorded DO values,
    calculates the line of best fit, and plots the calibration curve.
    """
    # Convert lists to numpy arrays for easier math
    x = np.array(raw_values)
    y = np.array(actual_values)

    # Calculate linear regression (degree 1 polynomial = straight line)
    # m is the slope, b is the y-intercept
    m, b = np.polyfit(x, y, 1)

    # Calculate R-squared (how well the line fits the data: 1.0 is perfect)
    correlation_matrix = np.corrcoef(x, y)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy ** 2

    # Print the results
    print("=== Calibration Results ===")
    print(f"Formula: DO (mg/L) = {m:.4f} * (Raw Value) + {b:.4f}")
    print(f"Slope (m): {m:.4f}")
    print(f"Intercept (b): {b:.4f}")
    print(f"R-squared: {r_squared:.4f}")

    if r_squared < 0.90:
        print("\nWarning: R-squared is low. Your sensor data might be noisy or non-linear.")
    else:
        print("\nGood fit! Your data points form a strong linear trend.")

    # --- Plotting ---
    # Generate points for the trendline
    trendline_x = np.linspace(min(x) * 0.8, max(x) * 1.1, 100)
    trendline_y = m * trendline_x + b

    plt.figure(figsize=(8, 5))

    # Plot the original data points
    plt.scatter(x, y, color='blue', label='Recorded Samples', zorder=5)

    # Plot the line of best fit
    plt.plot(trendline_x, trendline_y, color='red', linestyle='--',
             label=f'Fit: y = {m:.4f}x + {b:.2f}')

    # Formatting the chart
    plt.title('Dissolved Oxygen Sensor Calibration Curve')
    plt.xlabel('Raw Analog Value')
    plt.ylabel('DO (mg/L)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    # Display the plot
    plt.show()

raw_analog = [60, 139, 143, 160, 280]
manual_do = [2.01, 3.20, 3.34, 3.41, 7.8]

# Run the function
create_calibration_curve(raw_analog, manual_do)