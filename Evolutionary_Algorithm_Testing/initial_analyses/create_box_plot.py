import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def create_box_plots(file_list, cost_file_path, iteration_file_path):
    """
    Reads a list of CSV files and generates box plots for 'Iterations_Run'
    and 'Final_Cost_ITAE'.
    """
    dfs = []

    for file_path in file_list:
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Extract algorithm name from filename (assumes format like 'algo_restOfName.csv')
            # For example: 'de_results...' becomes 'DE'
            filename = os.path.basename(file_path)
            algo_name = filename.split('_')[0].upper()

            # Add a column for the algorithm to distinguish data in the plot
            df['Algorithm'] = algo_name

            # Append to list
            dfs.append(df)
            print(f"Loaded {filename} with {len(df)} rows.")

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    if not dfs:
        print("No valid data loaded.")
        return

    # Concatenate all dataframes into one
    combined_df = pd.concat(dfs, ignore_index=True)

    # Set plot style
    sns.set(style="whitegrid")

    # --- Plot 1: Iterations Run ---
    plt.figure(figsize=(10, 6))
    # Create boxplot: X-axis is the Algorithm, Y-axis is the Iterations
    sns.boxplot(x='Algorithm', y='Iterations_Run', data=combined_df)

    plt.title('Comparison of Iterations Run')
    plt.ylabel('Iterations')
    plt.xlabel('Algorithm')

    # Save the plot
    plt.savefig(iteration_file_path)
    plt.close()
    print(f"Saved {iteration_file_path}")

    # --- Plot 2: Final Cost (ITAE) ---
    plt.figure(figsize=(10, 6))
    # Create boxplot: X-axis is the Algorithm, Y-axis is the Final Cost
    sns.boxplot(x='Algorithm', y='Final_Cost_ITAE', data=combined_df)

    plt.title('Comparison of Final Cost (ITAE)')
    plt.ylabel('Final Cost (ITAE)')
    plt.xlabel('Algorithm')

    # Save the plot
    plt.savefig(cost_file_path)
    plt.close()
    print(f"Saved {cost_file_path}")


# --- usage ---
if __name__ == "__main__":
    # Add your filenames to this list
    files = [
        'de/de_results_batch_2_50trials_nighttime.csv',
        'ga/ga_results_batch_2_50trials_nighttime.csv',
        'pso/pso_results_batch_2_50trials_nighttime.csv'
    ]

    cost_file_path = "final_cost_boxplot_nighttime.png"
    iteration_file_path = "iterations_boxplot_nighttime.png"

    create_box_plots(files, cost_file_path, iteration_file_path)