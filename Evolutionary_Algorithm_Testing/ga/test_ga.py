import csv
import numpy as np
import matplotlib.pyplot as plt
import control as ct
import pygad
from pathlib import Path

# Assuming these are custom local modules
from Transfer_Function_Analysis.analyze_transfer_func_stability import *


# --- 1. HELPER FUNCTIONS ---
def simulate_response(Kp, Ki, plant):
    """Simulates the step response for a specific set of gains."""
    ctrl = ct.TransferFunction([Kp, Ki], [1, 0])
    try:
        sys = ct.feedback(plant * ctrl, 1)
        T_sim = np.linspace(0, 10000, 1000)
        T_sim, y_sim = ct.step_response(sys, T_sim)
        return T_sim, y_sim
    except:
        return None, None


def calculate_itae_cost(Kp, Ki, plant):
    """Calculates the ITAE cost for a single particle/solution."""
    if Kp < 0 or Ki < 0:
        return 1e9

    try:
        controller = ct.TransferFunction([Kp, Ki], [1, 0])
        closed_loop = ct.feedback(plant * controller, 1)
        T = np.linspace(0, 10000, 1000)
        T, y = ct.step_response(closed_loop, T)
        y = np.asarray(y).flatten()

        e = 1.0 - y
        dt = T[1] - T[0]
        # ITAE: Integral of Time-weighted Absolute Error
        itae = np.sum(T * np.abs(e)) * dt

        # Penalty for overshoot/undershoot
        if np.max(y) > 1.2 or np.min(y) < -0.2:
            itae += 1e9

        return float(itae) if not (np.isnan(itae) or np.isinf(itae)) else 1e9
    except:
        return 1e9


def setup_experiment_dir(folder_name):
    """Creates the output directory and returns the Path object."""
    output_dir = Path(folder_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_plots(output_dir, round_num, history, best_Kp, best_Ki, plant):
    """Generates and saves the convergence and response plots."""
    # Convergence Plot
    plt.figure(figsize=(10, 4))
    plt.plot(history, linewidth=2)
    plt.title(f'GA Convergence - Round {round_num}')
    plt.grid(True)
    plt.savefig(output_dir / f'convergence_round_{round_num}.png')
    plt.close()

    # Step Response Plot
    T_best, y_best = simulate_response(best_Kp, best_Ki, plant)
    plt.figure(figsize=(10, 6))
    if T_best is not None:
        plt.plot(T_best, y_best, 'b-', linewidth=3)
    plt.axhline(1.0, color='r', linestyle='--')
    plt.title(f'Response - Round {round_num}')
    plt.grid(True)
    plt.savefig(output_dir / f'response_round_{round_num}.png')
    plt.close()


# --- 2. TRACKING CLASS ---
class OptimizationTracker:
    def __init__(self, patience, tolerance, round_num, cost_function):
        self.patience = patience
        self.tolerance = tolerance
        self.round_num = round_num
        self.cost_function = cost_function  # Function to calculate cost
        self.counter = 0
        self.best_global_cost = float('inf')
        self.cost_history = []
        self.stop_triggered = False

    def on_generation(self, ga_instance):
        solution, _, _ = ga_instance.best_solution()
        # Use the passed cost function (which closes over 'plant')
        current_cost = self.cost_function(solution)

        self.cost_history.append(current_cost)
        gen_num = len(self.cost_history)

        if current_cost < 1e8:
            if current_cost < (self.best_global_cost - self.tolerance):
                self.best_global_cost = current_cost
                self.counter = 0
            else:
                self.counter += 1

        print(f"   [Round {self.round_num}] Gen {gen_num}: Cost={current_cost:.2f} "
              f"(Best={self.best_global_cost:.2f}) | Patience: {self.counter}/{self.patience}")

        if self.counter >= self.patience:
            self.stop_triggered = True
            return "stop"


# --- 3. MAIN EXPERIMENT LOGIC ---
def run_ga_experiment(
        # Configuration Params
        population_size=100,
        patience_limit=25,
        max_iters=200,
        improvement_tol=1.0,
        n_rounds=50,
        output_folder="experiment_images_ga",
        # Transfer Function Params
        tf_num=[44.93],
        tf_den=[1474.32, 1],
        tf_delay=343.93,
        tf_n_pade=2
):
    """
    Main entry point for running the Genetic Algorithm experiment.
    """

    # 1. Setup Environment
    output_dir = setup_experiment_dir(output_folder)
    csv_filename = output_dir / f"{output_dir.name}_detailed_log.csv"
    summary_filename = output_dir / f"{output_dir.name}_summary.txt"

    if not csv_filename.exists():
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Round', 'Iterations_Run', 'Final_Cost_ITAE', 'Best_Kp', 'Best_Ki'])

    # 2. Define Plant
    plant = define_transfer_func(tf_num, tf_den, tf_delay, tf_n_pade)
    max_kp_guardrail = define_guardrail_gain(plant)

    # 3. Define Objective Wrappers (Closures)
    # These functions capture 'plant' from the local scope, avoiding globals.
    def cost_wrapper(solution):
        return calculate_itae_cost(solution[0], solution[1], plant)

    def fitness_wrapper(ga_instance, solution, solution_idx):
        cost = cost_wrapper(solution)
        return float(1.0 / (cost + 1e-8))

    # 4. Initialize Aggregate History
    agg_history = {
        'iterations': [], 'costs': [], 'kp': [], 'ki': []
    }

    # 5. Main Loop
    for trial_no in range(n_rounds):
        current_round = trial_no + 1
        print(f"\n{'=' * 40}\nSTARTING GA ROUND {current_round} OF {n_rounds}\n{'=' * 40}")

        # Setup Tracker with the local cost wrapper
        tracker = OptimizationTracker(patience_limit, improvement_tol, current_round, cost_wrapper)

        # GA Configuration
        # Note: gene_space expects a list of dicts for separate bounds if using that format
        bounds = [{'low': 0.001, 'high': max_kp_guardrail}, {'low': 0.0, 'high': 0.001}]

        ga_instance = pygad.GA(
            num_generations=max_iters,
            num_parents_mating=10,
            fitness_func=fitness_wrapper,  # Uses the local wrapper
            sol_per_pop=population_size,
            num_genes=2,
            gene_space=bounds,
            parent_selection_type="rank",
            keep_parents=2,
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=20,
            on_generation=tracker.on_generation,
            suppress_warnings=True
        )

        # Run Optimization
        ga_instance.run()

        # Extract Results
        solution, _, _ = ga_instance.best_solution()
        best_Kp, best_Ki = solution
        cost = cost_wrapper(solution)
        iterations_run = len(tracker.cost_history)

        # Store Data
        agg_history['iterations'].append(iterations_run)
        agg_history['costs'].append(cost)
        agg_history['kp'].append(best_Kp)
        agg_history['ki'].append(best_Ki)

        # Save to CSV
        with open(csv_filename, mode='a', newline='') as file:
            csv.writer(file).writerow([current_round, iterations_run, cost, best_Kp, best_Ki])

        # Save Plots
        save_plots(output_dir, current_round, tracker.cost_history, best_Kp, best_Ki, plant)

    # 6. Final Summary
    summary_text = (
        f"--- GA EXPERIMENT SUMMARY ---\n"
        f"Total Rounds: {n_rounds}\n"
        f"Average ITAE Cost: {np.mean(agg_history['costs']):.2f}\n"
        f"Kp: {np.mean(agg_history['kp']):.5f} (+/- {np.std(agg_history['kp']):.5f})\n"
        f"Ki: {np.mean(agg_history['ki']):.6f} (+/- {np.std(agg_history['ki']):.6f})\n"
    )
    print(summary_text)
    with open(summary_filename, "w") as f:
        f.write(summary_text)


# --- 4. EXECUTION ---
if __name__ == "__main__":
    run_ga_experiment(
        # Configuration
        population_size=100,
        patience_limit=25,
        max_iters=200,
        improvement_tol=1.0,
        n_rounds=50,
        output_folder="experiment_images_ga_population_100",

        # Transfer Function
        tf_num=[44.93],
        tf_den=[1474.32, 1],
        tf_delay=343.93,
        tf_n_pade=2
    )