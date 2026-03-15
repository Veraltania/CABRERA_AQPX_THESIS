import csv
import numpy as np
import matplotlib.pyplot as plt
import control as ct
import pygad
from pathlib import Path

# Custom local modules
from Transfer_Function_Analysis.analyze_transfer_func_stability import define_transfer_func, define_guardrail_gain
from Evolutionary_Algorithm_Testing.solver_engine import fast_fbest_diffeq  # Imported Julia DDE solver


# --- 1. HELPER FUNCTIONS ---
def simulate_response(Kp, Ki, plant):
    """Simulates the step response purely for the final plotting."""
    ctrl = ct.TransferFunction([Kp, Ki], [1, 0])
    try:
        sys = ct.feedback(plant * ctrl, 1)
        T_sim = np.linspace(0, 10000, 1000)
        T_sim, y_sim = ct.step_response(sys, T_sim)
        return T_sim, y_sim
    except:
        return None, None


def setup_experiment_dir(folder_name):
    """Creates the output directory and returns the Path object."""
    output_dir = Path(folder_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_plots(output_dir, round_num, history, best_Kp, best_Ki, plant):
    """Generates and saves the convergence and response plots."""
    # Convergence Plot
    plt.figure(figsize=(10, 4))
    plt.plot(history, linewidth=2, color='blue')
    plt.title(f'GA Convergence - Round {round_num}')
    plt.xlabel('Generations')
    plt.ylabel('Cost (from Solver)')
    plt.grid(True)
    plt.savefig(output_dir / f'convergence_round_{round_num}.png')
    plt.close()

    # Step Response Plot
    T_best, y_best = simulate_response(best_Kp, best_Ki, plant)
    plt.figure(figsize=(10, 6))
    if T_best is not None:
        plt.plot(T_best, y_best, 'b-', linewidth=3, label='Best GA Params')
    plt.axhline(1.0, color='r', linestyle='--')
    plt.title(f'Step Response - Round {round_num}')
    plt.grid(True)
    plt.savefig(output_dir / f'response_round_{round_num}.png')
    plt.close()


# --- 2. TRACKING CLASS ---
class OptimizationTracker:
    def __init__(self, patience, tolerance, round_num, cost_function):
        self.patience = patience
        self.tolerance = tolerance
        self.round_num = round_num
        self.cost_function = cost_function
        self.counter = 0
        self.best_global_cost = float('inf')
        self.cost_history = []
        self.stop_triggered = False

    def on_generation(self, ga_instance):
        solution, _, _ = ga_instance.best_solution()
        current_cost = self.cost_function(solution)

        self.cost_history.append(current_cost)
        gen_num = len(self.cost_history)

        if current_cost < 1e8:
            if current_cost < (self.best_global_cost - self.tolerance):
                self.best_global_cost = current_cost
                self.counter = 0
            else:
                self.counter += 1

        print(f"   [Round {self.round_num}] Gen {gen_num}: Cost={current_cost:.4f} "
              f"(Best={self.best_global_cost:.4f}) | Patience: {self.counter}/{self.patience}")

        if self.counter >= self.patience:
            self.stop_triggered = True
            print(f"   --> Stopping Early: No improvement for {self.patience} generations.")
            return "stop"


# --- 3. MAIN EXPERIMENT LOGIC ---
def run_ga_experiment(
        # Configuration Params
        population_size=100,
        patience_limit=25,
        max_iters=200,
        improvement_tol=0.01,  # Lowered to match DE logic (useful for log scale)
        n_rounds=50,
        output_folder="experiment_images_ga",
        # Transfer Function Params (Defaults updated to match DE script for parity)
        tf_num=[-24.44],
        tf_den=[84487.79, 1],
        tf_delay=0.5,
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
            writer.writerow(['Round', 'Iterations_Run', 'Final_Cost', 'Best_Kp', 'Best_Ki'])

    # 2. Define Plant & Guardrails (Matching DE logic)
    is_reverse = tf_num[0] < 0
    plant = define_transfer_func(tf_num, tf_den, tf_delay, tf_n_pade)
    max_kp_guardrail = define_guardrail_gain(plant, find_negative_gain=is_reverse)

    # 3. Define Objective Wrappers (Closures targeting Julia Solver Engine)
    def cost_wrapper(solution):
        return fast_fbest_diffeq(
            Kp_ctrl=solution[0],
            Ki_ctrl=solution[1],
            K_plant=tf_num[0],
            T_plant=tf_den[0],
            delay=tf_delay
        )

    def fitness_wrapper(ga_instance, solution, solution_idx):
        cost = cost_wrapper(solution)
        # PyGAD maximizes fitness. Since we are minimizing cost, returning -cost
        # is the most robust translation, avoiding 1/0 errors or sign-flip issues.
        return -float(cost)

    # 4. Initialize Aggregate History
    agg_history = {
        'iterations': [], 'costs': [], 'kp': [], 'ki': []
    }

    # Dynamic Bounds Based on Plant Direction
    safe_limit = float(max_kp_guardrail) if max_kp_guardrail is not None else (-100.0 if is_reverse else 100.0)
    if is_reverse:
        bounds = [{'low': safe_limit, 'high': -0.001}, {'low': -0.01, 'high': -1e-6}]
    else:
        bounds = [{'low': 0.001, 'high': safe_limit}, {'low': 1e-6, 'high': 0.01}]

    print(f"Calculated Search Bounds (PyGAD): Kp {bounds[0]}, Ki {bounds[1]}")

    # 5. Main Loop
    for trial_no in range(n_rounds):
        current_round = trial_no + 1
        print(f"\n{'=' * 40}\nSTARTING GA ROUND {current_round} OF {n_rounds}\n{'=' * 40}")

        # Setup Tracker with the local cost wrapper
        tracker = OptimizationTracker(patience_limit, improvement_tol, current_round, cost_wrapper)

        ga_instance = pygad.GA(
            num_generations=max_iters,
            num_parents_mating=10,
            fitness_func=fitness_wrapper,  # Uses the local wrapper returning -cost
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
        f"Average Iterations: {np.mean(agg_history['iterations']):.1f}\n"
        f"Average Cost: {np.mean(agg_history['costs']):.2f}\n"
        f"-----------------------------\n"
        f"Kp: {np.mean(agg_history['kp']):.5f} (+/- {np.std(agg_history['kp']):.5f})\n"
        f"Ki: {np.mean(agg_history['ki']):.6f} (+/- {np.std(agg_history['ki']):.6f})\n"
    )
    print(summary_text)
    with open(summary_filename, "w") as f:
        f.write(summary_text)


# --- 4. EXECUTION ---
if __name__ == "__main__":
    run_ga_experiment()