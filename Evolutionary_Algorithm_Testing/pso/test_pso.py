from pathlib import Path
import pyswarms as ps
import csv
import numpy as np
import matplotlib.pyplot as plt
import control as ct

# Import the Julia-based fast solver engine
from Evolutionary_Algorithm_Testing.solver_engine import fast_fbest_diffeq

# Assuming these are custom local modules
from Transfer_Function_Analysis.analyze_transfer_func_stability import *


# --- 1. CUSTOM EXCEPTION ---
class EarlyStopping(Exception):
    def __init__(self, message="Convergence criteria met."):
        super().__init__(message)


# --- 2. HELPER FUNCTIONS ---
def simulate_response(Kp, Ki, K_plant, T_plant, delay):
    """Simulates the step response using the Julia DDE solver."""
    try:
        # We pass arbitrary weights (1.0, 0.05, 1.0) because we only need the arrays here
        _, _, _, y_vals, t_vals = jl.run_dde_solver(Kp, Ki, K_plant, T_plant, delay, 1.0, 0.05, 1.0)

        if len(y_vals) == 0:
            return None, None

        return np.array(t_vals), np.array(y_vals)
    except:
        return None, None


def objective_function(particles, K_plant, T_plant, delay, state, patience_limit, improvement_tol):
    """
    Calculates costs for the swarm using the Julia solver and handles patience/early stopping.
    State is a mutable dictionary passed via kwargs to maintain history across iterations.
    """
    n_particles = particles.shape[0]
    costs = []

    # 1. Calculate cost for each particle
    for i in range(n_particles):
        Kp, Ki = particles[i, 0], particles[i, 1]
        if Kp < 0 or Ki < 0:
            costs.append(1e9)
            continue

        try:
            # Evaluate using compiled DelayDiffEq engine
            cost = fast_fbest_diffeq(Kp, Ki, K_plant, T_plant, delay)
            costs.append(cost if not (np.isnan(cost) or np.isinf(cost)) else 1e9)
        except:
            costs.append(1e9)

    costs_array = np.array(costs)

    # 2. Update State (Best Cost & Patience)
    current_best_in_batch = np.min(costs_array)

    if current_best_in_batch < state['best_global_cost']:
        if current_best_in_batch < 1e8:
            # Check improvement tolerance
            if current_best_in_batch < (state['best_global_cost'] - improvement_tol):
                state['patience_counter'] = 0
            else:
                state['patience_counter'] += 1
            state['best_global_cost'] = current_best_in_batch
    else:
        if state['best_global_cost'] < 1e8:
            state['patience_counter'] += 1

    state['iter_count'] += 1
    state['history'].append(state['best_global_cost'])

    # 3. Logging
    print(f"   [Round {state['current_round']}] Gen {state['iter_count']}: "
          f"Best={state['best_global_cost']:.4f} | "
          f"Patience: {state['patience_counter']}/{patience_limit}")

    # 4. Check Termination
    if state['patience_counter'] >= patience_limit:
        raise EarlyStopping()

    return costs_array


def setup_experiment_dir(folder_name):
    """Creates the output directory and returns the Path object."""
    output_dir = Path(folder_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_plots(output_dir, round_num, history, best_Kp, best_Ki, K_plant, T_plant, delay):
    """Generates and saves the convergence and response plots."""
    # Convergence Plot
    plt.figure(figsize=(10, 4))
    plt.plot(history, linewidth=2)
    plt.title(f'PSO Convergence - Round {round_num}')
    plt.grid(True)
    plt.savefig(output_dir / f'convergence_round_{round_num}.png')
    plt.close()

    # Step Response Plot
    T_best, y_best = simulate_response(best_Kp, best_Ki, K_plant, T_plant, delay)
    plt.figure(figsize=(10, 6))
    if T_best is not None:
        plt.plot(T_best, y_best, 'g-', label='Best Params')
    plt.axhline(1.0, color='r', linestyle='--')
    plt.title(f'Response - Round {round_num}')
    plt.grid(True)
    plt.savefig(output_dir / f'response_round_{round_num}.png')
    plt.close()


# --- 3. MAIN EXPERIMENT LOGIC ---
def run_pso_experiment(
        # Configuration Params
        population_size=100,
        patience_limit=25,
        max_iters=200,
        improvement_tol=0.01,  # Lowered for log-scaled engine output
        n_rounds=50,
        output_folder="experiment_images_pso",
        # Transfer Function Params
        tf_num=[44.93],
        tf_den=[1474.32, 1],
        tf_delay=343.93,
        tf_n_pade=2
):
    """
    Main entry point for running the PSO experiment.
    """

    # 1. Setup Environment
    output_dir = setup_experiment_dir(output_folder)
    csv_filename = output_dir / f"{output_dir.name}_detailed_log.csv"
    summary_filename = output_dir / f"{output_dir.name}_summary.txt"

    # Initialize CSV
    if not csv_filename.exists():
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Round', 'Iterations_Run', 'Final_Cost_Log10', 'Best_Kp', 'Best_Ki'])

    # 2. Define Plant & Extract properties for Julia
    K_plant = tf_num[0]
    T_plant = tf_den[0]
    delay = tf_delay

    # Preserve the control-library plant just for guardrail gain calculations
    plant_ct = define_transfer_func(tf_num, tf_den, tf_delay, tf_n_pade)
    max_kp_guardrail = define_guardrail_gain(plant_ct)

    # 3. Initialize Aggregate History
    agg_history = {
        'iterations': [], 'costs': [], 'kp': [], 'ki': []
    }

    # 4. Main Loop
    for trial_no in range(n_rounds):
        current_round = trial_no + 1
        print(f"\n{'=' * 40}\nSTARTING PSO ROUND {current_round} OF {n_rounds}\n{'=' * 40}")

        # Reset State for this round
        run_state = {
            'best_global_cost': float('inf'),
            'patience_counter': 0,
            'iter_count': 0,
            'history': [],
            'current_round': current_round
        }

        # Optimizer Configuration
        bounds = (np.array([0.001, 0.0]), np.array([max_kp_guardrail, 0.001]))
        options = {'c1': 0.7, 'c2': 0.5, 'w': 0.9}
        optimizer = ps.single.GlobalBestPSO(n_particles=population_size, dimensions=2, options=options, bounds=bounds)

        # Run Optimization
        cost, pos = float('inf'), [0, 0]
        try:
            # Extra arguments passed here flow to objective_function
            cost, pos = optimizer.optimize(
                objective_function,
                iters=max_iters,
                verbose=False,
                K_plant=K_plant,
                T_plant=T_plant,
                delay=delay,
                state=run_state,
                patience_limit=patience_limit,
                improvement_tol=improvement_tol
            )
        except EarlyStopping:
            print("   --> Convergence criteria met.")
            cost, pos = optimizer.swarm.best_cost, optimizer.swarm.best_pos
        except Exception as e:
            print(f"   [ERROR] Unexpected error: {e}")
            if optimizer.swarm:
                cost, pos = optimizer.swarm.best_cost, optimizer.swarm.best_pos

        best_Kp, best_Ki = pos
        iterations_run = len(run_state['history'])

        # Store Data
        agg_history['iterations'].append(iterations_run)
        agg_history['costs'].append(cost)
        agg_history['kp'].append(best_Kp)
        agg_history['ki'].append(best_Ki)

        # Save CSV
        with open(csv_filename, mode='a', newline='') as file:
            csv.writer(file).writerow([current_round, iterations_run, cost, best_Kp, best_Ki])

        # Save Plots
        save_plots(output_dir, current_round, run_state['history'], best_Kp, best_Ki, K_plant, T_plant, delay)

    # 5. Final Summary
    summary_text = (
        f"--- PSO EXPERIMENT SUMMARY ---\n"
        f"Total Rounds: {n_rounds}\n"
        f"Average Cost (Log10 F_best): {np.mean(agg_history['costs']):.4f}\n"
        f"Kp: {np.mean(agg_history['kp']):.5f} (+/- {np.std(agg_history['kp']):.5f})\n"
        f"Ki: {np.mean(agg_history['ki']):.6f} (+/- {np.std(agg_history['ki']):.6f})\n"
    )

    print(summary_text)
    with open(summary_filename, "w") as f:
        f.write(summary_text)


# --- 4. EXECUTION ---
if __name__ == "__main__":
    run_pso_experiment(
        # Configuration
        population_size=100,
        patience_limit=25,
        max_iters=200,
        improvement_tol=0.01,  # Adjusted for log10(F_best)
        n_rounds=50,
        output_folder="experiment_images_pso_population_100",

        tf_num=[-24.44],
        tf_den=[84487.79, 1],
        tf_delay=0.5,
        tf_n_pade=2
    )