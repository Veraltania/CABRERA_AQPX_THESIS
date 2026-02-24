import csv
import numpy as np
import matplotlib.pyplot as plt
import control as ct
from pathlib import Path
from abc import ABC, abstractmethod
from numba import njit  # <-- NUMBA IMPORTED HERE

# Assuming your custom local modules are accessible
from Transfer_Function_Analysis.analyze_transfer_func_stability import *


# --- HIGH-SPEED NUMBA FITNESS FUNCTION ---
@njit
def fast_itae_numba(Kp, Ki, K_plant, T_plant, delay):
    """
    Blazing fast discrete simulation for any First-Order Plus Dead Time (FOPDT) system.
    G(s) = K_plant / (T_plant*s + 1) * e^(-delay*s)
    Now utilizing highly stable RK4 integration.
    """
    t_end = 10000.0
    steps = 1000
    dt = t_end / steps

    delay_steps = int(delay / dt)
    u_history = np.zeros(delay_steps + 1)
    buffer_idx = 0

    y = 0.0
    integral_e = 0.0
    itae = 0.0
    setpoint = 1.0

    max_y = -1e9
    min_y = 1e9

    for i in range(steps):
        t = i * dt
        error = setpoint - y

        # ITAE calculation
        itae += t * abs(error) * dt

        # PI Controller
        integral_e += error * dt
        u = (Kp * error) + (Ki * integral_e)

        # Delay Buffer Management
        u_history[buffer_idx] = u
        delayed_idx = (buffer_idx + 1) % len(u_history)
        u_delayed = u_history[delayed_idx]

        # ---------------------------------------------------------
        # Plant Dynamics (Runge-Kutta 4th Order Integration)
        # ---------------------------------------------------------
        k1 = (K_plant * u_delayed - y) / T_plant
        k2 = (K_plant * u_delayed - (y + 0.5 * dt * k1)) / T_plant
        k3 = (K_plant * u_delayed - (y + 0.5 * dt * k2)) / T_plant
        k4 = (K_plant * u_delayed - (y + dt * k3)) / T_plant

        y += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        # ---------------------------------------------------------

        # Early exit if y explodes to save computation time
        if abs(y) > 1e6:
            return 9.0

            # Track limits for constraints
        if y > max_y: max_y = y
        if y < min_y: min_y = y

        buffer_idx = delayed_idx

    # Apply standard constraints
    if max_y > 1.2 or min_y < -0.2:
        itae += 1e9

    return np.log10(max(itae, 1e-12))


# -----------------------------------------


class EvolutionaryOptimizer(ABC):
    def __init__(self, config, tf_params):
        """Initializes the generic optimizer and sets up the environment."""

        # 1. Auto-Detect Algorithm Name (e.g., "DEOptimizer" -> "DE")
        self.algo_name = self.__class__.__name__.replace('Optimizer', '')

        # 2. Extract Base Configurations
        self.pop_size = config.get('population_size', 100)
        self.patience = config.get('patience_limit', 25)
        self.max_iters = config.get('max_iters', 200)
        self.tol = config.get('improvement_tol', 1.0)
        self.n_rounds = config.get('n_rounds', 50)

        # 3. Auto-Generate Output Directory Name
        folder_name = config.get(
            'output_folder',
            f"experiment_images_{self.algo_name.lower()}_population_{self.pop_size}"
        )
        self.output_dir = self.setup_experiment_dir(folder_name)

        # --- EXTRACT RAW PARAMS FOR NUMBA ---
        self.K_plant = tf_params['tf_num'][0]
        self.T_plant = tf_params['tf_den'][0]

        # --- DELAY OVERRIDE LOGIC ---
        extracted_delay = tf_params.get('tf_delay', 0.0)
        if extracted_delay == 0.0:
            print("   [!] System delay is 0.0. Overriding to 0.5s to allow mathematical stability bounding.")
            self.delay = 1
        else:
            self.delay = extracted_delay

        # Determine if we need a negative gain guardrail
        self.is_reverse_acting = self.K_plant < 0

        # 4. Define Plant and Constraints (Using self.delay instead of the raw dictionary value)
        self.plant = define_transfer_func(
            tf_params['tf_num'],
            tf_params['tf_den'],
            self.delay,
            tf_params.get('tf_n_pade', 2)
        )

        # Pass the flag to the guardrail function
        self.max_kp = define_guardrail_gain(self.plant, find_negative_gain=self.is_reverse_acting)

        # 5. History Tracking for Summaries
        self.agg_history = {'iterations': [], 'costs': [], 'kp': [], 'ki': []}

    # -- Shared Helper Methods --

    def setup_experiment_dir(self, folder_name):
        output_dir = Path(folder_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def calculate_itae_cost(self, Kp, Ki):
        try:
            return fast_itae_numba(
                Kp,
                Ki,
                self.K_plant,
                self.T_plant,
                self.delay
            )
        except Exception as e:
            print(f"Cost calculation failed: {e}")
            return np.log10(1e9)

    # --- Refactored ea_optimizer.py snippets ---

    def simulate_response(self, Kp, Ki, amplitude=1.0):
        """Simulates response with a custom step amplitude."""
        ctrl = ct.TransferFunction([Kp, Ki], [1, 0])
        try:
            # The closed-loop system
            sys = ct.feedback(self.plant * ctrl, 1)
            T_sim = np.linspace(0, 10000, 1000)

            # Multiply the unit step response by the desired amplitude
            T, y = ct.step_response(sys, T_sim)
            return T, y * amplitude
        except:
            return None, None

    def save_plots(self, round_num, history, best_Kp, best_Ki):
        # Determine step direction based on plant gain
        step_amplitude = -1.0 if self.is_reverse_acting else 1.0

        # Step Response Plot
        T_best, y_best = self.simulate_response(best_Kp, best_Ki, amplitude=step_amplitude)

        plt.figure(figsize=(10, 6))
        if T_best is not None:
            label_text = f'Best Params (Step: {step_amplitude})'
            plt.plot(T_best, y_best, linewidth=3, label=label_text)

        # Update the target line to match the negative step
        plt.axhline(step_amplitude, color='r', linestyle='--', label=f'Target ({step_amplitude})')

        plt.title(
            f'Negative Step Response - Round {round_num}' if self.is_reverse_acting else f'Step Response - Round {round_num}')
        plt.ylabel('Output Response')
        plt.xlabel('Time (s)')
        plt.grid(True)
        plt.legend()
        plt.savefig(self.output_dir / f'response_round_{round_num}.png')
        plt.close()

    # -- Main Experiment Loop --

    def run_experiment(self):
        csv_file = self.output_dir / f"{self.output_dir.name}_detailed_log.csv"

        # Initialize CSV Header if the file is new
        if not csv_file.exists():
            with open(csv_file, mode='w', newline='') as file:
                csv.writer(file).writerow(['Round', 'Iterations_Run', 'Final_Cost_ITAE', 'Best_Kp', 'Best_Ki'])

        for current_round in range(1, self.n_rounds + 1):
            print(f"\n{'=' * 40}\nSTARTING {self.algo_name} ROUND {current_round} OF {self.n_rounds}\n{'=' * 40}")

            # Subclass executes its specific algorithm here
            best_Kp, best_Ki, cost, iterations_run, cost_history = self.optimize_round(current_round)
            print(f"   [RESULT] Best Params found: Kp = {best_Kp}, Ki = {best_Ki}")
            # Store and save data
            self.agg_history['iterations'].append(iterations_run)
            self.agg_history['costs'].append(cost)
            self.agg_history['kp'].append(best_Kp)
            self.agg_history['ki'].append(best_Ki)

            with open(csv_file, mode='a', newline='') as file:
                csv.writer(file).writerow([current_round, iterations_run, cost, best_Kp, best_Ki])

            self.save_plots(current_round, cost_history, best_Kp, best_Ki)

        self._print_and_save_summary()

    def _print_and_save_summary(self):
        summary_text = (
            f"--- {self.algo_name} EXPERIMENT SUMMARY ---\n"
            f"Total Rounds: {self.n_rounds}\n"
            f"Average Iterations: {np.mean(self.agg_history['iterations']):.1f}\n"
            f"Average ITAE Cost:  {np.mean(self.agg_history['costs']):.2f}\n"
            f"-----------------------------\n"
            f"Kp: {np.mean(self.agg_history['kp']):.5f} (+/- {np.std(self.agg_history['kp']):.5f})\n"
            f"Ki: {np.mean(self.agg_history['ki']):.6f} (+/- {np.std(self.agg_history['ki']):.6f})\n"
        )
        print(summary_text)
        with open(self.output_dir / f"{self.output_dir.name}_summary.txt", "w") as f:
            f.write(summary_text)

    # -- Abstract Method to Enforce Subclass Implementation --

    @abstractmethod
    def optimize_round(self, round_num):
        pass