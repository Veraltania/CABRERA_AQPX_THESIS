import csv
import numpy as np
import matplotlib.pyplot as plt
import control as ct
from pathlib import Path
from abc import ABC, abstractmethod

# Assuming your custom local modules are accessible
from Transfer_Function_Analysis.analyze_transfer_func_stability import *


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
        # If the user provides a custom folder name, use it. Otherwise, build a descriptive one.
        folder_name = config.get(
            'output_folder',
            f"experiment_images_{self.algo_name.lower()}_population_{self.pop_size}"
        )
        self.output_dir = self.setup_experiment_dir(folder_name)

        # 4. Define Plant and Constraints
        self.plant = define_transfer_func(
            tf_params['tf_num'],
            tf_params['tf_den'],
            tf_params['tf_delay'],
            tf_params['tf_n_pade']
        )
        self.max_kp = define_guardrail_gain(self.plant)

        # 5. History Tracking for Summaries
        self.agg_history = {'iterations': [], 'costs': [], 'kp': [], 'ki': []}

    # -- Shared Helper Methods --

    def setup_experiment_dir(self, folder_name):
        output_dir = Path(folder_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def simulate_response(self, Kp, Ki):
        ctrl = ct.TransferFunction([Kp, Ki], [1, 0])
        try:
            sys = ct.feedback(self.plant * ctrl, 1)
            T_sim = np.linspace(0, 10000, 1000)
            return ct.step_response(sys, T_sim)
        except:
            return None, None

    def calculate_itae_cost(self, Kp, Ki):
        if Kp < 0 or Ki < 0:
            return 1e9
        try:
            controller = ct.TransferFunction([Kp, Ki], [1, 0])
            closed_loop = ct.feedback(self.plant * controller, 1)
            T = np.linspace(0, 10000, 1000)
            T, y = ct.step_response(closed_loop, T)
            y = np.asarray(y).flatten()

            e = 1.0 - y
            dt = T[1] - T[0]
            itae = np.sum(T * np.abs(e)) * dt

            if np.max(y) > 1.2 or np.min(y) < -0.2:
                itae += 1e9

            return float(itae) if not (np.isnan(itae) or np.isinf(itae)) else 1e9
        except:
            return 1e9

    def save_plots(self, round_num, history, best_Kp, best_Ki):
        # Convergence Plot
        plt.figure(figsize=(10, 4))
        plt.plot(history, linewidth=2)
        plt.title(f'{self.algo_name} Convergence - Round {round_num}')
        plt.grid(True)
        plt.savefig(self.output_dir / f'convergence_round_{round_num}.png')
        plt.close()

        # Step Response Plot
        T_best, y_best = self.simulate_response(best_Kp, best_Ki)
        plt.figure(figsize=(10, 6))
        if T_best is not None:
            plt.plot(T_best, y_best, linewidth=3, label=f'Best {self.algo_name} Params')
        plt.axhline(1.0, color='r', linestyle='--')
        plt.title(f'Step Response - Round {round_num}')
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
        """
        Must be implemented by the child class (e.g., DEOptimizer, GAOptimizer).

        Returns:
            best_Kp (float): Best Proportional Gain found
            best_Ki (float): Best Integral Gain found
            final_cost (float): The final ITAE cost of the best parameters
            iterations_run (int): The number of iterations/generations until stopping
            cost_history (list): A list of the best costs per iteration/generation
        """
        pass