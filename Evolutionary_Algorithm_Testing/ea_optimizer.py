import csv
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import control as ct

from Evolutionary_Algorithm_Testing.solver_engine import fast_itae_diffeq


class EvolutionaryOptimizer(ABC):
    def __init__(self, config, tf_params):
        """Initializes the generic optimizer and sets up the environment."""
        self.algo_name = self.__class__.__name__.replace('Optimizer', '')

        # Base Configurations
        self.pop_size = config.get('population_size', 100)
        self.patience = config.get('patience_limit', 25)
        self.max_iters = config.get('max_iters', 200)
        self.tol = config.get('improvement_tol', 1.0)
        self.n_rounds = config.get('n_rounds', 50)

        folder_name = config.get(
            'output_folder',
            f"experiment_images_{self.algo_name.lower()}_population_{self.pop_size}"
        )
        self.output_dir = self.setup_experiment_dir(folder_name)

        # Raw params for Numba solver
        self.K_plant = tf_params['tf_num'][0]
        self.T_plant = tf_params['tf_den'][0]
        self.delay = tf_params.get('computed_delay', 0.5)

        # Pre-computed bounds and flags (Passed in, NOT calculated here)
        self.is_reverse_acting = tf_params.get('is_reverse_acting', self.K_plant < 0)
        self.max_kp = tf_params.get('max_kp', None)

        # Store raw TF params in case we need to build the plant lazily later
        self._raw_tf_params = tf_params
        self._lazy_plant = None

        # History Tracking
        self.agg_history = {'iterations': [], 'costs': [], 'kp': [], 'ki': []}

        # --- MEMOIZATION CACHE ---
        # Persists across all rounds for this specific optimizer instance
        self.memo_cache = {}

    def setup_experiment_dir(self, folder_name):
        output_dir = Path(folder_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def calculate_itae_cost(self, Kp, Ki):
        # 1. Fast-Fail Boundary Check
        if (self.is_reverse_acting and Kp > 0) or (not self.is_reverse_acting and Kp < 0):
            return 1e9  # Wrong sign, instantly penalize

        # 2. Fuzzy Memoization (Round to 5 decimal places to group micro-mutations)
        cache_key = (round(float(Kp), 5), round(float(Ki), 5))
        if cache_key in self.memo_cache:
            return self.memo_cache[cache_key]

        # 3. Compute and Cache
        try:
            cost = fast_itae_diffeq(Kp, Ki, self.K_plant, self.T_plant, self.delay)
            self.memo_cache[cache_key] = cost
            return cost
        except Exception as e:
            print(f"Cost calculation failed: {e}")
            return np.log10(1e9)

    @property
    def plant(self):
        """Lazy evaluation: Only build the control system plant if requested (e.g. for plotting)"""
        if self._lazy_plant is None:
            # Recreate it locally using standard control tools if needed for simulate_response
            num, den = ct.pade(self.delay, self._raw_tf_params.get('tf_n_pade', 2))
            pade_delay = ct.TransferFunction(num, den)
            base_tf = ct.TransferFunction(self._raw_tf_params['tf_num'], self._raw_tf_params['tf_den'])
            self._lazy_plant = base_tf * pade_delay
        return self._lazy_plant

    def simulate_response(self, Kp, Ki, amplitude=1.0):
        ctrl = ct.TransferFunction([Kp, Ki], [1, 0])
        try:
            sys = ct.feedback(self.plant * ctrl, 1)
            T_sim = np.linspace(0, 10000, 1000)
            T, y = ct.step_response(sys, T_sim)
            return T, y * amplitude
        except:
            return None, None

    # [Remaining methods: save_plots, run_experiment, _print_and_save_summary remain exactly the same]
    # ...

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

            # self.save_plots(current_round, cost_history, best_Kp, best_Ki)

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