import os
import csv
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import control as ct
import matplotlib.pyplot as plt

# ==========================================
# --- INTEGRATED JULIA SOLVER ENGINE ---
# ==========================================
os.environ["JULIA_IO_COLORED"] = "1"

print("Booting Julia and JIT compiling DelayDiffEq...")
print("(Go grab a coffee. This will take ~5 minutes on the Raspberry Pi...)")

from juliacall import Main as jl

# 1. DEFINE THE MATH AND SOLVER NATIVELY IN JULIA
jl.seval("""
using DelayDiffEq

function dde_system(du, u, h, p, t)
    Kp_c, Ki_c, K_p, T_p, tau, w1, w2, w4 = p
    y = u[1]

    past = h(p, t - tau)
    past_y = past[1]
    past_int_e = past[2]

    past_setpoint = (t - tau) >= 0.0 ? 1.0 : 0.0
    u_delayed = Kp_c * (past_setpoint - past_y) + Ki_c * past_int_e

    # Plant dynamics
    du[1] = (K_p * u_delayed - y) / T_p

    # Integral of error for the PI controller
    e = 1.0 - y
    du[2] = e

    # --- SEPARATED COST FUNCTION INTEGRANDS ---

    # 3. Error penalty
    du[3] = w1 * abs(e)

    # 4. Control effort penalty
    du[4] = w2 * (u_delayed^2)

    # 5. w4 penalty (Piecewise condition)
    delta_y = y - 1.0
    du[5] = delta_y < 0.0 ? w4 * abs(delta_y) : 0.0
end

function dde_history(p, t)
    return [0.0, 0.0, 0.0, 0.0, 0.0]
end

function run_dde_solver(Kp_ctrl, Ki_ctrl, K_plant, T_plant, delay, w1, w2, w4)
    u0 = [0.0, 0.0, 0.0, 0.0, 0.0]
    p = (Kp_ctrl, Ki_ctrl, K_plant, T_plant, delay, w1, w2, w4)
    tspan = (0.0, 10000.0)

    prob = DDEProblem(dde_system, u0, dde_history, tspan, p, constant_lags=[delay])

    try
        sol = solve(
            prob,
            MethodOfSteps(Tsit5()),
            abstol=1e-3,
            reltol=1e-3,
            tstops=[delay]
        )

        if sol.retcode != ReturnCode.Success
            return (9e9, 9e9, 9e9, Float64[], Float64[])
        end

        y_vals = [u[1] for u in sol.u]

        # Extract the individual integral values from the final timestep
        int_error = sol.u[end][3]
        int_control = sol.u[end][4]
        int_w4 = sol.u[end][5]

        t_vals = sol.t

        return (int_error, int_control, int_w4, y_vals, t_vals)
    catch e
        return (9e9, 9e9, 9e9, Float64[], Float64[])
    end
end
""")

def fast_fbest_diffeq(Kp_ctrl, Ki_ctrl, K_plant, T_plant, delay,
                      w_error=1.0, w_control=0.05, w_rise=0.2, w_overshoot=1.0):
    # Unpack the 3 distinct integrals from Julia
    int_error, int_control, int_w4, y_vals_jl, t_vals_jl = jl.run_dde_solver(
        Kp_ctrl, Ki_ctrl, K_plant, T_plant, delay, w_error, w_control, w_overshoot
    )

    # Check for early failure flags
    if int_error == 9e9 and len(y_vals_jl) == 0:
        return 9.0

    y_vals = np.array(y_vals_jl)
    t_vals = np.array(t_vals_jl)

    # Calculate Rise Time (t_u)
    crossings = t_vals[y_vals >= 1.0]
    if len(crossings) > 0:
        t_u = crossings[0]
    else:
        t_u = t_vals[-1]

    rise_time_penalty = w_rise * t_u

    f_best = int_error + int_control + int_w4 + rise_time_penalty

    # Apply hard constraints
    if np.max(y_vals) > 1.2 or np.min(y_vals) < -0.2:
        f_best += 1e9

    return np.log10(max(f_best, 1e-12))


# --- WARM-UP ---
_ = fast_fbest_diffeq(0.1, 0.01, 1.0, 10.0, 1.0)
print("Engine Ready! JIT compilation complete.")


# ==========================================
# --- EVOLUTIONARY OPTIMIZER BASE CLASS ---
# ==========================================

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
            cost = fast_fbest_diffeq(Kp, Ki, self.K_plant, self.T_plant, self.delay)
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

            # Only generate plots for the 25th and 50th rounds
            if current_round in [25, 50]:
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