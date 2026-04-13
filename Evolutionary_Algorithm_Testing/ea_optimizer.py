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

from juliacall import Main as jl

jl.seval("""
using DelayDiffEq

function dde_system(du, u, h, p, t)
    Kp_c, Ki_c, K_p, T_p, tau = p
    y = u[1]

    past = h(p, t - tau)
    past_y = past[1]
    past_int_e = past[2]

    if (t-tau) >= 0.0
         past_setpoint = 1.0
    else
         past_setpoint = 0.0
    end

    u_raw = Kp_c * (past_setpoint - past_y) + Ki_c * past_int_e
    
    # FIXED: Actuator must allow negative control effort (-1.0 to 1.0) 
    # to drive a reverse-acting (negative gain) plant to a positive setpoint.
    u_delayed = clamp(u_raw, -1.0, 1.0)
    
    du[1] = (K_p * u_delayed - y) / T_p
    e = 1.0 - y
    du[2] = e
    du[3] = abs(e)
    du[4] = u_delayed^2
end

function dde_history(p, t)
    return [0.0, 0.0, 0.0, 0.0]
end

function run_dde_solver(Kp_ctrl, Ki_ctrl, K_plant, T_plant, delay)
    u0 = [0.0, 0.0, 0.0, 0.0]
    p = (Kp_ctrl, Ki_ctrl, K_plant, T_plant, delay)
    simulation_time = T_plant * 3 + delay
    tspan = (0.0, simulation_time)

    prob = DDEProblem(dde_system, u0, dde_history, tspan, p, constant_lags=[delay])

    try
        sol = solve(prob, MethodOfSteps(Tsit5()), abstol=1e-3, reltol=1e-3, tstops=[delay])
        if sol.retcode != ReturnCode.Success
            return (9e9, 9e9, Float64[], Float64[], simulation_time)
        end

        y_vals = [u[1] for u in sol.u]
        int_error = sol.u[end][3]
        int_control = sol.u[end][4]
        t_vals = sol.t

        return (int_error, int_control, y_vals, t_vals, simulation_time)
    catch e
        return (9e9, 9e9, Float64[], Float64[], 0.0)
    end
end
""")

def fast_fbest_diffeq(Kp_ctrl, Ki_ctrl, K_plant, T_plant, delay, avg_rise_time):
    int_error, int_control, y_vals_jl, t_vals_jl, T_sim = jl.run_dde_solver(
        Kp_ctrl, Ki_ctrl, K_plant, T_plant, delay
    )

    penalty = (1e9, 1e9, 1e9, 1e9)

    if int_error == 9e9 and len(y_vals_jl) == 0:
        return penalty

    y_vals = np.array(y_vals_jl)
    t_vals = np.array(t_vals_jl)

    crossings_10 = np.where(y_vals >= 0.1)[0]
    crossings_90 = np.where(y_vals >= 0.9)[0]

    if len(crossings_10) > 0 and len(crossings_90) > 0:
        rise_time = t_vals[crossings_90[0]] - t_vals[crossings_10[0]]
    else:
        rise_time = T_sim * 10 

    norm_error = int_error / T_sim
    norm_effort = int_control / T_sim
    peak_y = np.max(y_vals)
    norm_overshoot = max(0.0, peak_y - 1.0) / 0.2
    norm_rise_time = rise_time / avg_rise_time

    if norm_error > 1.0 or norm_effort > 1.0 or norm_overshoot > 1.0 or norm_rise_time > 1.0:
        return penalty
        
    if np.max(y_vals) > 1.3 or np.min(y_vals) < -0.1:
        return penalty

    return (norm_error, norm_effort, norm_overshoot, norm_rise_time)

# --- WARM-UP ---
_ = fast_fbest_diffeq(0.1, 0.01, 1.0, 10.0, 1.0, 5.0)

# ==========================================
# --- SINGLE OBJECTIVE OPTIMIZER BASE ---
# ==========================================

class EvolutionaryOptimizer(ABC):
    def __init__(self, config, tf_params):
        self.config = config
        self.algo_name = self.__class__.__name__.replace('Optimizer', '')
        self.pop_size = config.get('population_size', 100)
        self.patience = config.get('patience_limit', 25)
        self.max_iters = config.get('max_iters', 200)
        self.tol = config.get('improvement_tol', 1e-4)
        self.n_rounds = config.get('n_rounds', 5)
        self.weights = config.get('weights', [1.0, 1.0, 1.0, 1.0])
        print(self.weights)

        folder_name = config.get('output_folder', f"experiment_images_{self.algo_name.lower()}")
        self.output_dir = Path(folder_name)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.K_plant = tf_params['tf_num'][0]
        self.T_plant = tf_params['tf_den'][0]
        self.delay = tf_params.get('computed_delay', 0.5)
        self.avg_rise_time = tf_params.get('avg_rise_time', self.T_plant * 2.2)

        self.is_reverse_acting = tf_params.get('is_reverse_acting', self.K_plant < 0)
        
        # --- Search Space Bounds ---
        # FIXED: Prioritize tf_params explicitly, then config, then logical defaults
        if self.is_reverse_acting:
            self.min_kp = float(tf_params.get('min_kp', config.get('min_kp', -2.0)))
            self.max_kp = float(tf_params.get('max_kp', config.get('max_kp', -0.001)))
            self.min_ki = float(tf_params.get('min_ki', config.get('min_ki', -0.05)))
            self.max_ki = float(tf_params.get('max_ki', config.get('max_ki', -1e-6)))
        else:
            self.min_kp = float(tf_params.get('min_kp', config.get('min_kp', 0.001)))
            self.max_kp = float(tf_params.get('max_kp', config.get('max_kp', 2.0)))
            self.min_ki = float(tf_params.get('min_ki', config.get('min_ki', 1e-6)))
            self.max_ki = float(tf_params.get('max_ki', config.get('max_ki', 0.05)))

        self._raw_tf_params = tf_params
        self._lazy_plant = None
        self.memo_cache = {}

    def calculate_cost(self, Kp, Ki):
        # FIXED: Ensure boundaries strictly enforce *both* parameters mathematically
        if self.is_reverse_acting and (Kp > 0 or Ki > 0):
            return (1e9, 1e9, 1e9, 1e9)
        if not self.is_reverse_acting and (Kp < 0 or Ki < 0):
            return (1e9, 1e9, 1e9, 1e9)

        cache_key = (round(float(Kp), 5), round(float(Ki), 5))
        if cache_key in self.memo_cache:
            return self.memo_cache[cache_key]

        try:
            cost_tuple = fast_fbest_diffeq(Kp, Ki, self.K_plant, self.T_plant, self.delay, self.avg_rise_time)
            self.memo_cache[cache_key] = cost_tuple
            return cost_tuple
        except:
            return (1e9, 1e9, 1e9, 1e9)

    @property
    def plant(self):
        if self._lazy_plant is None:
            num, den = ct.pade(self.delay, self._raw_tf_params.get('tf_n_pade', 2))
            pade_delay = ct.TransferFunction(num, den)
            base_tf = ct.TransferFunction(self._raw_tf_params['tf_num'], self._raw_tf_params['tf_den'])
            self._lazy_plant = base_tf * pade_delay
        return self._lazy_plant

    def simulate_response(self, Kp, Ki):
        ctrl = ct.TransferFunction([Kp, Ki], [1, 0])
        try:
            sys = ct.feedback(self.plant * ctrl, 1)
            simulation_time = self.delay + (self.T_plant * 5)
            T_sim = np.linspace(0, simulation_time, 1000)
            T, y = ct.step_response(sys, T_sim)
            return T, y
        except:
            return None, None

    def save_plot(self, round_num, best_Kp, best_Ki, best_cost):
        T_best, y_best = self.simulate_response(best_Kp, best_Ki)
        if T_best is None: return

        plt.figure(figsize=(10, 6))
        plt.plot(T_best, y_best, linewidth=2, color='#1f77b4', label=f'Best (Kp={best_Kp:.3f}, Ki={best_Ki:.3f})')
        plt.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Target (1.0)')
        plt.title(f'Final Best Step Response (Cost: {best_cost:.4f})', fontsize=14, fontweight='bold')
        plt.ylabel('Process Output (y)')
        plt.xlabel('Time (s)')
        plt.grid(True, linestyle=':', linewidth=0.7)
        plt.legend(loc='lower right')
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'final_response_round_{round_num:03d}.png', dpi=300)
        plt.close()

    def run_experiment(self):
        csv_file = self.output_dir / f"{self.output_dir.name}_detailed_log.csv"

        if not csv_file.exists():
            with open(csv_file, mode='w', newline='') as file:
                csv.writer(file).writerow([
                    'Round', 'Iterations', 'Best_Cost', 'Kp', 'Ki', 
                    'Error_Cost', 'Effort_Cost', 'Overshoot_Cost', 'RiseTime_Cost'
                ])

        costs_log = []
        for current_round in range(1, self.n_rounds + 1):
            best_sol, iterations, cost_history = self.optimize_round(current_round)
            Kp, Ki, total_cost, raw_costs = best_sol
            costs_log.append(total_cost)

            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    current_round, iterations, total_cost, Kp, Ki, 
                    raw_costs[0], raw_costs[1], raw_costs[2], raw_costs[3]
                ])
                
            if current_round == self.n_rounds:
                self.save_plot(current_round, Kp, Ki, total_cost)

            history_file = self.output_dir / f"raw_cost_history_round_{current_round:03d}.csv"
            with open(history_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Iteration', 'Cost'])
                for idx, c in enumerate(cost_history):
                    writer.writerow([idx + 1, c])

        print(f"--- {self.algo_name} FINISHED --- Average Best Cost: {np.mean(costs_log):.4f}")

    @abstractmethod
    def optimize_round(self, round_num):
        pass