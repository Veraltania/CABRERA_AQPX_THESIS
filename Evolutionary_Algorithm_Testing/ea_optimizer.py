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

# UPDATED: The DDE now handles [0, 1] clamping, calculates continuous anti-windup,
# and tracks pure `y` and `int_e` states so Python can perfectly reconstruct 
# the Total Variation (TV) control effort.
jl.seval("""
using DelayDiffEq

function dde_system(du, u, h, p, t)
    Kp_c, Ki_c, K_p, T_p, tau, target_sp = p
    y = u[1]
    int_e = u[2]

    # Calculate delayed control effort (what actually enters the plant)
    past = h(p, t - tau)
    past_y = past[1]
    past_int_e = past[2]
    
    past_sp = (t - tau) >= 0.0 ? target_sp : 0.0
    u_raw_delayed = Kp_c * (past_sp - past_y) + Ki_c * past_int_e
    
    # Actuator strictly clamped to [0.0, 1.0] 
    u_delayed = clamp(u_raw_delayed, 0.0, 1.0)
    
    du[1] = (K_p * u_delayed - y) / T_p
    
    # Calculate current state for integrator & anti-windup
    current_sp = t >= 0.0 ? target_sp : 0.0
    e_current = current_sp - y
    u_raw_current = Kp_c * e_current + Ki_c * int_e
    
    # Anti-windup: unconditionally stop integrating if saturated 
    # (Matches `if u_raw != u_clamped:` from step_response_comparison.py)
    if u_raw_current > 1.0 || u_raw_current < 0.0
        du[2] = 0.0 
    else
        du[2] = e_current
    end
end

function dde_history(p, t)
    return [0.0, 0.0]
end

function run_dde_solver(Kp_ctrl, Ki_ctrl, K_plant, T_plant, delay, target_sp)
    u0 = [0.0, 0.0]
    p = (Kp_ctrl, Ki_ctrl, K_plant, T_plant, delay, target_sp)
    simulation_time = T_plant * 3 + delay
    tspan = (0.0, simulation_time)

    prob = DDEProblem(dde_system, u0, dde_history, tspan, p, constant_lags=[delay])

    try
        # Force exactly 1000 uniform points to perfectly match np.linspace target
        saveat_interval = simulation_time / 999.0
        sol = solve(prob, MethodOfSteps(Tsit5()), abstol=1e-3, reltol=1e-3, saveat=saveat_interval)
        
        if sol.retcode != ReturnCode.Success
            return (Float64[], Float64[], Float64[], simulation_time)
        end

        t_vals = sol.t
        y_vals = [u[1] for u in sol.u]
        
        # Reconstruct exactly what the actuator produced at each time step
        u_vals = zeros(length(t_vals))
        for i in 1:length(t_vals)
            t = t_vals[i]
            current_y = y_vals[i]
            current_int_e = sol.u[i][2]
            current_sp = t >= 0.0 ? target_sp : 0.0
            
            u_raw = Kp_ctrl * (current_sp - current_y) + Ki_ctrl * current_int_e
            u_vals[i] = clamp(u_raw, 0.0, 1.0)
        end

        return (y_vals, u_vals, t_vals, simulation_time)
    catch e
        return (Float64[], Float64[], Float64[], 0.0)
    end
end
""")

def fast_fbest_diffeq(Kp_ctrl, Ki_ctrl, K_plant, T_plant, delay, avg_rise_time):
    target_sp = 1.0 if K_plant > 0 else -1.0
    
    y_vals_jl, u_vals_jl, t_vals_jl, T_sim = jl.run_dde_solver(
        Kp_ctrl, Ki_ctrl, K_plant, T_plant, delay, target_sp
    )

    penalty = (1e20, 1e20, 1e20, 1e20)

    if len(y_vals_jl) == 0:
        return penalty

    y_out = np.array(y_vals_jl)
    u_out = np.array(u_vals_jl)
    t_opt = np.array(t_vals_jl)

    # Normalize response relative to expected sign
    y_norm = y_out * np.sign(target_sp)

    if np.any(np.isnan(y_norm)) or np.any(np.isinf(y_norm)):
        return penalty

    # --- OVERSHOOT CALCULATION ---
    peak_val = np.max(y_norm)
    actual_overshoot = max(0.0, peak_val - 1.0)
    max_overshoot_limit = 0.20

    if actual_overshoot > max_overshoot_limit:
        return penalty

    # Stability Check
    if np.min(y_norm) < -0.1:
        return penalty

    error = 1.0 - y_norm
    int_error = np.trapezoid(np.abs(error), t_opt)

    # --- TOTAL VARIATION (CONTROL EFFORT) ---
    u_with_initial = np.concatenate(([0.0], u_out))
    norm_effort = np.sum(np.abs(np.diff(u_with_initial)))

    # Integral of Overshoot Area
    overshoot_array = np.where(error < 0, np.abs(error), 0.0)
    int_overshoot = np.trapezoid(overshoot_array, t_opt)

    crossings_10 = np.where(y_norm >= 0.1)[0]
    crossings_90 = np.where(y_norm >= 0.9)[0]

    if len(crossings_10) > 0 and len(crossings_90) > 0:
        rise_time = t_opt[crossings_90[0]] - t_opt[crossings_10[0]]
    else:
        rise_time = T_sim * 10

    norm_error = int_error / T_sim
    norm_overshoot = int_overshoot / T_sim
    norm_rise_time = rise_time / avg_rise_time

    if norm_error > 2.0 or norm_rise_time > 10.0:
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

        folder_name = config.get('output_folder', f"experiment_images_{self.algo_name.lower()}")
        self.output_dir = Path(folder_name)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.K_plant = tf_params['tf_num'][0]
        self.T_plant = tf_params['tf_den'][0]
        self.delay = tf_params.get('computed_delay', 0.5)
        self.avg_rise_time = tf_params.get('avg_rise_time', self.T_plant * 2.2)

        self.is_reverse_acting = tf_params.get('is_reverse_acting', self.K_plant < 0)
        
        # --- Search Space Bounds ---
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
        if self.is_reverse_acting and (Kp > 0 or Ki > 0):
            return (1e20, 1e20, 1e20, 1e20)
        if not self.is_reverse_acting and (Kp < 0 or Ki < 0):
            return (1e20, 1e20, 1e20, 1e20)

        cache_key = (round(float(Kp), 5), round(float(Ki), 5))
        if cache_key in self.memo_cache:
            return self.memo_cache[cache_key]

        try:
            cost_tuple = fast_fbest_diffeq(Kp, Ki, self.K_plant, self.T_plant, self.delay, self.avg_rise_time)
            self.memo_cache[cache_key] = cost_tuple
            return cost_tuple
        except:
            return (1e20, 1e20, 1e20, 1e20)

    @property
    def plant(self):
        # Kept for compatibility if sub-classes need access to the continuous theoretical plant
        if self._lazy_plant is None:
            num, den = ct.pade(self.delay, self._raw_tf_params.get('tf_n_pade', 2))
            pade_delay = ct.TransferFunction(num, den)
            base_tf = ct.TransferFunction(self._raw_tf_params['tf_num'], self._raw_tf_params['tf_den'])
            self._lazy_plant = base_tf * pade_delay
        return self._lazy_plant

    def simulate_response(self, Kp, Ki):
        # UPDATED: Avoid control.step_response(). Linear models don't enforce actuator
        # saturation bounds. We reuse the DDE to plot physically accurate curves.
        target_sp = 1.0 if self.K_plant > 0 else -1.0
        y_vals_jl, _, t_vals_jl, _ = jl.run_dde_solver(
            Kp, Ki, self.K_plant, self.T_plant, self.delay, target_sp
        )
        if len(y_vals_jl) == 0:
            return None, None
        return np.array(t_vals_jl), np.array(y_vals_jl)

    def save_plot(self, round_num, best_Kp, best_Ki, best_cost):
        T_best, y_best = self.simulate_response(best_Kp, best_Ki)
        if T_best is None: return

        target_sp = 1.0 if self.K_plant > 0 else -1.0

        plt.figure(figsize=(10, 6))
        plt.plot(T_best, y_best, linewidth=2, color='#1f77b4', label=f'Best (Kp={best_Kp:.3f}, Ki={best_Ki:.3f})')
        plt.axhline(target_sp, color='red', linestyle='--', linewidth=2, label=f'Target ({target_sp})')
        plt.title(f'Final Best Step Response (Cost: {best_cost:.4f})', fontsize=14, fontweight='bold')
        plt.ylabel('Process Output (y)')
        plt.xlabel('Time (s)')
        plt.grid(True, linestyle=':', linewidth=0.7)
        plt.legend(loc='lower right')
        
        # Set realistic y-limits depending on direction
        if self.is_reverse_acting:
            plt.ylim(top=0.1)
        else:
            plt.ylim(bottom=-0.1)
            
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