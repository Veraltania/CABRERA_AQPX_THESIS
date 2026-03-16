import os
import numpy as np
from juliacall import Main as jl

class JuliaSolverEngine:
    _instance = None

    def __new__(cls):
        # The Singleton logic: Only boot Julia if it hasn't been booted yet
        if cls._instance is None:
            cls._instance = super(JuliaSolverEngine, cls).__new__(cls)
            cls._instance._initialize_julia()
        return cls._instance

    def _initialize_julia(self):
        print("Booting Julia and JIT compiling DelayDiffEq...")

        os.environ["JULIA_IO_COLORED"] = "1"
        self.jl = jl

        # 1. DEFINE THE MATH AND SOLVER NATIVELY IN JULIA
        self.jl.seval("""
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

        # Run the warm-up once during initialization
        self.evaluate(0.1, 0.01, 1.0, 10.0, 1.0)
        print("Engine Ready! JIT compilation complete.")

    def evaluate(self, Kp_ctrl, Ki_ctrl, K_plant, T_plant, delay,
                 w_error=1.0, w_control=0.05, w_rise=0.2, w_overshoot=1.0):

        # Call the Julia solver
        int_error, int_control, int_w4, y_vals_jl, t_vals_jl = self.jl.run_dde_solver(
            Kp_ctrl, Ki_ctrl, K_plant, T_plant, delay, w_error, w_control, w_overshoot
        )

        if int_error == 9e9 and len(y_vals_jl) == 0:
            return 9.0

        y_vals = np.array(y_vals_jl)
        t_vals = np.array(t_vals_jl)

        crossings = t_vals[y_vals >= 1.0]
        t_u = crossings[0] if len(crossings) > 0 else t_vals[-1]

        rise_time_penalty = w_rise * t_u
        f_best = int_error + int_control + int_w4 + rise_time_penalty

        if np.max(y_vals) > 1.2 or np.min(y_vals) < -0.2:
            f_best += 1e9

        return np.log10(max(f_best, 1e-12))