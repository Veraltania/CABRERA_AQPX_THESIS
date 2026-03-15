import os
import numpy as np

# Tell Julia to use colored output in the terminal
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
    # Expanded to 5 states
    return [0.0, 0.0, 0.0, 0.0, 0.0]
end

function run_dde_solver(Kp_ctrl, Ki_ctrl, K_plant, T_plant, delay, w1, w2, w4)
    # Expanded to 5 states
    u0 = [0.0, 0.0, 0.0, 0.0, 0.0]
    p = (Kp_ctrl, Ki_ctrl, K_plant, T_plant, delay, w1, w2, w4)
    
    # 1. Adapt tspan to system dynamics
    # Response starts after 'delay', then needs ~3x T_plant to settle
    t_end = delay + 3.0 * T_plant
    tspan = (0.0, t_end)

    prob = DDEProblem(dde_system, u0, dde_history, tspan, p, constant_lags=[delay])

    # 2. Define exactly 1000 evenly spaced points
    save_points = range(0.0, stop=t_end, length=1000)

    try
        sol = solve(
            prob,
            MethodOfSteps(Tsit5()),
            abstol=1e-3,
            reltol=1e-3,
            tstops=[delay],
            saveat=save_points # Forces the solver to save at these specific points
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

    # --- PRINT THE BREAKDOWN ---
    #
    # print("\n--- Cost Function Breakdown ---")
    # print(f"Error Penalty (w1):      {int_error:.4f}")
    # print(f"Control Penalty (w2):    {int_control:.4f}")
    # print(f"w4 Penalty (undershoot): {int_w4:.4f}")
    # print(f"Rise Time Penalty:       {rise_time_penalty:.4f}")
    #
    # # Reassemble final F_best
    f_best = int_error + int_control + int_w4 + rise_time_penalty
    # print(f"TOTAL F_best (pre-log):  {f_best:.4f}")
    # print("-------------------------------")
    # Apply hard constraints
    if np.max(y_vals) > 1.2 or np.min(y_vals) < -0.2:
        f_best += 1e9

    return np.log10(max(f_best, 1e-12))


# --- WARM-UP ---
_ = fast_fbest_diffeq(0.1, 0.01, 1.0, 10.0, 1.0)

print("Engine Ready! JIT compilation complete.")
print("The Evolutionary Algorithm is now running F_best at compiled speeds.")