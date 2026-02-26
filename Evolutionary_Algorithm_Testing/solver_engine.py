import os
import numpy as np
from juliacall import Main as jl
from diffeqpy import de

os.environ["JULIA_IO_COLORED"] = "1"
print("Initializing Native Julia Solver Engine...")

# 1. DEFINE THE MATH IN NATIVE JULIA (Runs 100x - 1000x faster)
# This eliminates the Python-to-Julia communication overhead during integration.
jl.seval("""
function dde_system(du, u, h, p, t)
    # Extract parameters
    Kp_c, Ki_c, K_p, T_p, tau = p

    # Current State (Julia uses 1-based indexing!)
    y = u[1] 

    # Delayed States
    past = h(p, t - tau)
    past_y = past[1]
    past_int_e = past[2]

    # Control signal
    u_delayed = Kp_c * (1.0 - past_y) + Ki_c * past_int_e

    # Differential Equations
    du[1] = (K_p * u_delayed - y) / T_p  # Plant dynamics
    du[2] = 1.0 - y                      # Integral of error
    du[3] = t * abs(1.0 - y)             # ITAE tracking
end

function dde_history(p, t)
    # Return initial states for t < 0: [y, int_e, itae]
    return [0.0, 0.0, 0.0]
end
""")


def fast_itae_diffeq(Kp_ctrl, Ki_ctrl, K_plant, T_plant, delay):
    # Setup problem in Python, but reference the pure-Julia functions
    u0 = [0.0, 0.0, 0.0]
    p = (Kp_ctrl, Ki_ctrl, K_plant, T_plant, delay)  # Tuple for speed
    tspan = (0.0, 10000.0)

    # Notice we pass jl.dde_system and jl.dde_history directly
    prob = de.DDEProblem(jl.dde_system, u0, jl.dde_history, tspan, p, constant_lags=[delay])

    try:
        sol = de.solve(
            prob,
            de.MethodOfSteps(de.Tsit5()),
            saveat=10.0,
            maxiters=5000,
            dtmin=1e-3,
            force_dtmin=True,
            abstol=1e-3,
            reltol=1e-3
        )

        if sol.retcode != de.ReturnCode.Success:
            return 9.0

            # Extract results
        y_vals = np.array([state[0] for state in sol.u])
        final_itae = sol.u[-1][2]

        # Penalties
        if np.max(y_vals) > 1.2 or np.min(y_vals) < -0.2:
            final_itae += 1e9

        return np.log10(max(final_itae, 1e-12))

    except Exception:
        return 9.0


# --- WARM-UP ---
print("Compiling native Julia components...")
_ = fast_itae_diffeq(0.1, 0.01, 1.0, 10.0, 1.0)
print("Engine Ready! You are now running at full compiled speed.")