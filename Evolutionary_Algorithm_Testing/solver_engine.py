import os
import numpy as np

# Tell Julia to use colored output in the terminal
os.environ["JULIA_IO_COLORED"] = "1"

print("Booting Julia and JIT compiling DelayDiffEq...")
print("(Go grab a coffee. This will take ~5 minutes on the Raspberry Pi...)")

# This will automatically use your juliapkg environment
from juliacall import Main as jl

# 1. DEFINE THE MATH AND SOLVER NATIVELY IN JULIA
# This completely eliminates Python-to-Julia communication overhead during integration.
jl.seval("""
using DelayDiffEq

function dde_system(du, u, h, p, t)
    Kp_c, Ki_c, K_p, T_p, tau = p
    y = u[1] 
    
    past = h(p, t - tau)
    past_y = past[1]
    past_int_e = past[2]

    past_setpoint = (t - tau) >= 0.0 ? 1.0 : 0.0
    u_delayed = Kp_c * (past_setpoint - past_y) + Ki_c * past_int_e

    du[1] = (K_p * u_delayed - y) / T_p  
    du[2] = 1.0 - y                      
    du[3] = t * abs(1.0 - y)             
end

function dde_history(p, t)
    return [0.0, 0.0, 0.0]
end

function run_dde_solver(Kp_ctrl, Ki_ctrl, K_plant, T_plant, delay)
    u0 = [0.0, 0.0, 0.0]
    p = (Kp_ctrl, Ki_ctrl, K_plant, T_plant, delay)
    tspan = (0.0, 10000.0)
    
    prob = DDEProblem(dde_system, u0, dde_history, tspan, p, constant_lags=[delay])
    
    try
        sol = solve(
            prob,
            MethodOfSteps(Tsit5()),
            saveat=10.0,
            dtmin=1e-3,
            abstol=1e-3,
            reltol=1e-3
        )
        
        if sol.retcode != ReturnCode.Success
            return (9.0, Float64[])
        end
        
        # Extract y_vals and final_itae (Julia is 1-indexed)
        y_vals = [u[1] for u in sol.u]
        final_itae = sol.u[end][3] 
        
        return (final_itae, y_vals)
    catch e
        return (9.0, Float64[])
    end
end
""")

def fast_itae_diffeq(Kp_ctrl, Ki_ctrl, K_plant, T_plant, delay):
    # 2. CALL THE JULIA WRAPPER
    final_itae, y_vals_jl = jl.run_dde_solver(Kp_ctrl, Ki_ctrl, K_plant, T_plant, delay)
    
    # Check for early failure flags sent from Julia
    if final_itae == 9.0 and len(y_vals_jl) == 0:
        return 9.0

    # Convert the julia array to a standard NumPy array for the penalty check
    y_vals = np.array(y_vals_jl)

    # Apply constraints / Penalties
    if np.max(y_vals) > 1.2 or np.min(y_vals) < -0.2:
        final_itae += 1e9

    return np.log10(max(final_itae, 1e-12))


# --- WARM-UP ---
# The script will hang on this exact line while Julia compiles everything.
_ = fast_itae_diffeq(0.1, 0.01, 1.0, 10.0, 1.0)

print("Engine Ready! JIT compilation complete.")
print("The Evolutionary Algorithm can now run at compiled C/Fortran speeds.")