import math
import csv
import os

# ─────────────────────────────────────────────
# Tuning parameter
# ─────────────────────────────────────────────
DAMPING_RATIO_TARGET = 1      

# ─────────────────────────────────────────────
# Open-loop plant data: (label, Kp, tau, theta)
# ─────────────────────────────────────────────
plant_data = [
    # label              Kp        tau         theta
    ("DO-day-1",       1.346,   1551.955,    104.469),
    ("DO-day-2",       1.133,   2833.820,      0.0  ),
    ("DO-day-3",       2.287,   3010.296,      0.0  ),
    ("DO-day-4",       2.430,   3492.589,      0.0  ),
    # nighttime DO
    ("DO-night-1",     2.360,   3083.590,      0.0  ),
    ("DO-night-2",     2.050,   4500.000,      0.0  ),
    ("DO-night-3",     3.920,   3012.230,      0.0  ),
    ("DO-night-4",     3.130,   2530.050,      0.0  ),
    # TDS
    ("TDS-1",        -21.080,  71160.910,      0.0  ),
    ("TDS-2",        -15.520,  40156.080,      0.0  ),
    ("TDS-3",        -12.460,  16825.290,      0.0  ),
]

# ─────────────────────────────────────────────
# Compute Kc, Ki, and Damping Ratio (Zeta)
# ─────────────────────────────────────────────
results = []

for label, Kp, tau, theta in plant_data:
    if Kp == 0:
        print(f"Warning: Skipping '{label}' — Kp is 0.")
        continue

    # Lambda parameter 
    lb = tau * 3
    
    # Proportional gain
    Kc = tau / (Kp * (lb + theta))

    # Integral gain
    Ki = Kc / tau
    # Ki = ((1 + Kp * Kc)**2) / (4 * Kp * tau)

    # Calculated Damping Ratio (Zeta)

    zeta = (1 + Kp * Kc) / (2 * math.sqrt(Kp * Ki * tau))

    results.append((label, Kp, tau, theta, Kc, Ki, zeta))

# ─────────────────────────────────────────────
# Display Results
# ─────────────────────────────────────────────
col_w = {
    "label": 12,
    "Kp":    10,
    "tau":   12,
    "theta": 10,
    "Kc":    12,
    "Ki":    14,
    "zeta":  10,
}

header = (
    f"{'Label':<{col_w['label']}} "
    f"{'Kp':>{col_w['Kp']}} "
    f"{'tau':>{col_w['tau']}} "
    f"{'theta':>{col_w['theta']}} "
    f"{'Kc':>{col_w['Kc']}} "
    f"{'Ki':>{col_w['Ki']}} "
    f"{'Zeta':>{col_w['zeta']}}"
)

separator = "-" * len(header)

print()
print(f"  Lambda Tuning Results with Damping Ratio Verification")
print(separator)
print(header)
print(separator)

for label, Kp, tau, theta, Kc, Ki, zeta in results:
    print(
        f"{label:<{col_w['label']}} "
        f"{Kp:>{col_w['Kp']}.4f} "
        f"{tau:>{col_w['tau']}.3f} "
        f"{theta:>{col_w['theta']}.3f} "
        f"{Kc:>{col_w['Kc']}.6f} "
        f"{Ki:>{col_w['Ki']}.6f} "
        f"{zeta:>{col_w['zeta']}.4f}"
    )

print(separator)
print(f"  Zeta formula: (1 + Kc*Kp) / sqrt(2 * tau * Kc * Ki)")
print()

# ─────────────────────────────────────────────
# Export to CSV
# ─────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_filename = os.path.join(script_dir, "lambda_tuning_results.csv")

with open(csv_filename, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    # Write the header row
    writer.writerow(["Label", "Kp", "tau", "theta", "Kc", "Ki", "Zeta"])
    
    # Write the data rows
    for row in results:
        writer.writerow(row)

print(f"Data successfully exported to '{csv_filename}'.")