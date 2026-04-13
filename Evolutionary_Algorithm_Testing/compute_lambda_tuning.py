# List of 3D open-loop plant data: (Kp, tau, theta)
plant_data = [
    # daytime DO
    (1.346, 1551.955, 104.469),
    (1.133, 2833.82, 0.0),
    (2.287, 3010.296, 0.0),
    (2.430, 3492.589, 0.0),
    # nighttime DO
    (2.36, 3083.59, 0.0),
    (2.05, 4500.00, 0.0),
    (3.92, 3012.23, 0.0),
    (3.13, 2530.05, 0.0),
    # TDS
    (-21.08, 71160.91, 0.0),
    (-15.52, 40156.08, 0.0),
    (-12.46, 16825.29, 0.0)
]

# The output list that will store the computed controller gains (K)
K_list = []

# Iterate directly over the list of 3D data
for Kp, tau, theta in plant_data:
    if Kp == 0:
        print(f"Warning: Skipping data point {(Kp, tau, theta)} because Kp is 0.")
        continue
        
    # Tuning rules applied:
    # Lambda = tau
    # Ti = tau
    # Formula: Kc = tau / (Kp * (Lambda + theta))
    
    Kc = tau / (Kp * (tau + theta))
    K_list.append(Kc)

# Output the final list of K
print("Output list of Kp:")
print(K_list)

print("Output list of Ki")
for Kp, tau, theta in plant_data:
    print(1/tau)