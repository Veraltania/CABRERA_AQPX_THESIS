import sys
from Evolutionary_Algorithm_Testing.de.de_optimizer import DEOptimizer


def main():
    # ==========================================
    # 1. HARDCODED INPUTS
    # ==========================================

    # PI Controller Gains
    kp = 9.48910
    ki = 0.00592

    # Transfer Function (Plant) Parameters
    # G(s) = (K / (Tp*s + 1)) * exp(-delay*s)
    tf_params = {
        'tf_num': [86.7056], 'tf_den': [3287.0801, 1], 'tf_delay': 0.0,
        'tf_n_pade': 2, 'computed_delay': 0.05, 'is_reverse_acting': False, 'max_kp': 100.0
    }

    # Optimizer Configuration
    # These settings are required to initialize the DEOptimizer instance
    config = {
        'patience': 20,
        'tol': 1e-4,
        'mutation': (0.5, 1.0),
        'recombination': 0.745,
        'strategy': 'best1bin'
    }

    # ==========================================
    # 2. EXECUTION
    # ==========================================

    print("Initializing EA Solver Engine...")

    try:
        # Create an instance of the optimizer.
        # This initializes the internal Python solver/simulation environment.
        optimizer = DEOptimizer(config, tf_params)

        # Calculate the cost using the ITAE (Integral Time Absolute Error) logic
        # implemented in the ea_optimizer framework.
        cost = optimizer.calculate_itae_cost(kp, ki)

        # ==========================================
        # 3. OUTPUT
        # ==========================================
        print("\n" + "=" * 40)
        print("         EA SOLVER COST REPORT")
        print("=" * 40)
        print(f"CONTROLLER SETTINGS:")
        print(f"  Kp: {kp}")
        print(f"  Ki: {ki}")
        print(f"CALCULATED COST: {cost}")
        print("=" * 40)

    except ImportError:
        print("\n[!] ERROR: Could not find 'Evolutionary_Algorithm_Testing.ea_optimizer'.")
        print("Ensure the folder structure is correct relative to this script.")
    except Exception as e:
        print(f"\n[!] ERROR: An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()