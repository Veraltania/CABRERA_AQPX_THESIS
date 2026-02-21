import control as ct
import numpy as np
import matplotlib.pyplot as plt


def define_transfer_func(num, den, delay, n_pade):
    G_linear = ct.TransferFunction(num, den)

    num_delay, den_delay = ct.pade(delay, n_pade)
    G_delay = ct.TransferFunction(num_delay, den_delay)

    # Combined Plant G(s)
    plant = ct.series(G_linear, G_delay)
    return plant


def find_all_crossings(sys):
    """
    Finds ALL exact crossings of the imaginary axis for a rational transfer function.
    Returns a list of tuples: (frequency_rad_s, crossing_gain_k)
    """
    num = sys.num[0][0]
    den = sys.den[0][0]

    def poly_minus_s(coeffs):
        n = len(coeffs) - 1
        powers = np.arange(n, -1, -1)
        signs = np.where(powers % 2 == 1, -1, 1)
        return coeffs * signs

    num_neg = poly_minus_s(num)
    den_neg = poly_minus_s(den)

    term1 = np.convolve(num, den_neg)
    term2 = np.convolve(num_neg, den)
    poly_imag_zero = term1 - term2

    roots_s = np.roots(poly_imag_zero)
    crossings = []

    # 1. Check for high-frequency crossings (w > 0)
    for r in roots_s:
        if abs(r.real) > 1e-5:
            continue

        w_val = abs(r.imag)
        if w_val < 1e-6:
            continue

        g_val = ct.evalfr(sys, 1j * w_val)
        if g_val.real < 0:
            k_cross = -1 / g_val.real
            crossings.append((w_val, k_cross))

    # 2. NEW: Explicitly check for DC crossing (w = 0, s = 0)
    dc_gain = ct.evalfr(sys, 0).real
    if dc_gain != 0:
        k_cross_dc = -1.0 / dc_gain
        # We append w=0 and the associated crossing gain
        crossings.append((0.0, k_cross_dc))

    crossings = sorted(list(set(crossings)))
    return crossings


def define_guardrail_gain(plant):
    """
    Finds the lowest positive gain that causes instability by verifying closed-loop poles.
    """
    all_crossings = find_all_crossings(plant)
    if not all_crossings:
        return None

    valid_boundaries = []

    for w, k in all_crossings:
        # We generally care about positive controller gains for the boundary
        if k <= 0:
            continue

        # Create the closed-loop system AT this crossing gain
        cl_sys = ct.feedback(k * plant, 1)
        poles = cl_sys.poles()

        # Check if any pole is strictly in the Right-Half Plane (ignoring the crossing pole)
        # We use 1e-4 as a tolerance for floating point math on the imaginary axis
        is_valid_boundary = True
        for p in poles:
            if p.real > 1e-4:
                is_valid_boundary = False
                break

        if is_valid_boundary:
            valid_boundaries.append((w, k))

    if not valid_boundaries:
        return None

    # Return the valid crossing with the lowest gain
    return min(valid_boundaries, key=lambda x: x[1])[1]

def display_root_locus(plant, crossings=None):
    """
    Displays the root locus plot for the given plant.
    Optionally marks the crossing points found.
    """
    plt.figure(figsize=(10, 8))

    # Calculate and plot the root locus
    # Using grid=True to help visualize stable/unstable regions
    ct.root_locus(plant, plot=True, grid=True)

    # Optional: Highlight the imaginary axis crossings
    if crossings:
        print(f"\nHighlighting {len(crossings)} crossing(s) on the plot:")

        # Separate lists for labeling purposes to avoid duplicate legend entries
        w_vals = []

        for w, k in crossings:
            w_vals.append(w)
            print(f"Crossing coordinates: (0, {w:.4f}) and (0, {-w:.4f}) at Gain K = {k:.4f}")
            # Plot the positive imaginary crossing
            plt.plot([0], [w], 'rX', markersize=10, zorder=10)
            # Plot the negative imaginary crossing (conjugate)
            plt.plot([0], [-w], 'rX', markersize=10, zorder=10)

        # Add a dummy handle for the legend
        plt.plot([], [], 'rX', markersize=10, label='Stability Crossing')

    plt.title(
        f"Root Locus (Padé Order n={plant.ninputs})")  # Accessing hidden implementation detail for title if needed, or just generic
    plt.title("Root Locus of System with Padé Approximation")
    plt.xlabel("Real Axis ($s^{-1}$)")
    plt.ylabel("Imaginary Axis ($s^{-1}$)")

    # Manually set the limits here
    plt.xlim([-25, 25])  # Shows Real axis from -10 to 2
    plt.ylim([-100, 100])  # Shows Imaginary axis from -5 to 5

    plt.legend()
    plt.tight_layout()
    plt.show()


# --- Usage ---
if __name__ == '__main__':
    # System Parameters
    num = [-13.87]
    den = [30961.51, 1]
    delay = 0.00
    n_pade = 2

    # 1. Define System
    G = define_transfer_func(num, den, delay, n_pade)
    print(G)

    # 2. Find Crossings
    all_crossings = find_all_crossings(G)

    if not all_crossings:
        print("No crossings found (System might be unconditionally stable or unstable).")
    else:
        max_gain = define_guardrail_gain(G)
        print(f"Calculated Max Stable Gain (K_u): {max_gain:.4f}")

        # Print details of all crossings found
        print("\nAll Imaginary Axis Crossings found:")
        for w, k in all_crossings:
            print(f"  Frequency (w_u): {w:.4f} rad/s, Gain (K_u): {k:.4f}")

    # 3. Display Root Locus
    display_root_locus(G, all_crossings)