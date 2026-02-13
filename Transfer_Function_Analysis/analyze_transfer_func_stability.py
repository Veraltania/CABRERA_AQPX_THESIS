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
    # 1. Extract Numerator and Denominator coefficients
    # sys.num and sys.den are lists of lists, we need 1D arrays
    num = sys.num[0][0]
    den = sys.den[0][0]

    # 2. Construct the "Imaginary Part = 0" polynomial
    # The condition Im(G(jw)) = 0 is equivalent to:
    # N(s) * D(-s) - N(-s) * D(s) = 0  (where s = jw)

    # Helper to generate polynomial for P(-s)
    # We flip the sign of coefficients for odd powers
    def poly_minus_s(coeffs):
        n = len(coeffs) - 1
        # Create array of signs: [..., -1, 1, -1, 1] depends on power
        powers = np.arange(n, -1, -1)
        signs = np.where(powers % 2 == 1, -1, 1)
        return coeffs * signs

    num_neg = poly_minus_s(num)
    den_neg = poly_minus_s(den)

    # Perform polynomial multiplication (convolution)
    term1 = np.convolve(num, den_neg)
    term2 = np.convolve(num_neg, den)

    # The equation to solve: term1 - term2 = 0
    poly_imag_zero = term1 - term2

    # 3. Find roots of this polynomial (these are values of 's')
    # Since we set s = jw, the valid roots must be purely imaginary (Real part ~ 0)
    roots_s = np.roots(poly_imag_zero)

    crossings = []

    for r in roots_s:
        # Filter 1: Must be purely imaginary root (s = jw implies Re(s)=0)
        # We allow a tiny tolerance for numerical noise
        if abs(r.real) > 1e-5:
            continue

        # Extract w (frequency) from the imaginary part
        w_val = abs(r.imag)

        # Filter 2: We only care about positive frequencies
        if w_val < 1e-6:
            continue

        # 4. Calculate the Gain K at this frequency
        # G(jw) is purely real here. K = -1 / G(jw)
        g_val = ct.evalfr(sys, 1j * w_val)

        # Filter 3: Check if it's a valid Root Locus crossing (Phase = 180)
        # This means Real part must be NEGATIVE.
        # (If Real part is positive, Phase is 0, which is the start of the locus, not a crossing)
        if g_val.real < 0:
            k_cross = -1 / g_val.real
            crossings.append((w_val, k_cross))

    # Sort by Frequency (or by Gain if you prefer)
    crossings = sorted(list(set(crossings)))  # set removes duplicates
    return crossings


def define_guardrail_gain(plant):
    all_crossings = find_all_crossings(plant)
    if not all_crossings:
        return None  # Or a safe default value

    # Find the tuple with the minimum k and return only k
    return min(all_crossings, key=lambda x: x[1])[1]


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
    plt.xlim([-0.015, 0.015])  # Shows Real axis from -10 to 2
    plt.ylim([-0.015, 0.015])  # Shows Imaginary axis from -5 to 5

    plt.legend()
    plt.tight_layout()
    plt.show()


# --- Usage ---
if __name__ == '__main__':
    # System Parameters
    num = [71.77]
    den = [3887.85, 1]
    delay = 320.00
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