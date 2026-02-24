from scipy.optimize import differential_evolution
from Evolutionary_Algorithm_Testing.ea_optimizer import EvolutionaryOptimizer

class DEOptimizer(EvolutionaryOptimizer):
    def __init__(self, config, tf_params):
        # 1. Initialize base class settings
        super().__init__(config, tf_params)

        # 2. Extract DE-specific options with robust defaults
        self.mutation = config.get('mutation', (0.5, 1.0))
        self.recombination = config.get('recombination', 0.7)
        self.strategy = config.get('strategy', 'best1bin')

    def optimize_round(self, round_num):
        class Tracker:
            def __init__(self, patience, tol):
                self.patience = patience
                self.tol = tol
                self.counter = 0
                self.best_cost = float('inf')
                self.history = []

            def callback(self, xk, convergence=None):
                # We need to compute the cost inside callback to track it accurately per generation
                # Since Scipy DE callback doesn't pass the current cost directly, we recalculate it.
                cost = cost_wrapper(xk)
                self.history.append(cost)

                if cost < 1e8:
                    if cost < (self.best_cost - self.tol):
                        self.best_cost = cost
                        self.counter = 0
                    else:
                        self.counter += 1

                print(
                    f"   Gen {len(self.history)}: Cost={cost:.2f} (Best={self.best_cost:.2f}) | Patience: {self.counter}/{self.patience}")

                if self.counter >= self.patience:
                    print(f"   --> Stopping Early: No improvement for {self.patience} generations.")
                    return True

        def cost_wrapper(x):
            return self.calculate_itae_cost(x[0], x[1])

        tracker = Tracker(self.patience, self.tol)

        # --- DYNAMIC BOUNDS BASED ON PLANT DIRECTION ---
        is_reverse = getattr(self, 'is_reverse_acting', False)

        if self.max_kp is None:
            safe_limit = -100.0 if is_reverse else 100.0
            print(f"   [!] No stability crossing found. Using fallback Kp boundary: {safe_limit}")
        else:
            safe_limit = float(self.max_kp)

        if is_reverse:
            min_kp, max_kp = safe_limit, -0.001
            min_ki, max_ki = -0.01, -0.00001
        else:
            min_kp, max_kp = 0.001, safe_limit
            min_ki, max_ki = 0.00001, 0.01

        # Format bounds for SciPy DE (List of Tuples)
        bounds = [(min_kp, max_kp), (min_ki, max_ki)]

        # 3. Apply the dynamically loaded DE configurations
        result = differential_evolution(
            cost_wrapper,
            bounds,
            maxiter=self.max_iters,
            popsize=self.pop_size,
            mutation=self.mutation,
            recombination=self.recombination,
            strategy=self.strategy,
            callback=tracker.callback,
            disp=False
        )

        return result.x[0], result.x[1], result.fun, len(tracker.history), tracker.history