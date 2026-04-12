import numpy as np
from scipy.optimize import differential_evolution
from Evolutionary_Algorithm_Testing.ea_optimizer import EvolutionaryOptimizer

class DEOptimizer(EvolutionaryOptimizer):
    def __init__(self, config, tf_params):
        super().__init__(config, tf_params)
        self.mutation = config.get('mutation', (0.5, 1.0))
        self.recombination = config.get('recombination', 0.745)
        self.strategy = config.get('strategy', 'best1bin')

    def optimize_round(self, round_num):
        best_sol_tracker = {'x': None, 'cost': float('inf'), 'raw': None}

        def scalar_cost_wrapper(x):
            Kp, Ki = x[0], x[1]
            # Fetch unsummed parameters
            raw_costs = self.calculate_cost(Kp, Ki)
            
            if raw_costs[0] >= 1e8:
                return 1e9

            # Apply weights and sum
            weighted_cost = sum(w * c for w, c in zip(self.weights, raw_costs))

            # Track best manually for logging purposes
            if weighted_cost < best_sol_tracker['cost']:
                best_sol_tracker['cost'] = weighted_cost
                best_sol_tracker['x'] = (Kp, Ki)
                best_sol_tracker['raw'] = raw_costs

            return weighted_cost

        # Tracker for Patience Early Stopping
        class Tracker:
            def __init__(self, patience, tol):
                self.patience = patience
                self.tol = tol
                self.counter = 0
                self.best_cost = float('inf')

            def callback(self, xk, convergence=None):
                cost = scalar_cost_wrapper(xk)
                if cost < 1e8:
                    if cost < (self.best_cost - self.tol):
                        self.best_cost = cost
                        self.counter = 0
                    else:
                        self.counter += 1
                if self.counter >= self.patience:
                    return True

        tracker = Tracker(self.patience, self.tol)
        safe_limit = float(self.max_kp) if self.max_kp is not None else (-2.0 if self.is_reverse_acting else 2.0)

        if self.is_reverse_acting:
            bounds = [(safe_limit, -0.001), (-0.01, -0.0001)]
        else:
            bounds = [(0.001, safe_limit), (0.0001, 0.01)]

        result = differential_evolution(
            scalar_cost_wrapper,
            bounds,
            maxiter=self.max_iters,
            popsize=self.pop_size,
            mutation=self.mutation,
            recombination=self.recombination,
            strategy=self.strategy,
            callback=tracker.callback,
            polish=False
        )

        final_Kp, final_Ki = best_sol_tracker['x']
        return (final_Kp, final_Ki, best_sol_tracker['cost'], best_sol_tracker['raw']), result.nit