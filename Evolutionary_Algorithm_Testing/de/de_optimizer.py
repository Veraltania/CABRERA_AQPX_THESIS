import numpy as np
from scipy.optimize import differential_evolution
from Evolutionary_Algorithm_Testing.ea_optimizer import EvolutionaryOptimizer


class _CostConvergenceTracker:
    """Track one cost-based patience rule shared by all optimizers."""

    def __init__(self, patience, tolerance, penalty_cost):
        self.patience = patience
        self.tolerance = tolerance
        self.penalty_cost = penalty_cost
        self.counter = 0
        self.best_cost = float('inf')
        self.history = []

    def update(self, current_best):
        if not np.isfinite(current_best):
            current_best = self.penalty_cost

        if current_best < (self.best_cost - self.tolerance):
            self.best_cost = current_best
            self.counter = 0
        else:
            self.counter += 1

        self.history.append(self.best_cost)
        return self.counter >= self.patience


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
            raw_costs = self.calculate_cost(Kp, Ki)

            if self.is_penalty_costs(raw_costs):
                return np.inf

            weighted_cost = self.weighted_cost(raw_costs)
        
            if weighted_cost < best_sol_tracker['cost']:
                best_sol_tracker['cost'] = weighted_cost
                best_sol_tracker['x'] = (Kp, Ki)
                best_sol_tracker['raw'] = raw_costs

            return weighted_cost

        tracker = _CostConvergenceTracker(
            self.patience, self.tol, self.scalar_penalty
        )

        def convergence_callback(xk, convergence=None):
            return tracker.update(scalar_cost_wrapper(xk))
        
        # Applying boundaries extracted from config in base class
        bounds = [(self.min_kp, self.max_kp), (self.min_ki, self.max_ki)]

        result = differential_evolution(
            scalar_cost_wrapper,
            bounds,
            maxiter=self.max_iters,
            popsize=self.pop_size,
            mutation=self.mutation,
            recombination=self.recombination,
            strategy=self.strategy,
            callback=convergence_callback,
            # Disable SciPy's population-dispersion stop so all algorithms
            # use only the shared cost-patience rule and max_iters.
            tol=0.0,
            atol=-np.inf,
            polish=False
        )

        if best_sol_tracker['x'] is None:
            final_Kp, final_Ki = 0.0, 0.0
            best_sol_tracker['cost'] = self.scalar_penalty
            best_sol_tracker['raw'] = self.penalty_costs
        else:
            final_Kp, final_Ki = best_sol_tracker['x']

        return (final_Kp, final_Ki, best_sol_tracker['cost'], best_sol_tracker['raw']), result.nit, tracker.history
