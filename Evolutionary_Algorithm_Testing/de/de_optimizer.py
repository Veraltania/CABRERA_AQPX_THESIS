from scipy.optimize import differential_evolution
from Evolutionary_Algorithm_Testing.ea_optimizer import EvolutionaryOptimizer

class DEOptimizer(EvolutionaryOptimizer):
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
        bounds = [(0.001, self.max_kp), (0.0, 0.001)]

        result = differential_evolution(
            cost_wrapper,
            bounds,
            maxiter=self.max_iters,
            popsize=self.pop_size,
            mutation=(0.5, 1),
            recombination=0.7,
            strategy='best1bin',
            callback=tracker.callback,
            disp=False
        )

        return result.x[0], result.x[1], result.fun, len(tracker.history), tracker.history