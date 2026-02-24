import numpy as np
import pyswarms as ps
import math
from Evolutionary_Algorithm_Testing.ea_optimizer import EvolutionaryOptimizer


class EarlyStopping(Exception):
    pass


class PSOOptimizer(EvolutionaryOptimizer):
    def __init__(self, config, tf_params):
        # 1. Initialize base class settings
        super().__init__(config, tf_params)

        # 2. Extract PSO-specific options with robust defaults
        # Use the Clerc-Kennedy constriction factor for assured convergence
        phi1 = config.get('phi1', 2.05)  # Cognitive parameter
        phi2 = config.get('phi2', 2.05)  # Social parameter
        phi = phi1 + phi2

        # compute the Clerc-Kennedy constriction factor
        chi = 2.0 / abs(2.0 - phi - math.sqrt(phi ** 2 - 4 * phi))

        # Convert the Clerc-Kennedy-based params to usable values for PySwarm
        self.w = chi
        self.c1 = chi * phi1
        self.c2 = chi * phi2

        # Package them for PySwarms
        self.options = {'c1': self.c1, 'c2': self.c2, 'w': self.w}

    def optimize_round(self, round_num):
        run_state = {
            'best_cost': float('inf'),
            'patience_counter': 0,
            'history': []
        }

        def objective_function(particles):
            n_particles = particles.shape[0]
            costs = []

            for i in range(n_particles):
                cost = self.calculate_itae_cost(particles[i, 0], particles[i, 1])

                # --- FIX: Sanitize costs to prevent PySwarms broadcast error ---
                if np.isinf(cost) or np.isnan(cost):
                    cost = 9.0  # Assign a heavy penalty cost instead of infinity

                costs.append(cost)

            costs_array = np.array(costs)
            current_best = np.min(costs_array)

            if current_best < run_state['best_cost']:
                if current_best < 1e8:
                    if current_best < (run_state['best_cost'] - self.tol):
                        run_state['patience_counter'] = 0
                    else:
                        run_state['patience_counter'] += 1
                    run_state['best_cost'] = current_best
            else:
                if run_state['best_cost'] < 1e8:
                    run_state['patience_counter'] += 1

            run_state['history'].append(run_state['best_cost'])

            print(
                f"   Gen {len(run_state['history'])}: Best={run_state['best_cost']:.2f} | Patience: {run_state['patience_counter']}/{self.patience}")

            if run_state['patience_counter'] >= self.patience:
                raise EarlyStopping()

            return costs_array

        # --- DYNAMIC BOUNDS BASED ON PLANT DIRECTION ---
        is_reverse = getattr(self, 'is_reverse_acting', False)

        # 1. Provide a fallback if no stability crossing was found (infinite gain margin)
        if self.max_kp is None:
            # If no limit is found, use a reasonably large limit to prevent infinite searches
            safe_limit = -100.0 if is_reverse else 100.0
            print(f"   [!] No stability crossing found. Using fallback Kp boundary: {safe_limit}")
        else:
            safe_limit = float(self.max_kp)

        # 2. Set bounds depending on direction
        if is_reverse:
            # Searching in negative territory
            # min bound must be smaller (more negative) than max bound
            min_kp, max_kp = safe_limit, -0.001
            min_ki, max_ki = -0.01, -0.00001
        else:
            # Standard direct-acting
            min_kp, max_kp = 0.001, safe_limit
            min_ki, max_ki = 0.00001, 0.01

        # 3. Explicitly force float64 so PySwarms doesn't complain
        bounds = (
            np.array([min_kp, min_ki], dtype=np.float64),
            np.array([max_kp, max_ki], dtype=np.float64)
        )

        # 4. Initialize the optimizer
        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.pop_size,
            dimensions=2,
            options=self.options,
            bounds=bounds
        )

        try:
            cost, pos = optimizer.optimize(objective_function, iters=self.max_iters, verbose=False)
        except EarlyStopping:
            print(f"   --> Stopping Early: No improvement for {self.patience} generations.")
            cost, pos = optimizer.swarm.best_cost, optimizer.swarm.best_pos
        except Exception as e:
            print(f"   [ERROR] {e}")
            cost, pos = optimizer.swarm.best_cost, optimizer.swarm.best_pos

        return pos[0], pos[1], cost, len(run_state['history']), run_state['history']