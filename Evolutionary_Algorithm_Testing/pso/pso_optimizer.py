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

        bounds = (np.array([0.001, 0.0]), np.array([self.max_kp, 0.001]))

        # 3. Use the dynamically loaded options here
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