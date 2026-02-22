import numpy as np
import pyswarms as ps
from Evolutionary_Algorithm_Testing.ea_optimizer import EvolutionaryOptimizer

class EarlyStopping(Exception):
    pass

class PSOOptimizer(EvolutionaryOptimizer):
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
        options = {'c1': 0.7, 'c2': 0.5, 'w': 0.9}

        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.pop_size,
            dimensions=2,
            options=options,
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