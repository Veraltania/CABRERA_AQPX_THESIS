import numpy as np
import pyswarms as ps
from Evolutionary_Algorithm_Testing.ea_optimizer import EvolutionaryOptimizer

class EarlyStopping(Exception):
    pass

class PSOOptimizer(EvolutionaryOptimizer):
    def __init__(self, config, tf_params):
        super().__init__(config, tf_params)
        self.w = config.get('w', 0.6)
        self.c1 = config.get('c1', 2.0)
        self.c2 = config.get('c2', 2.0)
        self.v_min = config.get('v_min', -1.0)
        self.v_max = config.get('v_max', 1.0)

    def optimize_round(self, round_num):
        best_sol_tracker = {'x': None, 'cost': float('inf'), 'raw': None}
        particle_penalty = self.scalar_penalty
        
        run_state = {
            'best_cost': float('inf'),
            'patience_counter': 0,
            'history': []
        }

        def objective_function(particles):
            costs = []
            for p in particles:
                Kp, Ki = p[0], p[1]
                raw_costs = self.calculate_cost(Kp, Ki)

                if self.is_penalty_costs(raw_costs):
                    weighted_cost = particle_penalty
                else:
                    weighted_cost = self.weighted_cost(raw_costs)
                    if weighted_cost < best_sol_tracker['cost']:
                        best_sol_tracker['cost'] = weighted_cost
                        best_sol_tracker['x'] = (Kp, Ki)
                        best_sol_tracker['raw'] = raw_costs

                costs.append(weighted_cost)

            costs_array = np.array(costs)
            current_best = np.min(costs_array)

            if current_best < particle_penalty:
                if current_best < (run_state['best_cost'] - self.tol):
                    run_state['patience_counter'] = 0
                    run_state['best_cost'] = current_best
                elif np.isfinite(run_state['best_cost']):
                    run_state['patience_counter'] += 1

            run_state['history'].append(
                run_state['best_cost'] if np.isfinite(run_state['best_cost'])
                else self.scalar_penalty
            )

            if run_state['patience_counter'] >= self.patience:
                raise EarlyStopping()

            return costs_array

        # Applying boundaries extracted from config in base class
        bounds = (
            np.array([self.min_kp, self.min_ki], dtype=np.float64),
            np.array([self.max_kp, self.max_ki], dtype=np.float64)
        )
        velocity_clamp = (self.v_min, self.v_max)
        options = {'c1': self.c1, 'c2': self.c2, 'w': self.w}

        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.pop_size,
            dimensions=2,
            options=options,
            bounds=bounds,
            velocity_clamp=velocity_clamp
        )

        iterations_run = self.max_iters
        try:
            _ = optimizer.optimize(objective_function, iters=self.max_iters, verbose=False)
        except EarlyStopping:
            iterations_run = len(run_state['history'])

        if best_sol_tracker['x'] is None:
            final_Kp, final_Ki = 0.0, 0.0
            best_sol_tracker['cost'] = self.scalar_penalty
            best_sol_tracker['raw'] = self.penalty_costs
        else:
            final_Kp, final_Ki = best_sol_tracker['x']

        return (final_Kp, final_Ki, best_sol_tracker['cost'], best_sol_tracker['raw']), iterations_run, run_state['history']
