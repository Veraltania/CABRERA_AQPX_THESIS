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

        # 2. Extract/Set PSO-specific options based on the required configuration
        self.w = config.get('w', 0.6)  # Weight value W = 0.6
        self.c1 = config.get('c1', 2.0)  # Acceleration factor c1 = 2.0
        self.c2 = config.get('c2', 2.0)  # Acceleration factor c2 = 2.0

        # Override inherited settings to match requirements
        self.pop_size = config.get('pop_size', 100)
        self.max_iters = config.get('max_iters', 100)

        # Velocity bounds (Vmin = -1.0, Vmax = 1.0)
        self.v_min = config.get('v_min', -1.0)
        self.v_max = config.get('v_max', 1.0)

        # Package them for PySwarms
        self.options = {'c1': self.c1, 'c2': self.c2, 'w': self.w}

    def optimize_round(self, round_num):
        run_state = {
            'best_cost': float('inf'),
            'patience_counter': 0,
            'history': []
        }

        def objective_function(particles):
            # 1. Faster iteration: Evaluate costs using a list comprehension.
            # This avoids the heavy overhead of repeated Python .append() calls.
            costs_array = np.array([
                self.calculate_cost(p[0], p[1]) for p in particles
            ])

            # 2. Vectorized sanitization: Replace NaNs and Infs across the whole array instantly.
            # This replaces the element-wise `if np.isinf(cost)` checks.
            costs_array = np.nan_to_num(costs_array, nan=9.0, posinf=9.0, neginf=9.0)

            # 3. Find the best cost for this iteration
            current_best = np.min(costs_array)

            # --- Early Stopping & Tracking Logic ---
            if current_best < run_state['best_cost']:
                if current_best < 1e8:
                    if current_best < (run_state['best_cost'] - getattr(self, 'tol', 1e-4)):
                        run_state['patience_counter'] = 0
                    else:
                        run_state['patience_counter'] += 1
                    run_state['best_cost'] = current_best
            else:
                if run_state['best_cost'] < 1e8:
                    run_state['patience_counter'] += 1

            run_state['history'].append(run_state['best_cost'])

            if run_state['patience_counter'] >= getattr(self, 'patience', 10):
                raise EarlyStopping()

            return costs_array

        # --- DYNAMIC BOUNDS BASED ON PLANT DIRECTION ---
        is_reverse = getattr(self, 'is_reverse_acting', False)

        # 1. Provide a fallback if no stability crossing was found (infinite gain margin)
        if getattr(self, 'max_kp', None) is None:
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
            min_ki, max_ki = -0.01, -0.0001
        else:
            # Standard direct-acting
            min_kp, max_kp = 0.001, safe_limit
            min_ki, max_ki = 0.0001, 0.01

        # 3. Explicitly force float64 so PySwarms doesn't complain
        bounds = (
            np.array([min_kp, min_ki], dtype=np.float64),
            np.array([max_kp, max_ki], dtype=np.float64)
        )

        # 4. Pack the velocity clamp
        velocity_clamp = (self.v_min, self.v_max)

        # 5. Initialize the optimizer
        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.pop_size,
            dimensions=2,
            options=self.options,
            bounds=bounds,
            velocity_clamp=velocity_clamp
        )

        try:
            cost, pos = optimizer.optimize(objective_function, iters=self.max_iters, verbose=False)
        except EarlyStopping:
            #print(f"   --> Stopping Early: No improvement for {getattr(self, 'patience', 10)} generations.")
            cost, pos = optimizer.swarm.best_cost, optimizer.swarm.best_pos
        except Exception as e:
            print(f"   [ERROR] {e}")
            cost, pos = optimizer.swarm.best_cost, optimizer.swarm.best_pos

        return pos[0], pos[1], cost, len(run_state['history']), run_state['history']