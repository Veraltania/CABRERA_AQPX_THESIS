import pygad
from Evolutionary_Algorithm_Testing.ea_optimizer import EvolutionaryOptimizer


class GAOptimizer(EvolutionaryOptimizer):
    def __init__(self, config, tf_params):
        super().__init__(config, tf_params)

        self.num_parents_mating = config.get('num_parents_mating', 10)
        self.parent_selection_type = config.get('parent_selection_type', "rank")
        self.keep_elitism = config.get('keep_elitism', 2)
        self.crossover_type = config.get('crossover_type', "scattered")

        # Explicitly setting the requested Crossover Probability (Pc)
        self.crossover_probability = 0.90

    def optimize_round(self, round_num):
        def cost_wrapper(solution):
            return self.calculate_cost(solution[0], solution[1])

        class Tracker:
            def __init__(self, patience, tol, cost_func):
                self.patience = patience
                self.tol = tol
                self.cost_func = cost_func
                self.counter = 0
                self.best_cost = float('inf')
                self.history = []

            def on_generation(self, ga_instance):
                # Apply dynamic Mutation factor (Pm = 0.1 - 0.01 * n/P)
                n = ga_instance.generations_completed
                P = ga_instance.num_generations
                ga_instance.mutation_probability = 0.1 - 0.01 * (n / P)

                # Evaluate best solution
                solution, _, _ = ga_instance.best_solution()
                cost = self.cost_func(solution)
                self.history.append(cost)

                if cost < 1e8:
                    if cost < (self.best_cost - self.tol):
                        self.best_cost = cost
                        self.counter = 0
                    else:
                        self.counter += 1

                gen_num = len(self.history)
                #if gen_num % 25 == 0 or self.counter >= self.patience:
                    #print(
                        #f"   Gen {gen_num}: Best={self.best_cost:.2f} | P: {self.counter}/{self.patience} | Pm: {ga_instance.mutation_probability:.4f}")

                if self.counter >= self.patience:
                    #print(f"   --> Stopping Early: No improvement for {self.patience} generations.")
                    return "stop"

        def fitness_wrapper(ga_instance, solution, solution_idx):
            cost = cost_wrapper(solution)
            return float(1.0 / (cost + 1e-8))

        tracker = Tracker(self.patience, self.tol, cost_wrapper)

        if self.max_kp is None:
            safe_limit = -100.0 if self.is_reverse_acting else 100.0
            print(f"   [!] No stability crossing found. Using fallback Kp boundary: {safe_limit}")
        else:
            safe_limit = float(self.max_kp)

        if self.is_reverse_acting:
            min_kp, max_kp = safe_limit, -0.001
            min_ki, max_ki = -0.01, -1e-6
        else:
            min_kp, max_kp = 0.001, safe_limit
            min_ki, max_ki = 1e-6, 0.01

        bounds = [{'low': min_kp, 'high': max_kp}, {'low': min_ki, 'high': max_ki}]

        ga_kwargs = {
            "num_generations": self.max_iters,
            "num_parents_mating": self.num_parents_mating,
            "fitness_func": fitness_wrapper,
            "sol_per_pop": self.pop_size,
            "num_genes": 2,
            "gene_space": bounds,
            "parent_selection_type": self.parent_selection_type,
            "keep_elitism": self.keep_elitism,
            "crossover_type": self.crossover_type,
            "crossover_probability": self.crossover_probability,  # Set Pc = 0.90
            "mutation_type": "random",  # 'random' needed to utilize exact scalar probabilities
            "mutation_probability": 0.1,  # Set initial Pm for Generation 0
            "on_generation": tracker.on_generation,
            "suppress_warnings": True
        }

        ga_instance = pygad.GA(**ga_kwargs)
        ga_instance.run()

        solution, _, _ = ga_instance.best_solution()
        final_cost = cost_wrapper(solution)

        return solution[0], solution[1], final_cost, len(tracker.history), tracker.history