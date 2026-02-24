import pygad
from Evolutionary_Algorithm_Testing.ea_optimizer import EvolutionaryOptimizer


class GAOptimizer(EvolutionaryOptimizer):
    def __init__(self, config, tf_params):
        # 1. Initialize the base class to get self.pop_size, self.max_iters, etc.
        super().__init__(config, tf_params)

        # 2. Extract GA-specific configurations strictly from the user's config
        self.num_parents_mating = config.get('num_parents_mating', 10)
        self.parent_selection_type = config.get('parent_selection_type', "rank")
        self.keep_elitism = config.get('keep_elitism', 2)

        self.crossover_type = config.get('crossover_type', "scattered")
        self.mutation_type = config.get('mutation_type', "adaptive")

        # 3. Handle mutation parameters based on the mutation type chosen
        if self.mutation_type == "adaptive":
            self.mutation_probability = config.get('mutation_probability', [0.25, 0.05])
        else:
            self.mutation_percent_genes = config.get('mutation_percent_genes', 20)

    def optimize_round(self, round_num):
        class Tracker:
            def __init__(self, patience, tol, cost_func):
                self.patience = patience
                self.tol = tol
                self.cost_func = cost_func
                self.counter = 0
                self.best_cost = float('inf')
                self.history = []

            def on_generation(self, ga_instance):
                solution, _, _ = ga_instance.best_solution()
                cost = self.cost_func(solution)
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
                    return "stop"

        def cost_wrapper(solution):
            return self.calculate_itae_cost(solution[0], solution[1])

        def fitness_wrapper(ga_instance, solution, solution_idx):
            cost = cost_wrapper(solution)
            return float(1.0 / (cost + 1e-8))

        tracker = Tracker(self.patience, self.tol, cost_wrapper)
        bounds = [{'low': 0.001, 'high': self.max_kp}, {'low': 0.0, 'high': 0.01}]

        # 4. Construct the PyGAD arguments dynamically
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
            "mutation_type": self.mutation_type,
            "on_generation": tracker.on_generation,
            "suppress_warnings": True
        }

        # Apply specific mutation setting depending on the chosen mutation type
        if self.mutation_type == "adaptive":
            ga_kwargs["mutation_probability"] = self.mutation_probability
        else:
            ga_kwargs["mutation_percent_genes"] = self.mutation_percent_genes

        # Instantiate GA with unpacked dictionary
        ga_instance = pygad.GA(**ga_kwargs)

        ga_instance.run()
        solution, _, _ = ga_instance.best_solution()
        final_cost = cost_wrapper(solution)

        return solution[0], solution[1], final_cost, len(tracker.history), tracker.history