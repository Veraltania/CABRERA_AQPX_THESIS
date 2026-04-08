import os
import csv
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- GENERIC WORKER FUNCTION ---
# Defined at the top level for multiprocessing pickling
def generic_worker_run(args):
    val, val_label, pop_size, tf_params, base_config, master_sweep_dir, opt_class, param_keys = args

    run_config = base_config.copy()
    run_config['population_size'] = pop_size
    
    # Dynamically apply the sweep parameter(s) to the config
    for key in param_keys:
        run_config[key] = val

    bin_folder_name = os.path.join(master_sweep_dir, f"bin_{val_label}")
    run_config['output_folder'] = bin_folder_name

    # Instantiate the dynamic optimizer class
    optimizer = opt_class(run_config, tf_params)
    optimizer.run_experiment()

    # Extract histories
    bin_costs = optimizer.agg_history['costs']
    bin_iters = optimizer.agg_history['iterations']
    raw_histories = optimizer.agg_history.get('histories', [])
    
    avg_convergence_curve = None
    if raw_histories:
        padded_histories = []
        max_len = run_config['max_iters']
        for h in raw_histories:
            h_list = list(h)
            if not h_list: continue
            if len(h_list) < max_len:
                h_list.extend([h_list[-1]] * (max_len - len(h_list)))
            elif len(h_list) > max_len:
                h_list = h_list[:max_len]
            padded_histories.append(h_list)
        if padded_histories:
            avg_convergence_curve = np.mean(padded_histories, axis=0)

    # Compile dynamic stats
    bin_stats_cost = {
        'Population': pop_size, 'Param_Value': val_label,
        'Min_Cost': np.min(bin_costs), 'Max_Cost': np.max(bin_costs),
        'Avg_Cost': np.mean(bin_costs), 'Std_Cost': np.std(bin_costs)
    }

    bin_stats_iter = {
        'Population': pop_size, 'Param_Value': val_label,
        'Min_Iter': np.min(bin_iters), 'Max_Iter': np.max(bin_iters),
        'Avg_Iter': np.mean(bin_iters), 'Std_Iter': np.std(bin_iters)
    }
    
    return {
        'param_label': val_label,
        'bin_costs': bin_costs, 'bin_iters': bin_iters,
        'bin_stats_cost': bin_stats_cost, 'bin_stats_iter': bin_stats_iter,
        'avg_convergence_curve': avg_convergence_curve
    }


# --- MODULAR SWEEPER CLASS ---
class OptimizationSweeper:
    def __init__(self, optimizer_class, sweep_config, transfer_functions, base_config, output_dir):
        self.opt_class = optimizer_class
        self.transfer_functions = transfer_functions
        self.base_config = base_config
        self.output_dir = output_dir
        
        # Unpack Sweep Configuration
        self.param_label = sweep_config['label']       
        self.param_keys = sweep_config['keys']         
        self.param_values = sweep_config['values']     
        self.pop_sizes = sweep_config['pop_sizes']
        
        self.total_cores = multiprocessing.cpu_count()
        self.use_cores = max(1, int(self.total_cores * 0.75))
        os.makedirs(self.output_dir, exist_ok=True)

    def run_sweep(self):
        print(f"STARTING SWEEP: {self.opt_class.__name__} | Param: {self.param_label}")
        print(f"Global Directory: {self.output_dir}")
        print(f"Using {self.use_cores} of {self.total_cores} cores.\n")

        with multiprocessing.Pool(processes=self.use_cores) as pool:
            for tf_name, tf_params in self.transfer_functions.items():
                self._run_tf_sweep(tf_name, tf_params, pool)

    def _run_tf_sweep(self, tf_name, tf_params, pool):
        print(f"\n{'=' * 80}\nINITIATING SWEEP FOR TRANSFER FUNCTION: {tf_name}\n{'=' * 80}")
        tf_level_dir = os.path.join(self.output_dir, f"results_{tf_name}")
        os.makedirs(tf_level_dir, exist_ok=True)

        final_report_cost, final_report_iter = [], []

        for pop_size in self.pop_sizes:
            master_sweep_dir = os.path.join(tf_level_dir, f"sweep_{self.param_label.lower()}_pop-{pop_size}")
            os.makedirs(master_sweep_dir, exist_ok=True)

            tasks = [
                (val, f"{val:.4f}", pop_size, tf_params, self.base_config, master_sweep_dir, self.opt_class, self.param_keys)
                for val in self.param_values
            ]

            results = list(tqdm(
                pool.imap(generic_worker_run, tasks), 
                total=len(tasks), desc=f"Running (Pop: {pop_size})", unit="bin"
            ))

            self._generate_visualizations(results, tf_name, pop_size, master_sweep_dir)
            self._save_local_reports(results, pop_size, master_sweep_dir)

            final_report_cost.extend([r['bin_stats_cost'] for r in results])
            final_report_iter.extend([r['bin_stats_iter'] for r in results])

        self._save_master_reports(final_report_cost, final_report_iter, tf_name, tf_level_dir)

    def _generate_visualizations(self, results, tf_name, pop_size, output_dir):
        labels = [r['param_label'] for r in results]
        avg_costs = [r['bin_stats_cost']['Avg_Cost'] for r in results]
        all_costs = [r['bin_costs'] for r in results]
        avg_iters = [r['bin_stats_iter']['Avg_Iter'] for r in results]
        all_iters = [r['bin_iters'] for r in results]
        curves = [r['avg_convergence_curve'] for r in results]

        def save_plot(fig_func, filename):
            plt.figure(figsize=(10, 6))
            fig_func()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()

        save_plot(lambda: (
            plt.plot(labels, avg_costs, marker='o', color='b'),
            plt.title(f'Avg Cost vs {self.param_label} ({tf_name} | Pop: {pop_size})'),
            plt.xlabel(self.param_label), plt.ylabel('Average Cost'), plt.xticks(rotation=45),
            plt.ylim(bottom=0)
        ), 'average_cost_line_plot.png')

        save_plot(lambda: (
            plt.boxplot(all_costs, tick_labels=labels, showfliers=True),
            plt.title(f'Cost Distribution ({tf_name} | Pop: {pop_size})'),
            plt.xlabel(self.param_label), plt.ylabel('Cost'), plt.xticks(rotation=45),
            plt.ylim(bottom=0)
        ), 'cost_distribution_boxplot.png')

        save_plot(lambda: (
            plt.plot(labels, avg_iters, marker='s', color='g'),
            plt.title(f'Avg Iterations vs {self.param_label} ({tf_name} | Pop: {pop_size})'),
            plt.xlabel(self.param_label), plt.ylabel('Average Iterations'), plt.xticks(rotation=45),
            plt.ylim(bottom=0)
        ), 'average_iterations_line_plot.png')

        if any(c is not None for c in curves):
            plt.figure(figsize=(12, 8))
            colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
            for i, (label, curve) in enumerate(zip(labels, curves)):
                if curve is not None:
                    plt.plot(range(len(curve)), curve, label=f'{self.param_label}={label}', color=colors[i])
            plt.title(f'Convergence by {self.param_label} ({tf_name})')
            plt.yscale('log')
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'convergence_comparison.png'))
            plt.close()

    def _write_csv(self, path, headers, data_dicts, mapping_keys):
        with open(path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for d in data_dicts:
                writer.writerow([d[k] for k in mapping_keys])

    def _save_local_reports(self, results, pop_size, output_dir):
        cost_headers = ['Population_Size', f'{self.param_label}_Value', 'Lowest_Cost', 'Highest_Cost', 'Average_Cost', 'Std_Dev']
        cost_keys = ['Population', 'Param_Value', 'Min_Cost', 'Max_Cost', 'Avg_Cost', 'Std_Cost']
        self._write_csv(os.path.join(output_dir, f"report_costs_pop_{pop_size}.csv"), cost_headers, [r['bin_stats_cost'] for r in results], cost_keys)
        
        iter_headers = ['Population_Size', f'{self.param_label}_Value', 'Least_Iters', 'Most_Iters', 'Avg_Iters', 'Std_Dev']
        iter_keys = ['Population', 'Param_Value', 'Min_Iter', 'Max_Iter', 'Avg_Iter', 'Std_Iter']
        self._write_csv(os.path.join(output_dir, f"report_iters_pop_{pop_size}.csv"), iter_headers, [r['bin_stats_iter'] for r in results], iter_keys)

    def _save_master_reports(self, final_costs, final_iters, tf_name, output_dir):
        cost_headers = ['Population_Size', f'{self.param_label}_Value', 'Lowest_Cost', 'Highest_Cost', 'Average_Cost', 'Std_Dev']
        cost_keys = ['Population', 'Param_Value', 'Min_Cost', 'Max_Cost', 'Avg_Cost', 'Std_Cost']
        self._write_csv(os.path.join(output_dir, f"master_cost_report_{tf_name}.csv"), cost_headers, final_costs, cost_keys)

        iter_headers = ['Population_Size', f'{self.param_label}_Value', 'Least_Iters', 'Most_Iters', 'Avg_Iters', 'Std_Dev']
        iter_keys = ['Population', 'Param_Value', 'Min_Iter', 'Max_Iter', 'Avg_Iter', 'Std_Iter']
        self._write_csv(os.path.join(output_dir, f"master_iters_report_{tf_name}.csv"), iter_headers, final_iters, iter_keys)