import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
import itertools


def save_analysis_to_csv(file_list):
    dfs = []

    # 1. Load Data
    for file_path in file_list:
        try:
            df = pd.read_csv(file_path)
            algo_name = os.path.basename(file_path).split('_')[0].upper()
            df['Algorithm'] = algo_name
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    if not dfs:
        print("No valid data loaded.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)

    def analyze_and_save(metric_name, anova_file, fisher_file):
        print(f"Processing {metric_name}...")

        # --- ANOVA ---
        formula = f'{metric_name} ~ C(Algorithm)'
        model = ols(formula, data=combined_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Save ANOVA results
        anova_table.to_csv(anova_file)
        print(f"Saved ANOVA results to {anova_file}")

        # --- Fisher's LSD ---
        mse = model.mse_resid
        df_error = model.df_resid
        groups = combined_df['Algorithm'].unique()

        # Calculate group stats
        group_stats = {}
        for g in groups:
            sub = combined_df[combined_df['Algorithm'] == g][metric_name]
            group_stats[g] = {'mean': sub.mean(), 'n': len(sub)}

        fisher_results = []
        pairs = list(itertools.combinations(groups, 2))

        for g1, g2 in pairs:
            m1 = group_stats[g1]['mean']
            n1 = group_stats[g1]['n']
            m2 = group_stats[g2]['mean']
            n2 = group_stats[g2]['n']

            diff = m1 - m2

            # Standard Error
            se = np.sqrt(mse * (1 / n1 + 1 / n2))

            t_stat = diff / se
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=df_error))

            sig = "Yes" if p_val < 0.05 else "No"

            fisher_results.append({
                'Group1': g1,
                'Group2': g2,
                'Mean_Difference': diff,
                't_statistic': t_stat,
                'p_value': p_val,
                'Significant': sig
            })

        # Save Fisher's LSD results
        fisher_df = pd.DataFrame(fisher_results)
        fisher_df.to_csv(fisher_file, index=False)
        print(f"Saved Fisher's LSD results to {fisher_file}")

    # Run for Iterations
    analyze_and_save('Iterations_Run', 'anova_iterations.csv', 'fisher_lsd_iterations.csv')

    # Run for Final Cost
    analyze_and_save('Final_Cost_ITAE', 'anova_final_cost.csv', 'fisher_lsd_final_cost.csv')


if __name__ == "__main__":
    files = [
        'de/de_results_batch_2_50trials.csv',
        'ga/ga_results_batch_2_50trials.csv',
        'pso/pso_results_batch_2_50trials.csv'
    ]
    save_analysis_to_csv(files)