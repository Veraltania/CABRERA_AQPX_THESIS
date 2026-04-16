import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

try:
    import pingouin as pg
except ImportError:
    print("Warning: 'pingouin' is not installed. Parametric RM ANOVA will fail. Please run 'pip install pingouin'.")

def plot_results(df_long, out_path, metric_name="Score"):
    """Generates and saves a boxplot with overlaid strip plot for the data."""
    plt.figure(figsize=(10, 6))
    
    # Create a boxplot to show distribution
    sns.boxplot(data=df_long, x='Group', y='Score', color='lightgray', showfliers=False)
    # Add a stripplot to show individual trial data points
    sns.stripplot(data=df_long, x='Group', y='Score', size=6, alpha=0.7, jitter=True, hue='Group', legend=False)
    
    plt.title(f'Distribution of {metric_name} across Groups')
    plt.ylabel(metric_name)
    plt.xlabel('Set-up / Group')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_file = f"{out_path}_boxplot.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"SUCCESS: Visualization saved to '{plot_file}'")

def run_statistical_pipeline(csv_filepath):
    # --- 1. Set up Paths and Load Data ---
    input_path = Path(csv_filepath)
    if not input_path.exists():
        print(f"Error: Could not find the file '{csv_filepath}'.")
        return

    results_dir = input_path.parent / f"stat_analysis_{input_path.stem}"
    results_dir.mkdir(parents=True, exist_ok=True)
    base_out_name = results_dir / input_path.stem

    # Read the CSV. 
    # index_col=0 sets the first column (groups) as the index.
    # .T transposes it so that Groups = Columns, Trials = Rows.
    df = pd.read_csv(input_path, index_col=0).T
    
    print(f"--- DATA LOADED: {input_path.name} ---")
    print(f"Groups identified ({len(df.columns)}): {list(df.columns)}")
    print(f"Trials identified ({len(df)}): {list(df.index)}\n")

    # Clean missing data (required for RM ANOVA / Friedman)
    df_clean = df.dropna()
    if len(df_clean) < len(df):
        print(f"Note: Dropped {len(df) - len(df_clean)} trials with missing data to balance the design.\n")

    # Melt to long format for plotting and Pingouin
    df_long = df_clean.reset_index().melt(id_vars='index', var_name='Group', value_name='Score')
    df_long.rename(columns={'index': 'Trial'}, inplace=True)

    # Generate Visualization
    plot_results(df_long, base_out_name, metric_name=input_path.stem)

    # --- 2. Normality Check (Shapiro-Wilk) ---
    print("\n--- 1. NORMALITY CHECK (Shapiro-Wilk) ---")
    alpha = 0.05
    is_normal = True
    normality_data = []

    for col in df_clean.columns:
        stat, p_val = stats.shapiro(df_clean[col])
        passed = p_val >= alpha
        normality_data.append({
            'Group': col, 'W-Statistic': round(stat, 4), 
            'p-value': round(p_val, 4), 'Is_Normal': passed
        })
        print(f"Group '{col}': p-value = {p_val:.4f} ({'Normal' if passed else 'Violated'})")
        if not passed: is_normal = False

    pd.DataFrame(normality_data).to_csv(f"{base_out_name}_Normality.csv", index=False)

    # --- 3. Branching Logic: Parametric vs Non-Parametric ---
    if is_normal:
        print("\nConclusion: All groups are normally distributed. Proceeding with PARAMETRIC tests.")
        
        num_groups = len(df_clean.columns)

        # --- Handle 2 Groups (Paired t-test) ---
        if num_groups == 2:
            print("\n--- 2A. PAIRED T-TEST (2 Groups) ---")
            g1, g2 = df_clean.columns[0], df_clean.columns[1]
            
            # Perform paired t-test
            stat, p_val = stats.ttest_rel(df_clean[g1], df_clean[g2])
            print(f"Paired t-test Statistic: {stat:.4f}, p-value: {p_val:.4f}")
            
            pd.DataFrame([{'Test': 'Paired t-test', 'Group_A': g1, 'Group_B': g2, 'Statistic': stat, 'p-value': p_val}]).to_csv(
                f"{base_out_name}_Paired_Ttest.csv", index=False)
            
            if p_val < alpha:
                print(f"\nResult: Significant difference found between '{g1}' and '{g2}' (p = {p_val:.4f}).")
            else:
                print(f"\nResult: No significant difference found (p = {p_val:.4f}).")
                
            print("Action: Only 2 groups were analyzed, so no post-hoc testing is needed.")

        # --- Handle 3+ Groups (RM ANOVA) ---
        else:
            print("\n--- 2A. SPHERICITY CHECK (Mauchly's) ---")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    spher, W, chi_sq, ddof, p_val_spher = pg.sphericity(
                        data=df_long, dv='Score', subject='Trial', within='Group'
                    )
                print(f"Sphericity met? {spher} (p = {p_val_spher:.4f})")
                pd.DataFrame([{
                    'Test': "Mauchly's", 'Sphericity_Met': spher, 'p-value': round(p_val_spher, 4)
                }]).to_csv(f"{base_out_name}_Sphericity.csv", index=False)
            except Exception as e:
                print(f"Sphericity check failed: {e}")
                return

            # Repeated Measures ANOVA
            print("\n--- 3A. REPEATED MEASURES ANOVA ---")
            aovrm = pg.rm_anova(data=df_long, dv='Score', within='Group', subject='Trial', detailed=True)
            aovrm.to_csv(f"{base_out_name}_ANOVA.csv", index=False)
            
            # Dynamically check which columns Pingouin generated
            desired_cols = ['Source', 'ddof1', 'ddof2', 'DF', 'F', 'p-unc', 'p-GG-corr']
            actual_cols = [c for c in desired_cols if c in aovrm.columns]
            print(aovrm[actual_cols].round(4).to_string(index=False))

            # Determine which p-value to use based on Sphericity
            p_val_anova = aovrm['p-GG-corr'].iloc[0] if (not spher and 'p-GG-corr' in aovrm.columns) else aovrm['p-unc'].iloc[0]

            if p_val_anova >= alpha:
                print(f"\nResult: No significant difference found (p = {p_val_anova:.4f}). Analysis complete.")
                return
                
            print(f"\nResult: Significant difference found (p = {p_val_anova:.4f}). Running Post-hoc...")

            # Paired t-tests Post-hoc (Holm)
            print("\n--- 4A. POST-HOC: Paired T-Tests (Holm correction) ---")
            posthoc = pg.pairwise_tests(data=df_long, dv='Score', within='Group', subject='Trial', padjust='holm')
            posthoc.round(4).to_csv(f"{base_out_name}_PostHoc_TTest.csv", index=False)
            
            posthoc_desired_cols = ['A', 'B', 'T', 'dof', 'p-unc', 'p-corr']
            posthoc_actual_cols = [c for c in posthoc_desired_cols if c in posthoc.columns]
            print(posthoc[posthoc_actual_cols].round(4).to_string(index=False))
            
    else:
        print("\nConclusion: Normality violated. Proceeding with NON-PARAMETRIC tests.")
        
        num_groups = len(df_clean.columns)
        
        # --- Handle 2 Groups (Wilcoxon) ---
        if num_groups == 2:
            print("\n--- 2B. WILCOXON SIGNED-RANK TEST (2 Groups) ---")
            g1, g2 = df_clean.columns[0], df_clean.columns[1]
            
            # Perform Wilcoxon signed-rank test
            stat, p_val = stats.wilcoxon(df_clean[g1], df_clean[g2])
            print(f"Wilcoxon Statistic: {stat:.4f}, p-value: {p_val:.4f}")
            
            pd.DataFrame([{'Test': 'Wilcoxon Signed-Rank', 'Group_A': g1, 'Group_B': g2, 'Statistic': stat, 'p-value': p_val}]).to_csv(
                f"{base_out_name}_Wilcoxon.csv", index=False)
            
            if p_val < alpha:
                print(f"\nResult: Significant difference found between '{g1}' and '{g2}' (p = {p_val:.4f}).")
            else:
                print(f"\nResult: No significant difference found (p = {p_val:.4f}).")
            
            print("Action: Only 2 groups were analyzed, so no post-hoc testing is needed.")
            
        # --- Handle 3+ Groups (Friedman + Nemenyi) ---
        else:
            print(f"\n--- 2B. FRIEDMAN TEST ({num_groups} Groups) ---")
            data_arrays = [df_clean[col] for col in df_clean.columns]
            stat, p_val = stats.friedmanchisquare(*data_arrays)
            print(f"Friedman Chi-Square: {stat:.4f}, p-value: {p_val:.4f}")
            
            pd.DataFrame([{'Test': 'Friedman', 'Chi-Square': stat, 'p-value': p_val}]).to_csv(
                f"{base_out_name}_Friedman.csv", index=False)

            if p_val >= alpha:
                print("\nResult: No significant difference found. Analysis complete.")
                return

            print("\nResult: Significant difference found. Running Post-hoc...")

            # Nemenyi Post-hoc
            print("\n--- 3B. POST-HOC: Nemenyi Test ---")
            nemenyi = sp.posthoc_nemenyi_friedman(df_clean.to_numpy())
            nemenyi.columns = df_clean.columns
            nemenyi.index = df_clean.columns
            nemenyi.round(4).to_csv(f"{base_out_name}_PostHoc_Nemenyi.csv")
            print("Pairwise p-values (p < 0.05 is significant):")
            print(nemenyi.round(4))

if __name__ == "__main__":
    # You can loop through multiple files easily here:
    target_files = ['Simulation_Testing/simulation_graphs_controller_aging/controller_aging_control_effort.csv', 
                    'Simulation_Testing/simulation_graphs_controller_aging/controller_aging_iae.csv',
                    'Simulation_Testing/simulation_graphs_disturbance_do/Control_Effort_AUC_table.csv',
                    'Simulation_Testing/simulation_graphs_disturbance_do/IAE_table.csv']
    
    for file in target_files:
        print(f"\n{'='*50}")
        print(f"STARTING ANALYSIS FOR: {file}")
        print(f"{'='*50}")
        run_statistical_pipeline(file)