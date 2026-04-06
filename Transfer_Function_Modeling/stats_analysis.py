import pandas as pd
from scipy import stats
import scikit_posthocs as sp
from pathlib import Path
import warnings

try:
    import pingouin as pg
except ImportError:
    print("Warning: 'pingouin' is not installed. Parametric RM ANOVA will fail. Please run 'pip install pingouin'.")


def run_repeated_measures_analysis(csv_filepath):
    # --- Set up Output Directory using pathlib ---
    input_path = Path(csv_filepath)
    if not input_path.exists():
        print(f"Error: Could not find the file '{csv_filepath}'.")
        return

    # Create a new folder on the same level as the input file
    results_dir = input_path.parent / f"{input_path.stem}_Results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Base name for the output files
    base_out_name = results_dir / input_path.stem

    # Load the CSV data AND TRANSPOSE IT (.T) so models become the columns
    df = pd.read_csv(input_path, index_col=0).T
    print(f"Data loaded successfully. Found {len(df.columns)} models to compare across {len(df)} trials.\n")

    alpha = 0.05
    is_normal = True
    normality_data = []

    # --- 1. Normality Test (Shapiro-Wilk) ---
    print("--- 1. Normality Check (Shapiro-Wilk Test) ---")
    for col in df.columns:
        data_col = df[col].dropna()
        stat, p_val = stats.shapiro(data_col)

        passed_normality = p_val >= alpha
        normality_data.append({
            'Model': col,
            'W-Statistic': round(stat, 4),
            'p-value': round(p_val, 4),
            'Is_Normal': passed_normality
        })

        print(f"Model '{col}': Statistic={stat:.3f}, p-value={p_val:.3f}")

        if not passed_normality:
            print(f"  -> Violated: '{col}' does not appear to be normally distributed.")
            is_normal = False

    # Save Normality Results
    normality_csv = f"{base_out_name}_Normality_Results.csv"
    pd.DataFrame(normality_data).to_csv(normality_csv, index=False)
    print(f"SUCCESS: Normality results saved to '{normality_csv}'\n")

    # RM ANOVA and Friedman both require complete cases (no missing data). Drop trials with any NaNs.
    df_clean = df.dropna()
    if len(df_clean) < len(df):
        print(f"Note: Dropped {len(df) - len(df_clean)} trials with missing data to balance the design.\n")

    # Convert to "long" format for Pingouin and scikit-posthocs
    df_long = df_clean.reset_index().melt(id_vars='index', var_name='Model', value_name='Score')
    df_long.rename(columns={'index': 'Trial'}, inplace=True)

    # =========================================================
    # PARAMETRIC PATH (Mauchly's + RM ANOVA + Paired t-tests)
    # =========================================================
    if is_normal:
        print("Conclusion: All models are normally distributed (Normality is OK).")
        print("ACTION: Proceeding with parametric Repeated Measures ANOVA...\n")

        # --- 2A. Sphericity Check (Mauchly's Test) ---
        print("--- 2A. Sphericity Check (Mauchly's Test) ---")
        try:
            # Catch the harmless "divide by zero" RuntimeWarning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                spher, W, chi_sq, ddof, p_val_spher = pg.sphericity(data=df_long, dv='Score', subject='Trial',
                                                                    within='Model')

            print(f"Sphericity met? {spher} (p-value = {p_val_spher:.3f})")

            # Save Sphericity Results
            sphericity_data = [{
                'Test': "Mauchly's",
                'Sphericity_Met': spher,
                'W': round(W, 4),
                'Chi-Square': round(chi_sq, 4),
                'DOF': ddof,
                'p-value': round(p_val_spher, 4)
            }]
            sphericity_csv = f"{base_out_name}_Sphericity_Results.csv"
            pd.DataFrame(sphericity_data).to_csv(sphericity_csv, index=False)
            print(f"SUCCESS: Sphericity results saved to '{sphericity_csv}'")

            if not spher:
                print("  -> Violated: Variances of the differences between models are not equal.")
                print("  -> ACTION: The ANOVA will automatically apply the Greenhouse-Geisser correction.\n")
            else:
                print("  -> Sphericity assumption is met.\n")
        except Exception as e:
            print(f"ERROR: Sphericity check failed. Details: {e}")
            return

        # --- 3A. Repeated Measures ANOVA ---
        print("--- 3A. Repeated Measures ANOVA ---")
        try:
            aovrm = pg.rm_anova(data=df_long, dv='Score', within='Model', subject='Trial', detailed=True)
            print(aovrm.round(4).to_string())

            # Save the ANOVA table directly to CSV
            anova_csv = f"{base_out_name}_ANOVA_table.csv"
            aovrm.to_csv(anova_csv, index=False)
            print(f"\nSUCCESS: ANOVA table successfully saved to '{anova_csv}'\n")

            # Extract the correct p-value based on sphericity
            if not spher and 'p-GG-corr' in aovrm.columns:
                p_val_anova = aovrm['p-GG-corr'].iloc[0]
                print("Using Greenhouse-Geisser corrected p-value for significance check.")
            else:
                p_val_anova = aovrm['p-unc'].iloc[0]
                print("Using uncorrected p-value for significance check.")

        except Exception as e:
            print(f"ERROR: Repeated Measures ANOVA failed. Details: {e}")
            return

        # Check if the ANOVA was significant before proceeding to post-hoc
        if p_val_anova >= alpha:
            print(f"\nConclusion: No significant differences found among the models (p = {p_val_anova:.3f}).")
            print("ACTION: Stopping here. No post-hoc testing is needed.")
            return

        print(
            f"\nConclusion: Significant differences found among models (p = {p_val_anova:.3f}). Proceeding to post-hoc testing...\n")

        # --- 4A. Paired t-tests Post-hoc (with Holm Correction) ---
        print("--- 4A. Paired t-tests Post-hoc (with Holm Correction) ---")

        # Pingouin automatically performs paired t-tests when 'within' and 'subject' are provided.
        posthoc_results = pg.pairwise_tests(data=df_long, dv='Score', within='Model', subject='Trial', padjust='holm')

        print("Pairwise p-values (p-corr < 0.05 indicates a significant difference):")

        # Display the most relevant columns in the terminal
        cols_to_print = ['A', 'B', 'T', 'dof', 'p-unc', 'p-corr']
        print(posthoc_results[cols_to_print].round(4).to_string(index=False))

        # Save the full detailed table (including effect sizes like Hedges g) to CSV
        posthoc_csv = f"{base_out_name}_Paired_ttests_Holm.csv"
        posthoc_results.round(4).to_csv(posthoc_csv, index=False)
        print(f"\nSUCCESS: Paired t-test post-hoc results successfully saved to '{posthoc_csv}'")

        return

    # =========================================================
    # NON-PARAMETRIC PATH (Friedman + Nemenyi)
    # =========================================================
    print("Conclusion: Normality assumption violated. Proceeding to non-parametric tests...\n")

    # --- 2B. Friedman Test ---
    print("--- 2B. Friedman Test ---")
    data_arrays = [df_clean[col] for col in df_clean.columns]

    stat, p_val = stats.friedmanchisquare(*data_arrays)
    print(f"Friedman Chi-Square Statistic: {stat:.3f}")
    print(f"p-value: {p_val:.3f}")

    # Save Friedman results
    friedman_csv = f"{base_out_name}_Friedman_Results.csv"
    pd.DataFrame([{'Test': 'Friedman', 'Chi-Square': round(stat, 4), 'p-value': round(p_val, 4)}]).to_csv(friedman_csv,
                                                                                                          index=False)
    print(f"SUCCESS: Friedman results saved to '{friedman_csv}'")

    if p_val >= alpha:
        print("\nConclusion: No significant differences found among the models (p >= 0.05).")
        print("ACTION: Stopping here. No post-hoc testing is needed.")
        return

    print("\nConclusion: Significant differences found among models. Proceeding to post-hoc testing...\n")

    # --- 3B. Nemenyi Post-hoc Test ---
    print("--- 3B. Nemenyi Post-hoc Test ---")
    nemenyi_results = sp.posthoc_nemenyi_friedman(df_clean.to_numpy())
    nemenyi_results.columns = df_clean.columns
    nemenyi_results.index = df_clean.columns

    print("Pairwise p-values (values < 0.05 indicate a significant difference):")
    print(nemenyi_results.round(3))

    # Save to CSV
    nemenyi_csv = f"{base_out_name}_Nemenyi_PostHoc.csv"
    nemenyi_results.round(3).to_csv(nemenyi_csv)
    print(f"\nSUCCESS: Nemenyi post-hoc results successfully saved to '{nemenyi_csv}'")


if __name__ == "__main__":
    input_csv_file = 'Model_R2_DO_Nighttime.csv'

    run_repeated_measures_analysis(input_csv_file)