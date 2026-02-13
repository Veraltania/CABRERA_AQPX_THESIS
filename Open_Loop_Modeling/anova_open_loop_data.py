import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg # Great for blocked post-hocs

# 1. Load the CSV file
df_raw = pd.read_csv(r'C:\Users\Dindo\Documents\aqpx\Cabrera_Thesis_AQPX\Open_Loop_Modeling\accuracy_metrics_nighttime.csv')

# 2. Adaptive Transformation
# Assume 1st column is the Model name, and the rest are Date columns
model_col_name = df_raw.columns[0]
date_col_names = df_raw.columns[1:].tolist()

# Convert from 'Wide' format to 'Long' format (required for ANOVA)
df_long = pd.melt(
    df_raw,
    id_vars=[model_col_name],
    value_vars=date_col_names,
    var_name='Trial_Date',
    value_name='Metric_Value'
)

# Rename columns internally for easy formula writing
df_long.columns = ['Model', 'Date', 'Value']

# 3. Statistical Analysis (Two-Way ANOVA / RCBD)
# Value ~ C(Model) tests the treatments
# C(Date) accounts for the variation between days (trials)
model = ols('Value ~ C(Model) + C(Date)', data=df_long).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# 4. Results
print("--- ANOVA Results ---")
print(anova_table)

# Interpretation
p_val = anova_table.loc['C(Model)', 'PR(>F)']
print(f"\nP-value for Models: {p_val:.4f}")
if p_val < 0.05:
    print("Result: Significant difference found between models.")
else:
    print("Result: No significant difference found between models.")

# 1. Run the Tukey HSD test
# (Assuming df_long is the 'melted' dataframe created in previous steps)
tukey = pairwise_tukeyhsd(endog=df_long['Value'],
                          groups=df_long['Model'],
                          alpha=0.05)

# 2. Convert the summary table to a DataFrame
# We skip the first row of 'data' because it contains the headers
tukey_df = pd.DataFrame(data=tukey.summary().data[1:],
                        columns=tukey.summary().data[0])

# 3. Export to CSV
tukey_df.to_csv('tukey_hsd_results.csv', index=False)

print("Tukey results exported to 'tukey_hsd_results.csv'.")
print("\n--- Summary Table ---")
print(tukey_df)

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm

# 1. Re-run ANOVA to get the internal stats (using your long-format df)
model_fit = ols('Value ~ C(Model) + C(Date)', data=df_long).fit()
anova_table = sm.stats.anova_lm(model_fit, typ=2)

# Get key values for LSD calculation
mse = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']
df_residual = anova_table.loc['Residual', 'df']
n_per_group = df_long.groupby('Model')['Value'].count().iloc[0]  # Assuming balanced (n=3)

# 2. Calculate the LSD Threshold
# LSD = t_crit * sqrt(2 * MSE / n)
t_crit = stats.t.ppf(1 - 0.05 / 2, df_residual)
lsd_threshold = t_crit * np.sqrt(2 * mse / n_per_group)

# 3. Pairwise Comparisons
models = df_long['Model'].unique()
results = []

for i in range(len(models)):
    for j in range(i + 1, len(models)):
        m1, m2 = models[i], models[j]
        mean1 = df_long[df_long['Model'] == m1]['Value'].mean()
        mean2 = df_long[df_long['Model'] == m2]['Value'].mean()
        diff = abs(mean1 - mean2)

        # Calculate specific p-value for this pair using the pooled MSE
        t_stat = diff / np.sqrt(2 * mse / n_per_group)
        p_val = 2 * (1 - stats.t.cdf(t_stat, df_residual))

        results.append({
            'Group 1': m1,
            'Group 2': m2,
            'Mean Diff': mean1 - mean2,
            'Abs Diff': diff,
            'p-value': p_val,
            'Significant': diff > lsd_threshold
        })

# 4. Display and Export
fisher_df = pd.DataFrame(results)
fisher_df.to_csv('fisher_lsd_results_manual.csv', index=False)

print(f"LSD Threshold: {lsd_threshold:.4f}")
print(fisher_df[['Group 1', 'Group 2', 'p-value', 'Significant']])