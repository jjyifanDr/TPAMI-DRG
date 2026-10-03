"""
Theorem 3 Validation
read from SUMMARY.CSV to validation Theorem 3 
"""

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

# ============================================================
# 1. SUMMARY.CSV
# ============================================================
CSV_PATH = './results_theorem1./SUMMARY.CSV'   # 
df = pd.read_csv(CSV_PATH)

# Only retain T1a (Theorem 3 validation based on T1a subsamping experiments)
df_t1a = df[df['experiment'] == 'T1a'].copy()

# ============================================================
# 2. remove IIoT
# ============================================================
EXCLUDED = ['IIoT']
df_t1a = df_t1a[~df_t1a['dataset'].isin(EXCLUDED)]

# ============================================================
# 3. For each dataset, find the configuration with Δ<0 that has the lowest G value
# ============================================================
N_REPEATS = 10   # SUMMARY.CSV 

rows = []
for ds in df_t1a['dataset'].unique():
    sub = df_t1a[(df_t1a['dataset'] == ds) & (df_t1a['delta'] < 0)]
    if len(sub) == 0:
        print(f"  {ds}: no Δ<0 configuration, skipped.")
        continue

    idx_min_g = sub['gdi'].idxmin()
    row = sub.loc[idx_min_g]

    auc_mean = row['auc']
    auc_std  = row['auc_std']
    n = row['n']
    gdi = row['gdi']
    delta = row['delta']

    # 95% CI
    se = auc_std / np.sqrt(N_REPEATS)
    ci_low  = auc_mean - 1.96 * se
    ci_high = auc_mean + 1.96 * se

    # one-sample t-test vs 0.5 
    if auc_std > 0:
        t_stat = (auc_mean - 0.5) / se
        p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df=N_REPEATS - 1))
    else:
        t_stat = 0.0
        p_value = 1.0

    rows.append({
        'dataset': ds,
        'n': n,
        'gdi': gdi,
        'delta': delta,
        'auc_mean': auc_mean,
        'auc_std': auc_std,
        'ci_low': ci_low,
        'ci_high': ci_high,
        't_stat': t_stat,
        'p_value': p_value,
    })

    print(f"  {ds}: n={n}, G={gdi:.4f}, Δ={delta:+.4f}, "
          f"AUC={auc_mean:.4f}, 95% CI=[{ci_low:.4f}, {ci_high:.4f}], "
          f"t={t_stat:.4f}, p={p_value:.4f}")

# ============================================================
# 4. 
# ============================================================
df_result = pd.DataFrame(rows)
df_result.to_csv('./results_theorem3/theorem3_summary.csv', index=False)
print("\nSaved: ./theorem3_summary.csv")
print(df_result.to_string(index=False))