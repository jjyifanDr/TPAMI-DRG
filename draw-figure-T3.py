import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 150

RESULTS_DIR = './results_theorem3'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# 1.   theorem3_summary.csv
# ============================================================
df = pd.read_csv(os.path.join(RESULTS_DIR, 'theorem3_summary.csv'))
if 'delta' in df.columns and 'delta_mean' not in df.columns:
    df = df.rename(columns={'delta': 'delta_mean'})

#   Δ < 0  
df_neg = df[df['delta_mean'] < 0].copy()

# remove IIoT
DATASETS_ORDER = ['AD', 'MI', 'CreditCard']
df_neg = df_neg[df_neg['dataset'].isin(DATASETS_ORDER)]

COLORS = {
    'AD':         '#1f77b4',
    'MI':         '#9467bd',
    'CreditCard': '#d62728',
}
MARKERS = {
    'AD': 'o',
    'MI': 'D',
    'CreditCard': 's',
}

# ============================================================
# 2.  
# ============================================================
c2_emp = df_neg['gdi'].max()
print(f"Empirical C2^emp = {c2_emp:.4f}")

# ============================================================
# 3.  
# ============================================================
N_REPEATS_FIXED = 10   # FIX:  

lowest = {}
for ds in DATASETS_ORDER:
    sub = df_neg[df_neg['dataset'] == ds].sort_values('gdi')
    if len(sub) == 0:
        continue
    row = sub.iloc[0]
    lowest[ds] = {
        'auc_mean': row['auc_mean'],
        'auc_std': row['auc_std'],
        'n_repeats': N_REPEATS_FIXED,   # FIX
        'gdi': row['gdi'],
        'delta': row['delta_mean'],
    }

#  
for ds, v in lowest.items():
    n = v['n_repeats']
    se = v['auc_std'] / np.sqrt(n)
    if se > 0:
        t_stat = (v['auc_mean'] - 0.5) / se
        p_val = 2 * (1 - t_dist.cdf(abs(t_stat), df=n - 1))
    else:
        t_stat = 0.0
        p_val = 1.0
    v['t_stat'] = t_stat
    v['p_val'] = p_val
    v['ci_low'] = v['auc_mean'] - 1.96 * se
    v['ci_high'] = v['auc_mean'] + 1.96 * se

print("\n=== Lowest-GDI configuration per dataset ===")
for ds, v in lowest.items():
    print(f"  {ds}: G={v['gdi']:.4f}, Δ={v['delta']:+.4f}, "
          f"AUC={v['auc_mean']:.4f}±{v['auc_std']:.4f}, "
          f"95% CI=[{v['ci_low']:.4f}, {v['ci_high']:.4f}], "
          f"t={v['t_stat']:.4f}, p={v['p_val']:.4f}")

# ============================================================
# 4.  
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

# (a) AUC vs G
ax = axes[0]
for ds in DATASETS_ORDER:
    sub = df_neg[df_neg['dataset'] == ds].sort_values('gdi')
    if len(sub) == 0:
        continue
    ax.errorbar(sub['gdi'], sub['auc_mean'],
                yerr=sub['auc_std'] / np.sqrt(N_REPEATS_FIXED),
                marker=MARKERS[ds], color=COLORS[ds],
                markersize=8, capsize=3, lw=1.2, label=ds)

ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2.5, alpha=0.9)


ax.set_xscale('log')
ax.set_xlabel('$G$', fontsize=12)
ax.set_ylabel('AUC', fontsize=12)
ax.set_title(r'(a) AUC vs. $GDI$ ($\Delta < 0$)', fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12, loc='best')

# (b)  
ax = axes[1]
datasets_present = [ds for ds in DATASETS_ORDER if ds in lowest]
means    = [lowest[ds]['auc_mean'] for ds in datasets_present]
ci_lows  = [lowest[ds]['ci_low']  for ds in datasets_present]
ci_highs = [lowest[ds]['ci_high'] for ds in datasets_present]
p_vals   = [lowest[ds]['p_val']   for ds in datasets_present]
gdi_vals = [lowest[ds]['gdi']     for ds in datasets_present]

x = np.arange(len(datasets_present))
bar_colors = [COLORS[ds] for ds in datasets_present]

ax.bar(x, means, color=bar_colors, edgecolor='black',
       linewidth=0.8, width=0.5)
ax.errorbar(x, means,
            yerr=[np.array(means) - np.array(ci_lows),
                  np.array(ci_highs) - np.array(means)],
            fmt='none', ecolor='black', capsize=5, linewidth=1.2)

for i, (m, p, g) in enumerate(zip(means, p_vals, gdi_vals)):
    ax.text(i, m + 0.06, f'$*p$={p:.3f}\n$g$={g:.2f}',
            ha='center', va='bottom', fontsize=11)

ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2.5, alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(datasets_present)
ax.set_ylim(0.3, 0.7)
ax.set_ylabel('AUC', fontsize=12)
ax.set_title(r'(b) Lowest-$GDI$ configuration per dataset ($\Delta < 0$)', fontsize=13)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, 'T3_validation_1x2.tiff')
plt.savefig(out_path, dpi=300, format='tiff', bbox_inches='tight')
plt.show()
print(f"Saved: {out_path}")

# ============================================================
# 5.  
# ============================================================
print("\n=== Theorem 3 validation summary ===")
for ds in datasets_present:
    v = lowest[ds]
    print(f"  {ds}: GDI={v['gdi']:.4f}, Δ={v['delta']:+.4f}, "
          f"AUC={v['auc_mean']:.4f}, "
          f"95% CI=[{v['ci_low']:.4f}, {v['ci_high']:.4f}], "
          f"t={v['t_stat']:.4f}, p={v['p_val']:.4f}")