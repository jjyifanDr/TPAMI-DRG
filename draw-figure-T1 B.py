import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D

# ============================================================
#  
# ============================================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11

# ============================================================
# 1.  
# ============================================================
files = {
    'AD': './results_theorem1/AD_SUMMARY.CSV',
    'CreditCard': './results_theorem1/CreditCard_SUMMARY.CSV',
    'IIoT': './results_theorem1/IIoT_SUMMARY.CSV',
    'MI': './results_theorem1/MI_SUMMARY.CSV',
}

frames = []
for name, path in files.items():
    df = pd.read_csv(path)
    df['dataset'] = name
    frames.append(df)
data = pd.concat(frames, ignore_index=True)

#  
t1b = data[data['experiment'] == 'T1b'].copy()
t1b = t1b.dropna(subset=['p', 'auc', 'delta'])

# ============================================================
# 2.  
# ============================================================
colors_ds = {
    'AD':         '#1f77b4',
    'CreditCard': '#d62728',
    'IIoT':       '#ff7f0e',
    'MI':         '#9467bd',
}
markers_ds = {
    'AD': 'o',
    'CreditCard': 's',
    'IIoT': '^',
    'MI': 'D',
}
colors_method = {
    'pca':    '#1f77b4',
    'goe':    '#d62728',
    'random': '#9467bd',
}
markers_method = {
    'pca': 'o',
    'goe': 's',
    'random': '^',
}
method_labels = {
    'pca': 'PCA',
    'goe': 'GOE',
    'random': 'Random',
}

datasets_order = ['AD', 'MI', 'IIoT', 'CreditCard']
methods_order = ['pca', 'goe', 'random']

# ============================================================
# 3.  
# ============================================================

def is_degenerate(row, alpha=0.05):
    """
    Determine whether a configuration has degraded to AUC=0.5.
    Judgment criteria: Whether the 95% confidence interval of AUC includes 0.5.
    -If CI contains 0.5, it is judged as degradation;
    -If CI does not include 0.5, it is judged as non degenerate.
    """
    auc = row['auc']
    std = row['auc_std'] if 'auc_std' in row and not np.isnan(row['auc_std']) else np.nan
    n_rep = row['n_repeats'] if 'n_repeats' in row and not np.isnan(row['n_repeats']) else 1

    # Scenario 1: Missing std or n_rep, unable to calculate CI
    if np.isnan(std) or n_rep <= 1:
        return abs(auc - 0.5) < 0.05

    # Scenario 2: std==0, AUC fully determined
    if std == 0:
        return abs(auc - 0.5) < 1e-6

    # Scenario 3: std>0, calculate 95% CI using t-distribution
    # CI = auc ± t_{alpha/2, n_rep-1} * std / sqrt(n_rep)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n_rep - 1)
    half_width = t_crit * std / np.sqrt(n_rep)
    ci_low  = auc - half_width
    ci_high = auc + half_width

    # If CI contains 0.5, it is judged as degradation
    return ci_low <= 0.5 <= ci_high

t1b['degenerate'] = t1b.apply(is_degenerate, axis=1)
t1b['delta_sign'] = np.where(t1b['delta'] > 0, 'Δ > 0', 'Δ < 0')

# ============================================================
# 4. Wilson 95% confidence interval
# ============================================================
def wilson_ci(k, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))

# ============================================================
# 5. 
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes = axes.flatten()

#  
legend_ds = [Line2D([0], [0], marker=markers_ds[ds], color='w',
                    markerfacecolor='gray', markeredgecolor='black',
                    markersize=8, label=ds) for ds in datasets_order]
legend_m = [Line2D([0], [0], marker=markers_method[m], color='w',
                   markerfacecolor=colors_method[m], markeredgecolor='black',
                   markersize=8, label=method_labels[m]) for m in methods_order]

# ------------------------------------------------------------
# (a) AUC vs p， 
# ------------------------------------------------------------
ax = axes[0]
for ds in datasets_order:
    sub_ds = t1b[t1b['dataset'] == ds]
    for m in methods_order:
        sub = sub_ds[sub_ds['method'] == m]
        if len(sub) == 0:
            continue
        for _, row in sub.iterrows():
            ax.scatter(row['p'], row['auc'],
                       marker=markers_method[m], color=colors_method[m],
                       s=80, edgecolors='black', linewidths=0.5)

ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2.5, alpha=0.8)
ax.set_xlabel('Projected dimension  $p$', fontsize=16)
ax.set_ylabel('AUC', fontsize=16)
ax.set_title('(a) AUC vs. $p$', fontsize=16)
ax.grid(True, alpha=0.3)

#  
ax.legend(handles=legend_ds + legend_m, fontsize=10, loc='best', ncol=2)

# ------------------------------------------------------------
# (b) AUC vs Δ， 
# ------------------------------------------------------------
ax = axes[1]
for ds in datasets_order:
    sub_ds = t1b[t1b['dataset'] == ds]
    for m in methods_order:
        sub = sub_ds[sub_ds['method'] == m]
        if len(sub) == 0:
            continue
        for _, row in sub.iterrows():
            c = '#1f77b4' if row['delta'] > 0 else '#d62728'
            ax.scatter(row['delta'], row['auc'],
                       marker=markers_method[m], color=c, s=80,
                       edgecolors='black', linewidths=0.5)

ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2.5, alpha=0.9)
ax.axvline(x=0, color='black', linestyle=':', linewidth=2.5, alpha=0.9)
ax.set_xlabel(r'$\Delta$', fontsize=16)
ax.set_ylabel('AUC', fontsize=16)
ax.set_title(r'(b) AUC vs. $\Delta$', fontsize=16)
ax.grid(True, alpha=0.3)

#  
ax.legend(handles=legend_ds + legend_m, fontsize=10, loc='best', ncol=2)

# ------------------------------------------------------------
# (c) Degradation probability vs Δ sign, with Wilson 95% CI
# ------------------------------------------------------------
ax = axes[2]

groups = ['Δ > 0', 'Δ < 0']
probs = []
cis = []
ns = []

for g in groups:
    sub = t1b[t1b['delta_sign'] == g]
    n = len(sub)
    k = int(sub['degenerate'].sum())
    p_hat = k / n if n > 0 else np.nan
    lo, hi = wilson_ci(k, n)
    probs.append(p_hat)
    cis.append((lo, hi))
    ns.append(n)

xpos = np.arange(len(groups))
bars = ax.bar(xpos, probs,
              color=['#1f77b4', '#d62728'],
              edgecolor='black', linewidth=0.8, width=0.5)

# 加 Wilson CI 误差棒
for i, (p_hat, (lo, hi), n) in enumerate(zip(probs, cis, ns)):
    if not np.isnan(p_hat):
        ax.errorbar(i, p_hat,
                    yerr=[[p_hat - lo], [hi - p_hat]],
                    fmt='none', ecolor='black',
                    elinewidth=1.5, capsize=6, capthick=1.5)
        # 柱顶标 p_hat 和 n
        ax.text(i, hi + 0.03,
                f'{p_hat:.2f} (n={n})',
                ha='center', va='bottom', fontsize=16)

ax.set_xticks(xpos)
ax.set_xticklabels(groups)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Probability ', fontsize=16)
ax.set_title('(c) Degradation Probability to Random Guessing ', fontsize=16)
ax.grid(True, alpha=0.3, axis='y')

# 图上标明 Wilson CI
ax.text(0.98, 0.02, 'Error bars: Wilson 95% CI',
        transform=ax.transAxes, fontsize=16,
        ha='right', va='bottom', color='black',
        bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.9))

# ============================================================
# 6.  
# ============================================================
plt.tight_layout()
plt.savefig('T1b_main_1x3.tiff', dpi=300, format='tiff', bbox_inches='tight')
plt.show()
print("saved as: T1b_main_1x3.tiff")