import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import stats
from matplotlib.lines import Line2D

# ============================================================
# Times New Roman，
# ============================================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13

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
t1a = data[data['experiment'] == 'T1a'].copy()
t1a = t1a.dropna(subset=['n', 'gdi', 'auc', 'delta'])# (subset=['n', 'auc', 'delta'])

# 
rank = pd.read_csv('./results_theorem1/Rank_SUMMARY.csv')

# ============================================================
# 2. 
# ============================================================
anomaly_rate = {
    'AD': 0.019908,
    'CreditCard': 0.004266,
    'IIoT': 0.436759,
    'MI': 0.10,
}

# ============================================================
# 3. 
# ============================================================
colors = {
    'AD':         '#1f77b4',   # 蓝
    'CreditCard': '#d62728',   # 红
    'IIoT':       '#ff7f0e',   # 橙
    'MI':         '#9467bd',   # 紫
}
markers = {
    'AD': 'o',
    'CreditCard': 's',
    'IIoT': '^',
    'MI': 'D',
}

color_pos = '#1f77b4'  # Δ > 0
color_neg = '#d62728'  # Δ < 0

# ============================================================
# 4. Degradation judgment
# ============================================================
def is_degenerate(row, alpha=0.05):
    auc = row['auc']
    std = row['auc_std'] if 'auc_std' in row and not np.isnan(row['auc_std']) else np.nan
    n_rep = row['n_repeats'] if 'n_repeats' in row and not np.isnan(row['n_repeats']) else 1
    if np.isnan(std) or std == 0 or n_rep <= 1:
        return abs(auc - 0.5) < 0.05
    t_stat = (auc - 0.5) / (std / np.sqrt(n_rep))
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_rep - 1))
    return p_val > alpha

t1a['degenerate'] = t1a.apply(is_degenerate, axis=1)
t1a['delta_sign'] = np.where(t1a['delta'] > 0, 'Δ > 0', 'Δ < 0')

# ============================================================
# 5. Public legend element
# ============================================================
legend_delta = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color_pos,
           markeredgecolor='black', markersize=8, label='Δ > 0'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color_neg,
           markeredgecolor='black', markersize=8, label='Δ < 0'),
]
legend_dataset = [
    Line2D([0], [0], marker=markers[ds], color='w',
           markerfacecolor='gray', markeredgecolor='black',
           markersize=8, label=ds)
    for ds in files
]
legend_combined = legend_delta + legend_dataset

# ============================================================
# 6.  
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# ------------------------------------------------------------
# (a) AUC vs G
# ------------------------------------------------------------
ax = axes[0]
for ds in files:
    sub = t1a[t1a['dataset'] == ds]          
    ax.plot(sub['gdi'], sub['auc'], marker=markers[ds],
            color=colors[ds], lw=1.5, ms=8, label=ds)

ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2.5, alpha=0.9)
ax.set_xscale('log')
ax.set_xlabel('GDI', fontsize=16)
ax.set_ylabel('AUC', fontsize=16)
ax.set_title('(a) AUC vs. GDI', fontsize=16)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=13, loc='best')

# ------------------------------------------------------------
# (b) AUC vs Δ
# ------------------------------------------------------------
ax = axes[1]
for ds in files:
    sub = t1a[t1a['dataset'] == ds].sort_values('delta')
    ax.plot(sub['delta'], sub['auc'], marker=markers[ds],
            color=colors[ds], lw=1.5, ms=8, label=ds)

ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2.5, alpha=0.9)
ax.axvline(x=0, color='black', linestyle=':', linewidth=2.5, alpha=0.9)
ax.set_xlabel(r'$\Delta$', fontsize=16)
ax.set_ylabel('AUC', fontsize=16)
ax.set_title(r'(b) AUC vs. $\Delta$', fontsize=16)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=13, loc='best')

# ------------------------------------------------------------
# (c) Δ  vs Significant degradation probability
# ------------------------------------------------------------
ax = axes[2]
groups = ['Δ > 0', 'Δ < 0']
deg_probs = []
for g in groups:
    sub = t1a[t1a['delta_sign'] == g]
    deg_probs.append(sub['degenerate'].mean() if len(sub) > 0 else np.nan)

bars = ax.bar(groups, deg_probs, color=[color_pos, color_neg],
              edgecolor='black', linewidth=0.8, width=0.5)
for bar, p in zip(bars, deg_probs):
    if not np.isnan(p):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{p:.2f}', ha='center', va='bottom', fontsize=16)

ax.set_ylim(0, 1.1)
ax.set_ylabel('Degeneration fraction', fontsize=16)
ax.set_title('(c) Fraction of Configurations Degenerating to AUC = 0.5', fontsize=16)
ax.grid(True, alpha=0.3, axis='y')

# ------------------------------------------------------------
# (d) AUC vs anomaly rate，
# ------------------------------------------------------------
ax = axes[3]
for ds in files:
    sub = t1a[t1a['dataset'] == ds]
    rate = anomaly_rate[ds]
    for _, row in sub.iterrows():
        c = color_pos if row['delta'] > 0 else color_neg
        ax.scatter(rate, row['auc'], color=c, marker=markers[ds],
                   s=100, edgecolors='black', linewidths=1)

ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2.5, alpha=0.9)
ax.set_xscale('log')
ax.set_xlabel('Anomaly rate', fontsize=16)
ax.set_ylabel('AUC', fontsize=16)
ax.set_title('(d) AUC vs. Anomaly Rate (color: Δ sign)', fontsize=16)
ax.grid(True, alpha=0.3)
ax.legend(handles=legend_combined, fontsize=13, loc='best', ncol=2)

# ------------------------------------------------------------
# (e) Δ vs anomaly rate
# ------------------------------------------------------------
ax = axes[4]
for ds in files:
    sub = t1a[t1a['dataset'] == ds]
    rate = anomaly_rate[ds]
    ax.scatter([rate]*len(sub), sub['delta'], marker=markers[ds],
               color=colors[ds], s=100, edgecolors='black', linewidths=1,
               label=ds)

ax.axhline(y=0, color='black', linestyle='--', linewidth=2.5, alpha=0.9)
ax.set_xscale('log')
ax.set_xlabel('Anomaly rate', fontsize=16)
ax.set_ylabel(r'$\Delta$', fontsize=16)
ax.set_title(r'(e) $\Delta$ vs. Anomaly Rate', fontsize=16)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=13, loc='best')

# ------------------------------------------------------------
# (f) sentivity：S_Δ vs S_anomaly vs S_n
# ------------------------------------------------------------
ax = axes[5]

def sensitivity(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0:
        return np.nan
    return abs(np.polyfit(x, y, 1)[0]) * (np.max(x) - np.min(x)) / (np.max(y) - np.min(y) + 1e-9)

datasets_order = ['AD', 'MI', 'IIoT', 'CreditCard']
S_delta, S_rate, S_n = [], [], []

for ds in datasets_order:
    sub = t1a[t1a['dataset'] == ds]
    if len(sub) < 2:
        S_delta.append(np.nan); S_rate.append(np.nan); S_n.append(np.nan)
        continue
    S_delta.append(sensitivity(sub['delta'], sub['auc']))
    S_rate.append(sensitivity([anomaly_rate[ds]]*len(sub), sub['auc']))
    S_n.append(sensitivity(sub['n'], sub['auc']))

x = np.arange(len(datasets_order))
width = 0.25

ax.bar(x - width, S_delta, width, label=r'$S_\Delta$',
       color='#1f77b4', edgecolor='black', linewidth=0.6)
ax.bar(x, S_rate, width, label=r'$S_{\mathrm{anomaly}}$',
       color='#d62728', edgecolor='black', linewidth=0.6)
ax.bar(x + width, S_n, width, label=r'$S_n$',
       color='#9467bd', edgecolor='black', linewidth=0.6)

ax.set_xticks(x)
ax.set_xticklabels(datasets_order)
ax.set_ylabel('Sensitivity', fontsize=16)
ax.set_title('(f) Sensitivity comparison', fontsize=16)
ax.legend(fontsize=13, loc='best')
ax.grid(True, alpha=0.3, axis='y')

# ============================================================
# 7.  
# ============================================================
plt.tight_layout()
plt.savefig('T1a_main_2x3.tiff', dpi=300, format='tiff', bbox_inches='tight')
plt.show()
print("saved as: T1a_main_2x3.tiff")


# ============================================================
# 8. rank（GDI vs Δ）
# ============================================================
fig2, ax2 = plt.subplots(figsize=(8, 5))

datasets_order = ['AD', 'MI', 'IIoT', 'CreditCard']
rho_gdi, rho_delta, p_gdi, p_delta = [], [], [], []

for ds in datasets_order:
    row_g = rank[(rank['dataset'] == ds) & (rank['metric'] == 'GDI')]
    row_d = rank[(rank['dataset'] == ds) & (rank['metric'] == 'Delta')]
    rho_gdi.append(row_g['spearman_rho'].values[0] if not row_g.empty else np.nan)
    rho_delta.append(row_d['spearman_rho'].values[0] if not row_d.empty else np.nan)
    p_gdi.append(row_g['p_value'].values[0] if not row_g.empty else np.nan)
    p_delta.append(row_d['p_value'].values[0] if not row_d.empty else np.nan)

x = np.arange(len(datasets_order))
width = 0.35

bars1 = ax2.bar(x - width/2, rho_gdi, width, label='GDI',
                color='#1f77b4', edgecolor='black', linewidth=0.6)
bars2 = ax2.bar(x + width/2, rho_delta, width, label=r'$\Delta$',
                color='#d62728', edgecolor='black', linewidth=0.6)

for bar, rho in zip(bars1, rho_gdi):
    if not np.isnan(rho):
        offset = 0.02 if rho >= 0 else -0.06
        va = 'bottom' if rho >= 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                 f'{rho:.3f}', ha='center', va=va, fontsize=13, color='black')

for bar, rho, p in zip(bars2, rho_delta, p_delta):
    if np.isnan(rho):
        continue
    mid_y = bar.get_height() / 2
    label = f'{rho:.3f}'
    if not np.isnan(p) and p < 0.05:
        label += ' *'
    ax2.text(bar.get_x() + bar.get_width()/2, mid_y, label,
             ha='center', va='center', fontsize=11,
             color='black', fontweight='bold',
             path_effects=[pe.withStroke(linewidth=2, foreground='white')])

ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(datasets_order)
ax2.set_ylabel('Spearman ρ', fontsize=16)
ax2.set_title('Rank correlation: GDI vs Δ', fontsize=16)
ax2.legend(fontsize=13, loc='best')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('T1a_rank_corr.tiff', dpi=300, format='tiff', bbox_inches='tight')
plt.show()
print("saved as: T1a_rank_corr.tiff")