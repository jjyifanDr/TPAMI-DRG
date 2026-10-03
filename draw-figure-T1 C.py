import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

# ============================================================
#  
# ============================================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['axes.titlesize'] = 16

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
t1c = data[data['experiment'] == 'T1c'].copy()
t1c = t1c.dropna(subset=['noise_factor', 'auc', 'delta'])

datasets_order = ['AD', 'MI', 'IIoT', 'CreditCard']

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

legend_ds = [Line2D([0], [0], marker=markers_ds[ds], color='w',
                    markerfacecolor='gray', markeredgecolor='black',
                    markersize=9, label=ds) for ds in datasets_order]

# ============================================================
# 3.  
# ============================================================
fig = plt.figure(figsize=(24, 11))

 
gs = GridSpec(2, 6, figure=fig,
              height_ratios=[1.0, 1.0],
              hspace=0.40, wspace=0.75)

ax_a1 = fig.add_subplot(gs[0, 0:2])   #  
ax_a2 = fig.add_subplot(gs[0, 3:5])   #  
ax_b1 = fig.add_subplot(gs[1, 0:2])   #  
ax_b2 = fig.add_subplot(gs[1, 2:4])   #  
ax_b3 = fig.add_subplot(gs[1, 4:6])   #  

# ------------------------------------------------------------
#   Δ vs noise_factor
# ------------------------------------------------------------
ax = ax_a1
for ds in datasets_order:
    sub = t1c[t1c['dataset'] == ds].sort_values('noise_factor')
    if len(sub) == 0:
        continue
    ax.plot(sub['noise_factor'], sub['delta'],
            marker=markers_ds[ds], color=colors_ds[ds],
            lw=1.4, ms=8, label=ds)

ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
ax.set_xscale('log', base=2)
ax.set_xlabel(r'Noise factor $\log_2$', fontsize=16)
ax.set_ylabel(r'$\Delta$', fontsize=16)
ax.set_title(r'(a) $\Delta$ vs. noise factor', fontsize=16)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=14, loc='best')

# ------------------------------------------------------------
#  (b) AUC vs noise_factor
# ------------------------------------------------------------
ax = ax_a2
for ds in datasets_order:
    sub = t1c[t1c['dataset'] == ds].sort_values('noise_factor')
    if len(sub) == 0:
        continue
    ax.plot(sub['noise_factor'], sub['auc'],
            marker=markers_ds[ds], color=colors_ds[ds],
            lw=1.4, ms=8, label=ds)

ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
ax.set_xscale('log', base=2)
ax.set_xlabel(r'Noise factor $\log_2$', fontsize=16)
ax.set_ylabel('AUC', fontsize=16)
ax.set_title('(b) AUC vs. noise factor', fontsize=16)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=14, loc='best')

# ------------------------------------------------------------
#   (a) Δ–AUC  
# ------------------------------------------------------------
ax = ax_b1
all_noise = t1c['noise_factor'].values
norm = plt.Normalize(vmin=np.log2(all_noise.min()), vmax=np.log2(all_noise.max()))
cmap = plt.cm.coolwarm_r

for ds in datasets_order:
    sub = t1c[t1c['dataset'] == ds].sort_values('noise_factor')
    if len(sub) == 0:
        continue
    ax.plot(sub['delta'], sub['auc'],
            color=colors_ds[ds], lw=1.4, alpha=0.7,
            marker=markers_ds[ds], ms=8, markeredgecolor='black',
            markeredgewidth=0.5, label=ds)
    for _, row in sub.iterrows():
        c = cmap(norm(np.log2(row['noise_factor'])))
        ax.scatter(row['delta'], row['auc'],
                   color=c, s=90, marker=markers_ds[ds],
                   edgecolors='black', linewidths=0.5, zorder=3)

ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
ax.axvline(x=0, color='black', linestyle=':', linewidth=1.2, alpha=0.6)
ax.set_xlabel(r'$\Delta$', fontsize=16)
ax.set_ylabel('AUC', fontsize=16)
ax.set_title(r'(c) $\Delta$–AUC trajectory', fontsize=16)
ax.grid(True, alpha=0.3)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r'$\log_2$ (noise factor)', fontsize=16)
cbar.ax.tick_params(labelsize=16)

ax.legend(handles=legend_ds, fontsize=14, loc='best')

# ------------------------------------------------------------
#   (b)  
# ------------------------------------------------------------
ax = ax_b2
for ds in datasets_order:
    sub = t1c[t1c['dataset'] == ds].sort_values('noise_factor')
    if len(sub) == 0:
        continue
    ax.plot(sub['noise_factor'], sub['auc'],
            marker=markers_ds[ds], color=colors_ds[ds],
            lw=1.4, ms=8, label=ds)

ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
ax.set_xscale('log', base=2)
ax.set_xlabel(r'Noise factor $\log_2$', fontsize=16)
ax.set_ylabel('AUC (separability proxy)', fontsize=16)
ax.set_title('(d) Separability vs. noise', fontsize=16)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=14, loc='best')

# ------------------------------------------------------------
#  (c)  
# ------------------------------------------------------------
ax = ax_b3

def partial_corr(x, y, z):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if len(x) < 4:
        return np.nan, np.nan
    z_design = np.vstack([np.ones_like(z), z]).T
    beta_x, *_ = np.linalg.lstsq(z_design, x, rcond=None)
    x_res = x - z_design @ beta_x
    beta_y, *_ = np.linalg.lstsq(z_design, y, rcond=None)
    y_res = y - z_design @ beta_y
    if np.std(x_res) == 0 or np.std(y_res) == 0:
        return np.nan, np.nan
    r, p = stats.pearsonr(x_res, y_res)
    return r, p

coef_noise_delta = []
coef_noise_auc = []
coef_delta_auc = []

for ds in datasets_order:
    sub = t1c[t1c['dataset'] == ds].dropna(subset=['noise_factor', 'auc', 'delta'])
    if len(sub) < 4:
        coef_noise_delta.append(np.nan)
        coef_noise_auc.append(np.nan)
        coef_delta_auc.append(np.nan)
        continue
    log_noise = np.log2(sub['noise_factor'])
    r_nd, _ = stats.pearsonr(log_noise, sub['delta'])
    r_na, _ = partial_corr(log_noise, sub['auc'], sub['delta'])
    r_da, _ = partial_corr(sub['delta'], sub['auc'], log_noise)
    coef_noise_delta.append(r_nd)
    coef_noise_auc.append(r_na)
    coef_delta_auc.append(r_da)

x_pos = np.arange(len(datasets_order))
width = 0.25

ax.bar(x_pos - width, coef_noise_delta, width,
       label=r'noise $\to \Delta$',
       color='#1f77b4', edgecolor='black', linewidth=0.6)
ax.bar(x_pos, coef_noise_auc, width,
       label=r'noise $\to$ AUC $|$ $\Delta$',
       color='#d62728', edgecolor='black', linewidth=0.6)
ax.bar(x_pos + width, coef_delta_auc, width,
       label=r'$\Delta \to$ AUC $|$ noise',
       color='#9467bd', edgecolor='black', linewidth=0.6)

ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(datasets_order)
ax.set_ylabel('Partial correlation', fontsize=16)
ax.set_title('(e) Path coefficients: noise, Δ, AUC', fontsize=16)
ax.legend(fontsize=16, loc='best')
ax.grid(True, alpha=0.3, axis='y')

# ============================================================
# 4. 
# ============================================================
plt.savefig('T1c_main_2x3.tiff', dpi=300, format='tiff', bbox_inches='tight')
plt.show()
print("saved as: T1c_main_2x3.tiff")