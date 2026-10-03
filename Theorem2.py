import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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
# 1. READ T1a–T1c
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

#  T1a、T1b、T1c
real = data[data['experiment'].isin(['T1a', 'T1b', 'T1c'])].copy()
real = real.dropna(subset=['auc', 'delta'])

# Only retain configurations with Δ>0 (Theorem 2 only has information when Δ>0)
real_pos = real[real['delta'] > 0].copy()

print(f"Total real configurations: {len(real)}")
print(f"Configurations with Δ > 0: {len(real_pos)}")

# ============================================================
# 2. WD / Poisson PDF / CDF
# ============================================================
def wd_pdf(s):
    return (np.pi * s / 2.0) * np.exp(-np.pi * s**2 / 4.0)

def wd_cdf(s):
    return 1.0 - np.exp(-np.pi * s**2 / 4.0)

def poisson_pdf(s):
    return np.exp(-s)

def poisson_cdf(s):
    return 1.0 - np.exp(-s)

# ============================================================
# 3. DISTRIBUTION: normal = θ WD + (1-θ) P, anomaly = (1-θ) WD + θ P
# ============================================================
def f0_pdf(s, theta):
    return theta * wd_pdf(s) + (1.0 - theta) * poisson_pdf(s)

def f0_cdf(s, theta):
    return theta * wd_cdf(s) + (1.0 - theta) * poisson_cdf(s)

def f1_pdf(s, theta):
    return (1.0 - theta) * wd_pdf(s) + theta * poisson_pdf(s)

def f1_cdf(s, theta):
    return (1.0 - theta) * wd_cdf(s) + theta * poisson_cdf(s)

# ============================================================
# 4. SAMPLING
# ============================================================
def sample_wd(n, rng):
    u = rng.uniform(0, 1, n)
    return np.sqrt(-4.0 / np.pi * np.log(1.0 - u))

def sample_poisson(n, rng):
    return rng.exponential(1.0, n)

def sample_f0(n, theta, rng):
    mask = rng.uniform(0, 1, n) < theta
    out = np.empty(n)
    n_wd = int(np.sum(mask))
    n_p = n - n_wd
    if n_wd > 0:
        out[mask] = sample_wd(n_wd, rng)
    if n_p > 0:
        out[~mask] = sample_poisson(n_p, rng)
    return out

def sample_f1(n, theta, rng):
    mask = rng.uniform(0, 1, n) < theta
    out = np.empty(n)
    n_p = int(np.sum(mask))
    n_wd = n - n_p
    if n_p > 0:
        out[mask] = sample_poisson(n_p, rng)
    if n_wd > 0:
        out[~mask] = sample_wd(n_wd, rng)
    return out

# ============================================================
# 5. LRT score
# ============================================================
def lrt_score(s, theta):
    s = np.clip(np.asarray(s, dtype=float), 1e-12, None)
    return np.log(f0_pdf(s, theta) + 1e-300) - np.log(f1_pdf(s, theta) + 1e-300)

# ============================================================
# 6. AUC via Mann-Whitney U
# ============================================================
def auc_mwu(scores_pos, scores_neg):
    if len(scores_pos) < 2 or len(scores_neg) < 2:
        return np.nan
    u, _ = mannwhitneyu(scores_pos, scores_neg, alternative='greater')
    return u / (len(scores_pos) * len(scores_neg))

# ============================================================
# 7. Δ(θ)
# ============================================================
def w1_between(cdf_a, cdf_b, s_max=30.0, n_grid=20000):
    grid = np.linspace(0.0, s_max, n_grid)
    F_a = cdf_a(grid)
    F_b = cdf_b(grid)
    return np.trapz(np.abs(F_a - F_b), grid)

def compute_delta(theta):
    cdf_normal = lambda s: f0_cdf(s, theta)
    w1_wd = w1_between(cdf_normal, wd_cdf)
    w1_p  = w1_between(cdf_normal, poisson_cdf)
    return w1_wd - w1_p

# ============================================================
# 8. SCANNING θ,  
# ============================================================
theta_list = np.linspace(0.0, 0.5, 21)
N_SAMPLES  = 5000          # TRAINING CLASSIFER
N_REPS     = 10

deltas       = []
auc_lrt      = []
auc_logreg   = []

for theta in theta_list:
    delta = compute_delta(theta)

    a_lrt_rep  = []
    a_lr_rep   = []

    for rep in range(N_REPS):
        rng = np.random.default_rng(1000 + rep)
        s_normal  = sample_f0(N_SAMPLES, theta, rng)
        s_anomaly = sample_f1(N_SAMPLES, theta, rng)

        # ---- LRT ----
        scores_n = lrt_score(s_normal, theta)
        scores_a = lrt_score(s_anomaly, theta)
        a_lrt_rep.append(auc_mwu(scores_n, scores_a))

        # ---- LogReg: ----
        X_all = np.concatenate([s_normal, s_anomaly])[:, None]
        y_all = np.concatenate([np.zeros(len(s_normal)),
                                np.ones(len(s_anomaly))])

        # LogReg
        clf_lr = LogisticRegression(max_iter=1000)
        clf_lr.fit(X_all, y_all)
        s_lr = clf_lr.predict_proba(X_all)[:, 1]
        a_lr_rep.append(roc_auc_score(y_all, s_lr))

    deltas.append(delta)
    auc_lrt.append(float(np.mean(a_lrt_rep)))
    auc_logreg.append(float(np.mean(a_lr_rep)))

    print(f"theta={theta:.3f}  Δ={delta:+.4f}  "
          f"AUC_LRT={auc_lrt[-1]:.4f}  "
          f"AUC_LR={auc_logreg[-1]:.4f}")

deltas     = np.array(deltas)
auc_lrt    = np.array(auc_lrt)
auc_logreg = np.array(auc_logreg)

# ============================================================
# 9. Synthetic data: WD vs Poisson LRT
# ============================================================
rng = np.random.default_rng(42)

n_wd = 5000
n_p = 5000
s_wd = sample_wd(n_wd, rng)
s_p = sample_poisson(n_p, rng)

# Theoretical AUC of LRT in WD vs Poisson
scores_wd = lrt_score(s_wd, theta=1.0)  #  WD as normal
scores_p = lrt_score(s_p, theta=1.0)    #  Poisson as anomaly
auc_theory = auc_mwu(scores_p, scores_wd)  # anomaly as positive class
print(f"Synthetic LRT AUC (WD vs Poisson): {auc_theory:.4f}")

# ============================================================
# 10. 画图：1 行 3 列
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ------------------------------------------------------------
# (a) real data：AUC vs Δ，Δ > 0
# ------------------------------------------------------------
ax = axes[0]

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

# --- plot Δ > 0 ---
for ds in ['AD', 'CreditCard', 'IIoT', 'MI']:
    sub = real[(real['dataset'] == ds) & (real['delta'] > 0)]
    if len(sub) == 0:
        continue
    ax.scatter(sub['delta'], sub['auc'],
               marker=markers_ds[ds], color=colors_ds[ds],
               s=60, edgecolors='black', linewidths=0.5,
               label=f'{ds} (Δ>0)', zorder=3)
    
# --- plot Δ < 0  ---
for ds in ['AD', 'CreditCard', 'IIoT', 'MI']:
    sub = real[(real['dataset'] == ds) & (real['delta'] < 0)]
    if len(sub) == 0:
        continue
    ax.scatter(sub['delta'], sub['auc'],
               marker=markers_ds[ds], facecolors='none',
               edgecolors=colors_ds[ds], linewidths=1.5,
               s=60, label=f'{ds} (Δ<0)', zorder=2)

# AUC=0.5 
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2.5, alpha=0.9)

# Empirically enveloping: the minimum slope that falls below (0,0.5) for all points
# --- Diagonal line y=0.5+C3 * Δ, C3 is estimated only from points where Δ>0 ---
if len(real_pos) > 0:
    delta_pos = real_pos['delta'].values
    auc_pos   = real_pos['auc'].values
    c3_est = np.max((auc_pos - 0.5) / delta_pos)
    c3_est = max(c3_est, 0.0)

    delta_grid = np.linspace(real['delta'].min() * 1.1,
                             real['delta'].max() * 1.1, 200)
    ax.plot(delta_grid, 0.5 + c3_est * delta_grid,
            color='blue', linestyle='-', linewidth=1.5,
            label=f'Bound: $0.5 + {c3_est:.2f}\\Delta$')

ax.axvline(x=0, color='black', linestyle=':', linewidth=1.2, alpha=0.6)

ax.set_xlabel(r'$\Delta$', fontsize=16)
ax.set_ylabel('AUC', fontsize=16)
ax.set_ylim(0, 1) 
ax.set_title(r'(a) Real data: AUC vs. $\Delta$', fontsize=16)
ax.tick_params(axis='both', labelsize=16)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12, loc='lower left', ncol=2)





# ------------------------------------------------------------
# (b) sub figure a: LRT only
# ------------------------------------------------------------
ax = axes[1]
ax.scatter(deltas, auc_lrt, color='#9467bd', marker='o', s=80,
           edgecolors='black', linewidths=0.5,
           label='LRT (Neyman-Pearson optimal)', zorder=3)
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2.5, alpha=0.9)
ax.axvline(x=0.0, color='black', linestyle=':', linewidth=2.2, alpha=0.6)

mask_pos = deltas > 0
if np.sum(mask_pos) >= 3:
    coeffs = np.polyfit(deltas[mask_pos], auc_lrt[mask_pos], 1)
    slope, intercept = coeffs[0], coeffs[1]
    y_pred = np.polyval(coeffs, deltas[mask_pos])
    ss_res = np.sum((auc_lrt[mask_pos] - y_pred) ** 2)
    ss_tot = np.sum((auc_lrt[mask_pos] - np.mean(auc_lrt[mask_pos])) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    dgrid = np.linspace(0, deltas.max(), 100)
    ax.plot(dgrid, intercept + slope * dgrid,
            color='blue', linestyle='-', linewidth=1.5,
            label=f'Fit (Δ>0): {intercept:.3f} + {slope:.3f}Δ, R²={r2:.4f}')

ax.set_xlabel(r'$\Delta$', fontsize=16)
ax.set_ylabel('LRT AUC', fontsize=16)
ax.set_title(r'(b) Synthetic: LRT AUC vs. $\Delta$', fontsize=16)
ax.tick_params(axis='both', labelsize=16)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=16, loc='upper left')

# ------------------------------------------------------------
# (c) subfigur b: 
# ------------------------------------------------------------
ax = axes[2]
ax.scatter(deltas, auc_lrt,    color='#9467bd', marker='o', s=80,
           edgecolors='blue', linewidths=0.5,
           label='LRT (optimal)', zorder=4)
ax.scatter(deltas, auc_logreg, color='#1f77b4', marker='s', s=80,
           edgecolors='black', linewidths=0.5,
           label='Logistic Regression', zorder=3)

ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2.5, alpha=0.9)
ax.axvline(x=0.0, color='black', linestyle=':', linewidth=2.2, alpha=0.6)

#LRT upper boundary: Starting from (0,0.5), slope=LRT fitting slope
c3_est = float(slope)
dgrid2 = np.linspace(0, deltas.max(), 100)
ax.plot(dgrid2, 0.5 + c3_est * dgrid2,
        color='blue', linestyle='-', linewidth=1.5,
        label=f'LRT bound: 0.5 + {c3_est:.3f}Δ')

ax.set_xlabel(r'$\Delta$', fontsize=16)
ax.set_ylabel('AUC', fontsize=16)
ax.set_title(r'(c) Synthetic: all classifiers vs. Theorem 2 bound', fontsize=16)
ax.tick_params(axis='both', labelsize=16)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=16, loc='upper left')

# ============================================================
# 11. 
# ============================================================
plt.tight_layout()
plt.savefig('T2_validation_1x3.tiff', dpi=300, format='tiff', bbox_inches='tight')
plt.show()
print("saved as: T2_validation_1x3.tiff")

# ============================================================
# 12.  
# ============================================================
print("\n=== Theorem 2 Validation ===")
print(f"Real configurations with Δ > 0: {len(real_pos)}")
if len(real_pos) > 0:
    print(f"Empirical C3 (upper envelope slope): {c3_est:.4f}")
    violations = real_pos[real_pos['auc'] > 0.5 + c3_est * real_pos['delta']]
    print(f"Violations of the bound: {len(violations)}")

print(f"\nSynthetic LRT AUC (WD vs Poisson): {auc_theory:.4f}")
print("\n=== Theorem 2: LRT vs baselines ===")
print(f"Δ range: [{np.nanmin(deltas):.4f}, {np.nanmax(deltas):.4f}]")
print(f"LRT AUC range: [{np.nanmin(auc_lrt):.4f}, {np.nanmax(auc_lrt):.4f}]")
print(f"LogReg AUC range: [{np.nanmin(auc_logreg):.4f}, {np.nanmax(auc_logreg):.4f}]")
print(f"\nLRT fit (Δ>0): intercept={intercept:.4f}, slope={slope:.4f}, R²={r2:.4f}")
print(f"LRT bound slope C3 = {c3_est:.4f}")