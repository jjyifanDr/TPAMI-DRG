import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, mannwhitneyu

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12

RNG = np.random.default_rng(42)

# ============================================================
# WD / Poisson  
# ============================================================
def wd_pdf(s):
    return (np.pi * s / 2.0) * np.exp(-np.pi * s**2 / 4.0)

def wd_cdf(s):
    return 1.0 - np.exp(-np.pi * s**2 / 4.0)

def poisson_pdf(s):
    return np.exp(-s)

def poisson_cdf(s):
    return 1.0 - np.exp(-s)

def sample_wd(n, rng):
    u = rng.uniform(0, 1, n)
    return np.sqrt(-4.0 / np.pi * np.log(1.0 - u))

def sample_poisson(n, rng):
    return rng.exponential(1.0, n)

def sample_normal(n, theta, rng):
    mask = rng.uniform(0, 1, n) < theta
    out = np.empty(n)
    n_p = int(np.sum(mask))
    n_wd = n - n_p
    if n_wd > 0:
        out[~mask] = sample_wd(n_wd, rng)
    if n_p > 0:
        out[mask] = sample_poisson(n_p, rng)
    return out

def sample_anomaly(n, rng):
    return sample_wd(n, rng)

def compute_delta(theta):
    grid = np.linspace(0, 30, 5000)
    F0 = theta * poisson_cdf(grid) + (1 - theta) * wd_cdf(grid)
    w1_wd = np.trapz(np.abs(F0 - wd_cdf(grid)), grid)
    w1_p = np.trapz(np.abs(F0 - poisson_cdf(grid)), grid)
    return w1_wd - w1_p

def lrt_score(s, theta):
    s = np.clip(s, 1e-12, None)
    f0 = theta * poisson_pdf(s) + (1 - theta) * wd_pdf(s)
    f1 = wd_pdf(s)
    return np.log(f0 + 1e-300) - np.log(f1 + 1e-300)

def auc_mwu(scores_pos, scores_neg):
    u, _ = mannwhitneyu(scores_pos, scores_neg, alternative='greater')
    return u / (len(scores_pos) * len(scores_neg))


# ============================================================
#   KDE
# ============================================================
def safe_kde(x, x_grid, eps=1e-8, spike_width=None):
    x = np.asarray(x).ravel()
    if x.size == 0:
        return np.zeros_like(x_grid)
    if np.std(x) < eps:
        c = float(x[0])
        if spike_width is None:
            span = x_grid[-1] - x_grid[0]
            spike_width = max(span / 200.0, 1e-6)
        density = np.exp(-0.5 * ((x_grid - c) / spike_width) ** 2)
        density /= (np.sqrt(2 * np.pi) * spike_width)
        return density
    return gaussian_kde(x)(x_grid)


# ============================================================
#  
# ============================================================
def plot_2d_density():
    thetas = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_samples = 2000
    noise_std = 0.6

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

    yticks = list(range(len(thetas)))
    yticklabels = [f'θ={t:.2f}' for t in thetas]

    # 存储每行要在循环结束后统一绘制的文本内容
    text_info = {0: [], 1: []}

    for col, add_noise in enumerate([False, True]):
        ax = axes[col]
        for i, theta in enumerate(thetas):
            delta = compute_delta(theta)

            s_normal = sample_normal(n_samples, theta, RNG)
            s_anomaly = sample_anomaly(n_samples, RNG)

            if add_noise:
                s_normal = s_normal + RNG.normal(0, noise_std, n_samples)
                s_normal = np.clip(s_normal, 1e-6, None)

            scores_n = lrt_score(s_normal, theta)
            scores_a = lrt_score(s_anomaly, theta)
            auc = auc_mwu(scores_n, scores_a)

            x_min = min(scores_n.min(), scores_a.min())
            x_max = max(scores_n.max(), scores_a.max())
            span = x_max - x_min
            if span < 1e-6:
                x_min -= 0.5
                x_max += 0.5
                span = x_max - x_min
            pad = 0.05 * span
            x_grid = np.linspace(x_min - pad, x_max + pad, 400)

            kde_n = safe_kde(scores_n, x_grid)
            kde_a = safe_kde(scores_a, x_grid)

            height = 0.7
            if kde_n.max() > 0:
                kde_n = kde_n / kde_n.max() * height
            if kde_a.max() > 0:
                kde_a = kde_a / kde_a.max() * height

            offset = i * 1.0
            ax.fill_between(x_grid, offset, offset + kde_n,
                            color='#1f77b4', alpha=0.7,
                            label='Normal' if i == 0 else None)
            ax.fill_between(x_grid, offset, offset + kde_a,
                            color='#d62728', alpha=0.9,
                            label='Anomaly' if i == 0 else None)

            # 只记录文本内容，不在这里绘制
            text_info[col].append(
                (offset + height * 1.05,
                 f'Δ={delta:+.2f}, AUC={auc:.2f}')
            )

        ax.set_xlabel('LRT score  $t(s)$', fontsize=12)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=12)
        title = '(b) With noise on normal' if add_noise else '(a) No noise'
        ax.tick_params(axis='both', labelsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='lower right')

    #  
    fig.canvas.draw()

    #  
    for col, ax in enumerate(axes):
        #  
        x_right_data = ax.transData.inverted().transform(
            ax.transAxes.transform((1.0, 0.0))
        )[0]
        #  
        x_text = x_right_data - 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0])

        for y_text, label in text_info[col]:
            ax.text(x_text, y_text, label,
                    fontsize=12, ha='right', va='bottom')

    plt.tight_layout()
    plt.savefig('T3_degeneration_density_2d_v2.tiff', dpi=300, format='tiff',
                bbox_inches='tight')
    plt.show()
    print("Saved: T3_degeneration_density_2d_v2.tiff")


if __name__ == '__main__':
    plot_2d_density()