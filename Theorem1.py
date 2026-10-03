"""
Theorem 1 Validation: Geometric Factor Perturbation Experiment

No train-test split is used. All normal samples are used for geometry estimation
and spectral analysis. For T1a, we randomly draw n samples from the full normal
set to compute the empirical covariance and spectral statistics; this is repeated
30 times for each n.

All results are saved to ./results_theorem1/
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from utils import (
    RANDOM_SEED, ALPHA,
    get_random_state, get_subsample_indices,
    compute_auc_and_delta_for_config, project_data,
    GeometryEstimator, compute_gdi,
    P_FIXED, P_RANGE, N_RANGE, NOISE_FACTORS,
    load_dataset, load_mnist,
)

np.random.seed(RANDOM_SEED)
np.random.seed(42)

RESULTS_DIR = './results_theorem1'
os.makedirs(RESULTS_DIR, exist_ok=True)

N_REPEATS = 10


# ============================================================================
# T1a:  
# ============================================================================
def run_t1a(X_normal, X_anomaly, dataset_name, p_fixed, n_values, alpha=ALPHA, n_repeats=1):
    print(f"\n  T1a: Extract proportionally based on the overall anomaly rate (AUC-based)...")
    if n_repeats > 1:
        print(f"    Running {n_repeats} repeats for {dataset_name}")

    n_normal_full = len(X_normal)
    n_anomaly_full = len(X_anomaly)
    N_full = n_normal_full + n_anomaly_full
    anomaly_rate = n_anomaly_full / N_full
    print(f"    Full: N={N_full}, normal={n_normal_full}, anomaly={n_anomaly_full}, rate={anomaly_rate:.4%}")

    geo_estimator = GeometryEstimator()
    tau, kappa, sigma2, d = geo_estimator.estimate(X_normal)
    print(f"    Geometry: tau={tau:.4f}, kappa={kappa:.4f}, sigma2={sigma2:.6f}, d={d}")

    all_results = []

    for n in n_values:
        if n > N_full:
            print(f"    n={n} exceeds full N ({N_full}), skipping...")
            continue

        n_anomaly_sub = max(1, int(round(n * anomaly_rate)))
        n_normal_sub = n - n_anomaly_sub
        if n_normal_sub < 5:
            print(f"    n={n}: n_normal_sub={n_normal_sub} < 5, skipping...")
            continue

        aucs, deltas = [], []

        for rep in range(n_repeats):
            seed_offset = n + rep * 10000
            idx_normal = get_subsample_indices(n_normal_full, n_normal_sub, seed_offset)
            idx_anomaly = get_subsample_indices(n_anomaly_full, n_anomaly_sub, seed_offset + 777)
            X_normal_subset = X_normal[idx_normal]
            X_anomaly_subset = X_anomaly[idx_anomaly]

            p_eff = min(p_fixed, X_normal_subset.shape[1])
            if p_eff <= 0:
                continue

            from sklearn.decomposition import PCA
            pca = PCA(n_components=p_eff)
            X_normal_proj = pca.fit_transform(X_normal_subset)
            X_anomaly_proj = pca.transform(X_anomaly_subset)
            if p_eff < p_fixed:
                X_normal_proj = np.hstack([X_normal_proj, np.zeros((n_normal_sub, p_fixed - p_eff))])
                X_anomaly_proj = np.hstack([X_anomaly_proj, np.zeros((n_anomaly_sub, p_fixed - p_eff))])

            m_adaptive = max(1, p_fixed // 10)

            verbose_this = (n == min(n_values) or n == max(n_values))

            auc, delta = compute_auc_and_delta_for_config(
                X_normal_proj, X_anomaly_proj, alpha, m_adaptive,
                verbose=verbose_this
            )

            aucs.append(auc)
            deltas.append(delta)

        if len(aucs) > 0:
            auc_mean = np.mean(aucs)
            auc_std = np.std(aucs) if len(aucs) > 1 else 0.0
            delta_mean = np.mean(deltas)
            delta_std = np.std(deltas) if len(deltas) > 1 else 0.0

            X_c_sub = X_normal_subset - np.mean(X_normal_subset, axis=0)
            Sigma_sub = (X_c_sub.T @ X_c_sub) / len(X_normal_subset)
            lambda_max_sub = float(np.linalg.eigvalsh(Sigma_sub)[-1])
            gdi = compute_gdi(tau, kappa, sigma2, p_fixed, n, lambda_max=lambda_max_sub)

            all_results.append({
                'dataset': dataset_name, 'experiment': 'T1a',
                'n': n, 'p': p_fixed, 'm': m_adaptive,
                'tau': tau, 'kappa': kappa, 'sigma2': sigma2, 'gdi': gdi,
                'auc': auc_mean, 'auc_std': auc_std,
                'delta': delta_mean, 'delta_std': delta_std,
                'n_repeats': n_repeats
            })
            print(f"    n={n} (normal={n_normal_sub}, anomaly={n_anomaly_sub}): "
                  f"AUC={auc_mean:.4f}±{auc_std:.4f}, Delta={delta_mean:.4f}±{delta_std:.4f}, GDI={gdi:.4f}")

    return pd.DataFrame(all_results)


# ============================================================================
# T1b: 
# ============================================================================
def run_t1b(X_normal, X_anomaly, dataset_name, p_fixed, p_values, alpha=ALPHA, n_repeats=1):
    print(f"\n  T1b: Dimensionality Reduction (normal + anomaly)...")
    n_normal_full = len(X_normal)
    n_anomaly_full = len(X_anomaly)
    print(f"    Full: normal={n_normal_full}, anomaly={n_anomaly_full}")

    geo_estimator = GeometryEstimator()
    tau, kappa, sigma2, d = geo_estimator.estimate(X_normal)
    print(f"    Geometry: tau={tau:.4f}, kappa={kappa:.4f}, sigma2={sigma2:.6f}, d={d}")

    results = []
    for p in p_values:
        m_adaptive = max(1, p // 10)
        for method in ['pca', 'goe', 'random']:
            if method == 'pca' and p > X_normal.shape[1]:
                continue
            aucs, deltas = [], []
            for rep in range(n_repeats):
                seed = 42 + p + (0 if method == 'pca' else 1 if method == 'goe' else 2) * 1000 + rep * 50000

                if method == 'pca':
                    # ---- FIXED: fit on normal, transform anomaly with the SAME pca ----
                    from sklearn.decomposition import PCA
                    p_eff = min(p, X_normal.shape[1], X_normal.shape[0])
                    if p_eff <= 0:
                        continue
                    pca = PCA(n_components=p_eff)
                    X_normal_proj = pca.fit_transform(X_normal)
                    X_anomaly_proj = pca.transform(X_anomaly)
                    if p_eff < p:
                        X_normal_proj = np.hstack([
                            X_normal_proj,
                            np.zeros((n_normal_full, p - p_eff))
                        ])
                        X_anomaly_proj = np.hstack([
                            X_anomaly_proj,
                            np.zeros((n_anomaly_full, p - p_eff))
                        ])
                else:
                    X_normal_proj = project_data(X_normal, p, method, seed)
                    X_anomaly_proj = project_data(X_anomaly, p, method, seed)

                auc, delta = compute_auc_and_delta_for_config(
                    X_normal_proj, X_anomaly_proj, alpha, m_adaptive
                )
                aucs.append(auc); deltas.append(delta)

            auc_mean = np.mean(aucs)
            auc_std = np.std(aucs) if len(aucs) > 1 else 0.0
            delta_mean = np.mean(deltas)
            delta_std = np.std(deltas) if len(deltas) > 1 else 0.0
            gdi = compute_gdi(tau, kappa, sigma2, p, n_normal_full + n_anomaly_full)

            results.append({
                'dataset': dataset_name, 'experiment': 'T1b',
                'p': p, 'm': m_adaptive, 'method': method,
                'tau': tau, 'kappa': kappa, 'sigma2': sigma2, 'gdi': gdi,
                'auc': auc_mean, 'auc_std': auc_std,
                'delta': delta_mean, 'delta_std': delta_std,
                'n_repeats': n_repeats
            })
            print(f"    p={p}, method={method}: AUC={auc_mean:.4f}±{auc_std:.4f}, Delta={delta_mean:.4f}±{delta_std:.4f}")

    return pd.DataFrame(results)


# ============================================================================
# T1c:  
# ============================================================================
def run_t1c(X_normal, X_anomaly, dataset_name, p_fixed, noise_factors, alpha=ALPHA, n_repeats=1):
    print(f"\n  T1c: Noise Injection (normal + anomaly)...")
    n_normal_full = len(X_normal)
    n_anomaly_full = len(X_anomaly)

    geo_estimator = GeometryEstimator()
    tau, kappa, sigma2_base, d = geo_estimator.estimate(X_normal)
    print(f"    Geometry: tau={tau:.4f}, kappa={kappa:.4f}, sigma2_base={sigma2_base:.6f}, d={d}")

    results = []
    m_adaptive = max(1, p_fixed // 10)

    for factor in noise_factors:
        aucs, deltas = [], []
        for rep in range(n_repeats):
            rng = get_random_state(42 + factor + rep * 1000)
            noise_std = np.sqrt(sigma2_base * factor)
            X_normal_noisy = X_normal + rng.randn(*X_normal.shape) * noise_std

            p_eff = min(p_fixed, X_normal_noisy.shape[1])
            from sklearn.decomposition import PCA
            pca = PCA(n_components=p_eff)
            X_normal_proj = pca.fit_transform(X_normal_noisy)
            X_anomaly_proj = pca.transform(X_anomaly)
            if p_eff < p_fixed:
                X_normal_proj = np.hstack([X_normal_proj, np.zeros((n_normal_full, p_fixed - p_eff))])
                X_anomaly_proj = np.hstack([X_anomaly_proj, np.zeros((n_anomaly_full, p_fixed - p_eff))])

            auc, delta = compute_auc_and_delta_for_config(
                X_normal_proj, X_anomaly_proj, alpha, m_adaptive
            )
            aucs.append(auc); deltas.append(delta)

        if len(aucs) > 0:
            auc_mean = np.mean(aucs)
            auc_std = np.std(aucs) if len(aucs) > 1 else 0.0
            delta_mean = np.mean(deltas)
            delta_std = np.std(deltas) if len(deltas) > 1 else 0.0
            sigma2_effective = sigma2_base * factor
            gdi = compute_gdi(tau, kappa, sigma2_effective, p_fixed, n_normal_full + n_anomaly_full)

            results.append({
                'dataset': dataset_name, 'experiment': 'T1c',
                'noise_factor': factor, 'p': p_fixed, 'm': m_adaptive,
                'tau': tau, 'kappa': kappa,
                'sigma2_effective': sigma2_effective, 'gdi': gdi,
                'auc': auc_mean, 'auc_std': auc_std,
                'delta': delta_mean, 'delta_std': delta_std,
                'n_repeats': n_repeats
            })
            print(f"    noise_factor={factor}: AUC={auc_mean:.4f}±{auc_std:.4f}, Delta={delta_mean:.4f}±{delta_std:.4f}")

    return pd.DataFrame(results)


# ============================================================================
# T1d:  
# ============================================================================
def run_t1d(X_normal, X_anomaly, dataset_name, p_fixed, n_repeats=1):
    print(f"\n  T1d: Cross-Dataset tau Observation...")
    geo_estimator = GeometryEstimator()
    tau, kappa, sigma2, d = geo_estimator.estimate(X_normal)

    n_max = len(X_normal) + len(X_anomaly)
    gdi = compute_gdi(tau, kappa, sigma2, p_fixed, n_max)

    p_eff = min(p_fixed, X_normal.shape[1])

    aucs, deltas = [], []
    for rep in range(n_repeats):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=p_eff)
        X_normal_proj = pca.fit_transform(X_normal)
        X_anomaly_proj = pca.transform(X_anomaly)
        if p_eff < p_fixed:
            X_normal_proj = np.hstack([X_normal_proj, np.zeros((len(X_normal), p_fixed - p_eff))])
            X_anomaly_proj = np.hstack([X_anomaly_proj, np.zeros((len(X_anomaly), p_fixed - p_eff))])

        m_adaptive = max(1, p_fixed // 10)
        auc, delta = compute_auc_and_delta_for_config(
            X_normal_proj, X_anomaly_proj, ALPHA, m_adaptive
        )
        aucs.append(auc); deltas.append(delta)

    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs) if len(aucs) > 1 else 0.0
    delta_mean = np.mean(deltas)
    delta_std = np.std(deltas) if len(deltas) > 1 else 0.0

    print(f"    tau={tau:.4f}, kappa={kappa:.4f}, sigma2={sigma2:.6f}, d={d}")
    print(f"    GDI={gdi:.4f}")
    print(f"    AUC={auc_mean:.4f}±{auc_std:.4f}, Delta={delta_mean:.4f}±{delta_std:.4f}")

    return {
        'dataset': dataset_name, 'experiment': 'T1d',
        'p': p_fixed, 'n': n_max,
        'tau': tau, 'kappa': kappa, 'sigma2': sigma2, 'd': d, 'gdi': gdi,
        'auc': auc_mean, 'auc_std': auc_std,
        'delta': delta_mean, 'delta_std': delta_std,
        'n_repeats': n_repeats
    }


# ============================================================================
# MNIST: 
# ============================================================================
def run_mnist_t1a():
    print("\n" + "=" * 70)
    print("MNIST (只跑 T1a, 1 次)")
    print("=" * 70)
    X_mnist, y_mnist = load_mnist()
    X_normal = X_mnist[y_mnist == 0]
    X_anomaly = X_mnist[y_mnist == 1]
    print(f"MNIST: normal={len(X_normal)}, anomaly={len(X_anomaly)}")

    df = run_t1a(
        X_normal, X_anomaly, 'MNIST',
        p_fixed=P_FIXED['MNIST'], n_values=N_RANGE['MNIST'],
        n_repeats=1
    )
    return df


# ============================================================================
# Rank_SUMMARY
# ============================================================================
def compute_rank_summary(df_all):
    from scipy.stats import spearmanr

    rows = []

    def _calc(sub_df, label):
        gdi = sub_df['gdi'].values
        delta = sub_df['delta'].values
        auc = sub_df['auc'].values
        mask_g = np.isfinite(gdi) & np.isfinite(auc)
        mask_d = np.isfinite(delta) & np.isfinite(auc)
        if mask_g.sum() >= 3:
            rho_g, p_g = spearmanr(gdi[mask_g], auc[mask_g])
        else:
            rho_g, p_g = np.nan, np.nan
        if mask_d.sum() >= 3:
            rho_d, p_d = spearmanr(delta[mask_d], auc[mask_d])
        else:
            rho_d, p_d = np.nan, np.nan
        rows.append({'dataset': label, 'metric': 'GDI', 'spearman_rho': rho_g, 'p_value': p_g})
        rows.append({'dataset': label, 'metric': 'Delta', 'spearman_rho': rho_d, 'p_value': p_d})

    df_t1abc = df_all[df_all['experiment'].isin(['T1a', 'T1b', 'T1c'])].copy()

    _calc(df_t1abc, 'All')

    for ds in ['AD', 'MI', 'IIoT', 'CreditCard']:
        sub = df_t1abc[df_t1abc['dataset'] == ds]
        _calc(sub, ds)

    return pd.DataFrame(rows)


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Theorem 1 Validation (5 datasets)")
    print("=" * 70)
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Alpha={ALPHA}, Repeats={N_REPEATS} (MNIST=1)")
    print("Run order: MI, IIoT, AD, CreditCard")

    all_dfs = {'T1a': [], 'T1b': [], 'T1c': [], 'T1d': []}
    t1d_results = []

    run_order = ['CreditCard','IIoT','MI','AD']
    dataset_summaries = {}

    for dataset_name in run_order:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name} (30 repeats)")
        print(f"{'='*60}")

        config = {
            'AD': {'p_fixed': P_FIXED['AD'], 'p_range': P_RANGE['AD'], 'n_range': N_RANGE['AD']},
            'MI': {'p_fixed': P_FIXED['MI'], 'p_range': P_RANGE['MI'], 'n_range': N_RANGE['MI']},
            'CreditCard': {'p_fixed': P_FIXED['CreditCard'], 'p_range': P_RANGE['CreditCard'], 'n_range': N_RANGE['CreditCard']},
            'IIoT': {'p_fixed': P_FIXED['IIoT'], 'p_range': P_RANGE['IIoT'], 'n_range': N_RANGE['IIoT']},
        }[dataset_name]

        X_full, y_full = load_dataset(dataset_name)
        X_normal = X_full[y_full == 0]
        X_anomaly = X_full[y_full == 1]
        print(f"  Normal: {len(X_normal)}, Anomaly: {len(X_anomaly)}")

        from utils import GeometryEstimator, p_effective_from_d

        geo_tmp = GeometryEstimator()
        tau_tmp, kappa_tmp, sigma2_tmp, d_est = geo_tmp.estimate(X_normal, verbose=True)
        lambda_max_actual = geo_tmp.lambda_max
        lambda_max_theory = tau_tmp**2 / max(kappa_tmp**2, 1e-12)
        ratio_geom = lambda_max_actual / max(lambda_max_theory, 1e-12)
        print(f"  [GeomCheck] lambda_max_actual={lambda_max_actual:.6f}, "
              f"tau^2/kappa^2={lambda_max_theory:.6f}, "
              f"ratio={ratio_geom:.6f}")

        n_normal_count = len(X_normal)
        p_adaptive = p_effective_from_d(d_est, n_normal_count, C_max=1.0, k_exp=0.75)
        p_adaptive = min(p_adaptive, X_normal.shape[1])
        p_adaptive = max(10, p_adaptive)
        config['p_fixed'] = p_adaptive
        P_FIXED[dataset_name] = p_adaptive
        print(f"  [Corollary 8.1] d_est={d_est}, n={n_normal_count}, p_eff={p_adaptive}")

        base_p = p_adaptive
        D_ambient = X_normal.shape[1]
        p_candidates = sorted(set([
            max(2, base_p // 2),
            max(2, int(base_p * 0.75)),
            base_p,
            min(D_ambient, int(base_p * 1.5)),
            min(D_ambient, base_p * 2),
        ]))
        p_candidates = [p for p in p_candidates if 2 <= p <= D_ambient]
        if len(p_candidates) < 2:
            p_candidates = [max(2, min(D_ambient, base_p))]
        config['p_range'] = p_candidates
        print(f"  [Adaptive] p_range={p_candidates}")

        df_t1a = run_t1a(X_normal, X_anomaly, dataset_name,
                         config['p_fixed'], config['n_range'], n_repeats=N_REPEATS)
        all_dfs['T1a'].append(df_t1a)

        X_c = X_normal - np.mean(X_normal, axis=0)
        Sigma_hat = (X_c.T @ X_c) / len(X_normal)
        lambda_max_actual = np.linalg.eigvalsh(Sigma_hat)[-1]
        lambda_max_theory = tau_tmp**2 / kappa_tmp**2
        print(f"  [GeomCheck] lambda_max_actual={lambda_max_actual:.4f}, "
              f"tau^2/kappa^2={lambda_max_theory:.4f}, "
              f"ratio={lambda_max_actual / max(lambda_max_theory, 1e-12):.4f}")

        df_t1b = run_t1b(X_normal, X_anomaly, dataset_name,
                         config['p_fixed'], config['p_range'], n_repeats=N_REPEATS)
        all_dfs['T1b'].append(df_t1b)

        df_t1c = run_t1c(X_normal, X_anomaly, dataset_name,
                         config['p_fixed'], NOISE_FACTORS, n_repeats=N_REPEATS)
        all_dfs['T1c'].append(df_t1c)

        t1d_res = run_t1d(X_normal, X_anomaly, dataset_name,
                          config['p_fixed'], n_repeats=N_REPEATS)
        t1d_results.append(t1d_res)

        df_ds = pd.concat([df_t1a, df_t1b, df_t1c, pd.DataFrame([t1d_res])], ignore_index=True)
        dataset_summaries[dataset_name] = df_ds
        out_name = f'{dataset_name}_SUMMARY.CSV'
        df_ds.to_csv(os.path.join(RESULTS_DIR, out_name), index=False)
        print(f"  Saved: {out_name}")

    df_all = pd.concat(list(dataset_summaries.values()), ignore_index=True)
    df_all.to_csv(os.path.join(RESULTS_DIR, 'SUMMARY.CSV'), index=False)
    print(f"\nSaved: SUMMARY.CSV ({len(df_all)} rows)")

    #df_mnist = run_mnist_t1a()
    #df_mnist.to_csv(os.path.join(RESULTS_DIR, 'MNIST_summary.csv'), index=False)
    #print(f"Saved: MNIST_summary.csv ({len(df_mnist)} rows)")

    df_rank = compute_rank_summary(df_all)
    df_rank.to_csv(os.path.join(RESULTS_DIR, 'Rank_SUMMARY.csv'), index=False)
    print(f"Saved: Rank_SUMMARY.csv ({len(df_rank)} rows)")

    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)
    print(f"Output files in: {RESULTS_DIR}")
    print("  - MI_SUMMARY.CSV, AD_SUMMARY.CSV, CreditCard_SUMMARY.CSV, IIoT_SUMMARY.CSV")
    print("  - SUMMARY.CSV")
    print("  - Rank_SUMMARY.csv")


if __name__ == "__main__":
    main()