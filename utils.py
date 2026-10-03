"""
utils.py - Fixed: tau and kappa estimated independently, fast structure preserved.
"""

import os
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

ALPHA = 0.1
M = None


# ============================================================================
# Random Utility Functions
# ============================================================================
def get_random_state(seed):
    return np.random.RandomState(seed)


def get_subsample_indices(n_total, n_sample, seed_offset):
    rng = get_random_state(42 + seed_offset)
    return rng.choice(n_total, n_sample, replace=False)


# ============================================================================
# Intrinsic Dimension Estimator: MLE (Levina-Bickel, 2004)
# ============================================================================
def estimate_d_mle_levina_bickel(X, k=None):
    n, D = X.shape
    if n < 10:
        return 1
    if k is None:
        k = min(n - 1, max(10, int(np.sqrt(n))))
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    r = distances[:, 1:]
    r = np.maximum(r, 1e-12)
    r_k = r[:, -1:]
    log_ratios = np.log(r_k) - np.log(r[:, :-1])
    mean_log = np.mean(log_ratios, axis=1)
    mean_log = np.maximum(mean_log, 1e-12)
    d_i = 1.0 / mean_log
    d_i = d_i[np.isfinite(d_i) & (d_i > 0)]
    if len(d_i) == 0:
        return 1
    return max(1, int(round(np.median(d_i))))


def p_effective_from_d(d_est, n, C_max=1.0, k_exp=0.75):
    if d_est <= 1 or n <= 1:
        return 2
    return max(2, int(np.ceil(C_max * d_est * (np.log(n) ** k_exp))))


# ============================================================================
# Spectral Functions
# ============================================================================
def compute_empirical_covariance(X):
    n_samples = X.shape[0]
    X_centered = X - np.mean(X, axis=0)
    Sigma = (X_centered.T @ X_centered) / n_samples
    return Sigma


def extract_eigenspacings(Sigma, p=None, bulk_ratio=None):
    eigenvalues = eigh(Sigma, eigvals_only=True)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    n_eig = len(eigenvalues)
    if n_eig < 3:
        return np.array([])

    if bulk_ratio is None:
        if p is not None and p <= 10:
            bulk_ratio = 0.01
        else:
            bulk_ratio = 0.05

    start_idx = int(np.ceil(bulk_ratio * n_eig))
    end_idx = int(np.floor((1 - bulk_ratio) * n_eig))

    if end_idx - start_idx < 2:
        start_idx = 0
        end_idx = n_eig

    bulk_eigenvalues = eigenvalues[start_idx:end_idx]

    if len(bulk_eigenvalues) < 2:
        return np.array([])

    spacings = []
    for i in range(len(bulk_eigenvalues) - 1):
        window = max(1, int(0.1 * len(bulk_eigenvalues)))
        if i - window >= 0 and i + window < len(bulk_eigenvalues):
            Delta = (bulk_eigenvalues[i - window] - bulk_eigenvalues[i + window]) / (2 * window)
        else:
            Delta = np.mean(np.diff(bulk_eigenvalues))

        if Delta > 1e-10:
            s = (bulk_eigenvalues[i] - bulk_eigenvalues[i+1]) / Delta
            spacings.append(s)

    return np.array(spacings)


def compute_wasserstein_distance_to_wd(spacings):
    if len(spacings) < 3:
        return 1.0
    spacings_sorted = np.sort(spacings)
    emp_cdf = np.linspace(0, 1, len(spacings_sorted) + 1)[1:]
    wd_cdf_at_points = 1 - np.exp(-np.pi * spacings_sorted**2 / 4)
    emp_cdf = np.clip(emp_cdf, 1e-10, 1 - 1e-10)
    wd_cdf_at_points = np.clip(wd_cdf_at_points, 1e-10, 1 - 1e-10)
    w1_dist = np.trapz(np.abs(emp_cdf - wd_cdf_at_points), spacings_sorted)
    return w1_dist


def compute_wasserstein_distance_to_poisson(spacings):
    if len(spacings) < 3:
        return 1.0
    spacings_sorted = np.sort(spacings)
    emp_cdf = np.linspace(0, 1, len(spacings_sorted) + 1)[1:]
    poisson_cdf_at_points = 1 - np.exp(-spacings_sorted)
    emp_cdf = np.clip(emp_cdf, 1e-10, 1 - 1e-10)
    poisson_cdf_at_points = np.clip(poisson_cdf_at_points, 1e-10, 1 - 1e-10)
    w1_dist = np.trapz(np.abs(emp_cdf - poisson_cdf_at_points), spacings_sorted)
    return w1_dist


def compute_delta_from_spacings(spacings):
    if len(spacings) < 3:
        return 0.0
    d_wd = compute_wasserstein_distance_to_wd(spacings)
    d_p = compute_wasserstein_distance_to_poisson(spacings)
    return d_wd - d_p


def compute_delta(X, alpha=ALPHA, m=None):
    Sigma = compute_empirical_covariance(X)
    p = X.shape[1]
    spacings = extract_eigenspacings(Sigma, p=p)
    return compute_delta_from_spacings(spacings)


def w1_detector_score(X_normal, X_test, alpha=ALPHA, m=None):
    n_normal = X_normal.shape[0]
    n_test = X_test.shape[0]
    scores = np.zeros(n_test)
    if n_normal < 3 or n_test < 1:
        return scores
    p = X_normal.shape[1]
    for i in range(n_test):
        X_combined = np.vstack([X_normal, X_test[i:i+1]])
        Sigma = compute_empirical_covariance(X_combined)
        spacings = extract_eigenspacings(Sigma, p=p)
        if len(spacings) < 3:
            scores[i] = 0.0
            continue
        d_wd = compute_wasserstein_distance_to_wd(spacings)
        d_p = compute_wasserstein_distance_to_poisson(spacings)
        scores[i] = d_p - d_wd
    return scores


def compute_auc_and_delta_for_config(X_normal, X_anomaly, alpha=ALPHA, m=None, verbose=False):
    n_normal = len(X_normal)
    n_anomaly = len(X_anomaly)
    if n_normal < 5 or n_anomaly < 2:
        return 0.5, 0.0
    X_train = X_normal
    X_test_all = np.vstack([X_normal, X_anomaly])
    y_test_all = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])
    scores = w1_detector_score(X_train, X_test_all, alpha, m)
    try:
        auc = roc_auc_score(y_test_all, scores)
    except:
        auc = 0.5

    if verbose:
        p_eff = X_normal.shape[1]
        Sigma = compute_empirical_covariance(X_normal)
        spacings = extract_eigenspacings(Sigma, p=p_eff)
        if len(spacings) >= 3:
            d_wd = compute_wasserstein_distance_to_wd(spacings)
            d_p = compute_wasserstein_distance_to_poisson(spacings)
            qs = np.quantile(spacings, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
            s_sorted = np.sort(spacings)
            f_emp_at_05 = np.mean(s_sorted <= 0.5)
            f_wd_at_05 = 1 - np.exp(-np.pi * 0.25 / 4)
            f_p_at_05 = 1 - np.exp(-0.5)
            print(f"    [Spec] p={p_eff}, n_norm={n_normal}, n_ano={n_anomaly}")
            print(f"    [Spec] len(spacings)={len(spacings)}, "
                  f"mean={spacings.mean():.4f}, std={spacings.std():.4f}")
            print(f"    [Spec] quantiles(0/10/25/50/75/90/99/100): "
                  f"{qs[0]:.3f}/{qs[1]:.3f}/{qs[2]:.3f}/{qs[3]:.3f}/"
                  f"{qs[4]:.3f}/{qs[5]:.3f}/{qs[6]:.3f}/{qs[7]:.3f}")
            print(f"    [Spec] D_WD={d_wd:.4f}, D_P={d_p:.4f}, Delta={d_wd-d_p:+.4f}")
            print(f"    [Spec] CDF@0.5: emp={f_emp_at_05:.4f}, WD={f_wd_at_05:.4f}, P={f_p_at_05:.4f}")
        else:
            print(f"    [Spec] len(spacings)={len(spacings)} (too few!)")

    if verbose:
        n_norm = n_normal
        s_norm = scores[:n_norm]
        s_ano = scores[n_norm:]
        print(f"    [Det] score_normal: mean={s_norm.mean():+.4f}, std={s_norm.std():.4f}, "
              f"q25={np.quantile(s_norm,0.25):+.4f}, q50={np.quantile(s_norm,0.5):+.4f}, "
              f"q75={np.quantile(s_norm,0.75):+.4f}")
        print(f"    [Det] score_anomaly: mean={s_ano.mean():+.4f}, std={s_ano.std():.4f}, "
              f"q25={np.quantile(s_ano,0.25):+.4f}, q50={np.quantile(s_ano,0.5):+.4f}, "
              f"q75={np.quantile(s_ano,0.75):+.4f}")
        all_s = np.concatenate([s_norm, s_ano])
        bins = np.linspace(all_s.min(), all_s.max(), 30)
        h_n, _ = np.histogram(s_norm, bins=bins, density=True)
        h_a, _ = np.histogram(s_ano, bins=bins, density=True)
        overlap = np.sum(np.minimum(h_n, h_a)) * (bins[1] - bins[0])
        print(f"    [Det] score_overlap={overlap:.4f} (1.0=完全重叠, 0.0=完全分离)")

    delta = compute_delta(X_normal, alpha, m)
    return auc, delta


# ============================================================================
# Projection Methods
# ============================================================================
def pca_projection(X, p):
    n, D = X.shape
    p_eff = min(p, D, n)
    if p_eff <= 0:
        return X[:, :1] if D > 0 else np.zeros((n, 1))
    pca = PCA(n_components=p_eff)
    X_pca = pca.fit_transform(X)
    if p_eff < p:
        X_pca = np.hstack([X_pca, np.zeros((n, p - p_eff))])
    return X_pca


def goe_projection(X, p, seed=None):
    n, D = X.shape
    if p <= 0:
        return np.zeros((n, 1))
    if seed is None:
        seed = RANDOM_SEED
    rng = get_random_state(seed)
    if p <= D:
        W = rng.randn(p, D) / np.sqrt(p)
    else:
        W = rng.randn(p, D)
        U, _, Vt = np.linalg.svd(W, full_matrices=False)
        W = U @ Vt
        W = W / np.sqrt(p)
    return X @ W.T


def random_projection(X, p, seed=None):
    n, D = X.shape
    if p <= 0:
        return np.zeros((n, 1))
    if seed is None:
        seed = RANDOM_SEED
    rng = get_random_state(seed)
    proj_mat = rng.randn(p, D) / np.sqrt(p)
    return X @ proj_mat.T


def project_data(X, p, method='pca', seed=None):
    if method == 'pca':
        return pca_projection(X, p)
    elif method == 'goe':
        return goe_projection(X, p, seed)
    elif method == 'random':
        return random_projection(X, p, seed)
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================================
# Geometry Estimation (Fixed: tau and kappa independent, fast structure)
# ============================================================================
class GeometryEstimator:
    def __init__(self, k_neighbors=30):
        self.k_neighbors = k_neighbors
        self.tau = None
        self.kappa = None
        self.sigma2 = None
        self.d = None
        self.lambda_max = None
        self._last_eigvals = None
        self._last_reach_estimates = None
        self._last_curvature_estimates = None
        self._rng = get_random_state(42)

    def estimate(self, X, verbose=False):
        n, D = X.shape
        if n < 10 or D < 2:
            self.tau, self.kappa, self.sigma2, self.d = 1.0, 1.0, 1e-6, 2
            self.lambda_max = 1.0
            return self.tau, self.kappa, self.sigma2, self.d
        self._estimate_intrinsic_dimension(X)
        self.tau = self._estimate_reach(X)
        self.kappa = self._estimate_curvature(X)
        self.sigma2 = self._estimate_noise_variance(X)
        self._estimate_lambda_max(X)

        if verbose and self._last_eigvals is not None:
            ev = self._last_eigvals
            d = self.d
            print(f"    [Geo] d={d}, tau={self.tau:.4f}, kappa={self.kappa:.6f}, sigma2={self.sigma2:.6f}")
            print(f"    [Geo] kappa*tau={self.kappa * self.tau:.6f}")
            print(f"    [Geo] lambda_max_actual={self.lambda_max:.6f}")
            print(f"    [Geo] tau^2/kappa^2={self.tau**2 / max(self.kappa**2, 1e-12):.6f}")
            ratio = self.lambda_max / max(self.tau**2 / max(self.kappa**2, 1e-12), 1e-12)
            print(f"    [Geo] ratio (lambda_max / tau^2/kappa^2) = {ratio:.6f}")
            print(f"    [Geo] eigvals[:8] = {ev[:8]}")
            d_safe = min(d, len(ev) - 1)
            lo = max(0, d_safe - 2)
            hi = min(len(ev), d_safe + 3)
            print(f"    [Geo] eigvals[d-2:d+3] = {ev[lo:hi]}")
            if d_safe >= 1 and ev[d_safe] > 1e-12:
                ratio_eig = ev[d_safe - 1] / ev[d_safe]
                print(f"    [Geo] eigvals[d-1]/eigvals[d] = {ratio_eig:.4f} (>1.5 说明 d 在肘部)")
            if self._last_reach_estimates is not None and len(self._last_reach_estimates) > 1:
                r = np.array(self._last_reach_estimates)
                print(f"    [Geo] reach: median={np.median(r):.4f}, CV={r.std()/max(r.mean(),1e-12):.4f}, n={len(r)}")
            if self._last_curvature_estimates is not None and len(self._last_curvature_estimates) > 1:
                c = np.array(self._last_curvature_estimates)
                print(f"    [Geo] kappa: median={np.median(c):.6f}, CV={c.std()/max(c.mean(),1e-12):.4f}, n={len(c)}")

        return self.tau, self.kappa, self.sigma2, self.d

    def _estimate_intrinsic_dimension(self, X):
        n, D = X.shape
        if D <= 1 or n < 10:
            self.d = 1
            return
        X_c = X - np.mean(X, axis=0)
        Sigma = (X_c.T @ X_c) / n
        ev = np.linalg.eigvalsh(Sigma)
        ev = np.sort(ev)[::-1]
        ev = np.maximum(ev, 0)
        self._last_eigvals = ev
        d_mle = estimate_d_mle_levina_bickel(X)
        self.d = max(1, min(d_mle, D))

    def _estimate_lambda_max(self, X):
        n, D = X.shape
        X_c = X - np.mean(X, axis=0)
        Sigma = (X_c.T @ X_c) / n
        ev = np.linalg.eigvalsh(Sigma)
        self.lambda_max = float(max(ev[-1], 1e-12))

    def _estimate_reach(self, X):
        """
        原始快速结构: 每个采样点一次 eigvalsh。
        tau 用 normal_var / tangent_var 的比值。
        """
        n, D = X.shape
        if D >= 512:
            k = min(n, max(100, min(300, int(3 * np.sqrt(D) * np.log(max(D, 2))))))
        else:
            k = min(n, max(30, min(100, int(3 * np.sqrt(D) * np.log(max(D, 2))))))
        k = max(k, 10)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        reach_estimates = []
        sample_size = min(n, 200)
        sample_indices = self._rng.choice(n, sample_size, replace=False)
        for i in sample_indices:
            neighborhood = X[indices[i]]
            k_local = len(neighborhood)
            if k_local < 5:
                continue
            mu = np.mean(neighborhood, axis=0)
            centered = neighborhood - mu
            cov_local = centered.T @ centered / k_local
            eigvals = np.linalg.eigvalsh(cov_local)
            eigvals = np.sort(eigvals)[::-1]
            d = min(self.d, len(eigvals) - 1)
            if d >= 1 and d < len(eigvals):
                tangent_var = np.sum(eigvals[:d])
                normal_var = np.sum(eigvals[d:])
                if normal_var > 1e-10 and tangent_var > 1e-10:
                    r_local = np.median(distances[i, 1:k_local])
                    ratio = normal_var / max(tangent_var, 1e-12)
                    ratio = max(ratio, 1e-6)
                    reach_est = r_local / np.sqrt(ratio)
                    reach_estimates.append(reach_est)
        self._last_reach_estimates = reach_estimates
        if reach_estimates:
            return float(np.median(reach_estimates))
        return 1.0

    def _estimate_curvature(self, X):
        """
        Quick structure: one eigvalso and one lstsq for each sampling point.
        Accurately estimate the local principal curvature
        """
        n, D = X.shape
        if D >= 512:
            k = min(n, max(100, min(300, int(3 * np.sqrt(D) * np.log(max(D, 2))))))
        else:
            k = min(n, max(30, min(100, int(3 * np.sqrt(D) * np.log(max(D, 2))))))
        k = max(k, 10)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)
        curvature_estimates = []
        sample_size = min(n, 200)
        sample_indices = self._rng.choice(n, sample_size, replace=False)
        for i in sample_indices:
            neighborhood = X[indices[i]]
            k_local = len(neighborhood)
            if k_local < 5:
                continue
            mu = np.mean(neighborhood, axis=0)
            centered = neighborhood - mu
            cov_local = centered.T @ centered / k_local
            eigvals, eigvecs = np.linalg.eigh(cov_local)
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            eigvals = np.maximum(eigvals, 0)

            d_use = min(self.d, len(eigvals) - 1)
            if d_use < 1:
                continue

            T = eigvecs[:, :d_use]
            N = eigvecs[:, d_use:]

            if N.shape[1] == 0:
                continue

            # Local tangent plane coordinates (only counting the first two tangential directions)
            u = centered @ T[:, 0]
            if d_use >= 2:
                v = centered @ T[:, 1]
            else:
                v = np.zeros_like(u)

            # Normal coordinates (first normal direction)
            w_norm = centered @ N[:, 0]

            if d_use == 1:
                A = np.column_stack([u**2])
                try:
                    coef, _, _, _ = np.linalg.lstsq(A, w_norm, rcond=None)
                    kappa_est = abs(coef[0])
                except np.linalg.LinAlgError:
                    continue
            else:
                A = np.column_stack([u**2, u * v, v**2])
                try:
                    coef, _, _, _ = np.linalg.lstsq(A, w_norm, rcond=None)
                    a, b, c = coef
                    M = np.array([[a, b / 2.0], [b / 2.0, c]])
                    eigvals_M = np.linalg.eigvalsh(M)
                    kappa_est = max(abs(eigvals_M[0]), abs(eigvals_M[1]))
                except np.linalg.LinAlgError:
                    continue

            kappa_est = max(kappa_est, 1e-6)
            curvature_estimates.append(kappa_est)

        self._last_curvature_estimates = curvature_estimates
        if curvature_estimates:
            return float(np.median(curvature_estimates))
        return 1.0

    def _estimate_noise_variance(self, X):
        n, D = X.shape
        d = min(self.d, D)
        if d < D and d > 0:
            pca = PCA(n_components=d)
            X_proj = pca.fit_transform(X)
            X_recon = pca.inverse_transform(X_proj)
            residual = X - X_recon
            sigma2 = np.var(residual.flatten())
        else:
            sigma2 = np.var(X.flatten())
        return max(sigma2, 1e-6)


# ============================================================================
# compute_gdi
# ============================================================================
def compute_gdi(tau, kappa, sigma2, p, n, lambda_max=None):
    if kappa <= 0 or sigma2 <= 0 or n <= 0:
        return 0.0
    return (tau**2 * p) / (kappa**2 * sigma2 * n)


# ============================================================================
# Dataset Loaders 
# ============================================================================
class ADImageLoader:
    def __init__(self, data_root, normal_root='NonDemented', anomaly_root='ModerateDemented',
                 target_size=(32, 32)):
        from PIL import Image
        self.data_root = data_root
        self.normal_root = normal_root
        self.anomaly_root = anomaly_root
        self.target_size = target_size
        self.Image = Image

    def _load_from_folder(self, folder_path, label, prefix_filter=None):
        X, y = [], []
        if not os.path.isdir(folder_path):
            print(f"Warning: folder {folder_path} not found")
            return X, y
        all_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                    continue
                if prefix_filter and not file.startswith(prefix_filter):
                    continue
                all_files.append(os.path.join(root, file))
        all_files.sort()
        for img_path in all_files:
            try:
                img = self.Image.open(img_path).convert('L')
                img = img.resize(self.target_size)
                arr = np.array(img).flatten() / 255.0
                X.append(arr)
                y.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        return X, y

    def load_data(self):
        X_all, y_all = [], []
        normal_path = os.path.join(self.data_root, self.normal_root)
        X, y = self._load_from_folder(normal_path, 0, prefix_filter='nonDem')
        X_all.extend(X); y_all.extend(y)
        print(f"Loaded {len(X)} normal images (AD)")
        anomaly_path = os.path.join(self.data_root, self.anomaly_root)
        X, y = self._load_from_folder(anomaly_path, 1, prefix_filter='moderateDem')
        X_all.extend(X); y_all.extend(y)
        print(f"Loaded {len(X)} anomaly images (AD)")
        if len(X_all) == 0:
            raise ValueError(f"No images loaded! Check path: {self.data_root}")
        X = np.array(X_all)
        y = np.array(y_all)
        print(f"=== AD Dataset: total={len(X)}, normal={np.sum(y==0)}, anomaly={np.sum(y==1)}, rate={np.mean(y):.4%} ===")
        return X, y


class MILoader:
    def __init__(self, data_root, normal_root='normal', anomaly_root='anomaly',
                 target_size=(32, 32)):
        from PIL import Image
        self.data_root = data_root
        self.normal_root = normal_root
        self.anomaly_root = anomaly_root
        self.target_size = target_size
        self.Image = Image

    def _load_from_folder(self, folder_path, label):
        X, y = [], []
        if not os.path.isdir(folder_path):
            print(f"Warning: folder {folder_path} not found")
            return X, y
        for root, _, files in os.walk(folder_path):
            for file in files:
                if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                    continue
                img_path = os.path.join(root, file)
                try:
                    img = self.Image.open(img_path).convert('L')
                    img = img.resize(self.target_size)
                    arr = np.array(img).flatten() / 255.0
                    X.append(arr)
                    y.append(label)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        return X, y

    def load_data(self):
        X_all, y_all = [], []
        normal_path = os.path.join(self.data_root, self.normal_root)
        X, y = self._load_from_folder(normal_path, 0)
        X_all.extend(X); y_all.extend(y)
        print(f"Loaded {len(X)} normal images (MI)")
        anomaly_path = os.path.join(self.data_root, self.anomaly_root)
        X, y = self._load_from_folder(anomaly_path, 1)
        X_all.extend(X); y_all.extend(y)
        print(f"Loaded {len(X)} anomaly images (MI)")
        if len(X_all) == 0:
            raise ValueError(f"No images loaded! Check path: {self.data_root}")
        X = np.array(X_all)
        y = np.array(y_all)
        print(f"=== MI Dataset: total={len(X)}, normal={np.sum(y==0)}, anomaly={np.sum(y==1)}, rate={np.mean(y):.4%} ===")
        return X, y


class MNISTLoader:
    def __init__(self, data_root='./MNIST', target_size=(32, 32)):
        self.data_root = data_root
        self.target_size = target_size

    def load_data(self):
        from torchvision import datasets, transforms
        from PIL import Image
        train_dataset = datasets.MNIST(
            root=self.data_root, train=True, download=True,
            transform=transforms.ToTensor()
        )
        X_list, y_list = [], []
        data = train_dataset.data.numpy()
        targets = train_dataset.targets.numpy()

        for i in range(len(targets)):
            img_arr = data[i]
            if img_arr.shape != self.target_size:
                img = Image.fromarray(img_arr).resize(self.target_size)
                img_arr = np.array(img)
            arr = img_arr.flatten() / 255.0
            X_list.append(arr)
            y_list.append(1 if targets[i] == 0 else 0)

        X = np.array(X_list)
        y = np.array(y_list)
        print(f"=== MNIST Dataset: total={len(X)}, normal={np.sum(y==0)}, anomaly={np.sum(y==1)}, rate={np.mean(y):.4%} ===")
        return X, y


class IIoTDataLoader:
    def __init__(self):
        pass

    def load_and_preprocess_data(self):
        print("Loading IIoT data...")
        data = pd.read_csv('./Datasets/IOT/UKMNCT_IIoT_FDIA.csv', header=0)
        X = data.iloc[:, :-1]
        y_raw = data.iloc[:, -1]
        y = y_raw.map({'Attack': 1, 'Natural': 0})
        if y.isna().any():
            if y_raw.dtype in ['int64', 'float64']:
                y = y_raw.copy()
            else:
                y = y_raw.apply(lambda x: 1 if str(x).lower() in ['attack', '1', 'anomaly'] else 0)
        X_ = X.values.astype(float)
        y_ = y.values.flatten().astype(int)
        print(f"=== IIoT Dataset: total={len(X_)}, normal={np.sum(y_==0)}, anomaly={np.sum(y_==1)}, rate={np.mean(y_):.4%} ===")
        return X_, y_


class CreditCardLoader:
    def __init__(self, csv_path='./CC/CreditCard.csv'):
        self.csv_path = csv_path

    def load_data(self):
        print("Loading CreditCard data...")
        df = pd.read_csv(self.csv_path)

        n_before = len(df)
        df = df.dropna(subset=['Class']).reset_index(drop=True)
        v_cols = [f'V{i}' for i in range(1, 29)]
        df = df.dropna(subset=v_cols).reset_index(drop=True)
        n_dropped = n_before - len(df)
        print(f"[CreditCard] Dropped {n_dropped} rows with NaN "
              f"({n_before} -> {len(df)})")

        raw = df['Class']
        print(f"[CreditCard] Class unique (raw): {sorted(raw.unique())[:10]}")
        print(f"[CreditCard] Class dtype: {raw.dtype}, NaN count: {raw.isna().sum()}")
        y = (raw == 1).astype(int).values
        print(f"[CreditCard] bincount: {np.bincount(y)}")

        X_ = df[v_cols].values.astype(float)

        assert np.isfinite(X_).all(), "X_ still has NaN/inf!"

        print(f"=== CreditCard Dataset: total={len(X_)}, "
              f"normal={np.sum(y==0)}, anomaly={np.sum(y==1)}, "
              f"rate={np.mean(y):.4%} ===")
        return X_, y


# ============================================================================
# Dataset Configurations
# ============================================================================
def adaptive_p_from_d(d):
    if d <= 1:
        return 2
    return max(2, int(np.ceil(d * np.log(d))))

P_FIXED = {
    'AD':         adaptive_p_from_d(30),
    'MI':         adaptive_p_from_d(30),
    'IIoT':       adaptive_p_from_d(10),
    'CreditCard': adaptive_p_from_d(10),
    'MNIST':      adaptive_p_from_d(30),
}

P_RANGE = {
    'AD': [120, 240, 360, 480, 560],
    'MI': [120, 240, 360, 480, 560],
    'IIoT': [5, 10, 15, 20, 25],
    'CreditCard': [5, 10, 15, 20, 25],
    'MNIST': [120, 240, 360, 480, 560],
}

N_RANGE = {
    'AD': [100, 200, 400, 800, 1200,1500, 2000],
    'MI': [100, 200, 300, 500, 700, 800,  900],
    'IIoT': [100, 200, 500, 1000, 2000, 3000, 5000],
    'CreditCard': [1000, 2000, 3000, 5000, 7000, 9000, 10000],
    'MNIST': [1000, 3000, 5000, 10000, 20000, 30000,50000],
}

NOISE_FACTORS = [1, 2, 4, 8, 16, 32, 64]


DATASET_CONFIGS = {
    'MI': {
        'loader': MILoader,
        'loader_kwargs': {'data_root': './Datasets/MI', 'normal_root': 'normal', 'anomaly_root': 'anomaly', 'target_size': (32, 32)},
        'p_fixed': P_FIXED['MI'], 'p_range': P_RANGE['MI'], 'n_range': N_RANGE['MI'],
    },
    'AD': {
        'loader': ADImageLoader,
        'loader_kwargs': {'data_root': './Datasets/AD', 'normal_root': 'NonDemented', 'anomaly_root': 'ModerateDemented', 'target_size': (32, 32)},
        'p_fixed': P_FIXED['AD'], 'p_range': P_RANGE['AD'], 'n_range': N_RANGE['AD'],
    },
    'CreditCard': {
        'loader': CreditCardLoader,
        'loader_kwargs': {'csv_path': './Datasets/CC/CreditCard.csv'},
        'p_fixed': P_FIXED['CreditCard'], 'p_range': P_RANGE['CreditCard'], 'n_range': N_RANGE['CreditCard'],
    },
    'IIoT': {
        'loader': IIoTDataLoader,
        'loader_kwargs': {},
        'p_fixed': P_FIXED['IIoT'], 'p_range': P_RANGE['IIoT'], 'n_range': N_RANGE['IIoT'],
    },
}


def load_dataset(dataset_name):
    config = DATASET_CONFIGS[dataset_name]
    loader = config['loader'](**config['loader_kwargs'])
    if dataset_name == 'IIoT':
        data = loader.load_and_preprocess_data()
        if data is None:
            raise ValueError(f"Failed to load {dataset_name}")
        return data
    else:
        return loader.load_data()


def load_mnist():
    loader = MNISTLoader(data_root='./MNIST', target_size=(32, 32))
    return loader.load_data()