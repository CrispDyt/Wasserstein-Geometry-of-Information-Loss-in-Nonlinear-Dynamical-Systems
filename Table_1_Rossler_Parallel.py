from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Dict, Tuple, Any, List

import numpy as np
from numba import jit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

try:
    from joblib import Parallel, delayed
except Exception: 
    Parallel = None
    delayed = None

def rossler(xyz, *, a=0.2, b=0.2, c=5.7):
    x, y, z = xyz
    x_dot = -y - z
    y_dot = x + a * y
    z_dot = b + z * (x - c)
    return np.array([x_dot, y_dot, z_dot], dtype=np.float64)


def rossler_dm_y(x, y, z, a=0.2, b=0.2, c=5.7):
    x_dot = y
    y_dot = x + a * y
    z_dot = -y - z + a * x + a * a * y
    return np.array([x_dot, y_dot, z_dot], dtype=np.float64)


def rossler_dm_x(x, y, z, a=0.2, b=0.2, c=5.7):
    x_dot = x
    y_dot = -y - z
    z_dot = -x - a * y - b - z * (x - c)
    return np.array([x_dot, y_dot, z_dot], dtype=np.float64)


def rossler_dm_z(x, y, z, a=0.2, b=0.2, c=5.7):
    x_dot = z
    y_dot = b + z * (x - c)
    z_dot = (b + z * (x - c)) * (x - c) + z * (-y - z)
    return np.array([x_dot, y_dot, z_dot], dtype=np.float64)


def rossler_dm_yz(x, y, z, a=0.2, b=0.2, c=5.7):
    x_dot = y + z
    y_dot = b + x + a * y - c * z + x * z
    z_dot = (
        -b * c + (a + b) * x + (a * a - 1) * y + (c * c - 1) * z
        - (2 * c + 1) * x * z - z * z + x * x * z
    )
    return np.array([x_dot, y_dot, z_dot], dtype=np.float64)


def rossler_dm_zx(x, y, z, a=0.2, b=0.2, c=5.7):
    x_dot = z + x
    y_dot = -y - z + b + z * (x - c)
    z_dot = (
        -b * (c + 1) + (b - 1) * x - a * y + c * (c + 1) * z
        + (1 - 2 * c) * x * z - y * z - z * z
    )
    return np.array([x_dot, y_dot, z_dot], dtype=np.float64)


def rossler_dm_xy(x, y, z, a=0.2, b=0.2, c=5.7):
    x_dot = x + y
    y_dot = x + (a - 1) * y - z
    z_dot = -b + (a - 1) * x + (a * a - a + 1) * y + (c - 1) * z - x * z
    return np.array([x_dot, y_dot, z_dot], dtype=np.float64)


def rossler_y_z_y(x, y, z, a=0.2, b=0.2, c=5.7):
    x_dot = y
    y_dot = z
    z_dot = x + a * y
    return np.array([x_dot, y_dot, z_dot], dtype=np.float64)


def rossler_x_y_x(x, y, z, a=0.2, b=0.2, c=5.7):
    x_dot = x
    y_dot = y
    z_dot = -y - z
    return np.array([x_dot, y_dot, z_dot], dtype=np.float64)


def rossler_x_y_y(x, y, z, a=0.2, b=0.2, c=5.7):
    x_dot = x
    y_dot = y
    z_dot = x + a * y
    return np.array([x_dot, y_dot, z_dot], dtype=np.float64)


def rossler_y_z_z(x, y, z, a=0.2, b=0.2, c=5.7):
    x_dot = y
    y_dot = z
    z_dot = b + z * (x - c)
    return np.array([x_dot, y_dot, z_dot], dtype=np.float64)


def rossler_x_z_z(x, y, z, a=0.2, b=0.2, c=5.7):
    x_dot = x
    y_dot = z
    z_dot = b + z * (x - c)
    return np.array([x_dot, y_dot, z_dot], dtype=np.float64)


def rossler_x_z_x(x, y, z, a=0.2, b=0.2, c=5.7):
    x_dot = x
    y_dot = z
    z_dot = -y - z
    return np.array([x_dot, y_dot, z_dot], dtype=np.float64)


def build_embeddings_from_y(y_arr: np.ndarray) -> Dict[str, np.ndarray]:
    x_x_x = rossler_dm_x(y_arr[:, 0], y_arr[:, 1], y_arr[:, 2]).T
    y_y_y = rossler_dm_y(y_arr[:, 0], y_arr[:, 1], y_arr[:, 2]).T
    z_z_z = rossler_dm_z(y_arr[:, 0], y_arr[:, 1], y_arr[:, 2]).T
    y_z_y = rossler_y_z_y(y_arr[:, 0], y_arr[:, 1], y_arr[:, 2]).T
    x_y_x = rossler_x_y_x(y_arr[:, 0], y_arr[:, 1], y_arr[:, 2]).T
    x_y_y = rossler_x_y_y(y_arr[:, 0], y_arr[:, 1], y_arr[:, 2]).T
    y_z_z = rossler_y_z_z(y_arr[:, 0], y_arr[:, 1], y_arr[:, 2]).T
    x_z_z = rossler_x_z_z(y_arr[:, 0], y_arr[:, 1], y_arr[:, 2]).T
    x_z_x = rossler_x_z_x(y_arr[:, 0], y_arr[:, 1], y_arr[:, 2]).T
    y_plus_z = rossler_dm_yz(y_arr[:, 0], y_arr[:, 1], y_arr[:, 2]).T
    x_plus_z = rossler_dm_zx(y_arr[:, 0], y_arr[:, 1], y_arr[:, 2]).T
    x_plus_y = rossler_dm_xy(y_arr[:, 0], y_arr[:, 1], y_arr[:, 2]).T

    return {
        "x_x_x": x_x_x,
        "y_y_y": y_y_y,
        "z_z_z": z_z_z,
        "y_z_y": y_z_y,
        "x_y_x": x_y_x,
        "x_y_y": x_y_y,
        "y_z_z": y_z_z,
        "x_z_z": x_z_z,
        "x_z_x": x_z_x,
        "y_plus_z": y_plus_z,
        "x_plus_z": x_plus_z,
        "x_plus_y": x_plus_y,
    }


@jit(nopython=True, fastmath=True)
def compute_errors_numba(neighbor_clouds, max_iter=50, eps=1e-5):
    """Pointwise geometric-median loss for each future cloud."""
    n_samples, k, dim = neighbor_clouds.shape
    errors = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        cloud = neighbor_clouds[i]

        y = np.zeros(dim)
        for d in range(dim):
            s = 0.0
            for j in range(k):
                s += cloud[j, d]
            y[d] = s / k

        for _ in range(max_iter):
            sum_weights = 0.0
            y_next = np.zeros(dim)
            all_non_zero = True

            for j in range(k):
                dist_sq = 0.0
                for d in range(dim):
                    diff = cloud[j, d] - y[d]
                    dist_sq += diff * diff
                dist = np.sqrt(dist_sq)

                if dist < 1e-10:
                    all_non_zero = False
                    for d in range(dim):
                        y[d] = cloud[j, d]
                    break

                w = 1.0 / dist
                sum_weights += w
                for d in range(dim):
                    y_next[d] += cloud[j, d] * w

            if not all_non_zero:
                break

            diff_norm_sq = 0.0
            for d in range(dim):
                y_next[d] /= sum_weights
                diff = y[d] - y_next[d]
                diff_norm_sq += diff * diff
                y[d] = y_next[d]

            if np.sqrt(diff_norm_sq) < eps:
                break

        total_dist = 0.0
        for j in range(k):
            dist_sq = 0.0
            for d in range(dim):
                diff = cloud[j, d] - y[d]
                dist_sq += diff * diff
            total_dist += np.sqrt(dist_sq)

        errors[i] = total_dist / k

    return errors


@jit(nopython=True, fastmath=True)
def compute_baselines_numba(neighbor_clouds, query_futures):
    """
    Baselines for each query.

    cond_var_trace[q]
        Empirical conditional variance trace:
        (1/k) sum_j ||Y_j - mean(Y)||^2.

    pred_error[q]
        Short-horizon local-constant prediction error:
        ||X_{q+n} - mean(Y)||.

    mean_dispersion[q]
        Mean distance from neighbour futures to their centroid:
        (1/k) sum_j ||Y_j - mean(Y)||.
        This is not the prediction error; it is included for reference.

    pair_dispersion[q]
        Mean pairwise distance among neighbour futures.
    """
    n_samples, k, dim = neighbor_clouds.shape
    cond_var_trace = np.empty(n_samples, dtype=np.float64)
    pred_error = np.empty(n_samples, dtype=np.float64)
    mean_dispersion = np.empty(n_samples, dtype=np.float64)
    pair_dispersion = np.empty(n_samples, dtype=np.float64)

    for i in range(n_samples):
        cloud = neighbor_clouds[i]

        centroid = np.zeros(dim)
        for d in range(dim):
            s = 0.0
            for j in range(k):
                s += cloud[j, d]
            centroid[d] = s / k

        # Conditional variance trace 
        var_sum = 0.0
        dist_sum = 0.0
        for j in range(k):
            dist_sq = 0.0
            for d in range(dim):
                diff = cloud[j, d] - centroid[d]
                dist_sq += diff * diff
            var_sum += dist_sq
            dist_sum += np.sqrt(dist_sq)
        cond_var_trace[i] = var_sum / k
        mean_dispersion[i] = dist_sum / k

        # Local-constant prediction error 
        pred_sq = 0.0
        for d in range(dim):
            diff = query_futures[i, d] - centroid[d]
            pred_sq += diff * diff
        pred_error[i] = np.sqrt(pred_sq)

        # Mean pairwise neighbour-future dispersion.
        pair_sum = 0.0
        pair_count = 0
        for a in range(k):
            for b in range(a + 1, k):
                dist_sq = 0.0
                for d in range(dim):
                    diff = cloud[a, d] - cloud[b, d]
                    dist_sq += diff * diff
                pair_sum += np.sqrt(dist_sq)
                pair_count += 1
        pair_dispersion[i] = pair_sum / pair_count

    return cond_var_trace, pred_error, mean_dispersion, pair_dispersion

def check_embedding_condition(data: np.ndarray) -> float:
    data_centered = data - np.mean(data, axis=0)
    s = np.linalg.svd(data_centered, compute_uv=False)
    if s[-1] < 1e-12:
        return np.inf
    return float(s[0] / s[-1])


def robust_scale_embedding(embedding_data: np.ndarray) -> np.ndarray:
    data = np.asarray(embedding_data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    scaler = RobustScaler()
    return scaler.fit_transform(data)


def select_knn_indices(
    data_norm: np.ndarray,
    pushforward: int,
    k: int,
    n_samples: int,
    random_state: int,
    theiler_w: int,
    buffer_mult: int,
    strict_theiler: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if k <= 1:
        raise ValueError("k must be >= 2.")
    if pushforward <= 0:
        raise ValueError("pushforward must be >= 1.")
    if theiler_w < 0:
        raise ValueError("theiler_w must be >= 0.")
    if buffer_mult < 1:
        raise ValueError("buffer_mult must be >= 1.")

    T = data_norm.shape[0]
    max_start = T - pushforward
    if max_start <= k:
        raise ValueError("Time series is too short for the requested horizon and k.")

    X_curr = data_norm[:max_start]
    rng = np.random.default_rng(random_state)
    n_q = int(min(n_samples, max_start))
    query_idx = rng.choice(max_start, size=n_q, replace=False)

    k_query = int(min(max_start, max(k * buffer_mult, k + 2 * theiler_w + 5)))
    nbrs = NearestNeighbors(n_neighbors=k_query, algorithm="auto", n_jobs=1).fit(X_curr)
    dist_raw, idx_raw = nbrs.kneighbors(X_curr[query_idx])

    nn_idx = np.empty((n_q, k), dtype=np.int64)
    rk = np.empty(n_q, dtype=np.float64)
    not_enough = 0

    for r in range(n_q):
        q = int(query_idx[r])
        cand = idx_raw[r]
        cand_d = dist_raw[r]

        picked = 0
        last_d = 0.0
        for j in range(k_query):
            t = int(cand[j])
            if t == q:
                continue
            if theiler_w > 0 and abs(t - q) <= theiler_w:
                continue
            nn_idx[r, picked] = t
            last_d = float(cand_d[j])
            picked += 1
            if picked == k:
                break

        if picked < k:
            not_enough += 1
            if strict_theiler:
                raise RuntimeError("Not enough valid neighbours after Theiler exclusion.")
            picked = 0
            last_d = 0.0
            for j in range(k_query):
                t = int(cand[j])
                if t == q:
                    continue
                nn_idx[r, picked] = t
                last_d = float(cand_d[j])
                picked += 1
                if picked == k:
                    break
            if picked < k:
                raise RuntimeError("Not enough valid neighbours even after fallback.")

        rk[r] = last_d

    return query_idx, nn_idx, rk, not_enough


def estimate_intrinsic_and_baselines_on_scaled(
    data_norm: np.ndarray,
    pushforward: int,
    k: int,
    n_samples: int,
    random_state: int,
    theiler_w: int = 0,
    buffer_mult: int = 6,
    strict_theiler: bool = True,
) -> Dict[str, Any]:
    """Estimate E* and baseline metrics on already robust-scaled coordinates."""
    max_start = data_norm.shape[0] - pushforward
    Y_fut = data_norm[pushforward:]

    query_idx, nn_idx, rk, not_enough = select_knn_indices(
        data_norm=data_norm,
        pushforward=pushforward,
        k=k,
        n_samples=n_samples,
        random_state=random_state,
        theiler_w=theiler_w,
        buffer_mult=buffer_mult,
        strict_theiler=strict_theiler,
    )

    neighbor_clouds = Y_fut[nn_idx]
    query_futures = Y_fut[query_idx]

    local_e_star = compute_errors_numba(neighbor_clouds)
    cond_var, pred_error, mean_disp, pair_disp = compute_baselines_numba(
        neighbor_clouds, query_futures
    )

    return {
        "E_star_k": float(np.mean(local_e_star)),
        "CondVar": float(np.mean(cond_var)),
        "PredErr": float(np.mean(pred_error)),
        "MeanDisp": float(np.mean(mean_disp)),
        "NeighborDisp": float(np.mean(pair_disp)),
        "median_rk": float(np.median(rk)),
        "q90_rk": float(np.quantile(rk, 0.90)),
        "mean_rk": float(np.mean(rk)),
        "n_queries": int(len(query_idx)),
        "not_enough_queries": int(not_enough),
    }


def run_one_repeat(
    name: str,
    data_norm: np.ndarray,
    condition_number: float,
    rep: int,
    pushforward: int,
    k: int,
    n_samples: int,
    theiler_w: int,
    buffer_mult: int,
    strict_theiler: bool,
) -> Dict[str, Any]:
    try:
        out = estimate_intrinsic_and_baselines_on_scaled(
            data_norm=data_norm,
            pushforward=pushforward,
            k=k,
            n_samples=n_samples,
            random_state=rep,
            theiler_w=theiler_w,
            buffer_mult=buffer_mult,
            strict_theiler=strict_theiler,
        )
        out.update({
            "embedding": name,
            "repeat": rep,
            "condition_number": float(condition_number),
            "status": "ok",
        })
        return out
    except Exception as exc:
        return {
            "embedding": name,
            "repeat": rep,
            "condition_number": float(condition_number),
            "status": f"failed: {type(exc).__name__}: {exc}",
            "E_star_k": np.nan,
            "CondVar": np.nan,
            "PredErr": np.nan,
            "MeanDisp": np.nan,
            "NeighborDisp": np.nan,
            "median_rk": np.nan,
            "q90_rk": np.nan,
            "mean_rk": np.nan,
            "n_queries": 0,
            "not_enough_queries": np.nan,
        }


def nanmean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return np.nan, np.nan
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=0))


def fmt_mean_std(mean: float, std: float) -> str:
    if np.isnan(mean):
        return "nan"
    return f"{mean:.4f} ± {std:.4f}"


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

def main() -> None:
    # Experiment parameters.
    DT = 0.01
    NUM_STEPS = int(2e4)
    PUSHFORWARD = 20
    K = 50
    N_SAMPLES = 5000
    COND_THRESHOLD = 1000.0
    THEILER_W = 30
    BUFFER_MULT = 6
    N_REPEATS = 30
    N_JOBS = -1 
    STRICT_THEILER = True

    output_dir = Path("rossler_baseline_outputs")
    output_dir.mkdir(exist_ok=True)

    y_clean = np.zeros((NUM_STEPS + 1, 3), dtype=np.float64)
    y_clean[0] = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    for i in range(NUM_STEPS):
        y_clean[i + 1] = y_clean[i] + rossler(y_clean[i]) * DT

    embeddings_clean = build_embeddings_from_y(y_clean)

    scaled_embeddings: Dict[str, Tuple[np.ndarray, float, bool]] = {}
    for name, emb in embeddings_clean.items():
        data_norm = robust_scale_embedding(emb)
        cond_num = check_embedding_condition(data_norm)
        rank_ok = (cond_num <= COND_THRESHOLD) and np.isfinite(cond_num)
        if not rank_ok:
            warnings.warn(f"{name}: rank deficient or ill-conditioned; cond={cond_num}")
        scaled_embeddings[name] = (data_norm, cond_num, rank_ok)

    dummy_cloud = np.random.default_rng(123).normal(size=(2, K, 3))
    dummy_future = np.random.default_rng(456).normal(size=(2, 3))
    _ = compute_errors_numba(dummy_cloud)
    _ = compute_baselines_numba(dummy_cloud, dummy_future)

    tasks = []
    all_run_rows: List[Dict[str, Any]] = []

    for name, (data_norm, cond_num, rank_ok) in scaled_embeddings.items():
        if not rank_ok:
            for rep in range(N_REPEATS):
                all_run_rows.append({
                    "embedding": name,
                    "repeat": rep,
                    "condition_number": float(cond_num),
                    "status": "rank_deficient",
                    "E_star_k": np.nan,
                    "CondVar": np.nan,
                    "PredErr": np.nan,
                    "MeanDisp": np.nan,
                    "NeighborDisp": np.nan,
                    "median_rk": np.nan,
                    "q90_rk": np.nan,
                    "mean_rk": np.nan,
                    "n_queries": 0,
                    "not_enough_queries": np.nan,
                })
            continue

        for rep in range(N_REPEATS):
            tasks.append((name, data_norm, cond_num, rep))

    if tasks:
        if Parallel is None:
            warnings.warn("joblib is unavailable; running sequentially.")
            run_rows = [
                run_one_repeat(
                    name=name,
                    data_norm=data_norm,
                    condition_number=cond_num,
                    rep=rep,
                    pushforward=PUSHFORWARD,
                    k=K,
                    n_samples=N_SAMPLES,
                    theiler_w=THEILER_W,
                    buffer_mult=BUFFER_MULT,
                    strict_theiler=STRICT_THEILER,
                )
                for name, data_norm, cond_num, rep in tasks
            ]
        else:
            run_rows = Parallel(n_jobs=N_JOBS, verbose=10, prefer="processes")(
                delayed(run_one_repeat)(
                    name=name,
                    data_norm=data_norm,
                    condition_number=cond_num,
                    rep=rep,
                    pushforward=PUSHFORWARD,
                    k=K,
                    n_samples=N_SAMPLES,
                    theiler_w=THEILER_W,
                    buffer_mult=BUFFER_MULT,
                    strict_theiler=STRICT_THEILER,
                )
                for name, data_norm, cond_num, rep in tasks
            )
        all_run_rows.extend(run_rows)

    order = list(embeddings_clean.keys())
    all_run_rows.sort(key=lambda r: (order.index(r["embedding"]), int(r["repeat"])))

    metrics = [
        "E_star_k", "CondVar", "PredErr", "MeanDisp", "NeighborDisp",
        "median_rk", "q90_rk", "mean_rk",
    ]

    summary_rows: List[Dict[str, Any]] = []
    for name in order:
        rows = [r for r in all_run_rows if r["embedding"] == name]
        valid = sum(1 for r in rows if r["status"] == "ok")
        row: Dict[str, Any] = {
            "embedding": name,
            "valid_runs": valid,
            "total_runs": N_REPEATS,
            "condition_number": rows[0]["condition_number"] if rows else np.nan,
        }
        for metric in metrics:
            mean, std = nanmean_std([r[metric] for r in rows])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        summary_rows.append(row)

    run_fields = [
        "embedding", "repeat", "status", "condition_number", "E_star_k",
        "CondVar", "PredErr", "MeanDisp", "NeighborDisp",
        "median_rk", "q90_rk", "mean_rk", "n_queries", "not_enough_queries",
    ]
    summary_fields = ["embedding", "valid_runs", "total_runs", "condition_number"]
    for metric in metrics:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std"])

    write_csv(output_dir / "rossler_baseline_all_runs.csv", all_run_rows, run_fields)
    write_csv(output_dir / "rossler_baseline_summary_30runs.csv", summary_rows, summary_fields)

    print("\n30-repeat summary. All metrics are computed in robust-scaled coordinates.")
    print(
        f"{'Embedding':<12} | {'Valid':<7} | {'E*':<17} | {'CondVar':<17} | "
        f"{'PredErr':<17} | {'MeanDisp':<17} | {'NeighborDisp':<17}"
    )
    print("-" * 122)
    for row in summary_rows:
        print(
            f"{row['embedding']:<12} | "
            f"{int(row['valid_runs']):>2}/{int(row['total_runs']):<4} | "
            f"{fmt_mean_std(row['E_star_k_mean'], row['E_star_k_std']):<17} | "
            f"{fmt_mean_std(row['CondVar_mean'], row['CondVar_std']):<17} | "
            f"{fmt_mean_std(row['PredErr_mean'], row['PredErr_std']):<17} | "
            f"{fmt_mean_std(row['MeanDisp_mean'], row['MeanDisp_std']):<17} | "
            f"{fmt_mean_std(row['NeighborDisp_mean'], row['NeighborDisp_std']):<17}"
        )

    print(f"\nSaved per-run results to: {output_dir / 'rossler_baseline_all_runs.csv'}")
    print(f"Saved summary results to: {output_dir / 'rossler_baseline_summary_30runs.csv'}")


if __name__ == "__main__":
    main()
