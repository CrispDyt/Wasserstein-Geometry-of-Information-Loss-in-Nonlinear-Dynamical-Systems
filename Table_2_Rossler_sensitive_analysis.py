from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Dict, Tuple, Any, List, Iterable

import numpy as np
from numba import jit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover
    Parallel = None
    delayed = None

def rossler(xyz, *, a=0.2, b=0.2, c=5.7):
    x, y, z = xyz
    return np.array([-y - z, x + a * y, b + z * (x - c)], dtype=np.float64)


def rossler_dm_y(x, y, z, a=0.2, b=0.2, c=5.7):
    return np.array([y, x + a * y, -y - z + a * x + a * a * y], dtype=np.float64)


def rossler_dm_x(x, y, z, a=0.2, b=0.2, c=5.7):
    return np.array([x, -y - z, -x - a * y - b - z * (x - c)], dtype=np.float64)


def rossler_dm_z(x, y, z, a=0.2, b=0.2, c=5.7):
    zdot = b + z * (x - c)
    return np.array([z, zdot, zdot * (x - c) + z * (-y - z)], dtype=np.float64)


def rossler_dm_yz(x, y, z, a=0.2, b=0.2, c=5.7):
    return np.array([
        y + z,
        b + x + a * y - c * z + x * z,
        -b * c + (a + b) * x + (a * a - 1) * y + (c * c - 1) * z
        - (2 * c + 1) * x * z - z * z + x * x * z,
    ], dtype=np.float64)


def rossler_dm_zx(x, y, z, a=0.2, b=0.2, c=5.7):
    return np.array([
        z + x,
        -y - z + b + z * (x - c),
        -b * (c + 1) + (b - 1) * x - a * y + c * (c + 1) * z
        + (1 - 2 * c) * x * z - y * z - z * z,
    ], dtype=np.float64)


def rossler_dm_xy(x, y, z, a=0.2, b=0.2, c=5.7):
    return np.array([
        x + y,
        x + (a - 1) * y - z,
        -b + (a - 1) * x + (a * a - a + 1) * y + (c - 1) * z - x * z,
    ], dtype=np.float64)


def rossler_y_z_y(x, y, z, a=0.2, b=0.2, c=5.7):
    return np.array([y, z, x + a * y], dtype=np.float64)


def rossler_x_y_x(x, y, z, a=0.2, b=0.2, c=5.7):
    return np.array([x, y, -y - z], dtype=np.float64)


def rossler_x_y_y(x, y, z, a=0.2, b=0.2, c=5.7):
    return np.array([x, y, x + a * y], dtype=np.float64)


def rossler_y_z_z(x, y, z, a=0.2, b=0.2, c=5.7):
    return np.array([y, z, b + z * (x - c)], dtype=np.float64)


def rossler_x_z_z(x, y, z, a=0.2, b=0.2, c=5.7):
    return np.array([x, z, b + z * (x - c)], dtype=np.float64)


def rossler_x_z_x(x, y, z, a=0.2, b=0.2, c=5.7):
    return np.array([x, z, -y - z], dtype=np.float64)


def build_embeddings_from_y(y_arr: np.ndarray) -> Dict[str, np.ndarray]:
    x, y, z = y_arr[:, 0], y_arr[:, 1], y_arr[:, 2]
    return {
        "x_x_x": rossler_dm_x(x, y, z).T,
        "y_y_y": rossler_dm_y(x, y, z).T,
        "z_z_z": rossler_dm_z(x, y, z).T,
        "y_z_y": rossler_y_z_y(x, y, z).T,
        "x_y_x": rossler_x_y_x(x, y, z).T,
        "x_y_y": rossler_x_y_y(x, y, z).T,
        "y_z_z": rossler_y_z_z(x, y, z).T,
        "x_z_z": rossler_x_z_z(x, y, z).T,
        "x_z_x": rossler_x_z_x(x, y, z).T,
        "y_plus_z": rossler_dm_yz(x, y, z).T,
        "x_plus_z": rossler_dm_zx(x, y, z).T,
        "x_plus_y": rossler_dm_xy(x, y, z).T,
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
    """Conditional variance, prediction error, centroid dispersion, pairwise dispersion."""
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

        pred_sq = 0.0
        for d in range(dim):
            diff = query_futures[i, d] - centroid[d]
            pred_sq += diff * diff
        pred_error[i] = np.sqrt(pred_sq)

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

def simulate_rossler(num_steps: int, dt: float) -> np.ndarray:
    y_clean = np.zeros((num_steps + 1, 3), dtype=np.float64)
    y_clean[0] = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    for i in range(num_steps):
        y_clean[i + 1] = y_clean[i] + rossler(y_clean[i]) * dt
    return y_clean


def add_embedding_noise(embedding: np.ndarray, noise_level: float, rng: np.random.Generator) -> np.ndarray:
    """Add coordinate-wise Gaussian noise before robust scaling.

    noise_level = 0.01 means 1% of each coordinate's empirical standard deviation.
    """
    data = np.asarray(embedding, dtype=np.float64)
    if noise_level <= 0:
        return data.copy()
    scale = np.std(data, axis=0, ddof=0)
    scale = np.where(scale > 0, scale, 1.0)
    return data + noise_level * scale * rng.normal(size=data.shape)


def robust_scale_embedding(embedding_data: np.ndarray) -> np.ndarray:
    data = np.asarray(embedding_data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return RobustScaler().fit_transform(data)


def check_embedding_condition(data: np.ndarray) -> float:
    data_centered = data - np.mean(data, axis=0)
    s = np.linalg.svd(data_centered, compute_uv=False)
    if s[-1] < 1e-12:
        return np.inf
    return float(s[0] / s[-1])


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


def estimate_on_scaled(
    data_norm: np.ndarray,
    pushforward: int,
    k: int,
    n_samples: int,
    random_state: int,
    theiler_w: int,
    buffer_mult: int,
    strict_theiler: bool = True,
) -> Dict[str, Any]:
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
    cond_var, pred_error, mean_disp, pair_disp = compute_baselines_numba(neighbor_clouds, query_futures)
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


def rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(x, dtype=np.float64)
    n = len(x)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 3:
        return np.nan
    ra = rankdata_average_ties(a[mask])
    rb = rankdata_average_ties(b[mask])
    ra = ra - np.mean(ra)
    rb = rb - np.mean(rb)
    denom = np.sqrt(np.sum(ra * ra) * np.sum(rb * rb))
    if denom == 0:
        return np.nan
    return float(np.sum(ra * rb) / denom)


def nanmean_std(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if np.all(np.isnan(arr)):
        return np.nan, np.nan
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=0))


def fmt(mean: float, std: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "nan"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

def make_sensitivity_configs() -> List[Dict[str, Any]]:
    """One-at-a-time sensitivity around the baseline setting.

    Baseline: n=20, k=50, N_Q=5000, Theiler=30, noise=0.
    The reviewer asked about sensitivity to n, k, Theiler window, sample size,
    metric/sample density, and noise. This grid is compact enough for an appendix table.
    """
    base = dict(pushforward=20, k=50, n_samples=5000, theiler_w=30, noise_level=0.0)
    configs: List[Dict[str, Any]] = []

    def add(group: str, label: str, **updates: Any) -> None:
        cfg = base.copy()
        cfg.update(updates)
        cfg["group"] = group
        cfg["config"] = label
        configs.append(cfg)

    add("baseline", "baseline")

    for k in [20, 50, 100]:
        add("k", f"k={k}", k=k)
    for n in [5, 10, 20, 40]:
        add("horizon", f"n={n}", pushforward=n)
    for w in [0, 10, 30, 60]:
        add("theiler", f"w={w}", theiler_w=w)
    for nq in [1000, 2500, 5000, 10000]:
        add("query_count", f"Nq={nq}", n_samples=nq)
    for noise in [0.0, 0.01, 0.05, 0.10]:
        add("noise", f"noise={noise:g}", noise_level=noise)

    seen = set()
    unique = []
    for cfg in configs:
        key = (cfg["pushforward"], cfg["k"], cfg["n_samples"], cfg["theiler_w"], cfg["noise_level"])
        if key in seen and cfg["config"] != "baseline":
            continue
        seen.add(key)
        unique.append(cfg)
    return unique


def run_one_task(
    embedding_name: str,
    embedding_data: np.ndarray,
    config: Dict[str, Any],
    repeat: int,
    cond_threshold: float,
    buffer_mult: int,
    strict_theiler: bool,
) -> Dict[str, Any]:
    try:
        rng = np.random.default_rng(10_000 * repeat + abs(hash((embedding_name, config["config"]))) % 10_000)
        noisy_embedding = add_embedding_noise(embedding_data, float(config["noise_level"]), rng)
        data_norm = robust_scale_embedding(noisy_embedding)
        cond_num = check_embedding_condition(data_norm)
        if (cond_num > cond_threshold) or (not np.isfinite(cond_num)):
            return {
                "embedding": embedding_name, "group": config["group"], "config": config["config"],
                "repeat": repeat, "status": "rank_deficient", "condition_number": cond_num,
                "pushforward": config["pushforward"], "k": config["k"], "n_samples": config["n_samples"],
                "theiler_w": config["theiler_w"], "noise_level": config["noise_level"],
                "E_star_k": np.nan, "CondVar": np.nan, "PredErr": np.nan, "MeanDisp": np.nan,
                "NeighborDisp": np.nan, "median_rk": np.nan, "q90_rk": np.nan, "mean_rk": np.nan,
            }
        out = estimate_on_scaled(
            data_norm=data_norm,
            pushforward=int(config["pushforward"]),
            k=int(config["k"]),
            n_samples=int(config["n_samples"]),
            random_state=repeat,
            theiler_w=int(config["theiler_w"]),
            buffer_mult=buffer_mult,
            strict_theiler=strict_theiler,
        )
        out.update({
            "embedding": embedding_name, "group": config["group"], "config": config["config"],
            "repeat": repeat, "status": "ok", "condition_number": cond_num,
            "pushforward": config["pushforward"], "k": config["k"], "n_samples": config["n_samples"],
            "theiler_w": config["theiler_w"], "noise_level": config["noise_level"],
        })
        return out
    except Exception as exc:
        return {
            "embedding": embedding_name, "group": config["group"], "config": config["config"],
            "repeat": repeat, "status": f"failed: {type(exc).__name__}: {exc}",
            "condition_number": np.nan, "pushforward": config["pushforward"], "k": config["k"],
            "n_samples": config["n_samples"], "theiler_w": config["theiler_w"],
            "noise_level": config["noise_level"], "E_star_k": np.nan, "CondVar": np.nan,
            "PredErr": np.nan, "MeanDisp": np.nan, "NeighborDisp": np.nan,
            "median_rk": np.nan, "q90_rk": np.nan, "mean_rk": np.nan,
        }


def summarize_runs(rows: List[Dict[str, Any]], embeddings_order: List[str], configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metrics = ["E_star_k", "CondVar", "PredErr", "MeanDisp", "NeighborDisp", "median_rk", "q90_rk", "mean_rk"]
    summary: List[Dict[str, Any]] = []
    for cfg in configs:
        for emb in embeddings_order:
            sub = [r for r in rows if r["embedding"] == emb and r["config"] == cfg["config"]]
            valid = sum(1 for r in sub if r["status"] == "ok")
            row: Dict[str, Any] = {
                "embedding": emb, "group": cfg["group"], "config": cfg["config"],
                "valid_runs": valid, "total_runs": len(sub),
                "pushforward": cfg["pushforward"], "k": cfg["k"], "n_samples": cfg["n_samples"],
                "theiler_w": cfg["theiler_w"], "noise_level": cfg["noise_level"],
            }
            for metric in metrics:
                mean, std = nanmean_std([r[metric] for r in sub])
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
            summary.append(row)
    return summary


def rank_stability_summary(summary: List[Dict[str, Any]], configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Baseline ranking of E* across embeddings.
    baseline = [r for r in summary if r["config"] == "baseline"]
    baseline_by_embedding = {r["embedding"]: r["E_star_k_mean"] for r in baseline}
    embeddings = list(baseline_by_embedding.keys())
    baseline_vals = np.array([baseline_by_embedding[e] for e in embeddings], dtype=np.float64)

    out_rows: List[Dict[str, Any]] = []
    for cfg in configs:
        rows = [r for r in summary if r["config"] == cfg["config"]]
        by_embedding = {r["embedding"]: r for r in rows}
        vals = np.array([by_embedding[e]["E_star_k_mean"] for e in embeddings], dtype=np.float64)
        rho = spearman_corr(baseline_vals, vals)
        rel = np.abs(vals - baseline_vals) / np.maximum(np.abs(baseline_vals), 1e-12)
        mask = np.isfinite(rel)
        out_rows.append({
            "group": cfg["group"], "config": cfg["config"],
            "spearman_vs_baseline": rho,
            "median_relative_change_E": float(np.nanmedian(rel[mask])) if np.any(mask) else np.nan,
            "max_relative_change_E": float(np.nanmax(rel[mask])) if np.any(mask) else np.nan,
        })
    return out_rows


def main() -> None:
    # Global settings.
    DT = 0.01
    NUM_STEPS = int(2e4)
    COND_THRESHOLD = 1000.0
    BUFFER_MULT = 6
    STRICT_THEILER = True
    N_REPEATS = 10  
    N_JOBS = -1

    output_dir = Path("rossler_sensitivity_outputs")
    output_dir.mkdir(exist_ok=True)

    trajectory = simulate_rossler(NUM_STEPS, DT)
    embeddings = build_embeddings_from_y(trajectory)
    embeddings_order = list(embeddings.keys())
    configs = make_sensitivity_configs()

    # warm up 
    dummy_cloud = np.random.default_rng(123).normal(size=(2, 50, 3))
    dummy_future = np.random.default_rng(456).normal(size=(2, 3))
    _ = compute_errors_numba(dummy_cloud)
    _ = compute_baselines_numba(dummy_cloud, dummy_future)

    tasks = []
    for cfg in configs:
        for emb_name, emb_data in embeddings.items():
            for rep in range(N_REPEATS):
                tasks.append((emb_name, emb_data, cfg, rep))

    if Parallel is None:
        warnings.warn("joblib unavailable; running sequentially.")
        all_rows = [
            run_one_task(e, d, c, r, COND_THRESHOLD, BUFFER_MULT, STRICT_THEILER)
            for e, d, c, r in tasks
        ]
    else:
        all_rows = Parallel(n_jobs=N_JOBS, verbose=10, prefer="processes")(
            delayed(run_one_task)(e, d, c, r, COND_THRESHOLD, BUFFER_MULT, STRICT_THEILER)
            for e, d, c, r in tasks
        )

    config_order = {cfg["config"]: i for i, cfg in enumerate(configs)}
    embedding_order = {e: i for i, e in enumerate(embeddings_order)}
    all_rows.sort(key=lambda r: (config_order[r["config"]], embedding_order[r["embedding"]], int(r["repeat"])))

    run_fields = [
        "group", "config", "embedding", "repeat", "status", "condition_number",
        "pushforward", "k", "n_samples", "theiler_w", "noise_level",
        "E_star_k", "CondVar", "PredErr", "MeanDisp", "NeighborDisp",
        "median_rk", "q90_rk", "mean_rk", "n_queries", "not_enough_queries",
    ]
    write_csv(output_dir / "rossler_sensitivity_all_runs.csv", all_rows, run_fields)

    summary = summarize_runs(all_rows, embeddings_order, configs)
    summary_fields = [
        "group", "config", "embedding", "valid_runs", "total_runs",
        "pushforward", "k", "n_samples", "theiler_w", "noise_level",
    ]
    for metric in ["E_star_k", "CondVar", "PredErr", "MeanDisp", "NeighborDisp", "median_rk", "q90_rk", "mean_rk"]:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(output_dir / "rossler_sensitivity_summary_by_embedding.csv", summary, summary_fields)

    stability = rank_stability_summary(summary, configs)
    write_csv(output_dir / "rossler_sensitivity_rank_stability.csv", stability, [
        "group", "config", "spearman_vs_baseline", "median_relative_change_E", "max_relative_change_E",
    ])

    print("\nRank stability of E* relative to the baseline setting")
    print(f"{'Group':<12} | {'Config':<14} | {'Spearman':>9} | {'Median rel. change':>18} | {'Max rel. change':>15}")
    print("-" * 82)
    for r in stability:
        print(
            f"{r['group']:<12} | {r['config']:<14} | "
            f"{r['spearman_vs_baseline']:>9.3f} | "
            f"{r['median_relative_change_E']:>18.3f} | "
            f"{r['max_relative_change_E']:>15.3f}"
        )

    print(f"\nSaved per-run results to: {output_dir / 'rossler_sensitivity_all_runs.csv'}")
    print(f"Saved embedding-level summary to: {output_dir / 'rossler_sensitivity_summary_by_embedding.csv'}")
    print(f"Saved rank-stability summary to: {output_dir / 'rossler_sensitivity_rank_stability.csv'}")


if __name__ == "__main__":
    main()
