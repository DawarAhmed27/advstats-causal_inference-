"""Run: python -m tests.test_stats   (validates the statistics code on synthetic data)"""
import numpy as np
from src import stats as S
from tests.synth import make_tasks, make_runs

def coverage_of_intervals(n=58, p=0.25, reps=1500, seed=1):
    rng = np.random.default_rng(seed); hit = {"wilson": 0, "beta": 0, "boot": 0, "clt": 0}
    for _ in range(reps):
        x = (rng.random(n) < p).astype(float); k = int(x.sum())
        lo, hi = S.wilson(k, n); hit["wilson"] += lo <= p <= hi
        lo, hi = S.beta_credible(k, n); hit["beta"] += lo <= p <= hi
        lo, hi = S.cluster_bootstrap(x, B=400, seed=int(rng.integers(1 << 30))); hit["boot"] += lo <= p <= hi
        m = x.mean(); se = np.sqrt(m * (1 - m) / n) if 0 < m < 1 else 0; hit["clt"] += (m - 1.96 * se) <= p <= (m + 1.96 * se)
    return {k: v / reps for k, v in hit.items()}

def test_conformal(alpha=0.1, seeds=20):
    covs, sizes = [], []
    for s in range(seeds):
        tasks = make_tasks(seed=s); T = [t for t in tasks if t["subset"] == "hand"] + [t for t in tasks if t["subset"] == "algo"][:60]
        runs = make_runs(T, "m", 0.35, R=10, seed=s)
        by = {}; [by.setdefault(r["task_id"], []).append(r["pred_step"]) for r in runs]
        L = {t["id"]: t["n_steps"] for t in T}; y = {t["id"]: t["true_step"] for t in T}
        P = S.vote_matrix(by, L, smooth=0.05)
        out = S.conformal_sets(P, y, alpha=alpha, K=5, contiguous=True, seed=s)
        covs.append(np.mean([c for c, _ in out.values()])); sizes.append(np.mean([z / L[t] for t, (_, z) in out.items()]))
    return float(np.mean(covs)), float(np.mean(sizes))

if __name__ == "__main__":
    print("CI coverage at n=58, true p=0.25, nominal 0.95:", coverage_of_intervals())
    c, z = test_conformal(); print(f"conformal (alpha=0.1): empirical coverage {c:.3f}, mean set size as share of log {z:.3f}")
    rng = np.random.default_rng(0); a = rng.random(58) < .3; b = rng.random(58) < .3
    print("paired diff (null):", S.paired_diff(a.astype(float), b.astype(float)))
    print("holm:", S.holm([0.01, 0.04, 0.03]))
    print("min detectable diff, sd=0.4, n=58:", round(S.min_detectable_diff(0.4, 58), 3))
