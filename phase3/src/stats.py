"""Statistical toolkit: metrics, intervals, paired tests, conformal sets, label noise, power."""
import numpy as np
from scipy import stats as st

# ---------- metrics ----------
def per_task_scores(runs, key="pred_step", truth="true_step", tol=0):
    """runs: list of run dicts for ONE method. Returns {task_id: mean correctness over its runs}."""
    acc = {}
    for r in runs:
        acc.setdefault(r["task_id"], []).append(float(r[key] is not None and abs(r[key] - r[truth]) <= tol))
    return {k: float(np.mean(v)) for k, v in acc.items()}

def chance_adjusted(acc, rand):
    """(acc - chance) / (1 - chance): 0 = random guessing, 1 = perfect."""
    return (acc - rand) / (1 - rand) if rand < 1 else float("nan")

# ---------- intervals ----------
def wilson(k, n, conf=0.95):
    if n == 0: return (float("nan"),) * 2
    z = st.norm.ppf(1 - (1 - conf) / 2); p = k / n
    d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)

def beta_credible(k, n, conf=0.95, a=1.0, b=1.0):
    lo, hi = st.beta.ppf([(1 - conf) / 2, 1 - (1 - conf) / 2], a + k, b + n - k)
    return (float(lo), float(hi))

def cluster_bootstrap(x, B=10000, conf=0.95, seed=0, stat=np.mean):
    """x: per-task scores (task = the resampling unit, so run-to-run noise is nested inside)."""
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    bs = stat(x[idx], axis=1)
    a = (1 - conf) / 2
    return (float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a)))

def paired_diff(xa, xb, B=10000, conf=0.95, seed=0):
    """Paired difference A-B over the same tasks: bootstrap CI + sign-flip permutation p-value."""
    d = np.asarray(xa, float) - np.asarray(xb, float); rng = np.random.default_rng(seed)
    lo, hi = cluster_bootstrap(d, B, conf, seed)
    signs = rng.choice([-1, 1], size=(B, len(d)))
    perm = np.abs((signs * d).mean(1)); obs = abs(d.mean())
    p = (1 + (perm >= obs - 1e-15).sum()) / (B + 1)
    return {"diff": float(d.mean()), "lo": lo, "hi": hi, "p": float(p)}

def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1, run)
    return adj.tolist()

# ---------- reliability across repeated runs ----------
def run_consistency(runs):
    """Share of tasks where all runs give the same predicted step, and mean pairwise agreement."""
    by = {}
    for r in runs: by.setdefault(r["task_id"], []).append(r["pred_step"])
    same = np.mean([len(set(v)) == 1 for v in by.values() if len(v) > 1])
    pair = []
    for v in by.values():
        if len(v) > 1:
            pair.append(np.mean([a == b for i, a in enumerate(v) for b in v[i + 1:]]))
    return {"all_same": float(same), "pairwise_agree": float(np.mean(pair))}

# ---------- vote distributions & conformal sets ----------
def vote_matrix(runs_by_task, n_steps_by_task, smooth=0.0):
    """Returns {task: vector p over steps} from repeated-run predictions."""
    P = {}
    for t, preds in runs_by_task.items():
        L = n_steps_by_task[t]; p = np.full(L, smooth, float)
        for s in preds:
            if s is not None and not (isinstance(s, float) and np.isnan(s)) and 0 <= int(s) < L: p[int(s)] += 1
        P[t] = p / p.sum() if p.sum() > 0 else np.full(L, 1 / L)
    return P

def conformal_sets(P, truth, alpha=0.1, K=5, contiguous=True, seed=0, tol=0):
    """Cross-conformal prediction sets with held-out evaluation.
    Score s = 1 - p_true(x). For each fold, the quantile comes from the other folds, and
    coverage is measured on the held-out fold. Returns per-task (covered, set_size)."""
    ids = list(P); rng = np.random.default_rng(seed); rng.shuffle(ids)
    folds = [ids[i::K] for i in range(K)]; out = {}
    for k in range(K):
        cal = [t for j in range(K) if j != k for t in folds[j]]
        s = np.array([1 - P[t][truth[t]] for t in cal]); n = len(s)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        q = np.quantile(s, level, method="higher")
        for t in folds[k]:
            keep = np.where(1 - P[t] <= q + 1e-12)[0]
            if len(keep) == 0: keep = np.array([int(np.argmax(P[t]))])
            if contiguous: keep = np.arange(keep.min(), keep.max() + 1)
            out[t] = (bool(np.any(np.abs(keep - truth[t]) <= tol)), int(len(keep)))
    return out

# ---------- label noise ----------
def perturb_labels(truth, n_steps, eps, mode="uniform", seed=0, max_shift=2):
    """Replace each true step with prob eps by a wrong one (uniform over steps or a nearby step)."""
    rng = np.random.default_rng(seed); out = {}
    for t, y in truth.items():
        L = n_steps[t]
        if rng.random() < eps and L > 1:
            if mode == "uniform":
                y2 = int(rng.choice([s for s in range(L) if s != y]))
            else:
                cand = [s for s in range(max(0, y - max_shift), min(L, y + max_shift + 1)) if s != y]
                y2 = int(rng.choice(cand)) if cand else y
            out[t] = y2
        else:
            out[t] = y
    return out

def kendall_tau(a, b):
    return float(st.kendalltau(a, b)[0])

# ---------- power ----------
def min_detectable_diff(sd_diff, n, alpha=0.05, power=0.8):
    """Smallest true mean paired difference detectable (normal approximation)."""
    return float((st.norm.ppf(1 - alpha / 2) + st.norm.ppf(power)) * sd_diff / np.sqrt(n))
