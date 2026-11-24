"""Turn run logs into tables and figures.
  python -m src.analyze --runs results/runs_*.jsonl --out results/main
  python -m src.analyze --demo --out results/demo      # synthetic data, checks the code path
"""
import argparse, glob, json, os, itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import stats as S

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.grid": True, "grid.alpha": .3, "font.size": 10})

def load(paths):
    rows = []
    for p in paths:
        for l in open(p, encoding="utf-8"): rows.append(json.loads(l))
    df = pd.DataFrame(rows); df = df[df["ok"]] if "ok" in df else df
    df["config"] = df["method"] + "|" + df["backbone"].astype(str) + "|T" + df["temperature"].astype(str) + ("|gt" if "use_gt" in df and df["use_gt"].any() else "")
    return df

def task_table(df, tol=0):
    """One row per (subset, config, task): mean correctness over runs."""
    d = df.copy(); d["hit"] = (d["pred_step"].notna() & ((d["pred_step"] - d["true_step"]).abs() <= tol)).astype(float)
    return d.groupby(["subset", "config", "task_id"]).agg(acc=("hit", "mean"), L=("n_steps", "first"), runs=("run", "nunique")).reset_index()

def e1_main(df, out):
    rows = []
    for tol in (0, 1, 2):
        tt = task_table(df, tol)
        for (sub, cfg), g in tt.groupby(["subset", "config"]):
            rand = float(np.mean([min(1, (2 * tol + 1)) / l for l in g["L"]]))
            m = g["acc"].mean(); lo, hi = S.cluster_bootstrap(g["acc"].values)
            k = int(round(m * len(g))); wl, wh = S.wilson(k, len(g)); bl, bh = S.beta_credible(k, len(g))
            rows.append(dict(subset=sub, config=cfg, tol=tol, n_tasks=len(g), acc=m, boot_lo=lo, boot_hi=hi, wilson_lo=wl, wilson_hi=wh,
                             bayes_lo=bl, bayes_hi=bh, random=rand, chance_adj=S.chance_adjusted(m, rand)))
    r = pd.DataFrame(rows); r.to_csv(f"{out}/e1_main_accuracy.csv", index=False); return r

def fig_forest(r, out):
    d = r[r.tol == 0]; subs = sorted(d.subset.unique())
    fig, axs = plt.subplots(1, len(subs), figsize=(5.8 * len(subs), 0.5 * max(3, d.config.nunique()) + 1.8), squeeze=False)
    for ax, sub in zip(axs[0], subs):
        g = d[d.subset == sub].sort_values("acc"); y = np.arange(len(g))
        ax.errorbar(g.acc, y, xerr=[g.acc - g.boot_lo, g.boot_hi - g.acc], fmt="o", color="#1F3864", capsize=3)
        ax.scatter(g.random, y, marker="|", s=140, color="#C00000", label="random baseline")
        ax.set_yticks(y); ax.set_yticklabels(g.config, fontsize=8); ax.set_title(f"Step accuracy, {sub} (95% task-bootstrap CI)"); ax.set_xlim(0, 1); ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{out}/fig1_forest.png", dpi=170); plt.close()

def e2_consistency(df, out):
    rows = []
    for (sub, cfg), g in df.groupby(["subset", "config"]):
        if g["run"].nunique() > 1:
            c = S.run_consistency(g.to_dict("records")); rows.append(dict(subset=sub, config=cfg, **c))
    r = pd.DataFrame(rows); r.to_csv(f"{out}/e2_run_consistency.csv", index=False); return r

def e3_pairs(df, out):
    tt = task_table(df); rows = []
    for sub, g in tt.groupby("subset"):
        wide = g.pivot(index="task_id", columns="config", values="acc").dropna()
        for a, b in itertools.combinations(wide.columns, 2):
            d = S.paired_diff(wide[a].values, wide[b].values); rows.append(dict(subset=sub, A=a, B=b, n=len(wide), **d))
    r = pd.DataFrame(rows)
    if len(r): r["p_holm"] = r.groupby("subset")["p"].transform(lambda s: S.holm(s.values)); r["significant_holm_05"] = r.p_holm < .05
    r.to_csv(f"{out}/e3_paired_comparisons.csv", index=False); return r

def e4_conformal(df, out, alphas=(0.05, 0.1, 0.2, 0.3)):
    rows = []
    for (sub, cfg), g in df.groupby(["subset", "config"]):
        if g["run"].nunique() < 3: continue
        by = {}; [by.setdefault(t, []).append(int(p) if pd.notna(p) else None) for t, p in zip(g.task_id, g.pred_step)]
        L = g.groupby("task_id")["n_steps"].first().astype(int).to_dict(); y = g.groupby("task_id")["true_step"].first().astype(int).to_dict()
        P = S.vote_matrix(by, L, smooth=0.05)
        for a in alphas:
            for contig in (False, True):
                res = S.conformal_sets(P, y, alpha=a, K=5, contiguous=contig)
                cov = np.array([c for c, _ in res.values()], float); size = np.array([z / L[t] for t, (_, z) in res.items()])
                lo, hi = S.cluster_bootstrap(cov, B=2000)
                rows.append(dict(subset=sub, config=cfg, alpha=a, contiguous=contig, n=len(cov), target=1 - a, coverage=cov.mean(), cov_lo=lo, cov_hi=hi,
                                 mean_set_share=size.mean(), mean_set_size=np.mean([z for _, z in res.values()])))
    r = pd.DataFrame(rows); r.to_csv(f"{out}/e4_conformal.csv", index=False)
    if len(r):
        fig, axs = plt.subplots(1, 2, figsize=(10, 4)); d = r[~r.contiguous]
        for (sub, cfg), g in d.groupby(["subset", "config"]):
            axs[0].plot(g.target, g.coverage, "o-", label=f"{sub} {cfg}", ms=4); axs[1].plot(g.coverage, g.mean_set_share, "o-", ms=4)
        axs[0].plot([.7, 1], [.7, 1], "k--", lw=1); axs[0].set_xlabel("target coverage 1-α"); axs[0].set_ylabel("held-out coverage"); axs[0].set_title("Conformal coverage"); axs[0].legend(fontsize=6)
        axs[1].set_xlabel("held-out coverage"); axs[1].set_ylabel("set size / log length"); axs[1].set_title("Cost of the guarantee")
        plt.tight_layout(); plt.savefig(f"{out}/fig3_conformal.png", dpi=170); plt.close()
    return r

def e5_noise(df, out, eps_grid=(0, .05, .1, .15, .2, .3), reps=200):
    rows = []; tt0 = df.copy()
    for sub, g in tt0.groupby("subset"):
        L = g.groupby("task_id")["n_steps"].first().to_dict(); y = g.groupby("task_id")["true_step"].first().to_dict()
        cfgs = sorted(g.config.unique()); base = {}
        for c in cfgs:
            gc = g[g.config == c]; base[c] = np.mean(list(S.per_task_scores(gc.to_dict("records")).values()))
        for mode in ("uniform", "nearby"):
            for eps in eps_grid:
                accs = {c: [] for c in cfgs}; taus = []
                for r in range(reps):
                    y2 = S.perturb_labels(y, L, eps, mode, seed=r)
                    cur = {}
                    for c in cfgs:
                        gc = g[g.config == c]; hit = {}
                        for t, p, ts in zip(gc.task_id, gc.pred_step, gc.true_step): hit.setdefault(t, []).append(float(p == y2[t]) if pd.notna(p) else 0.0)
                        cur[c] = np.mean([np.mean(v) for v in hit.values()]); accs[c].append(cur[c])
                    if len(cfgs) > 2 and eps > 0: taus.append(S.kendall_tau([base[c] for c in cfgs], [cur[c] for c in cfgs]))
                for c in cfgs:
                    rows.append(dict(subset=sub, config=c, mode=mode, eps=eps, acc_mean=np.mean(accs[c]), acc_lo=np.quantile(accs[c], .025), acc_hi=np.quantile(accs[c], .975),
                                     rank_tau=(float(np.mean([t for t in taus if t == t])) if any(t == t for t in taus) else np.nan)))
    r = pd.DataFrame(rows); r.to_csv(f"{out}/e5_label_noise.csv", index=False)
    if len(r):
        d = r[r["mode"] == "uniform"]; subs = sorted(d.subset.unique())
        fig, axs = plt.subplots(1, len(subs), figsize=(5.5 * len(subs), 4), squeeze=False)
        for ax, sub in zip(axs[0], subs):
            for c, g in d[d.subset == sub].groupby("config"):
                ax.plot(g.eps, g.acc_mean, "o-", ms=4, label=c); ax.fill_between(g.eps, g.acc_lo, g.acc_hi, alpha=.15)
            ax.set_title(f"Accuracy under label noise, {sub}"); ax.set_xlabel("share of wrong labels ε"); ax.set_ylabel("step accuracy"); ax.legend(fontsize=6)
        plt.tight_layout(); plt.savefig(f"{out}/fig4_label_noise.png", dpi=170); plt.close()
    return r

def e6_runs_and_ablation(df, out):
    """Accuracy as a function of the number of runs averaged (shows how noisy a single run is)."""
    rows = []
    for (sub, cfg), g in df.groupby(["subset", "config"]):
        R = g["run"].nunique()
        if R < 2: continue
        for r in range(1, R + 1):
            sub_g = g[g["run"] < r]; sc = list(S.per_task_scores(sub_g.to_dict("records")).values())
            # single-run spread: accuracy of each individual run
        per_run = [np.mean([float(p == t) for p, t in zip(gg.pred_step, gg.true_step)]) for _, gg in g.groupby("run")]
        rows.append(dict(subset=sub, config=cfg, runs=R, acc_run_min=min(per_run), acc_run_max=max(per_run), acc_run_sd=float(np.std(per_run, ddof=1)),
                         acc_mean=float(np.mean(per_run))))
    r = pd.DataFrame(rows); r.to_csv(f"{out}/e6_single_run_spread.csv", index=False); return r

def e7_power(df, out):
    tt = task_table(df); rows = []
    for sub, g in tt.groupby("subset"):
        wide = g.pivot(index="task_id", columns="config", values="acc").dropna()
        for a, b in itertools.combinations(wide.columns, 2):
            d = (wide[a] - wide[b]).values; rows.append(dict(subset=sub, A=a, B=b, n=len(d), sd_diff=d.std(ddof=1), min_detectable_diff_80pct=S.min_detectable_diff(d.std(ddof=1), len(d))))
    r = pd.DataFrame(rows); r.to_csv(f"{out}/e7_power.csv", index=False); return r

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="+"); ap.add_argument("--out", required=True); ap.add_argument("--demo", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.demo:
        from tests.synth import make_tasks, make_runs
        T = make_tasks(); rows = []
        for m, p, bb in (("all_at_once", .30, "demo-A"), ("binary_search", .24, "demo-A"), ("all_at_once", .36, "demo-B")):
            rows += [dict(r, backbone=bb) for r in make_runs(T, m, p, R=5, seed={"all_at_once|demo-A":1,"binary_search|demo-A":2,"all_at_once|demo-B":3}[m+"|"+bb])]
        df = pd.DataFrame(rows); df["config"] = df["method"] + "|" + df["backbone"] + "|T0.7"
    else:
        df = load([p for pat in a.runs for p in glob.glob(pat)])
    r1 = e1_main(df, a.out); fig_forest(r1, a.out); e2_consistency(df, a.out); e3_pairs(df, a.out); e4_conformal(df, a.out); e5_noise(df, a.out, reps=60 if a.demo else 200)
    e6_runs_and_ablation(df, a.out); e7_power(df, a.out); print("wrote", sorted(os.listdir(a.out)))
if __name__ == "__main__":
    main()
