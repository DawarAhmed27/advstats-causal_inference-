"""Synthetic benchmark + synthetic 'LLM runs' with KNOWN accuracy, to validate the stats code."""
import numpy as np, json

def make_tasks(n_algo=126, n_hand=58, seed=0):
    rng = np.random.default_rng(seed); tasks = []
    for sub, n, lo, hi in (("algo", n_algo, 5, 10), ("hand", n_hand, 10, 90)):
        for i in range(n):
            L = int(rng.integers(lo, hi + 1)); y = int(rng.integers(0, L))
            hist = [{"agent": f"agent{j % 3}", "content": f"step {j}"} for j in range(L)]
            tasks.append({"id": f"{sub}-{i}", "subset": sub, "question": "q", "ground_truth": "a", "history": hist,
                          "true_agent": hist[y]["agent"], "true_step_raw": y, "true_step": y, "n_steps": L})
    return tasks

def make_runs(tasks, method, p_correct, R=5, seed=0, task_sd=0.15):
    """Each task has its own difficulty p_i around p_correct; each run is correct w.p. p_i,
    otherwise a random wrong step near the truth (to make conformal sets meaningful)."""
    rng = np.random.default_rng(seed); runs = []
    for t in tasks:
        pi = float(np.clip(rng.normal(p_correct, task_sd), 0.01, 0.99))
        for r in range(R):
            if rng.random() < pi: s = t["true_step"]
            else:
                c = [x for x in range(t["n_steps"]) if x != t["true_step"]]
                near = [x for x in c if abs(x - t["true_step"]) <= 2]
                s = int(rng.choice(near if near and rng.random() < 0.6 else c))
            runs.append({"task_id": t["id"], "subset": t["subset"], "method": method, "run": r, "pred_step": s,
                         "true_step": t["true_step"], "n_steps": t["n_steps"], "temperature": 0.7, "use_gt": False, "ok": True})
    return runs
