"""Load and audit the Who&When benchmark (Hugging Face: Kevin355/Who_and_When).

Normalised task dict:
  id, subset ('algo'|'hand'), question, ground_truth, history=[{'agent','content'}],
  true_agent, true_step (0-based index into history), n_steps
The step-index convention (0- vs 1-based) is checked at load time by the audit.
"""
import json, re
from collections import Counter

REPO = "Kevin355/Who_and_When"

def _agent_of(h):
    return str(h.get("name") or h.get("role") or h.get("agent") or "unknown")

def _norm(row, subset, idx):
    hist = row["history"]
    if isinstance(hist, str):
        hist = json.loads(hist)
    hist = [{"agent": _agent_of(h), "content": str(h.get("content", ""))} for h in hist]
    return {"id": f"{subset}-{idx}", "subset": subset, "question": str(row.get("question", "")),
            "ground_truth": str(row.get("ground_truth", "")), "history": hist,
            "true_agent": str(row.get("mistake_agent", "")),
            "true_step_raw": int(str(row.get("mistake_step", "-1")).strip() or -1),
            "n_steps": len(hist)}

def load_whowhen(local_dir=None):
    """Returns list of tasks. Needs network (or a local copy saved with save_local)."""
    if local_dir:
        return [json.loads(l) for l in open(f"{local_dir}/tasks.jsonl", encoding="utf-8")]
    from datasets import load_dataset, get_dataset_config_names
    tasks = []
    for cfg in get_dataset_config_names(REPO):
        subset = "algo" if "algo" in cfg.lower() else "hand"
        ds = load_dataset(REPO, cfg)
        for split in ds:
            for i, row in enumerate(ds[split]):
                tasks.append(_norm(row, subset, f"{cfg}-{i}"))
    return tasks

def audit(tasks):
    """Step 2 of the methodology: pick the step-index offset and report data facts."""
    def match(offset):
        ok = n = 0
        for t in tasks:
            s = t["true_step_raw"] - offset
            if 0 <= s < t["n_steps"]:
                n += 1; ok += t["history"][s]["agent"] == t["true_agent"]
        return ok / max(n, 1)
    m0, m1 = match(0), match(1)
    offset = 0 if m0 >= m1 else 1
    for t in tasks:
        t["true_step"] = t["true_step_raw"] - offset
    rep = {"n_tasks": len(tasks), "offset_chosen": offset, "agent_match_offset0": m0, "agent_match_offset1": m1}
    for sub in ("algo", "hand"):
        T = [t for t in tasks if t["subset"] == sub]
        if T:
            L = [t["n_steps"] for t in T]
            rep[sub] = {"n": len(T), "steps_min": min(L), "steps_median": sorted(L)[len(L)//2], "steps_max": max(L),
                        "random_step_acc": sum(1/l for l in L)/len(L),
                        "true_step_out_of_range": sum(not (0 <= t["true_step"] < t["n_steps"]) for t in T)}
    return rep

def save_local(tasks, out_dir):
    import os; os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/tasks.jsonl", "w", encoding="utf-8") as f:
        for t in tasks: f.write(json.dumps(t, ensure_ascii=False) + "\n")
