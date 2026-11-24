"""Run failure-attribution methods repeatedly and log every prediction.

Usage (Kaggle / GPU):
  python -m src.attribute --backend hf --model Qwen/Qwen2.5-7B-Instruct \
      --methods all_at_once binary_search --subsets algo hand --runs 5 --temperature 0.7 \
      --out results/runs_qwen7b.jsonl
Use --backend mock to test the plumbing without a GPU.
"""
import argparse, json, os, re, random, sys, time
from src.data import load_whowhen, audit

# ---------- backends ----------
class MockLLM:
    """Random answers; only for plumbing tests."""
    def __init__(self, seed=0): self.rng = random.Random(seed)
    def chat(self, prompt, temperature=0.0, max_new_tokens=64):
        return json.dumps({"agent_name": "x", "step_number": self.rng.randint(0, 5), "answer": self.rng.choice(["yes", "no", "first", "second"])})

class HFLLM:
    def __init__(self, model, load_4bit=True, max_ctx=14000, seed=0):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        self.torch, self.max_ctx = torch, max_ctx
        self.tok = AutoTokenizer.from_pretrained(model)
        kw = dict(device_map="auto")
        if load_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        else:
            kw["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw).eval()
        torch.manual_seed(seed)
    def chat(self, prompt, temperature=0.0, max_new_tokens=64):
        msgs = [{"role": "user", "content": prompt}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        if ids.shape[1] > self.max_ctx:  # keep head and tail of the prompt
            half = self.max_ctx // 2
            ids = self.torch.cat([ids[:, :half], ids[:, -half:]], dim=1)
        ids = ids.to(self.model.device)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.eos_token_id)
        kw.update(dict(do_sample=True, temperature=temperature, top_p=0.95) if temperature > 0 else dict(do_sample=False))
        with self.torch.no_grad():
            out = self.model.generate(ids, **kw)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

# ---------- prompt helpers ----------
def fmt(task, lo, hi, max_chars):
    lines = []
    for i in range(lo, hi):
        h = task["history"][i]; c = h["content"]
        if len(c) > max_chars: c = c[: max_chars // 2] + " ...[cut]... " + c[-max_chars // 2:]
        lines.append(f"[Step {i}] {h['agent']}: {c}")
    return "\n".join(lines)

HEAD = ("You are analysing a failed run of a multi-agent system. The problem was:\n{q}\n"
        "{gt}The system failed to solve it. Steps are numbered from 0.\n\n")

def _head(task, use_gt):
    return HEAD.format(q=task["question"], gt=(f"The correct final answer is: {task['ground_truth']}\n" if use_gt else ""))

def _json(text):
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try: return json.loads(m.group(0))
        except Exception: pass
    return {}

def _int(text, default=None):
    d = _json(text)
    for k in ("step_number", "step", "answer"):
        if k in d:
            try: return int(d[k])
            except Exception: pass
    m = re.search(r"\d+", text)
    return int(m.group(0)) if m else default

# ---------- methods: each returns predicted step (int or None) ----------
def all_at_once(llm, task, T, use_gt, max_chars):
    p = (_head(task, use_gt) + "Conversation log:\n" + fmt(task, 0, task["n_steps"], max_chars) +
         "\n\nIdentify the earliest step whose error made the whole attempt fail. "
         'Reply with JSON only: {"agent_name": "<agent>", "step_number": <int>}')
    out = llm.chat(p, T); s = _int(out)
    return (s if s is not None and 0 <= s < task["n_steps"] else None), out

def binary_search(llm, task, T, use_gt, max_chars):
    lo, hi = 0, task["n_steps"] - 1; log = []
    while lo < hi:
        mid = (lo + hi) // 2
        p = (_head(task, use_gt) + f"Log of steps {lo} to {hi}:\n" + fmt(task, lo, hi + 1, max_chars) +
             f"\n\nIs the decisive error in the first half (steps {lo}-{mid}) or the second half (steps {mid+1}-{hi})? "
             'Reply with JSON only: {"answer": "first"} or {"answer": "second"}')
        out = llm.chat(p, T, 16); log.append(out)
        if "second" in out.lower() and "first" not in out.lower(): lo = mid + 1
        else: hi = mid
    return lo, " | ".join(log)

def step_by_step(llm, task, T, use_gt, max_chars):
    log = []
    for i in range(task["n_steps"]):
        p = (_head(task, use_gt) + "Log so far:\n" + fmt(task, 0, i + 1, max_chars) +
             f"\n\nIs Step {i} an error that makes the final failure unavoidable? Reply with JSON only: " '{"answer": "yes"} or {"answer": "no"}')
        out = llm.chat(p, T, 16); log.append(out)
        if "yes" in out.lower(): return i, " | ".join(log)
    return None, " | ".join(log)

METHODS = {"all_at_once": all_at_once, "binary_search": binary_search, "step_by_step": step_by_step}

def agent_at(task, step, fallback):
    return task["history"][step]["agent"] if step is not None else fallback

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="hf", choices=["hf", "mock"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--methods", nargs="+", default=["all_at_once", "binary_search"])
    ap.add_argument("--subsets", nargs="+", default=["algo", "hand"])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--use_gt", action="store_true", help="give the correct final answer in the prompt")
    ap.add_argument("--max_step_chars", type=int, default=1500)
    ap.add_argument("--limit", type=int, default=0, help="first N tasks per subset (debug)")
    ap.add_argument("--data_dir", default=None, help="folder with tasks.jsonl from save_local")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tasks = load_whowhen(a.data_dir); print(json.dumps(audit(tasks), indent=1))
    llm = MockLLM() if a.backend == "mock" else HFLLM(a.model)
    done = set()
    if os.path.exists(a.out):  # resume
        for l in open(a.out): r = json.loads(l); done.add((r["task_id"], r["method"], r["run"], r["temperature"], r["use_gt"]))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        for sub in a.subsets:
            T = [t for t in tasks if t["subset"] == sub]
            T = T[: a.limit] if a.limit else T
            for t in T:
                for m in a.methods:
                    for run in range(a.runs if a.temperature > 0 else 1):
                        key = (t["id"], m, run, a.temperature, a.use_gt)
                        if key in done: continue
                        t0 = time.time()
                        try: step, raw = METHODS[m](llm, t, a.temperature, a.use_gt, a.max_step_chars); ok = True
                        except Exception as e: step, raw, ok = None, repr(e), False
                        f.write(json.dumps({"task_id": t["id"], "subset": sub, "method": m, "backbone": a.model if a.backend == "hf" else "mock",
                                            "temperature": a.temperature, "use_gt": a.use_gt, "run": run, "pred_step": step,
                                            "pred_agent": agent_at(t, step, None), "true_step": t["true_step"], "true_agent": t["true_agent"],
                                            "n_steps": t["n_steps"], "ok": ok, "sec": round(time.time() - t0, 2), "raw": raw[:300]}) + "\n")
                        f.flush()
if __name__ == "__main__":
    main()
