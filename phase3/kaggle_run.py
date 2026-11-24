# %% [markdown]
# # Phase 3 runner (Kaggle: GPU T4 x2, Internet ON). Paste each cell into a notebook, or run: python kaggle_run.py
# %%
import subprocess, sys, os
def sh(c): print(c); subprocess.run(c, shell=True, check=True)
if not os.path.exists("advstats-causal_inference-"):
    sh("git clone https://github.com/DawarAhmed27/advstats-causal_inference-.git")
os.chdir("advstats-causal_inference-/phase3")
sh(f"{sys.executable} -m pip install -q datasets bitsandbytes accelerate scipy")
# %% [markdown]
# ## 1. Data audit (step 2). Saves a local copy so later runs do not need the network.
# %%
from src.data import load_whowhen, audit, save_local
import json
tasks = load_whowhen(); rep = audit(tasks); print(json.dumps(rep, indent=1)); save_local(tasks, "data"); json.dump(rep, open("results/data_audit.json", "w"), indent=1)
# %% [markdown]
# ## 2. Smoke test: 3 tasks per subset, 2 runs (check the output file and the parse rate before the long run)
# %%
sh(f"{sys.executable} -m src.attribute --backend hf --model Qwen/Qwen2.5-7B-Instruct --data_dir data --methods all_at_once binary_search --runs 2 --limit 3 --out results/smoke.jsonl")
import pandas as pd; d = pd.read_json("results/smoke.jsonl", lines=True); print(d[["method", "pred_step", "true_step", "ok", "sec"]].head(12)); print("parse failures:", d.pred_step.isna().mean())
# %% [markdown]
# ## 3. Main run (resumable: re-running the cell continues where it stopped). Algo first, it is short.
# %%
sh(f"{sys.executable} -m src.attribute --backend hf --model Qwen/Qwen2.5-7B-Instruct --data_dir data --subsets algo --methods all_at_once binary_search step_by_step --runs 5 --temperature 0.7 --out results/runs_qwen7b.jsonl")
sh(f"{sys.executable} -m src.attribute --backend hf --model Qwen/Qwen2.5-7B-Instruct --data_dir data --subsets hand --methods all_at_once binary_search --runs 5 --temperature 0.7 --out results/runs_qwen7b.jsonl")
# %% [markdown]
# ## 4. Ablation runs: temperature 0 (single run), with ground-truth answer in the prompt
# %%
sh(f"{sys.executable} -m src.attribute --backend hf --model Qwen/Qwen2.5-7B-Instruct --data_dir data --methods all_at_once --temperature 0 --out results/runs_qwen7b_t0.jsonl")
sh(f"{sys.executable} -m src.attribute --backend hf --model Qwen/Qwen2.5-7B-Instruct --data_dir data --methods all_at_once --use_gt --runs 5 --temperature 0.7 --out results/runs_qwen7b_gt.jsonl")
# %% [markdown]
# ## 5. Second backbone (edit the model name, e.g. Qwen/Qwen3-8B), then analysis
# %%
# sh(f"{sys.executable} -m src.attribute --backend hf --model Qwen/Qwen3-8B --data_dir data --methods all_at_once binary_search --runs 5 --out results/runs_qwen3_8b.jsonl")
sh(f"{sys.executable} -m src.analyze --runs 'results/runs_*.jsonl' --out results/main")
# Download results/ (jsonl + results/main) before the session ends.
