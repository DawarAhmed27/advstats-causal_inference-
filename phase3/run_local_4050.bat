@echo off
REM Phase 3 on a Windows laptop with an NVIDIA GPU (RTX 4050 6 GB). Double-click to run; safe to re-run (resumes).
REM Uses a 3B model so it fits in 6 GB. Hand-crafted logs are long, so the long-context subset runs with fewer steps kept per step.
setlocal
set MODEL=Qwen/Qwen2.5-3B-Instruct
cd /d "%USERPROFILE%"
if not exist advstats_run (git clone https://github.com/DawarAhmed27/advstats-causal_inference-.git advstats_run)
cd advstats_run && git pull
cd phase3
if not exist .venv (python -m venv .venv)
call .venv\Scripts\activate.bat
python -m pip install -q --upgrade pip
python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -q numpy pandas matplotlib scipy datasets transformers accelerate bitsandbytes
python -c "import torch;print('CUDA available:',torch.cuda.is_available())"
python -c "from src.data import *; import json; t=load_whowhen(); r=audit(t); print(json.dumps(r,indent=1)); save_local(t,'data'); json.dump(r,open('results/data_audit.json','w'),indent=1)"
REM smoke test first
python -m src.attribute --backend hf --model %MODEL% --data_dir data --methods all_at_once binary_search --runs 2 --limit 3 --max_step_chars 600 --out results\smoke.jsonl
REM main run: algo subset (short logs) with repeated sampling
python -m src.attribute --backend hf --model %MODEL% --data_dir data --subsets algo --methods all_at_once binary_search --runs 5 --temperature 0.7 --max_step_chars 600 --out results\runs_qwen3b.jsonl
REM hand-crafted subset (long logs, slower)
python -m src.attribute --backend hf --model %MODEL% --data_dir data --subsets hand --methods all_at_once binary_search --runs 5 --temperature 0.7 --max_step_chars 400 --out results\runs_qwen3b.jsonl
python -m src.analyze --runs results\runs_qwen3b.jsonl --out results\main
echo Done. Results are in %CD%\results\main
pause
