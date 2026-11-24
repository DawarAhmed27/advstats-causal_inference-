@echo off
REM Phase 3 on a Windows laptop with an NVIDIA GPU (RTX 4050 6 GB). Double-click to run; safe to re-run (resumes).
setlocal
set MODEL=Qwen/Qwen2.5-3B-Instruct
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
cd /d "%USERPROFILE%"
if not exist advstats_run (git clone https://github.com/DawarAhmed27/advstats-causal_inference-.git advstats_run)
cd advstats_run && git pull
cd phase3
if not exist results mkdir results
python --version
if not exist .venv (python -m venv .venv)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
REM Make sure torch is a CUDA build. If the current one is CPU-only, replace it.
python -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>nul
if errorlevel 1 (
  python -m pip uninstall -y torch torchvision torchaudio
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
)
python -m pip install numpy pandas matplotlib scipy datasets transformers accelerate bitsandbytes
python -c "import torch;print('CUDA available:',torch.cuda.is_available(), torch.__version__);print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
python -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
  echo.
  echo STOP: PyTorch cannot see the GPU. Send the lines above to Claude. Nothing was run.
  pause
  exit /b 1
)
python -c "from src.data import *; import json; t=load_whowhen(); r=audit(t); print(json.dumps(r,indent=1)); save_local(t,'data'); json.dump(r,open('results/data_audit.json','w'),indent=1)"
if errorlevel 1 goto fail
REM smoke test
python -m src.attribute --backend hf --model %MODEL% --data_dir data --methods all_at_once binary_search --runs 2 --limit 3 --max_step_chars 600 --out results\smoke.jsonl
if errorlevel 1 goto fail
REM main run: algo subset (short logs)
python -m src.attribute --backend hf --model %MODEL% --data_dir data --subsets algo --methods all_at_once binary_search --runs 5 --temperature 0.7 --max_step_chars 600 --out results\runs_qwen3b.jsonl
if errorlevel 1 goto fail
python -m src.analyze --runs results\runs_qwen3b.jsonl --out results\main_algo
REM hand-crafted subset (long logs, slower)
python -m src.attribute --backend hf --model %MODEL% --data_dir data --subsets hand --methods all_at_once binary_search --runs 5 --temperature 0.7 --max_step_chars 400 --out results\runs_qwen3b.jsonl
if errorlevel 1 goto fail
python -m src.analyze --runs results\runs_qwen3b.jsonl --out results\main
echo Done. Results are in %CD%\results\main
pause
exit /b 0
:fail
echo.
echo FAILED. Send the last lines above to Claude.
pause
exit /b 1
