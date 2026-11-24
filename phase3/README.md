# Phase 3: Implementation and experiments

Statistically grounded evaluation of failure attribution on Who&When (see the Phase 1 and 2 documents).

## Status (read this first)
- The code is written and tested **on synthetic data only**. No real Who&When or LLM result exists yet.
- `results/demo/` holds outputs made from *synthetic* runs to check the code path. They say nothing about real methods.
- Real data (Hugging Face `Kevin355/Who_and_When`) and LLM calls need network and a GPU: run `kaggle_run.py` on Kaggle (T4 x2) or on a laptop GPU.

## Layout
- `src/data.py`: loads the benchmark, picks the step-index convention (0- or 1-based) by checking that the labelled agent speaks at the labelled step, reports log lengths and random-guess accuracy.
- `src/attribute.py`: All-at-Once, Binary Search, Step-by-Step with a Hugging Face 4-bit backend (default Qwen2.5-7B-Instruct) and a mock backend. Writes one JSON line per (task, method, run); resumable.
- `src/stats.py`: metrics, chance adjustment, Wilson, Beta-credible and task-level bootstrap intervals, paired bootstrap + sign-flip test, Holm correction, run-to-run consistency, cross-conformal step sets (optionally contiguous), label-noise simulation, power.
- `src/analyze.py`: experiments E1 to E7 and figures.
- `tests/`: synthetic generator and `python -m tests.test_stats`.

## Experiments
| Id | Question | Output |
|----|----------|--------|
| E1 | accuracy with intervals (tolerance 0, 1, 2), random baseline, chance-adjusted | `e1_main_accuracy.csv`, `fig1_forest.png` |
| E2 | how often do repeated runs agree | `e2_run_consistency.csv` |
| E3 | paired method differences with Holm-adjusted p | `e3_paired_comparisons.csv` |
| E4 | conformal coverage and set size | `e4_conformal.csv`, `fig3_conformal.png` |
| E5 | accuracy and ranking under label noise | `e5_label_noise.csv`, `fig4_label_noise.png` |
| E6 | spread of single-run accuracy | `e6_single_run_spread.csv` |
| E7 | smallest detectable paired difference | `e7_power.csv` |

Ablations come from the run configurations: temperature 0 vs 0.7, with vs without the correct answer in the prompt (`--use_gt`), number of runs, backbone, subset.

## Known limits
- Step-by-Step needs one LLM call per step and is only planned for the Algo subset.
- Long Hand-crafted logs are cut per step (`--max_step_chars`) and, if still too long, head and tail are kept. This lowers accuracy on long logs and must be stated in the paper.
- Prompts are our own wording, not the original Who&When prompts, so numbers are not directly comparable with published ones until we also run their code.
- Wilson/Beta intervals in E1 treat each task as one Bernoulli trial with the rounded run-average; the task-level bootstrap is the primary interval.
- The random baseline for tolerance > 0 ignores log edges.
- Conformal coverage is measured on held-out folds; with 58 Hand-crafted logs the calibration set is small.
