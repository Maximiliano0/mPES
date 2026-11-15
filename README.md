<div align="center">

# 🦠 mPES — *Multiple Pandemic Experiment Suite*

**A reinforcement-learning benchmark for resource allocation under uncertainty.**

[![Python](https://img.shields.io/badge/python-3.12-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-ff6f00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-4.7-1a73e8.svg)](https://optuna.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.2-7c4dff.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/license-private-lightgrey.svg)](#-license)

</div>

Seven individual RL agents and six ensembles allocate limited resources across sequences of simulated pandemic
outbreaks. The active, validated line is `h1/`; `h2/` is a suspended staging area. The LaTeX thesis that reports
the `h1/` results lives in `writings/`.

## 🗂️ Repository layout

| Path | Contents |
|------|----------|
| [`h1/tabular/`](h1/tabular/) | `pes_base` (Q-Learning baseline), `pes_ql` (Q-Learning + Optuna), `pes_dql` (Double Q-Learning + ε warm-up + PBRS) |
| [`h1/ml/`](h1/ml/) | `pes_dqn` (DQN), `pes_rdqn` (recurrent DQN, LSTM), `pes_a2c` (Advantage Actor-Critic), `pes_trf` (causal Transformer DQN) |
| [`h1/ens/`](h1/ens/) | `pes_ens` (weighted soft voting, inference-only, best ensemble), `pes_ens_sprb`, `pes_ens_accq`, `pes_ens_trf_guard`, `pes_ens_consensus`, `pes_ens_consensus_prior` |
| [`h1/general/`](h1/general/) | Stress benchmark, seed-variability studies, results and comparison report |
| [`h2/`](h2/) | Suspended: `tabular_conf/ql_conf`, `tabular_conf/dql_conf` |
| [`writings/`](writings/) | Thesis (`00_Main/Main.tex`, `01_Chapters/`, `02_Images/`), audit (`audit/audit.py`), analysis scripts |
| [`dict/`](dict/) | Committee pre-reviews, correction tracker ([`correcciones.md`](dict/correcciones.md)) and defence material |
| [`utils/`](utils/) | `config/` (requirements, pyright, pylint) and `scripts/` (helpers, long-run lanes) |
| [`.claude/`](.claude/) · [`.github/`](.github/) | Claude Code and GitHub Copilot agents, skills and prompts ([`.claude/CLAUDE.md`](.claude/CLAUDE.md)) |

Every package ships `doc/<pkg>_explained.md` (usage) and `doc/<pkg>_theory.md` (theory). The benchmark is
documented in [`h1/general/README.md`](h1/general/README.md) and
[`h1/general/doc/comparacion_modelos.md`](h1/general/doc/comparacion_modelos.md).

## ⚙️ Setup

Windows only, Python 3.12. From the repository root:

```powershell
py -3.12 -m venv win_mpes_env
win_mpes_env\Scripts\Activate.ps1
pip install -r utils/config/requirements.txt
git lfs pull                                   # per-seed models and bibliography PDFs
```

In every session (an active `VIRTUAL_ENV` avoids the "Press ENTER" prompt of each package):

```powershell
win_mpes_env\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"; $env:TF_ENABLE_ONEDNN_OPTS = "0"
```

## ▶️ Usage

All `python -m` commands run from inside `h1/`.

```powershell
cd h1
python -m tabular.pes_ql                       # run a package: tabular.<pkg>, ml.<pkg>, ens.<pkg>
python -m ml.pes_trf.ext.optimize_tr 50        # Bayesian optimisation (50 trials)
python -m ens.pes_ens_sprb.ext.optimize_ens 50
```

Optimisation entry points: `tabular.pes_{ql,dql}.ext.optimize_rl`, `ml.pes_dqn.ext.optimize_dqn`,
`ml.pes_rdqn.ext.optimize_rdqn`, `ml.pes_a2c.ext.optimize_a2c`, `ml.pes_trf.ext.optimize_tr` and
`ens.<pkg>.ext.optimize_ens` (`pes_ens` is inference-only).

## 📈 Stress benchmark

Both suites (7 individual models, 6 ensembles) are evaluated on the reference (`sev_base`, 64 fixed sequences),
21 perturbations of severity, length, both jointly and block structure, and 5 held-out replicas
(`heldout_s1`–`s5`) drawn from the reference distributions.

```powershell
cd h1
python -m general.scripts.benchmark run --suite both    # resumable
python -m general.scripts.analysis                      # matrices + report.md
python -m general.scripts.figures
```

## 🎲 Seed variability

The delivered models come from the best Optuna trial, so their training seed is not an unbiased draw.

- **Individual models** — [`seed_variability.py`](h1/general/scripts/seed_variability.py) retrains Q-Learning,
  Double Q-Learning, DQN, recurrent DQN, A2C and the Transformer with 10 seeds each (1001–1010), as their
  `train_*` scripts do but without overwriting `inputs/`, and evaluates them on the reference and the held-out
  replicas. Per-seed models live in `h1/general/work/seeds/` (Git LFS).
- **Ensembles** — [`ensemble_seeds.py`](h1/general/scripts/ensemble_seeds.py) keeps each ensemble's rule and
  parameters and swaps its members for retrained seeds (`MPES_MODEL_ROOT`): 10 combinations (best, worst and
  closest to the mean of every member, plus 7 random) shared by the six ensembles.

```powershell
cd h1
python -m general.scripts.seed_variability run --models trf --seeds 1001 1002   # resumable; summary -> results/
python -m general.scripts.ensemble_seeds run --ensembles pes_ens                # plan | run | summary
```

Cost per seed on a laptop CPU: minutes (tabular, DQN, A2C) to 3–7 h (recurrent DQN, Transformer). For long runs,
[`utils/scripts/seed_run/`](utils/scripts/seed_run/) launches hidden *lanes* at below-normal priority with one
physical core left free, so the machine stays usable (one ensemble per lane, at most):

```powershell
powershell -ExecutionPolicy Bypass -File utils\scripts\seed_run\power_on.ps1      # no sleep on AC
powershell -ExecutionPolicy Bypass -File utils\scripts\seed_run\launch_lanes.ps1 -Module seed_variability `
    -Lane '--models rdqn --seeds 1001;--models rdqn --seeds 1002', '--models a2c --seeds 1001'
powershell -ExecutionPolicy Bypass -File utils\scripts\seed_run\status.ps1        # also: stop.ps1, power_restore.ps1
```

## 📝 Thesis

```powershell
cd writings
python audit\audit.py                          # audit + pdflatex/bibtex -> out/ (report: audit/AUDIT.md)
python auxiliar\scripts\sync_figures.py        # copy generated h1/ figures into 02_Images
```

Every number in the thesis comes from `h1/general/results/` or from the analysis scripts in
[`writings/auxiliar/scripts/`](writings/auxiliar/scripts/) (`seed_comparison.py`, `policy_vs_rule.py`,
`heldout_gap.py`, `trf_vs_ens.py`, …), run from the repository root. The revision after the committee
pre-reviews is tracked in [`dict/correcciones.md`](dict/correcciones.md).

## 📄 License

Private repository — all rights reserved.
