<div align="center">

# 🦠 mPES — *Multiple Pandemic Experiment Suite*

**A reinforcement-learning benchmark for resource allocation under uncertainty.**

[![Python](https://img.shields.io/badge/python-3.12-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-ff6f00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-4.7-1a73e8.svg)](https://optuna.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.2-7c4dff.svg)](https://gymnasium.farama.org/)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-required-f64935.svg?logo=git&logoColor=white)](#-git-lfs)
[![License](https://img.shields.io/badge/license-private-lightgrey.svg)](#-license)

</div>

Seven individual RL agents and six ensembles allocate a limited budget across sequences of simulated pandemic
outbreaks, derived from the Pandemic Experiment Scenario (PES) of the BCI-NE Lab, University of Essex. Every
sequence is scored against the exact optimum, computed by dynamic programming with full information. The active,
validated line is `h1/`; `h2/` is a suspended staging area. The full report (Spanish) is
[`writings/out/`](writings/out/).

## 📊 Status and main results

The work is complete: all models are trained, optimised and evaluated, and the report is final.

Mean normalised performance $\bar r$ (1 = optimum, 0 = no allocation) on the reference (64 fixed sequences),
the 21 generalisation scenarios and the 5 held-out replicas (320 new sequences):

| Policy | Reference | Generalisation | Held-out |
|--------|:---------:|:--------------:|:--------:|
| Optimum (full information) | 1.000 | 1.000 | 1.000 |
| Fixed rule $a = \min(S + 2, R)$ | 0.940 | 0.937 | 0.943 |
| **Weighted ensemble** (`pes_ens`) | **0.937** | **0.939** | **0.939** |
| Transformer gate (`pes_ens_trf_guard`) | 0.928 | 0.931 | 0.928 |
| **Causal Transformer** (`pes_trf`) | **0.927** | **0.930** | **0.929** |
| Consensus with prior (`pes_ens_consensus_prior`) | 0.918 | 0.919 | 0.915 |
| Consensus (`pes_ens_consensus`) | 0.889 | 0.905 | 0.893 |
| Soft voting (`pes_ens_sprb`) | 0.914 | 0.902 | 0.917 |
| Action voting (`pes_ens_accq`) | 0.914 | 0.901 | 0.916 |
| DQN (`pes_dqn`) | 0.894 | 0.899 | 0.891 |
| A2C (`pes_a2c`) | 0.887 | 0.896 | 0.887 |
| Recurrent DQN (`pes_rdqn`) | 0.899 | 0.889 | 0.903 |
| Double Q-Learning (`pes_dql`) | 0.896 | 0.877 | 0.900 |
| Q-Learning (`pes_ql`) | 0.887 | 0.871 | 0.881 |
| Q-Learning baseline (`pes_base`) | 0.871 | 0.851 | 0.860 |

- The causal Transformer is the best individual model on average.
- Only the weighted ensemble beats it, by about 1 %. Its edge comes from a severity prior and a high temperature
  that bring its decisions close to a simple fixed rule, rather than from confidence weighting.
- A fixed rule calibrated on the optimum reaches the same level, because the optimal policy of this environment is
  simple.
- **Training-seed variability** (10 new seeds per model): the Transformer keeps the highest mean on the reference
  and the held-out replicas (0.919 / 0.920), but its range overlaps the recurrent DQN's (0.902 / 0.898). With 10
  seed combinations of its members, the weighted ensemble (0.928 / 0.932) beats the Transformer of the same
  combination in 9 of 10. On the 21 generalisation scenarios the Transformer also keeps the highest mean
  (0.921 ± 0.007, against 0.891 for the DQN and 0.885 for the recurrent DQN), and the weighted ensemble beats it in
  8 of 10 combinations, by a marginal 0.7 %.

## 🗂️ Repository layout

| Path | Contents |
|------|----------|
| [`h1/tabular/`](h1/tabular/) | `pes_base` (Q-Learning baseline), `pes_ql` (Q-Learning + Optuna), `pes_dql` (Double Q-Learning + ε warm-up + PBRS) |
| [`h1/ml/`](h1/ml/) | `pes_dqn` (DQN), `pes_rdqn` (recurrent DQN, LSTM), `pes_a2c` (Advantage Actor-Critic), `pes_trf` (causal Transformer DQN) |
| [`h1/ens/`](h1/ens/) | `pes_ens` (weighted soft voting, inference-only, best ensemble), `pes_ens_sprb`, `pes_ens_accq`, `pes_ens_trf_guard`, `pes_ens_consensus`, `pes_ens_consensus_prior` |
| [`h1/general/`](h1/general/) | Stress benchmark, seed-variability studies, post-hoc analyses, results (`results/`) and comparison report |
| [`h2/`](h2/) | Suspended: `tabular_conf/ql_conf`, `tabular_conf/dql_conf` |
| [`writings/out/`](writings/out/) | Compiled report (PDF, Spanish) |
| [`utils/`](utils/) | `config/` (requirements, pyright, pylint) and `scripts/` (helpers, long-run lanes) |
| [`.claude/`](.claude/) · [`.github/`](.github/) | Claude Code and GitHub Copilot agents, skills and prompts ([`.claude/CLAUDE.md`](.claude/CLAUDE.md)) |

Every package ships `doc/<pkg>_explained.md` (usage) and `doc/<pkg>_theory.md` (theory). The benchmark is
documented in [`h1/general/README.md`](h1/general/README.md) and
[`h1/general/doc/comparacion_modelos.md`](h1/general/doc/comparacion_modelos.md).

## 📦 Git LFS

The per-seed models of the seed-variability study are stored with [Git LFS](https://git-lfs.com/)
(`.gitattributes`):

| Pattern | Contents |
|---------|----------|
| `h1/general/work/seeds/**/*.keras` | DQN, recurrent DQN, A2C and Transformer retrained with seeds 1001–1010 |
| `h1/general/work/seeds/**/*.npy` | Q-Learning and Double Q-Learning Q-tables for the same seeds |

60 files, about 10 MB in total, under `h1/general/work/seeds/<pkg>/seed_<n>/` next to each `result.json`. The
delivered models in `h1/` and every other file are regular Git objects.

```powershell
git lfs install                                # once per machine
git clone https://github.com/Maximiliano0/mPES.git
cd mPES
git lfs pull                                   # download the LFS objects (if the clone skipped them)
git lfs ls-files                               # '*' = downloaded, '-' = pointer only
```

Without `git lfs pull` the seed models stay as text pointers. Everything else, including the benchmark and the
summaries in `h1/general/results/`, works without them; only re-evaluating individual seeds needs them. To clone
without LFS objects, set `GIT_LFS_SKIP_SMUDGE=1`.

## ⚙️ Setup

Windows only, Python 3.12. From the repository root:

```powershell
py -3.12 -m venv win_mpes_env
win_mpes_env\Scripts\Activate.ps1
pip install -r utils/config/requirements.txt
git lfs pull
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
- **Generalisation** — the same seeds and combinations evaluated, without retraining, on the 21 generalisation
  scenarios; [`seed_generalisation.py`](h1/general/scripts/seed_generalisation.py) gathers mean ± SD, 95 %
  intervals, Welch tests, Cohen's *d* and Spearman orderings across seeds (`results/*/seed_generalisation.json`).
  Per-sequence results are kept in `work/seeds/<pkg>/<run>/generalizacion.json` (force-added: `work/seeds/` is
  ignored) and `work/ens_seeds/generalizacion/<pkg>/<combination>.json`.

```powershell
cd h1
python -m general.scripts.seed_variability run --models trf --seeds 1001 1002   # resumable; summary -> results/
python -m general.scripts.ensemble_seeds run --ensembles pes_ens                # plan | run | summary
python -m general.scripts.seed_variability eval --models trf                    # saved models, 27 scenarios
python -m general.scripts.ensemble_seeds run --ensembles pes_ens --scenarios gen
python -m general.scripts.seed_generalisation
```

Cost per seed on a laptop CPU: minutes (tabular, DQN, A2C) to 3–7 h (recurrent DQN, Transformer). For long runs,
[`utils/scripts/seed_run/`](utils/scripts/seed_run/) launches hidden *lanes* at below-normal priority with one
physical core left free, so the machine stays usable. Each lane runs its jobs in sequence, separated by `;`:

```powershell
powershell -ExecutionPolicy Bypass -File utils\scripts\seed_run\power_on.ps1      # no sleep on AC
powershell -ExecutionPolicy Bypass -Command "& 'utils\scripts\seed_run\launch_lanes.ps1' -Module seed_variability -Lane '--models rdqn --seeds 1001;--models rdqn --seeds 1002','--models a2c --seeds 1001'"
powershell -ExecutionPolicy Bypass -Command "& 'utils\scripts\seed_run\launch_lanes.ps1' -Module seed_variability -Command eval -Lane '--models trf'"
powershell -ExecutionPolicy Bypass -Command "& 'utils\scripts\seed_run\launch_lanes.ps1' -Module ensemble_seeds -First 3 -Lane '--ensembles pes_ens --scenarios gen'"
powershell -ExecutionPolicy Bypass -File utils\scripts\seed_run\status.ps1        # progress, lanes, hours left
```

- Pass several `-Lane` values through `-Command "& '...ps1' ..."`. With `-File`, PowerShell does not parse the list.
- Run at most one ensemble per lane: each run temporarily replaces the CSVs in the package's `inputs/`.
- When adding lanes while others are still running, use `-First N` so the new lanes do not overwrite the `.cmd` of a
  live one.
- `stop.ps1` stops everything and restores those CSVs from git. Every run is resumable: at most the scenario in
  progress is lost. When finished, run `power_restore.ps1`.

## 🔬 Post-hoc analyses

Every table and figure of the report comes from `h1/general/results/` or from these scripts in
[`h1/general/scripts/`](h1/general/scripts/), run from the repository root after the benchmark. Their outputs go to
`h1/general/results/`.

| Script | Produces |
|--------|----------|
| `ensemble_decisions.py` | How often each fixed rule changes the decision of `pes_ens`, `pes_ens_consensus_prior` and `pes_ens_trf_guard` |
| `trf_vs_ens.py` | Each ensemble against the Transformer in the 21 generalisation scenarios |
| `heldout_gap.py` | Reference against the held-out replicas (paired test) |
| `weighted_ens_sensitivity.py` | One-at-a-time sensitivity of the fixed parameters of `pes_ens` (resumable, ~16 min) |
| `weighted_ens_oracle.py` | Exact optimal allocation and the effect of each rule of `pes_ens` |
| `fixed_rule.py` | Model-free rule $a = \min(S + k, R)$ in the 27 scenarios |
| `policy_vs_rule.py` | Learned policy vs. fixed rule vs. optimum, by severity (reads the recorded decisions) |
| `seed_comparison.py` | Seed spread against the differences between models and ensembles |
| `seed_generalisation.py` | Seed spread, intervals, Welch tests and Spearman orderings in the 21 generalisation scenarios (`python -m`, from `h1/`) |

```powershell
python h1/general/scripts/fixed_rule.py
python h1/general/scripts/seed_comparison.py
```

## 📄 License

Private repository — all rights reserved. © 2026 Maximiliano Vega.
