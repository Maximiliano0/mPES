<div align="center">

# 🦠 mPES — *Multiple Pandemic Experiment Suite*

**A reinforcement-learning benchmark for resource allocation under uncertainty.**

[![Python](https://img.shields.io/badge/python-3.12-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-ff6f00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-4.7-1a73e8.svg)](https://optuna.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.2-7c4dff.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/license-private-lightgrey.svg)](#-license)

</div>

> This repository contains two experiment lines. The active benchmark workflow is in `h1/`; `h2/` remains an experimental staging area and is not part of the current benchmark report. The LaTeX thesis that reports the `h1/` results lives in `writings/`.

---

## ✨ Current scope

- 🧠 **Active benchmark line:** `h1/` with tabular, neural and ensemble training packages.
- 🧪 **Experimental staging line:** `h2/` (`tabular_conf/ql_conf`, `tabular_conf/dql_conf`) remains out of the validated benchmark path.
- 📊 **Under Stress Experiments:** [`h1/general/`](h1/general/) over seven individual models and six ensemble variants.
- 🔬 **Bayesian optimisation:** Optuna for the trainable variants.
- 🌍 **Windows-only workflow:** Python 3.12 and the `win_mpes_env` environment.
- 📚 **Package-level docs:** every package ships `doc/<pkg>_explained.md` (usage) and `doc/<pkg>_theory.md` (theory), in Markdown only.
- 📝 **Thesis:** [`writings/`](writings/) holds the LaTeX manuscript (`00_Main/Main.tex`, chapters in `01_Chapters/`, figures in `02_Images/`) and its audit script (`audit/audit.py`).
- ⚖️ **Jury review:** [`dict/`](dict/) holds the two pre-reviews of the thesis committee, the correction tracker ([`dict/correcciones.md`](dict/correcciones.md)) and the defence material ([`dict/defensa/`](dict/defensa/)).
- 🎲 **Seed variability:** [`h1/general/scripts/seed_variability.py`](h1/general/scripts/seed_variability.py) retrains Q-Learning, DQN and the Transformer with new training seeds, without touching the delivered models.
- 🤖 **AI assistants:** GitHub Copilot agents and prompts in [`.github/`](.github/); their Claude Code equivalents in [`.claude/`](.claude/), described in [`.claude/CLAUDE.md`](.claude/CLAUDE.md).

### Verified workspace snapshot

The current repository snapshot in this workspace is aligned with the validated `h1/` branch:

- `h1/tabular/`: 3 active packages (`pes_base`, `pes_ql`, `pes_dql`)
- `h1/ml/`: 4 active packages (`pes_dqn`, `pes_rdqn`, `pes_a2c`, `pes_trf`)
- `h1/ens/`: 6 active ensemble variants (`pes_ens`, `pes_ens_sprb`, `pes_ens_accq`, `pes_ens_trf_guard`, `pes_ens_consensus`, `pes_ens_consensus_prior`)
- `h1/general/`: benchmark harness with `scripts/`, `results/`, `work/` and the comparison report
- `h2/`: suspended staging area for `tabular_conf/ql_conf` and `tabular_conf/dql_conf`
- `utils/` and `writings/`: shared configuration, helper scripts, and the thesis workflow
- `dict/`: committee pre-reviews, correction tracker and defence material

### Repository lines

| Line | Status | Current package(s) |
|------|--------|--------------------|
| [`h1/`](h1/) | Active and validated | `tabular/`, `ml/`, `ens/`, `general/` |
| [`h2/`](h2/) | Suspended / staging | `tabular_conf/ql_conf`, `tabular_conf/dql_conf` |

---

## 📦 Active packages

### `h1/tabular/`

| Package | Algorithm | Key files |
|---------|-----------|-----------|
| `pes_base` | Tabular Q-Learning baseline | [`ext/pandemic.py`](h1/tabular/pes_base/ext/pandemic.py), [`ext/train_rl.py`](h1/tabular/pes_base/ext/train_rl.py) |
| `pes_ql` | Q-Learning + Bayesian optimisation | [`ext/optimize_rl.py`](h1/tabular/pes_ql/ext/optimize_rl.py) |
| `pes_dql` | Double Q-Learning + ε-decay warm-up + PBRS | [`ext/pandemic.py`](h1/tabular/pes_dql/ext/pandemic.py), [`ext/optimize_rl.py`](h1/tabular/pes_dql/ext/optimize_rl.py) |

### `h1/ml/`

| Package | Algorithm | Key files |
|---------|-----------|-----------|
| `pes_dqn` | Deep Q-Network | [`ext/dqn_model.py`](h1/ml/pes_dqn/ext/dqn_model.py), [`ext/train_dqn.py`](h1/ml/pes_dqn/ext/train_dqn.py), [`ext/optimize_dqn.py`](h1/ml/pes_dqn/ext/optimize_dqn.py) |
| `pes_rdqn` | Recurrent DQN (LSTM) | [`ext/rdqn_model.py`](h1/ml/pes_rdqn/ext/rdqn_model.py), [`ext/train_rdqn.py`](h1/ml/pes_rdqn/ext/train_rdqn.py), [`ext/optimize_rdqn.py`](h1/ml/pes_rdqn/ext/optimize_rdqn.py) |
| `pes_a2c` | Advantage Actor-Critic | [`ext/ac_model.py`](h1/ml/pes_a2c/ext/ac_model.py), [`ext/train_a2c.py`](h1/ml/pes_a2c/ext/train_a2c.py), [`ext/optimize_a2c.py`](h1/ml/pes_a2c/ext/optimize_a2c.py) |
| `pes_trf` | Causal Transformer DQN | [`ext/transformer_model.py`](h1/ml/pes_trf/ext/transformer_model.py), [`ext/train_transformer.py`](h1/ml/pes_trf/ext/train_transformer.py), [`ext/optimize_tr.py`](h1/ml/pes_trf/ext/optimize_tr.py) |

### `h1/ens/`

| Package | Algorithm | Key files |
|---------|-----------|-----------|
| `pes_ens_sprb` | Confidence-weighted soft voting ensemble | [`__main__.py`](h1/ens/pes_ens_sprb/__main__.py), [`ext/ensemble.py`](h1/ens/pes_ens_sprb/ext/ensemble.py), [`ext/optimize_ens.py`](h1/ens/pes_ens_sprb/ext/optimize_ens.py) |
| `pes_ens_accq` | Confidence-weighted action/Q-value ensemble | [`__main__.py`](h1/ens/pes_ens_accq/__main__.py), [`ext/ensemble.py`](h1/ens/pes_ens_accq/ext/ensemble.py), [`ext/optimize_ens.py`](h1/ens/pes_ens_accq/ext/optimize_ens.py) |
| `pes_ens_trf_guard` | Transformer-first confidence-gated ensemble | [`__main__.py`](h1/ens/pes_ens_trf_guard/__main__.py), [`ext/ensemble.py`](h1/ens/pes_ens_trf_guard/ext/ensemble.py), [`ext/optimize_ens.py`](h1/ens/pes_ens_trf_guard/ext/optimize_ens.py) |
| `pes_ens_consensus` | Agreement/disagreement confidence consensus | [`__main__.py`](h1/ens/pes_ens_consensus/__main__.py), [`ext/ensemble.py`](h1/ens/pes_ens_consensus/ext/ensemble.py), [`ext/optimize_ens.py`](h1/ens/pes_ens_consensus/ext/optimize_ens.py) |
| `pes_ens` | Weighted soft-voting ensemble (DQN + RDQN + TRF; A2C configured but disabled), inference-only | [`__main__.py`](h1/ens/pes_ens/__main__.py), [`ext/ensemble_model.py`](h1/ens/pes_ens/ext/ensemble_model.py), [`ext/tools.py`](h1/ens/pes_ens/ext/tools.py) |
| `pes_ens_consensus_prior` | Confidence consensus with severity-informed prior | [`__main__.py`](h1/ens/pes_ens_consensus_prior/__main__.py), [`ext/ensemble.py`](h1/ens/pes_ens_consensus_prior/ext/ensemble.py), [`ext/optimize_ens.py`](h1/ens/pes_ens_consensus_prior/ext/optimize_ens.py) |

> All six ensemble variants (`pes_ens`, `pes_ens_sprb`, `pes_ens_accq`, `pes_ens_consensus`, `pes_ens_consensus_prior` and `pes_ens_trf_guard`) are part of the active benchmark workflow. `pes_ens` is the best-performing ensemble in the current results.

### Support directories

| Path | Purpose |
|------|---------|
| [`h1/general/`](h1/general/) | Under Stress Experiments harness and seed-variability study |
| [`utils/`](utils/) | Requirements, lint/type-check config (`config/`), helper scripts (`scripts/`) |
| [`writings/`](writings/) | LaTeX thesis, audit tooling and thesis-specific analysis scripts |
| [`dict/`](dict/) | Committee pre-reviews, correction tracker and defence material |
| [`.claude/`](.claude/) · [`.github/`](.github/) | Claude Code and GitHub Copilot agents, skills and prompts |

---

## ⚙️ Setup

Create the virtual environment once, from the repository root (PowerShell,
Python 3.12):

```powershell
py -3.12 -m venv win_mpes_env
win_mpes_env\Scripts\Activate.ps1
pip install -r utils/config/requirements.txt
```

Then, in every session, activate it and set the variables:

```powershell
win_mpes_env\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"
$env:TF_ENABLE_ONEDNN_OPTS = "0"
```

`VIRTUAL_ENV` must point to the active environment (set by the activation
script); otherwise each package's `__init__.py` pauses with a "Press ENTER"
prompt. Pinned dependencies live in
[`utils/config/requirements.txt`](utils/config/requirements.txt).

---

## ▶️ Usage

> Run all commands from inside `h1/`.

```powershell
cd h1

python -m tabular.pes_base
python -m tabular.pes_ql
python -m tabular.pes_dql

python -m ml.pes_dqn
python -m ml.pes_rdqn
python -m ml.pes_a2c
python -m ml.pes_trf

python -m ens.pes_ens_sprb
python -m ens.pes_ens_accq
python -m ens.pes_ens_trf_guard
python -m ens.pes_ens_consensus
python -m ens.pes_ens_consensus_prior
python -m ens.pes_ens
```

### Bayesian optimisation

```powershell
cd h1
python -m tabular.pes_ql.ext.optimize_rl 50
python -m tabular.pes_dql.ext.optimize_rl 50
python -m ml.pes_dqn.ext.optimize_dqn 50
python -m ml.pes_rdqn.ext.optimize_rdqn 50
python -m ml.pes_a2c.ext.optimize_a2c 50
python -m ml.pes_trf.ext.optimize_tr 50
python -m ens.pes_ens_sprb.ext.optimize_ens 50
python -m ens.pes_ens_accq.ext.optimize_ens 50
python -m ens.pes_ens_trf_guard.ext.optimize_ens 50
python -m ens.pes_ens_consensus.ext.optimize_ens 50
python -m ens.pes_ens_consensus_prior.ext.optimize_ens 50
```

> `pes_ens` is inference-only and has no optimisation entry point.

---

## 📚 Documentation

The active repository documentation is centred on the executable packages in `h1/`. The general benchmark harness compares individual and ensemble suites under the same stress catalogue.

| Package | Guide |
|---------|-------|
| `pes_base` | [`pes_base_explained.md`](h1/tabular/pes_base/doc/pes_base_explained.md) · [`pes_base_theory.md`](h1/tabular/pes_base/doc/pes_base_theory.md) |
| `pes_ql` | [`pes_ql_explained.md`](h1/tabular/pes_ql/doc/pes_ql_explained.md) · [`pes_ql_theory.md`](h1/tabular/pes_ql/doc/pes_ql_theory.md) |
| `pes_dql` | [`pes_dql_explained.md`](h1/tabular/pes_dql/doc/pes_dql_explained.md) · [`pes_dql_theory.md`](h1/tabular/pes_dql/doc/pes_dql_theory.md) |
| `pes_dqn` | [`pes_dqn_explained.md`](h1/ml/pes_dqn/doc/pes_dqn_explained.md) · [`pes_dqn_theory.md`](h1/ml/pes_dqn/doc/pes_dqn_theory.md) |
| `pes_rdqn` | [`pes_rdqn_explained.md`](h1/ml/pes_rdqn/doc/pes_rdqn_explained.md) · [`pes_rdqn_theory.md`](h1/ml/pes_rdqn/doc/pes_rdqn_theory.md) |
| `pes_a2c` | [`pes_a2c_explained.md`](h1/ml/pes_a2c/doc/pes_a2c_explained.md) · [`pes_a2c_theory.md`](h1/ml/pes_a2c/doc/pes_a2c_theory.md) |
| `pes_trf` | [`pes_trf_explained.md`](h1/ml/pes_trf/doc/pes_trf_explained.md) · [`pes_trf_theory.md`](h1/ml/pes_trf/doc/pes_trf_theory.md) |
| `pes_ens_sprb` | [`pes_ens_sprb_explained.md`](h1/ens/pes_ens_sprb/doc/pes_ens_sprb_explained.md) · [`pes_ens_sprb_theory.md`](h1/ens/pes_ens_sprb/doc/pes_ens_sprb_theory.md) |
| `pes_ens_accq` | [`pes_ens_accq_explained.md`](h1/ens/pes_ens_accq/doc/pes_ens_accq_explained.md) · [`pes_ens_accq_theory.md`](h1/ens/pes_ens_accq/doc/pes_ens_accq_theory.md) |
| `pes_ens_trf_guard` | [`pes_ens_trf_guard_explained.md`](h1/ens/pes_ens_trf_guard/doc/pes_ens_trf_guard_explained.md) · [`pes_ens_trf_guard_theory.md`](h1/ens/pes_ens_trf_guard/doc/pes_ens_trf_guard_theory.md) |
| `pes_ens_consensus` | [`pes_ens_consensus_explained.md`](h1/ens/pes_ens_consensus/doc/pes_ens_consensus_explained.md) · [`pes_ens_consensus_theory.md`](h1/ens/pes_ens_consensus/doc/pes_ens_consensus_theory.md) |
| `pes_ens` | [`pes_ens_explained.md`](h1/ens/pes_ens/doc/pes_ens_explained.md) · [`pes_ens_theory.md`](h1/ens/pes_ens/doc/pes_ens_theory.md) |
| `pes_ens_consensus_prior` | [`pes_ens_consensus_prior_explained.md`](h1/ens/pes_ens_consensus_prior/doc/pes_ens_consensus_prior_explained.md) · [`pes_ens_consensus_prior_theory.md`](h1/ens/pes_ens_consensus_prior/doc/pes_ens_consensus_prior_theory.md) |
| general benchmark | [`h1/general/README.md`](h1/general/README.md) |

---

## 📈 Current stress benchmark

The active benchmark in [`h1/general/`](h1/general/) evaluates two suites:

- Individual models: `pes_base`, `pes_ql`, `pes_dql`, `pes_dqn`, `pes_rdqn`,
  `pes_a2c` and `pes_trf`.
- Ensembles: `pes_ens`, `pes_ens_sprb`, `pes_ens_accq`,
  `pes_ens_consensus`, `pes_ens_consensus_prior` and `pes_ens_trf_guard`.

The scenario catalogue contains the reference (`sev_base`, the 64 fixed
sequences), 21 perturbations of severity, sequence length, both jointly and
the block structure, and 5 held-out replicas (`heldout_s1`–`heldout_s5`):
fresh draws from the reference distributions that never took part in
training or optimisation. The comparison report defines
the normalized performance, Welch $p$, Cohen $d$, degradation and KL metrics,
and links the generated heatmaps and sequence curves.

```powershell
cd h1
python -m general.scripts.benchmark run --suite both
python -m general.scripts.analysis
python -m general.scripts.figures
python -m general.scripts.random_baseline
python -m general.scripts.agent_internals
```

The full comparison is documented in
[`h1/general/doc/comparacion_modelos.md`](h1/general/doc/comparacion_modelos.md).

### Seed variability

The delivered models come from the best Optuna trial, so their training seed
is not an unbiased draw. This study retrains the six trainable individual
models (Q-Learning, Double Q-Learning, DQN, recurrent DQN, A2C and the
Transformer) with 10 new seeds each (1001–1010), evaluates them greedily on the
reference and the held-out replicas, and reports the delivered model apart
from the new seeds. It trains each package as its `train_*` script does (the
recurrent DQN takes its architecture from `config/CONFIG.py`) but calls the
training functions directly — never the `train_*` scripts, which overwrite
`inputs/` — and writes only to `h1/general/work/seeds/`. Runs are resumable.

```powershell
powershell -ExecutionPolicy Bypass -File utils\scripts\run_seed_variability.ps1   # detached run (from the root)

cd h1
python -m general.scripts.seed_variability run        # same run in the foreground (6 models, seeds 1001-1010)
python -m general.scripts.seed_variability run --models trf --seeds 1006 1007   # any subset
python -m general.scripts.seed_variability summary    # -> results/individual/seed_variability.json
```

Approximate cost on a laptop CPU per seed: 1–3 min (tabular), 20–25 min (DQN),
about 30 min (A2C) and 3–7 h (recurrent DQN, Transformer). DQN, the recurrent
DQN and the Transformer are single-threaded by design (determinism), so seeds
can run in parallel as separate processes with disjoint `--models`/`--seeds`;
A2C uses several threads and runs in its own process. Status: 60 seeds
finished (10 per model). Per-seed models are versioned in
`h1/general/work/seeds/` (binaries via Git LFS; run `git lfs pull`).

**Ensembles.** `h1/general/scripts/ensemble_seeds.py` keeps each ensemble's
rule and tuned parameters and swaps its members for retrained seeds (via
`MPES_MODEL_ROOT`): ten member-seed combinations (best, worst and closest to
the mean of every member, plus seven random ones) shared by the six ensembles,
evaluated on the reference and the held-out replicas exactly as the benchmark
runs them. The delivered ensemble is evaluated too as a control (it reproduces
the benchmark cells). One ensemble must not run in two processes at once.

```powershell
cd h1
python -m general.scripts.ensemble_seeds plan       # -> results/ensemble/ensemble_seeds_plan.json
python -m general.scripts.ensemble_seeds run --ensembles pes_ens pes_ens_sprb
python -m general.scripts.ensemble_seeds summary    # -> results/ensemble/ensemble_seeds.json
```

**Long runs on a laptop** (`utils/scripts/seed_run/`, from the repository root). Each *lane* is a hidden
console that runs its jobs one after another, detached from VS Code, at below-normal priority and with logical
CPUs 0–1 left free, so the machine stays usable. Do not launch more lanes than free CPUs, and never run one
ensemble in two lanes.

```powershell
powershell -ExecutionPolicy Bypass -File utils\scripts\seed_run\power_on.ps1        # no sleep on AC (saves settings)
powershell -ExecutionPolicy Bypass -File utils\scripts\seed_run\launch_lanes.ps1 -Module seed_variability `
    -Lane '--models rdqn --seeds 1001;--models rdqn --seeds 1002', '--models a2c --seeds 1001 1002'
powershell -ExecutionPolicy Bypass -File utils\scripts\seed_run\launch_lanes.ps1 -Module ensemble_seeds `
    -Lane '--ensembles pes_ens', '--ensembles pes_ens_sprb'
powershell -ExecutionPolicy Bypass -File utils\scripts\seed_run\status.ps1          # progress
powershell -ExecutionPolicy Bypass -File utils\scripts\seed_run\stop.ps1            # stop cleanly (resumable)
powershell -ExecutionPolicy Bypass -File utils\scripts\seed_run\power_restore.ps1   # restore power settings
```

---

## 📝 Thesis

```powershell
cd writings
python audit\audit.py                          # audit + pdflatex/bibtex -> writings/out/
python audit\audit.py --no-tex                 # audit only
python auxiliar\scripts\sync_figures.py        # copy generated h1/ figures into 02_Images
python auxiliar\scripts\ensemble_decisions.py  # decision frequencies of the fixed ensemble rules
python auxiliar\scripts\rebuild_thesis.py      # wrapper around audit.py that checks the PDF in out/
python auxiliar\scripts\build_results_context.py   # regenerate claude_context/mpes_resultados.{md,json}
```

The audit report is written to
[`writings/audit/AUDIT.md`](writings/audit/AUDIT.md).

Every number in the thesis comes from `h1/general/results/` or from one of the
analysis scripts in [`writings/auxiliar/scripts/`](writings/auxiliar/scripts/)
(run from the repository root):

| Script | Output |
|--------|--------|
| `heldout_gap.py` | Reference vs. held-out gap of the 13 models |
| `fixed_rule.py` | Fixed rule $a = \min(S + k, R)$ calibrated with the optimum |
| `weighted_ens_oracle.py` | Weighted ensemble rules vs. the exact optimal allocation |
| `weighted_ens_sensitivity.py` | One-at-a-time sensitivity of the weighted ensemble (Table 5.7) |
| `trf_vs_ens.py` | Each ensemble vs. the individual Transformer in generalisation |
| `policy_vs_rule.py` | Learned policies vs. the $S + k$ rule (Appendix E.2, defence slide) |
| `seed_comparison.py` | Seed-to-seed spread vs. model gaps (after `seed_variability summary`) |

The thesis is currently being revised after the committee pre-reviews; the
status of each observation is tracked in
[`dict/correcciones.md`](dict/correcciones.md).

---

## 📄 License

Private repository — all rights reserved.
