<div align="center">

# 🦠 mPES — *Multiple Pandemic Experiment Suite*

**A reinforcement-learning benchmark for resource allocation under uncertainty.**

[![Python](https://img.shields.io/badge/python-3.12-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-ff6f00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-4.7-1a73e8.svg)](https://optuna.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.2-7c4dff.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/license-private-lightgrey.svg)](#-license)

</div>

> The repository is split into three experiment lines: `h1/` is the active and
> validated line, while `h2/` and `h3/` are experimental or suspended branches.

---

## ✨ Highlights

- 🧠 **Eleven package folders under `h1/`** — three tabular, four neural, and four ensemble variants.
- 📊 **22-scenario Under Stress Experiments** ([`h1/general/`](h1/general/)) over six benchmarked models.
- 🔬 **Bayesian hyperparameter optimisation** via Optuna for the learnable models.
- 🌍 **Windows-first** — Python 3.12 and the `win_mpes_env` virtual environment.
- 📚 **Documentation per package** (Markdown with KaTeX-ready formulas).

---

## 📦 Packages

Packages are grouped by algorithm family under the active workspace lines.

### `h1/tabular/` — value-based tabular RL

| Package | Algorithm | Key files |
|---------|-----------|-----------|
| `pes_base` | Tabular Q-Learning *(baseline)* | [`ext/pandemic.py`](h1/tabular/pes_base/ext/pandemic.py), [`ext/train_rl.py`](h1/tabular/pes_base/ext/train_rl.py) |
| `pes_ql`   | Q-Learning + Bayesian optimisation | [`ext/optimize_rl.py`](h1/tabular/pes_ql/ext/optimize_rl.py) |
| `pes_dql`  | Double Q-Learning + ε-decay warm-up + PBRS | [`ext/pandemic.py`](h1/tabular/pes_dql/ext/pandemic.py), [`ext/optimize_rl.py`](h1/tabular/pes_dql/ext/optimize_rl.py) |

### `h1/ml/` — deep & neural RL

| Package | Algorithm | Key files |
|---------|-----------|-----------|
| `pes_dqn`  | Deep Q-Network (replay + target net) | [`ext/dqn_model.py`](h1/ml/pes_dqn/ext/dqn_model.py), [`ext/train_dqn.py`](h1/ml/pes_dqn/ext/train_dqn.py), [`ext/optimize_dqn.py`](h1/ml/pes_dqn/ext/optimize_dqn.py) |
| `pes_rdqn` | Recurrent DQN (LSTM over trial history) | [`ext/rdqn_model.py`](h1/ml/pes_rdqn/ext/rdqn_model.py), [`ext/train_rdqn.py`](h1/ml/pes_rdqn/ext/train_rdqn.py), [`ext/optimize_rdqn.py`](h1/ml/pes_rdqn/ext/optimize_rdqn.py) |
| `pes_a2c`  | Advantage Actor-Critic (separate actor + critic nets) | [`ext/ac_model.py`](h1/ml/pes_a2c/ext/ac_model.py), [`ext/train_a2c.py`](h1/ml/pes_a2c/ext/train_a2c.py), [`ext/optimize_a2c.py`](h1/ml/pes_a2c/ext/optimize_a2c.py) |
| `pes_trf`  | Causal Transformer encoder + DQN (sliding window) | [`ext/transformer_model.py`](h1/ml/pes_trf/ext/transformer_model.py), [`ext/train_transformer.py`](h1/ml/pes_trf/ext/train_transformer.py), [`ext/optimize_tr.py`](h1/ml/pes_trf/ext/optimize_tr.py) |

### `h1/ens/` — ensemble models

| Package | Algorithm | Key files |
|---------|-----------|-----------|
| `pes_ens_sprb` | Confidence-weighted soft voting ensemble | [`__main__.py`](h1/ens/pes_ens_sprb/__main__.py), [`ext/ensemble.py`](h1/ens/pes_ens_sprb/ext/ensemble.py), [`ext/optimize_ens.py`](h1/ens/pes_ens_sprb/ext/optimize_ens.py) |
| `pes_ens_accq` | Confidence-weighted action/Q-value ensemble | [`__main__.py`](h1/ens/pes_ens_accq/__main__.py), [`ext/ensemble.py`](h1/ens/pes_ens_accq/ext/ensemble.py), [`ext/optimize_ens.py`](h1/ens/pes_ens_accq/ext/optimize_ens.py) |
| `pes_ens_trf_guard` | Transformer-first confidence-gated ensemble | [`__main__.py`](h1/ens/pes_ens_trf_guard/__main__.py), [`ext/ensemble.py`](h1/ens/pes_ens_trf_guard/ext/ensemble.py), [`ext/optimize_ens.py`](h1/ens/pes_ens_trf_guard/ext/optimize_ens.py) |
| `pes_ens_consensus` | Confidence consensus with agreement bonus and disagreement penalty | [`__main__.py`](h1/ens/pes_ens_consensus/__main__.py), [`ext/ensemble.py`](h1/ens/pes_ens_consensus/ext/ensemble.py), [`ext/optimize_ens.py`](h1/ens/pes_ens_consensus/ext/optimize_ens.py) |

### Support directories

| Path | Purpose |
|------|---------|
| [`h1/general/`](h1/general/) | Cross-model Under Stress Experiments harness (22 scenarios × 6 benchmarked models) |
| [`h2/`](h2/) | Experimental branch with suspended work |
| [`h3/`](h3/) | Additional experimental branch (read-only for current active workflow) |
| [`utils/`](utils/) | Windows scripts, requirements, and lint config |
| [`h1/general/doc/`](h1/general/doc/) | Cross-package theoretical comparison material |

---

## 🗂️ Package layout

```text
<group>/<pkg>/                 # <group> ∈ { tabular, ml }
├── __init__.py          # Config re-exports, ANSI codes, numpy/TF setup
├── __main__.py          # Experiment entry point (blocks/sequences/trials)
├── config/CONFIG.py     # All tuneable constants
├── doc/                 # Markdown & HTML documentation
├── ext/                 # Core algorithms (Gym env, training, optimisation)
├── inputs/              # Generated data (date-stamped subdirs)
├── outputs/             # Logs and results (date-stamped subdirs)
└── src/                 # Support modules
    ├── exp_utils.py        # Severity calculations, sequence helpers
    ├── log_utils.py        # Dual-stream logging (console + file)
    ├── pygameMediator.py   # Pygame UI bridge
    ├── result_formatter.py # Matplotlib result plots
    └── terminal_utils.py   # Rich console output (header, section, info…)
```

---

## ⚙️ Setup

### Requirements

| Dependency | Version | Dependency | Version |
|------------|---------|------------|---------|
| Python     | 3.12    | Optuna     | 4.7.0   |
| TensorFlow | 2.21.0  | Gymnasium  | 1.2.3   |
| Keras      | 3.13.2  | matplotlib | 3.10.8  |
| NumPy      | 2.4.3   | scipy      | 1.17.1  |
| Pygame     | 2.5.2   |            |         |

> Full list in [`utils/config/requirements.txt`](utils/config/requirements.txt).

### Virtual environment

```bash
# Windows (PowerShell)
python -m venv win_mpes_env
win_mpes_env\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r utils/config/requirements.txt
```

### Environment variables

Set these **before** running training or optimisation:

| Variable | Value | Purpose |
|----------|-------|---------|
| `VIRTUAL_ENV`           | path to active venv | Prevents `__init__.py` interactive prompt |
| `PYTHONIOENCODING`      | `utf-8`             | Avoids `UnicodeEncodeError` on Windows    |
| `TF_ENABLE_ONEDNN_OPTS` | `0`                 | Suppresses oneDNN info messages           |

---

## ▶️ Usage

> `h1/` is a plain directory, not a Python package (no `__init__.py` at that
> level). Run every command below with `h1/` as the current working
> directory: `cd h1`.

### Run an experiment

```bash
cd h1
python -m tabular.pes_base    # Tabular Q-Learning (baseline)
python -m tabular.pes_ql      # Q-Learning  (Optuna-tuned)
python -m tabular.pes_dql     # Double Q-Learning + PBRS

python -m ml.pes_dqn          # Deep Q-Network
python -m ml.pes_rdqn         # Recurrent DQN (LSTM)
python -m ml.pes_a2c          # Advantage Actor-Critic
python -m ml.pes_trf          # Causal Transformer DQN
```

Ensembles de evaluación:

```bash
python -m ens.pes_ens_sprb
python -m ens.pes_ens_accq
python -m ens.pes_ens_trf_guard
python -m ens.pes_ens_consensus
```

### Train an agent

```bash
cd h1

# --- Tabular models (Q-table episodes) ---
python -m tabular.pes_base.ext.train_rl   1000000
python -m tabular.pes_ql.ext.train_rl     1000000
python -m tabular.pes_dql.ext.train_rl    1000000

# --- Deep models (gradient steps / episodes) ---
python -m ml.pes_dqn.ext.train_dqn          175000
python -m ml.pes_rdqn.ext.train_rdqn        175000
python -m ml.pes_a2c.ext.train_a2c          175000
python -m ml.pes_trf.ext.train_transformer  175000

```

### Evaluate and optimise ensembles

The ensembles do not train neural networks. They load the trained DQN, RDQN
and Transformer models, read the local `initial_severity.csv`,
`sequence_lengths.csv` and `best_params.json` files, and evaluate fixed
sequences. Optimisation is independent for each ensemble and writes its own
`best_params.json`.

```bash
cd h1
python -m ens.pes_ens_sprb.ext.optimize_ens 50
python -m ens.pes_ens_accq.ext.optimize_ens 50
python -m ens.pes_ens_trf_guard.ext.optimize_ens 50
python -m ens.pes_ens_consensus.ext.optimize_ens 50
```

### Bayesian hyperparameter optimisation

> **Known issue:** [`utils/win/run_bayesian_opt.ps1`](utils/win/run_bayesian_opt.ps1)
> still resolves package paths as `tabular\<pkg>` / `ml\<pkg>` (pre-`h1/h2`
> layout) and needs a code fix before these commands work again. This is a
> `.ps1` script issue, not a documentation issue.

```bash
# Windows (PowerShell)
.\utils\win\run_bayesian_opt.ps1 bayesian 100
.\utils\win\run_bayesian_opt.ps1 dqn       30
.\utils\win\run_bayesian_opt.ps1 ac        30
```

---

## 🦠 The Pandemic Scenario

| Aspect | Definition |
|--------|------------|
| **State space**  | `[resources_left (9–39), trial_number (0–10), severity (0–9)]` → 3 410 states |
| **Action space** | allocate 0–10 resources (11 discrete actions; over-allocation masked with `-1e9`) |
| **Dynamics**     | `new_severity = 1.4 · initial_severity − 0.4 · resources_allocated` |
| **Reward**       | negative cumulative severity (the agent **minimises** total damage) |

### Experiment hierarchy

```text
Experiment (1)
└── Block (8)
    └── Sequence / Map (8)
        └── Trial / City (3–10)
            └── Resource Decision (0–10)
```

**~ 360 total trials per experiment (~ 45 per block).**

---

## 📚 Documentation

Each documented package ships its own in-depth Markdown documentation under
[`h1/<group>/<pkg>/doc/`](h1/ml/pes_dqn/doc/) (Spanish, under `h1/`). The ensemble
packages currently expose source and runtime folders but do not have package-level
Markdown documentation. The cross-package
theoretical comparison lives at
[`h1/general/doc/comparacion_modelos.md`](h1/general/doc/comparacion_modelos.md).
All maths uses **inline KaTeX** (`$ … $`) so it renders natively on
GitHub.

| Package | Theory | Implementation guide |
|---------|--------|----------------------|
| `pes_base`  | [`theory_rl.md`](h1/tabular/pes_base/doc/theory_rl.md) | [`explained_pes.md`](h1/tabular/pes_base/doc/explained_pes.md) · [`how_to_train_and_test.md`](h1/tabular/pes_base/doc/how_to_train_and_test.md) |
| `pes_ql`    | [`pes_ql_theory.md`](h1/tabular/pes_ql/doc/pes_ql_theory.md) | [`pes_ql_explained.md`](h1/tabular/pes_ql/doc/pes_ql_explained.md) |
| `pes_dql`   | [`pes_dql_theory.md`](h1/tabular/pes_dql/doc/pes_dql_theory.md) | [`pes_dql_explained.md`](h1/tabular/pes_dql/doc/pes_dql_explained.md) |
| `pes_dqn`   | [`pes_dqn_theory.md`](h1/ml/pes_dqn/doc/pes_dqn_theory.md) | [`pes_dqn_explained.md`](h1/ml/pes_dqn/doc/pes_dqn_explained.md) |
| `pes_rdqn`  | [`pes_rdqn_theory.md`](h1/ml/pes_rdqn/doc/pes_rdqn_theory.md) | [`pes_rdqn_explained.md`](h1/ml/pes_rdqn/doc/pes_rdqn_explained.md) |
| `pes_a2c`   | [`pes_a2c_theory.md`](h1/ml/pes_a2c/doc/pes_a2c_theory.md) | [`pes_a2c_explained.md`](h1/ml/pes_a2c/doc/pes_a2c_explained.md) |
| `pes_trf`   | [`pes_trf_theory.md`](h1/ml/pes_trf/doc/pes_trf_theory.md) | [`pes_trf_explained.md`](h1/ml/pes_trf/doc/pes_trf_explained.md) |
| `pes_ens_sprb` | Source-only | [`ext/evaluate_ens.py`](h1/ens/pes_ens_sprb/ext/evaluate_ens.py) |
| `pes_ens_accq` | Source-only | [`ext/evaluate_ens.py`](h1/ens/pes_ens_accq/ext/evaluate_ens.py) |
| `pes_ens_trf_guard` | Source-only | [`ext/evaluate_ens.py`](h1/ens/pes_ens_trf_guard/ext/evaluate_ens.py) |
| `pes_ens_consensus` | Source-only | [`ext/evaluate_ens.py`](h1/ens/pes_ens_consensus/ext/evaluate_ens.py) |

---

## 📈 Cross-model Under Stress Experiments — `h1/general/`

The [`h1/general/`](h1/general/) harness evaluates six active models against a
catalogue of **22 under-stress scenarios** (severity, length,
joint, structural families) and aggregates the results into nine
`matrix_*.csv` files plus four publication-quality heatmaps under
[`h1/general/results/`](h1/general/results/). The active sweep is **132 cells**
(6 models × 22 scenarios).

> **Pending regeneration.** `benchmark_report.md`, the heatmaps, and the raw
> per-model JSON cells belong to the six-model catalogue and are independent
> of the current ensemble packages. Re-run the sweep below to regenerate them
> for the current 6-model catalogue.

### Run the sweep

```bash
cd h1
python -m general.scripts.orchestrate    # full 132-cell sweep (resumable)
python -m general.scripts.progress       # live progress bars + ETA
python -m general.scripts.aggregate      # build matrix_*.csv from raw/
python -m general.scripts.plot_matrix    # render heatmaps (PNG + PDF)
python -m general.scripts.report         # generate benchmark_report.md
```

### Heatmaps

All four heatmaps are written as both **`.png`** (300 dpi raster) and
**`.pdf`** (vector, TrueType-embedded) for direct inclusion in
publications. Cells use fixed colour-scale limits so figures from
different sweeps are directly comparable; clipped values are flagged
in-cell (`≤-10` in the Welch heatmap).

| Metric | Heatmap | Colour map | Scale |
|--------|---------|-----------|-------|
| Global mean performance | `general/results/heatmap_global_mean.png` | `viridis` | auto-bounded |
| Stress degradation (Δ vs baseline) | `general/results/heatmap_stress_degradation.png` | `RdBu_r` (diverging) | symmetric around 0 |
| Welch test, log₁₀(p) | `general/results/heatmap_welch_logp.png` | `magma_r` | clipped to `[-10, 0]` |
| Action-distribution KL | `general/results/heatmap_action_kl.png` | `cividis` | `LogNorm` |

---

## 📄 License

Private repository — all rights reserved.
