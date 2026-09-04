# GitHub Copilot Instructions

> Last updated: 2026-09-02

## Project Overview

**mPES** (Multiple Pandemic Experiment Suite) is a multi-package Python
workspace for reinforcement-learning experiments on a resource-allocation task
(the "Pandemic Scenario"). The repository currently includes three experiment
lines: `h1/` (active and validated), `h2/` (experimental and suspended), and
`h3/` (prototype / research staging). Each line shares the same package
scaffolding and implements one or more algorithmic variants.

### `h1/` — active line

| Package | Algorithm | Key files |
|---------|-----------|-----------|
| `tabular/pes_base` | Tabular Q-Learning (baseline) | `ext/pandemic.py`, `ext/train_rl.py` |
| `tabular/pes_ql` | Q-Learning + Bayesian hyperparam optimisation (Optuna) | `ext/optimize_rl.py` |
| `tabular/pes_dql` | Double Q-Learning, ε-decay warm-up, PBRS | `ext/pandemic.py`, `ext/optimize_rl.py` |
| `ml/pes_dqn` | Deep Q-Network (experience replay + target net) | `ext/dqn_model.py`, `ext/train_dqn.py`, `ext/optimize_dqn.py` |
| `ml/pes_rdqn` | Recurrent DQN (LSTM over trial history) | `ext/rdqn_model.py`, `ext/train_rdqn.py`, `ext/optimize_rdqn.py` |
| `ml/pes_a2c` | Advantage Actor-Critic (A2C, separate actor + critic nets) | `ext/ac_model.py`, `ext/train_a2c.py`, `ext/optimize_a2c.py` |
| `ml/pes_trf` | Causal Transformer encoder + RL | `ext/transformer_model.py`, `ext/train_transformer.py`, `ext/optimize_tr.py` |
| `ens/pes_ens_sprb` | Confidence-weighted soft voting over action probabilities (DQN + RDQN + TRF) | `ext/ensemble.py`, `ext/evaluate_ens.py`, `ext/optimize_ens.py` |
| `ens/pes_ens_accq` | Confidence-weighted action voting with Q-value tie-breaking (DQN + RDQN + TRF) | `ext/ensemble.py`, `ext/evaluate_ens.py`, `ext/optimize_ens.py` |
| `ens/pes_ens_trf_guard` | Transformer-first confidence-gated fallback ensemble | `ext/ensemble.py`, `ext/evaluate_ens.py`, `ext/optimize_ens.py` |
| `ens/pes_ens_consensus` | Confidence consensus with agreement and disagreement terms | `ext/ensemble.py`, `ext/evaluate_ens.py`, `ext/optimize_ens.py` |
| `general/` | Cross-model Under Stress Experiments harness (22 scenarios × 6 models) | `scripts/orchestrate.py`, `scripts/aggregate.py`, `scripts/report.py` |

> The ensemble implementations combine `pes_dqn`, `pes_rdqn`, and `pes_trf`.
> `pes_ens_sprb` uses soft voting, `pes_ens_accq` uses action voting with
> normalized-Q tie-breaking, `pes_ens_trf_guard` gives the Transformer a
> confidence gate with fallback, and `pes_ens_consensus` rewards agreement
> while penalizing disagreement. All expose tunable parameters through
> `ext/optimize_ens.py` and evaluate through `ext/evaluate_ens.py`.
>
> Legacy folders `h1/ens/pes_ens/` and `h1/ens/pes_ens_consensus_prior/` are
> retained in the repository but are not part of the active benchmark workflow.

### `h2/` — experimental line (suspended)

| Package | Algorithm | Status |
|---------|-----------|--------|
| `tabular_conf/ql_conf` | Tabular Q-Learning + experimental configuration variants | Suspended — staging only |

### `h3/` — prototype line

| Package | Algorithm | Status |
|---------|-----------|--------|
| `tabular_uq/ql_uq` | Q-Learning + uncertainty quantification prototype | Experimental — not part of active benchmark |

### Shared

| Path | Contents |
|------|----------|
| `utils/` | Windows shell scripts (`win/`), lint/type-check config (`config/`), helper scripts (`scripts/`) |

### Workspace directory structure

```
h1/                 # Active experiment line
├── tabular/        #   Value-based tabular RL (Q-Learning variants)
│   ├── pes_base/   #     Baseline tabular Q-Learning
│   ├── pes_ql/     #     Q-Learning + Bayesian optimisation
│   └── pes_dql/    #     Double Q-Learning + ε warm-up + PBRS
├── ml/             #   Deep / neural RL
│   ├── pes_dqn/    #     Deep Q-Network
│   ├── pes_rdqn/   #     Recurrent DQN (LSTM)
│   ├── pes_a2c/    #     Advantage Actor-Critic
│   ├── pes_trf/    #     Causal Transformer DQN
│   ├── pes_ens_sprb/ #   Confidence-weighted soft voting ensemble
│   ├── pes_ens_accq/ #   Confidence-weighted action/Q-value ensemble
│   ├── pes_ens_trf_guard/ # Transformer-first confidence-gated ensemble
│   └── pes_ens_consensus/ # Confidence consensus ensemble
├── ens/pes_ens/    #   Archived ensemble prototype
├── ens/pes_ens_consensus_prior/ # Archived consensus variant
└── general/        #   Cross-model Under Stress Experiments harness + comparison doc

h2/                  # Experimental line (suspended)
├── general/
├── tabular_conf/
│   └── ql_conf/    #   Experimental tabular Q-Learning variant
└── README.md

h3/                  # Prototype line
├── general/
├── tabular_uq/
│   └── ql_uq/      #   Research Q-Learning + UQ prototype
└── README.md

utils/               # Shared scripts and config (Windows only)
```

> **`h1/`, `h2/` and `h3/` are plain directories, not Python packages** — none
> has an `__init__.py` at its own level. Every `python -m ...` command below
> must be run from the relevant experiment line, e.g. `cd h1; python -m tabular.pes_base`.

### Run commands (from within `h1/`)

| Package | Command |
|---------|---------|
| `tabular/pes_base` | `python -m tabular.pes_base` |
| `tabular/pes_ql` | `python -m tabular.pes_ql` |
| `tabular/pes_dql` | `python -m tabular.pes_dql` |
| `ml/pes_dqn` | `python -m ml.pes_dqn` |
| `ml/pes_rdqn` | `python -m ml.pes_rdqn` |
| `ml/pes_a2c` | `python -m ml.pes_a2c` |
| `ml/pes_trf` | `python -m ml.pes_trf` |
| `ens/pes_ens_sprb` | `python -m ens.pes_ens_sprb` |
| `ens/pes_ens_accq` | `python -m ens.pes_ens_accq` |
| `ens/pes_ens_trf_guard` | `python -m ens.pes_ens_trf_guard` |
| `ens/pes_ens_consensus` | `python -m ens.pes_ens_consensus` |

### Bayesian optimisation commands (from within `h1/`)

| Package | Command |
|---------|---------|
| `tabular/pes_ql` | `python -m tabular.pes_ql.ext.optimize_rl [n_trials]` |
| `tabular/pes_dql` | `python -m tabular.pes_dql.ext.optimize_rl [n_trials]` |
| `ml/pes_dqn` | `python -m ml.pes_dqn.ext.optimize_dqn [n_trials]` |
| `ml/pes_rdqn` | `python -m ml.pes_rdqn.ext.optimize_rdqn [n_trials]` |
| `ml/pes_a2c` | `python -m ml.pes_a2c.ext.optimize_a2c [n_trials]` |
| `ml/pes_trf` | `python -m ml.pes_trf.ext.optimize_tr [n_trials]` |
| `ens/pes_ens_sprb` | `python -m ens.pes_ens_sprb.ext.optimize_ens [n_trials]` |
| `ens/pes_ens_accq` | `python -m ens.pes_ens_accq.ext.optimize_ens [n_trials]` |
| `ens/pes_ens_trf_guard` | `python -m ens.pes_ens_trf_guard.ext.optimize_ens [n_trials]` |
| `ens/pes_ens_consensus` | `python -m ens.pes_ens_consensus.ext.optimize_ens [n_trials]` |

### Common package layout

```
<GROUP>/<PKG>/
├── __init__.py          # Config re-exports, ANSI codes, numpy/TF setup
├── __main__.py          # Experiment entry point (blocks/sequences/trials)
├── config/CONFIG.py     # All tuneable constants
├── doc/                 # Markdown documentation (+ HTML exports)
├── ext/                 # Core algorithms (Gym env, training, optimisation)
├── inputs/              # Generated data (date-stamped subdirs)
├── outputs/             # Logs and results (date-stamped subdirs)
└── src/                 # Support modules
  ├── exp_utils.py       # Severity calculations, sequence helpers
  ├── log_utils.py       # Dual-stream logging (console + file)
  ├── pygameMediator.py  # Pygame UI bridge
  ├── result_formatter.py # Matplotlib result plots
  └── terminal_utils.py  # Rich console output (header, section, info…)
```

## Windows Setup

This project runs on **Windows only** (Python 3.12, `win_mpes_env`). There is
no supported Linux environment; do not add `linux_mpes_env` or `.sh` variants
unless explicitly requested.

### Virtual environment activation

Activate from the repository root, then move into `h1/` (or `h2/`) before
running any `python -m ...` command:

**Windows (PowerShell):**
```powershell
win_mpes_env\Scripts\Activate.ps1
```

**Windows (cmd):**
```cmd
win_mpes_env\Scripts\activate.bat
```

### Environment variables (required for deep-learning packages)

These must be set **before** launching optimisation or training processes:

| Variable | Value | Purpose |
|----------|-------|---------|
| `VIRTUAL_ENV` | Path to active venv | Prevents `__init__.py` "Press ENTER" prompt |
| `PYTHONIOENCODING` | `utf-8` | Avoids `UnicodeEncodeError` on redirected output (Windows cp1252) |
| `TF_ENABLE_ONEDNN_OPTS` | `0` | Suppresses oneDNN info messages |

### Path conventions

- Use `os.path.join()` in Python code — never hard-code `/` or `\`.
- Shell scripts under `utils/` are Windows-only (`.ps1`); do not add `.sh`
  variants unless the project explicitly adds Linux support again.
- All paths in scripts must be **relative** to the workspace root.

### Key dependencies

| Package | Version |
|---------|---------|
| TensorFlow | 2.21.0 |
| Keras | 3.13.2 |
| numpy | 2.4.3 |
| matplotlib | 3.10.8 |
| scipy | 1.17.1 |
| optuna | 4.7.0 |
| gymnasium | 1.2.3 |
| pygame | 2.5.2 |

Full list in `utils/config/requirements.txt`.

## Code Style

| Rule | Standard |
|------|----------|
| Max line length | 120 characters |
| Indentation | 4 spaces (PEP 8) |
| Variable naming | `snake_case` **and** `PascalCase` accepted (scientific convention) |
| NumPy alias | `numpy` (never `np`) |
| Type hints | Use where practical; pyright must pass with 0 errors |
| Docstrings | NumPy-style, required on every public function and class |
| Unused vars | Prefix with `_` (e.g., `_fig, ax = plt.subplots()`) |

### Import conventions

- **`__init__.py`** may use `from .config.CONFIG import *` (wildcard re-export).
- **All other modules** must use explicit imports:
  ```python
  from .. import ANSI, INPUTS_PATH, VERBOSE   # ✅
  from .. import *                              # ❌ (except __init__.py)
  ```
- Section comments above import blocks:
  ```python
  ##########################
  ##  Imports externos    ##
  ##########################

  ##########################
  ##  Imports internos    ##
  ##########################
  ```

## Functionality

Implement features and fixes that align with the project's goals and
requirements. Avoid adding unrelated functionality.

Each package is self-contained — do **not** cross-reference between packages.
`h2/` is suspended: do not add training runs, benchmark sweeps, or results for
`h2` packages unless the user explicitly reactivates that line.

## Testing

Ensure that any new code is properly tested and does not break existing
functionality. Write unit tests where appropriate.

## Documentation

- Every public function and class must have a docstring (NumPy-style).
- Maintain clear and concise documentation for any new features or changes.
- Update existing documentation if necessary to reflect changes.
- Every `doc/` directory contains `<pkg>_explained.md` (usage guide) and
  `<pkg>_theory.md` (theoretical foundations) exported to `.html`.
- The cross-package comparison document is at `h1/general/doc/comparacion_modelos.md`.
- Use `utils/scripts/_export_docs_html.py` to re-export `.md` → `.html`.
- When editing `.md` files, ensure relative links and image paths resolve
  from the file's own location, and that any in-file table of contents
  matches the actual headings (GitHub auto-generates anchors from headings).

## Version Control

Do not commit or push changes directly. Instead, provide code suggestions
and improvements through pull requests for review by the project maintainers.