# GitHub Copilot Instructions

> Last updated: 2026-04-20

## Project Overview

**mPES** (multiple Pandemic Experiment Scenario) is a multi-package Python
workspace for reinforcement-learning experiments on a resource-allocation task
(the "Pandemic Scenario"). Each package implements a different algorithmic
variant sharing the same experiment framework.

| Package | Algorithm | Key files |
|---------|-----------|-----------|
| `pes_base` | Tabular Q-Learning (baseline) | `ext/pandemic.py`, `ext/train_rl.py` |
| `pes_ql` | Q-Learning + Bayesian hyperparam optimisation (Optuna) | `ext/optimize_rl.py` |
| `pes_dql` | Double Q-Learning, ε-decay warm-up, PBRS | `ext/pandemic.py`, `ext/optimize_rl.py` |
| `pes_dqn` | Deep Q-Network (experience replay + target net) | `ext/dqn_model.py`, `ext/train_dqn.py`, `ext/optimize_dqn.py` |
| `pes_rdqn` | Recurrent DQN (LSTM over trial history) | `ext/rdqn_model.py`, `ext/train_rdqn.py`, `ext/optimize_rdqn.py` |
| `pes_a2c` | Advantage Actor-Critic (A2C, separate actor + critic nets) | `ext/ac_model.py`, `ext/train_a2c.py`, `ext/optimize_a2c.py` |
| `pes_trf` | Causal Transformer encoder + RL | `ext/transformer_model.py`, `ext/train_transformer.py`, `ext/optimize_tr.py` |
| `pes_ens` | Ensemble (soft voting of pes_dqn + pes_a2c + pes_rdqn + pes_trf) | `ext/ensemble_model.py` |
| `utils` | Shared helpers (shell scripts) | `linux/run_bayesian_opt.sh`, `win/run_bayesian_opt.ps1`, `config/.pylintrc` |

### Common package layout

```
<PKG>/
├── __init__.py          # Config re-exports, ANSI codes, numpy/TF setup
├── __main__.py          # Experiment entry point (blocks/sequences/trials)
├── config/CONFIG.py     # All tuneable constants
├── doc/                 # Markdown documentation
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

## Cross-Platform Setup

This project runs on **two machines**:

| Machine | OS | Virtual env | Python |
|---------|----|-------------|--------|
| Windows | Windows 10 | `win_mpes_env` | 3.12 |
| Linux | Ubuntu / GNOME | `linux_mpes_env` | 3.12 |

### Virtual environment activation

Detect the OS and activate the correct virtual environment:

**Linux / macOS:**
```bash
source linux_mpes_env/bin/activate
```

**Windows (PowerShell):**
```powershell
win_mpes_env\Scripts\Activate.ps1
```

**Windows (cmd):**
```cmd
win_mpes_env\Scripts\activate.bat
```

> **Tip:** When writing scripts or commands, always use OS detection or
> provide both variants. Never hard-code a single virtual environment name.

### Environment variables (required for deep-learning packages)

These must be set **before** launching optimisation or training processes:

| Variable | Value | Purpose |
|----------|-------|---------|
| `VIRTUAL_ENV` | Path to active venv | Prevents `__init__.py` "Press ENTER" prompt |
| `PYTHONIOENCODING` | `utf-8` | Avoids `UnicodeEncodeError` on redirected output (Windows cp1252) |
| `TF_ENABLE_ONEDNN_OPTS` | `0` | Suppresses oneDNN info messages |

### Path conventions

- Use `os.path.join()` in Python code — never hard-code `/` or `\`.
- In shell scripts, provide **both** `.sh` (Bash) and `.ps1` (PowerShell) variants.
- All paths in scripts must be **relative** to the workspace root.

### Key dependencies (both environments)

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
requirements. Avoid adding unrelated functionality. Each package is
self-contained — do **not** cross-reference between packages.

## Testing

Ensure that any new code is properly tested and does not break existing
functionality. Write unit tests where appropriate.

## Documentation

- Every public function and class must have a docstring (NumPy-style).
- Maintain clear and concise documentation for any new features or changes.
- Update existing documentation if necessary to reflect changes.

## Version Control

Do not commit or push changes directly. Instead, provide code suggestions
and improvements through pull requests for review by the project maintainers.