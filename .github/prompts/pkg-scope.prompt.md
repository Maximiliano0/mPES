# Package Scope — `<PKG>`

> Last updated: 2026-09-01

Restrict the chat to a **single package** in the mPES workspace.

## Workspace layout

Packages live under `h1/` (active) or `h2/` (experimental, suspended),
grouped by algorithm family:

```
h1/
  tabular/   # value-based tabular RL
    pes_base, pes_ql, pes_dql
  ml/        # deep / neural RL
    pes_dqn, pes_rdqn, pes_a2c, pes_trf
  general/   # cross-model OOD benchmark harness
h2/
  tabular_uq/
    ql_uq    # experimental Q-Learning + UQ (suspended)
utils/       # shared scripts and config (Windows only)
```

Throughout this prompt, `<PKG>` is the short package name and `<GROUP>`
is its parent directory (`tabular` or `ml`) within `<LINE>` (`h1` or `h2`).
Run commands always use `python -m <GROUP>.<PKG>` (e.g. `python -m ml.pes_dqn`)
with `<LINE>/` as the current working directory (e.g. `cd h1`).

## Usage

When invoking this prompt, specify the target package name. Examples:

```
@pkg-scope pes_base       # h1/tabular/pes_base
@pkg-scope pes_ql         # h1/tabular/pes_ql
@pkg-scope pes_dql        # h1/tabular/pes_dql
@pkg-scope pes_dqn        # h1/ml/pes_dqn
@pkg-scope pes_rdqn       # h1/ml/pes_rdqn
@pkg-scope pes_a2c        # h1/ml/pes_a2c
@pkg-scope pes_trf        # h1/ml/pes_trf
@pkg-scope ql_uq          # h2/tabular_uq/ql_uq (suspended)
@pkg-scope utils
```

Throughout this prompt, `<PKG>` refers to the package name provided by the
user.

## Directive

Work **exclusively** on `<LINE>/<GROUP>/<PKG>/`.  Do **not** read, modify, or
reference any other package in this workspace.

## Available Packages

| Line | Group | Package | Algorithm | Key files |
|------|-------|---------|-----------|-----------|
| `h1` | `tabular` | `pes_base` | Tabular Q-Learning (baseline) | `ext/pandemic.py`, `ext/train_rl.py` |
| `h1` | `tabular` | `pes_ql` | Q-Learning + Bayesian optimisation (Optuna) | `ext/optimize_rl.py` |
| `h1` | `tabular` | `pes_dql` | Double Q-Learning, ε-decay warm-up, PBRS | `ext/pandemic.py`, `ext/optimize_rl.py` |
| `h1` | `ml` | `pes_dqn` | Deep Q-Network (experience replay + target net) | `ext/dqn_model.py`, `ext/train_dqn.py`, `ext/optimize_dqn.py` |
| `h1` | `ml` | `pes_rdqn` | Recurrent DQN (LSTM over trial history) | `ext/rdqn_model.py`, `ext/train_rdqn.py`, `ext/optimize_rdqn.py` |
| `h1` | `ml` | `pes_a2c` | Advantage Actor-Critic (A2C, actor + critic nets) | `ext/ac_model.py`, `ext/train_a2c.py`, `ext/optimize_a2c.py` |
| `h1` | `ml` | `pes_trf` | Causal Transformer encoder + DQN (sliding window) | `ext/transformer_model.py`, `ext/train_transformer.py`, `ext/optimize_tr.py` |
| `h2` | `tabular_uq` | `ql_uq` | Q-Learning + uncertainty quantification (suspended) | `ext/pandemic.py` |
| — | — | `utils` | Shared helpers (Windows scripts) | `win/run_bayesian_opt.ps1`, `config/.pylintrc` |

## Discovery — Build the Package Map

Before starting any task, **automatically** discover the package structure:

1. List `<GROUP>/<PKG>/`, `<GROUP>/<PKG>/ext/`, `<GROUP>/<PKG>/src/`, and
   `<GROUP>/<PKG>/doc/`.
2. Read `<GROUP>/<PKG>/config/CONFIG.py` to learn all tuneable constants.
3. Skim `<GROUP>/<PKG>/__init__.py` for re-exports and global setup.
4. Identify the core algorithm files in `ext/` and support modules in `src/`.

### Common layout (most packages follow this)

```
<GROUP>/<PKG>/
├── __init__.py            # Config re-exports, ANSI codes, numpy/TF setup
├── __main__.py            # Experiment entry point (blocks / sequences / trials)
├── config/
│   └── CONFIG.py          # All tuneable constants
├── doc/                   # Markdown + HTML documentation
├── ext/                   # Core algorithms (Gym env, training, optimisation)
├── inputs/                # Generated data (date-stamped subdirs)
├── outputs/               # Logs and results (date-stamped subdirs)
└── src/
    ├── exp_utils.py       # Severity calculations, sequence helpers
    ├── log_utils.py       # Dual-stream logging (console + file)
    ├── pygameMediator.py  # Pygame UI bridge
    ├── result_formatter.py # Matplotlib result plots
    └── terminal_utils.py  # Rich console output (header, section, info…)
```

> **Note:** `ext/` contents vary by package — always list the directory to
> discover the actual files.

## Rules

1. Follow all conventions from `copilot-instructions.md` (style, imports,
   docstrings, quality gates).
2. Every code change must pass **pyright** (0 errors) and **pylint** (10.00/10)
   against `<GROUP>/<PKG>/`.
3. Keep documentation (`doc/*.md`) consistent with source — update if code
   changes affect documented behaviour. Re-export with
   `python utils/scripts/_export_html.py <PKG>` (the script resolves the
   group automatically).
4. Do **not** cross-reference or import from other packages (except `utils`
   when used as a shared dependency, and the `ml/pes_ens` ensemble which
   intentionally loads sibling `.keras` artefacts from disk).
