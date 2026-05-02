# Package Scope — `<PKG>`

> Last updated: 2026-05-02

Restrict the chat to a **single package** in the mPES workspace.

## Usage

When invoking this prompt, specify the target package name. Examples:

```
@pkg-scope pes_base
@pkg-scope pes_ql
@pkg-scope pes_dql
@pkg-scope pes_dqn
@pkg-scope pes_rdqn
@pkg-scope pes_a2c
@pkg-scope pes_trf
@pkg-scope pes_ens
@pkg-scope utils
```

Throughout this prompt, `<PKG>` refers to the package name provided by the
user.

## Directive

Work **exclusively** on `<PKG>/`.  Do **not** read, modify, or reference
any other package in this workspace.

## Available Packages

| Package | Algorithm | Key files |
|---------|-----------|-----------|
| `pes_base` | Tabular Q-Learning (baseline) | `ext/pandemic.py`, `ext/train_rl.py` |
| `pes_ql` | Q-Learning + Bayesian optimisation (Optuna) | `ext/optimize_rl.py` |
| `pes_dql` | Double Q-Learning, ε-decay warm-up, PBRS | `ext/pandemic.py`, `ext/optimize_rl.py` |
| `pes_dqn` | Deep Q-Network (experience replay + target net) | `ext/dqn_model.py`, `ext/train_dqn.py`, `ext/optimize_dqn.py` |
| `pes_rdqn` | Recurrent DQN (LSTM over trial history) | `ext/rdqn_model.py`, `ext/train_rdqn.py`, `ext/optimize_rdqn.py` |
| `pes_a2c` | Advantage Actor-Critic (A2C, actor + critic nets) | `ext/ac_model.py`, `ext/train_a2c.py`, `ext/optimize_a2c.py` |
| `pes_trf` | Causal Transformer encoder + DQN (sliding window) | `ext/transformer_model.py`, `ext/train_transformer.py`, `ext/optimize_tr.py` |
| `pes_ens` | Ensemble (soft voting of pes_dqn + pes_a2c + pes_rdqn + pes_trf) | `ext/ensemble_model.py` |
| `utils` | Shared helpers (shell scripts) | `linux/run_bayesian_opt.sh`, `win/run_bayesian_opt.ps1`, `config/.pylintrc` |

## Discovery — Build the Package Map

Before starting any task, **automatically** discover the package structure:

1. List `<PKG>/`, `<PKG>/ext/`, `<PKG>/src/`, and `<PKG>/doc/`.
2. Read `<PKG>/config/CONFIG.py` to learn all tuneable constants.
3. Skim `<PKG>/__init__.py` for re-exports and global setup.
4. Identify the core algorithm files in `ext/` and support modules in `src/`.

### Common layout (most packages follow this)

```
<PKG>/
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
   against `<PKG>/`.
3. Keep documentation (`doc/*.md`) consistent with source — update if code
   changes affect documented behaviour.
4. Do **not** cross-reference or import from other packages (except `utils`
   when used as a shared dependency).
