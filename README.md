# mPES — multiple Pandemic Experiment Scenario

Multi-package Python workspace for **reinforcement-learning** experiments on a
resource-allocation task (the *Pandemic Scenario*).

An agent must distribute **39 resources** across ~360 trials (8 blocks × 8
sequences × 3–10 trials) to minimise disease severity. Five algorithmic variants
share the same experiment framework, making side-by-side comparison
straightforward.

## Packages

| Package | Algorithm | Key files |
|---------|-----------|-----------|
| `pes_base` | Tabular Q-Learning (baseline) | `ext/pandemic.py`, `ext/train_rl.py` |
| `pes_ql` | Q-Learning + Bayesian optimisation (Optuna) | `ext/optimize_rl.py` |
| `pes_dql` | Double Q-Learning, ε-decay warm-up, PBRS | `ext/pandemic.py`, `ext/optimize_rl.py` |
| `pes_dqn` | Deep Q-Network (experience replay + target net) | `ext/dqn_model.py`, `ext/train_dqn.py`, `ext/optimize_dqn.py` |
| `pes_rdqn` | Recurrent DQN (LSTM over trial history) | `ext/rdqn_model.py`, `ext/train_rdqn.py`, `ext/optimize_rdqn.py` |
| `pes_a2c` | Advantage Actor-Critic (A2C) | `ext/ac_model.py`, `ext/train_a2c.py`, `ext/optimize_a2c.py` |
| `pes_trf` | Causal Transformer encoder + DQN (sliding window) | `ext/transformer_model.py`, `ext/train_transformer.py`, `ext/optimize_tr.py` |
| `pes_ens` | Ensemble (soft voting of pes_dqn + pes_a2c + pes_rdqn + pes_trf) | `ext/ensemble_model.py` |
| `utils` | Shared helpers (shell scripts) | `linux/run_bayesian_opt.sh`, `win/run_bayesian_opt.ps1`, `config/.pylintrc` |

## Package layout

```
<pkg>/
├── __init__.py          # Config re-exports, ANSI codes, numpy/TF setup
├── __main__.py          # Experiment entry point (blocks/sequences/trials)
├── config/CONFIG.py     # All tuneable constants
├── doc/                 # Markdown & HTML documentation
├── ext/                 # Core algorithms (Gym env, training, optimisation)
├── inputs/              # Generated data (date-stamped subdirs)
├── outputs/             # Logs and results (date-stamped subdirs)
└── src/                 # Support modules
    ├── exp_utils.py       # Severity calculations, sequence helpers
    ├── log_utils.py       # Dual-stream logging (console + file)
    ├── pygameMediator.py  # Pygame UI bridge
    ├── result_formatter.py# Matplotlib result plots
    └── terminal_utils.py  # Rich console output (header, section, info…)
```

## Setup

### Requirements

| Dependency | Version |
|------------|---------|
| Python | 3.12 (Windows & Linux) |
| TensorFlow | 2.21.0 |
| Keras | 3.13.2 |
| NumPy | 2.4.3 |
| matplotlib | 3.10.8 |
| scipy | 1.17.1 |
| Optuna | 4.7.0 |
| Gymnasium | 1.2.3 |
| Pygame | 2.5.2 |

### Virtual environment

```bash
# Linux
python3 -m venv linux_mpes_env
source linux_mpes_env/bin/activate

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
| `VIRTUAL_ENV` | Path to active venv | Prevents `__init__.py` interactive prompt |
| `PYTHONIOENCODING` | `utf-8` | Avoids `UnicodeEncodeError` on Windows |
| `TF_ENABLE_ONEDNN_OPTS` | `0` | Suppresses oneDNN info messages |

## Usage

### Run an experiment

```bash
python -m pes_base     # Tabular Q-Learning
python -m pes_ql       # Q-Learning (Bayesian-tuned)
python -m pes_dql      # Double Q-Learning + PBRS
python -m pes_dqn      # Deep Q-Network
python -m pes_rdqn     # Recurrent DQN (LSTM)
python -m pes_a2c      # Advantage Actor-Critic (A2C)
python -m pes_trf      # Causal Transformer DQN
python -m pes_ens      # Ensemble (soft voting of dqn+a2c+rdqn+trf, no training)
```

### Train an agent

```bash
# Tabular Q-Learning (1M episodes)
python -m pes_base.ext.train_rl 1000000

# Deep Q-Network
python -m pes_dqn.ext.train_dqn 500000

# Actor-Critic
python -m pes_a2c.ext.train_a2c 500000
```

### Bayesian hyperparameter optimisation

```bash
# Linux
./utils/linux/run_bayesian_opt.sh bayesian 100    # pes_ql, 100 trials
./utils/linux/run_bayesian_opt.sh dql 100         # pes_dql
./utils/linux/run_bayesian_opt.sh dqn 30          # pes_dqn
./utils/linux/run_bayesian_opt.sh rdqn 30         # pes_rdqn
./utils/linux/run_bayesian_opt.sh ac 30           # pes_a2c
./utils/linux/run_bayesian_opt.sh transformer 30  # pes_trf

# Windows (PowerShell)
.\utils\win\run_bayesian_opt.ps1 bayesian 100
.\utils\win\run_bayesian_opt.ps1 dqn 30
.\utils\win\run_bayesian_opt.ps1 ac 30
```

## The Pandemic Scenario

- **State space**: `[resources_left (9–39), trial_number (0–10), severity (0–9)]` → 3,410 states
- **Action space**: allocate 0–10 resources (11 discrete actions; allocations exceeding remaining resources are masked with sentinel `-1e9`)
- **Dynamics**: `new_severity = 1.4 × initial_severity − 0.4 × resources_allocated`
- **Reward**: negative cumulative severity (the agent minimises total damage)

## Experiment structure

```
Experimento (1)
├── Bloque (8)
│   ├── Secuencia / Mapa (8)
│   │   ├── Trial / Ciudad (3–10)
│   │   │   └── Decisión de Recursos (0–10)
```

- **1** experiment → **8** blocks → **8** sequences per block → **3–10** trials per sequence
- ~360 total trials per experiment (~45 per block)

## Documentation

Each package ships its own in-depth Markdown documentation under
`<pkg>/doc/` (Spanish), with matching HTML renderings (KaTeX math,
dark-mode CSS). Regenerate the HTML with:

```bash
python utils/scripts/_export_html.py            # all packages
python utils/scripts/_export_html.py pes_dqn    # single package
```

To update both source-level docstrings and the Markdown docs from the
current code, use the workflow described in
`.github/prompts/update-pkg-docs.prompt.md`.

## License

Private repository — all rights reserved.
