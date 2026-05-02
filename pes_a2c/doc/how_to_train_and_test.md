# How to Train and Test the A2C Model

> Package: **pes_a2c** — Advantage Actor-Critic variant of the Pandemic Experiment Scenario

---

## Prerequisites

| Requirement | Detail |
|-------------|--------|
| Python | 3.12 (Windows) / 3.12 (Linux) |
| Virtual environment | `win_mpes_env` (Windows) or `linux_mpes_env` (Linux) |
| TensorFlow | 2.21.0 (CPU — no GPU required) |
| Input data | `inputs/initial_severity.csv` and `inputs/sequence_lengths.csv` |

Activate the environment once per terminal session:

**Linux / macOS:**

```bash
source linux_mpes_env/bin/activate
```

**Windows (PowerShell):**

```powershell
win_mpes_env\Scripts\Activate.ps1
```

---

## 1. Training the A2C Agent

### 1.1 Quick Start

```bash
python3 -m pes_a2c.ext.train_a2c
```

This runs the **full training pipeline** with default settings (250 000 episodes,
hiperparámetros del mejor trial de la optimización bayesiana del 2026-04-23).

### 1.2 Custom Episode Count

Pass the number of episodes as the first argument:

```bash
python3 -m pes_a2c.ext.train_a2c 200000
```

### 1.3 What Happens During Training

The pipeline proceeds through these stages:

1. **Load data** — reads `initial_severity.csv` and `sequence_lengths.csv`
   from `pes_a2c/inputs/`.
2. **Random-player baseline** — runs 64 evaluation sequences with a random
   agent and saves performance plots.
3. **A2C training** — trains the Actor and Critic networks for the configured
   number of episodes using on-policy batched updates with GAE(λ) advantage
   estimation, gradient clipping, advantage normalisation, entropy bonus,
   **on-policy masked softmax sampling** (no ε-greedy overlay; actions are
   sampled from the Actor distribution renormalised over feasible actions),
   PBRS reward shaping, infeasible-action masking inside the policy gradient,
   and cosine annealing learning rate scheduling.
4. **Save artefacts** — writes the trained Actor Keras model (`.keras`),
   reward history (`rewards.npy`), and a configuration summary to a dated
   subdirectory.
5. **Evaluate** — runs the trained agent on the same 64 sequences and
   generates performance, confidence, and cumulative-performance plots.

### 1.4 Training Output Files

All outputs are saved to `pes_a2c/inputs/<YYYY-MM-DD>_A2C_TRAIN/`:

| File | Description |
|------|-------------|
| `ac_actor_<date>.keras` | Trained Actor Keras model (date-stamped copy) |
| `rewards_<date>.npy` | Average reward every 10 000 episodes |
| `training_config_<date>.txt` | Full hyperparameter record |
| `random_player_*.png` | Baseline performance plots |
| `ac_agent_rewards_vs_episodes_<date>.png` | Reward convergence curve |
| `ac_agent_sequence_performance_<date>.png` | Final severity per sequence |
| `ac_agent_normalised_performance_<date>.png` | Normalised performance |
| `ac_agent_cumulative_performance_<date>.png` | Cumulative performance trend |
| `ac_agent_confidences_<date>.png` | Entropy-based confidence scores |

Additionally, the Actor model and rewards are **copied** to the standard
paths consumed by the experiment runner:

- `pes_a2c/inputs/ac_actor.keras`
- `pes_a2c/inputs/rewards.npy`

### 1.5 Default Hyperparameters

Los hiperparámetros activos en `ext/train_a2c.py` corresponden al **trial #90**
de la optimización bayesiana ejecutada el 2026-04-23 (mean_perf = 0.887236).
Los valores de `CONFIG.py` (columna *CONFIG*) se actualizaron para coincidir
con ese trial #90 y se usan también como línea base para la semilla
(*warm-start*) de Optuna.

| Parámetro | Valor activo (`train_a2c.py`) | CONFIG | Fuente |
|-----------|------------------------------|--------|--------|
| Actor hidden layers | `[128]` | `[128]` | `CONFIG.AC_ACTOR_HIDDEN_UNITS` |
| Critic hidden layers | `[128]` | `[128]` | `CONFIG.AC_CRITIC_HIDDEN_UNITS` |
| Actor learning rate | 6.476 × 10⁻⁴ | 6.476 × 10⁻⁴ | `train_a2c.py` (Bayesian opt trial #90) |
| Critic learning rate | 4.299 × 10⁻³ | 4.299 × 10⁻³ | `train_a2c.py` (Bayesian opt trial #90) |
| Entropy coefficient | 5.278 × 10⁻³ | 5.278 × 10⁻³ | `train_a2c.py` (Bayesian opt trial #90) |
| Discount γ | 0.8545 | 0.8545 | `train_a2c.py` (Bayesian opt trial #90) |
| Initial ε *(legacy, unused)* | 0.0 | 0.0 | `CONFIG.AC_EPSILON_INITIAL` |
| Min ε *(legacy, unused)* | 0.0 | 0.0 | `CONFIG.AC_EPSILON_MIN` |
| Episodes | 250 000 | — | `train_a2c.py` (default; CLI override) |
| Random seed | 42 | 42 | `CONFIG.SEED` |

> **Note**: `AC_EPSILON_INITIAL` and `AC_EPSILON_MIN` are kept in CONFIG
> for API compatibility but are **not used** — training samples actions
> on-policy from the Actor distribution.  Exploration is provided
> intrinsically by the softmax randomness plus the entropy bonus.

**Hiperparámetros de mejora** (configurables en `CONFIG.py`):

| Parámetro | Valor por defecto | CONFIG |
|-----------|-------------------|--------|
| Warmup ratio *(legacy, unused)* | 0.05 (5%) | `AC_WARMUP_RATIO` |
| Target ratio *(legacy, unused)* | 0.60 (60%) | `AC_TARGET_RATIO` |
| PBRS coeff (β) | 0.153 | `AC_PENALTY_COEFF` |
| GAE(λ) | 0.913 | `AC_GAE_LAMBDA` |
| Max grad norm | 1.200 | `AC_MAX_GRAD_NORM` |
| LR min ratio (cosine) | 0.237 | `AC_LR_MIN_RATIO` |
| Spend cost coeff | 0.0104 | `AC_SPEND_COST_COEFF` |
| Last-action bias (logit) | -1.389 | `AC_LAST_ACTION_BIAS` |
| Optimization mode | `full` | `AC_OPTIMIZE_MODE` |

### 1.6 Expected Training Time

On a modest CPU (e.g. Intel i3-6006U @ 2 GHz, 4 threads), approximately
**40–80 minutes** for 250 000 episodes with CPU optimisations enabled
(`tf.function` per-trial compilation, no confidence computation during training,
TensorFlow thread-pool tuning).  Training time scales linearly with
episode count.

A2C may be slightly slower than DQN per episode due to maintaining two
networks and performing a full gradient update per episode (vs. DQN's
periodic mini-batch updates).

### 1.7 CPU Training Optimisations

The following optimisations are active by default and require no
configuration:

| Optimisation | Effect | Location |
|-------------|--------|----------|
| `tf.function` per-trial compiled training | Eliminates eager overhead for per-episode updates | `ext/pandemic.py` |
| `optimizer.build()` pre-initialisation | Creates optimiser variables outside `tf.function` graph | `ext/pandemic.py` |
| `tf.constant` scalar hyper-parameters | Prevents costly retracing across Optuna trials | `ext/pandemic.py` |
| Infeasible-action masking during training | Aligns training with masked inference; policy gradient trains only over legal actions | `ext/pandemic.py`, `ext/ac_model.py` |
| Skip confidence computation | `compute_confidence=False` skips entropy/masking per step | `ext/pandemic.py` |
| TF thread-pool tuning | `intra_op=0` (auto), `inter_op=2` for multi-core CPUs | `ext/ac_model.py` |
| Actor-only inference | Only Actor loaded at experiment time (Critic discarded) | `src/pygameMediator.py` |

To re-enable confidence tracking during training (at slightly slower speed):

```python
rewards, actor, confs = A2CTraining(
    env, actor_lr, critic_lr, discount, entropy_coeff,
    eps, min_eps, episodes,
    ...,
    compute_confidence=True,   # ← enables meta-cognitive observation
)
```

---

## 2. Bayesian Hyperparameter Optimisation (Optional)

If the default hyperparameters are not satisfactory, you can run an
automated search using Optuna:

```bash
python3 -m pes_a2c.ext.optimize_a2c 30
```

The integer argument is the number of trials (default: 30).

### 2.0.1 Optimization Mode

The optimizer supports two modes selectable via `--mode` (or
`CONFIG.AC_OPTIMIZE_MODE`):

- **`full`** (default in CONFIG) — optimizes all 14 sampled hyperparameters
  (8 base + 6 improvements).
- **`improvements_only`** — fixes the 8 base hyperparameters at the
  CONFIG.py values (trial #90) and only optimizes the 6 improvement
  parameters.  This reduces the search from 14D to 6D.

```bash
# Optimize all sampled hyperparameters (default mode)
python3 -m pes_a2c.ext.optimize_a2c 50

# Optimize only improvement params
python3 -m pes_a2c.ext.optimize_a2c 50 --mode improvements_only
```

### 2.1 Resume a Previous Search

Optimisation state is stored in an SQLite database.  To resume:

```bash
python3 -m pes_a2c.ext.optimize_a2c 50 --resume 2026-03-02
```

This loads the study from `inputs/<date>_BAYESIAN_OPT/optuna_study_<date>.db`
and runs additional trials until the total reaches 50.

### 2.2 Optimisation Outputs

Saved to `pes_a2c/inputs/<YYYY-MM-DD>_BAYESIAN_OPT/`:

| File | Description |
|------|-------------|
| `ac_best_<date>.keras` | Best Actor model found |
| `rewards_best_<date>.npy` | Reward history of the best trial |
| `optimization_results_<date>.txt` | Full report with all trial results |
| `optimization_history_<date>.png` | Convergence plot |
| `hyperparameter_importances_<date>.png` | Parameter importance ranking |
| `optuna_study_<date>.db` | SQLite database (resumable) |

### 2.3 A2C-Specific Search Space

The A2C optimisation searches over up to 14 sampled hyperparameters
(8 base + 6 improvement).  In **`full`** mode (default) the base
hyperparameters sampled are:

```
actor_lr             ∈ [1e-4, 1e-2]         (log scale)
critic_lr            ∈ [1e-4, 1e-2]         (log scale)
discount_factor      ∈ [0.85, 0.995]
entropy_coeff        ∈ [0.0, 0.1]           (linear; 0 disables the bonus)
num_episodes         ∈ [50000, 250000]      (step = 25000)
actor_hidden_dim     ∈ {32, 64, 128, 256}   (categorical)
critic_hidden_dim    ∈ {32, 64, 128, 256}   (categorical)
n_hidden_layers      ∈ {1, 2, 3}
```

The ε-greedy parameters (`epsilon_initial`, `epsilon_min`) are **fixed
at 0** and not sampled — training is now pure on-policy softmax sampling,
so ε has no effect.

The 6 always-sampled improvement parameters are:

```
penalty_coeff        ∈ [0.0, 0.3]           (linear; 0 disables PBRS)
gae_lambda           ∈ [0.90, 0.99]
max_grad_norm        ∈ [0.3, 1.5]
lr_min_ratio         ∈ [0.05, 0.25]
spend_cost_coeff     ∈ [0.0, 0.05]          (per-action training-only cost; 0 disables)
last_action_bias     ∈ [-2.0, 0.0]          (init logit of the max-spend action)
```

`warmup_ratio` and `target_ratio` are also fixed (0 / 1) since they only
govern the unused legacy ε-decay schedule.

In **`improvements_only`** mode, only the 6 improvement parameters are
sampled; the 8 base parameters are held at CONFIG values.

Each trial uses an independent seed (`SEED + trial.number + 1`) so that
repeat sampling of the same configuration produces independent
stochastic replicates — essential for TPE to estimate objective noise.

Compared to DQN: A2C adds two separate learning rates (Actor and Critic)
and `entropy_coeff`; it omits `batch_size`, `replay_buffer_size`, and
`target_sync_freq` since A2C is on-policy.

### 2.4 Running on Google Colab Pro+

If the local machine does not have enough RAM (the full optimisation needs
~900 MB per process), you can run it on Google Colab Pro+ using the
launcher notebook `utils/colab/launch.ipynb`.  See
`utils/colab/colab_workflow.md` for the complete step-by-step guide.

The launcher notebook now exposes a `USE_GPU` parameter:

| `USE_GPU` | Meaning | When to use |
|-----------|---------|-------------|
| `0` (default) | Force TF onto CPU on Colab | Recommended for the **Colab → local PC reproducibility pipeline** (§ 2.5).  CPU runs are bit-exact across Colab and the local Windows/Linux dev machines. |
| `1` | Use the Colab GPU runtime | Faster optimisation when you only need the trained `.keras` file (you will copy it directly to `pes_a2c/inputs/`).  Local retraining will diverge in the LSB. |

When `USE_GPU=1`, also switch the runtime via **Runtime → Change runtime
type → T4 / A100 GPU** before running cells.  The launcher prints a
warning if no GPU is detected.

`TF_DETERMINISTIC_OPS=1` and `TF_CUDNN_DETERMINISTIC=1` are exported
by `utils/colab/setup_colab.sh` so that even on GPU the same seed
produces the same trajectory on the same hardware.

### 2.5 Reproducing a Colab Best Trial Locally

The end-to-end pipeline is:

1. **(Colab)** Run the optimisation:
   ```python
   PKG = 'ac'; N_TRIALS = 100; USE_GPU = 0  # CPU for reproducibility
   ```
   The launcher writes everything to
   `Drive/MyDrive/mPES/pes_a2c/<DATE>_BAYESIAN_OPT/` including
   `optuna_study_<DATE>.db` (the SQLite study) and
   `optimization_results_<DATE>.txt` (now contains a
   **REPRODUCIBILITY** block listing the best `trial_seed` and the
   exact `train_a2c.py` command).

2. **(Local)** Copy the run directory from Drive into the workspace:
   ```text
   Drive/MyDrive/mPES/pes_a2c/<DATE>_BAYESIAN_OPT/
                     →  pes_a2c/inputs/<DATE>_BAYESIAN_OPT/
   ```

3. **(Local)** Train with the best params + best seed automatically:
   ```bash
   python -m pes_a2c.ext.train_a2c --from-best <DATE>
   ```
   > **Auto-load:** Omitting `--from-best` makes `train_a2c.py`
   > auto-discover the most recent `<DATE>_BAYESIAN_OPT/` directory
   > under `inputs/` and reproduce its best trial.  This matches
   > `pes_ql` / `pes_dql` behavior — a bare
   > `python -m pes_a2c.ext.train_a2c` is enough after a Bayesian run.
   > To fall back to the baked-in `CONFIG.py` values, remove or rename
   > the `_BAYESIAN_OPT/` directory.

   The script reads `optuna_study_<DATE>.db`, applies the best trial's
   hyperparameters and `trial_seed`, and runs an extra **parity
   evaluation** that uses the *exact same* `qf` as the optimiser.  Its
   `mean_perf` is printed alongside the Optuna-reported value with the
   absolute difference, classified as:

   | `|Δ|` | Meaning |
   |-------|---------|
   | `< 1e-6` | bit-exact match |
   | `< 1e-3` | within FP tolerance (different TF build / hardware) |
   | `≥ 1e-3` | mismatch — see Troubleshooting (§ 4) |

4. **(Local)** Run the experiment with the freshly trained Actor:
   ```bash
   python -m pes_a2c
   ```

> **Shortcut.** If you do not need to retrain locally, simply copy
> `ac_best_<DATE>.keras` over `pes_a2c/inputs/ac_actor.keras` and
> jump to step 4.  This is the **only** way to guarantee `mean_perf`
> equality when Colab used a GPU runtime (`USE_GPU=1`).

---

## 3. Testing the A2C Agent (Running the Experiment)

### 3.1 Verify Training Files Exist

Before running the experiment, ensure these files are present:

```
pes_a2c/inputs/ac_actor.keras    ← trained Actor Keras model
pes_a2c/inputs/rewards.npy       ← reward history
```

Both are automatically created by the training pipeline (§ 1).

### 3.2 Run the Experiment

```bash
python3 -m pes_a2c
```

This launches the full experiment lifecycle:

1. **Validates** the Actor model (loads it, checks parameter count and output
   shape).
2. **Sets up** the experiment session with logging and dated output folders.
3. **Iterates** through 8 blocks × 8 sequences × 3–10 trials per sequence.
4. At each trial, the A2C agent:
   - Normalises the current state `(resources_left, trial_no, severity)` to
     `[0, 1]³`.
   - Performs a **forward pass** through the Actor to obtain
     $\pi_\theta(a \mid s)$.
   - Selects the action with the highest probability (masking infeasible
     allocations where `action > resources_left`).
   - Computes **meta-cognitive confidence** from the entropy of the policy
     distribution — a theoretically grounded measure.
   - Simulates human-like **response timing** based on confidence.
5. **Saves** results (performance JSON, visualisation PNG, response logs) to
   `pes_a2c/outputs/<YYYY-MM-DD>_A2C_AGENT/`.

### 3.3 Experiment Outputs

| File | Description |
|------|-------------|
| `PES__<session_id>.txt` | Experiment configuration snapshot |
| `PES_responses_<session_id>.txt` | Trial-by-trial responses |
| `PES_log_<session_id>.txt` | Console log (dual-stream) |
| `PES_movement_log_<session_id>.npy` | Movement data |
| `PES_results_<session_id>.json` | Performance summary JSON |
| `PES_results_<session_id>.png` | Performance plots |

### 3.4 Configuration

All experiment parameters are in `config/CONFIG.py`.  Key settings:

```python
PLAYER_TYPE = {  # Decision maker type - SELECT ONE
    1: 'RL_AGENT',     # Tabular Q-Learning agent (fallback / baseline comparison)
    2: 'DQN_AGENT',    # Deep Q-Network (experience replay + target net)
    3: 'A2C_AGENT'     # Advantage Actor-Critic (A2C) agent
}[3]                   # <-- change index to select
AC_MODEL_ACTOR_FILE = 'ac_actor.keras'  # Actor model filename in inputs/
NUM_BLOCKS = 8
NUM_SEQUENCES = 8
```

### 3.5 Reproducibility check (`raw_mean_perf`)

After saving the results JSON, `__main__.py` prints a one-line consistency
check between the freshly computed mean performance and the value reported
by the Bayesian optimiser:

```text
raw_mean_perf = 0.898652  (std=0.045213, n=64)
best_params.json mean_perf = 0.898652  |Δ| = 0.000000
```

`raw_mean_perf` is the mean of `MyPerformances` across the full
`NUM_BLOCKS × NUM_SEQUENCES` session. `best_params.json['mean_perf']` is
the value Optuna recorded for the winning trial (loaded from
`inputs/best_params.json`). A small `|Δ|` (< 1e-3) confirms full
reproducibility; a larger gap usually means the run used a different
`SEED`, a different input CSV, or a different compute device (for
example a GPU vs CPU kernel).

---

## 4. Troubleshooting

### Model file not found

```
A2C Actor model file not found!
Expected path: .../pes_a2c/inputs/ac_actor.keras
```

**Solution:** Run training first:

```bash
python3 -m pes_a2c.ext.train_a2c
```

### Rewards file not found

Same as above — the training pipeline generates both `ac_actor.keras` and
`rewards.npy`.

### TensorFlow warnings

TensorFlow GPU/CUDA warnings are suppressed by default
(`TF_CPP_MIN_LOG_LEVEL=3`).  If you still see them, they are harmless — the
model runs on CPU.

### Poor agent performance

- Try increasing episodes: `python3 -m pes_a2c.ext.train_a2c 200000`
- Run Bayesian optimisation to find better hyperparameters:
  `python3 -m pes_a2c.ext.optimize_a2c 50`
- Adjust `AC_ENTROPY_COEFF` in `CONFIG.py` — higher values encourage more
  exploration, lower values favour exploitation.
- Check that `CONFIG.SEED = 42` for reproducible results.

### Slow training on CPU

The default configuration is already optimised for CPU.  If training is
still too slow:

- Reduce episode count: `python3 -m pes_a2c.ext.train_a2c 50000`
- Verify `compute_confidence=False` is set (default) in `train_a2c.py`

### Actor vs. Critic model confusion

Only the **Actor** model (`ac_actor.keras`) is needed for the experiment.
The Critic is used only during training and is not saved to the standard
inputs path.

---

## 5. Quick Reference

| Task | Command |
|------|---------|
| Activate environment | `source linux_mpes_env/bin/activate` |
| Train A2C (default) | `python3 -m pes_a2c.ext.train_a2c` |
| Train A2C (custom) | `python3 -m pes_a2c.ext.train_a2c 200000` |
| Optimise hyperparams | `python3 -m pes_a2c.ext.optimize_a2c 30` |
| Resume optimisation | `python3 -m pes_a2c.ext.optimize_a2c 50 --resume YYYY-MM-DD` |
| Run experiment | `python3 -m pes_a2c` |
