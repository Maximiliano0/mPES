# How to Train and Test the Double Q-Learning Agent

> Package: **pes_dql** — Double Q-Learning with exponential ε-decay warm-up
> and Potential-Based Reward Shaping (PBRS)

---

## Prerequisites

| Requirement | Detail |
|-------------|--------|
| Python | 3.12 (Windows) / 3.12 (Linux) |
| Virtual environment | `win_mpes_env` (Windows) or `linux_mpes_env` (Linux) |
| Optuna | 4.7.0 (included in `utils/config/requirements.txt`) |
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

## 1. Training the Double Q-Learning Agent

### 1.1 Quick Start

```bash
python3 -m pes_dql.ext.train_rl
```

This runs the **full training pipeline** with Bayesian-optimised
hyperparameters (860 000 episodes by default).

### 1.2 Custom Episode Count

Pass the number of episodes as the first argument:

```bash
python3 -m pes_dql.ext.train_rl 1000000
```

### 1.3 What Happens During Training

The pipeline proceeds through these stages:

1. **Load data** — reads `initial_severity.csv` and
   `sequence_lengths.csv` from `inputs/`.
2. **Random player baseline** — runs 64 sequences with uniformly random
   allocations and generates two baseline plots.
3. **Double Q-Learning training** — trains with three algorithmic
   improvements over standard Q-Learning (§ 1.7). Prints average reward
   every 10 000 episodes.
4. **Evaluation** — runs **two** evaluation passes on the same 64 fixed
   sequences:
   - A **deterministic pass** (`qf_eval`) that mirrors `optimize_rl.objective`
     bit-for-bit (same `-1e9` masking, same index clamping, no random
     draws). Its `mean_perf` is compared against the value recorded by the
     best Bayesian trial: an exact match (`abs(delta) < 1e-9`) confirms
     the pipeline reproduces the optimisation result. A mismatch raises a
     warning suggesting to check `seed`, `track_confidence`, and the CSV
     inputs.
   - A **stochastic pass** (`qf`) that uses `rl_agent_meta_cognitive` to
     also collect entropy-based confidence scores and human-like response
     timing for plotting.
5. **Save artefacts** — writes Q-table, reward history, config report,
   and visualisations to a dated subdirectory.

### 1.4 Training Output Files

All outputs are saved to `pes_dql/inputs/<YYYY-MM-DD>_RL_TRAIN/`:

| File | Description |
|------|-------------|
| `q_<date>.npy` | Trained Q-table — shape `(31, 11, 10, 11)` |
| `rewards_<date>.npy` | Average-reward history (one entry every 10 000 episodes) |
| `training_config_<date>.txt` | Summary of hyperparameters and settings |
| `confsrl_<date>.npy` | Confidence scores from evaluation |
| `random_player_sequence_performance_<date>.png` | Baseline severity per sequence |
| `random_player_normalised_performance_<date>.png` | Baseline normalised performance |
| `rl_agent_rewards_vs_episodes_<date>.png` | Learning curve |
| `rl_agent_sequence_performance_<date>.png` | Severity per sequence |
| `rl_agent_normalised_performance_<date>.png` | Performance (0–1) |
| `rl_agent_cumulative_performance_<date>.png` | Cumulative trend |
| `rl_agent_confidences_<date>.png` | Raw confidence scatter |
| `rl_agent_remapped_confidences_<date>.png` | Normalised confidence (0–1) |

Additionally, to consume the trained Q-table from the experiment runner
(`python3 -m pes_dql`), the standard paths must be populated:

- `pes_dql/inputs/q.npy`
- `pes_dql/inputs/rewards.npy`

Both `train_rl.py` and `optimize_rl.py` now **mirror** their best Q-table
and rewards into these paths automatically at the end of the run, in
addition to the dated copies. Manual promotion is no longer required.

### 1.5 Default Hyperparameters (Bayesian-Optimised)

`train_rl.py` resolves the hyperparameters in this order:

1. `pes_dql/inputs/best_params.json` — written by every successful
   `optimize_rl.py` run.
2. The newest `pes_dql/inputs/<date>_BAYESIAN_OPT/best_params_<date>.json`.
3. A hard-coded fallback (the values shown below) if neither file exists.

The JSON payload also carries the **per-trial seed** (`SEED + best_trial_number + 1`)
and `track_confidence=False`. `train_rl.py` consumes both so the resulting
Q-table reproduces the best Bayesian trial bit-for-bit and the local
deterministic `mean_perf` matches the value Optuna recorded for that trial.

The fallback values (used when no `best_params.json` is present):

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning rate α | 0.2593 | Q-table update step size |
| Discount factor γ | 0.9806 | Future-reward weighting |
| Initial ε | 0.8392 | Starting exploration probability |
| Minimum ε | 0.0799 | Final exploration probability |
| Episodes | 860 000 | Total training episodes |
| Warm-up ratio | 0.0240 | Fraction of episodes at constant ε |
| Target ratio | 0.5174 | Fraction where ε reaches ε_min |
| PBRS coefficient β | 0.2177 | Reward-shaping strength |
| Random seed | 42 | `CONFIG.SEED` |

### 1.6 Q-Table Dimensions

| Axis | Size | Meaning |
|------|------|---------|
| 0 | 31 | Resources left (0–30) |
| 1 | 11 | Trial number (0–10) |
| 2 | 10 | Severity (0–9), `MAX_SEVERITY = 9` |
| 3 | 11 | Actions (0–10 resources) |

**Total entries:** $31 \times 11 \times 10 \times 11 = 37{,}510$

During training, **two Q-tables** are maintained (`Q_A`, `Q_B`). The
saved Q-table is their average: $Q = \frac{Q_A + Q_B}{2}$.

### 1.7 Three Algorithmic Improvements

pes_dql extends the baseline with three improvements that preserve the
optimal policy while accelerating convergence:

#### A. Double Q-Learning (van Hasselt, 2010)

Eliminates maximisation bias by maintaining two independent Q-tables.
Action selection uses one table; value evaluation uses the other:

$$Q_A(s,a) \leftarrow Q_A(s,a) + \alpha \bigl[ r + \gamma \, Q_B\bigl(s', \arg\max_{a'} Q_A(s',a')\bigr) - Q_A(s,a) \bigr]$$

With probability 0.5, the roles of $Q_A$ and $Q_B$ are swapped.

#### B. Exponential ε-Decay with Warm-up

Replaces linear decay with three distinct phases:

| Phase | Episodes | ε behaviour |
|-------|----------|-------------|
| Warm-up | 0 → $W$ | Constant at $\varepsilon_0$ |
| Decay | $W$ → $T$ | Exponential: $\varepsilon_0 \cdot \lambda^{t-W}$ |
| Exploitation | $T$ → $N$ | ≈ $\varepsilon_{\min}$ |

Where $W = \text{warmup\_ratio} \times N$, $T = \text{target\_ratio} \times N$,
and $\lambda = (\varepsilon_{\min} / \varepsilon_0)^{1/((T - W))}$ is
auto-computed (equivalent to $\lambda = (\varepsilon_{\min}/\varepsilon_0)^{1/((\text{target\_ratio} - \text{warmup\_ratio}) \cdot N)}$).

#### C. Potential-Based Reward Shaping (Ng et al., 1999)

Augments the environment reward with a shaping term:

$$r' = r + \beta \bigl( \gamma \, \Phi(s') - \Phi(s) \bigr)$$

where $\Phi(s) = -\sum_i s_i$ (negative sum of current severities; the env
already clamps each $s_i$ to $\geq 0$). By the invariance theorem, the
optimal policy is unchanged for any $\beta \geq 0$.

### 1.8 Key Differences from Other Packages

| Aspect | pes_base | pes_ql | pes_dql |
|--------|----------|--------|---------|
| Algorithm | Q-Learning | Q-Learning | **Double** Q-Learning |
| ε-decay | Linear | Linear | **Exponential + warm-up** |
| Reward shaping | None | None | **PBRS** |
| Hyperparameters | Hand-tuned | Bayesian (5 params) | Bayesian (**8 params**) |
| Default episodes | 1 000 000 | 900 000 | 860 000 |

---

## 2. Bayesian Hyperparameter Optimisation

### 2.1 Quick Start

```bash
python3 -m pes_dql.ext.optimize_rl 50
```

This launches 50 Optuna trials. Each trial trains a Double Q-Learning
agent with sampled hyperparameters and evaluates it on 64 fixed sequences.

### 2.2 Custom Trial Count

```bash
python3 -m pes_dql.ext.optimize_rl 100
```

### 2.3 Resume a Previous Search

```bash
python3 -m pes_dql.ext.optimize_rl 100 --resume 2026-03-02
```

Optuna loads the existing SQLite database and continues from where it
stopped. If the requested total is already reached, no new trials run.

### 2.4 Search Space (8 Parameters)

| Parameter | Range | Scale |
|-----------|-------|-------|
| `learning_rate` | [0.05, 0.30] | Log |
| `discount_factor` | [0.90, 0.999] | Linear |
| `epsilon_initial` | [0.50, 1.00] | Linear |
| `epsilon_min` | [0.01, 0.10] | Log |
| `num_episodes` | [150 000, 500 000] (step 10 000) | Linear (int) |
| `warmup_ratio` | [0.02, 0.15] | Linear |
| `target_ratio` | [0.40, 0.80] | Linear |
| `penalty_coeff` | [1e-4, 0.30] | Log |

**Sampler:** `TPESampler(seed=42, n_startup_trials=10, multivariate=True, group=True)`.
The `multivariate` + `group` options let TPE model correlations between
hyperparameters that are coupled in the algorithm (warm-up vs target ratio,
ε_initial vs ε_min) instead of treating each axis independently.

**Per-trial seed:** `seed=SEED + trial.number + 1` is passed to `QLearning()`,
so every trial is individually reproducible while still differing from
its siblings (avoids fitting a single random init).

**Confidence tracking off:** `track_confidence=False` is passed during
optimisation to skip the per-step entropy computation, which dominates
wall-time at hundreds of thousands of episodes and never feeds the learner.

**Objective:** Maximise mean normalised performance (0–1) over 64
evaluation sequences. Infeasible actions are masked with a very negative
sentinel ($-10^9$) before `argmax` so the policy never selects an action
larger than the remaining budget.

### 2.5 Optimisation Outputs

Saved to `pes_dql/inputs/<YYYY-MM-DD>_BAYESIAN_OPT/`:

| File | Description |
|------|-------------|
| `q_best_<date>.npy` | Q-table from the best trial |
| `rewards_best_<date>.npy` | Reward history from the best trial |
| `best_params_<date>.json` | Sidecar: hyperparameters + `trial_seed` + `mean_perf` + `best_trial_number` |
| `repro_fingerprint_<date>.json` | Environment fingerprint (numpy/python versions, CSV SHA-256, git commit, SEED) |
| `optimization_results_<date>.txt` | Full report with all trial results |
| `optimization_history_<date>.png` | Convergence plot |
| `hyperparameter_importances_<date>.png` | Parameter importance ranking (fANOVA) |
| `optuna_study_<date>.db` | SQLite database (resumable) |

A copy of `best_params.json`, `q.npy` and `rewards.npy` is also mirrored
to `pes_dql/inputs/` so the experiment runner picks them up without
further steps.

### 2.5.1 Reproducibility Fingerprint

`pes_dql/ext/repro.py` captures a snapshot of the runtime that determines
the bit-for-bit outcome of `QLearning(seed=...)`:

| Field | Value |
|-------|-------|
| `numpy_version` | `numpy.__version__` |
| `python_version` | major.minor.patch |
| `platform` | OS name |
| `seed` | `CONFIG.SEED` |
| `git_commit` | current `HEAD` (or `unknown`) |
| `csv_sha256` | SHA-256 of `sequence_lengths.csv` and `initial_severity.csv` |

`train_rl.py` reads back the fingerprint stored at optimisation time and
reports any mismatch via `diff_fingerprints()` before retraining, so the
user is warned when `mean_perf` may not match the Optuna-reported value.

### 2.6 Running in the Background

**Linux:**

```bash
nohup python3 -m pes_dql.ext.optimize_rl 100 \
  > pes_dql/inputs/bayesian_opt.log 2>&1 &
```

Or use the helper scripts:

```bash
./utils/run_bayesian_opt.sh dql 100
```

**Windows (PowerShell):**

```powershell
.\utils\run_bayesian_opt.ps1 dql 100
```

---

## 3. Testing the Agent (Running the Experiment)

### 3.1 Verify Training Files Exist

Before running the experiment, ensure these files are present:

```
pes_dql/inputs/q.npy           ← trained Q-table (31 × 11 × 10 × 11)
pes_dql/inputs/rewards.npy     ← reward history
```

Both are automatically created by the training pipeline (§ 1).

### 3.2 Run the Experiment

```bash
python3 -m pes_dql
```

This launches the full experiment lifecycle:

1. **Validates** the Q-table (loads `q.npy`, checks shape).
2. **Sets up** the experiment session with logging and dated output folders.
3. **Iterates** through 8 blocks × 8 sequences × 3–10 trials per
   sequence (~360 total decisions).
4. At each trial, the RL agent:
   - Indexes the Q-table: `Q[resources_left, trial_no, severity]`.
   - Selects the action with the highest Q-value (greedy argmax).
   - Masks infeasible actions (allocation > resources_left).
   - Computes **meta-cognitive confidence** from the entropy of the
     Q-value distribution.
   - Simulates human-like **response timing** based on confidence.
5. **Saves** results (performance JSON, visualisation PNG, response logs)
   to `pes_dql/outputs/<YYYY-MM-DD>_RL_AGENT/`.

### 3.3 Experiment Outputs

| File | Description |
|------|-------------|
| `PES_DQL_<session_id>.txt` | Experiment configuration snapshot |
| `PES_DQL_responses_<session_id>.txt` | Trial-by-trial responses (severity, allocation, confidence, timing) |
| `PES_DQL_log_<session_id>.txt` | Console log (dual-stream, saved next to `outputs/`) |
| `PES_DQL_movement_log_<session_id>.npy` | Movement data |
| `PES_DQL_results_<session_id>.json` | Performance summary (per-block mean/std/min/max, percentiles, improvement) |
| `PES_DQL_results_<session_id>.png` | Performance plots |

### 3.4 Configuration

All experiment parameters are in `config/CONFIG.py`. Key settings:

```python
PLAYER_TYPE = 'RL_AGENT'
SEED = 42
NUM_BLOCKS = 8
NUM_SEQUENCES = 8
AVAILABLE_RESOURCES_PER_SEQUENCE = 39
MAX_SEVERITY = 9
```

### 3.5 Reproducibility check (`raw_mean_perf`)

After saving the results JSON, `__main__.py` prints a one-line consistency
check between the freshly computed mean performance and the value reported
by the Bayesian optimiser:

```text
raw_mean_perf = 0.896344  (std=0.047708, n=64)
best_params.json mean_perf = 0.896344  |Δ| = 0.000000
```

`raw_mean_perf` is the mean of `MyPerformances` across the full
`NUM_BLOCKS × NUM_SEQUENCES` session. `best_params.json['mean_perf']` is
the value Optuna recorded for the winning trial (loaded from
`inputs/best_params.json`). A small `|Δ|` (< 1e-3) confirms full
reproducibility; a larger gap usually means the run used a different
`SEED`, a different input CSV, or a different compute device.

---

## 4. Complete Workflow (3-Stage Pipeline)

The recommended workflow from scratch:

```bash
# Stage 1 — Find optimal hyperparameters
python3 -m pes_dql.ext.optimize_rl 50

# Stage 2 — Train with those hyperparameters (already hard-coded)
python3 -m pes_dql.ext.train_rl

# Stage 3 — Run the experiment
python3 -m pes_dql
```

If you have run a new Bayesian search, no manual step is required:
`optimize_rl.py` mirrors `best_params.json`, `q.npy` and `rewards.npy` to
`pes_dql/inputs/`, and `train_rl.py` consumes them automatically (Stage 2).

---

## 5. Troubleshooting

### Q-table not found

```
Q‑table file not found: .../pes_dql/inputs/q.npy
```

**Solution:** Run training first:

```bash
python3 -m pes_dql.ext.train_rl
```

### Poor agent performance

- Run Bayesian optimisation with more trials:
  `python3 -m pes_dql.ext.optimize_rl 100`
- Increase episode count:
  `python3 -m pes_dql.ext.train_rl 1200000`
- Verify `SEED = 42` for reproducible results.

### Optimisation appears stuck or slow

Each trial trains a full Double Q-Learning agent — expect longer per trial
than pes_ql due to the three improvements (especially PBRS).

---

## 6. Empirical Results — Run 2026-04-21

Reference end-to-end run on Windows / `win_mpes_env` / Python 3.12, with the
default 100-trial Bayesian search.

### 6.1 Bayesian Optimisation (`inputs/2026-04-21_BAYESIAN_OPT/`)

| Metric | Value |
|--------|-------|
| Total trials | 100 |
| Completed | 14 |
| Pruned by `MedianPruner` | 86 |
| Failed | 0 |
| Best trial number | #3 (0-indexed) |
| Best `mean_perf` | **0.896344** |

Best hyperparameters (from `best_params_2026-04-21.json`):

| Parameter | Optimised value |
|-----------|-----------------|
| `learning_rate` ($\alpha$) | 0.11320 |
| `discount_factor` ($\gamma$) | 0.97773 |
| `epsilon_initial` ($\varepsilon_0$) | 0.59984 |
| `epsilon_min` ($\varepsilon_{\min}$) | 0.03268 |
| `num_episodes` ($N$) | 360 000 |
| `warmup_ratio` ($w$) | 0.02604 (≈ 9 373 episodes) |
| `target_ratio` ($f$) | 0.64302 (≈ 231 486 episodes) |
| `penalty_coeff` ($\beta$) | 3.917 × 10⁻⁴ |
| `trial_seed` | 46 (= `SEED + 3 + 1`) |

Auto-computed $\lambda \approx 0.9999869$ over $(f - w) \cdot N \approx 350\,627$
decay episodes; the remaining $\approx 128\,514$ episodes (35.7 %) run at
$\varepsilon_{\min}$.

These values **differ** from the legacy hard-coded fallback in § 1.5; the
fallback is only used when `best_params.json` is absent. After this run,
`pes_dql/inputs/best_params.json` was overwritten with the values above and
`__main__.py` / `train_rl.py` will pick them up automatically.

### 6.2 RL-Agent Training (`inputs/2026-04-21_RL_TRAIN/`)

`train_rl.py` reproduced the best trial bit-for-bit using `seed=46` and
`track_confidence=False`:

- Algorithm: Double Q-Learning (2 tables, 600 160 bytes during training).
- Q-table shape: `(31, 11, 10, 11)` = 37 510 cells (saved as the average of
  `Q_A` and `Q_B`).
- Rewards history length: 36 averaged samples (one every 10 000 episodes).
- Local deterministic `mean_perf` matches the optimisation reference to
  ≈ 0 difference (no warning emitted by `train_rl.py`).

### 6.3 Experiment (`outputs/2026-04-21_RL_AGENT/`)

Aggregated performance over the 8 blocks × 8 sequences (64 runs total),
read from `PES_DQL_results_2026-04-21_RL_AGENT.json`:

| Statistic | Value |
|-----------|-------|
| Overall mean | 0.8963 |
| Overall median | 0.8960 |
| Overall std | 0.0477 |
| Overall min | 0.7768 |
| Overall max | 1.0000 |
| 1st-block mean | 0.8896 |
| Last-block mean | 0.9038 |
| Improvement (last − first) | +0.0142 |
| Percentile 25 / 75 | 0.8623 / 0.9277 |

Per-block means range from 0.8872 (block 2) to 0.9040 (block 3), with the
block-3 maximum reaching the theoretical ceiling of 1.0. The slight
positive improvement between the first and last block is consistent with
fixed-policy evaluation (the agent does not learn during the experiment;
variability is sequence-dependent).

---

## 7. Quick Reference

| Task | Command |
|------|---------|
| Activate environment (Linux) | `source linux_mpes_env/bin/activate` |
| Activate environment (Windows) | `win_mpes_env\Scripts\Activate.ps1` |
| Train agent (default) | `python3 -m pes_dql.ext.train_rl` |
| Train agent (custom) | `python3 -m pes_dql.ext.train_rl 1000000` |
| Optimise hyperparameters | `python3 -m pes_dql.ext.optimize_rl 50` |
| Resume optimisation | `python3 -m pes_dql.ext.optimize_rl 100 --resume YYYY-MM-DD` |
| Run experiment | `python3 -m pes_dql` |
