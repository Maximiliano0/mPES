# How to Train and Test the Q-Learning Agent

> Package: **pes_base** — Tabular Q-Learning baseline for the Pandemic Experiment Scenario

---

## Prerequisites

| Requirement | Detail |
|-------------|--------|
| Python | 3.12 (Windows) / 3.12 (Linux) |
| Virtual environment | `win_mpes_env` (Windows) or `linux_mpes_env` (Linux) |
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

## 1. Training the Q-Learning Agent

### 1.1 Quick Start

```bash
python3 -m tabular.pes_base.ext.train_rl
```

This runs the **full training pipeline** with default settings (1 000 000
episodes).

### 1.2 Custom Episode Count

Pass the number of episodes as the first argument:

```bash
python3 -m tabular.pes_base.ext.train_rl 500000
```

### 1.3 What Happens During Training

The pipeline proceeds through these stages:

1. **Load data** — reads `initial_severity.csv` and
   `sequence_lengths.csv` from `inputs/`.
2. **Random player baseline** — runs 64 sequences with uniformly random
   allocations and generates two baseline plots.
3. **Q-Learning training** — standard tabular Q-Learning with linear
   ε-decay. Prints average reward every 10 000 episodes.
4. **Evaluation** — runs the trained greedy policy on 64 fixed sequences,
   collects meta-cognitive confidence scores.
5. **Save artefacts** — writes Q-table, reward history, config report,
   and visualisations to a dated subdirectory.

### 1.4 Training Output Files

All outputs are saved to `tabular/pes_base/inputs/<YYYY-MM-DD>_RL_TRAIN/`:

| File | Description |
|------|-------------|
| `q_<date>.npy` | Trained Q-table — shape `(31, 11, 10, 11)` |
| `rewards_<date>.npy` | Average-reward history (100 values, sampled every 10 000 eps) |
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

Additionally, the Q-table and rewards must be **manually copied** to the standard
paths consumed by the experiment runner:

- `tabular/pes_base/inputs/q.npy`
- `tabular/pes_base/inputs/rewards.npy`

### 1.5 Default Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Learning rate α | 0.2 | Q-table update step size |
| Discount factor γ | 0.9 | Future-reward weighting |
| Initial ε | 0.8 | Starting exploration probability |
| Minimum ε | 0.0 | Final exploration probability |
| Episodes | 1 000 000 | Total training episodes |
| ε-decay | Linear | $\varepsilon_t = \varepsilon_0 - t \cdot \frac{\varepsilon_0 - \varepsilon_{\min}}{N}$ |

### 1.6 Q-Table Dimensions

| Axis | Size | Meaning |
|------|------|---------|
| 0 | 31 | Resources left (0–30) |
| 1 | 11 | Trial number (0–10) |
| 2 | 10 | Severity (0–9), `MAX_SEVERITY = 9` |
| 3 | 11 | Actions (0–10 resources) |

**Total entries:** $31 \times 11 \times 10 \times 11 = 37{,}510$

### 1.7 Algorithm Summary

Standard tabular Q-Learning with update rule:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \bigl[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \bigr]$$

- **Initialisation:** Uniform random in $[-1, 1)$
- **Exploration:** ε-greedy with linear decay
- **Reward:** $r = -\sum_i \text{severity}_i$ (negative sum of all current city severities)

---

## 2. Testing the Agent (Running the Experiment)

### 2.1 Verify Training Files Exist

Before running the experiment, ensure these files are present:

```
tabular/pes_base/inputs/q.npy           ← trained Q-table (31 × 11 × 10 × 11)
tabular/pes_base/inputs/rewards.npy     ← reward history
```

Both are automatically created by the training pipeline (§ 1).

### 2.2 Run the Experiment

```bash
python3 -m tabular.pes_base
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
   to `tabular/pes_base/outputs/<YYYY-MM-DD>_RL_AGENT/`.

### 2.3 Experiment Outputs

| File | Description |
|------|-------------|
| `PES__<session_id>.txt` | Experiment configuration snapshot |
| `PES_responses_<session_id>.txt` | Trial-by-trial responses (severity, allocation, confidence, timing) |
| `PES_log_<session_id>.txt` | Console log (dual-stream) |
| `PES_movement_log_<session_id>.npy` | Movement data |
| `PES_results_<session_id>.json` | Performance summary JSON |
| `PES_results_<session_id>.png` | Performance plots |

### 2.4 Configuration

All experiment parameters are in `config/CONFIG.py`. Key settings:

```python
PLAYER_TYPE = 'RL_AGENT'
SEED = 42
NUM_BLOCKS = 8
NUM_SEQUENCES = 8
AVAILABLE_RESOURCES_PER_SEQUENCE = 39
MAX_SEVERITY = 9
```

---

## 3. Troubleshooting

### Q-table not found

```
Q‑table file not found: .../tabular/pes_base/inputs/q.npy
```

**Solution:** Run training first:

```bash
python3 -m tabular.pes_base.ext.train_rl
```

### Poor agent performance

- Increase episode count: `python3 -m tabular.pes_base.ext.train_rl 2000000`
- Verify `SEED = 42` in `config/CONFIG.py` for reproducible results.
- Check that `inputs/initial_severity.csv` and `inputs/sequence_lengths.csv`
  are present and well-formed.

### Training takes too long

1 000 000 episodes is the default. Reduce to 500 000 for faster
iteration (lower final quality).

---

## 4. Quick Reference

| Task | Command |
|------|---------|
| Activate environment (Linux) | `source linux_mpes_env/bin/activate` |
| Activate environment (Windows) | `win_mpes_env\Scripts\Activate.ps1` |
| Train agent (default) | `python3 -m tabular.pes_base.ext.train_rl` |
| Train agent (custom) | `python3 -m tabular.pes_base.ext.train_rl 500000` |
| Run experiment | `python3 -m tabular.pes_base` |
