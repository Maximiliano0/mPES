# QL+UQ: Q-Learning with Uncertainty Quantification — Usage Guide

## Overview

**QL+UQ** is an experimental variant of tabular Q-Learning that augments decision-making with **entropy-based confidence estimation** and **risk-controlled action scaling**. The agent learns resource-allocation policies while quantifying its own uncertainty, then adjusts allocations based on confidence scores.

## Core Concepts

### 1. Q-Learning Foundation

- **State space**: `(resources_left, trial_number, severity)` → 31 × 11 × 10 states
- **Action space**: Discrete allocations 0–10 resources
- **Learning**: Temporal-difference (TD) updates with ε-greedy exploration
- **Q-table shape**: (31, 11, 10, 11) = 37,510 cells

### 2. Confidence Estimation (Meta-Cognitive Function)

Confidence is computed **per decision** using the **entropy of the Q-value distribution** across actions at each state:

```
State (s) → Q(s, :) = [Q(s,0), Q(s,1), ..., Q(s,10)]
Convert to probability: PDF ∝ softmax(Q(s, :))
Confidence = 1 - Entropy(PDF)
```

- **High confidence**: Q-values are concentrated on one or two actions (low entropy)
- **Low confidence**: Q-values are spread across all actions (high entropy)

**Where**: Computed in `rl_agent_meta_cognitive()` function (pandemic.py line 459)

### 3. Risk-Controlled Action Scaling

Once confidence is estimated, the selected action is **scaled down** based on uncertainty:

```
risk_scale = f(confidence, γ, severity, mode)
adjusted_action = floor(selected_action × risk_scale)
```

- **When confident**: `risk_scale ≈ 1.0` → allocate the full action
- **When uncertain**: `risk_scale < 1.0` → allocate conservatively

**Where**: Applied in `apply_confidence_risk_penalty()` function (pandemic.py line 45)

---

## Configuration Parameters

All parameters are defined in `config/CONFIG.py`:

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `CONFIDENCE_ALLOCATION_PENALTY` | `bool` | `True` | Enable risk-scaling mechanism |
| `CONFIDENCE_RISK_FACTOR` | `float` | 0.5 | Scaling intensity (γ) |
| `PENALTY_MODE` | `str` | `'linear'` | Risk calculation mode |
| `BOLTZMANN_EPSILON` | `float` | 1e-8 | Numerical stability in Boltzmann modes |

### Penalty Modes

#### `'linear'` (Default)

Confidence directly scales action allocation:

$$\text{risk\_scale} = 1 - \gamma \cdot (1 - C)$$

- `C` = confidence ∈ [0, 1]
- `γ` = risk factor ∈ [0, 1]

**Behavior:**

- Confident (`C=1`): scale = 1.0 → full action
- Uncertain (`C=0`): scale = 1 − γ → reduced action

#### `'boltzmann_option_a'`

Severity-modulated Boltzmann penalty:

$$\text{risk\_scale} = \exp\left(-\frac{\gamma(1-C)}{(S_t+1)(C+\epsilon)}\right)$$

- `S_t` = current severity
- Higher severity amplifies penalty
- Smooth, non-linear transitions

#### `'boltzmann_option_b'`

Pure uncertainty Boltzmann penalty (ignores severity):

$$\text{risk\_scale} = \exp\left(-\frac{\gamma(1-C)}{C+\epsilon}\right)$$

---

## Workflow: Training & Inference

### Training Phase

```bash
cd h3
python -m tabular_uq.ql_uq
```

1. **Load** initial severities and sequence lengths from CSVs
2. **Initialize** Q-table (zeros)
3. **Run** Q-Learning episodes (default: 900,000)
   - Per episode: epsilon-greedy action selection
   - **Track confidence = False** (Q-table only; confidence computed at inference)
   - Update Q-values via TD rule
4. **Save** Q-table to `inputs/q.npy`, rewards history

### Inference Phase

```python
from tabular_uq.ql_uq.ext.pandemic import Pandemic, rl_agent_meta_cognitive
from tabular_uq.ql_uq.ext.train_rl import run_experiment

# Load trained Q-table
q_table = numpy.load('inputs/q.npy')

# Run greedy policy with confidence
def agent_with_confidence(state, resources, severity):
    confidence, action = rl_agent_meta_cognitive(q_table, resources, state[1], severity)
    # Action is already scaled by confidence internally
    return action, confidence

results = run_experiment(env, agent_with_confidence, sequences)
```

**At each step:**

1. Select greedy action from Q-table: `arg max_a Q(s, a)`
2. Compute entropy of `softmax(Q(s, :))`
3. Calculate confidence = `1 - entropy`
4. Scale action by risk factor → final allocation
5. Execute in environment, record confidence score

---

## Output & Artifacts

After training, outputs are saved to `outputs/<date>_RL_TRAIN/`:

```
outputs/
├── 2026-09-01_14-32-56_RL_TRAIN/
│   ├── q.npy                      # Trained Q-table (37510,)
│   ├── rewards.npy                # Per-episode rewards
│   ├── confidences.npy            # Per-decision confidences
│   ├── training_config.json       # Hyperparams, seed, n_episodes
│   ├── baseline_comparison.csv    # Agent vs. random baseline
│   ├── confidence_histogram.png   # Distribution plot
│   └── performance_metrics.json   # Mean, std, median of rewards
```

---

## Key Methods & Functions

### `rl_agent_meta_cognitive(options, resources_left, response_timeout)`

**Computes action & confidence from Q-table.**

```python
confidence, action = rl_agent_meta_cognitive(
    q_table,              # (31, 11, 10, 11) Q-table
    resources_left=15,    # int ∈ [0, 30]
    response_timeout=5    # int ∈ [0, 10] (trial number)
)
```

- `confidence`: float ∈ [0, 1]
- `action`: int ∈ [0, 10]

### `apply_confidence_risk_penalty(action, confidence, resources_left, severity, risk_factor, penalty_mode)`

**Scales action by confidence & severity.**

```python
adjusted = apply_confidence_risk_penalty(
    action=8,
    confidence=0.85,
    resources_left=15,
    severity=5.0,
    risk_factor=0.5,
    penalty_mode='linear'
)
# Returns: 7 or 8 (depending on risk_scale)
```

### `entropy_from_pdf(pdf)`

**Calculates Shannon entropy of a probability distribution.**

```python
from tabular_uq.ql_uq.ext.tools import entropy_from_pdf

H = entropy_from_pdf(softmax_q_values)  # returns bits
confidence = 1 - (H / numpy.log2(num_actions))  # normalized
```

---

## Comparison to pes_base (Tabular Q-Learning)

| Aspect | pes_base | ql_uq |
|--------|----------|-------|
| Q-Learning | ✓ | ✓ |
| Confidence tracking | ✗ | ✓ Entropy-based |
| Risk-controlled actions | ✗ | ✓ penalty modes |
| Action scaling | None | Linear / Boltzmann |
| Config params | 6 | 10 (+ UQ-specific) |
| Episode count | 20,000 (default) | 900,000 (default) |
| Use case | Baseline | Cautious, UQ-aware agent |

---

## Experimental Notes

- **Status**: Suspended — no active benchmark yet
- **Protocol**: Training config, evaluation criteria, and h1 comparison still TBD
- **Next steps**: Define reference h1 baseline, establish UQ evaluation metrics, run head-to-head scenarios
