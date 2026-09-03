# QL+UQ: Theoretical Foundations

## 1. Meta-Cognitive Confidence via Entropy

### Motivation

In reinforcement learning, uncertainty arises in two forms:

- **Aleatoric uncertainty**: Inherent randomness in the environment
- **Epistemic uncertainty**: Agent's lack of knowledge (e.g., unexplored states)

When the Q-values across actions at a state are **sharply peaked**, the agent has high confidence in its choice. When they are **diffuse**, the agent is uncertain.

### Definition

Given a state $s$ and Q-table $Q(s, a)$ for all $a \in \mathcal{A}$, we compute:

$$\text{PDF}(s) = \frac{\exp(\beta \cdot Q(s, :))}{\sum_a \exp(\beta \cdot Q(s, a))}$$

where $\beta$ is an inverse temperature parameter (default: $\beta=1$, standard softmax).

The **Shannon entropy** of this distribution is:

$$H(s) = -\sum_a \text{PDF}(s, a) \log_2 \text{PDF}(s, a)$$

**Confidence** is then defined as:

$$C(s) = 1 - \frac{H(s)}{H_{\max}} = 1 - \frac{H(s)}{\log_2(|\mathcal{A}|)}$$

where $H_{\max} = \log_2(|\mathcal{A}|)$ is the maximum entropy (uniform distribution over $|\mathcal{A}| = 11$ actions).

### Properties

- $H(s) = 0$ (one action has PDF = 1) → $C(s) = 1$ (maximum confidence)
- $H(s) = H_{\max}$ (uniform PDF) → $C(s) = 0$ (no confidence)
- $C(s) \in [0, 1]$ by construction

### Implementation

```python
def entropy_from_pdf(pdf):
    """Shannon entropy in bits."""
    pdf = pdf + numpy.abs(numpy.min(pdf))  # shift to positive
    p = pdf / numpy.sum(pdf)
    p[p == 0] += 1e-6  # avoid log(0)
    return -numpy.dot(p, numpy.log2(p))

confidence = 1 - (entropy_from_pdf(softmax(Q(s, :))) / numpy.log2(11))
```

---

## 2. Risk-Controlled Action Scaling

### Motivation

An uncertain agent should act conservatively. When confidence is low, scaling down the allocated resources reduces the risk of poor decisions.

Three penalty modes balance **aggressiveness** vs. **conservatism**:

### Mode 1: Linear Penalty

**Equation:**

$$\text{risk\_scale}(s, a, c) = 1 - \gamma(1 - c)$$

where:

- $c = C(s)$ = confidence ∈ [0, 1]
- $\gamma$ = risk factor ∈ [0, 1]

**Properties:**

- Linear in confidence deficit $(1 - c)$
- Independent of severity
- Fastest computation

**Behavior:**

- $c = 1$ (confident): scale = 1.0 → no penalty
- $c = 0.5$ (medium): scale = $1 - 0.5 \gamma$
- $c = 0$ (no confidence): scale = $1 - \gamma$ (linear floor)

---

### Mode 2: Boltzmann Option A (Severity-Modulated)

**Equation:**

$$\text{risk\_scale}(s, a, c) = \exp\left(-\frac{\gamma(1-c)}{(S_t + 1)(c + \epsilon)}\right)$$

where:

- $S_t$ = current severity ∈ [0, 9]
- $\epsilon \approx 10^{-8}$ = regularization constant
- Denominator $(S_t + 1)$ ∈ [1, 10]

**Properties:**

- Exponential decay in uncertainty $(1 - c)$
- Severity amplifies the penalty: higher $S_t$ → stronger penalty
- Non-linear, smooth transitions
- Avoids division-by-zero via $\epsilon$

**Behavior:**

Let $\gamma = 0.5$, $\epsilon = 10^{-8}$:

| $c$ | $S_t=0$ | $S_t=5$ | $S_t=9$ |
|-----|---------|---------|---------|
| 1.0 | 1.000   | 1.000   | 1.000   |
| 0.8 | 0.905   | 0.667   | 0.491   |
| 0.5 | 0.607   | 0.330   | 0.182   |
| 0.2 | 0.223   | 0.081   | 0.032   |

High severity + low confidence → dramatic action reduction.

---

### Mode 3: Boltzmann Option B (Pure Uncertainty)

**Equation:**

$$\text{risk\_scale}(s, a, c) = \exp\left(-\frac{\gamma(1-c)}{c + \epsilon}\right)$$

where:

- Ignores severity; depends only on confidence
- Denominator $(c + \epsilon)$ prevents division-by-zero

**Properties:**

- Intermediate between linear and severity-modulated Boltzmann
- Emphasizes confidence; ignores state severity
- Useful when severity is not observable or is unreliable

**Behavior:**

| $c$ | $\gamma=0.3$ | $\gamma=0.5$ | $\gamma=1.0$ |
|-----|-------------|-------------|-------------|
| 1.0 | 1.000       | 1.000       | 1.000       |
| 0.8 | 0.969       | 0.945       | 0.891       |
| 0.5 | 0.862       | 0.779       | 0.606       |
| 0.2 | 0.549       | 0.368       | 0.135       |
| 0.1 | 0.340       | 0.175       | 0.018       |

Lower $c$ with higher $\gamma$ yields steeper penalty.

---

## 3. Action Adjustment Pipeline

Given state $s$, the final action $a'$ is computed as:

$$a_{\text{greedy}} = \arg\max_a Q(s, a)$$

$$c(s) = \text{Confidence}(Q(s, :))$$

$$\text{scale} = \text{RiskPenalty}(c, \gamma, S_t, \text{mode})$$

$$a' = \text{clip}\left(\lfloor a_{\text{greedy}} \times \text{scale} \rfloor, 0, \min(\text{resources\_left}, 10)\right)$$

**Key steps:**

1. Greedy action selection from Q-table
2. Entropy-based confidence calculation
3. Risk scale computed (mode-dependent)
4. Action multiplied by scale, floored to integer
5. Clipped to resource constraints

---

## 4. Q-Learning Update Rule

During training, the Q-table is updated via **temporal-difference learning**:

$$Q(s, a) \gets Q(s, a) + \alpha [r + \gamma_q \max_a Q(s', a) - Q(s, a)]$$

where:

- $\alpha$ = learning rate
- $r$ = immediate reward
- $\gamma_q$ = discount factor
- $s'$ = next state

### Reward Structure

In the Pandemic environment:

$$r_t = \text{RESPONSE\_MULTIPLIER} \times (\text{initial\_severity} - \text{updated\_severity})$$

Rewards are **shaped** to encourage severity reduction.

---

## 5. Exploration Strategy

### Epsilon-Greedy with Decay

Training uses ε-greedy action selection:

$$a_t = \begin{cases}
\text{random action} & \text{with probability } \epsilon_t \\
\arg\max_a Q(s, a) & \text{with probability } 1 - \epsilon_t
\end{cases}$$

**Epsilon decay** (linear):

$$\epsilon_t = \epsilon_0 - \frac{t}{T}(\epsilon_0 - \epsilon_{\min})$$

where:
- $\epsilon_0$ ≈ 0.68 (Bayesian-optimized)
- $\epsilon_{\min}$ ≈ 0.08 (Bayesian-optimized)
- $T$ = total episodes

### Rationale

Early training: High exploration to discover diverse policies.
Late training: Exploitation to refine discovered policies.

## 6. State & Action Spaces

### Observation Space (Tabular)

States are discretized tuples $(r, t, s)$:

$$\mathcal{S} = \{0..30\} \times \{0..10\} \times \{0..9\}$$

**Dimensions:**
- $r$ = resources remaining (31 states)
- $t$ = trial number in sequence (11 states)
- $s$ = current severity (10 states)

**Total states:** $31 \times 11 \times 10 = 3,410$

### Action Space

$$\mathcal{A} = \{0, 1, 2, ..., 10\} \quad (|\mathcal{A}| = 11)$$

Each action represents allocating 0–10 resources in a single trial.

### Q-Table Dimensionality

$$Q : \mathcal{S} \times \mathcal{A} \to \mathbb{R}$$

Shape: $(31, 11, 10, 11) = 37,510$ scalar values

---

## 7. Comparison to Uncertainty Quantification Methods

| Method | Approach | Computational Cost | Requires Ensemble? |
|--------|----------|-------------------|--------------------|
| **Entropy (QL+UQ)** | Post-hoc softmax entropy | O(n_actions) | No |
| **Bootstrapped DQN** | Multiple networks | O(n_networks × net_size) | Yes |
| **Bayesian NNs** | Variational inference | O(forward pass) | Implicit ensemble |
| **Monte Carlo Dropout** | Stochastic forward passes | O(n_samples × forward pass) | Implicit |

**QL+UQ advantage:** Simple, interpretable, minimal overhead.

---

## 8. Hyperparameter Tuning (Bayesian Optimization)

The following hyperparameters are Bayesian-optimized (Optuna TPE sampler):

| Parameter | Search Space | Default | Optimized Value |
|-----------|--------------|---------|-----------------|
| $\alpha$ (learning rate) | [0.01, 0.99] | 0.2 | ≈0.360 |
| $\gamma_q$ (discount) | [0.5, 0.99] | 0.9 | ≈0.865 |
| $\epsilon_0$ (init explore) | [0.1, 0.9] | 0.6 | ≈0.679 |
| $\epsilon_{\min}$ (final explore) | [0.01, 0.5] | 0.1 | ≈0.085 |

**Objective:** Maximize mean reward on validation sequences.

---

## 9. Limitations & Future Work

### Current Limitations

1. **Tabular only**: Scalability limited to discrete, low-dimensional spaces
2. **Greedy confidence**: Softmax entropy alone may not capture all uncertainty sources
3. **No training feedback**: Confidence is computed post-hoc; Q-learning doesn't optimize for confidence
4. **Single seed**: Reproducibility via seed, but no ensemble uncertainty quantification

### Research Directions

- **Approximate Bayesian inference**: Variational posterior over Q-values
- **Uncertainty-aware learning**: Integrate confidence into the TD loss
- **Multi-seed ensemble**: Average predictions across multiple trained agents
- **Active learning**: Use confidence to guide exploration or data acquisition

---

## References & Related Work

- **Shannon Entropy**: Shannon, C. E. (1948). "A Mathematical Theory of Communication"
- **Softmax & Exploration**: Cesa-Bianchi & Lugosi (2006). "Prediction, Learning, and Games"
- **Confidence in RL**: Lakshminarayanan et al. (2017). "Deep Ensembles" (but tabular)
- **Boltzmann Exploration**: Gibbs sampling, statistical mechanics analog
