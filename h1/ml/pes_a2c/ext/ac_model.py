"""
pes_a2c - Advantage Actor-Critic (A2C) model components.

Provides the neural-network building blocks used by the A2C training loop
in :pymod:`pandemic`:

- **build_actor**:  Constructs a Keras model that maps a normalised state
  vector to an action probability distribution π(a|s) via softmax.
- **build_critic**:  Constructs a Keras model that maps a normalised state
  vector to a scalar state-value estimate V(s).
- **normalize_state**:  Scales raw integer state components to the [0, 1]
  range expected by the networks.
- **train_step_actor_critic**:  Single gradient-descent update for both
  Actor and Critic using the Advantage Actor-Critic objective with
  gradient clipping, GAE(λ), and advantage normalisation.
  Not decorated with ``@tf.function`` at module level — each
  Optuna trial wraps it locally via ``tf.function`` to scope
  the traced graph per trial.

Training Improvements
---------------------
- **Gradient clipping**: ``tf.clip_by_global_norm`` on both Actor and
  Critic gradients to prevent exploding gradients (Mnih et al., 2016).
- **GAE(λ)**: Generalized Advantage Estimation with reverse
  accumulation (Schulman et al., 2016).  λ=0 → TD(0); λ=1 → MC.
- **Advantage normalisation**: Per-batch zero-mean unit-variance scaling
  to stabilise Actor gradient magnitudes independently of reward scale.
- **Infeasible-action masking**: Per-transition feasibility masks zero
  out actions exceeding available resources before computing log-probs
  and entropy, so the policy gradient trains only over legal actions.

Architecture
------------
::

    Actor:   Input(3) → Dense(128, ReLU) → Dense(11, softmax)
    Critic:  Input(3) → Dense(128, ReLU) → Dense(1, linear)

The default hidden-layer sizes are configurable via ``config/CONFIG.py``
(``AC_ACTOR_HIDDEN_UNITS = [128]``, ``AC_CRITIC_HIDDEN_UNITS = [128]``).

CPU Optimisations
-----------------
- TensorFlow intra/inter-op thread pools are configured at import time
  to match the host CPU (defaults to auto-detect / 2 inter-op threads).
"""

##########################
##  Imports externos    ##
##########################
from typing import List, Optional, Tuple

import numpy
import tensorflow as tf

##########################
##  Imports internos    ##
##########################


# ---------------------------------------------------------------------------
#  CPU threading optimisation (skipped when GPU is available)
# ---------------------------------------------------------------------------
if not tf.config.list_physical_devices('GPU'):
    tf.config.threading.set_intra_op_parallelism_threads(0)   # 0 = auto-detect
    tf.config.threading.set_inter_op_parallelism_threads(2)


# ---------------------------------------------------------------------------
#  Network builders
# ---------------------------------------------------------------------------
def build_actor(state_dim: int, action_dim: int,
                hidden_units: List[int],
                seed: Optional[int] = None,
                last_action_bias: float = 0.0) -> tf.keras.Model:
    """Build a fully-connected Actor (policy) network with softmax output.

    The Actor maps a normalised state vector to a probability distribution
    over discrete actions.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the (normalised) state vector.
    action_dim : int
        Number of discrete actions (network output width).
    hidden_units : list of int
        Widths of the hidden dense layers (e.g. ``[64, 64]``).
    seed : int or None, optional
        Seed forwarded to ``GlorotUniform`` for cross-process
        reproducibility (independent of TF's global op counter).
        Each layer is offset by its index so weights differ per layer.
    last_action_bias : float, optional
        Initial bias logit applied to the last action of the policy
        head.  When < 0, the "spend the maximum feasible amount"
        action starts with reduced softmax probability, breaking the
        symmetry that otherwise drives the policy toward the trivial
        ``argmax → max-feasible`` collapse observed in early Optuna
        trials.  All other action biases are initialised to 0.
        Default: ``0.0`` (uniform zero-bias init).

    Returns
    -------
    tf.keras.Model
        Keras ``Sequential`` model with softmax output.
    """
    model = tf.keras.Sequential(name="Actor")
    model.add(tf.keras.layers.Input(shape=(int(state_dim),)))
    for idx, units in enumerate(hidden_units):
        layer_seed = None if seed is None else int(seed) + idx
        model.add(tf.keras.layers.Dense(
            int(units), activation="relu", name=f"actor_hidden_{idx}",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=layer_seed)))
    out_seed = None if seed is None else int(seed) + len(hidden_units)

    # Custom bias initialiser: zeros everywhere except last index.
    bias_vec = numpy.zeros((int(action_dim),), dtype=numpy.float32)
    if float(last_action_bias) != 0.0:
        bias_vec[-1] = float(last_action_bias)
    bias_init = tf.keras.initializers.Constant(bias_vec)

    model.add(tf.keras.layers.Dense(
        int(action_dim), activation="softmax", name="policy",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=out_seed),
        bias_initializer=bias_init))
    return model


def build_critic(state_dim: int,
                 hidden_units: List[int],
                 seed: Optional[int] = None) -> tf.keras.Model:
    """Build a fully-connected Critic (value) network with linear output.

    The Critic maps a normalised state vector to a scalar state-value
    estimate V(s).

    Parameters
    ----------
    state_dim : int
        Dimensionality of the (normalised) state vector.
    hidden_units : list of int
        Widths of the hidden dense layers (e.g. ``[64, 64]``).
    seed : int or None, optional
        Seed forwarded to ``GlorotUniform`` for cross-process
        reproducibility (independent of TF's global op counter).
        Each layer is offset by its index so weights differ per layer.

    Returns
    -------
    tf.keras.Model
        Keras ``Sequential`` model with single linear output.
    """
    model = tf.keras.Sequential(name="Critic")
    model.add(tf.keras.layers.Input(shape=(int(state_dim),)))
    for idx, units in enumerate(hidden_units):
        layer_seed = None if seed is None else int(seed) + idx
        model.add(tf.keras.layers.Dense(
            int(units), activation="relu", name=f"critic_hidden_{idx}",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=layer_seed)))
    out_seed = None if seed is None else int(seed) + len(hidden_units)
    model.add(tf.keras.layers.Dense(1, activation="linear", name="value",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=out_seed)))
    return model


# ---------------------------------------------------------------------------
#  State normalisation
# ---------------------------------------------------------------------------
def normalize_state(state, max_resources: int,
                    max_trials: int, max_severity: int) -> numpy.ndarray:
    """Scale raw integer state components to [0, 1].

    Parameters
    ----------
    state : array-like
        Raw state ``[resources_left, trial_no, severity]``.
    max_resources : int
        Upper bound of the resources dimension.
    max_trials : int
        Upper bound of the trial-number dimension.
    max_severity : int
        Upper bound of the severity dimension.

    Returns
    -------
    ndarray, shape ``(3,)``, dtype ``float32``
    """
    return numpy.array([
        state[0] / max(max_resources, 1),
        state[1] / max(max_trials, 1),
        state[2] / max(max_severity, 1),
    ], dtype=numpy.float32)


# ---------------------------------------------------------------------------
#  Training step (Actor-Critic)
# ---------------------------------------------------------------------------
def train_step_actor_critic(
    actor: tf.keras.Model,
    critic: tf.keras.Model,
    actor_optimizer: tf.keras.optimizers.Optimizer,
    critic_optimizer: tf.keras.optimizers.Optimizer,
    states: tf.Tensor,
    actions: tf.Tensor,
    rewards: tf.Tensor,
    next_states: tf.Tensor,
    dones: tf.Tensor,
    discount: tf.Tensor,
    entropy_coeff: tf.Tensor,
    max_grad_norm: tf.Tensor,
    gae_lambda: tf.Tensor,
    masks: Optional[tf.Tensor] = None
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Execute a single gradient-descent step for both Actor and Critic.

    Uses the Advantage Actor-Critic (A2C) objective with the following
    improvements over vanilla A2C:

    - **Gradient clipping** (clip-by-global-norm) on both Actor and Critic
      gradients to prevent exploding gradients (Mnih et al., 2016).
    - **GAE(λ)** — Generalized Advantage Estimation (Schulman et al., 2016)
      to interpolate between TD(0) (λ=0) and Monte-Carlo (λ=1) advantage
      estimates, reducing variance without significant bias.
    - **Advantage normalisation** — per-batch zero-mean unit-variance
      normalisation of advantages to stabilise Actor gradient magnitudes.
    - **Infeasible-action masking** — when *masks* are provided, the
      Actor's softmax output is multiplied element-wise by the mask and
      renormalised before computing log-probabilities and entropy.  This
      ensures the policy gradient and entropy bonus operate only over
      feasible actions (allocation ≤ resources_left), aligning training
      with the masked inference used in evaluation and ``__main__``.

    Loss functions:

    - **Critic loss**: MSE between V(s) and the TD target  r + γ·V(s')·(1-done).
    - **Actor loss**: -log π(a|s) · Â_GAE(s,a), where Â_GAE is the
      normalised GAE advantage, plus an entropy bonus for exploration.

    This function is **not** decorated with ``@tf.function`` at the
    module level because Optuna calls ``A2CTraining`` multiple times
    with different model/optimiser instances.  Each trial wraps this
    function via ``tf.function`` locally so the traced graph is
    scoped to a single optimisation trial and no ``tf.Variable``
    leaks between trials.

    Parameters
    ----------
    actor : tf.keras.Model
        Policy network whose weights are updated.
    critic : tf.keras.Model
        Value network whose weights are updated.
    actor_optimizer : tf.keras.optimizers.Optimizer
        Optimiser for the Actor (e.g. Adam with cosine annealing).
    critic_optimizer : tf.keras.optimizers.Optimizer
        Optimiser for the Critic (e.g. Adam with cosine annealing).
    states : tf.Tensor, shape ``(B, state_dim)``
    actions : tf.Tensor, shape ``(B,)``   int32
    rewards : tf.Tensor, shape ``(B,)``
    next_states : tf.Tensor, shape ``(B, state_dim)``
    dones : tf.Tensor, shape ``(B,)``
    discount : tf.Tensor
        Scalar ``tf.Tensor`` with discount factor γ.  Passed as a
        tensor (not a Python float) to avoid ``@tf.function``
        retracing when the value changes across Optuna trials.
    entropy_coeff : tf.Tensor
        Scalar ``tf.Tensor`` with the entropy-bonus weight.  Passed
        as a tensor for the same reason as *discount*.
    max_grad_norm : tf.Tensor
        Scalar ``tf.Tensor`` with the global gradient norm clipping
        threshold.  Applied to both Actor and Critic gradients before
        ``apply_gradients()``.
    gae_lambda : tf.Tensor
        Scalar ``tf.Tensor`` with the GAE(λ) parameter.  Controls
        the bias-variance trade-off of the advantage estimator:
        λ=0 recovers TD(0); λ=1 gives Monte-Carlo returns.
    masks : tf.Tensor or None, optional
        Boolean-style tensor of shape ``(B, action_dim)`` with 1.0 for
        feasible actions and 0.0 for infeasible ones.  When provided,
        the Actor's probabilities are masked and renormalised before
        computing ``log_probs`` and ``entropy``.  Default: ``None``
        (no masking — all actions treated as feasible).

    Returns
    -------
    actor_loss : tf.Tensor
        Scalar Actor loss (policy gradient + entropy).
    critic_loss : tf.Tensor
        Scalar Critic loss (MSE on TD error).
    entropy : tf.Tensor
        Scalar mean entropy of the policy distribution.
    """
    # ---------- Critic update ----------
    with tf.GradientTape() as critic_tape:
        values = tf.squeeze(critic(states, training=True), axis=1)
        next_values = tf.squeeze(critic(next_states, training=False), axis=1)
        td_targets = rewards + discount * next_values * (1.0 - dones)
        critic_loss = tf.reduce_mean(tf.square(td_targets - values))

    critic_grads = critic_tape.gradient(critic_loss, critic.trainable_variables)
    critic_grads, _ = tf.clip_by_global_norm(critic_grads, max_grad_norm)
    critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

    # ---------- GAE(λ) Advantage ----------
    # Recompute values after Critic update for a cleaner advantage signal
    values_updated = tf.squeeze(critic(states, training=False), axis=1)
    next_values_updated = tf.squeeze(critic(next_states, training=False), axis=1)
    deltas = rewards + discount * next_values_updated * (1.0 - dones) - values_updated

    # Generalized Advantage Estimation — reverse accumulation
    # λ=0 → TD(0) (original behaviour); λ=1 → Monte-Carlo returns
    T = tf.shape(deltas)[0]
    gae_buffer = tf.TensorArray(dtype=tf.float32, size=T, dynamic_size=False)
    last_gae = tf.constant(0.0)
    for t in tf.range(T - 1, -1, -1):
        last_gae = deltas[t] + discount * gae_lambda * (1.0 - dones[t]) * last_gae
        gae_buffer = gae_buffer.write(t, last_gae)
    advantages = gae_buffer.stack()

    # Advantage normalisation — zero-mean unit-variance per batch
    adv_mean = tf.reduce_mean(advantages)
    adv_std = tf.math.reduce_std(advantages) + 1e-8
    advantages = (advantages - adv_mean) / adv_std

    # ---------- Actor update ----------
    with tf.GradientTape() as actor_tape:
        probs = actor(states, training=True)

        # Infeasible-action masking: zero out actions > resources_left
        # and renormalise so that log_probs and entropy reflect only
        # feasible actions — consistent with evaluation and __main__.
        if masks is not None:
            probs = probs * masks
            probs = probs / (tf.reduce_sum(probs, axis=1, keepdims=True) + 1e-8)

        # Clip probabilities to avoid log(0)
        probs_clipped = tf.clip_by_value(probs, 1e-8, 1.0)
        action_mask = tf.one_hot(actions, depth=tf.shape(probs)[1])
        log_probs = tf.reduce_sum(tf.math.log(probs_clipped) * action_mask, axis=1)

        # Entropy bonus: H(π) = -Σ π(a) log π(a)  (feasible actions only)
        entropy = -tf.reduce_sum(probs_clipped * tf.math.log(probs_clipped), axis=1)
        mean_entropy = tf.reduce_mean(entropy)

        # Policy gradient loss: -E[ log π(a|s) · Â_GAE(s,a) ] - c_ent · H(π)
        actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantages))
        actor_loss = actor_loss - entropy_coeff * mean_entropy

    actor_grads = actor_tape.gradient(actor_loss, actor.trainable_variables)
    actor_grads, _ = tf.clip_by_global_norm(actor_grads, max_grad_norm)
    actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))

    return actor_loss, critic_loss, mean_entropy
