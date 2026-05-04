"""
pes_dqn — Deep Q-Network model components.

Provides the neural-network building blocks used by the DQN training loop
in :pymod:`pandemic`:

- **build_q_network**:  Constructs a Keras model that maps a normalised
  state vector to Q-values for every discrete action.  Accepts an
  optional ``seed`` argument that seeds every Glorot-uniform kernel
  initialiser deterministically.
- **normalize_state**:  Scales raw integer state components to the [0, 1]
  range expected by the network.
- **ReplayBuffer**:  Fixed-size circular buffer storing experience tuples
  ``(state, action, reward, next_state, done)`` for off-policy learning.
- **train_step_dqn**:  Single gradient-descent update implementing the
  **Double DQN** target (van Hasselt et al., 2016) with per-sample
  feasibility masking on the bootstrapped action and Huber loss on the
  resulting TD error.  Not decorated with ``@tf.function`` at module
  level — each Optuna trial wraps it locally via
  ``tf.function(..., reduce_retracing=True)`` to scope the traced graph
  per trial.

Architecture
------------
::

    Q-Network:  Input(3) → Dense(h₀, ReLU) → … → Dense(hₙ, ReLU) → Dense(11, linear)

The default hidden-layer sizes are configurable via ``config/CONFIG.py``
(``DQN_HIDDEN_UNITS = [64, 64]``).

CPU Optimisations
-----------------
- TensorFlow intra/inter-op thread pools are pinned to a single thread
  at import time and ``tf.config.experimental.enable_op_determinism()``
  is enabled, so per-step ordering is deterministic.
"""

##########################
##  Imports externos    ##
##########################
from typing import List, Tuple
from collections import deque

import numpy
import random as python_random
import tensorflow as tf

##########################
##  Imports internos    ##
##########################


# ---------------------------------------------------------------------------
#  CPU threading optimisation — single-threaded for determinism
# ---------------------------------------------------------------------------
if not tf.config.list_physical_devices('GPU'):
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

try:
    tf.config.experimental.enable_op_determinism()
except AttributeError:
    pass  # TF < 2.9: determinism API not available


# ---------------------------------------------------------------------------
#  Network builder
# ---------------------------------------------------------------------------
def build_q_network(state_dim: int, action_dim: int,
                    hidden_units: List[int],
                    seed: int | None = None) -> tf.keras.Model:
    """Build a fully-connected Q-network with linear output.

    The Q-network maps a normalised state vector to Q-values for every
    discrete action.  When ``seed`` is provided, every ``Dense`` layer is
    initialised with a deterministic ``GlorotUniform(seed=...)`` so the
    same network can be rebuilt in a fresh process.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the (normalised) state vector.
    action_dim : int
        Number of discrete actions (network output width).
    hidden_units : list of int
        Widths of the hidden dense layers (e.g. ``[64, 64]``).
    seed : int or None, optional
        Seed for the Glorot-uniform kernel initialisers.  Each layer
        gets a distinct derived seed (``seed + layer_index``).
        Default: ``None`` (Keras default RNG).

    Returns
    -------
    tf.keras.Model
        Keras ``Sequential`` model with linear output of shape ``(action_dim,)``.
    """
    def _init(layer_idx: int):
        if seed is None:
            return "glorot_uniform"
        return tf.keras.initializers.GlorotUniform(seed=int(seed) + layer_idx)

    model = tf.keras.Sequential(name="Q_Network")
    model.add(tf.keras.layers.Input(shape=(int(state_dim),)))
    for idx, units in enumerate(hidden_units):
        model.add(tf.keras.layers.Dense(
            int(units), activation="relu",
            kernel_initializer=_init(idx),
            name=f"q_hidden_{idx}"))
    model.add(tf.keras.layers.Dense(
        int(action_dim), activation="linear",
        kernel_initializer=_init(len(hidden_units)),
        name="q_values"))
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
#  Replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Fixed-size circular buffer for storing experience tuples.

    Stores ``(state, action, reward, next_state, done)`` transitions and
    supports uniform random sampling for mini-batch training.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions the buffer can hold.  When full,
        the oldest transition is discarded.
    seed : int or None, optional
        Random seed for reproducible sampling.  Default: ``None``.
    """

    def __init__(self, capacity: int, seed: int | None = None):
        self._buffer: deque[Tuple[numpy.ndarray, int, float,
                                  numpy.ndarray, bool]] = deque(maxlen=capacity)
        self._rng = python_random.Random(seed)

    def push(self, state: numpy.ndarray, action: int, reward: float,
             next_state: numpy.ndarray, done: bool) -> None:
        """Append a transition to the buffer.

        Parameters
        ----------
        state : ndarray
            Normalised state vector.
        action : int
            Action taken.
        reward : float
            Reward received.
        next_state : ndarray
            Normalised next-state vector.
        done : bool
            Whether the episode ended after this transition.
        """
        self._buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[numpy.ndarray, numpy.ndarray,
                                                numpy.ndarray, numpy.ndarray,
                                                numpy.ndarray]:
        """Sample a random mini-batch of transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        states : ndarray, shape ``(batch_size, state_dim)``
        actions : ndarray, shape ``(batch_size,)``
        rewards : ndarray, shape ``(batch_size,)``
        next_states : ndarray, shape ``(batch_size, state_dim)``
        dones : ndarray, shape ``(batch_size,)``
        """
        batch = self._rng.sample(list(self._buffer), batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            numpy.array(states, dtype=numpy.float32),
            numpy.array(actions, dtype=numpy.int32),
            numpy.array(rewards, dtype=numpy.float32),
            numpy.array(next_states, dtype=numpy.float32),
            numpy.array(dones, dtype=numpy.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
#  Training step (DQN)
# ---------------------------------------------------------------------------
def train_step_dqn(
    online_net: tf.keras.Model,
    target_net: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    states: tf.Tensor,
    actions: tf.Tensor,
    rewards: tf.Tensor,
    next_states: tf.Tensor,
    dones: tf.Tensor,
    discount: tf.Tensor,
    max_grad_norm: tf.Tensor,
    max_resources: tf.Tensor,
) -> tf.Tensor:
    """Execute a single gradient-descent step for the online Q-network.

    Uses the **Double DQN** objective (van Hasselt et al., 2016) with
    feasibility masking on the bootstrapped action:

    - **Double DQN**: the next-state action is selected by the *online*
      network (``argmax_a Q_θ(s', a)``) and evaluated with the *target*
      network (``Q_{θ⁻}(s', a*)``).  Eliminates the maximisation bias of
      vanilla DQN.
    - **Feasibility mask**: actions exceeding ``next_state[0]`` (resources
      left) are masked with ``-∞`` before the ``argmax``, so the bootstrap
      never relies on impossible actions.
    - **Huber loss**: Smooth L1 loss, less sensitive to outlier rewards
      than MSE.
    - **Gradient clipping**: ``tf.clip_by_global_norm`` to prevent
      exploding gradients.

    Loss function:

    .. math::
        a^{*}_i &= \\arg\\max_{a \\in \\mathcal{F}(s'_i)} Q_{\\theta}(s'_i, a) \\\\
        L &= \\frac{1}{B} \\sum_{i} \\mathrm{Huber}\\!\\left(
            Q_{\\theta}(s_i, a_i)
            - \\bigl[ r_i + \\gamma\\, Q_{\\theta^-}(s'_i, a^{*}_i)
              \\cdot (1 - d_i) \\bigr]
        \\right)

    where :math:`\\mathcal{F}(s')` is the set of feasible actions in state
    :math:`s'`.

    Parameters
    ----------
    online_net : tf.keras.Model
        Q-network whose weights are updated.
    target_net : tf.keras.Model
        Frozen Q-network used to compute TD targets.
    optimizer : tf.keras.optimizers.Optimizer
        Optimiser instance (e.g. Adam).
    states : tf.Tensor, shape ``(B, state_dim)``
    actions : tf.Tensor, shape ``(B,)``   int32
    rewards : tf.Tensor, shape ``(B,)``
    next_states : tf.Tensor, shape ``(B, state_dim)``
    dones : tf.Tensor, shape ``(B,)``
    discount : tf.Tensor
        Scalar discount factor γ.
    max_grad_norm : tf.Tensor
        Scalar global gradient norm clipping threshold.
    max_resources : tf.Tensor
        Scalar (float32) — environment ``max_resources``, used to recover
        the integer ``resources_left`` from the normalised ``next_states``
        for the feasibility mask.

    Returns
    -------
    loss : tf.Tensor
        Scalar Huber loss for the mini-batch.
    """
    # ---- Double DQN target with feasibility mask ----
    online_next_q = online_net(next_states, training=False)        # (B, A)
    target_next_q = target_net(next_states, training=False)        # (B, A)

    action_dim = tf.shape(online_next_q)[1]
    # Recover integer resources_left from normalised next_state[:, 0]
    res_left = tf.round(next_states[:, 0] * max_resources)         # (B,)
    action_indices = tf.cast(tf.range(action_dim), tf.float32)     # (A,)
    feasible = action_indices[tf.newaxis, :] <= res_left[:, tf.newaxis]  # (B, A)
    neg_inf = tf.constant(-1e9, dtype=tf.float32)
    masked_online_next_q = tf.where(feasible, online_next_q, neg_inf)

    best_actions = tf.argmax(masked_online_next_q, axis=1, output_type=tf.int32)
    target_mask = tf.one_hot(best_actions, depth=action_dim)
    bootstrap_q = tf.reduce_sum(target_next_q * target_mask, axis=1)
    td_targets = rewards + discount * bootstrap_q * (1.0 - dones)

    with tf.GradientTape() as tape:
        q_values = online_net(states, training=True)
        action_mask = tf.one_hot(actions, depth=action_dim)
        predicted_q = tf.reduce_sum(q_values * action_mask, axis=1)
        loss = tf.reduce_mean(tf.keras.losses.huber(td_targets, predicted_q))

    grads = tape.gradient(loss, online_net.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
    optimizer.apply_gradients(zip(grads, online_net.trainable_variables))

    return loss


def sync_target_network(online_net: tf.keras.Model,
                        target_net: tf.keras.Model) -> None:
    """Hard-copy online network weights to the target network.

    Parameters
    ----------
    online_net : tf.keras.Model
        Source network with up-to-date weights.
    target_net : tf.keras.Model
        Destination network to overwrite.
    """
    target_net.set_weights(online_net.get_weights())
