"""
pes_trf — Transformer Deep Q-Network model components.

Provides the neural-network building blocks used by the TRF training loop
in :pymod:`pandemic`:

- **build_q_network**:  Constructs a Keras model that maps a *windowed*
  state sequence (the last ``history_len`` normalised states of the
  current episode) to Q-values for every discrete action.  The trunk
  is a stack of **causal Transformer encoder blocks** (multi-head
  self-attention + position-wise feed-forward + residual + layer
  normalisation) applied to a learned linear projection of the input
  with additive learned positional embeddings; the Q-head is a small
  MLP fed by the *last* token of the sequence.
- **normalize_state**:  Scales a raw integer state to ``[0, 1]``.
- **make_history_window**:  Materialises a deque of past normalised
  states into a ``(history_len, state_dim)`` array, left-padded with
  zeros and consumable by the ``Masking`` layer.
- **HistoryDeque**:  Convenience wrapper that resets at the start of
  every episode and exposes ``append_step`` / ``current_window``.
- **ReplayBuffer**:  Fixed-size circular buffer storing experience
  tuples ``(state_window, action, reward, next_state_window, done)``
  for off-policy learning.
- **train_step_trf**:  Single gradient-descent update implementing the
  **Double DQN** target (van Hasselt et al., 2016) on the Transformer
  Q-network with per-sample feasibility masking on the bootstrapped
  action and Huber loss on the resulting TD error.

Architecture
------------
::

    Q-Network: Input(history_len, state_dim)
               -> Masking(0.0)
               -> Dense(d_model)            (token embedding)
               -> + learned positional embedding
               -> [ MultiHeadAttention(num_heads, key_dim, causal)
                    -> Add & LayerNorm
                    -> FFN(ff_dim, ReLU) -> Dense(d_model)
                    -> Add & LayerNorm ] x num_layers
               -> take last token
               -> Dense(h_0, ReLU) -> ... -> Dense(h_n, ReLU)
               -> Dense(action_dim, linear)

Defaults are configurable via ``config/CONFIG.py``:
``TRF_HISTORY_LEN``, ``TRF_D_MODEL``, ``TRF_NUM_HEADS``,
``TRF_KEY_DIM``, ``TRF_FF_DIM``, ``TRF_NUM_LAYERS``, ``TRF_DROPOUT``,
``TRF_HIDDEN_UNITS``.

CPU Optimisations
-----------------
- TensorFlow intra/inter-op thread pools are pinned to a single thread
  at import time and ``tf.config.experimental.enable_op_determinism()``
  is enabled, so per-step ordering is deterministic.
"""

##########################
##  Imports externos    ##
##########################
from typing import Deque, List, Tuple
from collections import deque

import numpy
import random as python_random
import tensorflow as tf

##########################
##  Imports internos    ##
##########################


# ---------------------------------------------------------------------------
#  CPU threading optimisation -- single-threaded for determinism
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
                    history_len: int = 6,
                    d_model: int = 32,
                    num_heads: int = 4,
                    key_dim: int = 16,
                    ff_dim: int = 64,
                    num_layers: int = 2,
                    dropout: float = 0.0,
                    seed: int | None = None) -> tf.keras.Model:
    """Build a Transformer-encoder Q-network with a dense head.

    The network consumes a left-padded sequence of the last
    ``history_len`` normalised states (zero-padded at the start of an
    episode) and outputs Q-values for every discrete action.  A
    ``Masking`` layer marks the all-zero pad rows as missing tokens, a
    learned linear projection lifts each timestep to ``d_model`` features,
    a learned positional embedding is added, and ``num_layers`` causal
    Transformer encoder blocks (multi-head self-attention +
    position-wise feed-forward, both with residual connection and layer
    normalisation) summarise the sequence.  Only the *last* token feeds
    the dense Q-head.

    Parameters
    ----------
    state_dim : int
        Dimensionality of one (normalised) state vector.
    action_dim : int
        Number of discrete actions (network output width).
    hidden_units : list of int
        Widths of the dense head applied after the encoder
        (e.g. ``[64]`` for a single hidden layer).
    history_len : int, optional
        Number of past timesteps fed to the encoder.  Default: 6.
    d_model : int, optional
        Token-embedding (and residual-stream) width.  Default: 32.
    num_heads : int, optional
        Number of attention heads per encoder block.  Default: 4.
    key_dim : int, optional
        Per-head key/query dimensionality.  Default: 16.
    ff_dim : int, optional
        Hidden width of the position-wise feed-forward sub-layer.
        Default: 64.
    num_layers : int, optional
        Number of stacked encoder blocks.  Default: 2.
    dropout : float, optional
        Dropout rate applied inside the attention and FFN sub-layers.
        Default: 0.0.
    seed : int or None, optional
        Seed for the Glorot-uniform kernel initialisers.  Each layer
        gets a distinct derived seed.  Default: ``None`` (Keras default
        RNG).

    Returns
    -------
    tf.keras.Model
        Keras functional model with linear output of shape
        ``(action_dim,)``.
    """
    def _init(layer_idx: int):
        if seed is None:
            return "glorot_uniform"
        return tf.keras.initializers.GlorotUniform(seed=int(seed) + layer_idx)

    inputs = tf.keras.layers.Input(shape=(int(history_len), int(state_dim)),
                                   name="history_window")
    x = tf.keras.layers.Masking(mask_value=0.0, name="history_mask")(inputs)
    # Token embedding: lift each (state_dim,) vector to (d_model,).
    x = tf.keras.layers.Dense(int(d_model), kernel_initializer=_init(0),
                              name="token_embed")(x)
    # Learned positional embedding added to every token.
    positions = tf.range(start=0, limit=int(history_len), delta=1)
    pos_emb = tf.keras.layers.Embedding(
        input_dim=int(history_len), output_dim=int(d_model),
        embeddings_initializer=_init(1), name="pos_embed")(positions)
    x = x + pos_emb

    # Stack of causal Transformer encoder blocks.
    for blk in range(int(num_layers)):
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=int(num_heads), key_dim=int(key_dim),
            dropout=float(dropout),
            kernel_initializer=_init(2 + 4 * blk),
            name=f"encoder_{blk}_mhsa")(x, x, use_causal_mask=True)
        x = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f"encoder_{blk}_ln1")(x + attn)
        ffn = tf.keras.layers.Dense(
            int(ff_dim), activation="relu",
            kernel_initializer=_init(3 + 4 * blk),
            name=f"encoder_{blk}_ffn1")(x)
        if dropout > 0.0:
            ffn = tf.keras.layers.Dropout(
                float(dropout), name=f"encoder_{blk}_drop")(ffn)
        ffn = tf.keras.layers.Dense(
            int(d_model),
            kernel_initializer=_init(4 + 4 * blk),
            name=f"encoder_{blk}_ffn2")(ffn)
        x = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f"encoder_{blk}_ln2")(x + ffn)

    # Read out the last token (most recent timestep).
    last_token = tf.keras.layers.Lambda(
        lambda t: t[:, -1, :], name="last_token",
        output_shape=(int(d_model),))(x)

    head = last_token
    init_offset = 2 + 4 * int(num_layers)
    for idx, units in enumerate(hidden_units):
        head = tf.keras.layers.Dense(
            int(units), activation="relu",
            kernel_initializer=_init(init_offset + idx),
            name=f"q_hidden_{idx}")(head)
    outputs = tf.keras.layers.Dense(
        int(action_dim), activation="linear",
        kernel_initializer=_init(init_offset + len(hidden_units)),
        name="q_values")(head)

    return tf.keras.Model(inputs=inputs, outputs=outputs,
                          name="Transformer_Q_Network")


# ---------------------------------------------------------------------------
#  State normalisation
# ---------------------------------------------------------------------------
def normalize_state(state, max_resources: int,
                    max_trials: int, max_severity: int) -> numpy.ndarray:
    """Scale raw integer state components to ``[0, 1]``.

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
#  Sliding history window
# ---------------------------------------------------------------------------
def make_history_window(history: "Deque[numpy.ndarray]",
                        history_len: int,
                        state_dim: int) -> numpy.ndarray:
    """Materialise a deque of past states into a fixed-shape array.

    The most recent state appears in the *last* row.  Missing prefix
    rows (beginning of an episode) are filled with zeros so the
    ``Masking`` layer can ignore them.

    Parameters
    ----------
    history : collections.deque of ndarray
        Past normalised states, oldest first.  Length <= ``history_len``.
    history_len : int
        Total window length.
    state_dim : int
        Dimensionality of each state vector.

    Returns
    -------
    ndarray, shape ``(history_len, state_dim)``, dtype ``float32``
    """
    window = numpy.zeros((int(history_len), int(state_dim)), dtype=numpy.float32)
    items = list(history)[-int(history_len):]
    if items:
        window[-len(items):] = numpy.asarray(items, dtype=numpy.float32)
    return window


class HistoryDeque:
    """Per-episode sliding window of normalised states.

    Wraps a :class:`collections.deque` with a fixed maximum length and
    exposes the helpers used by the training loop and the inference
    bridge (``pygameMediator``):

    - :meth:`reset` clears the buffer at the start of an episode.
    - :meth:`append_step` records a new normalised state.
    - :meth:`current_window` materialises the current window for the
      Q-network.

    Parameters
    ----------
    history_len : int
        Window length (number of past timesteps retained).
    state_dim : int
        Dimensionality of each state vector.
    """

    def __init__(self, history_len: int, state_dim: int):
        self._history_len = int(history_len)
        self._state_dim = int(state_dim)
        self._buffer: Deque[numpy.ndarray] = deque(maxlen=self._history_len)

    def reset(self) -> None:
        """Clear the buffer at the start of a new episode."""
        self._buffer.clear()

    def append_step(self, state: numpy.ndarray) -> None:
        """Record a normalised state in the sliding window.

        Parameters
        ----------
        state : ndarray, shape ``(state_dim,)``
            Normalised state vector.
        """
        self._buffer.append(numpy.asarray(state, dtype=numpy.float32))

    def current_window(self) -> numpy.ndarray:
        """Return the current ``(history_len, state_dim)`` window."""
        return make_history_window(self._buffer, self._history_len, self._state_dim)

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
#  Replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Fixed-size circular buffer for storing experience tuples.

    Stores ``(state_window, action, reward, next_state_window, done)``
    transitions and supports uniform random sampling for mini-batch
    training.  ``state_window`` and ``next_state_window`` are both
    ``(history_len, state_dim)`` arrays.

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
        state : ndarray, shape ``(history_len, state_dim)``
            Current sliding-window snapshot.
        action : int
            Action taken.
        reward : float
            Reward received.
        next_state : ndarray, shape ``(history_len, state_dim)``
            Sliding-window snapshot after the step.
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
        states : ndarray, shape ``(batch_size, history_len, state_dim)``
        actions : ndarray, shape ``(batch_size,)``
        rewards : ndarray, shape ``(batch_size,)``
        next_states : ndarray, shape ``(batch_size, history_len, state_dim)``
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
#  Training step (Transformer DQN -- Double DQN target)
# ---------------------------------------------------------------------------
def train_step_trf(
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
    """Execute a single gradient-descent step for the Transformer Q-network.

    Uses the **Double DQN** objective (van Hasselt et al., 2016) on top
    of the Transformer encoder trunk, with feasibility masking on the
    bootstrapped action.  The feasibility mask is read from the *last*
    timestep of each next-state window (``next_states[:, -1, 0]``),
    which holds the normalised ``resources_left`` at decision time.

    Parameters
    ----------
    online_net : tf.keras.Model
        Transformer Q-network whose weights are updated.
    target_net : tf.keras.Model
        Frozen Transformer Q-network used to compute TD targets.
    optimizer : tf.keras.optimizers.Optimizer
        Optimiser instance (e.g. Adam).
    states : tf.Tensor, shape ``(B, T, state_dim)``
    actions : tf.Tensor, shape ``(B,)``   int32
    rewards : tf.Tensor, shape ``(B,)``
    next_states : tf.Tensor, shape ``(B, T, state_dim)``
    dones : tf.Tensor, shape ``(B,)``
    discount : tf.Tensor
        Scalar discount factor gamma.
    max_grad_norm : tf.Tensor
        Scalar global gradient norm clipping threshold.
    max_resources : tf.Tensor
        Scalar (float32) -- environment ``max_resources``, used to
        recover the integer ``resources_left`` from the normalised
        ``next_states`` for the feasibility mask.

    Returns
    -------
    loss : tf.Tensor
        Scalar Huber loss for the mini-batch.
    """
    # ---- Double DQN target with feasibility mask ----
    online_next_q = online_net(next_states, training=False)        # (B, A)
    target_next_q = target_net(next_states, training=False)        # (B, A)

    action_dim = tf.shape(online_next_q)[1]
    # Recover integer resources_left from the most recent next-state row.
    res_left = tf.round(next_states[:, -1, 0] * max_resources)     # (B,)
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
