"""Ensemble inference engine for pes_ens.

This module implements an *ensemble agent* that fuses the policies of the
four sibling RL packages -- ``pes_dqn``, ``pes_a2c``, ``pes_rdqn`` and
``pes_trf`` -- by **soft voting**.  At every trial each member produces
its own action-probability distribution over the ``11`` discrete
allocations ``{0, 1, ..., 10}``; the distributions are averaged using
configurable weights, infeasible actions (allocation > resources_left)
are masked, the distribution is renormalised and ``argmax`` selects the
final allocation.

Key design points
-----------------
- **No cross-package imports.**  Each member model is a self-contained
  ``.keras`` artefact; pes_ens reads them straight off the filesystem
  via ``tf.keras.models.load_model``.  Required local helpers
  (``normalize_state``, ``HistoryDeque``) are therefore re-implemented
  here verbatim from the sibling packages so they remain
  bit-equivalent.
- **safe_mode=False** is mandatory when loading the pes_trf model
  because its causal-attention output uses a ``Lambda`` layer; Keras
  3.x refuses to deserialise ``Lambda`` layers in the default
  ``safe_mode=True`` (CWE-502 mitigation).  The flag is also harmless
  for the other three members.  We trust the artefacts because they
  are produced by our own training pipeline and shipped with the
  workspace.
- **Per-episode history isolation.**  Recurrent members (RDQN, TRF)
  require a sliding window of past normalised states.  The
  :class:`EnsembleAgent` keeps one independent ``HistoryDeque`` per
  member per ``(session_no, sequence_no)`` key and resets it whenever
  a new episode is detected.
- **Soft voting only.**  Hard voting (one-hot per member followed by
  majority) is intentionally not implemented: with four members it
  ties too often and discards uncertainty information that the
  averaged distribution preserves.

Public API
----------
- :class:`EnsembleAgent` -- stateful inference engine with
  :meth:`EnsembleAgent.predict` returning the averaged feasible
  distribution.
- :func:`normalize_state` -- ``[0, 1]^3`` state scaler shared with
  every member's training pipeline.
- :class:`HistoryDeque` -- fixed-length sliding window of past
  normalised states (verbatim copy of the sibling implementation).
"""

##########################
##  Imports externos    ##
##########################
import os
from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy
import tensorflow as tf

##########################
##  Imports internos    ##
##########################
from .. import INPUTS_PATH, MAX_SEVERITY


# ---------------------------------------------------------------------------
#  State normalisation (verbatim from sibling packages)
# ---------------------------------------------------------------------------
def normalize_state(state, max_resources: int,
                    max_trials: int, max_severity: int) -> numpy.ndarray:
    """Scale a raw integer state to ``[0, 1]^3``.

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
#  Sliding history window (verbatim from sibling packages)
# ---------------------------------------------------------------------------
def make_history_window(history: "Deque[numpy.ndarray]",
                        history_len: int,
                        state_dim: int) -> numpy.ndarray:
    """Materialise a deque of past states into a fixed-shape array.

    The most recent state appears in the *last* row.  Missing prefix
    rows (beginning of an episode) are filled with zeros.

    Parameters
    ----------
    history : collections.deque of ndarray
        Past normalised states, oldest first.
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
        """
        self._buffer.append(numpy.asarray(state, dtype=numpy.float32))

    def current_window(self) -> numpy.ndarray:
        """Return the current ``(history_len, state_dim)`` window."""
        return make_history_window(self._buffer, self._history_len, self._state_dim)

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _softmax(logits: numpy.ndarray, temperature: float = 1.0) -> numpy.ndarray:
    """Numerically stable softmax with optional temperature.

    Parameters
    ----------
    logits : ndarray
        Raw scores (e.g. Q-values) of shape ``(action_dim,)``.
    temperature : float, optional
        Softmax temperature; lower values produce sharper distributions.
        Default ``1.0``.

    Returns
    -------
    ndarray
        Probability distribution summing to 1, same shape as ``logits``.
    """
    scaled = numpy.asarray(logits, dtype=numpy.float64) / max(float(temperature), 1e-6)
    scaled = scaled - numpy.max(scaled)
    exps = numpy.exp(scaled)
    return (exps / numpy.sum(exps)).astype(numpy.float32)


def _resolve_path(rel_path: str) -> str:
    """Resolve a member-model path to an absolute filesystem location.

    Paths in ``CONFIG.ENS_MEMBER_MODELS`` are written relative to the
    pes_ens package root (``../pes_xxx/inputs/...``).  This helper
    joins them with ``INPUTS_PATH``'s parent (the package directory)
    and normalises the result.

    Parameters
    ----------
    rel_path : str
        Path string from CONFIG, possibly using forward slashes.

    Returns
    -------
    str
        Absolute, OS-native path.
    """
    pkg_root = os.path.dirname(INPUTS_PATH)
    return os.path.normpath(os.path.join(pkg_root, rel_path))


# ---------------------------------------------------------------------------
#  Ensemble engine
# ---------------------------------------------------------------------------
ACTION_DIM = 11      # discrete allocations 0..10
STATE_DIM = 3        # [resources_left, trial_no, severity]


class EnsembleAgent:
    """Soft-voting ensemble over four pre-trained sibling models.

    Each member is one of three roles -- ``q_dense`` (DQN-style),
    ``actor`` (A2C policy network) or ``q_recurrent`` (RDQN/TRF) --
    selected via the ``role`` field of ``CONFIG.ENS_MEMBER_MODELS``.
    Models are loaded eagerly in :meth:`__init__` and cached for the
    lifetime of the agent.

    Parameters
    ----------
    members_config : list of dict
        Subset of ``CONFIG.ENS_MEMBER_MODELS`` with ``enabled=True``.
    softmax_temperature : float, optional
        Temperature applied to Q-value members before softmax.
        Default ``1.0``.
    severity_prior_weight : float, optional
        Mixing coefficient ``w`` in ``[0, 1]`` for the Gaussian
        severity prior; ``0.0`` disables the prior.  Default ``0.0``.
    severity_prior_sigma : float, optional
        Standard deviation of the Gaussian severity prior in *raw*
        severity units.  Smaller => sharper bias toward
        ``action == round(severity)``.  Default ``1.5``.

    Attributes
    ----------
    members : list of dict
        Loaded member descriptors with extra keys ``model`` (the Keras
        model) and ``weight_norm`` (re-normalised weight in ``[0, 1]``).
    """

    def __init__(self,
                 members_config: List[dict],
                 softmax_temperature: float = 1.0,
                 severity_prior_weight: float = 0.0,
                 severity_prior_sigma: float = 1.5) -> None:
        self._softmax_temperature = float(softmax_temperature)
        self._prior_weight = float(numpy.clip(severity_prior_weight, 0.0, 1.0))
        self._prior_sigma = max(float(severity_prior_sigma), 1e-3)
        self.members: List[dict] = []
        # ``_history_caches[member_name]`` is itself a dict keyed by
        # ``(session_no, sequence_no)``, mapping to a HistoryDeque.
        self._history_caches: Dict[str, Dict[Tuple[int, int], HistoryDeque]] = {}

        # Load every enabled member exactly once.
        weight_sum = 0.0
        for cfg in members_config:
            if not cfg.get('enabled', True):
                continue
            abs_path = _resolve_path(cfg['path'])
            if not os.path.isfile(abs_path):
                raise FileNotFoundError(
                    f"\nFATAL ERROR: ensemble member '{cfg['name']}' model not "
                    f"found at:\n  {abs_path}\n"
                    f"Please train the producing package first."
                )
            try:
                # safe_mode=False is required by the pes_trf model (Lambda
                # layer) and is harmless for the other roles.  See module
                # docstring for the security justification.
                model = tf.keras.models.load_model(abs_path, safe_mode=False)
            except Exception as exc:
                raise RuntimeError(
                    f"\nFATAL ERROR: failed to load ensemble member "
                    f"'{cfg['name']}' from {abs_path}\nError: {exc}"
                ) from exc

            weight = float(cfg.get('weight', 1.0))
            weight_sum += weight
            self.members.append({
                'name':        cfg['name'],
                'role':        cfg['role'],
                'history_len': int(cfg.get('history_len', 1)),
                'model':       model,
                'weight':      weight,
            })
            self._history_caches[cfg['name']] = {}

        if not self.members:
            raise ValueError("EnsembleAgent requires at least one enabled member.")
        if weight_sum <= 0.0:
            raise ValueError("Sum of ensemble member weights must be > 0.")
        for m in self.members:
            m['weight_norm'] = m['weight'] / weight_sum

    # ------------------------------------------------------------------ #
    def reset_episode(self, session_no: int, sequence_no: int) -> None:
        """Clear the history buffers of every recurrent member.

        Should be called once at the start of each new sequence to
        prevent state leakage across episodes.

        Parameters
        ----------
        session_no : int
            Block index of the new episode.
        sequence_no : int
            Sequence index of the new episode within the block.
        """
        key = (int(session_no), int(sequence_no))
        for member in self.members:
            cache = self._history_caches[member['name']]
            if key in cache:
                cache[key].reset()

    # ------------------------------------------------------------------ #
    def _member_distribution(self,
                             member: dict,
                             state_norm: numpy.ndarray,
                             session_no: int,
                             sequence_no: int) -> numpy.ndarray:
        """Compute the action-probability distribution of one member.

        Parameters
        ----------
        member : dict
            Loaded member descriptor (entry of :attr:`members`).
        state_norm : ndarray
            Current normalised state, shape ``(3,)``.
        session_no : int
            Block index, used to key the history cache.
        sequence_no : int
            Sequence index, used to key the history cache.

        Returns
        -------
        ndarray
            Probability distribution over ``ACTION_DIM`` actions.
        """
        role = member['role']
        model = member['model']

        if role == 'q_recurrent':
            # Maintain an independent sliding window per episode.
            cache = self._history_caches[member['name']]
            key = (int(session_no), int(sequence_no))
            history = cache.get(key)
            if history is None:
                history = HistoryDeque(member['history_len'], STATE_DIM)
                cache[key] = history
            history.append_step(state_norm)
            window = history.current_window()
            tensor = window[numpy.newaxis]               # (1, T, 3)
            q_values = model(tensor, training=False).numpy().flatten()
            return _softmax(q_values, self._softmax_temperature)

        if role == 'q_dense':
            tensor = state_norm[numpy.newaxis, :]        # (1, 3)
            q_values = model(tensor, training=False).numpy().flatten()
            return _softmax(q_values, self._softmax_temperature)

        if role == 'actor':
            tensor = state_norm[numpy.newaxis, :]        # (1, 3)
            probs = model(tensor, training=False).numpy().flatten()
            # Defensive: enforce non-negativity & renormalisation.
            probs = numpy.clip(probs, 0.0, None)
            total = float(numpy.sum(probs))
            if total <= 0.0:
                return numpy.full(ACTION_DIM, 1.0 / ACTION_DIM, dtype=numpy.float32)
            return (probs / total).astype(numpy.float32)

        raise ValueError(f"Unknown ensemble member role: {role!r}")

    # ------------------------------------------------------------------ #
    def predict(self,
                state_norm: numpy.ndarray,
                resources_left: int,
                session_no: int,
                sequence_no: int) -> Tuple[numpy.ndarray, Dict[str, numpy.ndarray]]:
        """Return the soft-voted ensemble distribution for one trial.

        Pipeline applied per call:

        1. Per-member inference + softmax (Q-value roles only).
        2. Per-member feasibility masking + renormalisation.
        3. Confidence-weighted voting with dynamic weight
           ``w_norm * (0.1 + (1 - H_norm))``.
        4. Action-0 penalty ``* 0.3`` when ``max_feasible > 0``.
        5. Mixing with a Gaussian severity prior centred at the raw
           severity, weighted by ``severity_prior_weight``.
        6. Severity-floor safety override: if
           ``severity_raw >= 6`` and ``argmax < floor`` (with
           ``floor = severity_raw // 2``), the distribution becomes a
           one-hot at ``floor``.

        Parameters
        ----------
        state_norm : ndarray
            Current normalised state, shape ``(3,)``.
        resources_left : int
            Remaining resource budget for the current sequence.  Used
            to mask infeasible actions.
        session_no : int
            Block index (for history cache keying).
        sequence_no : int
            Sequence index (for history cache keying).

        Returns
        -------
        ensemble_probs : ndarray
            Final ensemble distribution of shape ``(ACTION_DIM,)``,
            already feasibility-masked, prior-mixed and (possibly)
            floor-overridden.
        per_member_probs : dict of ndarray
            ``{member_name: distribution}`` with the *unmasked*
            per-member distributions; useful for diagnostics and
            logging only.
        """
        max_feasible = max(0, int(resources_left))
        ensemble = numpy.zeros(ACTION_DIM, dtype=numpy.float64)
        per_member: Dict[str, numpy.ndarray] = {}
        log2_n = numpy.log2(ACTION_DIM)

        for member in self.members:
            raw = self._member_distribution(
                member, state_norm, session_no, sequence_no)
            per_member[member['name']] = raw

            # (e) Mask infeasible actions *per member* and renormalise
            # before mixing, so each member only "votes" within its
            # feasible support.
            masked = raw.astype(numpy.float64).copy()
            if max_feasible < ACTION_DIM - 1:
                masked[max_feasible + 1:] = 0.0
            ms = float(numpy.sum(masked))
            if ms <= 0.0:
                masked = numpy.zeros(ACTION_DIM, dtype=numpy.float64)
                masked[0] = 1.0
            else:
                masked = masked / ms

            # (d) Confidence-weighted voting: scale the static config
            # weight by the inverse normalised entropy of this member's
            # *feasible* distribution.  Sharp distributions weigh more.
            p = numpy.clip(masked, 1e-9, 1.0)
            entropy_norm = float(-numpy.sum(p * numpy.log2(p)) / log2_n)
            confidence = 1.0 - entropy_norm  # in [0, 1]
            dyn_weight = member['weight_norm'] * (0.1 + confidence)

            ensemble += dyn_weight * masked

        # (3) Penalise the "do nothing" action when there are still
        # resources available: the experiment record marks r==0 as
        # "no response" (confidence=-1), wasting the trial.
        if max_feasible > 0:
            ensemble[0] *= 0.3

        total = float(numpy.sum(ensemble))
        if total <= 0.0:
            # Pathological fallback: collapse to "allocate 0".
            ensemble = numpy.zeros(ACTION_DIM, dtype=numpy.float64)
            ensemble[0] = 1.0
        else:
            ensemble = ensemble / total

        # (4) Severity-prior bias: blend the ensemble with a Gaussian
        # prior centred at the current raw severity.  Recovers domain
        # knowledge "action ~ severity" when the models are uncertain.
        severity_raw = float(state_norm[2]) * float(MAX_SEVERITY)
        if self._prior_weight > 0.0:
            actions = numpy.arange(ACTION_DIM, dtype=numpy.float64)
            prior = numpy.exp(-((actions - severity_raw) ** 2)
                              / (2.0 * self._prior_sigma ** 2))
            if max_feasible < ACTION_DIM - 1:
                prior[max_feasible + 1:] = 0.0
            ps = float(numpy.sum(prior))
            if ps > 0.0:
                prior = prior / ps
                ensemble = ((1.0 - self._prior_weight) * ensemble
                            + self._prior_weight * prior)
                ensemble = ensemble / float(numpy.sum(ensemble))

        # (B) Severity-floor safety net.  If the trial faces a high
        # severity *and* there is enough budget, refuse to pick an
        # allocation below sev/2: this single rule eliminates the
        # catastrophic "severity 8, ensemble says 0-2" outliers that
        # produced the 0.754 minima in early blocks.
        if severity_raw >= 6.0:
            floor = int(severity_raw // 2)
            if max_feasible >= floor and int(numpy.argmax(ensemble)) < floor:
                ensemble = numpy.zeros(ACTION_DIM, dtype=numpy.float64)
                ensemble[floor] = 1.0

        return ensemble.astype(numpy.float32), per_member
