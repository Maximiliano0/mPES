"""Ensemble Agent Game Display Mediator and Response Handler.

Bridges the Pygame interface with the ensemble agent (pes_dqn +
pes_a2c + pes_rdqn + pes_trf), handling per-trial inference, response
timing and entropy-based confidence calculation.

Key Functions
-------------
- :func:`ens_agent_meta_cognitive` -- entropy-based confidence and
  reaction-time simulation, identical in spirit to the per-package
  ``*_agent_meta_cognitive`` helpers.
- :func:`provide_ens_agent_response` -- main entry point invoked from
  ``__main__`` for every trial.
- :func:`reset` -- pygame teardown placeholder (the ensemble agent
  needs no display).

Module Globals
--------------
``first_severity`` and ``number_of_trials`` are populated by the caller
(``__main__``) before the first trial.

A single :class:`EnsembleAgent` instance is created lazily on first use
and cached at module level so the four member models are loaded only
once per experiment session.
"""

##########################
##  Imports externos    ##
##########################
import os

import numpy

##########################
##  Imports internos    ##
##########################
from .. import (ANSI, AVAILABLE_RESOURCES_PER_SEQUENCE, INPUTS_PATH,
                MAX_SEVERITY, NUM_MAX_TRIALS, NUM_SEQUENCES,
                SEQ_LENGTHS_FILE, VERBOSE)
from ..config.CONFIG import (ENS_MEMBER_MODELS, ENS_SEVERITY_PRIOR_SIGMA,
                             ENS_SEVERITY_PRIOR_WEIGHT,
                             ENS_SOFTMAX_TEMPERATURE)
from ..ext.ensemble_model import EnsembleAgent, normalize_state
from ..ext.tools import convert_globalseq_to_seqs, entropy_from_pdf
from . import log_utils

##########################################################
## Variables requiring initialisation before module use ##
##########################################################
first_severity = None
number_of_trials = None

#################################
## Module-Specific constants   ##
#################################
FONT = 'ubuntumono'
BACKGROUND_COLOUR = ANSI.GRAY
RESPONSE_TIMEOUT = 5000  # in milliseconds

# Cached EnsembleAgent shared across trials (created lazily).
_ensemble_agent: EnsembleAgent | None = None


def _get_agent() -> EnsembleAgent:
    """Return the lazily instantiated module-level :class:`EnsembleAgent`.

    Returns
    -------
    EnsembleAgent
        The cached ensemble inference engine; built on first call from
        ``CONFIG.ENS_MEMBER_MODELS`` and reused on every subsequent
        trial.
    """
    global _ensemble_agent  # pylint: disable=global-statement
    if _ensemble_agent is None:
        _ensemble_agent = EnsembleAgent(
            members_config=ENS_MEMBER_MODELS,
            softmax_temperature=ENS_SOFTMAX_TEMPERATURE,
            severity_prior_weight=ENS_SEVERITY_PRIOR_WEIGHT,
            severity_prior_sigma=ENS_SEVERITY_PRIOR_SIGMA,
        )
        if VERBOSE:
            print(f"{ANSI.GREEN}EnsembleAgent ready with {len(_ensemble_agent.members)} "
                  f"members:{ANSI.RESET}")
            for m in _ensemble_agent.members:
                print(f"  - {m['name']:<6} role={m['role']:<12} "
                      f"weight={m['weight_norm']:.3f}")
    return _ensemble_agent


def reset() -> None:
    """No-op kept for API parity with sibling pygameMediator modules."""
    return


def ens_agent_meta_cognitive(probs, resources_left, response_timeout):
    """Generate ensemble response with meta-cognitive confidence and timing.

    Mirrors the entropy-based heuristic used by every sibling package
    so that ensemble outputs are directly comparable to single-model
    runs: confidence is the inverse-entropy of the *feasible* part of
    the averaged distribution, and reaction times are sampled from a
    Gaussian whose mean is mapped from the confidence value.

    Parameters
    ----------
    probs : array-like
        Pre-masked ensemble distribution of shape ``(11,)``.
    resources_left : float or int
        Remaining resource budget; used to clamp the final response.
    response_timeout : int
        Max response time in milliseconds (e.g. ``5000``).

    Returns
    -------
    response : int
        Selected resource allocation (``argmax`` of ``probs``).
    confidence : float
        Confidence in ``[0, 1]`` derived from distribution entropy.
    rt_hold : float
        Simulated reaction time from stimulus to button press, in
        seconds.
    rt_release : float
        Simulated total response duration in seconds; always
        ``>= rt_hold``.
    """
    probs = numpy.asarray(probs, dtype=numpy.float64).copy()
    log_utils.tee('Ensemble probs:', probs)

    # Reference distributions for confidence normalisation.
    m_entropy = numpy.zeros((11,))
    m_entropy[0] = 1.0
    M_entropy = numpy.ones((11,))

    o = numpy.arange(11, dtype=numpy.float32)
    probs[o > resources_left] = 0.00001

    log_utils.tee('Ensemble Feasible Probs:', probs)

    dec_entropy = entropy_from_pdf(probs)
    M_entropy = entropy_from_pdf(M_entropy)
    m_entropy = entropy_from_pdf(m_entropy)

    confidence = (1.0 / (m_entropy - M_entropy)) * (dec_entropy - M_entropy)

    response = int(numpy.argmax(probs))
    response = int(numpy.clip(response, 0, int(resources_left)))

    def map_to_response_time(x):
        """Map confidence to a reaction-time mean (seconds)."""
        return x * (-2) + 1

    mu, sigma = int(map_to_response_time(confidence) * 10), 3
    rt_hold = numpy.random.normal(mu, sigma, 1)[0]
    rt_release = rt_hold + numpy.random.normal(mu, 1, 1)[0]
    rt_hold = numpy.clip(rt_hold, 0, response_timeout / 1000.0)
    rt_release = numpy.clip(rt_release, 0, response_timeout / 1000.0)

    return response, float(confidence), float(rt_hold), float(rt_release)


def provide_ens_agent_response(
    _resources,
    resources_left,
    session_no,
    sequence_no,
    trial_no,
):
    """Generate an ensemble response for the current trial.

    Loads (lazily) the four member models, builds the current state
    vector, queries every member, averages their distributions with
    feasibility masking and returns the chosen allocation together
    with simulated timing and confidence metadata.

    Parameters
    ----------
    _resources : float
        Total resources available in the session (informational).
    resources_left : float or int
        Remaining resources available for allocation.
    session_no : int
        Block index (0-based).
    sequence_no : int
        Sequence index within the block (0-based).
    trial_no : int
        Trial index within the sequence (0-based).

    Returns
    -------
    confidence : float
        Decision confidence in ``[0, 1]``.
    response : int
        Resource allocation in ``[0, resources_left]``.
    rt_hold : float
        Reaction time in seconds (stimulus to button press).
    rt_release : float
        Total response duration in seconds.
    movement : list
        Empty placeholder for movement data.

    Raises
    ------
    AssertionError
        If ``first_severity`` was not initialised by the caller.
    FileNotFoundError
        If any enabled member model file is missing.
    RuntimeError
        If a member model file fails to deserialise.
    """
    assert first_severity is not None, \
        "The 'first_severity' module-global variable needs to be set by caller before calling this function"

    agent = _get_agent()

    # Reset per-episode history at the very first trial of every sequence.
    if int(trial_no) == 0:
        agent.reset_episode(session_no, sequence_no)

    SequenceLengthsCsv = os.path.join(INPUTS_PATH, SEQ_LENGTHS_FILE)
    sequence_length = numpy.loadtxt(SequenceLengthsCsv, delimiter=',')
    sevs = convert_globalseq_to_seqs(sequence_length, first_severity)

    sever = sevs[session_no * NUM_SEQUENCES + sequence_no][trial_no]
    city_number = trial_no

    try:
        resources_idx = int(resources_left)
    except (ValueError, TypeError):
        resources_idx = int(resources_left.numpy())
    try:
        city_idx = int(city_number)
    except (ValueError, TypeError):
        city_idx = int(city_number.numpy())
    try:
        sever_idx = int(sever)
    except (ValueError, TypeError):
        sever_idx = int(sever.numpy())

    # Normalisation limits derived from CONFIG (single source of truth).
    # ``-9`` mirrors the offset used in every sibling Pandemic env so the
    # state encoding is bit-equivalent to the producing packages.
    max_res = AVAILABLE_RESOURCES_PER_SEQUENCE - 9
    max_tri = NUM_MAX_TRIALS
    max_sev = MAX_SEVERITY

    resources_idx = max(0, min(resources_idx, max_res))
    city_idx = max(0, min(city_idx, max_tri))
    sever_idx = max(0, min(sever_idx, max_sev))

    state_norm = normalize_state(
        [resources_idx, city_idx, sever_idx], max_res, max_tri, max_sev
    )

    if VERBOSE:
        print(f"State indices - Resources: {resources_idx}, "
              f"City: {city_idx}, Severity: {sever_idx}")

    ensemble_probs, per_member = agent.predict(
        state_norm=state_norm,
        resources_left=int(resources_left),
        session_no=int(session_no),
        sequence_no=int(sequence_no),
    )

    if VERBOSE:
        for name, probs in per_member.items():
            top = int(numpy.argmax(probs))
            print(f"  member {name:<6} top={top:<2} p_top={probs[top]:.3f}")
        top_ens = int(numpy.argmax(ensemble_probs))
        print(f"  ENSEMBLE   top={top_ens:<2} p_top={ensemble_probs[top_ens]:.3f}")

    resp, confidence, rt_hold, rt_release = ens_agent_meta_cognitive(
        ensemble_probs, resources_left, RESPONSE_TIMEOUT)

    resp = int(numpy.clip(resp, 0, int(resources_left)))

    if VERBOSE:
        print(f"ENS Agent Response: {resp}, Confidence: {confidence}")
        print(f"Resources available: {int(resources_left)}, Response clamped to: {resp}")

    movement: list = []
    return confidence, resp, rt_hold, rt_release, movement
