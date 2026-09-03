"""A2C Agent Game Display Mediator and Response Handler.

This module bridges the Pygame game interface with the trained A2C agent
(Advantage Actor-Critic model), handling agent decision-making, response
timing, and confidence calculations.  It manages the communication between
the game display and the A2C agent by processing game states and generating
appropriately timed and confident responses.

Key Functions:
    - ac_agent_meta_cognitive: Meta-cognitive decision making with entropy-based confidence
    - provide_ac_agent_response: Main interface returning AC agent response and timing

Module Dependencies:
    External: numpy, tensorflow, os
    Internal: log_utils, convert_globalseq_to_seqs, CONFIG constants, ac_model

Global Variables:
    first_severity: Initial severity array loaded by caller
    number_of_trials: Total trial count for experiment

Author: PES Development Team
Version: 1.0  (A2C variant)
"""

##########################
##  Imports externos    ##
##########################
import numpy
import os
import tensorflow as tf

##########################
##  Imports internos    ##
##########################
from .. import (ANSI, AVAILABLE_RESOURCES_PER_SEQUENCE, INPUTS_PATH,
                 MAX_SEVERITY, NUM_MAX_TRIALS, NUM_SEQUENCES,
                 SEQ_LENGTHS_FILE, VERBOSE)
from ..config.CONFIG import AC_MODEL_ACTOR_FILE
from ..ext.tools import convert_globalseq_to_seqs, entropy_from_pdf
from ..ext.ac_model import normalize_state
from . import log_utils

##########################################################
## Variables requiring initialisation before module use ##
##########################################################
first_severity = None
number_of_trials = None

#################################
## Module-Specific constants   ##
#################################
FONT = 'ubuntumono'   # previously: Arial
BACKGROUND_COLOUR = ANSI.GRAY
RESPONSE_TIMEOUT = 5000  # in milliseconds


def ac_agent_meta_cognitive(policy_probs, resources_left, response_timeout):
    """Generate A2C agent response with meta-cognitive confidence and timing simulation.

    Implements a meta-cognitive decision-making process where the agent:
    1. Evaluates confidence based on entropy of Actor policy probabilities
    2. Filters infeasible options (exceeding available resources)
    3. Generates decision entropy from feasible options
    4. Maps confidence to response timing (reaction time, release time)

    Unlike Q-value-based entropy (a heuristic), this uses the Actor's π(a|s)
    directly — a true probability distribution, making entropy theoretically
    grounded as an uncertainty measure.

    Parameters
    ----------
    policy_probs : array_like
        1-D array of action probabilities from the Actor's softmax output.
        Typically shape (11,) representing π(a|s) for each allocation action.
    resources_left : float or int
        Available resources that constrain feasible actions.
    response_timeout : int
        Maximum response time in milliseconds (e.g., 5000 for 5 seconds).

    Returns
    -------
    response : int
        Selected resource allocation (argmax of feasible probability-masked actions).
        Guaranteed to not exceed resources_left.
    confidence : float
        Meta-cognitive confidence metric (0-1 range).
        - Based on entropy of feasible action-probability distribution.
        - 0 = uniform distribution (low confidence).
        - 1 = peaked distribution (high confidence).
    rt_hold : float
        Reaction time from stimulus to response button press (seconds).
        Sampled from Gaussian scaled by confidence.
    rt_release : float
        Total time from stimulus to releasing response button (seconds).
        Always >= rt_hold (represents commitment duration).

    Notes
    -----
    - Confidence inverted from entropy: (1/(m_entropy - M_entropy)) * (H - M_entropy)
    - High confidence → faster response (lower RT)
    - Low confidence → longer deliberation (higher RT)
    - All response times clipped to [0, response_timeout/1000]
    - Response always clipped to feasible range [0, resources_left]
    """

    # Min entropy from a univalue distribution (0)
    m_entropy = numpy.zeros((11,),)
    m_entropy[0] = 1

    # Max entropy from a uniform distribution (3.55....)
    M_entropy = numpy.ones((11,),)

    # Policy probabilities from the Actor
    log_utils.tee('Policy probs:', policy_probs)

    _entrp1 = entropy_from_pdf(policy_probs)

    o = [i for i in range(len(policy_probs))]
    o = numpy.asarray(o, dtype=numpy.float32)

    # Mask infeasible actions (allocation exceeds available resources)
    policy_probs[o > resources_left] = 0.00001

    log_utils.tee('Agent Feasible Probs:', policy_probs)

    # Compute entropy of feasible options for confidence metric
    dec_entropy = entropy_from_pdf(policy_probs)
    M_entropy = entropy_from_pdf(M_entropy)
    m_entropy = entropy_from_pdf(m_entropy)

    confidence = (1. / (m_entropy - M_entropy)) * (dec_entropy - M_entropy)

    response = numpy.argmax(policy_probs)

    # Ensure response never exceeds available resources
    response = int(numpy.clip(response, 0, int(resources_left)))

    def map_to_response_time(x): return x * (-2) + 1

    mu, sigma = int(map_to_response_time(confidence) * 10), 3

    rt_hold = numpy.random.normal(mu, sigma, 1)[0]
    rt_release = rt_hold + numpy.random.normal(mu, 1, 1)[0]

    rt_hold = numpy.clip(rt_hold, 0, response_timeout / 1000.0)
    rt_release = numpy.clip(rt_release, 0, response_timeout / 1000.0)

    return response, confidence, rt_hold, rt_release


def provide_ac_agent_response(
    _resources,
    resources_left,
    session_no,
    sequence_no,
    trial_no
):
    """Generate A2C agent response using trained Actor policy.

    Main interface for obtaining agent responses.  Loads the trained Keras
    Actor model from disk, retrieves the current game state (severity,
    resources, trial), computes π(a|s) via a forward pass, and generates
    a response with confidence and timing metadata.

    Parameters
    ----------
    _resources : float
        Total resources available in session (informational, not used for indexing).
    resources_left : float or int
        Remaining resources available for allocation.
    session_no : int
        Session identifier (0-indexed) for multi-session experiments.
    sequence_no : int
        Sequence within session (0-indexed).
    trial_no : int
        Trial within sequence (0-indexed).

    Returns
    -------
    confidence : float
        Decision confidence metric (0-1 range) from meta-cognitive processing.
    response : int
        Resource allocation decision (0 to resources_left).
    rt_hold : float
        Reaction time in seconds (stimulus to button press).
    rt_release : float
        Total response duration in seconds (stimulus to button release).
    movement : list
        Placeholder for movement data (currently empty list).

    Raises
    ------
    AssertionError
        If ``first_severity`` module variable not initialised by caller.
    FileNotFoundError
        If the Actor model file is not found in ``INPUTS_PATH``.
    RuntimeError
        If the model file is corrupted and cannot be loaded.

    Notes
    -----
    - Requires A2C pre-training via: ``python3 -m ml.pes_a2c.ext.train_a2c``
    - Requires ``first_severity`` initialised before calling this function.
    - State is normalised to [0, 1]³ before the forward pass.
    """

    assert first_severity is not None, \
        "The 'first_severity' module-global variable needs to be set by caller before calling this function"

    # Load and validate Actor model
    model_path = os.path.join(INPUTS_PATH, AC_MODEL_ACTOR_FILE)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"\nFATAL ERROR: Actor model file not found at {model_path}\n"
            f"Please train the A2C Agent first by running: python3 -m ml.pes_a2c.ext.train_a2c\n"
        )

    try:
        actor_model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise RuntimeError(
            f"\nFATAL ERROR: Failed to load Actor model!\n"
            f"Error: {str(e)}\n"
            f"File may be corrupted. Please retrain by running: python3 -m ml.pes_a2c.ext.train_a2c\n"
        ) from e

    if VERBOSE:
        print("Reading preloaded Actor model for A2C Agent")

    resources_remaining = tf.Variable(resources_left, dtype=tf.float32)

    if VERBOSE:
        print('Resources remaining...')
        print(int(resources_remaining.numpy()) if hasattr(resources_remaining, 'numpy') else resources_remaining)
        print()

    SequenceLengthsCsv = os.path.join(INPUTS_PATH, SEQ_LENGTHS_FILE)
    sequence_length = numpy.loadtxt(SequenceLengthsCsv, delimiter=',')
    sevs = convert_globalseq_to_seqs(sequence_length, first_severity)

    sever = sevs[session_no * NUM_SEQUENCES + sequence_no][trial_no]
    city_number = trial_no

    # Convert to integers for state construction
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
    # Mirrors the (env.available_resources_states - 1) etc. used in
    # train_a2c.py / pandemic.py so that any CONFIG change propagates here.
    max_res = AVAILABLE_RESOURCES_PER_SEQUENCE - 9
    max_tri = NUM_MAX_TRIALS
    max_sev = MAX_SEVERITY

    # Clamp to valid ranges
    resources_idx = max(0, min(resources_idx, max_res))
    city_idx = max(0, min(city_idx, max_tri))
    sever_idx = max(0, min(sever_idx, max_sev))

    # Forward pass through Actor to get π(a|s)
    state_norm = normalize_state(
        [resources_idx, city_idx, sever_idx], max_res, max_tri, max_sev
    )
    policy_probs = actor_model(state_norm[numpy.newaxis, :], training=False)[0].numpy()

    if VERBOSE:
        print(f"State indices - Resources: {resources_idx}, City: {city_idx}, Severity: {sever_idx}")
        print(f"Policy probs for this state: {policy_probs}")

    # Calculate the response and confidence using meta-cognitive entropy-based evaluation
    resp, confidence, rt_hold, rt_release = ac_agent_meta_cognitive(
        policy_probs, resources_left, RESPONSE_TIMEOUT)

    # Final validation: ensure response never exceeds available resources
    resp = int(numpy.clip(resp, 0, int(resources_left)))

    if VERBOSE:
        print(f"A2C Agent Response: {resp}, Confidence: {confidence}")
        print(f"Resources available: {int(resources_left)}, Response clamped to: {resp}")

    movement: list = []

    return confidence, resp, rt_hold, rt_release, movement
