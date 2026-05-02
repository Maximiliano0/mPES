"""RL Agent Game Display Mediator and Response Handler.

This module bridges the Pygame game interface with the trained RL agent (Q-Learning model),
handling agent decision-making, response timing, and confidence calculations. It manages
the communication between the game display and the RL agent by processing game states
and generating appropriately timed and confident responses.

Key Functions:
    - rl_agent_meta_cognitive: Meta-cognitive decision making with entropy-based confidence
    - provide_rl_agent_response: Main interface returning agent response and timing

Module Dependencies:
    External: numpy, tensorflow, os
    Internal: log_utils, convert_globalseq_to_seqs, CONFIG constants

Global Variables:
    first_severity: Initial severity array loaded by caller
    number_of_trials: Total trial count for experiment

Author: PES Development Team
Version: 1.0
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
from .. import ANSI, INPUTS_PATH, NUM_SEQUENCES, SEQ_LENGTHS_FILE, VERBOSE
from .. ext.tools import convert_globalseq_to_seqs
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


def rl_agent_meta_cognitive(options, resources_left, response_timeout):
    """Generate RL agent response with meta-cognitive confidence and timing simulation.

    Implements a meta-cognitive decision-making process where the agent:
    1. Evaluates confidence based on entropy of Q-table options
    2. Filters infeasible options (exceeding available resources)
    3. Generates decision entropy from feasible options
    4. Maps confidence to response timing (reaction time, release time)

    The meta-cognitive mechanism assumes that higher uncertainty (entropy) in the
    decision space correlates with longer response time to deliberate before committing.

    Parameters
    ----------
    options : array_like
        1-D array of Q-values from the Q-table state [resources, trial, severity]
        Typically shape (11,) representing resource allocation probabilities
    resources_left : float or int
        Available resources that constrain feasible actions
    response_timeout : int
        Maximum response time in milliseconds (e.g., 5000 for 5 seconds)

    Returns
    -------
    response : int
        Selected resource allocation (argmax of feasible options)
        Guaranteed to not exceed resources_left
    confidence : float
        Meta-cognitive confidence metric (0-1 range)
        - Based on entropy of feasible option distribution
        - 0 = uniform distribution (low confidence)
        - 1 = peaked distribution (high confidence)
    rt_hold : float
        Reaction time from stimulus to response button press (seconds)
        Sampled from Gaussian scaled by confidence
    rt_release : float
        Total time from stimulus to releasing response button (seconds)
        Always >= rt_hold (represents commitment duration)

    Notes
    -----
    - Confidence inverted from entropy: (1/(m_entropy - M_entropy)) * (H - M_entropy)
    - High confidence → faster response (lower RT)
    - Low confidence → longer deliberation (higher RT)
    - All response times clipped to [0, response_timeout/1000]
    - Response always clipped to feasible range [0, resources_left]

    Examples
    --------
    >>> q_options = np.array([0.1, 0.2, 0.5, 0.15, 0.05, 0, 0, 0, 0, 0, 0])
    >>> response, conf, rt_h, rt_r = rl_agent_meta_cognitive(
    ...     q_options, resources_left=15, response_timeout=5000)
    >>> print(f"Response: {response}, Confidence: {conf:.2f}, "
    ...       f"Hold time: {rt_h:.3f}s, Release time: {rt_r:.3f}s")
    # Output: Response: 2, Confidence: 0.68, Hold time: 0.234s, Release time: 0.567s
    """

    def entropy_from_pdf(pdf):
        """Compute Shannon entropy (base-2) of a shifted-positive PDF.

        Parameters
        ----------
        pdf : numpy.ndarray
            Raw score vector (may contain non-positive values).

        Returns
        -------
        float
            Shannon entropy in bits.
        """
        pdf = pdf + numpy.abs(numpy.min(pdf))
        p = pdf / numpy.sum(pdf)  # log(0)
        p[p == 0] += 0.000001
        print(p)
        H = -numpy.dot(p, numpy.log2(p))
        return H

  # Min entropy from a univalue distribution (0)
    m_entropy = numpy.zeros((11,),)
    m_entropy[0] = 1

  # Max entropy from a uniform distribution (3.55....)
    M_entropy = numpy.ones((11,),)

  # Options are the available choices from the Q Table
    log_utils.tee('Options:', options)

    _entrp1 = entropy_from_pdf(options)

    o = [i for i in range(len(options))]
    o = numpy.asarray(o, dtype=numpy.float32)

    # Set options that are not feasible (greater than resources left)
    # to a very negative sentinel so they never win argmax. A small positive
    # value would dominate the typically-negative learned Q-values and force
    # the policy to pick an infeasible index (silently clamped by the env).
    options[o > resources_left] = -1e9

    log_utils.tee('Agent Feasible Options:', options)

  # available resources, trial, severity
    dec_entropy = entropy_from_pdf(options)
    M_entropy = entropy_from_pdf(M_entropy)
    m_entropy = entropy_from_pdf(m_entropy)

    confidence = (1. / (m_entropy - M_entropy)) * (dec_entropy - M_entropy)

    response = numpy.argmax(options)

    # Ensure response never exceeds available resources
    response = int(numpy.clip(response, 0, int(resources_left)))

    def map_to_response_time(x):
        """Map confidence to a response-time scale via linear transform."""
        return x * (-2) + 1

    mu, sigma = int(map_to_response_time(confidence) * 10), 3

    rt_hold = numpy.random.normal(mu, sigma, 1)[0]
    rt_release = rt_hold + numpy.random.normal(mu, 1, 1)[0]

    rt_hold = numpy.clip(rt_hold, 0, response_timeout / 1000.0)
    rt_release = numpy.clip(rt_release, 0, response_timeout / 1000.0)

    return response, confidence, rt_hold, rt_release


def provide_rl_agent_response(
    _resources,
    resources_left,
    session_no,
    sequence_no,
    trial_no
):
    """Generate RL agent response using trained Q-Learning policy.

    Main interface for obtaining agent responses. Loads the trained Q-table from disk,
    retrieves the current game state (severity, resources, trial), selects the
    appropriate Q-value, and generates a response with confidence and timing metadata.

    This function handles all file I/O, state indexing, Q-table lookup, and calls
    the meta-cognitive decision function to produce a realistic response.

    Parameters
    ----------
    resources : float
        Total resources available in session (informational, not used for indexing)
    resources_left : float or int
        Remaining resources available for allocation (Q-table first dimension)
    session_no : int
        Session identifier (0-indexed) for multi-session experiments
    sequence_no : int
        Sequence within session (0-indexed)
    trial_no : int
        Trial within sequence (0-indexed, Q-table second dimension)

    Returns
    -------
    confidence : float
        Decision confidence metric (0-1 range) from meta-cognitive processing
    response : int
        Resource allocation decision (0 to resources_left)
    rt_hold : float
        Reaction time in seconds (stimulus to button press)
    rt_release : float
        Total response duration in seconds (stimulus to button release)
    movement : list
        Placeholder for movement data (currently empty list)

    Raises
    ------
    AssertionError
        If first_severity module variable not initialized by caller
    FileNotFoundError
        If Q-table (q.npy) or rewards file (rewards.npy) not found in INPUTS_PATH
    RuntimeError
        If Q-table or rewards files are corrupted and cannot be loaded

    Notes
    -----
    - Requires Q-table pre-training via: python3 -m pes_ql.ext.train_rl
    - Requires first_severity initialized: call before using this function
    - Q-table dimensions: [resources (31) × trials (11) × severity (10) × actions (11)]
    - State indices automatically clamped to valid ranges
    - All Q-table values converted to integers for safe array indexing
    - Uses VERBOSE flag to enable debug output during execution

    Examples
    --------
    Initialize module variable before first call::

        pygameMediator.first_severity = severity_array  # shape (num_sequences,)

    Then use function in game loop::

        conf, resp, rth, rtr, mov = provide_rl_agent_response(
            resources=30, resources_left=15, session_no=0,
            sequence_no=2, trial_no=7)
        print(f"Agent allocated {resp} resources with {conf:.2f} confidence")
        print(f"Response timing: {rth:.3f}s to press, {rtr:.3f}s to release")
    """

    assert first_severity is not None, \
        "The 'first_severity' module-global variable needs to be set by caller before calling this function"

    # Load and validate Q-Table
    q_file = os.path.join(INPUTS_PATH, 'q.npy')
    rewards_file = os.path.join(INPUTS_PATH, 'rewards.npy')

    if not os.path.isfile(q_file):
        raise FileNotFoundError(
            f"\nFATAL ERROR: Q-Table file not found at {q_file}\n"
            f"Please train the RL-Agent first by running: python3 -m pes_ql.ext.train_rl\n"
        )

    if not os.path.isfile(rewards_file):
        raise FileNotFoundError(
            f"\nFATAL ERROR: Rewards file not found at {rewards_file}\n"
            f"Please train the RL-Agent first by running: python3 -m pes_ql.ext.train_rl\n"
        )

    try:
        Q = numpy.load(q_file)
        _rewards = numpy.load(rewards_file)
    except Exception as e:
        raise RuntimeError(
            f"\nFATAL ERROR: Failed to load training files!\n"
            f"Error: {str(e)}\n"
            f"Files may be corrupted. Please retrain by running: python3 -m pes_ql.ext.train_rl\n"
        ) from e

    if VERBOSE:
        print("Reading preloaded Q-Table for RL-Agent")

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

    # Convert to integers for array indexing (handle numpy types and tensors)
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

    # Clamp indices to valid ranges
    resources_idx = max(0, min(resources_idx, Q.shape[0] - 1))
    city_idx = max(0, min(city_idx, Q.shape[1] - 1))
    sever_idx = max(0, min(sever_idx, Q.shape[2] - 1))

    if VERBOSE:
        print(f"State indices - Resources: {resources_idx}, City: {city_idx}, Severity: {sever_idx}")
        print(f"Q-Table shape: {Q.shape}")
        print(f"Q values for this state: {Q[resources_idx, city_idx, sever_idx]}")

    # Calculate the response and confidence using meta-cognitive entropy-based Q-table evaluation.
    resp, confidence, rt_hold, rt_release = rl_agent_meta_cognitive(
        Q[resources_idx, city_idx, sever_idx], resources_left, RESPONSE_TIMEOUT)

    # Final validation: ensure response never exceeds available resources
    resp = int(numpy.clip(resp, 0, int(resources_left)))

    if VERBOSE:
        print(f"RL-Agent Response: {resp}, Confidence: {confidence}")
        print(f"Resources available: {int(resources_left)}, Response clamped to: {resp}")

    movement = []

    return confidence, resp, rt_hold, rt_release, movement
