"""DQN Agent Game Display Mediator and Response Handler.

This module bridges the Pygame game interface with the trained DQN agent
(Deep Q-Network model), handling agent decision-making, response timing,
and confidence calculations. It manages the communication between the game
display and the DQN agent by processing game states and generating
appropriately timed and confident responses.

Key Functions:
    - dqn_agent_meta_cognitive: Meta-cognitive decision making with entropy-based confidence
    - provide_dqn_agent_response: Main interface returning agent response and timing

Module Dependencies:
    External: numpy, tensorflow, os
    Internal: log_utils, convert_globalseq_to_seqs, CONFIG constants, dqn_model

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
from .. import (ANSI, INPUTS_PATH,
                MAX_SEVERITY, NUM_MAX_TRIALS, NUM_SEQUENCES,
                SEQ_LENGTHS_FILE, VERBOSE)
from ..ext.tools import convert_globalseq_to_seqs
from ..ext.dqn_model import normalize_state

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




def provide_dqn_agent_response(
    _resources,
    resources_left,
    session_no,
    sequence_no,
    trial_no
):
    """Generate DQN agent response using trained Deep Q-Network policy.

    Main interface for obtaining agent responses. Loads the trained DQN model
    from disk, retrieves the current game state (severity, resources, trial),
    performs a forward pass through the network, and generates a response with
    confidence and timing metadata.

    This function handles all file I/O, state normalisation, model inference,
    and calls the meta-cognitive decision function to produce a realistic response.

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
        If first_severity module variable not initialized by caller.
    FileNotFoundError
        If DQN model (.keras) or rewards file (rewards.npy) not found in INPUTS_PATH.
    RuntimeError
        If DQN model or rewards files are corrupted and cannot be loaded.

    Notes
    -----
    - Requires DQN model pre-training via: python3 -m pes_dqn.ext.train_dqn
    - Requires first_severity initialized: call before using this function
    - State vector: [resources_left, trial_number, severity] (normalised to [0,1])
    - Uses VERBOSE flag to enable debug output during execution
    """

    assert first_severity is not None, \
        "The 'first_severity' module-global variable needs to be set by caller before calling this function"

    from ..config.CONFIG import DQN_MODEL_FILE

    # Load and validate DQN model
    model_file = os.path.join(INPUTS_PATH, DQN_MODEL_FILE)
    rewards_file = os.path.join(INPUTS_PATH, 'rewards.npy')

    if not os.path.isfile(model_file):
        raise FileNotFoundError(
            f"\nFATAL ERROR: DQN model file not found at {model_file}\n"
            f"Please train the DQN-Agent first by running: python3 -m pes_dqn.ext.train_dqn\n"
        )

    if not os.path.isfile(rewards_file):
        raise FileNotFoundError(
            f"\nFATAL ERROR: Rewards file not found at {rewards_file}\n"
            f"Please train the DQN-Agent first by running: python3 -m pes_dqn.ext.train_dqn\n"
        )

    try:
        dqn_model = tf.keras.models.load_model(model_file)
        _rewards = numpy.load(rewards_file)
    except Exception as e:
        raise RuntimeError(
            f"\nFATAL ERROR: Failed to load training files!\n"
            f"Error: {str(e)}\n"
            f"Files may be corrupted. Please retrain by running: python3 -m pes_dqn.ext.train_dqn\n"
        ) from e

    if VERBOSE:
        print("Reading preloaded DQN model for DQN-Agent")

    if VERBOSE:
        print('Resources remaining...')
        print(int(resources_left))
        print()

    SequenceLengthsCsv = os.path.join(INPUTS_PATH, SEQ_LENGTHS_FILE)
    sequence_length = numpy.loadtxt(SequenceLengthsCsv, delimiter=',')
    sevs = convert_globalseq_to_seqs(sequence_length, first_severity)

    sever = sevs[session_no * NUM_SEQUENCES + sequence_no][trial_no]
    city_number = trial_no

    # Build state for DQN forward pass
    try:
        resources_val = int(resources_left)
    except (ValueError, TypeError):
        resources_val = int(resources_left.numpy()) if hasattr(resources_left, 'numpy') else 0

    try:
        city_val = int(city_number)
    except (ValueError, TypeError):
        city_val = int(city_number.numpy()) if hasattr(city_number, 'numpy') else 0

    try:
        sever_val = int(sever)
    except (ValueError, TypeError):
        sever_val = int(sever.numpy()) if hasattr(sever, 'numpy') else 0

    if VERBOSE:
        print(f"State - Resources: {resources_val}, City: {city_val}, Severity: {sever_val}")

    # Normalise state and perform forward pass through DQN.
    # ``max_resources`` is read from a freshly built Pandemic env to avoid
    # silently desyncing if AVAILABLE_RESOURCES_PER_SEQUENCE or the offset
    # in Pandemic.__init__ ever changes.
    from ..ext.pandemic import Pandemic  # lazy import to avoid circular dependency
    _env_for_norm = Pandemic()
    max_res = _env_for_norm.max_resources
    raw_state = [resources_val, city_val, sever_val]
    state = normalize_state(raw_state, max_res,
                            NUM_MAX_TRIALS, MAX_SEVERITY)
    state_tensor = tf.expand_dims(state, axis=0)  # shape (1, 3)
    q_values = dqn_model(state_tensor, training=False).numpy().flatten()  # shape (11,)

    if VERBOSE:
        print(f"Q-values from DQN: {q_values}")

    # Calculate the response and confidence using meta-cognitive entropy-based evaluation
    from ..ext.pandemic import dqn_agent_meta_cognitive  # lazy import to avoid circular dependency
    resp, confidence, rt_hold, rt_release = dqn_agent_meta_cognitive(
        q_values, resources_left, RESPONSE_TIMEOUT)

    # Final validation: ensure response never exceeds available resources
    resp = int(numpy.clip(resp, 0, int(resources_left)))

    if VERBOSE:
        print(f"DQN-Agent Response: {resp}, Confidence: {confidence}")
        print(f"Resources available: {int(resources_left)}, Response clamped to: {resp}")

    movement = []

    return confidence, resp, rt_hold, rt_release, movement
