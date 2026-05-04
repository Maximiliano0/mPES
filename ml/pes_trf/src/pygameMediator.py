"""TRF Agent Game Display Mediator and Response Handler.

This module bridges the Pygame game interface with the trained TRF agent
(Transformer Deep Q-Network model), handling agent decision-making, response timing,
and confidence calculations. It manages the communication between the game
display and the TRF agent by processing game states and generating
appropriately timed and confident responses.

Key Functions:
    - trf_agent_meta_cognitive: Meta-cognitive decision making with entropy-based confidence
    - provide_trf_agent_response: Main interface returning agent response and timing

Module Dependencies:
    External: numpy, tensorflow, os
    Internal: log_utils, convert_globalseq_to_seqs, CONFIG constants, trf_model

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
from ..config.CONFIG import TRF_HISTORY_LEN
from ..ext.tools import convert_globalseq_to_seqs
from ..ext.transformer_model import normalize_state, HistoryDeque

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

# Per-(session, sequence) sliding-window cache so the Transformer trunk receives
# the actual trial history during inference. Reset implicitly when the
# (session, sequence) key changes.
_history_cache: dict = {}



def provide_trf_agent_response(
    _resources,
    resources_left,
    session_no,
    sequence_no,
    trial_no
):
    """Generate TRF agent response using trained Transformer Deep Q-Network policy.

    Main interface for obtaining agent responses. Loads the trained TRF model
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
        If TRF model (.keras) or rewards file (rewards.npy) not found in INPUTS_PATH.
    RuntimeError
        If TRF model or rewards files are corrupted and cannot be loaded.

    Notes
    -----
    - Requires TRF model pre-training via: python3 -m ml.pes_trf.ext.train_transformer
    - Requires first_severity initialized: call before using this function
    - State vector: [resources_left, trial_number, severity] (normalised to [0,1])
    - Uses VERBOSE flag to enable debug output during execution
    """

    assert first_severity is not None, \
        "The 'first_severity' module-global variable needs to be set by caller before calling this function"

    from ..config.CONFIG import TRF_MODEL_FILE

    # Load and validate TRF model
    model_file = os.path.join(INPUTS_PATH, TRF_MODEL_FILE)
    rewards_file = os.path.join(INPUTS_PATH, 'rewards.npy')

    if not os.path.isfile(model_file):
        raise FileNotFoundError(
            f"\nFATAL ERROR: TRF model file not found at {model_file}\n"
            f"Please train the TRF-Agent first by running: python3 -m ml.pes_trf.ext.train_transformer\n"
        )

    if not os.path.isfile(rewards_file):
        raise FileNotFoundError(
            f"\nFATAL ERROR: Rewards file not found at {rewards_file}\n"
            f"Please train the TRF-Agent first by running: python3 -m ml.pes_trf.ext.train_transformer\n"
        )

    try:
        # ``safe_mode=False`` is required because ``build_q_network`` uses a
        # ``tf.keras.layers.Lambda`` (``lambda t: t[:, -1, :]``) to read out
        # the final transformer token. Keras refuses lambda deserialisation
        # by default (CWE-502 mitigation). The model is produced by our own
        # ``train_transformer.py`` pipeline (Colab or local), so trusting it
        # is safe; do NOT remove this flag for unknown / third-party files.
        trf_model = tf.keras.models.load_model(model_file, safe_mode=False)
        _rewards = numpy.load(rewards_file)
    except Exception as e:
        raise RuntimeError(
            f"\nFATAL ERROR: Failed to load training files!\n"
            f"Error: {str(e)}\n"
            f"Files may be corrupted. Please retrain by running: python3 -m ml.pes_trf.ext.train_transformer\n"
        ) from e

    if VERBOSE:
        print("Reading preloaded TRF model for TRF-Agent")

    if VERBOSE:
        print('Resources remaining...')
        print(int(resources_left))
        print()

    SequenceLengthsCsv = os.path.join(INPUTS_PATH, SEQ_LENGTHS_FILE)
    sequence_length = numpy.loadtxt(SequenceLengthsCsv, delimiter=',')
    sevs = convert_globalseq_to_seqs(sequence_length, first_severity)

    sever = sevs[session_no * NUM_SEQUENCES + sequence_no][trial_no]
    city_number = trial_no

    # Build state for TRF forward pass
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

    # Normalise state and perform forward pass through TRF.
    # ``max_resources`` is read from a freshly built Pandemic env to avoid
    # silently desyncing if AVAILABLE_RESOURCES_PER_SEQUENCE or the offset
    # in Pandemic.__init__ ever changes.
    from ..ext.pandemic import Pandemic  # lazy import to avoid circular dependency
    _env_for_norm = Pandemic()
    max_res = _env_for_norm.max_resources
    raw_state = [resources_val, city_val, sever_val]
    state = normalize_state(raw_state, max_res,
                            NUM_MAX_TRIALS, MAX_SEVERITY)

    # Maintain a sliding window of past normalised states for the Transformer.
    cache_key = (int(session_no), int(sequence_no))
    history = _history_cache.get(cache_key)
    if history is None:
        history = HistoryDeque(TRF_HISTORY_LEN, 3)
        _history_cache[cache_key] = history
    history.append_step(state)
    window = history.current_window()
    state_tensor = window[numpy.newaxis]  # shape (1, history_len, 3)
    q_values = trf_model(state_tensor, training=False).numpy().flatten()  # shape (11,)

    if VERBOSE:
        print(f"Q-values from TRF: {q_values}")

    # Calculate the response and confidence using meta-cognitive entropy-based evaluation
    from ..ext.pandemic import trf_agent_meta_cognitive  # lazy import to avoid circular dependency
    resp, confidence, rt_hold, rt_release = trf_agent_meta_cognitive(
        q_values, resources_left, RESPONSE_TIMEOUT)

    # Final validation: ensure response never exceeds available resources
    resp = int(numpy.clip(resp, 0, int(resources_left)))

    if VERBOSE:
        print(f"TRF-Agent Response: {resp}, Confidence: {confidence}")
        print(f"Resources available: {int(resources_left)}, Response clamped to: {resp}")

    movement = []

    return confidence, resp, rt_hold, rt_release, movement
