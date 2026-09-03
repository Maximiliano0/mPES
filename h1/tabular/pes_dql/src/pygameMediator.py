"""pes_dql — RL Agent Response Handler and Confidence Estimator.

Bridges the experiment loop (__main__.py) with the trained Q-Learning model,
handling agent decision-making, response timing, and entropy-based confidence
calculations.

Key Functions:
    - rl_agent_meta_cognitive: Entropy-based confidence over Q-value distribution.
    - provide_rl_agent_response: Main interface — loads Q-table (q.npy),
      selects greedy action, computes confidence, returns formatted response.

Q-table layout (loaded from ``q.npy``):
    Shape: (31, 11, 10, 11)
        - 31 resource-left states   (0 … 30)
        - 11 trial-number states    (0 … 10)
        - 10 severity states        (0 … MAX_SEVERITY=9)
        - 11 actions                (allocate 0 … 10 resources)

Module Dependencies:
    External: numpy, tensorflow, os
    Internal: convert_globalseq_to_seqs, rl_agent_meta_cognitive, CONFIG constants

Global Variables:
    first_severity: Initial severity array set by caller before first use.
    number_of_trials: Total trial count for the experiment.
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
from .. import (
    ANSI,
    INPUTS_PATH,
    NUM_SEQUENCES,
    SEQ_LENGTHS_FILE,
    VERBOSE,
)
from .. ext.tools import convert_globalseq_to_seqs
from .. ext.pandemic import rl_agent_meta_cognitive

##########################################################
## Variables requiring initialisation before module use ##
##########################################################
first_severity        = None
number_of_trials      = None

#################################
## Module-Specific constants   ##
#################################
FONT              = 'ubuntumono'   # previously: Arial
BACKGROUND_COLOUR = ANSI.GRAY
RESPONSE_TIMEOUT  = 5000  # in milliseconds


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
    - Requires Q-table pre-training via: python3 -m tabular.pes_dql.ext.train_rl
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
            f"Please train the RL-Agent first by running: python3 -m tabular.pes_dql.ext.train_rl\n"
        )

    if not os.path.isfile(rewards_file):
        raise FileNotFoundError(
            f"\nFATAL ERROR: Rewards file not found at {rewards_file}\n"
            f"Please train the RL-Agent first by running: python3 -m tabular.pes_dql.ext.train_rl\n"
        )

    try:
        Q = numpy.load(q_file)
        _rewards = numpy.load(rewards_file)
    except Exception as e:
        raise RuntimeError(
            f"\nFATAL ERROR: Failed to load training files!\n"
            f"Error: {str(e)}\n"
            f"Files may be corrupted. Please retrain by running: python3 -m tabular.pes_dql.ext.train_rl\n"
        ) from e

    if VERBOSE:
        print( "Reading preloaded Q-Table for RL-Agent" )

    resources_remaining = tf.Variable(resources_left, dtype=tf.float32)

    if VERBOSE:
        print( 'Resources remaining...' )
        print( int(resources_remaining.numpy()) if hasattr(resources_remaining, 'numpy') else resources_remaining )
        print()

    SequenceLengthsCsv = os.path.join( INPUTS_PATH, SEQ_LENGTHS_FILE )
    sequence_length = numpy.loadtxt( SequenceLengthsCsv , delimiter=',')
    sevs = convert_globalseq_to_seqs(sequence_length, first_severity)

    sever = sevs[ session_no * NUM_SEQUENCES + sequence_no ][ trial_no ]
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
    resources_idx = max(0, min(resources_idx, Q.shape[0]-1))
    city_idx = max(0, min(city_idx, Q.shape[1]-1))
    sever_idx = max(0, min(sever_idx, Q.shape[2]-1))

    if VERBOSE:
        print(f"State indices - Resources: {resources_idx}, City: {city_idx}, Severity: {sever_idx}")
        print(f"Q-Table shape: {Q.shape}")
        print(f"Q values for this state: {Q[resources_idx, city_idx, sever_idx]}")

    # Calculate the response and confidence using meta-cognitive entropy-based Q-table evaluation.
    resp, confidence, rt_hold, rt_release = rl_agent_meta_cognitive(Q[resources_idx, city_idx, sever_idx], resources_left, RESPONSE_TIMEOUT)

    # Final validation: ensure response never exceeds available resources
    resp = int(numpy.clip(resp, 0, int(resources_left)))

    if VERBOSE:
        print(f"RL-Agent Response: {resp}, Confidence: {confidence}")
        print(f"Resources available: {int(resources_left)}, Response clamped to: {resp}")

    movement = []

    return confidence, resp, rt_hold, rt_release, movement
