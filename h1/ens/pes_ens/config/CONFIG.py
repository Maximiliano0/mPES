"""
Configuration file for the pes_ens experiment.

Centralises all tunable parameters for the *Ensemble* agent that fuses the
trained policies of pes_dqn, pes_a2c, pes_rdqn and pes_trf (A2C disabled
by default) via **confidence-weighted soft voting** of action-probability
distributions.

Sections
--------
- Resource Allocation Settings  (budget, initial cities)
- Data Files & Initialization   (CSV paths, random flags)
- Decision Aggregation           (mean / median / mode selector)
- Value Ranges & Limits          (severity, allocation bounds)
- Experiment Structure           (blocks, sequences, trials)
- Output & Logging               (file prefixes)
- Player & Agent Settings        (ENS_AGENT)
- Pandemic Dynamics              (alpha / beta multipliers)
- UI & Interaction               (trust scale, fixed sequences)
- Ensemble Settings              (member model paths, voting weights,
                                  history-window lengths for recurrent
                                  members)
- Reproducibility                (SEED for deterministic tie-breaking)
- Runtime & Persistence          (verbose, save flags)

Notes
-----
- pes_ens has **no training or optimisation phase**.  Every enabled
  member model must already exist as a ``.keras`` artefact in the
  canonical ``inputs/`` directory of its producing package.
- ``ENS_MEMBER_MODELS`` paths are resolved at runtime to absolute
  filesystem paths.  Members marked ``enabled=False`` are skipped.
- Voting strategy is **confidence-weighted soft voting** (see
  ``EnsembleAgent.predict`` in ``ext/ensemble_model.py``):  each
  member's raw output is converted to an action-probability
  distribution (``softmax(Q / ENS_SOFTMAX_TEMPERATURE)`` for Q-network
  members, renormalised policy for the A2C actor), infeasible actions
  (allocation > resources_left) are masked and each distribution is
  renormalised; the distributions are then mixed with the dynamic
  weight ``w_norm * (0.1 + confidence)``, where ``w_norm`` is
  ``ENS_MEMBER_MODELS[i]['weight']`` renormalised over the enabled
  members and ``confidence = 1 - H(p) / log2(11)``.  The mixture's
  action-0 mass is multiplied by ``0.3`` when resources remain, the
  result is renormalised and blended with the Gaussian severity prior
  (``ENS_SEVERITY_PRIOR_WEIGHT``/``ENS_SEVERITY_PRIOR_SIGMA``); if the
  raw severity is ``>= 6`` and the ``argmax`` falls below
  ``floor = severity // 2`` (and ``floor`` is feasible), the decision
  is forced to ``floor``.  Otherwise ``argmax`` selects the final
  action.
"""

# ==================== RESOURCE ALLOCATION SETTINGS ====================
AVAILABLE_RESOURCES_PER_SEQUENCE = 39   # Total resource budget per sequence
ALLOCATABLE_RESOURCES_PER_SEQUENCE = AVAILABLE_RESOURCES_PER_SEQUENCE - 9

INIT_NO_OF_CITIES = 2   # Cities visible at the start of each sequence

# ==================== DATA FILES & INITIALIZATION ====================
INITIAL_SEVERITY_FILE = 'initial_severity.csv'
SEQ_LENGTHS_FILE = 'sequence_lengths.csv'
RANDOM_INITIAL_SEVERITY = False
SAVE_INITIAL_SEVERITY_TO_FILE = False

# ==================== DECISION AGGREGATION ====================
AGGREGATION_METHOD = {1: 'confidence_weighted_median',
                      2: 'confidence_weighted_mean',
                      3: 'confidence_weighted_mode'
                      }[2]

# ==================== VALUE RANGES & LIMITS ====================
MAX_ALLOCATABLE_RESOURCES = 10
MAX_SEVERITY = 9
MIN_ALLOCATABLE_RESOURCES = 0
MAX_INIT_SEVERITY = 5
MIN_INIT_SEVERITY = 2
MAX_INIT_RESOURCES = 6
MIN_INIT_RESOURCES = 3

# ==================== EXPERIMENT STRUCTURE ====================
NUM_BLOCKS = 8
NUM_SEQUENCES = 8
NUM_MIN_TRIALS = 3
NUM_MAX_TRIALS = 10
TOTAL_NUM_TRIALS_IN_BLOCK = 45
NUM_ATTEMPTS_TO_ASSIGN_SEQ = 8

# ==================== OUTPUT & LOGGING ====================
OUTPUT_FILE_PREFIX = 'PES_ENS_'

# ==================== PLAYER & AGENT SETTINGS ====================
PLAYER_TYPE = 'ENS_AGENT'   # Ensemble of pes_dqn + pes_a2c + pes_rdqn + pes_trf

STARTING_BLOCK_INDEX = 0
STARTING_SEQ_INDEX = 0

# ==================== PANDEMIC DYNAMICS ====================
PANDEMIC_PARAMETER = 0.4

# ==================== UI & INTERACTION ====================
TRUST_MAX = 100
USE_FIXED_BLOCK_SEQUENCES = True

# ==================== REPRODUCIBILITY ====================
SEED = 42

# ==================== ENSEMBLE SETTINGS ====================
# Paths to the four pre-trained member models, expressed *relative to the
# pes_ens package directory* so they resolve to the canonical inputs/
# folder of each producing package.  ``role`` selects the inference
# adapter inside ``ext/ensemble_model.py``:
#
#   - ``q_dense``   : DQN-style dense Q-network (output Q-values, shape (11,))
#   - ``actor``     : A2C actor (output policy probabilities, shape (11,))
#   - ``q_recurrent``: RDQN/TRF recurrent Q-network (input window of past
#                     normalised states, output Q-values shape (11,))
#
# ``history_len`` is only used by recurrent members; ignore for others.
# ``weight`` is the relative voting weight (re-normalised internally).
# Set ``enabled=False`` to skip a member without removing its config.
ENS_MEMBER_MODELS = [
    {
        'name':        'dqn',
        'role':        'q_dense',
        'path':        '../../ml/pes_dqn/inputs/dqn_model.keras',
        'history_len': 1,
        'weight':      0.18,
        'enabled':     True,
    },
    {
        'name':        'a2c',
        'role':        'actor',
        'path':        '../../ml/pes_a2c/inputs/ac_actor.keras',
        'history_len': 1,
        'weight':      1.0,
        'enabled':     False,  # A2C actor is currently very weak; disable by default
    },
    {
        'name':        'rdqn',
        'role':        'q_recurrent',
        'path':        '../../ml/pes_rdqn/inputs/rdqn_model.keras',
        'history_len': 6,
        'weight':      0.9,
        'enabled':     True,
    },
    {
        'name':        'trf',
        'role':        'q_recurrent',
        'path':        '../../ml/pes_trf/inputs/trf_model.keras',
        'history_len': 6,
        'weight':      5.0,
        'enabled':     True,
    },
]

# Softmax temperature applied to Q-value members before mixing.
# Lower => sharper distribution, more confident; higher => smoother.
# 1.0 is the neutral value; 15.0 is used here, which strongly flattens
# the Q-member distributions (and therefore lowers their confidence).
ENS_SOFTMAX_TEMPERATURE = 15.0
# Severity-prior bias.  Mixes the ensemble distribution with a
# Gaussian prior centred at the current trial's raw severity:
#   prior[a] = exp(-(a - severity)^2 / (2 * sigma^2))   (masked to feasible a, renormalised)
#   final   = (1 - w) * ensemble + w * prior
# ``ENS_SEVERITY_PRIOR_WEIGHT`` in [0, 1]; 0.0 disables the prior.
# Smaller ``ENS_SEVERITY_PRIOR_SIGMA`` => sharper bias toward
# ``action == round(severity)``.
ENS_SEVERITY_PRIOR_WEIGHT = 0.17
ENS_SEVERITY_PRIOR_SIGMA = 3.0
# ==================== RUNTIME & PERSISTENCE ====================
VERBOSE = True
SAVE_RESULTS = True
