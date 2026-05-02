"""
Configuration file for the pes_base experiment.

Centralises all tunable experiment parameters in one place.
Values are imported by pes_base/__init__.py and re-exported at package level.

Sections
--------
- Resource Allocation Settings  (budget, initial cities)
- Data Files & Initialization   (CSV paths, random flags)
- Decision Aggregation           (mean / median / mode selector)
- Value Ranges & Limits          (severity, allocation bounds)
- Experiment Structure           (blocks, sequences, trials)
- Output & Logging               (file prefixes)
- Player & Agent Settings        (RL_AGENT selection)
- Pandemic Dynamics              (α / β multipliers)
- UI & Interaction               (trust scale, fixed sequences)
- Runtime & Persistence          (verbose, save flags)
- Reproducibility                (SEED = 42 for Q-Learning training)

Experiment Structure
--------------------
::

    Experimento (1)
    ├─ Bloque (8)
    │    ├─ Secuencia / Mapa (8)
    │    │    ├─ Trial / Ciudad (3~10)
    │    │    │    └─ Decisión de Recursos (0-10)

Note
----
All constants here are imported via ``from .config import CONFIG`` in ``__init__.py``.
"""

# ==================== RESOURCE ALLOCATION SETTINGS ====================
AVAILABLE_RESOURCES_PER_SEQUENCE = 39   # Total resource budget allocatable across all trials in a sequence

INIT_NO_OF_CITIES = 2   # Number of cities visible at the start of each sequence

# ==================== DATA FILES & INITIALIZATION ====================
INITIAL_SEVERITY_FILE = 'initial_severity.csv'   # CSV file containing initial severity values for cities
SEQ_LENGTHS_FILE = 'sequence_lengths.csv'       # CSV file specifying trial count for each sequence
RANDOM_INITIAL_SEVERITY = False                 # Generate random severities instead of loading from file
SAVE_INITIAL_SEVERITY_TO_FILE = False           # Save generated severities to CSV for reproducibility

# ==================== DECISION AGGREGATION ====================
# Method for combining resource allocation decisions from multiple participants
AGGREGATION_METHOD = {  1: 'confidence_weighted_median',    # Robust to outliers
                        2: 'confidence_weighted_mean',      # Standard weighted average
                        3: 'confidence_weighted_mode'       # Most common value (experimental)
                     }[ 2 ]    # <-- SELECT: Change index (1/2/3) to choose method

# ==================== VALUE RANGES & LIMITS ====================
MAX_ALLOCATABLE_RESOURCES = 10  # Maximum resources allocatable per trial (Suggested: 10)
MAX_SEVERITY = 9                # Maximum possible severity value for cities (Suggested: 9)
MIN_ALLOCATABLE_RESOURCES = 0   # Minimum resources allocatable per trial (Suggested: 0)
MAX_INIT_SEVERITY  = 5          # Maximum initial city severity (Suggested: 5)
MIN_INIT_SEVERITY  = 2          # Minimum initial city severity (Suggested: 2)
MAX_INIT_RESOURCES = 6          # Maximum initial resource allocation (Suggested: 6)
MIN_INIT_RESOURCES = 3          # Minimum initial resource allocation (Suggested: 3)

# ==================== EXPERIMENT STRUCTURE ====================
NUM_BLOCKS = 8                  # Number of experimental blocks (Suggested: 8, Range: 6-8)
NUM_SEQUENCES = 8               # Number of sequences (maps) per block (Suggested: 8, Range: 8-12)
NUM_MIN_TRIALS = 3              # Minimum trials per sequence (Suggested: 3)
NUM_MAX_TRIALS = 10             # Maximum trials per sequence (Suggested: 10)
TOTAL_NUM_TRIALS_IN_BLOCK  = 45 # Exact sum of trials across all sequences in a block
                                 # Ensures consistent block duration and break scheduling
NUM_ATTEMPTS_TO_ASSIGN_SEQ = 8  # Retry attempts when assigning sequences to satisfy constraints

# ==================== OUTPUT & LOGGING ====================
OUTPUT_FILE_PREFIX = 'PES_BASE_'  # Prefix for all output filenames

# ==================== PLAYER & AGENT SETTINGS ====================
PLAYER_TYPE = { # Decision maker type - SELECT ONE
    1: 'RL_AGENT'  # Q-Learning Reinforcement Learning agent
    }[1]

STARTING_BLOCK_INDEX = 0   # Resume from block index (0 = start from beginning)
STARTING_SEQ_INDEX   = 0   # Resume from sequence index (0 = start from beginning)

# ==================== PANDEMIC DYNAMICS ====================
PANDEMIC_PARAMETER = 0.4   # Alpha (α) parameter controlling disease dynamics
                           # α = RESPONSE_MULTIPLIER (resource effectiveness)
                           # β = SEVERITY_MULTIPLIER = 1 + α (disease propagation)
                           # Formula: new_severity = β*initial - α*resources

# ==================== UI & INTERACTION ====================
TRUST_MAX = 100            # Maximum scale value for confidence slider (upgraded from 4 to 100)
USE_FIXED_BLOCK_SEQUENCES = True  # Load sequence trial lengths from CSV file (vs. random)

# ==================== RUNTIME & PERSISTENCE ====================
VERBOSE = True             # Enable detailed console logging and initialization messages
SAVE_RESULTS = True        # Save experiment results to JSON/TXT files after each sequence

# ==================== REPRODUCIBILITY ====================
SEED = 42                  # Random seed for Q-Learning training reproducibility
