"""
pes_rdqn — Configuration file.

Centralises every tuneable parameter for the Pandemic Experiment Scenario
(Recurrent Deep Q-Network).  Values here apply to the experiment loop (__main__.py),
RDQN training (train_rdqn.py), and Bayesian optimisation (optimize_rdqn.py).

Experiment Structure::

    Experimento (1)
    ├─ Bloque (8)
    │    ├─ Secuencia / Mapa (8)
    │    │    ├─ Trial / Ciudad (3~10)
    │    │    │    └─ Decisión de Recursos (0-10)

Key Differences vs pes_dql (Double Q-Learning):
    - PLAYER_TYPE = 'RDQN_AGENT'
    - RDQN-specific hyperparameters: hidden units, replay buffer, target sync,
      batch size, gradient clipping, PBRS penalty, etc.
    - Uses neural network approximation instead of tabular Q-table
    - MAX_SEVERITY = 9 (same as all packages)
    - SEED = 42 (fixed for reproducible RDQN training & optimisation)
    - AVAILABLE_RESOURCES_PER_SEQUENCE = 39 (same as pes_dql)

Sections:
    - Resource Allocation Settings  (budget, initial cities)
    - Data Files & Initialization   (CSV paths, random flags)
    - Decision Aggregation           (mean / median / mode selector)
    - Value Ranges & Limits          (severity, allocation bounds)
    - Experiment Structure           (blocks, sequences, trials)
    - Output & Logging               (file prefixes)
    - Player & Agent Settings        (RDQN_AGENT selection)
    - Pandemic Dynamics              (α / β multipliers)
    - UI & Interaction               (trust scale, fixed sequences)
    - Reproducibility                (SEED = 42)
    - RDQN Hyperparameters            (network, replay buffer, target sync)
    - Runtime & Persistence          (verbose, save flags)
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
AGGREGATION_METHOD = {1: 'confidence_weighted_median',    # Robust to outliers
                      2: 'confidence_weighted_mean',      # Standard weighted average
                      3: 'confidence_weighted_mode'       # Most common value (experimental)
                      }[2]    # <-- SELECT: Change index (1/2/3) to choose method

# ==================== VALUE RANGES & LIMITS ====================
MAX_ALLOCATABLE_RESOURCES = 10  # Maximum resources allocatable per trial (Suggested: 10)
MAX_SEVERITY = 9                # Maximum possible severity value for cities (Suggested: 9)
MIN_ALLOCATABLE_RESOURCES = 0   # Minimum resources allocatable per trial (Suggested: 0)
MAX_INIT_SEVERITY = 5          # Maximum initial city severity (Suggested: 5)
MIN_INIT_SEVERITY = 2          # Minimum initial city severity (Suggested: 2)
MAX_INIT_RESOURCES = 6          # Maximum initial resource allocation (Suggested: 6)
MIN_INIT_RESOURCES = 3          # Minimum initial resource allocation (Suggested: 3)

# ==================== EXPERIMENT STRUCTURE ====================
NUM_BLOCKS = 8                  # Number of experimental blocks (Suggested: 8, Range: 6-8)
NUM_SEQUENCES = 8               # Number of sequences (maps) per block (Suggested: 8, Range: 8-12)
NUM_MIN_TRIALS = 3              # Minimum trials per sequence (Suggested: 3)
NUM_MAX_TRIALS = 10             # Maximum trials per sequence (Suggested: 10)
TOTAL_NUM_TRIALS_IN_BLOCK = 45  # Exact sum of trials across all sequences in a block
# Ensures consistent block duration and break scheduling
NUM_ATTEMPTS_TO_ASSIGN_SEQ = 8  # Retry attempts when assigning sequences to satisfy constraints

# ==================== OUTPUT & LOGGING ====================
OUTPUT_FILE_PREFIX = 'PES_RDQN_'    # Prefix for all output filenames

# ==================== PLAYER & AGENT SETTINGS ====================
PLAYER_TYPE = {  # Decision maker type - SELECT ONE
    1: 'RL_AGENT',  # Q-Learning Reinforcement Learning agent
    2: 'RDQN_AGENT'  # Recurrent Deep Q-Network Reinforcement Learning agent
}[2]

STARTING_BLOCK_INDEX = 0   # Resume from block index (0 = start from beginning)
STARTING_SEQ_INDEX = 0   # Resume from sequence index (0 = start from beginning)

# ==================== PANDEMIC DYNAMICS ====================
PANDEMIC_PARAMETER = 0.4   # Alpha (α) parameter controlling disease dynamics
# α = RESPONSE_MULTIPLIER (resource effectiveness)
# β = SEVERITY_MULTIPLIER = 1 + α (disease propagation)
# Formula: new_severity = β*initial - α*resources

# ==================== UI & INTERACTION ====================
TRUST_MAX = 100            # Maximum scale value for confidence slider (upgraded from 4 to 100)
USE_FIXED_BLOCK_SEQUENCES = True  # Load sequence trial lengths from CSV file (vs. random)

# ==================== RDQN HYPERPARAMETERS ====================
# RDQN = Recurrent DQN: an LSTM trunk consumes a sliding window of the last
# RDQN_HISTORY_LEN normalised states; a dense head outputs Q-values.
# Defaults below are starting points for Bayesian optimisation; tune via
# ``python -m pes_rdqn.ext.optimize_rdqn`` and write back into this file.
RDQN_HISTORY_LEN = 6                                # Sliding-window length fed to the LSTM
RDQN_LSTM_UNITS = 64                                # LSTM hidden-state width
RDQN_HIDDEN_UNITS = [64]                            # Dense-head widths after the LSTM trunk
RDQN_LEARNING_RATE = 0.0015083436603048935          # Adam learning rate for the Q-network
RDQN_BATCH_SIZE = 128                               # Mini-batch size sampled from replay buffer
RDQN_REPLAY_BUFFER_SIZE = 20_000                    # Maximum transitions stored in the replay buffer
RDQN_TARGET_SYNC_FREQ = 1_000                       # Steps between hard copies of online → target network
RDQN_DISCOUNT = 0.9634244388615337                  # Discount factor (γ) for TD targets
RDQN_EPSILON_INITIAL = 0.9627337198502147           # Initial exploration rate (ε-greedy)
RDQN_EPSILON_MIN = 0.06914686776995618              # Minimum exploration rate after decay
RDQN_EPISODES = 175_000                             # Default number of training episodes
RDQN_MAX_GRAD_NORM = 3.9528553802652735             # Global gradient norm clipping threshold
RDQN_PENALTY_COEFF = 0.02258267089059471            # PBRS reward shaping coefficient (β)
RDQN_WARMUP_RATIO = 0.2779025551585237              # Fraction of episodes with ε = ε₀ (pure exploration)
RDQN_TARGET_RATIO = 0.6290206520891799              # Fraction at which ε reaches ε_min via exponential decay
RDQN_LEARNING_STARTS_FRAC = 0.16154748671160965     # Fraction of buffer_size that must accumulate before training starts
RDQN_MODEL_FILE = 'rdqn_model.keras'                 # Filename for the saved Q-network weights

# ==================== REPRODUCIBILITY ====================
SEED = 42                  # Random seed for RDQN training reproducibility

# ==================== RUNTIME & PERSISTENCE ====================
VERBOSE = True             # Enable detailed console logging and initialization messages
SAVE_RESULTS = True        # Save experiment results to JSON/TXT files after each sequence
