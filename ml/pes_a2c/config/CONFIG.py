"""
Configuration file for the pes_a2c experiment.

Centralises all tunable experiment parameters in one place.
Values are imported by pes_a2c/__init__.py and re-exported at package level.

Sections
--------
- Resource Allocation Settings  (budget, initial cities)
- Data Files & Initialization   (CSV paths, random flags)
- Decision Aggregation           (mean / median / mode selector)
- Value Ranges & Limits          (severity, allocation bounds)
- Experiment Structure           (blocks, sequences, trials)
- Output & Logging               (file prefixes)
- Player & Agent Settings        (RL_AGENT / A2C_AGENT selector)
- Pandemic Dynamics              (α / β multipliers)
- UI & Interaction               (trust scale, fixed sequences)
- Advantage Actor-Critic (A2C)   (architecture & training knobs)
- Reproducibility                (SEED = 42 for A2C training)
- Runtime & Persistence          (verbose, save flags)

Key differences from pes_ql/config/CONFIG.py
----------------------------------------------------
- Replaces tabular Q-Learning with Advantage Actor-Critic (A2C)
- Separate Actor and Critic networks with independent learning rates
- Entropy regularisation coefficient for exploration
- MAX_SEVERITY = 9  → state normalisation to [0, 1]³

Experiment Structure
--------------------
::

    Experimento (1)
    ├─ Bloque (8)
    │    ├─ Secuencia / Mapa (8)
    │    │    ├─ Trial / Ciudad (3~10)
    │    │    │    └─ Decisión de Recursos (0-10)

Summary
-------
- 1 Experimento
- 8 Bloques
- 8 Secuencias por Bloque
- 3-10 Trials por Secuencia
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
OUTPUT_FILE_PREFIX = 'PES_A2C_'    # Prefix for all output filenames

# ==================== PLAYER & AGENT SETTINGS ====================
PLAYER_TYPE = {  # Decision maker type - SELECT ONE
    1: 'RL_AGENT',     # Tabular Q-Learning agent (fallback / baseline comparison)
    2: 'DQN_AGENT',    # Deep Q-Network (experience replay + target net)
    3: 'A2C_AGENT'     # Advantage Actor-Critic (A2C) agent
}[3]

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

# ==================== REPRODUCIBILITY ====================
SEED = 42                  # Random seed for A2C training reproducibility

# ==================== ADVANTAGE ACTOR-CRITIC (A2C) ====================
# Architecture from Bayesian opt trial #90 (2026-04-23, mean_perf = 0.887236)
AC_ACTOR_HIDDEN_UNITS: list[int] = [128]         # Hidden layers for the Actor (policy) network
AC_CRITIC_HIDDEN_UNITS: list[int] = [128]        # Hidden layers for the Critic (value) network
AC_ACTOR_LR = 0.0006475886789033675              # Adam learning rate for the Actor (trial #90)
AC_CRITIC_LR = 0.004299396121439639              # Adam learning rate for the Critic (trial #90)
AC_ENTROPY_COEFF = 0.005278050634190006          # Entropy regularisation coefficient (trial #90)
AC_DISCOUNT = 0.8545464903255913                 # Discount factor γ for return computation (trial #90)
AC_EPSILON_INITIAL = 0.0                         # Legacy: fixed to 0.0 (on-policy training)
AC_EPSILON_MIN = 0.0                             # Legacy: fixed to 0.0 (on-policy training)
AC_MODEL_ACTOR_FILE = 'ac_actor.keras'           # Saved Actor model filename

# ==================== A2C TRAINING IMPROVEMENTS ====================
# Exponential ε-decay with warm-up (legacy: kept for API compatibility,
# but the on-policy action-selection branch ignores ε entirely)
AC_WARMUP_RATIO = 0.05                               # Legacy: fraction of episodes with ε = ε₀ (unused)
AC_TARGET_RATIO = 0.60                                # Legacy: fraction at which ε reaches ε_min (unused)
# Potential-Based Reward Shaping — PBRS (Ng et al., 1999)
AC_PENALTY_COEFF = 0.1527141911212227                 # PBRS coefficient β; r' = r + β·(γ·Φ(s') − Φ(s)), 0 = disabled
# Generalized Advantage Estimation — GAE(λ) (Schulman et al., 2016)
AC_GAE_LAMBDA = 0.9134752045271409                    # λ: 0 = TD(0), 1 = Monte-Carlo (recommended: 0.90–0.99)
# Gradient clipping (Mnih et al., 2016)
AC_MAX_GRAD_NORM = 1.200061343068206                  # clip_by_global_norm threshold for Actor and Critic gradients
# Cosine annealing learning rate schedule
AC_LR_MIN_RATIO = 0.23749310142258717                 # Minimum LR as fraction of initial (α in CosineDecay)
# Per-action spending cost (training-only; 0 = disabled)
AC_SPEND_COST_COEFF = 0.010385577983525567            # r' = r − coeff·action; discourages trivial max-spend policy
# Initial logit bias for the last action in the Actor policy head
AC_LAST_ACTION_BIAS = -1.3894095635313632             # Reduces initial probability of "spend max" action

# ==================== OPTIMIZATION STRATEGY ====================
AC_OPTIMIZE_MODE = 'full'                             # 'full' = optimize all hyperparams (10 existing + 6 new)
                                                      # 'improvements_only' = fix existing at trial #90 values,
                                                      #   optimize only the 6 new improvement hyperparams

# ==================== RUNTIME & PERSISTENCE ====================
VERBOSE = True             # Enable detailed console logging and initialization messages
SAVE_RESULTS = True        # Save experiment results to JSON/TXT files after each sequence
