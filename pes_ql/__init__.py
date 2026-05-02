"""
Package initialization module for pes_ql (Pandemic Experiment Scenario
with Bayesian Optimisation).

Variant of pes_base that adds Bayesian hyperparameter optimisation (Optuna / TPE)
on top of Q-Learning.  This __init__ is structurally identical to pes/__init__;
the Bayesian-specific behaviour lives in ext/optimize_rl.py and ext/train_rl.py.

Handles package setup including:
- Configuration loading from config/CONFIG.py (includes SEED for reproducibility)
- Path definitions for documentation, outputs, and inputs directories
- ANSI color class for styled terminal output (BOLD, RED, GREEN, etc.)
- Virtual environment validation with user prompt
- NumPy print/error configuration and TensorFlow log suppression
- Pandemic dynamic parameters (RESPONSE_MULTIPLIER α, SEVERITY_MULTIPLIER β)
- Package exports via __all__ (36 symbols)
"""
######################
## External Imports ##
######################
import os

# CUDA / TF env vars kept for parity with sibling packages, even though
# pes_ql is pure-NumPy tabular Q-Learning and does not import TensorFlow.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import warnings
import numpy
from .config import CONFIG

# Suppress non-critical NumPy/SciPy compatibility warnings
warnings.filterwarnings('ignore', message='.*A NumPy version.*SciPy.*')

###########
## PATHs ##
###########
PKG_ROOT = os.path.dirname(os.path.abspath(__file__))

DOCUMENTATION_PATH = os.path.join(PKG_ROOT, 'doc')
OUTPUTS_PATH = os.path.join(PKG_ROOT, 'outputs')
INPUTS_PATH = os.path.join(PKG_ROOT, 'inputs')

#############################
## ANSI Color Escape Codes ##
#############################


class ANSI:
    """ANSI escape-code constants for styled terminal output."""

    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    ORANGE = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    GRAY = '\033[90m'
    RESET = '\033[0m'


#################################
## Suggest virtual-environment ##
#################################
if not os.getenv('VIRTUAL_ENV'):
    print(
        f"""{ANSI.PURPLE}
Warning: No suitable VIRTUAL_ENV environmental variable detected.

In order to ensure consistency / reproducibility between runs, you might want to
consider always running this experiment from within a suitable python virtual
environment, containing the python package versions specified in the package's
requirements.txt file.

Press ENTER if you'd like to continue regardless (or Ctrl-C to abort).
{ANSI.RESET}""")
    try:
        input()   # i.e. press Enter
    except KeyboardInterrupt:
        print('\n\nExiting...')
        sys.exit()

#######################################
### Process selected CONFIG.py file ###
#######################################

AVAILABLE_RESOURCES_PER_SEQUENCE = CONFIG.AVAILABLE_RESOURCES_PER_SEQUENCE
INIT_NO_OF_CITIES = CONFIG.INIT_NO_OF_CITIES
INITIAL_SEVERITY_FILE = CONFIG.INITIAL_SEVERITY_FILE
SEQ_LENGTHS_FILE = CONFIG.SEQ_LENGTHS_FILE
MAX_ALLOCATABLE_RESOURCES = CONFIG.MAX_ALLOCATABLE_RESOURCES
MAX_INIT_RESOURCES = CONFIG.MAX_INIT_RESOURCES
MAX_INIT_SEVERITY = CONFIG.MAX_INIT_SEVERITY
MIN_ALLOCATABLE_RESOURCES = CONFIG.MIN_ALLOCATABLE_RESOURCES
MIN_INIT_RESOURCES = CONFIG.MIN_INIT_RESOURCES
MIN_INIT_SEVERITY = CONFIG.MIN_INIT_SEVERITY
NUM_ATTEMPTS_TO_ASSIGN_SEQ = CONFIG.NUM_ATTEMPTS_TO_ASSIGN_SEQ
NUM_BLOCKS = CONFIG.NUM_BLOCKS
NUM_MAX_TRIALS = CONFIG.NUM_MAX_TRIALS
NUM_MIN_TRIALS = CONFIG.NUM_MIN_TRIALS
NUM_SEQUENCES = CONFIG.NUM_SEQUENCES
OUTPUT_FILE_PREFIX = CONFIG.OUTPUT_FILE_PREFIX
PANDEMIC_PARAMETER = CONFIG.PANDEMIC_PARAMETER
PLAYER_TYPE = CONFIG.PLAYER_TYPE
RANDOM_INITIAL_SEVERITY = CONFIG.RANDOM_INITIAL_SEVERITY
SAVE_INITIAL_SEVERITY_TO_FILE = CONFIG.SAVE_INITIAL_SEVERITY_TO_FILE
SAVE_RESULTS = CONFIG.SAVE_RESULTS
STARTING_BLOCK_INDEX = CONFIG.STARTING_BLOCK_INDEX
STARTING_SEQ_INDEX = CONFIG.STARTING_SEQ_INDEX
TOTAL_NUM_TRIALS_IN_BLOCK = CONFIG.TOTAL_NUM_TRIALS_IN_BLOCK
TRUST_MAX = CONFIG.TRUST_MAX
USE_FIXED_BLOCK_SEQUENCES = CONFIG.USE_FIXED_BLOCK_SEQUENCES
VERBOSE = CONFIG.VERBOSE
AGGREGATION_METHOD = CONFIG.AGGREGATION_METHOD
MAX_SEVERITY = CONFIG.MAX_SEVERITY

##############################################
### Process imported configuration options ###
##############################################
# sev_n = * β * sev_(n-1) - α * a  --> a is allocated resources, sev is severity
RESPONSE_MULTIPLIER = PANDEMIC_PARAMETER  # α (Alpha)
SEVERITY_MULTIPLIER = 1 + PANDEMIC_PARAMETER  # β (Beta)

# Set some nice numpy printing defaults and error handling
numpy.set_printoptions(threshold=sys.maxsize, precision=3, suppress=True,
                       linewidth=80, nanstr="--", infstr="∞")
numpy.seterr(all='raise', under='ignore')   # underflow silently truncates to 0.0

#########################################
### Print final init variables to log ###
#########################################
if VERBOSE:
    print(f"\n{'=' * 100}")
    print(f"  EXPERIMENT CONFIGURATION PARAMETERS")
    print(f"{'=' * 100}\n")
    print(f"{'Variable Name':<45} {'Variable Value':<30} {'Suggested Value':<20}")
    print(f"{'-' * 100}")
    print(f"{'AVAILABLE_RESOURCES_PER_SEQUENCE':<45} {str(AVAILABLE_RESOURCES_PER_SEQUENCE):<30} {'39':<20}")
    print(f"{'INIT_NO_OF_CITIES':<45} {str(INIT_NO_OF_CITIES):<30}")
    print(f"{'NUM_BLOCKS':<45} {str(NUM_BLOCKS):<30} {'8':<20}")
    print(f"{'NUM_MAX_TRIALS':<45} {str(NUM_MAX_TRIALS):<30} {'10':<20}")
    print(f"{'NUM_MIN_TRIALS':<45} {str(NUM_MIN_TRIALS):<30} {'3':<20}")
    print(f"{'NUM_SEQUENCES':<45} {str(NUM_SEQUENCES):<30} {'8':<20}")
    print(f"{'PANDEMIC_PARAMETER':<45} {str(PANDEMIC_PARAMETER):<30} {'0.4':<20}")
    print(f"{'PLAYER_TYPE':<45} {str(PLAYER_TYPE):<30}")
    print(f"{'RESPONSE_MULTIPLIER':<45} {str(RESPONSE_MULTIPLIER):<30} {'0.4':<20}")
    print(f"{'SEVERITY_MULTIPLIER':<45} {str(SEVERITY_MULTIPLIER):<30} {'1.4':<20}")
    print(f"{'TOTAL_NUM_TRIALS_IN_BLOCK':<45} {str(TOTAL_NUM_TRIALS_IN_BLOCK):<30} {'45':<20}")
    print(f"{'USE_FIXED_BLOCK_SEQUENCES':<45} {str(USE_FIXED_BLOCK_SEQUENCES):<30}")
    print(f"{'INITIAL_SEVERITY_FILE':<45} {str(INITIAL_SEVERITY_FILE):<30}")
    print(f"{'SEQ_LENGTHS_FILE':<45} {str(SEQ_LENGTHS_FILE):<30}")
    print(f"{'AGGREGATION_METHOD':<45} {str(AGGREGATION_METHOD):<30}")
    print(f"{'=' * 100}\n")

##############################
### Define package exports ###
##############################
__all__ = [
    'PKG_ROOT',
    'DOCUMENTATION_PATH',
    'OUTPUTS_PATH',
    'INPUTS_PATH',
    'ANSI',
    'AVAILABLE_RESOURCES_PER_SEQUENCE',
    'INIT_NO_OF_CITIES',
    'INITIAL_SEVERITY_FILE',
    'SEQ_LENGTHS_FILE',
    'MAX_ALLOCATABLE_RESOURCES',
    'MAX_INIT_RESOURCES',
    'MAX_INIT_SEVERITY',
    'MAX_SEVERITY',
    'MIN_ALLOCATABLE_RESOURCES',
    'MIN_INIT_RESOURCES',
    'MIN_INIT_SEVERITY',
    'NUM_ATTEMPTS_TO_ASSIGN_SEQ',
    'NUM_BLOCKS',
    'NUM_MAX_TRIALS',
    'NUM_MIN_TRIALS',
    'NUM_SEQUENCES',
    'OUTPUT_FILE_PREFIX',
    'PANDEMIC_PARAMETER',
    'PLAYER_TYPE',
    'SAVE_RESULTS',
    'STARTING_BLOCK_INDEX',
    'STARTING_SEQ_INDEX',
    'TOTAL_NUM_TRIALS_IN_BLOCK',
    'TRUST_MAX',
    'USE_FIXED_BLOCK_SEQUENCES',
    'VERBOSE',
    'RANDOM_INITIAL_SEVERITY',
    'SAVE_INITIAL_SEVERITY_TO_FILE',
    'RESPONSE_MULTIPLIER',
    'SEVERITY_MULTIPLIER',
    'AGGREGATION_METHOD'
]
