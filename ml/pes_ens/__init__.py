"""
Package initialization module for pes_ens (Pandemic Experiment Scenario
with Ensemble Agent).

The pes_ens variant performs **inference only**: at every trial it loads
four pre-trained sibling models (``pes_dqn``, ``pes_a2c``, ``pes_rdqn``,
``pes_trf``) from their canonical ``inputs/`` directories, converts each
model's output to an action-probability distribution, averages them
(weighted soft voting), masks infeasible actions and selects the
allocation via ``argmax``.

Handles package setup including:
- Configuration loading from config/CONFIG.py
- Path definitions for documentation, outputs and inputs directories
- ANSI color class for styled terminal output
- Virtual-environment validation with user prompt
- NumPy print/error configuration and TensorFlow log suppression
- Pandemic dynamic parameters (RESPONSE_MULTIPLIER alpha, SEVERITY_MULTIPLIER beta)
- Package exports via __all__
"""
######################
## External Imports ##
######################
import os

# TensorFlow/CUDA log suppression -- must precede any transitive TF import.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# GPU policy: by default we pin TF to CPU.  On Colab Pro+ the launcher sets
# ``MPES_USE_GPU=1`` and we leave CUDA_VISIBLE_DEVICES alone.  TF is also
# configured for *deterministic* ops so the same seed reproduces the same
# trajectory (within FP tolerance).
if os.environ.get('MPES_USE_GPU', '0') != '1':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
os.environ.setdefault('TF_CUDNN_DETERMINISTIC', '1')

# Quiet-import TF: suppress native-library stderr (cudart_stub.cc) emitted
# before absl::InitializeLog().
_devnull = os.open(os.devnull, os.O_WRONLY)
_old_stderr_fd = os.dup(2)
os.dup2(_devnull, 2)
os.close(_devnull)
try:
    import tensorflow  # noqa: F401
finally:
    os.dup2(_old_stderr_fd, 2)
    os.close(_old_stderr_fd)

# Configure GPU memory growth when available.
for _gpu in tensorflow.config.list_physical_devices('GPU'):
    tensorflow.config.experimental.set_memory_growth(_gpu, True)

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
        input()
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
# sev_n = beta * sev_(n-1) - alpha * a  (a=allocated resources, sev=severity)
RESPONSE_MULTIPLIER = PANDEMIC_PARAMETER  # alpha
SEVERITY_MULTIPLIER = 1 + PANDEMIC_PARAMETER  # beta

# === BENCHMARK OVERRIDE HOOK (general/) ============================
# When mPES is launched by the OOD benchmark harness in ``general/``,
# these env vars redirect outputs and resize the experiment grid
# without modifying CONFIG.py. Inputs (CSVs) are swapped in-place by
# the harness so INPUTS_PATH stays untouched.
_bench_outputs = os.environ.get('MPES_OUTPUTS_PATH')
if _bench_outputs:
    OUTPUTS_PATH = _bench_outputs
_bench_blocks = os.environ.get('MPES_NUM_BLOCKS')
if _bench_blocks:
    NUM_BLOCKS = int(_bench_blocks)
_bench_seqs = os.environ.get('MPES_NUM_SEQUENCES')
if _bench_seqs:
    NUM_SEQUENCES = int(_bench_seqs)
# === END BENCHMARK OVERRIDE HOOK ===================================

# Set some nice numpy printing defaults and error handling
numpy.set_printoptions(threshold=sys.maxsize, precision=3, suppress=True,
                       linewidth=80, nanstr="--", infstr="inf")
numpy.seterr(all='raise', under='ignore')

#########################################
### Print final init variables to log ###
#########################################
if VERBOSE:
    print(f"\n{'=' * 100}")
    print("  EXPERIMENT CONFIGURATION PARAMETERS")
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
