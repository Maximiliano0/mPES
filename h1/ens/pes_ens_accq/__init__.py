"""Confidence-weighted action/Q-value ensemble package."""
######################
## External Imports ##
######################
import os

# TensorFlow/CUDA log suppression — must precede any transitive TF import.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# GPU policy: pin CPU unless the launcher explicitly opts in via MPES_USE_GPU=1.
if os.environ.get('MPES_USE_GPU', '0') != '1':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Determinism flags (honoured by TF ≥ 2.9 on both CPU and cuDNN-enabled GPU).
os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
os.environ.setdefault('TF_CUDNN_DETERMINISTIC', '1')

# Quiet-import TF: suppress native-library stderr (cudart_stub.cc) emitted
# before absl::InitializeLog(). The module is cached in sys.modules, so
# subsequent ``import tensorflow`` calls in the package are free.
_devnull = os.open(os.devnull, os.O_WRONLY)
_old_stderr_fd = os.dup(2)
os.dup2(_devnull, 2)
os.close(_devnull)
try:
    import tensorflow  # noqa: F401
finally:
    os.dup2(_old_stderr_fd, 2)
    os.close(_old_stderr_fd)
