"""Pandemic environment constants, mirrored from ``ml/pes_dqn/config/CONFIG.py``.

Kept local so this ensemble's evaluation harness never imports from ``ml/``.
"""

AVAILABLE_RESOURCES_PER_SEQUENCE = 39
NUM_MAX_TRIALS = 10
NUM_SEQUENCES = 8
MAX_SEVERITY = 9
MAX_ALLOCATABLE_RESOURCES = 10
MIN_ALLOCATABLE_RESOURCES = 0
PANDEMIC_PARAMETER = 0.4
RESPONSE_MULTIPLIER = PANDEMIC_PARAMETER          # alpha
SEVERITY_MULTIPLIER = 1 + PANDEMIC_PARAMETER      # beta
