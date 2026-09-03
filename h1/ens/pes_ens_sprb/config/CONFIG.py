"""Configuration for confidence-weighted soft-voting ensemble."""
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
MODEL_PATHS = {
    'dqn': os.path.join(REPO_ROOT, 'h1', 'ml', 'pes_dqn', 'inputs', 'dqn_model.keras'),
    'rdqn': os.path.join(REPO_ROOT, 'h1', 'ml', 'pes_rdqn', 'inputs', 'rdqn_model.keras'),
    'trf': os.path.join(REPO_ROOT, 'h1', 'ml', 'pes_trf', 'inputs', 'trf_model.keras'),
}
ACTION_COUNT = 11
DEFAULT_WEIGHTS = {'dqn': 1.0, 'rdqn': 1.0, 'trf': 1.0}
DEFAULT_TEMPERATURE = 1.0
DEFAULT_CONFIDENCE_POWER = 1.0
SEED = 42
