"""Configuration for confidence-weighted action/Q-value ensemble."""
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
MODEL_ROOT = os.environ.get('MPES_MODEL_ROOT', '').strip()
H1_ROOT = os.environ.get('H1_DIR', os.path.join(REPO_ROOT, 'h1'))
MODEL_PATHS = {
    'dqn': os.path.join(MODEL_ROOT or os.path.join(H1_ROOT, 'ml', 'pes_dqn', 'inputs'), 'dqn_model.keras'),
    'rdqn': os.path.join(MODEL_ROOT or os.path.join(H1_ROOT, 'ml', 'pes_rdqn', 'inputs'), 'rdqn_model.keras'),
    'trf': os.path.join(MODEL_ROOT or os.path.join(H1_ROOT, 'ml', 'pes_trf', 'inputs'), 'trf_model.keras'),
}
ACTION_COUNT = 11
DEFAULT_WEIGHTS = {'dqn': 1.0, 'rdqn': 1.0, 'trf': 1.0}
DEFAULT_CONFIDENCE_POWER = 1.0
SEED = 42
