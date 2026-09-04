"""Configuration for confidence-weighted action/Q-value ensemble."""
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
MODEL_ROOT = os.environ.get('MPES_MODEL_ROOT', '').strip()
H_ROOT = os.environ.get('H_DIR', os.path.join(REPO_ROOT, 'h1'))
MODEL_PATHS = {
    'dqn': os.path.join(MODEL_ROOT or os.path.join(H_ROOT, 'ml', 'pes_dqn', 'inputs'), 'dqn_model.keras'),
    'a2c': os.path.join(MODEL_ROOT or os.path.join(H_ROOT, 'ml', 'pes_a2c', 'inputs'), 'ac_actor.keras'),
    'rdqn': os.path.join(MODEL_ROOT or os.path.join(H_ROOT, 'ml', 'pes_rdqn', 'inputs'), 'rdqn_model.keras'),
    'trf': os.path.join(MODEL_ROOT or os.path.join(H_ROOT, 'ml', 'pes_trf', 'inputs'), 'trf_model.keras'),
}
MODEL_ROLES = {'dqn': 'q', 'a2c': 'policy', 'rdqn': 'q', 'trf': 'q'}
ACTION_COUNT = 11
DEFAULT_WEIGHTS = {'dqn': 0.15, 'a2c': 0.10, 'rdqn': 0.25, 'trf': 0.50}
DEFAULT_CONFIDENCE_POWER = 1.0
SEED = 42
