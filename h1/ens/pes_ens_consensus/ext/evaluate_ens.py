"""Evaluate the confidence consensus ensemble."""
import argparse
from datetime import datetime
import glob
import json
import os
from typing import Any
import numpy
from .ensemble import ConfidenceConsensusEnsemble
from ..config.CONFIG import NUM_SEQUENCES
from ..src.pandemic_env import Pandemic, run_experiment
from ..src.tools_env import convert_globalseq_to_seqs
from ..src.result_formatter_env import generate_results_report


def _load_params(path: str) -> dict[str, Any]:
    """Load ensemble parameters from a JSON sidecar."""
    with open(path, 'r', encoding='utf-8') as file:
        payload = json.load(file)
    return dict(payload.get('hyperparameters', payload))


def _find_best_params(default_inputs: str) -> str:
    """Find the optimization result with the highest recorded objective value."""
    candidates = [os.path.join(default_inputs, 'best_params.json')]
    candidates.extend(glob.glob(os.path.join(default_inputs, '*_BAYESIAN_OPT', 'best_params.json')))
    valid_candidates: list[tuple[float, float, str]] = []
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as file:
                payload = json.load(file)
            value = float(payload.get('value', float('-inf')))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        valid_candidates.append((value, os.path.getmtime(path), path))
    if not valid_candidates:
        return candidates[0]
    return max(valid_candidates)[2]


def _normalize_state(state: Any, max_resources: int, max_trials: int,
                     max_severity: int) -> numpy.ndarray:
    """Scale a raw environment state to the model input range."""
    return numpy.array([
        state[0] / max(max_resources, 1),
        state[1] / max(max_trials, 1),
        state[2] / max(max_severity, 1),
    ], dtype=numpy.float32)


def main() -> None:
    """Evaluate the ensemble on the fixed sequences used by model evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate the confidence consensus ensemble.')
    package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_inputs = os.path.join(package_root, 'inputs')
    default_outputs = os.path.join(package_root, 'outputs')
    parser.add_argument('--severity', default=os.path.join(default_inputs, 'initial_severity.csv'))
    parser.add_argument('--lengths', default=os.path.join(default_inputs, 'sequence_lengths.csv'))
    parser.add_argument('--params', default=None)
    parser.add_argument('--output', default=default_outputs)
    arguments = parser.parse_args()
    params_path = arguments.params or _find_best_params(default_inputs)
    params = _load_params(params_path) if os.path.isfile(params_path) else {}
    weights = {name: float(params.get(f'weight_{name}', default))
               for name, default in {'dqn': 0.15, 'a2c': 0.10, 'rdqn': 0.25, 'trf': 0.50}.items()}
    ensemble = ConfidenceConsensusEnsemble(weights, float(params.get('confidence_power', 1.0)),
                                           float(params.get('agreement_bonus', 0.5)),
                                           float(params.get('disagreement_penalty', 0.1)))
    trials_per_sequence = numpy.loadtxt(arguments.lengths, delimiter=',')
    initial_severity = numpy.loadtxt(arguments.severity, delimiter=',')
    severities = convert_globalseq_to_seqs(trials_per_sequence, initial_severity)
    environment = Pandemic()
    environment.verbose = False
    current_sequence = [-1]

    def action_function(_environment: Any, state: Any, sequence_id: int) -> int:
        """Select an action from the ensemble for one environment state."""
        if sequence_id != current_sequence[0]:
            if current_sequence[0] >= 0:
                ensemble.reset((0, current_sequence[0]))
            current_sequence[0] = sequence_id
        normalized_state = _normalize_state(
            state, environment.max_resources, environment.max_seq_length, environment.max_severity)
        action, _confidence, _diagnostics = ensemble.predict(
            normalized_state, int(state[0]), (0, sequence_id))
        return action

    _, performances, _ = run_experiment(
        environment, action_function, False, trials_per_sequence, severities)
    subject_id = f'{datetime.now():%Y-%m-%d}_ENS_CONF_CONSENSUS'
    output_path = os.path.join(arguments.output, subject_id)
    os.makedirs(output_path, exist_ok=True)
    block_size = int(NUM_SEQUENCES)
    performances_by_block = [performances[index:index + block_size]
                             for index in range(0, len(performances), block_size)]
    resource_data = {
        'total_resources_per_sequence': environment.max_resources,
        'agent_type': 'ENS_CONF_CONSENSUS',
        'num_blocks': len(performances_by_block),
        'num_sequences': block_size,
        'total_trials': len(performances),
    }
    json_path, png_path = generate_results_report(
        subject_id, output_path, performances, performances_by_block, resource_data, 'PES_ENS_CONF_CONSENSUS_')
    numpy.save(os.path.join(output_path, f'PES_ENS_CONF_CONSENSUS_performances_{subject_id}.npy'), numpy.asarray(performances))
    print(json.dumps({'results_json': json_path, 'results_png': png_path}, indent=2))


if __name__ == '__main__':
    main()
