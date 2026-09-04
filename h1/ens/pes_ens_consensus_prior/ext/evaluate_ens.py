"""Evaluate the confidence consensus ensemble with a severity prior."""
import argparse
from datetime import datetime
import importlib
import json
import os
from typing import Any
import numpy
from .ensemble import ConfidenceConsensusEnsemble


def _load_params(path: str) -> dict[str, Any]:
    """Load ensemble parameters from a JSON sidecar."""
    with open(path, 'r', encoding='utf-8') as file:
        payload = json.load(file)
    return dict(payload.get('hyperparameters', payload))


def _load_evaluation_helpers() -> tuple[Any, Any, Any, Any]:
    """Load the shared environment and fixed-sequence evaluation helpers."""
    pandemic_module = importlib.import_module('ml.pes_dqn.ext.pandemic')
    tools_module = importlib.import_module('ml.pes_dqn.ext.tools')
    formatter_module = importlib.import_module('ml.pes_dqn.src.result_formatter')
    return (getattr(pandemic_module, 'Pandemic'), getattr(pandemic_module, 'run_experiment'),
            getattr(tools_module, 'convert_globalseq_to_seqs'),
            getattr(formatter_module, 'generate_results_report'))


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
    parser.add_argument('--params', default=os.path.join(default_inputs, 'best_params.json'))
    parser.add_argument('--output', default=default_outputs)
    arguments = parser.parse_args()
    params = _load_params(arguments.params) if os.path.isfile(arguments.params) else {}
    weights = {name: float(params.get(f'weight_{name}', default))
               for name, default in {'dqn': 0.15, 'a2c': 0.10, 'rdqn': 0.25, 'trf': 0.50}.items()}
    ensemble = ConfidenceConsensusEnsemble(weights, float(params.get('confidence_power', 1.0)),
                                           float(params.get('agreement_bonus', 0.5)),
                                           float(params.get('disagreement_penalty', 0.1)),
                                           float(params.get('prior_weight', 0.10)),
                                           float(params.get('prior_sigma', 1.5)))
    pandemic_class, run_experiment, convert_globalseq_to_seqs, generate_results_report = _load_evaluation_helpers()
    trials_per_sequence = numpy.loadtxt(arguments.lengths, delimiter=',')
    initial_severity = numpy.loadtxt(arguments.severity, delimiter=',')
    severities = convert_globalseq_to_seqs(trials_per_sequence, initial_severity)
    environment = pandemic_class()
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
    subject_id = f'{datetime.now():%Y-%m-%d}_ENS_CONSENSUS_PRIOR'
    output_path = os.path.join(arguments.output, subject_id)
    os.makedirs(output_path, exist_ok=True)
    block_size = int(getattr(importlib.import_module('ml.pes_dqn'), 'NUM_SEQUENCES'))
    performances_by_block = [performances[index:index + block_size]
                             for index in range(0, len(performances), block_size)]
    resource_data = {
        'total_resources_per_sequence': environment.max_resources,
        'agent_type': 'ENS_CONSENSUS_PRIOR',
        'num_blocks': len(performances_by_block),
        'num_sequences': block_size,
        'total_trials': len(performances),
    }
    json_path, png_path = generate_results_report(
        subject_id, output_path, performances, performances_by_block, resource_data, 'PES_ENS_CONSENSUS_PRIOR_')
    numpy.save(os.path.join(output_path, f'PES_ENS_CONSENSUS_PRIOR_performances_{subject_id}.npy'), numpy.asarray(performances))
    print(json.dumps({'results_json': json_path, 'results_png': png_path}, indent=2))


if __name__ == '__main__':
    main()
