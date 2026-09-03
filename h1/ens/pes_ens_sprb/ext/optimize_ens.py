"""Bayesian optimization for the SPRB ensemble."""
import argparse
import importlib
import json
import os
from typing import Any
import optuna
import numpy
from .ensemble import SoftVotingEnsemble


def _load_evaluation_helpers() -> tuple[Any, Any, Any, Any]:
    """Load the shared Pandemic evaluation helpers from the DQN package."""
    model_module = importlib.import_module('ml.pes_dqn.ext.dqn_model')
    pandemic_module = importlib.import_module('ml.pes_dqn.ext.pandemic')
    tools_module = importlib.import_module('ml.pes_dqn.ext.tools')
    return (getattr(model_module, 'normalize_state'), getattr(pandemic_module, 'Pandemic'),
            getattr(pandemic_module, 'run_experiment'), getattr(tools_module, 'convert_globalseq_to_seqs'))


def _objective(trial: optuna.Trial, trials_per_sequence: numpy.ndarray,
               severities: list[numpy.ndarray]) -> float:
    """Score one confidence-weighted soft-voting configuration in the environment."""
    normalize_state, pandemic_class, run_experiment, _ = _load_evaluation_helpers()
    params = {
        'weight_dqn': trial.suggest_float('weight_dqn', 0.0, 3.0),
        'weight_rdqn': trial.suggest_float('weight_rdqn', 0.0, 3.0),
        'weight_trf': trial.suggest_float('weight_trf', 0.0, 3.0),
        'temperature': trial.suggest_float('temperature', 0.1, 3.0),
        'confidence_power': trial.suggest_float('confidence_power', 0.25, 3.0),
    }
    ensemble = SoftVotingEnsemble(
        {name: params[f'weight_{name}'] for name in ('dqn', 'rdqn', 'trf')},
        params['temperature'], params['confidence_power'])
    environment = pandemic_class()
    environment.verbose = False
    max_resources = environment.max_resources
    max_trials = environment.max_seq_length
    max_severity = environment.max_severity
    current_sequence = [-1]

    def action_function(_environment, state, sequence_id):
        if sequence_id != current_sequence[0]:
            if current_sequence[0] >= 0:
                ensemble.reset((0, current_sequence[0]))
            current_sequence[0] = sequence_id
        normalized_state = normalize_state(state, max_resources, max_trials, max_severity)
        action, _confidence, _diagnostics = ensemble.predict(
            normalized_state, int(state[0]), (0, sequence_id))
        return action

    _, perfs, _ = run_experiment(
        environment, action_function, False, trials_per_sequence, severities)
    mean_perf = float(numpy.mean(perfs))
    trial.set_user_attr('mean_perf', mean_perf)
    trial.set_user_attr('std_perf', float(numpy.std(perfs)))
    trial.set_user_attr('min_perf', float(numpy.min(perfs)))
    trial.set_user_attr('max_perf', float(numpy.max(perfs)))
    return mean_perf


def main() -> None:
    """Optimize SPRB parameters on the shared fixed experiment sequences."""
    parser = argparse.ArgumentParser(description='Optimize the SPRB ensemble.')
    parser.add_argument('n_trials', nargs='?', type=int, default=50)
    default_inputs = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'ml', 'pes_dqn', 'inputs'))
    parser.add_argument('--severity', default=os.path.join(default_inputs, 'initial_severity.csv'))
    parser.add_argument('--lengths', default=os.path.join(default_inputs, 'sequence_lengths.csv'))
    parser.add_argument('--output', default=os.path.join(os.path.dirname(__file__), '..', 'inputs', 'best_params.json'))
    parser.add_argument('--out-dir', default=None, help='Directory for resumable optimisation artifacts.')
    parser.add_argument('--storage', default=None, help='Optuna storage URL.')
    parser.add_argument('--resume', default='', help='Existing run date; retained for launcher compatibility.')
    arguments = parser.parse_args()
    if arguments.out_dir:
        os.makedirs(arguments.out_dir, exist_ok=True)
        arguments.output = os.path.join(arguments.out_dir, 'best_params.json')
    storage = arguments.storage
    if storage and storage.startswith('sqlite:///'):
        storage_dir = os.path.dirname(os.path.abspath(storage.removeprefix('sqlite:///')))
        os.makedirs(storage_dir, exist_ok=True)
    study_name = 'pes_ens_sprb'
    _, _, _, convert_globalseq_to_seqs = _load_evaluation_helpers()
    trials_per_sequence = numpy.loadtxt(arguments.lengths, delimiter=',')
    initial_severity = numpy.loadtxt(arguments.severity, delimiter=',')
    severities = convert_globalseq_to_seqs(trials_per_sequence, initial_severity)
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42),
                                study_name=study_name, storage=storage, load_if_exists=True)
    study.optimize(lambda trial: _objective(trial, trials_per_sequence, severities), n_trials=arguments.n_trials)
    os.makedirs(os.path.dirname(os.path.abspath(arguments.output)), exist_ok=True)
    payload = {'hyperparameters': study.best_params, 'value': study.best_value,
               'mean_perf': study.best_trial.user_attrs['mean_perf'],
               'std_perf': study.best_trial.user_attrs['std_perf']}
    with open(arguments.output, 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
