"""Bayesian optimization for the accQ ensemble."""
import argparse
import json
import os
from datetime import datetime
from typing import Any
import optuna
import numpy
from .ensemble import ActionVotingEnsemble
from ..src.dqn_model import normalize_state as _normalize_state
from ..src.pandemic_env import Pandemic, run_experiment as _run_experiment
from ..src.tools_env import convert_globalseq_to_seqs as _convert_globalseq_to_seqs


def _load_evaluation_helpers() -> tuple[Any, Any, Any, Any]:
    """Load the local Pandemic evaluation helpers."""
    return _normalize_state, Pandemic, _run_experiment, _convert_globalseq_to_seqs


def _objective(trial: optuna.Trial, trials_per_sequence: numpy.ndarray,
               severities: list[numpy.ndarray]) -> float:
    """Score one confidence-weighted action-voting configuration in the environment."""
    normalize_state, pandemic_class, run_experiment, _ = _load_evaluation_helpers()
    params = {f'weight_{name}': trial.suggest_float(f'weight_{name}', 0.0, 3.0)
              for name in ('dqn', 'a2c', 'rdqn', 'trf')}
    params['confidence_power'] = trial.suggest_float('confidence_power', 0.25, 3.0)
    ensemble = ActionVotingEnsemble(
        {name: params[f'weight_{name}'] for name in ('dqn', 'a2c', 'rdqn', 'trf')},
        params['confidence_power'])
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
    """Optimize accQ parameters on the shared fixed experiment sequences."""
    parser = argparse.ArgumentParser(description='Optimize the accQ ensemble.')
    parser.add_argument('n_trials', nargs='?', type=int, default=50)
    default_inputs = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'inputs'))
    parser.add_argument('--severity', default=os.path.join(default_inputs, 'initial_severity.csv'))
    parser.add_argument('--lengths', default=os.path.join(default_inputs, 'sequence_lengths.csv'))
    parser.add_argument('--output', default=os.path.join(os.path.dirname(__file__), '..', 'inputs', 'best_params.json'))
    parser.add_argument('--out-dir', default=None, help='Directory for resumable optimisation artifacts.')
    parser.add_argument('--storage', default=None, help='Optuna storage URL.')
    parser.add_argument('--resume', default='', help='Existing run date; retained for launcher compatibility.')
    arguments = parser.parse_args()
    opt_date = arguments.resume or datetime.now().strftime('%Y-%m-%d')
    opt_dir = arguments.out_dir or os.path.join(default_inputs, f'{opt_date}_BAYESIAN_OPT')
    arguments.out_dir = opt_dir
    if arguments.out_dir:
        os.makedirs(arguments.out_dir, exist_ok=True)
        arguments.output = os.path.join(arguments.out_dir, 'best_params.json')
    db_path = os.path.join(opt_dir, f'optuna_study_{opt_date}.db')
    storage = arguments.storage or f'sqlite:///{db_path}'
    if storage and storage.startswith('sqlite:///'):
        storage_dir = os.path.dirname(os.path.abspath(storage.removeprefix('sqlite:///')))
        os.makedirs(storage_dir, exist_ok=True)
    study_name = 'pes_ens_accq'
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
