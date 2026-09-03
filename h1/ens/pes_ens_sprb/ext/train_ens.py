"""Evaluate the confidence-weighted soft-voting ensemble."""
import argparse
import json
import os
from typing import Any
import numpy
from .ensemble import SoftVotingEnsemble


def _load_params(path: str) -> dict[str, Any]:
    """Load ensemble parameters from a JSON sidecar."""
    with open(path, 'r', encoding='utf-8') as file:
        payload = json.load(file)
    return dict(payload.get('hyperparameters', payload))


def main() -> None:
    """Run ensemble inference over states stored in an NPZ file."""
    parser = argparse.ArgumentParser(description='Evaluate the SPRB ensemble.')
    parser.add_argument('--states', required=True, help='NPZ containing normalized states and resources_left.')
    parser.add_argument('--params', default=os.path.join(os.path.dirname(__file__), '..', 'inputs', 'best_params.json'))
    arguments = parser.parse_args()
    params = _load_params(arguments.params) if os.path.isfile(arguments.params) else {}
    weights = {name: float(params.get(f'weight_{name}', 1.0)) for name in ('dqn', 'rdqn', 'trf')}
    ensemble = SoftVotingEnsemble(weights, float(params.get('temperature', 1.0)), float(params.get('confidence_power', 1.0)))
    data = numpy.load(arguments.states)
    states = numpy.asarray(data['states'], dtype=numpy.float32)
    resources = numpy.asarray(data.get('resources_left', numpy.full(len(states), 10)), dtype=numpy.int32)
    actions = []
    confidences = []
    for index, state in enumerate(states):
        action, confidence, _ = ensemble.predict(state, int(resources[index]), (0, index))
        actions.append(action)
        confidences.append(confidence)
    print(json.dumps({'actions': actions, 'mean_confidence': float(numpy.mean(confidences))}, indent=2))


if __name__ == '__main__':
    main()
