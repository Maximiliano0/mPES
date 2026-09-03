"""Confidence-weighted action voting with normalized Q-value tie-breaking."""
from collections import defaultdict
from typing import Any
import os
import numpy
import tensorflow as tf
from ..config.CONFIG import ACTION_COUNT, DEFAULT_CONFIDENCE_POWER, DEFAULT_WEIGHTS, MODEL_PATHS


def _softmax(values: numpy.ndarray) -> numpy.ndarray:
    """Return a numerically stable softmax distribution."""
    shifted = values - numpy.max(values)
    probabilities = numpy.exp(shifted)
    return probabilities / numpy.sum(probabilities)


def _confidence(values: numpy.ndarray) -> float:
    """Calculate normalized inverse entropy confidence from feasible Q-values."""
    probabilities = _softmax(values)
    entropy = -numpy.sum(probabilities * numpy.log(numpy.maximum(probabilities, 1e-12)))
    return float(numpy.clip(1.0 - entropy / max(numpy.log(len(values)), 1e-12), 0.0, 1.0))


def _normalize(values: numpy.ndarray) -> numpy.ndarray:
    """Normalize Q-values for fair cross-model tie-breaking."""
    return (values - numpy.mean(values)) / max(float(numpy.std(values)), 1e-8)


class ActionVotingEnsemble:
    """Vote on each model's action and break ties using normalized Q-values."""

    def __init__(self, weights: dict[str, float] | None = None,
                 confidence_power: float = DEFAULT_CONFIDENCE_POWER) -> None:
        self._models = {name: self._load_model(path) for name, path in MODEL_PATHS.items()}
        self._weights = weights or dict(DEFAULT_WEIGHTS)
        self._confidence_power = float(confidence_power)
        self._histories: dict[tuple[int, int], list[numpy.ndarray]] = defaultdict(list)

    @staticmethod
    def _load_model(path: str) -> tf.keras.Model:
        """Load and validate one Keras model."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Ensemble model not found: {path}')
        model = tf.keras.models.load_model(path, safe_mode=False)
        if len(model.output_shape) != 2 or model.output_shape[-1] != ACTION_COUNT:
            raise ValueError(f'Model {path} must output {ACTION_COUNT} actions, got {model.output_shape}')
        return model

    def _model_input(self, name: str, state: numpy.ndarray, key: tuple[int, int]) -> numpy.ndarray:
        """Build the input tensor required by one model."""
        model = self._models[name]
        if len(model.input_shape) == 2:
            return state.reshape(1, -1)
        history = self._histories[key]
        history.append(state)
        history_len = int(model.input_shape[1])
        window = numpy.zeros((history_len, state.size), dtype=numpy.float32)
        window[-min(history_len, len(history)):] = numpy.asarray(history[-history_len:])
        return window[None, ...]

    def predict(self, state: list[float] | numpy.ndarray, resources_left: int,
                sequence_key: tuple[int, int] = (0, 0)) -> tuple[int, float, dict[str, Any]]:
        """Return voted action, aggregate confidence and model diagnostics."""
        feasible = min(max(int(resources_left), 0), ACTION_COUNT - 1)
        scores = numpy.zeros(ACTION_COUNT)
        q_tie = numpy.zeros(ACTION_COUNT)
        diagnostics: dict[str, Any] = {}
        for name in self._models:
            normalized_state = numpy.asarray(state, dtype=numpy.float32)
            model_input = self._model_input(name, normalized_state, sequence_key)
            q_values = numpy.asarray(self._models[name](model_input, training=False))[0].astype(numpy.float64)
            masked = q_values[:feasible + 1].copy()
            confidence = _confidence(masked)
            vote_weight = max(float(self._weights.get(name, 1.0)), 0.0) * confidence ** self._confidence_power
            selected = int(numpy.argmax(masked))
            scores[selected] += vote_weight
            q_tie[:feasible + 1] += _normalize(masked)
            diagnostics[name] = {'action': selected, 'confidence': confidence, 'q_values': q_values}
        feasible_scores = scores[:feasible + 1]
        candidates = numpy.flatnonzero(feasible_scores == numpy.max(feasible_scores))
        action = int(candidates[numpy.argmax(q_tie[candidates])])
        confidence = float(numpy.clip(numpy.max(scores) / max(sum(
            max(float(self._weights.get(name, 1.0)), 0.0) for name in self._models), 1e-12), 0.0, 1.0))
        return action, confidence, diagnostics

    def reset(self, sequence_key: tuple[int, int]) -> None:
        """Reset recurrent history for a sequence."""
        self._histories.pop(sequence_key, None)
