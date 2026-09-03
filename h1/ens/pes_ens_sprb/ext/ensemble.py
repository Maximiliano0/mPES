"""Confidence-weighted soft-voting ensemble over DQN, RDQN and TRF."""
from collections import defaultdict
from typing import Any
import os
import numpy
import tensorflow as tf
from ..config.CONFIG import ACTION_COUNT, DEFAULT_CONFIDENCE_POWER, DEFAULT_TEMPERATURE, DEFAULT_WEIGHTS, MODEL_PATHS


def _softmax(values: numpy.ndarray, temperature: float) -> numpy.ndarray:
    """Return a numerically stable softmax distribution."""
    scaled = values / max(float(temperature), 1e-6)
    scaled -= numpy.max(scaled)
    probabilities = numpy.exp(scaled)
    return probabilities / numpy.sum(probabilities)


def _confidence(values: numpy.ndarray, temperature: float) -> float:
    """Calculate normalized inverse entropy confidence from feasible Q-values."""
    probabilities = _softmax(values, temperature)
    entropy = -numpy.sum(probabilities * numpy.log(numpy.maximum(probabilities, 1e-12)))
    maximum = numpy.log(len(values))
    return float(numpy.clip(1.0 - entropy / max(maximum, 1e-12), 0.0, 1.0))


class SoftVotingEnsemble:
    """Combine the three trained models using confidence-weighted probabilities."""

    def __init__(self, weights: dict[str, float] | None = None,
                 temperature: float = DEFAULT_TEMPERATURE,
                 confidence_power: float = DEFAULT_CONFIDENCE_POWER) -> None:
        self._models = {name: self._load_model(path) for name, path in MODEL_PATHS.items()}
        self._weights = weights or dict(DEFAULT_WEIGHTS)
        self._temperature = float(temperature)
        self._confidence_power = float(confidence_power)
        self._histories: dict[tuple[int, int], list[numpy.ndarray]] = defaultdict(list)

    @staticmethod
    def _load_model(path: str) -> tf.keras.Model:
        """Load and validate one Keras model."""
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f'Ensemble model not found: {path}')
        model = tf.keras.models.load_model(path, safe_mode=False)
        output_shape = model.output_shape
        if len(output_shape) != 2 or output_shape[-1] != ACTION_COUNT:
            raise ValueError(f'Model {path} must output {ACTION_COUNT} actions, got {output_shape}')
        return model

    def _model_input(self, name: str, state: numpy.ndarray, key: tuple[int, int]) -> numpy.ndarray:
        """Build the input tensor required by one model."""
        model = self._models[name]
        normalized = numpy.asarray(state, dtype=numpy.float32)
        if len(model.input_shape) == 2:
            return normalized.reshape(1, -1)
        history = self._histories[key]
        history.append(normalized)
        history_len = int(model.input_shape[1])
        window = numpy.zeros((history_len, normalized.size), dtype=numpy.float32)
        window[-min(history_len, len(history)):] = numpy.asarray(history[-history_len:])
        return window[None, ...]

    def predict(self, state: list[float] | numpy.ndarray, resources_left: int,
                sequence_key: tuple[int, int] = (0, 0)) -> tuple[int, float, dict[str, Any]]:
        """Return ensemble action, confidence and model diagnostics.

        Parameters
        ----------
        state : array-like
            Normalized state `[resources, trial, severity]` expected by the models.
        resources_left : int
            Feasible action upper bound.
        sequence_key : tuple[int, int], optional
            Session and sequence key used to maintain recurrent history.
        """
        feasible = min(max(int(resources_left), 0), ACTION_COUNT - 1)
        distributions = []
        diagnostics: dict[str, Any] = {}
        for name in self._models:
            model_input = self._model_input(name, numpy.asarray(state), sequence_key)
            q_values = numpy.asarray(self._models[name](model_input, training=False))[0].astype(numpy.float64)
            masked = q_values[:feasible + 1].copy()
            confidence = _confidence(masked, self._temperature)
            weight = max(float(self._weights.get(name, 1.0)), 0.0) * confidence ** self._confidence_power
            distribution = numpy.zeros(ACTION_COUNT)
            distribution[:feasible + 1] = _softmax(masked, self._temperature)
            distributions.append(weight * distribution)
            diagnostics[name] = {'action': int(numpy.argmax(masked)), 'confidence': confidence, 'q_values': q_values}
        combined = numpy.sum(distributions, axis=0)
        action = int(numpy.argmax(combined[:feasible + 1]))
        confidence = float(numpy.clip(numpy.sum(combined) / max(sum(
            max(float(self._weights.get(name, 1.0)), 0.0) for name in self._models), 1e-12), 0.0, 1.0))
        return action, confidence, diagnostics

    def reset(self, sequence_key: tuple[int, int]) -> None:
        """Reset recurrent history for a sequence."""
        self._histories.pop(sequence_key, None)
