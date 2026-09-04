"""Transformer-first confidence-gated ensemble."""
from collections import defaultdict
from typing import Any
import os
import numpy
import tensorflow as tf
from ..config.CONFIG import ACTION_COUNT, DEFAULT_WEIGHTS, MODEL_PATHS, MODEL_ROLES


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


def _policy_confidence(values: numpy.ndarray) -> float:
    """Calculate normalized inverse entropy confidence from policy probabilities."""
    probabilities = numpy.clip(values, 1e-12, 1.0)
    entropy = -numpy.sum(probabilities * numpy.log(probabilities))
    return float(numpy.clip(1.0 - entropy / max(numpy.log(len(values)), 1e-12), 0.0, 1.0))


class TransformerGuardEnsemble:
    """Use Transformer predictions unless its confidence requires a fallback."""

    def __init__(self, weights: dict[str, float] | None = None,
                 confidence_power: float = 1.0,
                 trf_confidence_threshold: float = 0.5,
                 gate_slope: float = 10.0) -> None:
        self._models = {name: self._load_model(path) for name, path in MODEL_PATHS.items()}
        self._weights = weights or dict(DEFAULT_WEIGHTS)
        self._confidence_power = float(confidence_power)
        self._threshold = float(trf_confidence_threshold)
        self._gate_slope = float(gate_slope)
        self._histories: dict[tuple[str, tuple[int, int]], list[numpy.ndarray]] = defaultdict(list)

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
        history = self._histories[(name, key)]
        history.append(state)
        history_len = int(model.input_shape[1])
        window = numpy.zeros((history_len, state.size), dtype=numpy.float32)
        window[-min(history_len, len(history)):] = numpy.asarray(history[-history_len:])
        return window[None, ...]

    def predict(self, state: list[float] | numpy.ndarray, resources_left: int,
                sequence_key: tuple[int, int] = (0, 0)) -> tuple[int, float, dict[str, Any]]:
        """Return a Transformer-guarded action, confidence and diagnostics."""
        feasible = min(max(int(resources_left), 0), ACTION_COUNT - 1)
        state_array = numpy.asarray(state, dtype=numpy.float32)
        q_values_by_model: dict[str, numpy.ndarray] = {}
        confidences: dict[str, float] = {}
        actions: dict[str, int] = {}
        diagnostics: dict[str, Any] = {}
        for name in self._models:
            model_input = self._model_input(name, state_array, sequence_key)
            values = numpy.asarray(self._models[name](model_input, training=False))[0].astype(numpy.float64)
            masked = values[:feasible + 1].copy()
            if MODEL_ROLES[name] == 'policy':
                masked = numpy.clip(masked, 0.0, None)
                masked /= max(float(numpy.sum(masked)), 1e-12)
                confidences[name] = _policy_confidence(masked)
                distribution = masked
            else:
                confidences[name] = _confidence(masked)
                distribution = _softmax(masked)
            q_values_by_model[name] = distribution
            actions[name] = int(numpy.argmax(masked))
            diagnostics[name] = {'action': actions[name], 'confidence': confidences[name], 'values': values}
        trf_confidence = confidences['trf']
        gate = 1.0 / (1.0 + numpy.exp(-self._gate_slope * (trf_confidence - self._threshold)))
        fallback = numpy.zeros(feasible + 1)
        for name, values in q_values_by_model.items():
            weight = max(float(self._weights.get(name, 1.0)), 0.0) * confidences[name] ** self._confidence_power
            fallback += weight * values
        action = actions['trf'] if gate >= 0.5 else int(numpy.argmax(fallback))
        confidence = float(numpy.clip(gate * trf_confidence + (1.0 - gate) * max(confidences.values()), 0.0, 1.0))
        diagnostics['gate'] = float(gate)
        return action, confidence, diagnostics

    def reset(self, sequence_key: tuple[int, int]) -> None:
        """Reset recurrent history for a sequence."""
        for name in self._models:
            self._histories.pop((name, sequence_key), None)
