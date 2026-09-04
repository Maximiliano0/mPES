"""Confidence consensus ensemble with agreement and disagreement terms."""
from collections import defaultdict
from typing import Any
import os
import numpy
import tensorflow as tf
from ..config.CONFIG import (ACTION_COUNT, DEFAULT_CONFIDENCE_POWER, DEFAULT_PRIOR_SIGMA,
                             DEFAULT_PRIOR_WEIGHT, DEFAULT_WEIGHTS, MAX_SEVERITY, MODEL_PATHS,
                             MODEL_ROLES)


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
    """Normalize Q-values for cross-model comparison."""
    return (values - numpy.mean(values)) / max(float(numpy.std(values)), 1e-8)


class ConfidenceConsensusEnsemble:
    """Select actions using soft voting and a severity-informed prior."""

    def __init__(self, weights: dict[str, float] | None = None,
                 confidence_power: float = DEFAULT_CONFIDENCE_POWER,
                 agreement_bonus: float = 0.5,
                 disagreement_penalty: float = 0.1,
                 prior_weight: float = DEFAULT_PRIOR_WEIGHT,
                 prior_sigma: float = DEFAULT_PRIOR_SIGMA) -> None:
        self._models = {name: self._load_model(path) for name, path in MODEL_PATHS.items()}
        self._weights = weights or dict(DEFAULT_WEIGHTS)
        self._confidence_power = float(confidence_power)
        self._agreement_bonus = float(agreement_bonus)
        self._disagreement_penalty = float(disagreement_penalty)
        self._prior_weight = float(numpy.clip(prior_weight, 0.0, 1.0))
        self._prior_sigma = max(float(prior_sigma), 1e-3)
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
        """Return a prior-aware consensus action, confidence and diagnostics."""
        feasible = min(max(int(resources_left), 0), ACTION_COUNT - 1)
        state_array = numpy.asarray(state, dtype=numpy.float32)
        scores = numpy.zeros(feasible + 1, dtype=numpy.float64)
        actions: dict[str, int] = {}
        confidences: dict[str, float] = {}
        diagnostics: dict[str, Any] = {}
        for name in self._models:
            model_input = self._model_input(name, state_array, sequence_key)
            values = numpy.asarray(self._models[name](model_input, training=False))[0].astype(numpy.float64)
            if MODEL_ROLES[name] == 'actor':
                probabilities = numpy.clip(values, 0.0, None)
                probabilities /= max(float(numpy.sum(probabilities)), 1e-12)
            else:
                probabilities = _softmax(values)
            masked = probabilities[:feasible + 1]
            masked /= max(float(numpy.sum(masked)), 1e-12)
            confidence = _confidence(masked)
            action = int(numpy.argmax(masked))
            weight = max(float(self._weights.get(name, 1.0)), 0.0) * confidence ** self._confidence_power
            scores += weight * masked
            actions[name] = action
            confidences[name] = confidence
            diagnostics[name] = {'action': action, 'confidence': confidence, 'values': values}
        for action_index in range(feasible + 1):
            agreeing_confidence = sum(confidences[name] for name in self._models if actions[name] == action_index)
            disagreeing_confidence = sum(confidences[name] for name in self._models if actions[name] != action_index)
            scores[action_index] += self._agreement_bonus * agreeing_confidence
            scores[action_index] -= self._disagreement_penalty * disagreeing_confidence
        if feasible > 0:
            scores[0] *= 0.3
        scores = numpy.clip(scores, 0.0, None)
        scores /= max(float(numpy.sum(scores)), 1e-12)
        severity_raw = float(state_array[2]) * MAX_SEVERITY
        prior_actions = numpy.arange(feasible + 1, dtype=numpy.float64)
        prior = numpy.exp(-((prior_actions - severity_raw) ** 2) / (2.0 * self._prior_sigma ** 2))
        prior /= max(float(numpy.sum(prior)), 1e-12)
        scores = ((1.0 - self._prior_weight) * scores) + (self._prior_weight * prior)
        action = int(numpy.argmax(scores))
        if severity_raw >= 6.0:
            severity_floor = int(severity_raw // 2)
            if action < severity_floor <= feasible:
                action = severity_floor
                scores = numpy.zeros(feasible + 1, dtype=numpy.float64)
                scores[action] = 1.0
        confidence = float(numpy.clip(numpy.mean(list(confidences.values())), 0.0, 1.0))
        diagnostics['agreement'] = float(sum(value for name, value in confidences.items() if actions[name] == action))
        diagnostics['prior_weight'] = self._prior_weight
        diagnostics['prior_sigma'] = self._prior_sigma
        return action, confidence, diagnostics

    def reset(self, sequence_key: tuple[int, int]) -> None:
        """Reset recurrent history for a sequence."""
        for name in self._models:
            self._histories.pop((name, sequence_key), None)
