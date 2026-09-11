"""Agent-internals figures of the Transformer (``pes_trf``) in the shared style.

Re-renders the four panels documented in the thesis (Materials, "Variables
internas del agente Transformer") from artefacts already produced by ``h1/``:

* per-decision entropy confidence — ``confsrl_<date>.npy`` written by
  ``ml/pes_trf/ext/train_transformer.py`` in ``inputs/<date>_TRF_TRAIN``;
* normalised performance per sequence — ``per_sequence_perf`` of the
  benchmark cell ``results/individual/cells/pes_trf__sev_base.json``.

Outputs under ``results/agent_internals``:

=====================================  ==============================================
``trf_agent_confidences``              raw confidence per decision
``trf_agent_remapped_confidences``     confidence rescaled linearly to ``[0, 1]``
``trf_agent_normalised_performance``   normalised performance per sequence
``trf_agent_cumulative_performance``   running mean of the normalised performance
=====================================  ==============================================

Usage
-----
.. code-block:: powershell

    python -m general.scripts.agent_internals
    python -m general.scripts.agent_internals --train-date 2026-05-02
"""
##########################
##  Imports externos    ##
##########################
import argparse
import glob
import json
import os

from matplotlib import pyplot
import numpy

##########################
##  Imports internos    ##
##########################
from .benchmark import PACKAGE_GROUPS, REFERENCE_SCENARIO, RESULTS_ROOT, WORKSPACE_ROOT, cell_path
from .plotting import (MEAN_LINESTYLE, MEAN_LINEWIDTH, PUB_RC, REFERENCE_COLOURS,
                       model_colour, save_figure, style_axes)


###############
##  Constants
###############
MODEL = 'pes_trf'
FIGURES_DIR = os.path.join(RESULTS_ROOT, 'agent_internals')
UNSET_CONFIDENCE = -1.0   # train_transformer.py marks decisions with no resources left


###############
##  Data access
###############
def find_confidences(train_date: "str | None") -> str:
    """Return the ``confsrl_<date>.npy`` path of a ``pes_trf`` training run.

    Parameters
    ----------
    train_date : str or None
        ``YYYY-MM-DD`` of the run; the most recent run is used when ``None``.

    Returns
    -------
    str
        Absolute path of the confidence array.
    """
    inputs = os.path.join(WORKSPACE_ROOT, PACKAGE_GROUPS[MODEL], MODEL, 'inputs')
    pattern = f'{train_date}_TRF_TRAIN' if train_date else '*_TRF_TRAIN'
    candidates = sorted(glob.glob(os.path.join(inputs, pattern, 'confsrl_*.npy')))
    if not candidates:
        raise FileNotFoundError(f'No confsrl_*.npy under {os.path.join(inputs, pattern)}')
    return candidates[-1]


def load_confidences(path: str) -> "tuple[numpy.ndarray, numpy.ndarray]":
    """Load the raw confidences and their ``[0, 1]`` rescaling.

    Parameters
    ----------
    path : str
        ``confsrl_<date>.npy`` file.

    Returns
    -------
    raw : ndarray
        Confidence per decision, including the ``-1`` sentinel.
    remapped : ndarray
        Valid confidences rescaled linearly to ``[0, 1]`` (sentinels dropped).
    """
    raw = numpy.load(path).astype(numpy.float64)
    valid = raw[raw != UNSET_CONFIDENCE]
    remapped = (valid - valid.min()) / (valid.max() - valid.min())
    return raw, numpy.clip(remapped, 0.0, 1.0)


def load_performance() -> numpy.ndarray:
    """Return the per-sequence normalised performance of ``pes_trf`` at ``sev_base``."""
    with open(cell_path('individual', MODEL, REFERENCE_SCENARIO), 'r', encoding='utf-8') as handle:
        cell = json.load(handle)
    return numpy.asarray(cell['per_sequence_perf'], dtype=numpy.float64)


###############
##  Figures
###############
def _scatter(values: numpy.ndarray, ylabel: str, name: str, colour: str) -> None:
    figure, axis = pyplot.subplots(figsize=(9, 3.4))
    x = numpy.arange(len(values))
    axis.scatter(x, values, s=14, color=colour, alpha=0.75, linewidths=0)
    axis.axhline(float(values.mean()), color=colour, linestyle=MEAN_LINESTYLE,
                 linewidth=MEAN_LINEWIDTH, label=f'Media = {values.mean():.3f}')
    axis.set_xlabel('Decisión (trial de evaluación)')
    axis.set_ylabel(ylabel)
    axis.set_xlim(-5, len(values) + 5)
    axis.set_ylim(-0.05, 1.05)
    style_axes(axis)
    axis.legend(loc='upper left', frameon=False)
    save_figure(figure, os.path.join(FIGURES_DIR, name))


def plot_confidences(raw: numpy.ndarray, remapped: numpy.ndarray, colour: str) -> None:
    """Render the raw and rescaled confidence panels."""
    valid = numpy.where(raw == UNSET_CONFIDENCE, numpy.nan, raw)
    _scatter(valid[~numpy.isnan(valid)], 'Confianza (entropía de $Q$)',
             'trf_agent_confidences', colour)
    _scatter(remapped, 'Confianza reescalada a $[0,1]$',
             'trf_agent_remapped_confidences', colour)


def plot_performance(perf: numpy.ndarray, colour: str) -> None:
    """Render the per-sequence and running-mean performance panels."""
    x = numpy.arange(len(perf))

    figure, axis = pyplot.subplots(figsize=(9, 3.4))
    axis.plot(x, perf, color=colour, marker='s', markersize=3.5, linewidth=1.6, label=MODEL)
    axis.axhline(float(perf.mean()), color=colour, linestyle=MEAN_LINESTYLE,
                 linewidth=MEAN_LINEWIDTH, label=f'Media = {perf.mean():.3f}')
    axis.axhline(1.0, color=REFERENCE_COLOURS['best'], linewidth=0.9, alpha=0.7)
    axis.text(len(perf) - 0.5, 1.03, r'$S_{\mathrm{mejor}}$ (óptimo factible)',
              ha='right', va='bottom', fontsize=8, color=REFERENCE_COLOURS['best'])
    axis.set_xlabel('Secuencia de validación')
    axis.set_ylabel(r'Desempeño normalizado $\bar{r}$')
    axis.set_xlim(-1, len(perf))
    axis.set_ylim(0.0, 1.12)
    style_axes(axis)
    axis.legend(loc='lower left', frameon=False)
    save_figure(figure, os.path.join(FIGURES_DIR, 'trf_agent_normalised_performance'))

    running = numpy.cumsum(perf) / numpy.arange(1, len(perf) + 1)
    figure, axis = pyplot.subplots(figsize=(9, 3.4))
    axis.plot(x, running, color=colour, marker='^', markersize=3.5, linewidth=1.6, label=MODEL)
    axis.axhline(float(perf.mean()), color=colour, linestyle=MEAN_LINESTYLE,
                 linewidth=MEAN_LINEWIDTH, label=f'Media final = {perf.mean():.3f}')
    axis.set_xlabel('Secuencia de validación')
    axis.set_ylabel(r'Media acumulada de $\bar{r}$')
    axis.set_xlim(-1, len(perf))
    axis.set_ylim(0.5, 1.05)
    style_axes(axis)
    axis.legend(loc='lower right', frameon=False)
    save_figure(figure, os.path.join(FIGURES_DIR, 'trf_agent_cumulative_performance'))


###############
##  CLI
###############
def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description='Agent-internals figures of pes_trf.')
    parser.add_argument('--train-date', default=None,
                        help='YYYY-MM-DD of the pes_trf training run (default: latest)')
    args = parser.parse_args()

    confidences_path = find_confidences(args.train_date)
    raw, remapped = load_confidences(confidences_path)
    perf = load_performance()

    pyplot.rcParams.update(PUB_RC)
    colour = model_colour(MODEL)
    plot_confidences(raw, remapped, colour)
    plot_performance(perf, colour)

    print(f'Confidences : {confidences_path}  (n={len(raw)}, valid={int((raw != UNSET_CONFIDENCE).sum())})')
    print(f'Performance : {cell_path("individual", MODEL, REFERENCE_SCENARIO)}  '
          f'(n={len(perf)}, mean={perf.mean():.4f})')
    print(f'Figures     : {FIGURES_DIR}')


if __name__ == '__main__':
    main()
