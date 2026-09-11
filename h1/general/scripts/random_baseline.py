"""Random-player baseline figures for the mPES benchmark.

Replays a uniformly random decision maker over the 64 empirical validation
sequences shared by every ``h1/`` package and renders two figures under
``results/baseline``:

=========================================  =========================================
``random_player_sequence_performance``     raw accumulated severity per sequence
``random_player_normalised_performance``   normalised performance per sequence
=========================================  =========================================

The environment dynamics, the zero-allocation reference (``S_peor``) and the
budget-constrained optimum (``S_mejor``) mirror ``ext/pandemic.py`` and
``src/exp_utils.py`` of the reference package; constants are read from its
``config/CONFIG.py`` without importing the package (no TensorFlow start-up).

Usage
-----
.. code-block:: powershell

    python -m general.scripts.random_baseline
    python -m general.scripts.random_baseline --pkg pes_dqn --seed 7
"""
##########################
##  Imports externos    ##
##########################
import argparse
import importlib.util
import os
from types import ModuleType

from matplotlib import pyplot
import numpy

##########################
##  Imports internos    ##
##########################
from .benchmark import PACKAGE_GROUPS, RESULTS_ROOT, WORKSPACE_ROOT, find_baseline_paths
from .plotting import (MEAN_LINESTYLE, MEAN_LINEWIDTH, PUB_RC, REFERENCE_COLOURS,
                       save_figure, style_axes)


###############
##  Constants
###############
REFERENCE_PACKAGE = 'pes_trf'
PREASSIGNED_RESOURCES = 9   # Pandemic.__init__: max_resources = AVAILABLE_RESOURCES_PER_SEQUENCE - 9
FIGURES_DIR = os.path.join(RESULTS_ROOT, 'baseline')


###############
##  Config
###############
def load_package_config(pkg: str) -> ModuleType:
    """Load ``config/CONFIG.py`` of ``pkg`` as a standalone module.

    Parameters
    ----------
    pkg : str
        Package name (``pes_trf``, ``pes_dqn`` …).

    Returns
    -------
    ModuleType
        Executed configuration module; the package ``__init__`` is not run.
    """
    path = os.path.join(WORKSPACE_ROOT, PACKAGE_GROUPS[pkg], pkg, 'config', 'CONFIG.py')
    spec = importlib.util.spec_from_file_location(f'{pkg}_CONFIG', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load configuration from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sequences(pkg: str) -> "list[numpy.ndarray]":
    """Return the initial severities of every validation sequence of ``pkg``.

    Parameters
    ----------
    pkg : str
        Package whose ``inputs/`` CSVs define the sequences.

    Returns
    -------
    list of ndarray
        One array of initial severities per sequence, in file order.
    """
    severity_path, lengths_path = find_baseline_paths(pkg)
    lengths = numpy.loadtxt(lengths_path, delimiter=',').astype(int)
    severities = numpy.loadtxt(severity_path, delimiter=',')
    bounds = numpy.concatenate(([0], numpy.cumsum(lengths)))
    return [severities[bounds[i]:bounds[i + 1]] for i in range(len(lengths))]


###############
##  Dynamics
###############
def evolve(severities: "list[float]", allocations: "list[int]",
           alpha: float, beta: float) -> "list[float]":
    """Apply one evolution step to every visible city.

    Parameters
    ----------
    severities : list of float
        Current severity of each visible city.
    allocations : list of int
        Resources allocated to each visible city.
    alpha, beta : float
        ``RESPONSE_MULTIPLIER`` and ``SEVERITY_MULTIPLIER`` of the package.

    Returns
    -------
    list of float
        Updated severities, clipped at zero (``get_updated_severity``).
    """
    return [max(0.0, beta * s - alpha * a) for s, a in zip(severities, allocations)]


def simulate_sequence(initial: numpy.ndarray, allocations: "list[int]",
                      alpha: float, beta: float) -> float:
    """Return the accumulated final severity of a sequence under ``allocations``.

    Cities join one per trial and every visible city evolves once per trial,
    exactly as in ``Pandemic.step``.

    Parameters
    ----------
    initial : ndarray
        Initial severity of each city in the sequence.
    allocations : list of int
        Resource allocation decided at each trial.
    alpha, beta : float
        Pandemic multipliers.

    Returns
    -------
    float
        Sum of the per-city severities after the last trial.
    """
    severities: list[float] = []
    resources: list[int] = []
    for s0, action in zip(initial, allocations):
        severities.append(float(s0))
        resources.append(int(action))
        severities = evolve(severities, resources, alpha, beta)
    return float(numpy.sum(severities))


def random_allocations(length: int, budget: int, max_alloc: int,
                       rng: numpy.random.Generator) -> "list[int]":
    """Draw the uniformly random policy of the baseline for one sequence.

    Parameters
    ----------
    length : int
        Number of trials in the sequence.
    budget : int
        Resources the agent controls in the sequence.
    max_alloc : int
        Per-trial allocation cap (``MAX_ALLOCATABLE_RESOURCES``).
    rng : numpy.random.Generator
        Seeded generator.

    Returns
    -------
    list of int
        Allocations after the budget clipping applied by ``Pandemic.step``.
    """
    available = budget
    allocations = []
    for _ in range(length):
        action = int(rng.integers(0, max_alloc + 1))
        if available - action <= 0:
            action = available
        available -= action
        allocations.append(action)
    return allocations


def best_feasible_severity(initial: numpy.ndarray, budget: int, max_alloc: int,
                           alpha: float, beta: float) -> float:
    """Minimum accumulated severity under the per-sequence budget (``S_mejor``).

    Bounded-knapsack dynamic programme over integer per-city allocations in
    ``[0, max_alloc]`` whose sum does not exceed ``budget``; mirrors
    ``_best_feasible_sequence_severity`` in ``src/exp_utils.py``.

    Parameters
    ----------
    initial : ndarray
        Initial severity of each city.
    budget : int
        Total resources available across the sequence.
    max_alloc : int
        Per-trial allocation cap.
    alpha, beta : float
        Pandemic multipliers.

    Returns
    -------
    float
        Optimal accumulated severity.
    """
    length = len(initial)

    def final_severity(s0: float, allocation: int, steps: int) -> float:
        s = s0
        for _ in range(steps):
            s = max(0.0, beta * s - alpha * allocation)
        return s

    per_city = [[final_severity(float(initial[c]), a, length - c) for a in range(max_alloc + 1)]
                for c in range(length)]
    inf = float('inf')
    dp = [inf] * (budget + 1)
    dp[0] = 0.0
    for c in range(length):
        new_dp = [inf] * (budget + 1)
        for b, base in enumerate(dp):
            if base == inf:
                continue
            for a in range(max_alloc + 1):
                if b + a > budget:
                    break
                cost = base + per_city[c][a]
                if cost < new_dp[b + a]:
                    new_dp[b + a] = cost
        dp = new_dp
    return min(dp)


def run_random_player(pkg: str, seed: int) -> "tuple[numpy.ndarray, numpy.ndarray]":
    """Simulate the random player over every validation sequence of ``pkg``.

    Parameters
    ----------
    pkg : str
        Reference package providing constants and validation CSVs.
    seed : int
        Seed of the random policy.

    Returns
    -------
    raw : ndarray
        Accumulated severity ``S_cruda`` per sequence.
    normalised : ndarray
        Normalised performance ``(S_peor - S_agente) / (S_peor - S_mejor)``.
    """
    config = load_package_config(pkg)
    alpha = float(config.PANDEMIC_PARAMETER)
    beta = 1.0 + alpha
    max_alloc = int(config.MAX_ALLOCATABLE_RESOURCES)
    min_alloc = int(config.MIN_ALLOCATABLE_RESOURCES)
    budget = int(config.AVAILABLE_RESOURCES_PER_SEQUENCE) - PREASSIGNED_RESOURCES
    rng = numpy.random.default_rng(seed)

    raw, normalised = [], []
    for initial in load_sequences(pkg):
        allocations = random_allocations(len(initial), budget, max_alloc, rng)
        agent = simulate_sequence(initial, allocations, alpha, beta)
        worst = simulate_sequence(initial, [min_alloc] * len(initial), alpha, beta)
        best = best_feasible_severity(initial, budget, max_alloc, alpha, beta)
        raw.append(agent)
        normalised.append((worst - agent) / (worst - best))
    return numpy.asarray(raw), numpy.asarray(normalised)


###############
##  Figures
###############
def _sequence_axis(axis, n_sequences: int) -> None:
    axis.set_xlabel('Secuencia de validación')
    axis.set_xlim(-1, n_sequences)
    style_axes(axis)


def _bounds(axis, n_sequences: int) -> None:
    """Draw the S_peor / S_mejor bounds of the normalised metric."""
    axis.axhline(0.0, color=REFERENCE_COLOURS['worst'], linewidth=0.9, alpha=0.7)
    axis.axhline(1.0, color=REFERENCE_COLOURS['best'], linewidth=0.9, alpha=0.7)
    axis.text(n_sequences - 0.5, 0.02, r'$S_{\mathrm{peor}}$ (sin asignación)',
              ha='right', va='bottom', fontsize=8, color=REFERENCE_COLOURS['worst'])
    axis.text(n_sequences - 0.5, 0.98, r'$S_{\mathrm{mejor}}$ (óptimo factible)',
              ha='right', va='top', fontsize=8, color=REFERENCE_COLOURS['best'])


def plot_raw(raw: numpy.ndarray, colour: str) -> None:
    """Render the accumulated-severity figure of the random player."""
    figure, axis = pyplot.subplots(figsize=(9, 4.2))
    x = numpy.arange(len(raw))
    axis.plot(x, raw, color=colour, marker='o', markersize=3.5, linewidth=1.6,
              label='Decisor aleatorio')
    axis.axhline(float(raw.mean()), color=colour, linestyle=MEAN_LINESTYLE,
                 linewidth=MEAN_LINEWIDTH, label=f'Media = {raw.mean():.1f}')
    axis.set_ylabel(r'Severidad final acumulada $S_{\mathrm{cruda}}$')
    axis.set_ylim(bottom=0)
    _sequence_axis(axis, len(raw))
    axis.legend(loc='upper left', frameon=False)
    save_figure(figure, os.path.join(FIGURES_DIR, 'random_player_sequence_performance'))


def plot_normalised(normalised: numpy.ndarray, colour: str) -> None:
    """Render the normalised-performance figure of the random player."""
    figure, axis = pyplot.subplots(figsize=(9, 4.2))
    x = numpy.arange(len(normalised))
    axis.plot(x, normalised, color=colour, marker='s', markersize=3.5, linewidth=1.6,
              label='Decisor aleatorio')
    axis.axhline(float(normalised.mean()), color=colour, linestyle=MEAN_LINESTYLE,
                 linewidth=MEAN_LINEWIDTH, label=f'Media = {normalised.mean():.3f}')
    _bounds(axis, len(normalised))
    axis.set_ylabel(r'Desempeño normalizado $\bar{r}$')
    axis.set_ylim(-0.05, 1.05)
    _sequence_axis(axis, len(normalised))
    axis.legend(loc='lower left', bbox_to_anchor=(0.0, 0.07), frameon=False)
    save_figure(figure, os.path.join(FIGURES_DIR, 'random_player_normalised_performance'))


###############
##  CLI
###############
def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description='Random-player baseline figures for the mPES benchmark.')
    parser.add_argument('--pkg', default=REFERENCE_PACKAGE, choices=sorted(PACKAGE_GROUPS),
                        help='package providing CONFIG constants and validation CSVs')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed of the random policy (default: SEED of the package CONFIG)')
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else int(load_package_config(args.pkg).SEED)
    raw, normalised = run_random_player(args.pkg, seed)

    pyplot.rcParams.update(PUB_RC)
    colour = REFERENCE_COLOURS['random']
    plot_raw(raw, colour)
    plot_normalised(normalised, colour)

    print(f'Reference package : {args.pkg}  (seed={seed}, n={len(raw)})')
    print(f'S_cruda           : mean={raw.mean():.2f}  std={raw.std(ddof=1):.2f}  '
          f'min={raw.min():.2f}  max={raw.max():.2f}')
    print(f'Normalised r      : mean={normalised.mean():.4f}  std={normalised.std(ddof=1):.4f}  '
          f'min={normalised.min():.4f}  max={normalised.max():.4f}')
    print(f'Figures           : {FIGURES_DIR}')


if __name__ == '__main__':
    main()
