"""Scenario taxonomy and CSV synthesisers for the mPES OOD benchmark.

A *scenario* is a fully specified perturbation of the two CSV inputs that
drive the Pandemic experiment plus optional structural overrides
(``num_blocks``, ``num_sequences_per_block``).

Each scenario is identified by a string ID (e.g. ``sev_weibull``) and is
described by a :class:`Scenario` dataclass exposing ``severity_fn`` and
``length_fn`` callables that, given an RNG, return the per-trial severity
vector and per-sequence length vector respectively.

The full scenario set is built lazily by :func:`build_scenarios` and
written to disk by :func:`materialise_scenario`.

References
----------
- ``comparacion_modelos.md`` (n=64 baseline, in-distribution definitions).
- Severity range training distribution: integer in [0, 9] (``MAX_SEVERITY=9``).
- Length range training distribution: integer in [3, 10].
"""
##########################
##  Imports externos    ##
##########################
import os
from dataclasses import dataclass, field
from typing import Callable

import numpy

##########################
##  Imports internos    ##
##########################
# (none -- this module is dependency-free w.r.t. mPES packages)


###############
##  Constants
###############
SEV_MIN = 0
SEV_MAX = 9                       # in-distribution severity upper bound
LEN_MIN = 3
LEN_MAX = 10                      # in-distribution length upper bound
DEFAULT_NUM_BLOCKS = 8
DEFAULT_NUM_SEQUENCES = 8         # per block; 8x8 = 64 sequences (matches comparacion_modelos.md)
DEFAULT_SEED = 42


###############
##  Dataclass
###############
@dataclass
class Scenario:
    """One benchmark cell.

    Parameters
    ----------
    scenario_id : str
        Stable filesystem-safe identifier.
    family : str
        High-level group: ``severity`` | ``length`` | ``joint`` | ``structural`` | ``baseline``.
    description : str
        Human-readable summary used in plots and report tables.
    severity_fn : Callable[[numpy.random.Generator, int], numpy.ndarray]
        Produces the per-trial initial-severity vector of length
        ``total_trials`` (sum of the length array).
    length_fn : Callable[[numpy.random.Generator, int, int], numpy.ndarray]
        Produces the per-sequence length vector of shape
        ``(num_blocks, num_sequences_per_block)`` with integer entries.
    num_blocks : int
        Override for the experiment's block count.
    num_sequences_per_block : int
        Override for sequences per block.
    is_baseline : bool
        Marks the in-distribution reference cell used for OOD-degradation
        and Welch / Cohen's d comparisons.
    """

    scenario_id: str
    family: str
    description: str
    severity_fn: Callable[[numpy.random.Generator, int], numpy.ndarray]
    length_fn:   Callable[[numpy.random.Generator, int, int], numpy.ndarray]
    num_blocks: int = DEFAULT_NUM_BLOCKS
    num_sequences_per_block: int = DEFAULT_NUM_SEQUENCES
    is_baseline: bool = False
    extra: dict = field(default_factory=dict)


###############
##  Severity generators
###############
def _sev_empirical(empirical_path: str) -> Callable:
    """Return a generator that re-uses an existing per-package
    ``initial_severity.csv`` file (the model's training distribution)."""
    def _gen(_rng: numpy.random.Generator, n_trials: int) -> numpy.ndarray:
        arr = numpy.loadtxt(empirical_path, delimiter=',').astype(int)
        if arr.size < n_trials:
            # Tile if the empirical sample is shorter than required.
            reps = int(numpy.ceil(n_trials / arr.size))
            arr = numpy.tile(arr, reps)
        return arr[:n_trials]
    return _gen


def _sev_uniform(low: int = SEV_MIN, high: int = SEV_MAX) -> Callable:
    def _gen(rng, n):
        return rng.integers(low=low, high=high + 1, size=n)
    return _gen


def _sev_truncgauss(mean: float, std: float, low: int = SEV_MIN, high: int = SEV_MAX) -> Callable:
    def _gen(rng, n):
        out = numpy.empty(n, dtype=int)
        i = 0
        while i < n:
            sample = rng.normal(mean, std, size=n * 2)
            sample = sample[(sample >= low) & (sample <= high)]
            take = min(len(sample), n - i)
            out[i:i + take] = numpy.round(sample[:take]).astype(int)
            i += take
        return out
    return _gen


def _sev_weibull(k: float = 1.5, target_mean: float = 4.5,
                 low: int = SEV_MIN, high: int = SEV_MAX) -> Callable:
    # scale chosen so E[clipped] is approximately target_mean
    try:
        from math import gamma as _gamma
        scale = target_mean / _gamma(1 + 1 / k)
    except Exception:  # pylint: disable=broad-except
        scale = target_mean

    def _gen(rng, n):
        raw = rng.weibull(k, size=n) * scale
        clipped = numpy.clip(numpy.round(raw), low, high).astype(int)
        return clipped
    return _gen


def _sev_beta(alpha: float, beta: float, low: int = SEV_MIN, high: int = SEV_MAX) -> Callable:
    def _gen(rng, n):
        raw = rng.beta(alpha, beta, size=n) * (high - low) + low
        return numpy.clip(numpy.round(raw), low, high).astype(int)
    return _gen


def _sev_bimodal(m1: float, s1: float, m2: float, s2: float, mix: float = 0.5,
                 low: int = SEV_MIN, high: int = SEV_MAX) -> Callable:
    def _gen(rng, n):
        choose = rng.random(n) < mix
        out = numpy.where(choose, rng.normal(m1, s1, size=n), rng.normal(m2, s2, size=n))
        return numpy.clip(numpy.round(out), low, high).astype(int)
    return _gen


def _sev_extrapolate_high(low: int = SEV_MAX + 1, high: int = 12) -> Callable:
    """OOD: severities ABOVE the training upper bound."""
    def _gen(rng, n):
        return rng.integers(low=low, high=high + 1, size=n)
    return _gen


###############
##  Length generators
###############
def _len_empirical(empirical_path: str) -> Callable:
    def _gen(_rng, n_blocks: int, n_seq: int) -> numpy.ndarray:
        arr = numpy.loadtxt(empirical_path, delimiter=',').astype(int)
        flat = arr.flatten()
        need = n_blocks * n_seq
        if flat.size < need:
            reps = int(numpy.ceil(need / flat.size))
            flat = numpy.tile(flat, reps)
        return flat[:need].reshape(n_blocks, n_seq)
    return _gen


def _len_constant(value: int) -> Callable:
    def _gen(_rng, n_blocks, n_seq):
        return numpy.full((n_blocks, n_seq), value, dtype=int)
    return _gen


def _len_geometric(p: float = 0.2, low: int = LEN_MIN, high: int = LEN_MAX) -> Callable:
    def _gen(rng, n_blocks, n_seq):
        raw = rng.geometric(p, size=(n_blocks, n_seq)) + (low - 1)
        return numpy.clip(raw, low, high).astype(int)
    return _gen


def _len_poisson(lam: float = 5.0, low: int = LEN_MIN, high: int = LEN_MAX) -> Callable:
    def _gen(rng, n_blocks, n_seq):
        raw = rng.poisson(lam, size=(n_blocks, n_seq))
        return numpy.clip(raw, low, high).astype(int)
    return _gen


def _len_extrapolate_long(low: int = LEN_MAX + 1, high: int = 20) -> Callable:
    """OOD: lengths LONGER than what the agent ever saw during training."""
    def _gen(rng, n_blocks, n_seq):
        return rng.integers(low=low, high=high + 1, size=(n_blocks, n_seq))
    return _gen


###############
##  Scenario catalogue
###############
def build_scenarios(empirical_severity_path: str,
                    empirical_lengths_path: str) -> "list[Scenario]":
    """Construct the canonical 24-scenario benchmark catalogue.

    Parameters
    ----------
    empirical_severity_path : str
        Path to a reference ``initial_severity.csv`` (used for the
        in-distribution baseline and length-only sweeps).
    empirical_lengths_path : str
        Path to a reference ``sequence_lengths.csv``.

    Returns
    -------
    list[Scenario]
    """
    sev_emp = _sev_empirical(empirical_severity_path)
    len_emp = _len_empirical(empirical_lengths_path)

    scenarios: "list[Scenario]" = []

    # ---- A. Severity sweep (length = empirical) ----
    scenarios.append(Scenario(
        'sev_empirical', 'baseline',
        'Empirical training distribution (in-distribution baseline).',
        sev_emp, len_emp, is_baseline=True,
    ))
    scenarios.append(Scenario(
        'sev_uniform', 'severity',
        'Uniform U(0, 9) severity.',
        _sev_uniform(), len_emp,
    ))
    scenarios.append(Scenario(
        'sev_gauss_low', 'severity',
        'Truncated Gaussian N(2, 1.5) -- easy regime.',
        _sev_truncgauss(2.0, 1.5), len_emp,
    ))
    scenarios.append(Scenario(
        'sev_gauss_mid', 'severity',
        'Truncated Gaussian N(4.5, 2.0) -- matched mean.',
        _sev_truncgauss(4.5, 2.0), len_emp,
    ))
    scenarios.append(Scenario(
        'sev_gauss_high', 'severity',
        'Truncated Gaussian N(7, 1.5) -- hard regime.',
        _sev_truncgauss(7.0, 1.5), len_emp,
    ))
    scenarios.append(Scenario(
        'sev_weibull', 'severity',
        'Weibull(k=1.5) clipped to [0,9], heavy upper tail.',
        _sev_weibull(1.5, 4.5), len_emp,
    ))
    scenarios.append(Scenario(
        'sev_beta_lowskew', 'severity',
        'Beta(2, 5) * 9 -- skewed low.',
        _sev_beta(2.0, 5.0), len_emp,
    ))
    scenarios.append(Scenario(
        'sev_beta_highskew', 'severity',
        'Beta(5, 2) * 9 -- skewed high.',
        _sev_beta(5.0, 2.0), len_emp,
    ))
    scenarios.append(Scenario(
        'sev_bimodal', 'severity',
        'Bimodal mixture 0.5 N(2,1) + 0.5 N(7,1).',
        _sev_bimodal(2.0, 1.0, 7.0, 1.0), len_emp,
    ))
    # Adversarial constant / ramp scenarios were intentionally dropped:
    # they produce degenerate per-sequence severity vectors where
    # WorstCaseSeverity == BestCaseSeverity, which makes the
    # normalised-final-severity performance metric (in every package's
    # exp_utils.py) divide by zero.  Excluding them keeps the matrices
    # well-defined; OOD coverage is preserved by the gauss / weibull /
    # beta / bimodal / extrapolate scenarios.
    scenarios.append(Scenario(
        'sev_extrapolate_high', 'severity',
        'OOD severities U(10, 12) -- ABOVE training range.',
        _sev_extrapolate_high(), len_emp,
    ))

    # ---- B. Length sweep (severity = empirical) ----
    scenarios.append(Scenario(
        'len_all_short', 'length',
        'Every sequence has length 3 (minimum in-distribution).',
        sev_emp, _len_constant(3),
    ))
    scenarios.append(Scenario(
        'len_all_long', 'length',
        'Every sequence has length 10 (maximum in-distribution).',
        sev_emp, _len_constant(10),
    ))
    scenarios.append(Scenario(
        'len_geometric', 'length',
        'Lengths ~ Geometric(p=0.2) clipped to [3, 10].',
        sev_emp, _len_geometric(0.2),
    ))
    scenarios.append(Scenario(
        'len_poisson', 'length',
        'Lengths ~ Poisson(lambda=5) clipped to [3, 10].',
        sev_emp, _len_poisson(5.0),
    ))
    scenarios.append(Scenario(
        'len_extrapolate_long', 'length',
        'OOD lengths U{11..20} -- LONGER than training.',
        sev_emp, _len_extrapolate_long(),
    ))

    # ---- C. Joint stress ----
    scenarios.append(Scenario(
        'joint_high_long', 'joint',
        'High severity Gauss(7,1.5) x all-long (length 10).',
        _sev_truncgauss(7.0, 1.5), _len_constant(10),
    ))
    scenarios.append(Scenario(
        'joint_low_short', 'joint',
        'Low severity Gauss(2,1.5) x all-short (length 3).',
        _sev_truncgauss(2.0, 1.5), _len_constant(3),
    ))
    scenarios.append(Scenario(
        'joint_uniform_geom', 'joint',
        'Uniform severity x geometric lengths.',
        _sev_uniform(), _len_geometric(0.2),
    ))
    scenarios.append(Scenario(
        'joint_extrap_both', 'joint',
        'OOD severity x OOD length (full extrapolation).',
        _sev_extrapolate_high(), _len_extrapolate_long(),
    ))
    # joint_adv9_long dropped: constant-severity sequences make the
    # performance metric undefined (see comment above sev_extrapolate_high).

    # ---- D. Structural sweep ----
    scenarios.append(Scenario(
        'struct_few_long_blocks', 'structural',
        '4 blocks x 16 sequences = 64 (fewer, larger blocks).',
        sev_emp, len_emp, num_blocks=4, num_sequences_per_block=16,
    ))
    scenarios.append(Scenario(
        'struct_many_short_blocks', 'structural',
        '16 blocks x 4 sequences = 64 (more, smaller blocks).',
        sev_emp, len_emp, num_blocks=16, num_sequences_per_block=4,
    ))
    scenarios.append(Scenario(
        'struct_more_total', 'structural',
        '8 blocks x 16 sequences = 128 (double sample size).',
        sev_emp, len_emp, num_blocks=8, num_sequences_per_block=16,
    ))

    return scenarios


###############
##  Materialisation
###############
def materialise_scenario(scenario: Scenario, target_dir: str,
                         seed: int = DEFAULT_SEED) -> "tuple[str, str]":
    """Generate scenario CSVs into ``target_dir``.

    Parameters
    ----------
    scenario : Scenario
    target_dir : str
        Directory in which ``initial_severity.csv`` and
        ``sequence_lengths.csv`` will be written.
    seed : int
        RNG seed (single-seed protocol -- see ``comparacion_modelos.md``).

    Returns
    -------
    tuple[str, str]
        (severity_csv_path, lengths_csv_path)
    """
    os.makedirs(target_dir, exist_ok=True)
    rng = numpy.random.default_rng(seed)

    lengths_2d = scenario.length_fn(rng,
                                    scenario.num_blocks,
                                    scenario.num_sequences_per_block)
    n_trials = int(lengths_2d.sum())
    severity_1d = scenario.severity_fn(rng, n_trials)

    sev_path = os.path.join(target_dir, 'initial_severity.csv')
    len_path = os.path.join(target_dir, 'sequence_lengths.csv')

    numpy.savetxt(sev_path, severity_1d, fmt='%d', delimiter=',')
    # The package loaders expect a flat 1D vector indexed by global sequence
    # position (see ``exp_utils.next_seq_length``), so we flatten before save.
    numpy.savetxt(len_path, lengths_2d.flatten(), fmt='%d', delimiter=',')

    return sev_path, len_path
