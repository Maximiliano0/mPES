"""Scenario taxonomy and CSV synthesisers for the mPES Under Stress Experiments.

A *scenario* is a fully specified perturbation of the two CSV inputs that
drive the Pandemic experiment plus optional structural overrides
(``num_blocks``, ``num_sequences_per_block``).

Each scenario is identified by a string ID (e.g. ``sev_weibull``) and is
described by a :class:`Scenario` dataclass exposing ``severity_fn`` and
``length_fn`` callables that, given an RNG, return the per-trial severity
vector and per-sequence length vector respectively.

The full scenario set is built lazily by :func:`build_scenarios` and
written to disk by :func:`materialise_scenario`.

The ``heldout`` family holds fresh i.i.d. draws from the empirical
distribution of the reference CSVs (the same distribution the individual
models sample during training). They measure how much the reference score
is inflated by tuning on those 64 fixed sequences, so they are excluded
from every generalisation aggregate (see :func:`is_heldout`). Each held-out
scenario uses its own seed offset and stores its sampling distribution in
``sampling_distribution.json`` next to the CSVs.

References
----------
- ``comparacion_modelos.md`` (n=64 baseline definitions).
- Severity range training distribution: integer in [0, 9] (``MAX_SEVERITY=9``).
- Length range training distribution: integer in [3, 10].
"""
##########################
##  Imports externos    ##
##########################
import hashlib
import json
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
SEV_MAX = 9                       # baseline severity upper bound
LEN_MIN = 3
LEN_MAX = 10                      # baseline length upper bound
DEFAULT_NUM_BLOCKS = 8
DEFAULT_NUM_SEQUENCES = 8         # per block; 8x8 = 64 sequences (matches comparacion_modelos.md)
DEFAULT_SEED = 42
HELDOUT_FAMILY = 'heldout'
HELDOUT_PREFIX = 'heldout_s'
HELDOUT_REPLICAS = 5              # heldout_s1 .. heldout_s5, seed = DEFAULT_SEED + k
SAMPLING_FILE = 'sampling_distribution.json'
H1_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
        High-level group: ``severity`` | ``length`` | ``joint`` | ``structural`` |
        ``heldout`` | ``baseline``.
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
        Marks the baseline reference cell used for stress-degradation
        and Welch / Cohen's d comparisons.
    extra : dict
        Optional metadata. ``seed_offset`` shifts the materialisation seed;
        ``sampling`` holds the empirical distribution written to
        ``sampling_distribution.json``.
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


def is_heldout(scenario_id: str) -> bool:
    """Return True for the held-out replicas of the reference distribution."""
    return scenario_id.startswith(HELDOUT_PREFIX)


###############
##  Empirical distribution
###############
def empirical_distribution(csv_path: str) -> dict:
    """Return the value frequencies of an integer CSV (the i.i.d. sampling table).

    Parameters
    ----------
    csv_path : str
        ``initial_severity.csv`` or ``sequence_lengths.csv`` of the reference package.

    Returns
    -------
    dict
        ``values``, ``counts`` and ``probabilities`` (``counts / n``) plus ``n``.
    """
    arr = numpy.loadtxt(csv_path, delimiter=',').astype(int).flatten()
    values, counts = numpy.unique(arr, return_counts=True)
    return {'values': values.tolist(), 'counts': counts.tolist(),
            'probabilities': (counts / counts.sum()).tolist(), 'n': int(counts.sum())}


def sha256_file(path: str) -> str:
    """Hex SHA-256 of a file (used to pin the source and drawn CSVs)."""
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _observed(values: numpy.ndarray) -> dict:
    """Frequencies actually drawn, to compare against the sampling table."""
    unique, counts = numpy.unique(values, return_counts=True)
    return {'values': unique.tolist(), 'counts': counts.tolist(),
            'frequencies': (counts / counts.sum()).tolist(), 'n': int(counts.sum())}


###############
##  Severity generators
###############
def _sev_base(empirical_path: str) -> Callable:
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


def _sev_empirical_resample(distribution: dict) -> Callable:
    """Draw each severity i.i.d. from the empirical frequencies of the reference CSV."""
    values = numpy.asarray(distribution['values'], dtype=int)
    probabilities = numpy.asarray(distribution['probabilities'], dtype=float)

    def _gen(rng: numpy.random.Generator, n_trials: int) -> numpy.ndarray:
        return rng.choice(values, size=n_trials, p=probabilities)
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
    """Under stress: severities ABOVE the training upper bound."""
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


def _len_empirical_resample(distribution: dict) -> Callable:
    """Draw each sequence length i.i.d. from the empirical frequencies of the reference CSV."""
    values = numpy.asarray(distribution['values'], dtype=int)
    probabilities = numpy.asarray(distribution['probabilities'], dtype=float)

    def _gen(rng: numpy.random.Generator, n_blocks: int, n_seq: int) -> numpy.ndarray:
        return rng.choice(values, size=(n_blocks, n_seq), p=probabilities)
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
    """Under stress: lengths LONGER than what the agent ever saw during training."""
    def _gen(rng, n_blocks, n_seq):
        return rng.integers(low=low, high=high + 1, size=(n_blocks, n_seq))
    return _gen


###############
##  Scenario catalogue
###############
def build_scenarios(empirical_severity_path: str,
                    empirical_lengths_path: str) -> "list[Scenario]":
    """Construct the canonical 27-scenario benchmark catalogue.

    The catalogue contains the baseline cell (``sev_base``), 21 stress
    scenarios (9 severity, 5 length, 4 joint and 3 structural) and 5
    held-out replicas of the baseline distribution, which are not part of
    the stress aggregates.

    Parameters
    ----------
    empirical_severity_path : str
        Path to a reference ``initial_severity.csv`` (used for the
        baseline and length-only sweeps).
    empirical_lengths_path : str
        Path to a reference ``sequence_lengths.csv``.

    Returns
    -------
    list[Scenario]
    """
    sev_emp = _sev_base(empirical_severity_path)
    len_emp = _len_empirical(empirical_lengths_path)

    scenarios: "list[Scenario]" = []

    # ---- A. Severity sweep (length = empirical) ----
    scenarios.append(Scenario(
        'sev_base', 'baseline',
        'Empirical training distribution (baseline).',
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
        'Weibull(k=1.5, scale ~4.99, unclipped mean 4.5) rounded and clipped to [0,9], long upper tail.',
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
    # well-defined; stress coverage is preserved by the gauss / weibull /
    # beta / bimodal / extrapolate scenarios.
    scenarios.append(Scenario(
        'sev_extrapolate_high', 'severity',
        'Under-stress severities U(10, 12) -- ABOVE training range.',
        _sev_extrapolate_high(), len_emp,
    ))

    # ---- B. Length sweep (severity = empirical) ----
    scenarios.append(Scenario(
        'len_all_short', 'length',
        'Every sequence has length 3 (minimum baseline length).',
        sev_emp, _len_constant(3),
    ))
    scenarios.append(Scenario(
        'len_all_long', 'length',
        'Every sequence has length 10 (maximum baseline length).',
        sev_emp, _len_constant(10),
    ))
    scenarios.append(Scenario(
        'len_geometric', 'length',
        'Lengths ~ 2 + Geometric(p=0.2) (support >= 3) clipped to [3, 10].',
        sev_emp, _len_geometric(0.2),
    ))
    scenarios.append(Scenario(
        'len_poisson', 'length',
        'Lengths ~ Poisson(lambda=5) clipped to [3, 10].',
        sev_emp, _len_poisson(5.0),
    ))
    scenarios.append(Scenario(
        'len_extrapolate_long', 'length',
        'Under-stress lengths U{11..20} -- LONGER than training.',
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
        'Uniform severity U(0, 9) x lengths 2 + Geometric(p=0.2) clipped to [3, 10].',
        _sev_uniform(), _len_geometric(0.2),
    ))
    scenarios.append(Scenario(
        'joint_extrap_both', 'joint',
        'Under-stress severity x under-stress length (full extrapolation).',
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

    # ---- E. Held-out replicas (not part of the stress aggregates) ----
    sampling = {
        'severity': {**empirical_distribution(empirical_severity_path),
                     'source': os.path.relpath(empirical_severity_path, H1_ROOT).replace(os.sep, '/'),
                     'source_sha256': sha256_file(empirical_severity_path)},
        'length': {**empirical_distribution(empirical_lengths_path),
                   'source': os.path.relpath(empirical_lengths_path, H1_ROOT).replace(os.sep, '/'),
                   'source_sha256': sha256_file(empirical_lengths_path)},
    }
    for replica in range(1, HELDOUT_REPLICAS + 1):
        scenarios.append(Scenario(
            f'{HELDOUT_PREFIX}{replica}', HELDOUT_FAMILY,
            f'Held-out replica {replica}: i.i.d. draws from the empirical severity and '
            f'length frequencies (seed offset {replica}).',
            _sev_empirical_resample(sampling['severity']),
            _len_empirical_resample(sampling['length']),
            extra={'seed_offset': replica, 'sampling': sampling},
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
    effective_seed = seed + int(scenario.extra.get('seed_offset', 0))
    rng = numpy.random.default_rng(effective_seed)

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

    if 'sampling' in scenario.extra:
        _write_sampling_record(scenario, target_dir, seed, effective_seed,
                               severity_1d, lengths_2d, sev_path, len_path)
    return sev_path, len_path


def _write_sampling_record(scenario: Scenario, target_dir: str, seed: int, effective_seed: int,
                           severity_1d: numpy.ndarray, lengths_2d: numpy.ndarray,
                           sev_path: str, len_path: str) -> str:
    """Write ``sampling_distribution.json``: the i.i.d. table, the seed and what was drawn."""
    record = {
        'scenario': scenario.scenario_id,
        'family': scenario.family,
        'description': scenario.description,
        'procedure': ('i.i.d. resampling: each sequence length is drawn independently from '
                      'sampling_distribution.length and each initial severity independently from '
                      'sampling_distribution.severity; severities do not depend on the length '
                      'or on the position in the sequence. Lengths are drawn first '
                      '(num_blocks x num_sequences_per_block), then one severity per trial.'),
        'rng': 'numpy.random.default_rng(effective_seed).choice(values, p=probabilities)',
        'base_seed': seed,
        'seed_offset': int(scenario.extra.get('seed_offset', 0)),
        'effective_seed': effective_seed,
        'structure': {'num_blocks': scenario.num_blocks,
                      'num_sequences_per_block': scenario.num_sequences_per_block,
                      'n_sequences': int(lengths_2d.size), 'n_trials': int(lengths_2d.sum())},
        'sampling_distribution': scenario.extra['sampling'],
        'observed': {'severity': _observed(severity_1d), 'length': _observed(lengths_2d)},
        'outputs_sha256': {'initial_severity.csv': sha256_file(sev_path),
                           'sequence_lengths.csv': sha256_file(len_path)},
    }
    path = os.path.join(target_dir, SAMPLING_FILE)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2)
    return path
