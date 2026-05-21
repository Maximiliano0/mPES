"""Unified reward reconstruction and plotting utilities.

This module merges what used to be three separate scripts (reward
reconstruction + two plot variants). It exposes:

* :func:`reconstruct` — replay env dynamics from saved artefacts and
  return per-sequence reward statistics.
* :func:`plot_growth_vs_perf` — per-package figure
  ``reward_growth_vs_perf.png`` that answers two questions at once:

  1. *How does the cumulative reward grow as sequences are evaluated?*
     The raw cumulative reward (sum of per-step rewards across every
     sequence seen so far) is plotted on the left Y axis (negative,
     monotonically decreasing).
  2. *How does it differ from ``mean_perf``?*
     The cumulative normalised perf (``Σ perf_i``, ∈ ``[0, n]``) is
     plotted on the right Y axis. ``mean_perf`` is the slope of this
     curve divided by ``n``. Different colours, twin axes, same x axis
     make the difference of scale and shape obvious.

This module does NOT modify any training/evaluation code. It only reads
existing artefacts under each package's ``outputs/<latest>/`` directory
plus the benchmark raw JSONs in ``general/results/raw/``.

Pandemic environment recap
--------------------------
Per-step reward: :math:`r_t = -\\sum_i s_{i, t}`. Severity update:
:math:`s_{i, t+1} = \\max(0, \\beta s_{i, t} - \\alpha a_i)` with
:math:`\\alpha = \\text{PANDEMIC\\_PARAMETER} = 0.4` and
:math:`\\beta = 1 + \\alpha`.

Normalisation (1)
-----------------
``perf_i = (S_worst_i - S_final_i) / (S_worst_i - S_best_i) âˆˆ [0, 1]``.
Already stored as ``per_sequence_perf`` in
``general/results/raw/<pkg>__sev_empirical.json``.

CLI
---
::

    python -m utils.scripts.reward_plots
"""

##########################
##  Imports externos    ##
##########################
import csv
import glob
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy


##########################
##  Constantes          ##
##########################
PANDEMIC_PARAMETER = 0.4  # α (matches every package's CONFIG.PANDEMIC_PARAMETER)
ALPHA = PANDEMIC_PARAMETER
BETA = 1.0 + PANDEMIC_PARAMETER

PACKAGES: List[Tuple[str, str]] = [
    ('tabular', 'pes_base'),
    ('tabular', 'pes_ql'),
    ('tabular', 'pes_dql'),
    ('ml', 'pes_dqn'),
    ('ml', 'pes_rdqn'),
    ('ml', 'pes_a2c'),
    ('ml', 'pes_trf'),
    ('ml', 'pes_ens'),
]
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(REPO_ROOT, 'general', 'results', 'raw')


##########################
##  Tipos               ##
##########################
@dataclass
class SequenceReward:
    """Per-sequence reconstructed reward stats.

    ``cumulative_reward_worst`` is the cumulative reward obtained by the
    no-action policy (``a_t = 0`` for every trial) on the *same* initial
    severities, which is the theoretical lower bound for this sequence
    given the env dynamics. ``normalized_reward`` rescales the actual
    cumulative reward against this bound so that it lives in ``[0, 1]``:

    .. math::
        \\text{normalized\\_reward}_i = 1 - \\frac{R_i}{R_{\\text{worst}, i}}
        \\in [0, 1]

    (both numerator and denominator are negative; ``R_i = R_{worst}`` is
    a 0 score and ``R_i = 0`` would be the unattainable upper bound).
    """
    num_trials: int
    rewards: List[float]
    cumulative_reward: float
    cumulative_reward_worst: float
    mean_reward: float
    normalized_reward: float


##########################
##  ReconstrucciÃ³n      ##
##########################
def _read_responses(path: str) -> List[dict]:
    """Parse ``PES_*_responses_*.txt`` into a list of dicts."""
    rows: List[dict] = []
    with open(path, 'r', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        header = None
        for raw in reader:
            if not raw:
                continue
            stripped = [c.strip() for c in raw]
            if header is None:
                header = [h.lstrip('#').strip() for h in stripped]
                continue
            rows.append(dict(zip(header, stripped)))
    return rows


def _read_trials_per_sequence(movement_log_path: str) -> List[int]:
    """Return per-sequence trial counts in block-major order."""
    obj = numpy.load(movement_log_path, allow_pickle=True).item()
    counts: List[int] = []
    for block_idx in sorted(obj.keys()):
        block = obj[block_idx]
        for seq_idx in sorted(block.keys()):
            counts.append(len(block[seq_idx]))
    return counts


def _simulate_cum_reward(initial_severities: List[float],
                         actions: List[float]) -> Tuple[float, List[float]]:
    """Replay one sequence; return ``(cumulative_reward, per_step_rewards)``.

    Mirrors the env dynamics in ``ext/pandemic.py::step``: severities
    list starts with ``[initial_severities[0]]``; at step *t* the action
    is appended to resources, all present severities update with the
    formula, reward = -sum(severities); then if not done, append
    ``initial_severities[t + 1]`` as a new city.
    """
    n = len(initial_severities)
    severities: List[float] = [float(initial_severities[0])]
    resources: List[float] = []
    rewards: List[float] = []
    for t in range(n):
        resources.append(float(actions[t]))
        severities = [
            max(0.0, BETA * s - ALPHA * r)
            for s, r in zip(severities, resources)
        ]
        rewards.append(-float(numpy.sum(severities)))
        if t < n - 1:
            severities.append(float(initial_severities[t + 1]))
    return float(numpy.sum(rewards)), rewards


def _simulate_sequence(initial_severities: List[float],
                       actions: List[int]) -> SequenceReward:
    """Replay one sequence and return per-step + normalised statistics.

    The "worst" reference trajectory is the no-action policy on the same
    initial severities, used to map the cumulative reward into
    ``[0, 1]`` (see :class:`SequenceReward`).
    """
    n = len(initial_severities)
    cum, rewards = _simulate_cum_reward(initial_severities,
                                        [float(a) for a in actions])
    cum_worst, _ = _simulate_cum_reward(initial_severities, [0.0] * n)
    mean = cum / n if n > 0 else 0.0
    if cum_worst < 0.0:
        norm = 1.0 - cum / cum_worst
    else:
        norm = 1.0  # degenerate: all initial severities were zero
    return SequenceReward(num_trials=n, rewards=rewards,
                          cumulative_reward=cum,
                          cumulative_reward_worst=cum_worst,
                          mean_reward=mean,
                          normalized_reward=float(norm))


def reconstruct(responses_path: str,
                movement_log_path: str) -> List[SequenceReward]:
    """Reconstruct per-sequence reward statistics from saved files."""
    if not os.path.isfile(responses_path):
        raise FileNotFoundError(responses_path)
    if not os.path.isfile(movement_log_path):
        raise FileNotFoundError(movement_log_path)
    rows = _read_responses(responses_path)
    counts = _read_trials_per_sequence(movement_log_path)
    total = sum(counts)
    if total != len(rows):
        if len(rows) > total:
            rows = rows[:total]
        else:
            counts = counts[:max(1, len([c for c in counts if c <= len(rows)]))]
    out: List[SequenceReward] = []
    idx = 0
    for n in counts:
        chunk = rows[idx:idx + n]
        idx += n
        if len(chunk) != n:
            break
        try:
            init_sev = [float(r['InitialSeverity']) for r in chunk]
            actions = [int(float(r['Response'])) for r in chunk]
        except (KeyError, ValueError):
            continue
        out.append(_simulate_sequence(init_sev, actions))
    return out


def per_sequence_arrays(stats: List[SequenceReward]):
    """Return parallel numpy arrays.

    Returns
    -------
    tuple of numpy.ndarray
        ``(mean_reward, cum_reward, num_trials, normalized_reward,
        cum_reward_worst)``.
    """
    if not stats:
        z = numpy.empty(0)
        return (z, z, numpy.empty(0, dtype=int), z, z)
    mean_r = numpy.asarray([s.mean_reward for s in stats], dtype=numpy.float64)
    cum_r = numpy.asarray([s.cumulative_reward for s in stats], dtype=numpy.float64)
    nt = numpy.asarray([s.num_trials for s in stats], dtype=numpy.int64)
    norm_r = numpy.asarray([s.normalized_reward for s in stats], dtype=numpy.float64)
    cum_w = numpy.asarray([s.cumulative_reward_worst for s in stats], dtype=numpy.float64)
    return mean_r, cum_r, nt, norm_r, cum_w


def find_pair(output_dir: str) -> Optional[Tuple[str, str]]:
    """Return ``(responses_path, movement_log_path)`` if both exist."""
    resp = glob.glob(os.path.join(output_dir, 'PES_*_responses_*.txt'))
    mov = glob.glob(os.path.join(output_dir, 'PES_*_movement_log_*.npy'))
    if resp and mov:
        return resp[0], mov[0]
    return None


##########################
##  IO helpers          ##
##########################
def _latest_output_dir(pkg_path: str) -> Optional[str]:
    """Return the most recent ``<date>_<TYPE>_AGENT`` directory or None."""
    out_root = os.path.join(pkg_path, 'outputs')
    if not os.path.isdir(out_root):
        return None
    candidates = [
        os.path.join(out_root, d) for d in os.listdir(out_root)
        if os.path.isdir(os.path.join(out_root, d)) and '_AGENT' in d
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _load_per_sequence_perf(pkg: str) -> Optional[numpy.ndarray]:
    """Load benchmark ``sev_empirical`` per-sequence perf if available."""
    path = os.path.join(RAW_DIR, f'{pkg}__sev_empirical.json')
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        ps = data.get('per_sequence_perf')
        return numpy.asarray(ps, dtype=numpy.float64) if ps else None
    except (OSError, json.JSONDecodeError, ValueError):
        return None


##########################
##  Plot: norm vs perf  ##
##########################
def plot_norm_reward_vs_perf(pkg_group: str, pkg_name: str) -> Optional[str]:
    """Generate ``normalized_reward_vs_perf.png`` for one package.

    Compares two **per-sequence normalised** metrics, both living in
    ``[0, 1]``, so they share a single y-axis:

    * **normalized_reward**:math:`_i = 1 - R_i / R_{\\text{worst}, i}`,
      derived from the cumulative reward of sequence *i* relative to the
      no-action lower bound. Reflects *trajectory quality* across every
      step.
    * **perf**:math:`_i = (S_{\\text{worst}, i} - S_{\\text{final}, i})
      / (S_{\\text{worst}, i} - S_{\\text{best}, i})`, the value already
      stored in ``general/results/raw/<pkg>__sev_empirical.json``.
      Reflects *only the final summed severity*.

    The figure has two stacked panels (shared x-axis):

    * **Top — per sequence**: instantaneous ``normalized_reward_i`` (red)
      and ``perf_i`` (blue) markers, with their running means.
    * **Bottom — cumulative growth**: :math:`\\Sigma_{i \\leq k}` of
      each metric, against the ideal :math:`y = k` line.
    """
    pkg_path = os.path.join(REPO_ROOT, pkg_group, pkg_name)
    out_dir = _latest_output_dir(pkg_path)
    if out_dir is None:
        print(f'[skip] {pkg_name}: no output directory')
        return None

    pair = find_pair(out_dir)
    if pair is None:
        print(f'[skip] {pkg_name}: missing responses/movement_log files')
        return None
    try:
        stats = reconstruct(*pair)
    except (OSError, ValueError, KeyError) as exc:
        print(f'[skip] {pkg_name}: reconstruction failed ({exc})')
        return None
    _, _, n_trials, norm_r, _ = per_sequence_arrays(stats)
    if norm_r.size == 0:
        print(f'[skip] {pkg_name}: empty reconstruction')
        return None
    perf_full = _load_per_sequence_perf(pkg_name)

    n = norm_r.size
    x = numpy.arange(1, n + 1)
    running_norm = numpy.cumsum(norm_r) / x
    cum_norm = numpy.cumsum(norm_r)

    has_perf = perf_full is not None and perf_full.size >= n
    if has_perf:
        perf = perf_full[:n]
        running_perf = numpy.cumsum(perf) / x
        cum_perf = numpy.cumsum(perf)

    color_norm = '#d62728'  # red — normalized reward
    color_perf = '#1f77b4'  # blue — perf

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    # ----- Top: instantaneous values + running means -----
    ax_top.plot(x, norm_r, 'o', color=color_norm, alpha=0.45, markersize=4,
                label=r'normalized_reward$_i = 1 - R_i / R_{\mathrm{worst},i}$')
    ax_top.plot(x, running_norm, '-', color=color_norm, linewidth=2.0,
                label=f'Running mean -> mean_norm_reward = {norm_r.mean():.4f}')
    ax_top.axhline(norm_r.mean(), color=color_norm, linestyle=':',
                   linewidth=0.9)
    if has_perf:
        ax_top.plot(x, perf, 's', color=color_perf, alpha=0.45, markersize=4,
                    label=r'perf$_i$ (final-severity based)')
        ax_top.plot(x, running_perf, '-', color=color_perf, linewidth=2.0,
                    label=f'Running mean -> mean_perf = {perf.mean():.4f}')
        ax_top.axhline(perf.mean(), color=color_perf, linestyle=':',
                       linewidth=0.9)
    ax_top.set_ylim(-0.02, 1.05)
    ax_top.set_ylabel(r'Normalised metric  $\in [0, 1]$')
    ax_top.grid(True, alpha=0.3)
    ax_top.set_title('Per sequence: normalized_reward (red) vs. perf (blue)')
    ax_top.legend(loc='lower right', fontsize=9, framealpha=0.9)

    # ----- Bottom: cumulative growth (both in [0, k]) -----
    ax_bot.plot(x, cum_norm, '-', color=color_norm, linewidth=2.0,
                label=r'$\Sigma_{i \leq k}$ normalized_reward$_i$')
    if has_perf:
        ax_bot.plot(x, cum_perf, '-', color=color_perf, linewidth=2.0,
                    label=r'$\Sigma_{i \leq k}$ perf$_i$')
    ax_bot.plot(x, x, ':', color='#444444', linewidth=0.9, alpha=0.7,
                label=r'Ideal $y = k$ (perfect agent)')
    ax_bot.set_xlabel('Sequence index $k$')
    ax_bot.set_ylabel(r'Cumulative normalised metric')
    ax_bot.grid(True, alpha=0.3)
    ax_bot.set_title('Cumulative growth (slopes = running means above)')
    ax_bot.legend(loc='upper left', fontsize=9, framealpha=0.9)

    perf_txt = (f'mean_perf={perf.mean():.4f}'
                if has_perf else 'mean_perf=N/A')
    fig.suptitle(
        f'Normalised reward vs. mean_perf -- {pkg_group}/{pkg_name}\n'
        f'(n={n} sequences, total {int(n_trials.sum())} trials, '
        f'alpha={ALPHA}, beta={BETA}; '
        f'mean_norm_reward={norm_r.mean():.4f}, {perf_txt})',
        fontsize=11, y=1.0,
    )
    fig.tight_layout()
    out_path = os.path.join(out_dir, 'normalized_reward_vs_perf.png')
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[ok]   {pkg_name}: wrote '
          f'{os.path.relpath(out_path, REPO_ROOT)}  | '
          f'mean_norm_reward={norm_r.mean():.4f}  {perf_txt}')
    return out_path


##########################
##  Main / CLI          ##
##########################
def main() -> None:
    """Generate the normalized-reward-vs-perf figure for every package."""
    print('=' * 84)
    print('Normalised reward vs. mean_perf (per package)')
    print('=' * 84)
    n_ok = 0
    for group, name in PACKAGES:
        if plot_norm_reward_vs_perf(group, name):
            n_ok += 1
    print('-' * 84)
    print(f'Done: {n_ok} figure(s) written.')


if __name__ == '__main__':
    main()
