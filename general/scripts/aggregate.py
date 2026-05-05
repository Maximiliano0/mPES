"""Aggregate per-cell raw JSONs into model x scenario matrices and compute
the OOD metrics requested by the user (mean / std / min / max /
OOD-degradation / Welch t / Cohen's d / KL action drift).

Run after :mod:`general.scripts.orchestrate` has populated
``general/results/raw/``.

Outputs (under ``general/results/``):

* ``matrix_global_mean.csv``
* ``matrix_std.csv``
* ``matrix_min.csv``
* ``matrix_max.csv``
* ``matrix_ood_degradation.csv``
* ``matrix_welch_p.csv``
* ``matrix_cohen_d.csv``
* ``matrix_action_kl.csv``
* ``matrix_summary.json`` (machine-readable consolidation)
"""
##########################
##  Imports externos    ##
##########################
import csv
import glob
import json
import math
import os

import numpy

##########################
##  Imports internos    ##
##########################
from .runner import ALL_PACKAGES, GENERAL_ROOT, RAW_RESULTS_DIR
from .scenarios import build_scenarios
from .runner import _find_baseline_paths

RESULTS_DIR = os.path.join(GENERAL_ROOT, 'results')


###############
##  Statistics
###############
def _welch_t(x: numpy.ndarray, y: numpy.ndarray) -> "tuple[float, float, float]":
    """Two-sample Welch t-test.

    Returns
    -------
    t : float
        Welch t-statistic.
    p : float
        Two-sided p-value (may underflow to ``0.0`` for very large |t|).
    log10_p : float
        Base-10 logarithm of the two-sided p-value, computed via the
        survival function in log-space so that extreme tails do not
        underflow.  Useful for publication heatmaps where raw ``p``
        collapses to zero.
    """
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float('nan'), float('nan'), float('nan')
    mx, my = float(numpy.mean(x)), float(numpy.mean(y))
    vx, vy = float(numpy.var(x, ddof=1)), float(numpy.var(y, ddof=1))
    se = math.sqrt(vx / nx + vy / ny) if (vx + vy) > 0 else 0.0
    if se == 0.0:
        return float('nan'), float('nan'), float('nan')
    t = (mx - my) / se
    df_num = (vx / nx + vy / ny) ** 2
    df_den = (vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1)
    df = df_num / df_den if df_den > 0 else (nx + ny - 2)
    try:
        from scipy.stats import t as _student_t
        # Two-sided p via survival function; logsf is accurate in the deep tail.
        log_sf = float(_student_t.logsf(abs(t), df=df))  # natural log of one-sided p
        log10_p = (log_sf + math.log(2.0)) / math.log(10.0)
        p = 2.0 * float(_student_t.sf(abs(t), df=df))
    except Exception:  # pylint: disable=broad-except
        # Crude normal-tail fallback (less accurate at extreme |t|).
        p = math.erfc(abs(t) / math.sqrt(2.0))
        log10_p = math.log10(p) if p > 0 else float('-inf')
    return t, p, log10_p


def _cohen_d(x: numpy.ndarray, y: numpy.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float('nan')
    vx, vy = float(numpy.var(x, ddof=1)), float(numpy.var(y, ddof=1))
    pooled = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled == 0.0:
        return float('nan')
    return (float(numpy.mean(x)) - float(numpy.mean(y))) / pooled


def _kl(p: numpy.ndarray, q: numpy.ndarray, eps: float = 1e-9) -> float:
    p = numpy.asarray(p, dtype=float) + eps
    q = numpy.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(numpy.sum(p * numpy.log(p / q)))


###############
##  Loading
###############
def _load_all() -> dict:
    """Returns ``{model: {scenario: cell_dict}}``."""
    out: dict = {}
    for path in glob.glob(os.path.join(RAW_RESULTS_DIR, '*.json')):
        with open(path, 'r', encoding='utf-8') as f:
            cell = json.load(f)
        out.setdefault(cell['model'], {})[cell['scenario']] = cell
    return out


###############
##  Matrix writers
###############
def _write_matrix(path: str, models: list, scenarios: list, getter):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model'] + scenarios)
        for m in models:
            row = [m]
            for s in scenarios:
                row.append(getter(m, s))
            w.writerow(row)


###############
##  Main
###############
def aggregate(reference_pkg: str = 'pes_dqn') -> str:
    """Build matrices + summary JSON from raw per-cell results.

    Parameters
    ----------
    reference_pkg : str
        Package whose ``inputs/`` directory provides the empirical
        baseline CSVs used to instantiate the scenario catalogue.

    Returns
    -------
    str
        Absolute path of the written ``matrix_summary.json`` file.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = _load_all()
    if not data:
        raise RuntimeError(f'No raw cells found under {RAW_RESULTS_DIR}.')

    # Canonical column order = scenario catalogue order.
    sev_path, len_path = _find_baseline_paths(reference_pkg)
    catalogue = build_scenarios(sev_path, len_path)
    scenarios = [s.scenario_id for s in catalogue]
    baseline_id = next(s.scenario_id for s in catalogue if s.is_baseline)

    models = [m for m in ALL_PACKAGES if m in data]

    def cell(m, s):
        return data.get(m, {}).get(s, {})

    def perf_vec(m, s) -> numpy.ndarray:
        return numpy.asarray(cell(m, s).get('per_sequence_perf', []), dtype=float)

    def actions(m, s) -> numpy.ndarray:
        a = cell(m, s).get('action_distribution')
        return numpy.asarray(a, dtype=float) if a else numpy.array([])

    # Pre-compute baseline arrays per model.
    baseline_perf = {m: perf_vec(m, baseline_id) for m in models}
    baseline_actions = {m: actions(m, baseline_id) for m in models}

    def gm(m, s):
        v = cell(m, s).get('global_mean_perf')
        return f'{v:.6f}' if v is not None else ''

    def st(m, s):
        v = cell(m, s).get('std_perf')
        return f'{v:.6f}' if v is not None else ''

    def mn(m, s):
        v = cell(m, s).get('min_perf')
        return f'{v:.6f}' if v is not None else ''

    def mx(m, s):
        v = cell(m, s).get('max_perf')
        return f'{v:.6f}' if v is not None else ''

    def degr(m, s):
        b = cell(m, baseline_id).get('global_mean_perf')
        c = cell(m, s).get('global_mean_perf')
        if b is None or c is None:
            return ''
        return f'{(b - c):.6f}'

    # Cache log10(p) so we can write two CSVs (raw p + log10 p) without recomputing.
    _welch_cache: dict = {}

    def _welch_for(m, s):
        key = (m, s)
        if key in _welch_cache:
            return _welch_cache[key]
        x = perf_vec(m, s)
        y = baseline_perf.get(m, numpy.array([]))
        if x.size == 0 or y.size == 0 or s == baseline_id:
            res = (float('nan'), float('nan'), float('nan'))
        else:
            res = _welch_t(x, y)
        _welch_cache[key] = res
        return res

    def welch_p(m, s):
        _, p, _ = _welch_for(m, s)
        return f'{p:.6e}' if not math.isnan(p) else ''

    def welch_logp(m, s):
        _, _, lp = _welch_for(m, s)
        if math.isnan(lp) or math.isinf(lp):
            return ''
        return f'{lp:.6f}'

    def cohen(m, s):
        x = perf_vec(m, s)
        y = baseline_perf.get(m, numpy.array([]))
        if x.size == 0 or y.size == 0 or s == baseline_id:
            return ''
        d = _cohen_d(x, y)
        return f'{d:.6f}' if not math.isnan(d) else ''

    def kl(m, s):
        a = actions(m, s)
        b = baseline_actions.get(m, numpy.array([]))
        if a.size == 0 or b.size == 0 or s == baseline_id:
            return ''
        return f'{_kl(a, b):.6f}'

    _write_matrix(os.path.join(RESULTS_DIR, 'matrix_global_mean.csv'),     models, scenarios, gm)
    _write_matrix(os.path.join(RESULTS_DIR, 'matrix_std.csv'),             models, scenarios, st)
    _write_matrix(os.path.join(RESULTS_DIR, 'matrix_min.csv'),             models, scenarios, mn)
    _write_matrix(os.path.join(RESULTS_DIR, 'matrix_max.csv'),             models, scenarios, mx)
    _write_matrix(os.path.join(RESULTS_DIR, 'matrix_ood_degradation.csv'), models, scenarios, degr)
    _write_matrix(os.path.join(RESULTS_DIR, 'matrix_welch_p.csv'),         models, scenarios, welch_p)
    _write_matrix(os.path.join(RESULTS_DIR, 'matrix_welch_logp.csv'),      models, scenarios, welch_logp)
    _write_matrix(os.path.join(RESULTS_DIR, 'matrix_cohen_d.csv'),         models, scenarios, cohen)
    _write_matrix(os.path.join(RESULTS_DIR, 'matrix_action_kl.csv'),       models, scenarios, kl)

    # Machine-readable consolidation.
    summary = {
        'baseline_scenario': baseline_id,
        'models': models,
        'scenarios': scenarios,
        'cells': data,
    }
    summary_path = os.path.join(RESULTS_DIR, 'matrix_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    return summary_path


def _main():
    out = aggregate()
    print(f"[aggregate] wrote summary -> {out}")


if __name__ == '__main__':
    _main()
