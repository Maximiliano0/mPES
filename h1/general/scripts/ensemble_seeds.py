"""Seed variability of the ensembles: members taken from the retrained seeds of the individual models.

Each ensemble keeps its rule and its tuned parameters (``inputs/best_params.json`` or ``config/CONFIG.py``); only
the member models change. A combination fixes one training seed (1001-1010, from
``general/scripts/seed_variability.py``) for each member: DQN, recurrent DQN, Transformer and A2C (``pes_ens`` does
not use A2C). The same ten combinations are applied to the six ensembles:

* ``c01_mejores``: the best seed of every member (highest mean ``r̄`` over the reference and the five held-out
  replicas);
* ``c02_peores``: the worst seed of every member;
* ``c03_media``: the seed of every member closest to the mean of its ten seeds;
* ``c04_azar1`` ... ``c10_azar7``: one seed per member drawn at random (fixed generator seed), without repeating a
  combination.

Every (ensemble, combination) is evaluated on the reference (``sev_base``) and the five held-out replicas by
running the ensemble package exactly as ``general/scripts/benchmark.py`` does (scenario CSVs swapped into
``inputs/`` and restored), with ``MPES_MODEL_ROOT`` pointing to a folder with the member files of the combination.
The delivered ensemble (members from each package's ``inputs/``) is evaluated with the same code as a control:
it must reproduce the benchmark cells. Nothing is retrained.

Each ensemble swaps the CSVs of its own package, so different ensembles can run in parallel, but one ensemble must
not run in two processes at once. Results are resumable: a finished (ensemble, combination) is skipped.

Usage (from ``h1/``, with ``win_mpes_env`` activated)::

    python -m general.scripts.ensemble_seeds plan
    python -m general.scripts.ensemble_seeds run --ensembles pes_ens pes_ens_sprb
    python -m general.scripts.ensemble_seeds summary
"""

##########################
##  Imports externos    ##
##########################
import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy

##########################
##  Imports internos    ##
##########################
from . import benchmark
from .seed_variability import SCENARIOS, keep_awake

##########################
##  Configuración       ##
##########################
H1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
GENERAL = os.path.join(H1, 'general')
SEEDS_WORK = os.path.join(GENERAL, 'work', 'seeds')
WORK = os.path.join(GENERAL, 'work', 'ens_seeds')
RESULTS = os.path.join(GENERAL, 'results', 'ensemble')
PLAN_FILE = os.path.join(RESULTS, 'ensemble_seeds_plan.json')
SUMMARY_FILE = os.path.join(RESULTS, 'ensemble_seeds.json')
LOG_FILE = os.path.join(WORK, 'ensemble_seeds.log')
CELLS = os.path.join(RESULTS, 'cells')
ENSEMBLES = list(benchmark.ENSEMBLE_PACKAGE_GROUPS)
MEMBERS = {'dqn': ('pes_dqn', 'dqn_model.keras'), 'rdqn': ('pes_rdqn', 'rdqn_model.keras'),
           'trf': ('pes_trf', 'trf_model.keras'), 'a2c': ('pes_a2c', 'ac_actor.keras')}
SEEDS = list(range(1001, 1011))
RANDOM_COMBOS = 7
PLAN_SEED = 20261115
DELIVERED = 'entregado'
STRUCTURE = (8, 8)          # blocks x sequences of sev_base and the held-out replicas


###############
##  Helpers
###############
def log(message: str) -> None:
    """Print ``message`` with a timestamp and append it to the run log."""
    line = f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}'
    print(line, flush=True)
    os.makedirs(WORK, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as handle:
        handle.write(line + '\n')


def write_json(path: str, payload: dict) -> None:
    """Write ``payload`` atomically (parallel processes read these files)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    partial = path + '.tmp'
    with open(partial, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
    os.replace(partial, path)


def read_json(path: str) -> dict:
    """Read a UTF-8 JSON file."""
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def scenario_csvs(pkg: str, scenario: str) -> tuple:
    """Return the ``(initial_severity.csv, sequence_lengths.csv)`` the benchmark materialised for ``pkg``."""
    folder = os.path.join(benchmark.WORK_ROOT, pkg, 'scenarios', scenario)
    return os.path.join(folder, 'initial_severity.csv'), os.path.join(folder, 'sequence_lengths.csv')


def check_inputs(pkg: str) -> None:
    """Abort unless the package inputs are the reference CSVs and no swap was left half-done."""
    inputs = os.path.join(H1, benchmark.PACKAGE_GROUPS[pkg], pkg, 'inputs')
    for name, reference in zip(('initial_severity.csv', 'sequence_lengths.csv'), scenario_csvs(pkg, 'sev_base')):
        current = os.path.join(inputs, name)
        if os.path.exists(current + '.bench_stash'):
            raise RuntimeError(f'{current}.bench_stash existe: un cambio de CSV quedó a medias')
        ours = numpy.loadtxt(current, delimiter=',')
        ref = numpy.loadtxt(reference, delimiter=',')
        if ours.shape != ref.shape or not numpy.array_equal(ours, ref):
            raise RuntimeError(f'{current} no es la referencia: ¿hay un benchmark en curso?')


###############
##  Combinations
###############
def seed_scores() -> Dict[str, Dict[int, float]]:
    """Mean ``r̄`` over the reference and the held-out replicas of every retrained seed of every member."""
    scores: Dict[str, Dict[int, float]] = {}
    for member, (pkg, _file) in MEMBERS.items():
        scores[member] = {}
        for seed in SEEDS:
            medias = read_json(os.path.join(SEEDS_WORK, pkg, f'seed_{seed}', 'result.json'))['medias']
            scores[member][seed] = float(numpy.mean([medias[s] for s in SCENARIOS]))
    return scores


def build_plan() -> dict:
    """Build the ten member-seed combinations shared by every ensemble.

    Returns
    -------
    dict
        Criterion, per-seed scores and the list of combinations (``id``, ``tipo`` and ``semillas``).
    """
    scores = seed_scores()

    def pick(rule) -> Dict[str, int]:
        return {member: int(rule(values)) for member, values in scores.items()}

    combos = [
        {'id': 'c01_mejores', 'tipo': 'mejores', 'semillas': pick(lambda v: max(v, key=v.get))},
        {'id': 'c02_peores', 'tipo': 'peores', 'semillas': pick(lambda v: min(v, key=v.get))},
        {'id': 'c03_media', 'tipo': 'media',
         'semillas': pick(lambda v: min(v, key=lambda s: abs(v[s] - numpy.mean(list(v.values())))))},
    ]
    generator = numpy.random.default_rng(PLAN_SEED)
    while len(combos) < 3 + RANDOM_COMBOS:
        seeds = {member: int(generator.choice(SEEDS)) for member in MEMBERS}
        if all(seeds != combo['semillas'] for combo in combos):
            combos.append({'id': f'c{len(combos) + 1:02d}_azar{len(combos) - 2}', 'tipo': 'azar', 'semillas': seeds})
    return {'criterio': 'r̄ medio en la referencia y las 5 réplicas held-out de cada semilla',
            'semilla_del_sorteo': PLAN_SEED, 'puntajes': {m: {str(s): v for s, v in d.items()} for m, d in
                                                          scores.items()},
            'combinaciones': combos}


def load_plan(force: bool = False) -> dict:
    """Return the stored plan, building and saving it the first time (or when ``force``)."""
    if force or not os.path.isfile(PLAN_FILE):
        write_json(PLAN_FILE, build_plan())
    return read_json(PLAN_FILE)


def model_root(combo: dict) -> str:
    """Folder with the member files of ``combo`` (copied from the per-seed models on first use)."""
    folder = os.path.join(WORK, 'models', combo['id'])
    os.makedirs(folder, exist_ok=True)
    for member, (pkg, filename) in MEMBERS.items():
        target = os.path.join(folder, filename)
        if not os.path.isfile(target):
            shutil.copyfile(os.path.join(SEEDS_WORK, pkg, f'seed_{combo["semillas"][member]}', filename), target)
    return folder


###############
##  Evaluation
###############
def evaluate(pkg: str, combo_id: str, root: Optional[str], scenarios: Optional[List[str]] = None) -> dict:
    """Run one ensemble with the members in ``root`` (``None`` = delivered members) on every scenario.

    Parameters
    ----------
    pkg : str
        Ensemble package.
    combo_id : str
        Name of the combination (output folder).
    root : str or None
        ``MPES_MODEL_ROOT`` for the run; ``None`` keeps each package's ``inputs/``.
    scenarios : list of str, optional
        Scenarios to evaluate (default ``SCENARIOS``).

    Returns
    -------
    dict
        Per-sequence performance vector of every scenario.
    """
    per_sequence = {}
    for scenario in scenarios or SCENARIOS:
        csvs = scenario_csvs(pkg, scenario)
        if numpy.loadtxt(csvs[1], delimiter=',').size != STRUCTURE[0] * STRUCTURE[1]:
            raise RuntimeError(f'{scenario}: se esperaban {STRUCTURE[0]} x {STRUCTURE[1]} secuencias')
        outputs = os.path.join(WORK, 'outputs', pkg, combo_id, scenario)
        shutil.rmtree(outputs, ignore_errors=True)
        os.makedirs(outputs)
        env = {'MPES_OUTPUTS_PATH': outputs, 'MPES_NUM_BLOCKS': str(STRUCTURE[0]),
               'MPES_NUM_SEQUENCES': str(STRUCTURE[1]), 'VIRTUAL_ENV': os.environ.get('VIRTUAL_ENV', ''),
               'MPES_MODEL_ROOT': root or ''}
        code, values = benchmark.run_package(pkg, csvs, env, os.path.join(outputs, '_subprocess.log'), outputs)
        if code != 0 or len(values) != STRUCTURE[0] * STRUCTURE[1]:
            raise RuntimeError(f'{pkg} {combo_id} {scenario}: código {code}, {len(values)} secuencias '
                               f'(ver {outputs})')
        per_sequence[scenario] = [float(v) for v in values]
    return per_sequence


def run_ensemble(pkg: str, combo_ids: Optional[List[str]] = None) -> None:
    """Evaluate the delivered ensemble and every pending combination of ``pkg``, skipping finished ones."""
    plan = load_plan()
    combos = [{'id': DELIVERED, 'tipo': DELIVERED, 'semillas': None}] + plan['combinaciones']
    if combo_ids:
        combos = [combo for combo in combos if combo['id'] in combo_ids]
    check_inputs(pkg)
    for combo in combos:
        result_file = os.path.join(WORK, 'results', pkg, f'{combo["id"]}.json')
        if os.path.isfile(result_file):
            log(f'{pkg} {combo["id"]}: ya hecha, se omite')
            continue
        log(f'{pkg} {combo["id"]}: inicio {combo["semillas"] or "(miembros entregados)"}')
        start = time.perf_counter()
        root = None if combo['semillas'] is None else model_root(combo)
        per_sequence = evaluate(pkg, combo['id'], root)
        medias = {s: float(numpy.mean(v)) for s, v in per_sequence.items()}
        write_json(result_file, {'ensamble': pkg, 'combinacion': combo['id'], 'tipo': combo['tipo'],
                                 'semillas': combo['semillas'], 'segundos': time.perf_counter() - start,
                                 'medias': medias, 'por_secuencia': per_sequence})
        log(f'{pkg} {combo["id"]}: fin; referencia {medias["sev_base"]:.4f}, '
            f'held-out {numpy.mean([medias[s] for s in SCENARIOS[1:]]):.4f}')


###############
##  Summary
###############
def spread(values: List[float]) -> Dict[str, float]:
    """Mean, sample standard deviation, minimum and maximum (``nan`` when empty)."""
    if not values:
        nan = float('nan')
        return {'media': nan, 'desviacion': nan, 'minimo': nan, 'maximo': nan}
    array = numpy.asarray(values, dtype=float)
    return {'media': float(array.mean()), 'desviacion': float(array.std(ddof=1)) if array.size > 1 else 0.0,
            'minimo': float(array.min()), 'maximo': float(array.max())}


def delivered_check(pkg: str, per_sequence: dict) -> float:
    """Largest per-sequence difference between the delivered run and the benchmark cells (``nan`` if absent)."""
    worst = 0.0
    for scenario, values in per_sequence.items():
        cell = os.path.join(CELLS, f'{pkg}__{scenario}.json')
        if not os.path.isfile(cell):
            return float('nan')
        reference = numpy.asarray(read_json(cell)['per_sequence_perf'], dtype=float)
        worst = max(worst, float(numpy.max(numpy.abs(numpy.asarray(values) - reference))))
    return worst


def summarise() -> dict:
    """Collect every result into per-ensemble statistics and write ``results/ensemble/ensemble_seeds.json``."""
    plan = load_plan()
    summary: dict = {'escenarios': SCENARIOS, 'plan': plan, 'ensambles': {}}
    for pkg in ENSEMBLES:
        folder = os.path.join(WORK, 'results', pkg)
        if not os.path.isdir(folder):
            continue
        rows = {}
        for name in sorted(os.listdir(folder)):
            if name.endswith('.json'):
                result = read_json(os.path.join(folder, name))
                rows[result['combinacion']] = {
                    'tipo': result['tipo'], 'semillas': result['semillas'],
                    'referencia': result['medias']['sev_base'],
                    'heldout': float(numpy.mean([result['medias'][s] for s in SCENARIOS[1:]]))}
                if result['combinacion'] == DELIVERED:
                    rows[DELIVERED]['dif_max_con_benchmark'] = delivered_check(pkg, result['por_secuencia'])
        combos = [row for name, row in rows.items() if name != DELIVERED]
        summary['ensambles'][pkg] = {
            'entregado': rows.get(DELIVERED), 'combinaciones': {n: r for n, r in rows.items() if n != DELIVERED},
            'semillas_variadas': {'n': len(combos), 'referencia': spread([r['referencia'] for r in combos]),
                                  'heldout': spread([r['heldout'] for r in combos])}}
    write_json(SUMMARY_FILE, summary)
    return summary


###############
##  Main
###############
def main() -> None:
    """Build the plan, run the pending evaluations or rebuild the summary."""
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n', 1)[0])
    parser.add_argument('command', choices=('plan', 'run', 'summary'))
    parser.add_argument('--ensembles', nargs='+', choices=ENSEMBLES, default=ENSEMBLES)
    parser.add_argument('--combos', nargs='+', default=None, help='combination ids (default: all + entregado)')
    parser.add_argument('--force-plan', action='store_true', help='rebuild the combinations')
    args = parser.parse_args()
    plan = load_plan(force=args.force_plan)
    if args.command == 'plan':
        for combo in plan['combinaciones']:
            log(f'{combo["id"]:12s} {combo["semillas"]}')
        return
    if args.command == 'run':
        keep_awake(True)
        try:
            for pkg in args.ensembles:
                run_ensemble(pkg, args.combos)
                summarise()
        finally:
            keep_awake(False)
    for pkg, info in summarise()['ensambles'].items():
        varied = info['semillas_variadas']
        delivered = info['entregado'] or {}
        log(f"{pkg}: entregado ref {delivered.get('referencia', float('nan')):.4f} | combinaciones n={varied['n']} "
            f"ref {varied['referencia']['media']:.4f} ± {varied['referencia']['desviacion']:.4f}  held-out "
            f"{varied['heldout']['media']:.4f} ± {varied['heldout']['desviacion']:.4f}")


if __name__ == '__main__':
    sys.exit(main())
