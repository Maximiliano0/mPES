"""Variabilidad entre semillas frente a las diferencias entre modelos y ensambles (Apéndice E.3).

Lee los resúmenes de ``h1/general/scripts/seed_variability.py``
(``h1/general/results/individual/seed_variability.json``) y de
``h1/general/scripts/ensemble_seeds.py`` (``h1/general/results/ensemble/ensemble_seeds.json``) y responde la
pregunta de Bernardez: ¿las diferencias entre modelos superan la variabilidad que produce sólo cambiar la semilla
de entrenamiento?

* Modelos individuales: el entregado (elegido por Optuna, no es un sorteo insesgado) y las semillas nuevas (media,
  desvío, mínimo y máximo), en la referencia y en la media de las cinco réplicas *held-out*.
* Comparaciones entre modelos con las semillas nuevas: diferencia de medias en puntos de ``r̄`` y si los rangos se
  superponen.
* Ensambles: con la regla y los parámetros fijos, las diez combinaciones de semillas de sus miembros (media, desvío,
  rango), el entregado y la diferencia con el Transformer que contiene cada combinación.
* Las filas de las tablas LaTeX (decimales con coma) se imprimen al final.

Uso (desde la raíz del repositorio, después de ``seed_variability summary`` y ``ensemble_seeds summary``)::

    python writings/auxiliar/scripts/seed_comparison.py
"""

##########################
##  Imports externos    ##
##########################
import argparse
import json
import os

import numpy

##########################
##  Configuración       ##
##########################
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
RESULTS = os.path.join(ROOT, 'h1', 'general', 'results')
SEEDS_FILE = os.path.join(RESULTS, 'individual', 'seed_variability.json')
ENSEMBLES_FILE = os.path.join(RESULTS, 'ensemble', 'ensemble_seeds.json')
DEFAULT_OUTPUT = os.path.join(RESULTS, 'individual', 'seed_comparison.json')
NAMES = {'ql': 'Q-Learning', 'dql': 'Double Q-Learning', 'dqn': 'DQN', 'rdqn': 'DQN recurrente', 'a2c': 'A2C',
         'trf': 'Transformer'}
ENSEMBLE_NAMES = {'pes_ens': 'Ensamble ponderado', 'pes_ens_trf_guard': 'Compuerta del Transformer',
                  'pes_ens_consensus_prior': r'Consenso con \emph{prior}', 'pes_ens_accq': r"Voto por acci\'on",
                  'pes_ens_sprb': 'Voto suave', 'pes_ens_consensus': 'Consenso'}
PAIRS = (('trf', 'rdqn'), ('trf', 'dqn'), ('rdqn', 'dqn'), ('dqn', 'ql'), ('dql', 'ql'))
GROUPS = ('referencia', 'heldout')
DELIVERED = 'entregado'


###############
##  Helpers
###############
def load(path: str) -> dict:
    """Load a JSON file.

    Parameters
    ----------
    path : str
        File to read.

    Returns
    -------
    dict
        Parsed content.
    """
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def compare(high: dict, low: dict) -> dict:
    """Compare the retrained seeds of two models in one scenario group.

    Parameters
    ----------
    high, low : dict
        ``spread`` dictionaries (``media``, ``desviacion``, ``minimo``, ``maximo``) of the model expected to be
        better and of the other one.

    Returns
    -------
    dict
        Difference of means in points of ``r̄`` (x100) and whether the ranges overlap.
    """
    return {'diferencia_puntos': 100.0 * (high['media'] - low['media']),
            'rangos_superpuestos': bool(high['minimo'] <= low['maximo']),
            'peor_alto_menos_mejor_bajo_puntos': 100.0 * (high['minimo'] - low['maximo'])}


def latex_number(value: float) -> str:
    """Format ``value`` with three decimals and a decimal comma for the thesis (``$0{,}927$``)."""
    return f'${value:.3f}$'.replace('.', '{,}')


def latex_spread(stats: dict) -> list:
    """``mean $\\pm$ sd`` and ``min--max`` cells of one ``spread`` dictionary."""
    return [f"{latex_number(stats['media'])} $\\pm$ {latex_number(stats['desviacion'])}",
            f"{latex_number(stats['minimo'])}--{latex_number(stats['maximo'])}"]


def latex_row(key: str, info: dict) -> str:
    """One row of the individual seed-variability table.

    Parameters
    ----------
    key : str
        Key of ``NAMES``.
    info : dict
        Entry of ``seed_variability.json`` for the model.

    Returns
    -------
    str
        LaTeX row: model, number of new seeds, delivered (reference / held-out) and, per group, mean ± standard
        deviation and range of the new seeds.
    """
    delivered = info['entregado']
    new = info['semillas_nuevas']
    cells = [NAMES[key], str(new['n']), latex_number(delivered['referencia']), latex_number(delivered['heldout'])]
    for group in GROUPS:
        cells += latex_spread(new[group])
    return ' & '.join(cells) + r' \\'


def trf_seeds(models: dict) -> dict:
    """``r̄`` per group of every retrained Transformer seed, keyed by seed."""
    return {run['semilla']: run for run in models['trf']['corridas'] if run['semilla'] != 0}


def ensemble_entry(info: dict, trf: dict) -> dict:
    """Statistics of one ensemble over its member-seed combinations.

    Parameters
    ----------
    info : dict
        Entry of ``ensemble_seeds.json`` for the ensemble.
    trf : dict
        Output of :func:`trf_seeds`.

    Returns
    -------
    dict
        Delivered ``r̄``, spread of the combinations and, per group, the mean difference with the Transformer seed
        of each combination and how many combinations exceed it.
    """
    rows = list(info['combinaciones'].values())
    entry: dict = {'entregado': info[DELIVERED], 'n': len(rows)}
    for group in GROUPS:
        delta = numpy.asarray([row[group] - trf[row['semillas']['trf']][group] for row in rows])
        stats = info['semillas_variadas'][group]
        entry[group] = {**stats, 'entregado_menos_media_puntos': 100.0 * (info[DELIVERED][group] - stats['media']),
                        'menos_su_transformer_puntos': float(100.0 * delta.mean()),
                        'supera_a_su_transformer': int((delta > 0).sum())}
    return entry


def latex_ensemble_row(pkg: str, entry: dict) -> str:
    """One row of the ensemble table: name, delivered, spread per group and combinations above their Transformer."""
    cells = [ENSEMBLE_NAMES[pkg], latex_number(entry['entregado']['referencia']),
             latex_number(entry['entregado']['heldout'])]
    for group in GROUPS:
        cells += latex_spread(entry[group])
    cells.append(f"{entry['referencia']['supera_a_su_transformer']}/{entry['heldout']['supera_a_su_transformer']}")
    return ' & '.join(cells) + r' \\'


###############
##  Main
###############
def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0])
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='JSON file for the comparison')
    args = parser.parse_args()

    models = load(SEEDS_FILE)['modelos']
    payload: dict = {'modelos': {}, 'comparaciones': {}, 'ensambles': {}}

    for key, info in models.items():
        new = info['semillas_nuevas']
        payload['modelos'][key] = {
            'entregado': info['entregado'], 'n_semillas_nuevas': new['n'],
            **{group: {**new[group], 'entregado_menos_media_puntos':
                       100.0 * (info['entregado'][group] - new[group]['media']),
                       'entregado_supera_todas': bool(info['entregado'][group] > new[group]['maximo'])}
               for group in GROUPS}}
        print(f"{NAMES[key]:18s} n={new['n']}  " + '  '.join(
            f"{group}: entregado {info['entregado'][group]:.4f} | nuevas {new[group]['media']:.4f} "
            f"± {new[group]['desviacion']:.4f} [{new[group]['minimo']:.4f}, {new[group]['maximo']:.4f}]"
            for group in GROUPS))

    for high, low in PAIRS:
        if high in models and low in models:
            name = f'{high}_vs_{low}'
            payload['comparaciones'][name] = {
                group: compare(models[high]['semillas_nuevas'][group], models[low]['semillas_nuevas'][group])
                for group in GROUPS}
            print(f'{NAMES[high]} - {NAMES[low]}: {payload["comparaciones"][name]}')

    if os.path.isfile(ENSEMBLES_FILE) and 'trf' in models:
        trf = trf_seeds(models)
        for pkg, info in load(ENSEMBLES_FILE)['ensambles'].items():
            payload['ensambles'][pkg] = ensemble_entry(info, trf)
            print(f"{pkg:24s} " + '  '.join(
                f"{group}: entregado {info[DELIVERED][group]:.4f} | combinaciones "
                f"{payload['ensambles'][pkg][group]['media']:.4f} ± {payload['ensambles'][pkg][group]['desviacion']:.4f}"
                f" | - su TRF {payload['ensambles'][pkg][group]['menos_su_transformer_puntos']:+.2f} pts "
                f"({payload['ensambles'][pkg][group]['supera_a_su_transformer']}/10)" for group in GROUPS))

    print('\nFilas LaTeX, modelos (modelo & n & entregado ref & entregado held-out & ref media ± DE & ref rango '
          '& held-out media ± DE & held-out rango):')
    for key in NAMES:
        if key in models and models[key]['entregado'] and models[key]['semillas_nuevas']['n']:
            print(latex_row(key, models[key]))
    if payload['ensambles']:
        print('\nFilas LaTeX, ensambles (ensamble & entregado ref & entregado held-out & ref media ± DE & ref rango '
              '& held-out media ± DE & held-out rango & combinaciones que superan a su Transformer ref/held-out):')
        for pkg in ENSEMBLE_NAMES:
            if pkg in payload['ensambles']:
                print(latex_ensemble_row(pkg, payload['ensambles'][pkg]))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f'-> {os.path.relpath(args.output, ROOT)}')


if __name__ == '__main__':
    main()
