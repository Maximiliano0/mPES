"""Variabilidad entre semillas frente a las diferencias entre modelos (Apéndice E.3).

Lee el resumen del arnés ``h1/general/scripts/seed_variability.py``
(``h1/general/results/individual/seed_variability.json``) y el ``r̄`` de los
modelos entregados (``h1/general/results/ensemble/policy_vs_rule.json``) y
responde la pregunta de Bernardez: ¿las diferencias entre modelos superan la
variabilidad que produce sólo cambiar la semilla de entrenamiento?

* Por modelo: el entregado (elegido por Optuna, no es un sorteo insesgado) y
  las semillas nuevas (media, desvío, mínimo y máximo), en la referencia y en
  la media de las cinco réplicas *held-out*.
* Comparaciones: Transformer frente a DQN y DQN frente a Q-Learning con las
  semillas nuevas (diferencia de medias en puntos de ``r̄`` y si los rangos se
  superponen), y el ensamble ponderado entregado frente al rango del
  Transformer.
* Las filas de la tabla LaTeX (decimales con coma) se imprimen al final.

Uso (desde la raíz del repositorio, después de ``seed_variability summary``)::

    python writings/auxiliar/scripts/seed_comparison.py
"""

##########################
##  Imports externos    ##
##########################
import argparse
import json
import os

##########################
##  Configuración       ##
##########################
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
RESULTS = os.path.join(ROOT, 'h1', 'general', 'results')
SEEDS_FILE = os.path.join(RESULTS, 'individual', 'seed_variability.json')
POLICY_FILE = os.path.join(RESULTS, 'ensemble', 'policy_vs_rule.json')
DEFAULT_OUTPUT = os.path.join(RESULTS, 'individual', 'seed_comparison.json')
NAMES = {'ql': 'Q-Learning', 'dqn': 'DQN', 'trf': 'Transformer'}
PAIRS = (('trf', 'dqn'), ('dqn', 'ql'))
GROUPS = ('referencia', 'heldout')
WEIGHTED = 'pes_ens'


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


def latex_row(key: str, info: dict) -> str:
    """One row of the seed-variability table.

    Parameters
    ----------
    key : str
        ``ql``, ``dqn`` or ``trf``.
    info : dict
        Entry of ``seed_variability.json`` for the model.

    Returns
    -------
    str
        LaTeX row: model, delivered (reference / held-out), number of new seeds and, per group,
        mean ± standard deviation and range of the new seeds.
    """
    delivered = info['entregado']
    new = info['semillas_nuevas']
    cells = [NAMES[key], latex_number(delivered['referencia']), latex_number(delivered['heldout']), str(new['n'])]
    for group in GROUPS:
        stats = new[group]
        cells.append(f"{latex_number(stats['media'])} $\\pm$ {latex_number(stats['desviacion'])}")
        cells.append(f"{latex_number(stats['minimo'])}--{latex_number(stats['maximo'])}")
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
    policy = load(POLICY_FILE)
    payload: dict = {'modelos': {}, 'comparaciones': {}, 'ensamble_ponderado': {}}

    for key, info in models.items():
        new = info['semillas_nuevas']
        payload['modelos'][key] = {
            'entregado': info['entregado'], 'n_semillas_nuevas': new['n'],
            **{group: {**new[group], 'entregado_menos_media_puntos':
                       100.0 * (info['entregado'][group] - new[group]['media']),
                       'entregado_supera_todas': bool(info['entregado'][group] > new[group]['maximo'])}
               for group in GROUPS}}
        print(f"{NAMES[key]:12s} n={new['n']}  " + '  '.join(
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

    if 'trf' in models:
        for group in GROUPS:
            weighted = float(policy[group][WEIGHTED]['rbar'])
            trf = models['trf']['semillas_nuevas'][group]
            payload['ensamble_ponderado'][group] = {
                'rbar': weighted, 'menos_media_trf_puntos': 100.0 * (weighted - trf['media']),
                'menos_maximo_trf_puntos': 100.0 * (weighted - trf['maximo']),
                'supera_todas_trf': bool(weighted > trf['maximo'])}
        print(f'Ensamble ponderado (entregado) frente al Transformer: {payload["ensamble_ponderado"]}')

    print('\nFilas LaTeX (modelo & entregado ref & entregado held-out & n & ref media ± DE & ref rango '
          '& held-out media ± DE & held-out rango):')
    for key in ('ql', 'dqn', 'trf'):
        if key in models and models[key]['entregado'] and models[key]['semillas_nuevas']['n']:
            print(latex_row(key, models[key]))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f'-> {os.path.relpath(args.output, ROOT)}')


if __name__ == '__main__':
    main()
