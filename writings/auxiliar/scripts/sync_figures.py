"""Sincroniza las figuras de la tesis con las que genera ``h1/``.

Copia a ``writings/02_Images/<carpeta>/`` cada figura que usa la tesis,
tomándola de los resultados del proyecto, y borra de ``02_Images`` todo lo que
no esté en la lista (copias viejas, duplicados y figuras que la tesis no usa).
El logo de la portada no se genera en ``h1/`` y se conserva tal cual.

Carpetas de ``02_Images``:

=================  ==========================================================
``frontpage``      logo de la portada
``baseline``       jugador aleatorio (``general.scripts.random_baseline``)
``per_model``      resultados por secuencia en la referencia (benchmark)
``individual``     mapas y curvas de los modelos individuales
``ensemble``       mapas y curvas de los ensambles
``agent_internals`` confianza del Transformer (``general.scripts.agent_internals``)
=================  ==========================================================

Uso (desde la raíz del repositorio)::

    python writings/auxiliar/scripts/sync_figures.py
"""

##########################
##  Imports externos    ##
##########################
import glob
import os
import re
import shutil

##########################
##  Configuración       ##
##########################
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
RESULTS = os.path.join(ROOT, 'h1', 'general', 'results')
WORK = os.path.join(ROOT, 'h1', 'general', 'work')
IMAGES = os.path.join(ROOT, 'writings', '02_Images')
CHAPTERS = os.path.join(ROOT, 'writings', '01_Chapters')
KEEP = {os.path.join('frontpage', 'LOGO-ITBA.jpg')}

PER_MODEL = {'PES_BASE': 'pes_base', 'PES_QL': 'pes_ql', 'PES_DQL': 'pes_dql', 'PES_DQN': 'pes_dqn',
             'PES_RDQN': 'pes_rdqn', 'PES_A2C': 'pes_a2c', 'PES_TRF': 'pes_trf', 'PES_ENS': 'pes_ens'}
SUITE_FIGURES = {'ind': ('individual', ['01_desempeno_por_escenario', '03_welch_logp_por_escenario',
                                         '04_kl_acciones_por_escenario', '05_curvas_por_familia',
                                         '13_pares_cohen_d']),
                 'ens': ('ensemble', ['01_desempeno_por_escenario', '03_welch_logp_por_escenario',
                                      '04_kl_acciones_por_escenario', '06_curvas_estresores_universales',
                                      '07_cohen_d_por_escenario', '12_pares_welch_logp', '13_pares_cohen_d'])}
RENAMES = {'ens_06_curvas_estresores_universales': 'ens_06_curvas_extrapolacion'}


###############
##  Helpers
###############
def latest(pattern: str) -> str:
    """Return the most recently modified file matching ``pattern``.

    Parameters
    ----------
    pattern : str
        Recursive glob pattern.

    Returns
    -------
    str
        Path of the newest match.
    """
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(f'No generated figure matches {pattern}')
    return max(matches, key=os.path.getmtime)


def figure_map() -> dict:
    """Map each thesis figure (``carpeta/nombre``) to the generated file it comes from.

    Returns
    -------
    dict
        ``{relative destination under 02_Images: absolute source path}``.
    """
    mapping = {os.path.join('baseline', name): os.path.join(RESULTS, 'baseline', name)
               for name in ('random_player_sequence_performance.png', 'random_player_normalised_performance.png')}
    for prefix, pkg in PER_MODEL.items():
        mapping[os.path.join('per_model', f'{prefix}_results.png')] = latest(
            os.path.join(WORK, pkg, 'outputs', 'sev_base', '**', f'{prefix}_results_*.png'))
    for tag, (suite, stems) in SUITE_FIGURES.items():
        for stem in stems:
            name = RENAMES.get(f'{tag}_{stem}', f'{tag}_{stem}')
            mapping[os.path.join(suite, f'{name}.png')] = os.path.join(RESULTS, suite, 'figures', f'{stem}.png')
    mapping[os.path.join('agent_internals', 'trf_agent_confidences.png')] = os.path.join(
        RESULTS, 'agent_internals', 'trf_agent_confidences.png')
    return mapping


def referenced_figures() -> set:
    """Return the file names used by ``\\includegraphics`` in the thesis chapters."""
    names = set()
    for tex in glob.glob(os.path.join(CHAPTERS, '*.tex')):
        with open(tex, encoding='utf-8') as handle:
            names.update(os.path.basename(m) for m in re.findall(
                r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', handle.read()))
    return names


###############
##  Main
###############
def main() -> None:
    """Copy the generated figures into ``02_Images`` and delete everything else."""
    mapping = figure_map()
    wanted = set(mapping) | KEEP
    for dest, source in sorted(mapping.items()):
        target = os.path.join(IMAGES, dest)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(source, target)
        print(f'copiada  {dest:55s} <- {os.path.relpath(source, ROOT)}')
    for path in sorted(glob.glob(os.path.join(IMAGES, '**', '*'), recursive=True)):
        rel = os.path.relpath(path, IMAGES)
        if os.path.isfile(path) and rel not in wanted:
            os.remove(path)
            print(f'borrada  {rel}')
    for folder in sorted(glob.glob(os.path.join(IMAGES, '**', ''), recursive=True), reverse=True):
        if folder.rstrip(os.sep) != IMAGES and not os.listdir(folder):
            os.rmdir(folder)
            print(f'borrada  {os.path.relpath(folder, IMAGES)}{os.sep}')
    available = {os.path.basename(p) for p in wanted}
    missing = sorted(referenced_figures() - available)
    unused = sorted(available - referenced_figures())
    print(f'\n{len(mapping)} figuras sincronizadas. Sin fuente: {missing or "ninguna"}. '
          f'Sin usar en la tesis: {unused or "ninguna"}.')


if __name__ == '__main__':
    main()
