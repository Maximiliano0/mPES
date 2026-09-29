"""Genera ``mpes_resultados.md`` y ``mpes_resultados.json`` (contexto numérico de la tesis).

Reconstruye los dos archivos de ``writings/auxiliar/claude_context/`` a partir
de las fuentes del repositorio, para poder regenerarlos después de cualquier
re-ejecución del benchmark:

* ``h1/general/results/{individual,ensemble}/``: ``summary.json``,
  ``comparison_metrics.json`` y ``matrices/*.csv`` (27 escenarios:
  ``sev_base``, 21 de generalización y 5 réplicas ``heldout_s*``; los
  agregados de "21 escenarios" excluyen las réplicas).
* ``h1/general/results/heldout/``: ``heldout_gap.json`` y
  ``heldout_catalogue.json``.
* Decisiones de los ensambles (§5.4): uno o más JSON escritos por
  ``ensemble_decisions.py --output`` (``--decisions``; si varios contienen el
  mismo paquete y escenario, gana el último).
* ``trf_vs_ens.compare`` (Tabla ``tab:trf-vs-ens``), ``random_baseline`` y
  ``agent_internals`` de ``general.scripts`` (sin escribir figuras).
* ``inputs/best_params.json``, ``inputs/**/optimization_results_*.txt`` y
  ``config/CONFIG.py`` de cada paquete (hiperparámetros, arquitecturas y
  número de parámetros entrenables, calculado con las fórmulas de Keras).
* Los ``.tex`` de ``writings/`` (orden de ``Main.tex``, secciones, etiquetas,
  títulos cortos, figuras, citas) y ``writings/audit/AUDIT.md`` (§13).

El texto redactado a mano (definiciones, protocolo, resumen ejecutivo,
advertencias, puntos de atención, reproducción, descripciones de reglas y
escenarios) está en ``claude_context/mpes_resultados_notas.md``, en bloques
con nombre; los números de ese texto son marcadores ``{{ruta|formato}}`` que
el generador llena con los valores calculados (ver la cabecera de las notas).
Además de cualquier clave del JSON, las notas pueden usar valores derivados
``_/...``: ``n_gen``, ``n_heldout``, ``n_ref_gen``, ``n_modelos``,
``n_estructurales``, ``max_asignacion``, ``presupuesto_*``, ``ver/<paquete>``,
``cfg/<pkg>/<CONSTANTE>``, ``arq/*`` y los resúmenes de las réplicas
``ho_*`` (lista completa en :func:`derived_values`).

Salida: UTF-8 sin BOM con fin de línea CRLF. La fecha y el commit de la
cabecera no cuentan como cambio: si sólo difieren ellos, el archivo en disco
no se reescribe.

Uso (desde la raíz del repositorio, con ``win_mpes_env`` activado)::

    python writings/auxiliar/scripts/build_results_context.py
    python writings/auxiliar/scripts/build_results_context.py --check
    python writings/auxiliar/scripts/build_results_context.py --decisions a.json b.json
"""

##########################
##  Imports externos    ##
##########################
import argparse
import ast
import contextlib
import datetime
import difflib
import glob
import importlib.metadata
import inspect
import io
import json
import math
import os
import re
import subprocess
import sys

import numpy

##########################
##  Configuración       ##
##########################
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
H1 = os.path.join(ROOT, 'h1')
RESULTS = os.path.join(H1, 'general', 'results')
WORK = os.path.join(H1, 'general', 'work')
WRITINGS = os.path.join(ROOT, 'writings')
SCRIPTS = os.path.join(WRITINGS, 'auxiliar', 'scripts')
CONTEXT = os.path.join(WRITINGS, 'auxiliar', 'claude_context')
DEFAULT_MD = os.path.join(CONTEXT, 'mpes_resultados.md')
DEFAULT_JSON = os.path.join(CONTEXT, 'mpes_resultados.json')
DEFAULT_NOTES = os.path.join(CONTEXT, 'mpes_resultados_notas.md')
DEFAULT_DECISIONS = os.path.join(RESULTS, 'ensemble', 'ens_decisions.json')
sys.path.insert(0, H1)
sys.path.insert(0, SCRIPTS)

##########################
##  Imports internos    ##
##########################
# pylint: disable=wrong-import-position
from general.scripts import agent_internals, random_baseline
from general.scripts.benchmark import (PACKAGE_GROUPS, REFERENCE_SCENARIO, generalisation_scenarios,
                                       heldout_scenarios)
from general.scripts.plotting import histogram_pmf, kl_divergence, read_matrix_csv
import sync_figures
from trf_vs_ens import compare

SUITES = ('individual', 'ensemble')
TRANSFORMER, WEIGHTED, QLEARNING = 'pes_trf', 'pes_ens', 'pes_ql'
FAMILY_ES = {'baseline': 'referencia', 'severity': 'severidad', 'length': 'longitud', 'joint': 'conjunta',
             'structural': 'estructural', 'heldout': 'fuera de muestra'}
FAMILIES = ('severity', 'length', 'joint', 'structural')
FAMILY_DEGRADATION_ORDER = ('joint', 'length', 'severity', 'structural')   # orden de comparison_metrics
MEMBER_ORDER = ('dqn', 'rdqn', 'trf', 'a2c')
ENSEMBLE_PARAMETERS = {'confidence_power': 'rho', 'temperature': 'tau', 'agreement_bonus': 'beta_a',
                       'disagreement_penalty': 'beta_d', 'prior_sigma': 'sigma_prior', 'prior_weight': 'w_prior',
                       'gate_slope': 'kappa_g', 'trf_confidence_threshold': 'tau_g'}
NETWORK_PREFIX = {'pes_dqn': 'DQN', 'pes_rdqn': 'RDQN', 'pes_trf': 'TRF'}
SELECTION_MD = {'optuna': 'Optuna', 'manual': 'manual (sin optimizar)', 'sin optimizar': 'sin optimizar'}
SELECTION_JSON = {'optuna': 'Optuna/TPE sobre las 64 secuencias de referencia',
                  'manual': 'valores fijos de config/CONFIG.py (sin optimización)',
                  'sin optimizar': 'valores originales de PES (sin optimizar)'}
# (clave JSON, evento en la tabla, contador de ensemble_decisions.py, clave JSON en las réplicas)
FINAL_DIFF = ('accion_final_distinta_del_transformer', 'Acción final distinta de la del Transformer', 'trf_diff',
              'accion_final_distinta_del_transformer')
DECISION_EVENTS = {
    'pes_ens_trf_guard': (('sigue_al_transformer_g>=0.5', 'Sigue al Transformer (g ≥ 0,5)', 'follow',
                           'sigue_al_transformer'), FINAL_DIFF),
    'pes_ens_consensus_prior': (('prior_cambia_accion_votada', 'El prior cambia la acción votada', 'prior',
                                 'prior_cambia_accion_votada'),
                                ('cota_cambia_accion', 'La cota cambia la acción', 'bound', 'cota_cambia_accion'),
                                FINAL_DIFF)}
DECISION_EVENTS[WEIGHTED] = DECISION_EVENTS['pes_ens_consensus_prior']
OPTIMUM_TOLERANCE = 1e-9      # rbar = 1 salvo error de redondeo (hay secuencias óptimas con 0.9999999999999998)
SCORE_TOLERANCE = 1e-4          # |media del benchmark - score de Optuna| para "reproduce el score"
LABEL_RE = re.compile(r'\b(?:sec|tab|fig|eq|ap):[A-Za-z0-9-]+')
PLACEHOLDER_RE = re.compile(r'\{\{(.+?)\}\}')
SUPERSCRIPT = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')


##########################
##  Formato de números  ##
##########################
def _drop_negative_zero(text: str) -> str:
    """Return ``text`` without the sign when it represents zero (``-0.00`` -> ``0.00``)."""
    return text[1:] if text[:1] in '+-' and not re.search(r'[1-9]', text) else text


def fixed(value, decimals: int) -> str:
    """Format ``value`` with fixed decimals, without a sign on zero; ``None``/NaN -> ``—``."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '—'
    return _drop_negative_zero(f'{value:.{decimals}f}')


def signed(value, decimals: int) -> str:
    """Format ``value`` with an explicit sign, except for zero (``+0.039``, ``0.000``)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '—'
    return _drop_negative_zero(f'{value:+.{decimals}f}')


def significant(value: float, digits: int) -> str:
    """Format ``value`` with ``digits`` significant digits keeping trailing zeros (``4.170``)."""
    if value == 0:
        return '0'
    decimals = max(digits - 1 - int(math.floor(math.log10(abs(value)))), 0)
    return f'{value:.{decimals}f}'


def thousands(value) -> str:
    """Format an integer with a space as thousands separator (``1 000 000``)."""
    return f'{int(value):,}'.replace(',', ' ')


def power_bound(p_value: float) -> str:
    """Return the smallest power of ten above ``p_value`` (``7.5e-08`` -> ``10⁻⁷``)."""
    return '10' + str(int(math.ceil(math.log10(p_value)))).translate(SUPERSCRIPT)


def format_value(value, spec: str) -> str:
    """Format a placeholder value with the mini-language documented in the notes file.

    Parameters
    ----------
    value : object
        Value resolved from the JSON data or the derived namespace.
    spec : str
        Python format spec or one of ``s2``/``s3``/``s4``, ``pct0``/``pct1``,
        ``miles``, ``sup``, ``e1``; a trailing comma selects a decimal comma.

    Returns
    -------
    str
        Formatted text.
    """
    if value is None:
        return '—'
    if not spec:
        return str(value)
    comma, minus = ',' in spec[-2:], '−' in spec[-2:]
    spec = spec.rstrip(',−')
    if spec in ('s2', 's3', 's4'):
        text = signed(value, int(spec[1]))
    elif spec in ('pct0', 'pct1'):
        text = fixed(100.0 * value, int(spec[3]))
    elif spec == 'miles':
        text = thousands(value)
    elif spec == 'sup':
        text = power_bound(value)
    elif spec == 'e1':
        text = f'{value:.1e}'
    else:
        text = _drop_negative_zero(format(value, spec))
    text = text.replace('-', '−') if minus else text
    return text.replace('.', ',') if comma else text


def with_article(name: str) -> str:
    """Prefix a display name with ``El`` (``Ensamble ponderado`` -> ``El ensamble ponderado``)."""
    return 'El ' + (name[0].lower() + name[1:] if name[1:2].islower() else name)


def r6(value) -> 'float | None':
    """Round to 6 decimals (precision of ``matrices/*.csv``); NaN -> ``None``."""
    if value is None or math.isnan(float(value)):
        return None
    return round(float(value), 6)


def round_significant(value: float, digits: int) -> float:
    """Round ``value`` to ``digits`` significant digits."""
    return float(f'{value:.{digits}g}')


##########################
##  Notas               ##
##########################
class Pending:
    """Narrative template (and local placeholder values) rendered once all numeric values exist."""

    def __init__(self, template: str, local: 'dict | None' = None) -> None:
        self.template = template
        self.local = local or {}

    def __repr__(self) -> str:
        return f'Pending({self.template[:40]!r})'


class Notes:
    """Named blocks (``## [nombre]``) of ``mpes_resultados_notas.md``."""

    def __init__(self, path: str) -> None:
        with open(path, encoding='utf-8') as handle:
            text = re.sub(r'<!--.*?-->[ \t]*\n?', '', handle.read(), flags=re.S)
        self.blocks: 'dict[str, str]' = {}
        name, lines = None, []
        for line in text.splitlines():
            match = re.fullmatch(r'## \[([A-Za-z0-9_.\-]+)\]\s*', line)
            if match:
                if name:
                    self.blocks[name] = '\n'.join(lines).strip('\n')
                name, lines = match.group(1), []
            elif name:
                lines.append(line)
        if name:
            self.blocks[name] = '\n'.join(lines).strip('\n')

    def text(self, name: str) -> str:
        """Return the raw text of block ``name``."""
        if name not in self.blocks:
            raise KeyError(f'Falta el bloque [{name}] en el archivo de notas')
        return self.blocks[name]

    def items(self, name: str) -> 'list[str]':
        """Return the ``- item`` lines of block ``name`` (continuation lines are joined)."""
        items: 'list[str]' = []
        for line in self.text(name).splitlines():
            if line.startswith('- '):
                items.append(line[2:].strip())
            elif line.strip() and items:
                items[-1] += ' ' + line.strip()
        return items

    def mapping(self, name: str) -> dict:
        """Return the ``- clave: valor`` pairs of block ``name`` (``@bloque`` nests another block)."""
        result: dict = {}
        for item in self.items(name):
            key, _, value = item.partition(': ')
            result[key.strip()] = self.mapping(value[1:].strip()) if value.startswith('@') else value.strip()
        return result

    def table(self, name: str) -> 'list[dict]':
        """Return the rows of the Markdown table in block ``name`` as dictionaries."""
        rows = []
        for line in self.text(name).splitlines():
            if not line.startswith('|'):
                continue
            cells = [cell.strip().replace('\\|', '|') for cell in re.split(r'(?<!\\)\|', line.strip())[1:-1]]
            if all(set(cell) <= set('-: ') for cell in cells):
                continue
            rows.append(cells)
        header = rows[0]
        return [dict(zip(header, row)) for row in rows[1:]]


def resolve(data: dict, path: str):
    """Return the value at ``path`` of ``data`` (``/``-separated keys; integers index lists)."""
    node = data
    for part in path.split('/'):
        if isinstance(node, list) and part.isdigit():
            node = node[int(part)]
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise KeyError(f'Marcador {{{{{path}}}}}: no existe la clave {part!r}')
    if isinstance(node, Pending):
        raise KeyError(f'Marcador {{{{{path}}}}}: apunta a otro texto de las notas')
    return node


def render(template: str, data: dict, mode: str, local: 'dict | None' = None) -> str:
    """Fill the ``{{...}}`` placeholders of ``template``; ``mode`` (``'md'``/``'json'``) selects the ``{{md:...|json:...}}`` alternatives and ``local`` values take precedence over ``data``."""
    def substitute(match: 're.Match') -> str:
        body = match.group(1).strip()
        if body.startswith('md:'):
            md_text, _, json_text = body[3:].partition('|json:')
            return md_text if mode == 'md' else json_text
        path, _, spec = body.partition('|')
        value = local[path] if local and path in local else resolve(data, path)
        return format_value(value, spec)
    return PLACEHOLDER_RE.sub(substitute, template)


def resolve_pending(node, data: dict):
    """Replace every :class:`Pending` inside ``node`` by its JSON rendering."""
    if isinstance(node, Pending):
        return render(node.template, data, 'json', node.local)
    if isinstance(node, dict):
        for key, value in node.items():
            node[key] = resolve_pending(value, data)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            node[index] = resolve_pending(value, data)
    return node


##########################
##  Carga de fuentes    ##
##########################
def load_json(path: str):
    """Load a UTF-8 JSON file."""
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def relative(path: str) -> str:
    """Return ``path`` relative to the repository root (absolute if outside it) with forward slashes."""
    path = os.path.abspath(path)
    inside = os.path.normcase(path).startswith(os.path.normcase(ROOT) + os.sep)
    return (os.path.relpath(path, ROOT) if inside else path).replace(os.sep, '/')


def package_dir(pkg: str) -> str:
    """Return ``h1/<grupo>/<pkg>``."""
    return os.path.join(H1, PACKAGE_GROUPS[pkg], pkg)


def load_config(pkg: str):
    """Execute ``config/CONFIG.py`` of ``pkg`` without importing the package (no TensorFlow)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return random_baseline.load_package_config(pkg)


def find_best_params(pkg: str) -> str:
    """Return the ``best_params.json`` that ``ens/<pkg>/ext/evaluate_ens.py`` would load.

    Mirrors ``_find_best_params``: the candidate with the highest ``value``
    (ties broken by modification time) among ``inputs/best_params.json`` and
    ``inputs/*_BAYESIAN_OPT/best_params.json``.
    """
    inputs = os.path.join(package_dir(pkg), 'inputs')
    candidates = [os.path.join(inputs, 'best_params.json')]
    candidates += glob.glob(os.path.join(inputs, '*_BAYESIAN_OPT', 'best_params.json'))
    scored = []
    for path in candidates:
        if os.path.isfile(path):
            try:
                scored.append((float(load_json(path).get('value', float('-inf'))), os.path.getmtime(path), path))
            except (OSError, TypeError, ValueError):
                continue
    return max(scored)[2] if scored else candidates[0]


def optuna_trials(pkg: str) -> dict:
    """Parse the number of Optuna trials (and pruned trials) of the latest study of ``pkg``."""
    files = sorted(glob.glob(os.path.join(package_dir(pkg), 'inputs', '**', 'optimization_results_*.txt'),
                             recursive=True), key=os.path.basename)
    if not files:
        return {}
    with open(files[-1], encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    result = {}
    match = re.search(r'Total trials:\s*(\d+)', text)
    if match:
        result['total'] = int(match.group(1))
    match = re.search(r'Pruned:\s*(\d+)', text)
    if match:
        result['pruned'] = int(match.group(1))
    return result


def base_training_config(pkg: str) -> dict:
    """Parse ``inputs/*_RL_TRAIN/training_config_*.txt`` of an unoptimised tabular package."""
    files = sorted(glob.glob(os.path.join(package_dir(pkg), 'inputs', '*_RL_TRAIN', 'training_config_*.txt')))
    with open(files[-1], encoding='utf-8') as handle:
        text = handle.read()

    def number(label: str) -> float:
        match = re.search(re.escape(label) + r'[^:]*:\s*([\d.,]+)', text)
        if not match:
            raise ValueError(f'{label} no está en {files[-1]}')
        return float(match.group(1).replace(',', ''))
    return {'alpha': number('Learning Rate'), 'gamma': number('Discount Factor'),
            'epsilon_0': number('Initial Epsilon'), 'epsilon_min': number('Minimum Epsilon'),
            'episodios': int(number('Number of Episodes'))}


def load_decisions(paths: 'list[str]') -> 'tuple[dict, list[str]]':
    """Merge the ``ensemble_decisions.py --output`` files (later files win)."""
    merged: dict = {}
    used = []
    for path in paths:
        if not os.path.isfile(path):
            print(f'AVISO: no existe {path}; se omite', file=sys.stderr)
            continue
        for pkg, scenarios in load_json(path).items():
            merged.setdefault(pkg, {}).update(scenarios)
        used.append(path)
    return merged, used


class Suite:
    """Benchmark outputs of one suite (``individual`` or ``ensemble``)."""

    def __init__(self, name: str) -> None:
        folder = os.path.join(RESULTS, name)
        self.name = name
        self.summary = load_json(os.path.join(folder, 'summary.json'))
        self.metrics = load_json(os.path.join(folder, 'comparison_metrics.json'))
        self.models: 'list[str]' = list(self.summary['models'])
        self.cells: dict = self.summary['cells']
        self.matrices = {}
        for metric in ('cohen_d', 'welch_logp', 'action_kl', 'stress_degradation'):
            models, scenarios, matrix = read_matrix_csv(os.path.join(folder, 'matrices', f'{metric}.csv'))
            self.matrices[metric] = {m: dict(zip(scenarios, matrix[i])) for i, m in enumerate(models)}

    def mean(self, pkg: str, scenario: str) -> float:
        """Mean normalised performance of one cell."""
        return float(self.cells[pkg][scenario]['global_mean_perf'])

    def perf(self, pkg: str, scenario: str) -> numpy.ndarray:
        """Per-sequence normalised performance of one cell."""
        return numpy.asarray(self.cells[pkg][scenario]['per_sequence_perf'], dtype=float)


##########################
##  Modelos             ##
##########################
def dense_chain(sizes: 'list[int]') -> int:
    """Trainable parameters of a chain of Dense layers ``sizes[0] -> ... -> sizes[-1]``."""
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


def trainable_parameters(pkg: str, cfg, n_inputs: int, n_actions: int) -> int:
    """Number of trainable parameters of the deployed network, from ``config/CONFIG.py``.

    DQN: dense chain. RDQN: LSTM (4 gates) + dense head. TRF: input projection,
    ``TRF_NUM_LAYERS`` Post-LN blocks (Q/K/V/O projections with bias, FFN, two
    LayerNorms) and dense head; the fixed positional vector is not trained.
    """
    if pkg == 'pes_dqn':
        return dense_chain([n_inputs, *cfg.DQN_HIDDEN_UNITS, n_actions])
    if pkg == 'pes_rdqn':
        units = int(cfg.RDQN_LSTM_UNITS)
        return 4 * (units * (n_inputs + units) + units) + dense_chain([units, *cfg.RDQN_HIDDEN_UNITS, n_actions])
    d_model, heads, key_dim, ff_dim = (int(cfg.TRF_D_MODEL), int(cfg.TRF_NUM_HEADS), int(cfg.TRF_KEY_DIM),
                                       int(cfg.TRF_FF_DIM))
    attention = 3 * (d_model * heads * key_dim + heads * key_dim) + heads * key_dim * d_model + d_model
    block = attention + dense_chain([d_model, ff_dim, d_model]) + 2 * 2 * d_model
    return (n_inputs * d_model + d_model + int(cfg.TRF_NUM_LAYERS) * block
            + dense_chain([d_model, *cfg.TRF_HIDDEN_UNITS, n_actions]))


def text_fields(notes: Notes, block: str, keys: 'tuple[str, ...]') -> dict:
    """Return the requested narrative fields of a notes block as :class:`Pending` values."""
    mapping = notes.mapping(block)
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f'Faltan {missing} en el bloque [{block}] de las notas')
    return {key: Pending(mapping[key]) for key in keys}


def individual_hyperparameters(pkg: str, notes: Notes, reference_mean: float) -> dict:
    """Hyperparameter block of one individual package (JSON layout of ``modelos_individuales``)."""
    cfg = load_config(pkg)
    block = f'hiper.{pkg}'
    if pkg == 'pes_base':
        values = base_training_config(pkg)
        text = text_fields(notes, block, ('tipo', 'tabla_Q', 'decaimiento_epsilon', 'optimizacion_bayesiana'))
        return {'tipo': text['tipo'], 'tabla_Q': text['tabla_Q'], 'alpha': values['alpha'],
                'gamma': values['gamma'], 'epsilon_0': values['epsilon_0'], 'epsilon_min': values['epsilon_min'],
                'decaimiento_epsilon': text['decaimiento_epsilon'], 'episodios': values['episodios'],
                'semilla': int(cfg.SEED), 'optimizacion_bayesiana': text['optimizacion_bayesiana']}
    best = load_json(os.path.join(package_dir(pkg), 'inputs', 'best_params.json'))
    hp = best['hyperparameters']
    trials = optuna_trials(pkg)
    score = {'score_optuna': round(float(best['mean_perf']), 4),
             'reproduce_score_optuna': abs(reference_mean - float(best['mean_perf'])) < SCORE_TOLERANCE}
    if pkg in ('pes_ql', 'pes_dql'):
        text = text_fields(notes, block, ('tipo', 'decaimiento_epsilon') + (('potencial_pbrs',)
                                                                             if pkg == 'pes_dql' else ()))
        out = {'tipo': text['tipo'], 'alpha': round(hp['learning_rate'], 4), 'gamma': round(hp['discount_factor'], 4),
               'epsilon_0': round(hp['epsilon_initial'], 4), 'epsilon_min': round(hp['epsilon_min'], 4),
               'decaimiento_epsilon': text['decaimiento_epsilon']}
        if pkg == 'pes_ql':
            return {**out, 'episodios': int(hp['num_episodes']), 'mejor_ensayo': int(best['best_trial_number']),
                    'semilla': int(best['training_seed']), 'ensayos_optuna': trials.get('total'), **score}
        return {**out, 'w_calentamiento': round(hp['warmup_ratio'], 4),
                'q_fraccion_epsilon_min': round(hp['target_ratio'], 4),
                'pbrs_kappa': round_significant(hp['penalty_coeff'], 3), 'potencial_pbrs': text['potencial_pbrs'],
                'episodios': int(hp['num_episodes']), 'semilla': int(best['seed']),
                'ensayos_optuna': trials.get('total'), 'ensayos_podados_median_pruner': trials.get('pruned'), **score}
    if pkg == 'pes_a2c':
        text = text_fields(notes, block, ('tipo', 'arquitectura', 'decaimiento_lr', 'penalizacion_gasto', 'nota'))
        return {'tipo': text['tipo'], 'arquitectura': text['arquitectura'], 'lr_actor': round(hp['actor_lr'], 6),
                'lr_critico': round(hp['critic_lr'], 6), 'decaimiento_lr': text['decaimiento_lr'],
                'gamma': round(hp['discount_factor'], 4), 'coef_entropia': round_significant(hp['entropy_coeff'], 3),
                'gae_lambda': round(hp['gae_lambda'], 4), 'clip_grad': round_significant(hp['max_grad_norm'], 4),
                'pbrs_kappa': round(hp['penalty_coeff'], 4), 'penalizacion_gasto': text['penalizacion_gasto'],
                'sesgo_logit_a10': round_significant(hp['last_action_bias'], 4),
                'episodios': int(hp['num_episodes']), 'mejor_ensayo': int(best['best_trial_number']),
                'semilla': int(best['trial_seed']), 'ensayos_optuna': trials.get('total'), **score,
                'nota': text['nota']}
    prefix = NETWORK_PREFIX[pkg]

    def const(name: str):
        return getattr(cfg, f'{prefix}_{name}')
    optimised = pkg == 'pes_dqn'
    text = text_fields(notes, block, ('tipo', 'arquitectura', 'arquitectura_origen') + (() if optimised else ('nota',)))
    n_actions = int(cfg.MAX_ALLOCATABLE_RESOURCES) + 1
    out = {'tipo': text['tipo'], 'arquitectura': text['arquitectura'],
           'parametros_entrenables': trainable_parameters(pkg, cfg, 3, n_actions),
           'arquitectura_origen': text['arquitectura_origen'], 'lr': round(const('LEARNING_RATE'), 6),
           'gamma': round(const('DISCOUNT'), 4), 'epsilon_0': round(const('EPSILON_INITIAL'), 4),
           'epsilon_min': round(const('EPSILON_MIN'), 4), 'w': round(const('WARMUP_RATIO'), 4),
           'q': round(const('TARGET_RATIO'), 4), 'batch': int(const('BATCH_SIZE')),
           'buffer': int(const('REPLAY_BUFFER_SIZE')), 'sync_objetivo_C': int(const('TARGET_SYNC_FREQ')),
           'clip_grad': round_significant(const('MAX_GRAD_NORM'), 4),
           'pbrs_kappa': round_significant(const('PENALTY_COEFF'), 3) if const('PENALTY_COEFF') else 0.0,
           'inicio_aprendizaje_f': round(const('LEARNING_STARTS_FRAC'), 4),
           'inicio_aprendizaje_transiciones': int(const('LEARNING_STARTS_FRAC') * const('REPLAY_BUFFER_SIZE')),
           'episodios': int(const('EPISODES'))}
    if optimised:
        return {**out, 'mejor_ensayo': int(best['best_trial_number']), 'semilla': int(best['trial_seed']),
                'ensayos_optuna': trials.get('total'), **score}
    return {**out, 'semilla': int(best['trial_seed']), 'nota': text['nota']}


def ensemble_definition(pkg: str, notes: Notes, reference_mean: float) -> dict:
    """Rule, members and parameters of one ensemble (JSON layout of ``ensambles.<pkg>.definicion``)."""
    cfg = load_config(pkg)
    block = f'ens.{pkg}'
    if pkg == WEIGHTED:
        text = text_fields(notes, block, ('regla', 'a2c', 'origen_parametros', 'confianza', 'nota'))
        members = {m['name']: m for m in cfg.ENS_MEMBER_MODELS if m.get('enabled', True)}
        names = [name for name in MEMBER_ORDER if name in members]
        total = sum(float(members[name]['weight']) for name in names)
        parameters = {'tau': float(cfg.ENS_SOFTMAX_TEMPERATURE),
                      **{f'w_{name}': float(members[name]['weight']) for name in names},
                      'peso_normalizado_trf': round(float(members['trf']['weight']) / total, 3),
                      'w_prior': float(cfg.ENS_SEVERITY_PRIOR_WEIGHT), 'sigma_prior': float(cfg.ENS_SEVERITY_PRIOR_SIGMA)}
        return {'regla': text['regla'], 'miembros': names, 'a2c': text['a2c'], 'parametros': parameters,
                'origen_parametros': text['origen_parametros'], 'confianza': text['confianza'], 'nota': text['nota']}
    text = text_fields(notes, block, ('regla', 'origen_parametros', 'confianza'))
    best = load_json(find_best_params(pkg))
    parameters = {}
    for key, value in best['hyperparameters'].items():
        short = ENSEMBLE_PARAMETERS.get(key, key.replace('weight_', 'w_'))
        parameters[short] = r6(value)
    parameters.setdefault('tau', float(getattr(cfg, 'DEFAULT_TEMPERATURE', 1.0)))
    members = [name for name in MEMBER_ORDER if name in cfg.MODEL_PATHS]
    return {'regla': text['regla'], 'miembros': members, 'parametros': dict(sorted(parameters.items())),
            'origen_parametros': text['origen_parametros'], 'score_optuna': r6(best['value']),
            'score_optuna_igual_a_referencia_benchmark': abs(float(best['value']) - reference_mean) < 1e-6,
            'confianza': text['confianza']}


def model_entry(suite: Suite, pkg: str, gen: 'list[str]', ql: dict, names: dict,
                detail: dict, out_of_range: 'list[str]') -> 'tuple[dict, dict]':
    """JSON summary block of one model and its raw (unrounded) values for the Markdown tables."""
    metrics = suite.metrics['models'][pkg]
    ref = suite.mean(pkg, REFERENCE_SCENARIO)
    means = {s: suite.mean(pkg, s) for s in gen}
    gen_mean = float(numpy.mean(list(means.values())))
    if abs(gen_mean - metrics['stress_mean']) > 1e-9:
        raise ValueError(f'{pkg}: media de generalización {gen_mean} != comparison_metrics {metrics["stress_mean"]}')
    families = {f: [s for s in gen if suite.cells[pkg][s]['family'] == f] for f in FAMILIES}
    rounded = {s: round(v, 6) for s, v in means.items()}   # precisión de matrices/global_mean.csv
    worst = min(gen, key=lambda s: rounded[s])
    best = max(gen, key=lambda s: rounded[s])
    no_struct = [s for s in gen if s not in families['structural']]
    cell = suite.cells[pkg][REFERENCE_SCENARIO]
    perf = suite.perf(pkg, REFERENCE_SCENARIO).reshape(int(cell['num_blocks']), -1)
    raw = {'ref': ref, 'ref_sd': float(cell['std_perf']), 'gen': gen_mean, 'deg_mean': metrics['mean_degradation'],
           'deg_max': metrics['worst_degradation'], 'worst': worst, 'worst_mean': means[worst],
           'delta_ref': (ref - ql['ref']) / (1.0 - ql['ref']), 'delta_gen': (gen_mean - ql['gen']) / (1.0 - ql['gen']),
           'family_deg': metrics['family_degradation'],
           'no_struct': float(numpy.mean([rounded[s] for s in no_struct])),
           'out_of_range': float(numpy.mean([rounded[s] for s in gen if s in out_of_range]))}
    entry = {'nombre_en_texto': names[pkg], 'referencia_media': r6(ref), 'referencia_sd': r6(cell['std_perf']),
             'referencia_min': r6(cell['min_perf']), 'referencia_max': r6(cell['max_perf']),
             'generalizacion_media_21': r6(gen_mean), 'generalizacion_sd_agrupada_21': r6(metrics['stress_std']),
             'degradacion_media': r6(metrics['mean_degradation']), 'mayor_degradacion': r6(metrics['worst_degradation']),
             'escenario_mayor_degradacion': metrics['worst_scenario'], 'peor_escenario': worst,
             'peor_escenario_media': r6(means[worst]), 'mejor_escenario': best, 'mejor_escenario_media': r6(means[best]),
             'degradacion_por_familia': {FAMILY_ES[f]: r6(metrics['family_degradation'][f])
                                         for f in FAMILY_DEGRADATION_ORDER},
             'media_por_familia': {FAMILY_ES[f]: r6(numpy.mean([rounded[s] for s in families[f]])) for f in FAMILIES},
             'derivado_media_18_sin_estructurales': r6(raw['no_struct']),
             'derivado_media_3_fuera_de_rango': r6(raw['out_of_range']),
             'delta_rel_vs_ql_referencia': round(raw['delta_ref'], 3),
             'delta_rel_vs_ql_generalizacion': round(raw['delta_gen'], 3),
             ('hiperparametros' if suite.name == 'individual' else 'definicion'): detail,
             'referencia_detalle': {
                 'distribucion_acciones_0a10': [round(float(v), 3) for v in cell['action_distribution']],
                 'media_por_bloque': [r6(v) for v in perf.mean(axis=1)],
                 'sd_por_bloque': [r6(v) for v in perf.std(axis=1, ddof=1)]}}
    return entry, raw


def cell_entry(suite: Suite, pkg: str, scenario: str) -> dict:
    """Per-cell values of ``celdas.<pkg>.<escenario>``."""
    cell = suite.cells[pkg][scenario]
    perf = suite.perf(pkg, scenario)
    reference = scenario == REFERENCE_SCENARIO

    def matrix(metric: str):
        return None if reference else r6(suite.matrices[metric][pkg][scenario])
    return {'mu': r6(cell['global_mean_perf']), 'sd': r6(cell['std_perf']), 'min': r6(cell['min_perf']),
            'max': r6(cell['max_perf']), 'deg': r6(suite.matrices['stress_degradation'][pkg][scenario]),
            'd': matrix('cohen_d'), 'log10p': matrix('welch_logp'), 'kl_acc': matrix('action_kl'),
            'f_opt': round(float(numpy.mean(perf >= 1.0 - OPTIMUM_TOLERANCE)), 4), 'f_lt08': round(float(numpy.mean(perf < 0.8)), 4),
            'a_media': round(float(numpy.mean(cell['per_trial_actions'])), 4), 'n': int(cell['n_sequences'])}


def winners(values: 'dict[str, float]') -> dict:
    """Winners (ties at 6 decimals) and their mean."""
    top = max(values.values())
    return {'ganador': [pkg for pkg, value in values.items() if value == top], 'media': top}


##########################
##  Documento LaTeX     ##
##########################
ACCENTS = {"'": dict(zip('aeiouAEIOU', 'áéíóúÁÉÍÓÚ')), '~': {'n': 'ñ', 'N': 'Ñ'}, '"': {'u': 'ü', 'U': 'Ü'}}


def strip_comments(text: str) -> str:
    """Remove LaTeX comments (unescaped ``%`` to end of line)."""
    return re.sub(r'(?<!\\)%.*', '', text)


def latex_to_text(text: str) -> str:
    """Convert a short LaTeX string (caption, title) to plain text."""
    text = re.sub(r"\\(['~\"])\{?(\\i|[A-Za-z])\}?",
                  lambda m: ACCENTS[m.group(1)].get('i' if m.group(2) == '\\i' else m.group(2), m.group(2)), text)
    for _ in range(3):
        text = re.sub(r'\\(?:texttt|emph|textbf|textit|mbox|text)\{([^{}]*)\}', r'\1', text)
    text = text.replace('\\_', '_').replace('\\%', '%').replace('~', ' ').replace('---', '—').replace('--', '–')
    text = text.replace('\\ ', ' ').replace('{,}', ',')
    return re.sub(r'\s+', ' ', text).strip()


def braced(text: str, start: int) -> 'tuple[str, int]':
    """Return the content of the balanced ``{...}``/``[...]`` group opening at ``text[start]`` and the index after it."""
    opening = text[start]
    closing, depth, index = {'{': '}', '[': ']'}[opening], 0, start
    while index < len(text):
        char = text[index]
        if char == '\\':
            index += 2
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start + 1:index], index + 1
        index += 1
    raise ValueError('grupo sin cerrar')


def command_arguments(text: str, index: int) -> 'tuple[str | None, str, int]':
    """Parse ``[optional]{mandatory}`` after a command ending at ``index``."""
    while index < len(text) and text[index] in ' \n\t*':
        index += 1
    optional = None
    if index < len(text) and text[index] == '[':
        optional, index = braced(text, index)
    while index < len(text) and text[index] in ' \n\t':
        index += 1
    mandatory, index = braced(text, index)
    return optional, mandatory, index


TOKEN_RE = re.compile(r'\\(begin|end)\{(figure|table|subfigure)\*?\}|\\(section|subsection|subsubsection)\b|'
                      r'\\(label)\{([^}]*)\}|\\(caption)\b|\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}|'
                      r'\\cite[a-zA-Z]*\*?(?:\[[^\]]*\]){0,2}\{([^}]*)\}|\\(?:auto|eq)?ref\{([^}]*)\}')


def parse_chapter(path: str, number: str, appendix: bool) -> dict:
    """Extract sections (numbered as ``number``, or A, B, ... in the appendix), labels, floats, images, citations and references of one chapter file."""
    with open(path, encoding='utf-8') as handle:
        text = strip_comments(handle.read())
    result: dict = {'labels': [], 'sections': [], 'floats': [], 'images': [], 'cites': [], 'refs': 0,
                    'chapter_label': None}
    counters = [0, 0, 0]
    stack: 'list[dict]' = []
    current: 'list[dict]' = []   # last sectioning command
    for match in TOKEN_RE.finditer(text):
        begin_end, env, section, label, caption, image, cite, ref = (match.group(1), match.group(2), match.group(3),
                                                                     match.group(5), match.group(6), match.group(7),
                                                                     match.group(8), match.group(9))
        if begin_end == 'begin':
            stack.append({'env': env, 'labels': [], 'caption': None, 'images': [], 'children': []})
        elif begin_end == 'end' and stack:
            node = stack.pop()
            if stack:
                stack[-1]['children'].append(node)
            else:
                result['floats'].append(node)
        elif section:
            _, title, _ = command_arguments(text, match.end())
            starred = text[match.end():match.end() + 1] == '*'
            level = ('section', 'subsection', 'subsubsection').index(section)
            if not starred:
                counters[level] += 1
                counters[level + 1:] = [0] * (2 - level)
            if appendix:
                parts = [chr(ord('A') + counters[0] - 1)] + [str(c) for c in counters[1:level + 1]]
            else:
                parts = [number] + [str(c) for c in counters[1:level + 1]]
            current.append({'nivel': section, 'numero': '' if starred else '.'.join(p for p in parts if p),
                            'titulo': latex_to_text(title), 'label': None})
            result['sections'].append(current[-1])
        elif label:
            result['labels'].append(label)
            if stack:
                stack[-1]['labels'].append(label)
            elif not label.startswith('eq:'):
                if current and current[-1]['label'] is None:
                    current[-1]['label'] = label
                elif not current and result['chapter_label'] is None:
                    result['chapter_label'] = label
        elif caption and stack:
            short, long_caption, _ = command_arguments(text, match.end())
            stack[-1]['caption'] = latex_to_text(short if short is not None else long_caption)
        elif image:
            name = os.path.basename(image)
            result['images'].append(name)
            for node in stack:
                node['images'].append(name)
        elif cite:
            result['cites'] += [key.strip() for key in cite.split(',') if key.strip()]
        elif ref:
            result['refs'] += 1
    return result


def main_structure() -> 'tuple[list[dict], list[str]]':
    """Return the chapters in ``Main.tex`` order and the ``\\graphicspath`` folders."""
    with open(os.path.join(WRITINGS, '00_Main', 'Main.tex'), encoding='utf-8') as handle:
        text = strip_comments(handle.read())
    graphics = []
    match = re.search(r'\\graphicspath\{(.*?)\n\}', text, flags=re.S)
    if match:
        for folder in re.findall(r'\{([^{}]+)\}', match.group(1)):
            name = folder.strip('/').split('/')[-1]
            if name not in graphics:
                graphics.append(name)
    chapters, title, number, appendix = [], None, 0, False
    for match in re.finditer(r'\\section(\*?)\{([^}]*)\}|\\(subfile|include)\{([^}]*)\}|\\begin\{appendices\}',
                             text):
        if match.group(0).startswith('\\begin'):
            appendix, title = True, 'Apéndice'
        elif match.group(2) is not None:
            starred = bool(match.group(1))
            if not starred:
                number += 1
            title = (f'{latex_to_text(match.group(2))} (sin numerar)' if starred
                     else f'{number} {latex_to_text(match.group(2))}')
        else:
            path = os.path.normpath(os.path.join(WRITINGS, '00_Main', match.group(4) + '.tex'))
            if match.group(3) == 'include':
                label = 'Portada' if 'Frontpage' in path else latex_to_text(os.path.basename(path))
            else:
                label = title or ''
            with open(path, encoding='utf-8') as handle:
                if 'otherlanguage' in handle.read():
                    label += ', dentro de otherlanguage{english}'
            chapter_number = '' if appendix or match.group(3) == 'include' or '(sin numerar)' in label else str(number)
            chapters.append({'path': path, 'title': label, 'number': chapter_number, 'appendix': appendix})
            title = None if not appendix else title
    return chapters, graphics


def parse_audit(path: str) -> dict:
    """Parse ``writings/audit/AUDIT.md`` into a status dictionary."""
    with open(path, encoding='utf-8') as handle:
        text = handle.read()

    def count(pattern: str) -> 'int | None':
        match = re.search(pattern, text)
        return int(match.group(1)) if match else None
    orphans = re.findall(r'^- `([^`]+\.tex)`', text.split('**Orden de capítulos', 1)[0], flags=re.M)
    warnings = [line.strip('- ').strip() for line in text.splitlines() if '⚠' in line or '❌' in line]
    return {'archivo': relative(path),
            'compilacion_ok': 'Compilación exitosa' in text, 'paginas': count(r'\*\*(\d+) páginas\*\*'),
            'figuras_con_label': count(r'Figuras con label: \*\*(\d+)\*\*'),
            'tablas_con_label': count(r'Tablas con label: \*\*(\d+)\*\*'),
            'referencias_internas': count(r'Referencias internas[^*]*\*\*(\d+)\*\*'),
            'imagenes': count(r'Imágenes referenciadas[^*]*\*\*(\d+)\*\*'),
            'tex_huerfanos': [os.path.basename(p) for p in orphans], 'avisos': warnings}


def script_summaries() -> 'dict[str, str]':
    """First docstring line of every ``writings/auxiliar/scripts/*.py``."""
    out = {}
    for path in sorted(glob.glob(os.path.join(SCRIPTS, '*.py'))):
        with open(path, encoding='utf-8') as handle:
            doc = ast.get_docstring(ast.parse(handle.read())) or ''
        out[os.path.basename(path)] = doc.splitlines()[0] if doc else ''
    return out


def latex_document(notes: Notes) -> 'tuple[dict, dict]':
    """Build the LaTeX map (``documento_latex``) from ``Main.tex``, the chapters and ``AUDIT.md``.

    Returns
    -------
    tuple of dict
        JSON block and helper data for the Markdown section (label set, figure list...).
    """
    chapters, graphics = main_structure()
    mapa = notes.mapping('mapa')
    figure_notes = notes.mapping('mapa_figuras_notas')
    excluded = [name.strip() for name in mapa['excluidos_a_proposito'].split(',')]
    image_root = os.path.join(WRITINGS, '02_Images')
    image_folder = {name: folder for folder in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, folder))
                    for name in os.listdir(os.path.join(image_root, folder))}
    capitulos, figures, tables, all_labels, used_images, cited, refs = [], {}, {}, [], [], [], 0
    subfigures = {}
    for chapter in chapters:
        parsed = parse_chapter(chapter['path'], chapter['number'], chapter['appendix'])
        name = os.path.basename(chapter['path'])
        labels = parsed['labels']
        all_labels += labels
        used_images += parsed['images']
        cited += parsed['cites']
        refs += parsed['refs']
        capitulos.append({'archivo': name, 'seccion': chapter['title'], 'labels': labels,
                          'secciones': parsed['sections'], 'label_capitulo': parsed['chapter_label']})
        for node in parsed['floats']:
            label = next((lab for lab in reversed(node['labels'])), None)
            if not label:
                continue
            info = {'capitulo': name, 'titulo': node['caption'] or '',
                    'imagenes': [f'{image_folder.get(img, "?")}/{img}' for img in node['images']]}
            if node['env'] == 'table':
                tables[label] = info
            else:
                figures[label] = info
                for child in node['children']:
                    for child_label in child['labels']:
                        subfigures[child_label] = {'capitulo': name, 'titulo': child['caption'] or '',
                                                   'imagenes': [f'{image_folder.get(i, "?")}/{i}'
                                                                for i in child['images']], 'padre': label}
    chapter_dir = os.path.join(WRITINGS, '01_Chapters')
    included = [os.path.basename(c['path']) for c in chapters]
    others = sorted(f for f in os.listdir(chapter_dir) if f.endswith('.tex') and f not in included)
    chapter_files = included + [f'{f} ({mapa["nota_excluidos"]})' if f in excluded
                                else f'{f} (huérfano: no incluido en Main.tex)' for f in others]
    images_tree = {folder: sorted(os.listdir(os.path.join(image_root, folder)), key=str.lower)
                   for folder in graphics + sorted(set(os.listdir(image_root)) - set(graphics))
                   if os.path.isdir(os.path.join(image_root, folder))}

    def files(folder: str) -> 'list[str]':
        return sorted((f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))), key=str.lower)
    used = set(used_images)
    unused: 'dict[str, list[str]]' = {}
    for suite, tag in (('individual', 'ind'), ('ensemble', 'ens')):
        folder = os.path.join(RESULTS, suite, 'figures')
        stems = [os.path.splitext(f)[0] for f in files(folder) if f.endswith('.png')]
        unused[suite] = [s for s in stems
                         if f'{sync_figures.RENAMES.get(f"{tag}_{s}", f"{tag}_{s}")}.png' not in used]
        subfolders = sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))
        if subfolders:
            unused[suite] += [f'{d}/ (subcarpeta)' for d in subfolders]
    for folder in ('agent_internals', 'baseline'):
        unused[folder] = [os.path.splitext(f)[0] for f in files(os.path.join(RESULTS, folder))
                          if f.endswith('.png') and f not in used]
    with open(os.path.join(WRITINGS, '00_Main', 'References.bib'), encoding='utf-8') as handle:
        bib = re.findall(r'^@(?!comment|string|preamble)\w+\{([^,\s]+),', handle.read(), flags=re.M | re.I)
    cited_unique = list(dict.fromkeys(cited))
    audit = parse_audit(os.path.join(WRITINGS, 'audit', 'AUDIT.md'))
    own = {'figuras_con_label': sum(1 for lab in set(all_labels) if lab.startswith('fig:')),
           'tablas_con_label': sum(1 for lab in set(all_labels) if lab.startswith('tab:')),
           'referencias_internas': refs, 'imagenes': len(set(used_images))}
    audit['recuento_del_generador'] = own
    figure_json = {}
    for label, info in {**figures, **subfigures}.items():
        extra = f'; {figure_notes[label]}' if label in figure_notes else ''
        figure_json[label] = Pending(f'{" + ".join(info["imagenes"])} ({info["titulo"]}; {info["capitulo"]}{extra})')
    document = {
        'estructura_repo': {
            'writings/00_Main/': files(os.path.join(WRITINGS, '00_Main')),
            'writings/01_Chapters/': chapter_files,
            'writings/02_Images/': images_tree,
            'writings/audit/': files(os.path.join(WRITINGS, 'audit')),
            'writings/auxiliar/scripts/': [f for f in files(SCRIPTS) if f.endswith('.py')]},
        'capitulos': capitulos,
        'figuras': figure_json,
        'tablas': {label: f'{info["titulo"]} ({info["capitulo"]})' for label, info in tables.items()},
        'figuras_generadas_no_incluidas': {**unused, 'como_incluirlas': Pending(mapa['como_incluirlas'])},
        'scripts_auxiliares': script_summaries(),
        'bib_keys': bib,
        'bib_citadas': cited_unique,
        'bib_no_citadas': [k for k in bib if k not in cited_unique],
        'bib_faltantes': [k for k in cited_unique if k not in bib],
        'estado_audit': audit}
    helper = {'labels': set(all_labels), 'figures': figures, 'subfigures': subfigures, 'tables': tables,
              'excluded': excluded, 'figure_notes': figure_notes}
    return document, helper


##########################
##  Datos               ##
##########################
def git_state() -> dict:
    """Short commit hash and dirty flag of the repository."""
    def run(*args: str) -> str:
        try:
            return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ''
    return {'commit': run('rev-parse', '--short', 'HEAD') or 'desconocido', 'sucio': bool(run('status', '--porcelain'))}


def decision_shares(decisions: dict, pkg: str, subset: 'list[str]', counter: str) -> 'float | None':
    """Share (%) of decisions with resources in which ``counter`` happened, pooled as in ``ensemble_decisions.py``."""
    rows = [decisions.get(pkg, {}).get(s) for s in subset]
    rows = [row for row in rows if row]
    total = sum(row['n'] for row in rows)
    return round(100.0 * sum(row[counter] for row in rows) / total, 1) if total else None


def build_decisions(decisions: dict, sources: 'list[str]', suites: dict, gen: 'list[str]',
                    held: 'list[str]', notes: Notes) -> 'tuple[dict, dict]':
    """Block ``decisiones_ensambles`` and replay diagnostics."""
    ensemble = suites['ensemble']
    diffs, held_diffs, missing = [], [], []
    for pkg in DECISION_EVENTS:
        for scenario in [REFERENCE_SCENARIO] + gen + held:
            row = decisions.get(pkg, {}).get(scenario)
            if row is None:
                if scenario not in held:
                    missing.append(f'{pkg}/{scenario}')
                continue
            diff = abs(float(row['mean']) - ensemble.mean(pkg, scenario))
            (held_diffs if scenario in held else diffs).append(diff)
    source_text = ', '.join(relative(p) for p in sources) or 'ninguno'
    fuente = (f'Tesis, Tabla tab:ens-freq; replay de writings/auxiliar/scripts/ensemble_decisions.py --output '
              f'({source_text}). Diferencia máxima entre la media del replay y la del benchmark actual: '
              f'{max(diffs):.1e} (generalización y referencia)' if diffs else 'sin datos de replay')
    if held_diffs:
        fuente += f', {max(held_diffs):.1e} (réplicas)'
    if missing:
        fuente += (f'. Faltan {len(missing)} celdas en el replay: {", ".join(missing[:5])}'
                   + (', …' if len(missing) > 5 else ''))
        print(f'AVISO: faltan celdas en el replay de decisiones: {missing}', file=sys.stderr)
    block: dict = {'fuente': fuente + '.', 'unidad': Pending(notes.mapping('decisiones')['unidad'])}
    for pkg in ('pes_ens_trf_guard', 'pes_ens_consensus_prior', WEIGHTED):
        entry: dict = {}
        for key, _label, counter, _held_key in DECISION_EVENTS[pkg]:
            entry[key] = {'referencia': decision_shares(decisions, pkg, [REFERENCE_SCENARIO], counter),
                          'generalizacion': decision_shares(decisions, pkg, gen, counter)}
        entry['replicas_fuera_de_muestra'] = {held_key: decision_shares(decisions, pkg, held, counter)
                                              for _key, _label, counter, held_key in DECISION_EVENTS[pkg]}
        if pkg == WEIGHTED:
            entry['otros_hechos_tesis'] = [Pending(item) for item in notes.items('decisiones_otros_hechos')]
        block[pkg] = entry
    extrap = [decisions.get(WEIGHTED, {}).get(s) for s in ('sev_extrapolate_high', 'joint_extrap_both')]
    extrap_diff = sum(row['trf_diff'] for row in extrap if row) if all(extrap) else None
    return block, {'max_diff': max(diffs) if diffs else float('nan'),
                   'max_diff_heldout': max(held_diffs) if held_diffs else float('nan'), 'extrap_diff': extrap_diff}


def build_heldout(notes: Notes, names: dict) -> 'tuple[dict, dict]':
    """Block ``replicas_fuera_de_muestra`` from ``heldout_gap.json`` and ``heldout_catalogue.json``."""
    gap = load_json(os.path.join(RESULTS, 'heldout', 'heldout_gap.json'))
    catalogue = load_json(os.path.join(RESULTS, 'heldout', 'heldout_catalogue.json'))
    models = gap['models']
    replicas = {r['scenario']: r for r in catalogue['replicas']}
    reps = list(replicas)
    sampling = catalogue['sampling_distribution']
    gaps = {k: m['gap'] for k, m in models.items()}
    optuna = [k for k in models if models[k]['selection'] == 'optuna']
    moved = [k for k in sorted(models, key=lambda k: models[k]['rank_reference'])
             if models[k]['rank_reference'] != models[k]['rank_heldout']]
    weighted_vs_trf = gap['weighted_vs_transformer_heldout']
    differences = {r: models[WEIGHTED]['replica_means'][r] - models[TRANSFORMER]['replica_means'][r] for r in reps}
    minimum, maximum = min(gaps, key=lambda k: gaps[k]), max(gaps, key=lambda k: gaps[k])
    json_text = {key: Pending(value) for key, value in notes.mapping('heldout_json').items()}
    block = {
        'fuente': json_text['fuente'], 'proposito': json_text['proposito'],
        'procedimiento': json_text['procedimiento'], 'exclusion': json_text['exclusion'],
        'distribucion_de_sorteo': {
            name_es: {'valores': sampling[name]['values'], 'conteos': sampling[name]['counts'],
                      'probabilidades': [round(p, 4) for p in sampling[name]['probabilities']],
                      'n': sampling[name]['n'],
                      'fuente': f'{os.path.basename(sampling[name]["source"])} de la referencia '
                                f'(idéntico en los {len(models)} paquetes)'}
            for name, name_es in (('severity', 'severidad'), ('length', 'longitud'))},
        'replicas': {k: {'semilla': r['effective_seed'], 'n_secuencias': r['structure']['n_sequences'],
                         'n_pasos': r['structure']['n_trials'],
                         'bloques_x_secuencias': f'{r["structure"]["num_blocks"]}x'
                                                 f'{r["structure"]["num_sequences_per_block"]}'}
                     for k, r in replicas.items()},
        'total': {'n_secuencias': sum(r['structure']['n_sequences'] for r in replicas.values()),
                  'n_pasos': sum(r['structure']['n_trials'] for r in replicas.values())},
        'verificacion_copias': json_text['verificacion_copias'],
        'convenciones': {k: Pending(v) for k, v in notes.mapping('heldout_convenciones').items()},
        'modelos': {k: {'nombre_en_texto': names[k], 'seleccion': SELECTION_JSON[m['selection']],
                        'referencia_media': r6(m['reference_mean']), 'referencia_sd_ddof1': r6(m['reference_std']),
                        'n_referencia': m['n_reference'], 'fuera_de_muestra_media': r6(m['heldout_mean']),
                        'fuera_de_muestra_sd': r6(m['heldout_std']), 'n_fuera_de_muestra': m['n_heldout'],
                        'medias_por_replica': {r: r6(m['replica_means'][r]) for r in reps},
                        'sd_entre_replicas': r6(m['sd_between_replicas']), 'caida': r6(m['gap']),
                        'caida_se': r6(m['gap_se']), 'cohen_d': r6(m['cohen_d']), 'welch_p': r6(m['welch_p']),
                        'welch_log10p': r6(m['welch_log10_p']), 'rango_referencia': m['rank_reference'],
                        'rango_fuera_de_muestra': m['rank_heldout']} for k, m in models.items()},
        'hechos_derivados': {
            'caida_min': {'modelo': minimum, 'valor': r6(gaps[minimum])},
            'caida_max': {'modelo': maximum, 'valor': r6(gaps[maximum])},
            'caida_rango_11_optuna': [r6(min(gaps[k] for k in optuna)), r6(max(gaps[k] for k in optuna))],
            'max_abs_d': r6(max(abs(m['cohen_d']) for m in models.values())),
            'min_welch_p': r6(min(m['welch_p'] for m in models.values())),
            'cambios_de_rango': {k: [models[k]['rank_reference'], models[k]['rank_heldout']] for k in moved}},
        'spearman_referencia_vs_fuera_de_muestra': {
            'rho': r6(gap['spearman_reference_vs_heldout']['rho']),
            'p': round_significant(gap['spearman_reference_vs_heldout']['p'], 4),
            'n_modelos': gap['spearman_reference_vs_heldout']['n_models']},
        'ensamble_ponderado_vs_transformer': {
            'diferencia_media': r6(weighted_vs_trf['mean_difference']),
            't_pareada_p': round_significant(weighted_vs_trf['paired_t_p'], 4),
            'wilcoxon_p': round_significant(weighted_vs_trf['wilcoxon_p'], 4),
            'replicas_ganadas': weighted_vs_trf['replicas_won'], 'n_replicas': weighted_vs_trf['n_replicas'],
            'n_secuencias': weighted_vs_trf['n_sequences'],
            'diferencia_por_replica': {r: r6(v) for r, v in differences.items()}},
        'afirmacion_tesis': json_text['afirmacion_tesis'], 'salvedad': json_text['salvedad'],
        'figuras': json_text['figuras'],
        'reproduccion': [Pending(item) for item in notes.items('heldout_reproduccion')]}
    copies = [status for r in catalogue['replicas'] for status in r['work_copies'].values()]
    inverted = [(a, b) for i, a in enumerate(models) for b in list(models)[i + 1:]
                if (models[a]['reference_mean'] - models[b]['reference_mean'])
                * (models[a]['heldout_mean'] - models[b]['heldout_mean']) < 0]
    first_ref = min(models, key=lambda k: models[k]['rank_reference'])
    first_held = min(models, key=lambda k: models[k]['rank_heldout'])
    max_optuna = max(optuna, key=lambda k: gaps[k])
    helper = {
        'order': sorted(models, key=lambda k: models[k]['rank_reference']), 'reps': reps, 'models': models,
        'sampling': sampling, 'replicas': replicas, 'differences': differences, 'moved': moved,
        'derived': {
            'ho_n_optuna': len(optuna), 'ho_semillas': f'{replicas[reps[0]]["effective_seed"]}..'
                                                       f'{replicas[reps[-1]]["effective_seed"]}',
            'ho_se_min': min(m['gap_se'] for m in models.values()), 'ho_se_max': max(m['gap_se'] for m in models.values()),
            'ho_caida_min_nombre': names[minimum], 'ho_caida_max_nombre': names[maximum],
            'ho_caida_max_optuna_nombre': names[max_optuna],
            'ho_caida_max_nota': '' if models[maximum]['selection'] == 'optuna' else ', que no se optimizó',
            'ho_caida_max_nota_corta': '' if models[maximum]['selection'] == 'optuna' else ', no se optimizó',
            'ho_significativa': ('nunca significativa' if min(m['welch_p'] for m in models.values()) >= 0.05
                                 else 'significativa en algún modelo'),
            'ho_copias': (f'{sum(1 for c in catalogue["replicas"][0]["work_copies"].values() if c == "match")} de '
                          f'{len(models)} coinciden en las {len(reps)} réplicas' if all(c == 'match' for c in copies)
                          else f'hay copias que no coinciden ({copies.count("mismatch")} distintas, '
                               f'{copies.count("missing")} ausentes)'),
            'ho_swap_max_diff': max((max(abs(models[a]['reference_mean'] - models[b]['reference_mean']),
                                         abs(models[a]['heldout_mean'] - models[b]['heldout_mean']))
                                     for a, b in inverted), default=0.0),
            'ho_cambios_rango': ', '.join(f'{names[k]} {models[k]["rank_reference"]} → {models[k]["rank_heldout"]}'
                                          for k in moved) or 'ninguno',
            'ho_primero': (f'{with_article(names[first_ref])} es 1.º en ambas.' if first_ref == first_held
                           else f'1.º en la referencia: {names[first_ref]}; en las réplicas: {names[first_held]}.'),
            'ho_n_celdas': len(models) * len(reps)}}
    return block, helper


def derived_values(ctx: dict) -> dict:
    """Derived namespace ``_`` available to the notes placeholders.

    Keys: ``n_gen``, ``n_heldout``, ``n_ref_gen``, ``n_modelos``, ``n_estructurales``, ``n_sev_en_rango``,
    ``familias_conteo``, ``max_asignacion``, ``n_acciones``, ``dim_entrada``, ``presupuesto_total``,
    ``preasignados``, ``presupuesto_agente``, ``max_pasos``, ``min_pasos``, ``max_severidad``, ``dim_r/t/s``,
    ``n_estados``, ``alpha``, ``beta``, ``ref_bloques``, ``ref_secuencias_bloque``, ``sev_ref_min/max``,
    ``semilla_escenarios``, ``semilla_aleatorio``, ``kl_bins``, ``kl_eps``, ``kl_eps_tex``, ``ver/<paquete>``,
    ``cfg/<pkg>/<CONSTANTE>``, ``tabla_q``, ``arq/*``, ``a2c_lr_min_pct``, ``a2c_coef_gasto``,
    ``gen_otros_opt_min/max``, ``ind_peor_len_long``, ``n_ens_peor_len_long``, ``n_struct_identicos``,
    ``reproducen_score``, ``n_ens_optuna``, ``ens_score_coincide``, ``ens_tau_optimizado``,
    ``ens_extrap_coincide``, ``dec_max_diff``, ``dec_max_diff_heldout``, ``dec_heldout_en_rango`` and the
    held-out summaries ``ho_*`` of :func:`build_heldout`.
    """
    data, suites, gen, held, names = ctx['data'], ctx['suites'], ctx['gen'], ctx['held'], ctx['names']
    cfg = {pkg: load_config(pkg) for pkg in ('pes_dqn', 'pes_rdqn', 'pes_trf', 'pes_a2c', WEIGHTED)}
    base = cfg['pes_trf']
    budget = int(base.AVAILABLE_RESOURCES_PER_SEQUENCE) - random_baseline.PREASSIGNED_RESOURCES
    n_actions = int(base.MAX_ALLOCATABLE_RESOURCES) + 1
    ref_cell = suites['individual'].cells[TRANSFORMER][REFERENCE_SCENARIO]
    severity_path, _ = random_baseline.find_baseline_paths(TRANSFORMER)
    severities = numpy.loadtxt(severity_path, delimiter=',')
    families = [suites['individual'].cells[TRANSFORMER][s]['family'] for s in gen]
    structural = [s for s, f in zip(gen, families) if f == 'structural']
    individual, ensembles = data['modelos_individuales'], data['ensambles']
    kl_default = inspect.signature(kl_divergence).parameters['epsilon'].default
    a2c_best = load_json(os.path.join(package_dir('pes_a2c'), 'inputs', 'best_params.json'))

    def layers(units, with_relu: bool = True, arrow: str = ' -> ', sep: str = ',') -> str:
        suffix = f'{sep}ReLU' if with_relu else ''
        return arrow.join(f'Dense({u}{suffix})' for u in units)
    struct_same = 0
    for suite in suites.values():
        for pkg in suite.models:
            ref = suite.cells[pkg][REFERENCE_SCENARIO]
            struct_same += all(suite.cells[pkg][s]['global_mean_perf'] == ref['global_mean_perf']
                               and suite.cells[pkg][s]['std_perf'] == ref['std_perf'] for s in structural)
    others = [individual[p]['generalizacion_media_21'] for p in individual if p not in (TRANSFORMER, 'pes_base')]
    optimised_ens = [p for p in ensembles if 'score_optuna' in ensembles[p]['definicion']]
    tau_opt = [p for p in optimised_ens if ensembles[p]['definicion']['parametros'].get('tau', 1.0) != 1.0]
    decisions = data['decisiones_ensambles']
    outside = []
    for pkg, events in DECISION_EVENTS.items():
        for key, label, _counter, held_key in events:
            ref, gen_share = decisions[pkg][key]['referencia'], decisions[pkg][key]['generalizacion']
            value = decisions[pkg]['replicas_fuera_de_muestra'][held_key]
            if None in (ref, gen_share, value):
                continue
            if not min(ref, gen_share) <= value <= max(ref, gen_share):
                outside.append(f'{names[pkg]}: {label[0].lower() + label[1:]} {value:.1f} frente a '
                               f'{ref:.1f}–{gen_share:.1f}')
    n_events = sum(len(v) for v in DECISION_EVENTS.values())
    derived = {
        'n_gen': len(gen), 'n_heldout': len(held), 'n_ref_gen': len(gen) + 1,
        'n_fuera_de_rango': len(ctx['out_of_range']),
        'n_modelos': sum(len(s.models) for s in suites.values()), 'n_estructurales': len(structural),
        'n_sev_en_rango': sum(1 for s, f in zip(gen, families) if f == 'severity' and s not in ctx['out_of_range']),
        'familias_conteo': ' · '.join(f'{FAMILY_ES[f]} {families.count(f)}' for f in FAMILIES),
        'max_asignacion': int(base.MAX_ALLOCATABLE_RESOURCES), 'n_acciones': n_actions, 'dim_entrada': 3,
        'presupuesto_total': int(base.AVAILABLE_RESOURCES_PER_SEQUENCE),
        'preasignados': random_baseline.PREASSIGNED_RESOURCES, 'presupuesto_agente': budget,
        'max_pasos': int(base.NUM_MAX_TRIALS), 'min_pasos': int(base.NUM_MIN_TRIALS),
        'max_severidad': int(base.MAX_SEVERITY), 'dim_r': budget + 1, 'dim_t': int(base.NUM_MAX_TRIALS) + 1,
        'dim_s': int(base.MAX_SEVERITY) + 1,
        'n_estados': (budget + 1) * (int(base.NUM_MAX_TRIALS) + 1) * (int(base.MAX_SEVERITY) + 1),
        'alpha': float(base.PANDEMIC_PARAMETER), 'beta': 1.0 + float(base.PANDEMIC_PARAMETER),
        'ref_bloques': int(ref_cell['num_blocks']), 'ref_secuencias_bloque': int(ref_cell['num_sequences_per_block']),
        'sev_ref_min': int(severities.min()), 'sev_ref_max': int(severities.max()),
        'semilla_escenarios': int(ref_cell['seed']), 'semilla_aleatorio': int(base.SEED),
        'kl_bins': inspect.signature(histogram_pmf).parameters['bins'].default, 'kl_eps': f'1e{int(round(math.log10(kl_default)))}',
        'kl_eps_tex': f'10^{{{int(round(math.log10(kl_default)))}}}',
        'ver': {'python': f'{sys.version_info.major}.{sys.version_info.minor}',
                **{name: '.'.join(importlib.metadata.version(name).split('.')[:2])
                   for name in ('tensorflow', 'keras', 'numpy', 'optuna', 'gymnasium', 'scipy')}},
        'cfg': {pkg: {k: getattr(c, k) for k in dir(c) if k.isupper()} for pkg, c in cfg.items()},
        'tabla_q': f'{budget + 1}x{int(base.NUM_MAX_TRIALS) + 1}x{int(base.MAX_SEVERITY) + 1}x{n_actions} = '
                   f'{thousands((budget + 1) * (int(base.NUM_MAX_TRIALS) + 1) * (int(base.MAX_SEVERITY) + 1) * n_actions)}'
                   f' valores',
        'arq': {'dqn_capas': layers(cfg['pes_dqn'].DQN_HIDDEN_UNITS),
                'dqn_capas_md': layers(cfg['pes_dqn'].DQN_HIDDEN_UNITS, arrow=' → ', sep=', '),
                'rdqn_capas': layers(cfg['pes_rdqn'].RDQN_HIDDEN_UNITS),
                'rdqn_capas_corto': layers(cfg['pes_rdqn'].RDQN_HIDDEN_UNITS, with_relu=False, arrow=' → '),
                'trf_capas': layers(cfg['pes_trf'].TRF_HIDDEN_UNITS),
                'trf_capas_md': layers(cfg['pes_trf'].TRF_HIDDEN_UNITS, arrow=' → ', sep=', '),
                'a2c_actor_capas': layers(a2c_best['actor_hidden_units'], arrow='->'),
                'a2c_critico_capas': layers(a2c_best['critic_hidden_units'], arrow='->'),
                'a2c_actor_corto': layers(a2c_best['actor_hidden_units'], with_relu=False, arrow='→'),
                'a2c_critico_corto': layers(a2c_best['critic_hidden_units'], with_relu=False, arrow='→')},
        'a2c_lr_min_pct': 100.0 * float(a2c_best['hyperparameters']['lr_min_ratio']),
        'a2c_coef_gasto': float(a2c_best['hyperparameters']['spend_cost_coeff']),
        'gen_otros_opt_min': min(others), 'gen_otros_opt_max': max(others),
        'ind_peor_len_long': ', '.join(names[p] for p in individual
                                       if individual[p]['peor_escenario'] == 'len_extrapolate_long'),
        'n_ens_peor_len_long': sum(1 for p in ensembles if ensembles[p]['peor_escenario'] == 'len_extrapolate_long'),
        'n_struct_identicos': struct_same,
        'reproducen_score': ('reproducen' if all(individual[p]['hiperparametros'].get('reproduce_score_optuna', True)
                                                 for p in individual) else 'no reproducen todos'),
        'n_ens_optuna': len(optimised_ens),
        'ens_score_coincide': ('coincide' if all(ensembles[p]['definicion']['score_optuna_igual_a_referencia_benchmark']
                                                 for p in optimised_ens) else 'no coincide en todos'),
        'ens_tau_optimizado': ('Sólo ' + ', '.join('el ' + names[p].lower() for p in tau_opt) + ' optimiza τ ('
                               + ', '.join(format_value(ensembles[p]['definicion']['parametros']['tau'], '.3f,')
                                           for p in tau_opt) + '); los demás usan τ = 1.') if tau_opt
                              else 'Ninguno optimiza τ (τ = 1).',
        'ens_extrap_coincide': ('—' if ctx['decisions']['extrap_diff'] is None else 'todas las decisiones'
                                if ctx['decisions']['extrap_diff'] == 0
                                else f'no todas las decisiones ({ctx["decisions"]["extrap_diff"]} distintas)'),
        'dec_max_diff': ctx['decisions']['max_diff'], 'dec_max_diff_heldout': ctx['decisions']['max_diff_heldout'],
        'dec_heldout_en_rango': (f'Quedan dentro del rango referencia–generalización en {n_events - len(outside)} de '
                                 f'{n_events} eventos' + (f'; fuera: {"; ".join(outside)}' if outside else '')
                                 + '. La tesis no agrega una columna para las réplicas.'),
        **ctx['heldout_derived']}
    return derived


def build_data(args, notes: Notes) -> 'tuple[dict, dict]':
    """Build the full JSON structure and the helper context for the Markdown."""
    suites = {name: Suite(name) for name in SUITES}
    scenarios = list(suites['individual'].summary['scenarios'])
    if scenarios != list(suites['ensemble'].summary['scenarios']):
        raise ValueError('Los dos grupos no tienen los mismos escenarios')
    gen, held = generalisation_scenarios(scenarios), heldout_scenarios(scenarios)
    main_scenarios = [REFERENCE_SCENARIO] + gen
    names = {row['Paquete']: row['Nombre en el texto'] for row in notes.table('paquetes')}
    abbreviations = {row['Paquete']: row['Abreviatura'] for row in notes.table('paquetes')}
    individual, ensemble = suites['individual'], suites['ensemble']
    out_of_range = [row['Escenario'] for row in notes.table('escenarios') if row['Fuera de rango'] == 'sí']
    ql = {'ref': individual.mean(QLEARNING, REFERENCE_SCENARIO),
          'gen': float(numpy.mean([individual.mean(QLEARNING, s) for s in gen]))}
    raw: dict = {}
    blocks = {'individual': {}, 'ensemble': {}}
    for suite in suites.values():
        for pkg in suite.models:
            reference = suite.mean(pkg, REFERENCE_SCENARIO)
            detail = (individual_hyperparameters(pkg, notes, reference) if suite.name == 'individual'
                      else ensemble_definition(pkg, notes, reference))
            blocks[suite.name][pkg], raw[pkg] = model_entry(suite, pkg, gen, ql, names, detail, out_of_range)

    # Agente aleatorio
    seed = int(load_config(TRANSFORMER).SEED)
    random_raw, random_norm = random_baseline.run_random_player(TRANSFORMER, seed)
    random_text = notes.mapping('agente_aleatorio')

    # Escenarios
    catalogue = {row['Escenario']: row for row in notes.table('escenarios')}
    heldout_catalogue = {r['scenario']: r for r in load_json(os.path.join(RESULTS, 'heldout',
                                                                          'heldout_catalogue.json'))['replicas']}
    escenarios = {}
    for scenario in scenarios:
        cell = individual.cells[TRANSFORMER][scenario]
        sizes = {int(s.cells[p][scenario]['n_sequences']) for s in suites.values() for p in s.models}
        if len(sizes) != 1:
            print(f'AVISO: {scenario} tiene distinto número de secuencias según el modelo: {sizes}', file=sys.stderr)
        entry = {'familia': FAMILY_ES[cell['family']], 'n_secuencias': int(cell['n_sequences']),
                 'n_pasos': len(cell['per_trial_actions']),
                 'bloques_x_secuencias': f'{cell["num_blocks"]}x{cell["num_sequences_per_block"]}'}
        if scenario in held:
            k = scenario.replace('heldout_s', '')
            local = {'k': k, 'semilla': heldout_catalogue[scenario]['effective_seed'],
                     'bloques': cell['num_blocks'], 'secuencias_bloque': cell['num_sequences_per_block']}
            entry.update({'descripcion': Pending(notes.text('escenario_heldout'), local),
                          'nombre_en_texto': 'réplica fuera de muestra', 'fuera_de_rango': False,
                          'semilla': local['semilla'], 'excluido_de_generalizacion': True})
        else:
            row = catalogue[scenario]
            entry.update({'descripcion': Pending(row['Descripción']),
                          'nombre_en_texto': None if row['Nombre en el texto'] == '—' else row['Nombre en el texto'],
                          'fuera_de_rango': row['Fuera de rango'] == 'sí'})
        escenarios[scenario] = entry
    check_out_of_range(escenarios)

    # Celdas y ganadores
    celdas = {pkg: {s: cell_entry(suite, pkg, s) for s in main_scenarios} for suite in suites.values() for pkg in suite.models}
    ganadores = {}
    for scenario in main_scenarios:
        ind_means = {p: round(individual.mean(p, scenario), 6) for p in individual.models}
        ens_means = {p: round(ensemble.mean(p, scenario), 6) for p in ensemble.models}
        ganadores[scenario] = {
            'individual': winners(ind_means), 'ensemble': winners(ens_means), 'todos': winners({**ind_means, **ens_means}),
            'ensambles_menos_transformer': {p: r6(ens_means[p] - ind_means[TRANSFORMER]) for p in ensemble.models}}
    hechos = derived_facts(individual, ensemble, main_scenarios, gen, ganadores)
    hechos['pes_ens_peor_menos_trf_peor'] = r6(raw[WEIGHTED]['worst_mean'] - raw[TRANSFORMER]['worst_mean'])
    trf_cells = individual.cells[TRANSFORMER]
    comparison = {}
    for pkg in ensemble.models:
        row = compare(ensemble.cells[pkg], trf_cells, [s for s in gen if s in trf_cells])
        comparison[pkg] = {'media_generalizacion': r6(row['gen_mean']), 'diferencia': r6(row['difference']),
                           'cohen_d': r6(row['cohen_d']), 'welch_p': round_significant(row['welch_p'], 4),
                           'welch_log10p': r6(row['welch_log10_p']), 'escenarios_mayor_media': row['wins'],
                           'n_escenarios': row['n_scenarios']}
    hechos['ensambles_vs_transformer_generalizacion'] = comparison

    # Pares
    pares, matrices = {}, {}
    for suite in suites.values():
        pairwise = suite.metrics['pairwise']
        models = pairwise['models']
        matrices[suite.name] = pairwise
        pares[suite.name] = {f'{a} vs {b}': {'d': r6(pairwise['cohen_d'][i][j]),
                                              'log10p': r6(pairwise['welch_log10_p'][i][j]),
                                              'kl_sym': r6(pairwise['symmetric_kl'][i][j])}
                             for i, a in enumerate(models) for j, b in enumerate(models) if j > i}
    citados = {}
    for label, pair in notes.mapping('pares_citados').items():
        first, _, second = pair.partition(' vs ')
        suite = next(s for s in suites.values() if first in s.models)
        models = matrices[suite.name]['models']
        i, j = models.index(first), models.index(second)
        citados[label] = {'d': r6(matrices[suite.name]['cohen_d'][i][j]),
                          'log10p': r6(matrices[suite.name]['welch_log10_p'][i][j]),
                          'kl_sym': r6(matrices[suite.name]['symmetric_kl'][i][j])}

    decisions, decision_files = load_decisions(args.decisions)
    decision_block, decision_info = build_decisions(decisions, decision_files, suites, gen, held, notes)
    confidence_path = agent_internals.find_confidences(None)
    confidences, _ = agent_internals.load_confidences(confidence_path)
    valid = confidences[confidences != agent_internals.UNSET_CONFIDENCE]
    heldout_block, heldout_helper = build_heldout(notes, names)
    document, latex = latex_document(notes)
    confidence_figure = 'trf_agent_confidences.png'
    in_thesis = confidence_figure in {os.path.basename(i) for info in latex['figures'].values()
                                      for i in info['imagenes']}
    reproduction = notes.mapping('reproduccion')
    meta_text = notes.mapping('meta')
    git = git_state()
    now = datetime.datetime.now(datetime.timezone.utc)
    data = {
        'meta': {**{k: Pending(v) for k, v in meta_text.items()},
                 'generado_por': 'writings/auxiliar/scripts/build_results_context.py (no editar a mano)',
                 'generado_utc': now.strftime('%Y-%m-%dT%H:%M:%SZ'), 'git': git,
                 'fuente': ('h1/general/results/{individual,ensemble}/ (summary.json, comparison_metrics.json, '
                            'matrices/*.csv), h1/general/results/heldout/, decisiones: '
                            + (', '.join(relative(p) for p in decision_files) or 'ninguna')
                            + '; inputs/best_params.json, inputs/*_BAYESIAN_OPT/ y config/CONFIG.py de cada '
                              'paquete; writings/ (Main.tex, 01_Chapters, References.bib, audit/AUDIT.md). Texto '
                              'redactado: writings/auxiliar/claude_context/mpes_resultados_notas.md.'),
                 'convenciones': {k: (Pending(v) if isinstance(v, str) else v)
                                  for k, v in notes.mapping('convenciones').items()}},
        'entorno': {k: Pending(v) for k, v in notes.mapping('entorno').items()},
        'agente_aleatorio': {'descripcion': Pending(random_text['descripcion']),
                             'rbar_media': round(float(random_norm.mean()), 4),
                             'rbar_sd': round(float(random_norm.std(ddof=1)), 4),
                             'rbar_min': round(float(random_norm.min()), 4), 'rbar_max': round(float(random_norm.max()), 4),
                             'S_cruda_media': round(float(random_raw.mean()), 2),
                             'S_cruda_sd': round(float(random_raw.std(ddof=1)), 2), 'n': int(len(random_norm)),
                             'figura': Pending(random_text['figura'])},
        'modelos_individuales': blocks['individual'],
        'ensambles': blocks['ensemble'],
        'escenarios': escenarios,
        'celdas': celdas,
        'ganadores_por_escenario': ganadores,
        'hechos_derivados': hechos,
        'pares': pares,
        'pares_citados_en_tesis': citados,
        'decisiones_ensambles': decision_block,
        'confianza_transformer_registro': {
            'decisiones_con_recursos_referencia': int(len(valid)), 'confianza_media': round(float(valid.mean()), 3),
            'calculo': Pending(notes.mapping('confianza_trf')['calculo']),
            'figura': (f'{"en la tesis" if in_thesis else "no incluida en la tesis"} '
                       f'(h1/general/results/agent_internals/{confidence_figure})'),
            'fuente': (f'general.scripts.agent_internals.load_confidences({relative(confidence_path)})'
                       + ('; Apéndice ap:trf-conf' if 'ap:trf-conf' in latex['labels'] else ''))},
        'degradacion_media_por_familia_promedio_del_grupo': {
            suite.name: {FAMILY_ES[f]: r6(numpy.mean([raw[p]['family_deg'][f] for p in suite.models]))
                         for f in FAMILIES} for suite in suites.values()},
        'replicas_fuera_de_muestra': heldout_block,
        'advertencias_metodologicas': [Pending(item) for item in notes.items('advertencias')],
        'puntos_de_atencion_borrador_actual': [
            {'severidad': row['Severidad'], 'archivo': row['Archivo'], 'lugar': row['Lugar'],
             'hallazgo': Pending(row['Hallazgo']), 'correccion': Pending(row['Corrección sugerida'])}
            for row in notes.table('puntos_de_atencion')],
        'documento_latex': document,
        'reproduccion': {'entorno': Pending(reproduction['entorno']),
                         'benchmark': notes.items('reproduccion_benchmark'),
                         'tesis_desde_raiz': notes.items('reproduccion_tesis'),
                         'replicas_fuera_de_muestra': Pending(reproduction['replicas_fuera_de_muestra']),
                         'contexto': Pending(reproduction['contexto'])}}
    ctx = {'out_of_range': out_of_range, 'random': (random_raw, random_norm), 'data': data, 'suites': suites, 'gen': gen, 'held': held, 'main': main_scenarios, 'names': names,
           'abbr': abbreviations, 'raw': raw, 'decisions': decision_info, 'heldout': heldout_helper,
           'heldout_derived': heldout_helper['derived'], 'latex': latex, 'git': git, 'now': now}
    lookup = dict(data)
    lookup['_'] = derived_values(ctx)
    ctx['lookup'] = lookup
    resolve_pending(data, lookup)
    return data, ctx


def check_out_of_range(escenarios: dict) -> None:
    """Warn when the notes' out-of-range flag disagrees with the materialised CSVs of ``h1/general/work``."""
    folder = os.path.join(WORK, 'pes_dqn', 'scenarios')
    if not os.path.isdir(folder):
        return
    cfg = load_config('pes_dqn')
    for scenario, entry in escenarios.items():
        path = os.path.join(folder, scenario)
        if not os.path.isdir(path):
            continue
        severity = numpy.loadtxt(os.path.join(path, 'initial_severity.csv'), delimiter=',')
        lengths = numpy.loadtxt(os.path.join(path, 'sequence_lengths.csv'), delimiter=',')
        actual = bool(severity.max() > cfg.MAX_SEVERITY or lengths.max() > cfg.NUM_MAX_TRIALS)
        if actual != entry['fuera_de_rango']:
            print(f'AVISO: {scenario}: fuera_de_rango en las notas = {entry["fuera_de_rango"]}, CSV = {actual}',
                  file=sys.stderr)


def derived_facts(individual: Suite, ensemble: Suite, scenarios: 'list[str]', gen: 'list[str]',
                  ganadores: dict) -> dict:
    """Block ``hechos_derivados`` (counts of scenarios won by the Transformer and by each ensemble)."""
    exact = [s for s in scenarios if TRANSFORMER in ganadores[s]['individual']['ganador']]
    first2, ties = [], []
    for scenario in scenarios:
        rounded = {p: round(r6(individual.mean(p, scenario)) or 0.0, 2) for p in individual.models}
        top = max(rounded.values())
        if rounded[TRANSFORMER] == top:
            first2.append(scenario)
            if sum(1 for v in rounded.values() if v == top) > 1:
                ties.append(scenario)
    beats = {}
    for pkg in ensemble.models:
        won = [s for s in gen if round(ensemble.mean(pkg, s), 6) > round(individual.mean(TRANSFORMER, s), 6)]
        beats[pkg] = {'n': len(won), 'escenarios': won}
    return {'transformer_gana_individuales_exacto': {'n': len(exact), 'escenarios': exact},
            'transformer_primero_o_empatado_a_2_decimales': {'n': len(first2), 'escenarios': first2,
                                                            'empates_2_decimales': ties},
            'escenarios_donde_ensamble_supera_al_transformer': beats}


##########################
##  Markdown            ##
##########################
def md_table(header: 'list[str]', align: str, rows: 'list[list[str]]') -> 'list[str]':
    """Render a Markdown table; ``align`` has one of ``l``, ``r``, ``c`` per column."""
    marks = {'l': '---', 'r': '---:', 'c': ':---:'}
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(marks[a] for a in align) + '|']
    lines += ['| ' + ' | '.join(row) + ' |' for row in rows]
    return lines


def md_cell(text: str) -> str:
    """Escape the pipes of a table cell."""
    return str(text).replace('|', '\\|')


class Markdown:
    """Renderer of ``mpes_resultados.md`` from the resolved data, the build context and the notes."""

    def __init__(self, data: dict, ctx: dict, notes: Notes) -> None:
        self.data, self.ctx, self.notes = data, ctx, notes
        self.lines: 'list[str]' = []
        self.names, self.abbr = ctx['names'], ctx['abbr']
        self.out_of_range = {s for s, e in data['escenarios'].items() if e['fuera_de_rango']}

    def add(self, *lines: str) -> None:
        """Append lines."""
        self.lines.extend(lines)

    def note(self, name: str) -> str:
        """Rendered text block of the notes."""
        return render(self.notes.text(name), self.ctx['lookup'], 'md')

    def note_items(self, name: str) -> 'list[str]':
        """Rendered list items of the notes."""
        return [render(item, self.ctx['lookup'], 'md') for item in self.notes.items(name)]

    def bullets(self, name: str) -> None:
        """Append the items of a notes block as bullets."""
        self.add(*[f'- {item}' for item in self.note_items(name)])

    def scenario_label(self, scenario: str) -> str:
        """Scenario id in backticks, with ``⚠`` when out of range."""
        return f'`{scenario}`' + (' ⚠' if scenario in self.out_of_range else '')

    # -- secciones ---------------------------------------------------------------------------------------------
    def header(self) -> None:
        """Title and provenance line."""
        git, now = self.ctx['git'], self.ctx['now']
        dirty = ' (con cambios sin confirmar)' if git['sucio'] else ''
        self.add('# mPES — Síntesis de métricas, salidas y resultados (contexto para la tesis)', '',
                 f'> Generado por `writings/auxiliar/scripts/build_results_context.py` — no editar a mano. '
                 f'Fecha (UTC): {now.strftime("%Y-%m-%dT%H:%M:%SZ")}; commit `{git["commit"]}`{dirty}. '
                 + self.note('encabezado'), '')

    def summary_sections(self) -> None:
        """Sections 0 to 3."""
        self.add('## 0. Resumen ejecutivo (hallazgos verificados)', '')
        self.add(*[f'{i}. {item}' for i, item in enumerate(self.note_items('resumen_ejecutivo'), 1)], '')
        self.add('## 1. Definiciones', '')
        self.bullets('definiciones')
        self.add('', '## 2. Entorno y protocolo', '')
        self.add(*[f'- **{key}**: {value}' for key, value in self.data['entorno'].items()])
        self.bullets('protocolo_md')
        raw, norm = self.ctx['random']
        self.add('', '## 3. Agente aleatorio (cota inferior de referencia)', '',
                 f'- {self.data["agente_aleatorio"]["descripcion"]}',
                 f'- $\\bar r$: media **{norm.mean():.3f}**, σ {norm.std(ddof=1):.3f}, rango [{norm.min():.3f}; '
                 f'{norm.max():.3f}]; $S_{{cruda}}$ media {raw.mean():.2f} (σ {raw.std(ddof=1):.2f}). '
                 f'Figura `fig:baseline-random`.', '')

    def performance_tables(self, group: str) -> None:
        """Summary and family-degradation tables of one group (§4.2/4.3 or §5.2/5.3)."""
        models = self.data['modelos_individuales' if group == 'individual' else 'ensambles']
        raw = self.ctx['raw']
        n_gen = len(self.ctx['gen'])
        numbers = ('4.2', '4.3') if group == 'individual' else ('5.2', '5.3')
        rows = []
        for pkg in sorted(models, key=lambda p: -raw[p]['gen']):
            r = raw[pkg]
            rows.append([f'`{pkg}`', self.names[pkg], f'{r["ref"]:.3f}', f'{r["ref_sd"]:.3f}', f'{r["gen"]:.3f}',
                         signed(r['deg_mean'], 4), f'`{r["worst"]}` ({r["worst_mean"]:.3f})', f'{r["deg_max"]:.3f}',
                         f'{fixed(100 * r["delta_ref"], 1)} %'])
        self.add(f'### {numbers[0]} Resumen de desempeño (ordenado por generalización)', '')
        self.add(*md_table(['Paquete', 'Nombre en el texto', 'Ref.', 'σ ref.', f'Gen. ({n_gen})', 'Deg. media',
                            'Peor escenario (media)', 'Mayor deg.', 'Δrel vs QL ref.'], 'llrrrrlrr', rows), '')
        n_struct = n_gen - self.ctx['lookup']['_']['n_estructurales']
        rows = [[f'`{pkg}`', *[signed(raw[pkg]['family_deg'][f], 4) for f in FAMILIES],
                 f'{raw[pkg]["no_struct"]:.3f}', f'{raw[pkg]["out_of_range"]:.3f}'] for pkg in models]
        self.add(f'### {numbers[1]} Degradación por familia', '')
        self.add(*md_table(['Paquete', 'Deg. severidad', 'Deg. longitud', 'Deg. conjunta', 'Deg. estructural',
                            f'Media {n_struct} sin estruct. (derivado)',
                            f'Media {len(self.out_of_range)} fuera de rango (derivado)'], 'lrrrrrr', rows), '')
        averages = self.data['degradacion_media_por_familia_promedio_del_grupo'][group]
        label = 'individual' if group == 'individual' else 'de ensambles'
        self.add(f'Promedio del grupo {label}: ' + ', '.join(f'{name} {signed(value, 4)}'
                                                             for name, value in averages.items()) + '.', '')

    def individual_section(self) -> None:
        """Section 4."""
        models = self.data['modelos_individuales']
        origins = self.notes.mapping('origen_hiperparametros')
        table_q = models['pes_base']['hiperparametros']['tabla_Q']
        n_table = int(table_q.split(' = ')[1].split(' valores')[0].replace(' ', ''))
        rows = []
        for pkg, entry in models.items():
            hp = entry['hiperparametros']
            if 'arquitectura' in hp:
                architecture, parameters = hp['arquitectura'], str(hp.get('parametros_entrenables', '—'))
            else:
                architecture = hp['tabla_Q'] if pkg == 'pes_base' else f'tabla Q {table_q}'
                parameters = f'{thousands(n_table)} (tabla)'
            rows.append([f'`{pkg}`', self.names[pkg], hp['tipo'], architecture, parameters, str(hp['episodios']),
                         str(hp['semilla']), render(origins[pkg], self.ctx['lookup'], 'md')])
        self.add('## 4. Modelos individuales', '', '### 4.1 Paquetes, nombres y configuración', '')
        self.add(*md_table(['Paquete', 'Nombre en el texto', 'Tipo', 'Arquitectura / tabla', 'Parámetros', 'Episodios',
                            'Semilla', 'Origen de hiperparámetros'], 'llllrrrl', rows), '')
        self.add(self.note('individuales_tabulares'), '', self.note('individuales_dqn_intro'), '')
        nets = [models[p]['hiperparametros'] for p in ('pes_dqn', 'pes_rdqn', 'pes_trf')]
        cfg = self.ctx['lookup']['_']['cfg']
        windows = ['estado actual', f'ventana de {cfg["pes_rdqn"]["RDQN_HISTORY_LEN"]} estados',
                   f'ventana de {cfg["pes_trf"]["TRF_HISTORY_LEN"]} estados']

        def comma(text: str) -> str:
            return text.replace('.', ',')
        rows = [['Entrada', *windows],
                ['Parámetros entrenables', *[thousands(h['parametros_entrenables']) for h in nets]],
                ['Tasa de aprendizaje', *[comma(format(h['lr'], 'g')) for h in nets]],
                ['γ', *[comma(f'{h["gamma"]:.4f}') for h in nets]],
                ['ε0 / εmin', *[comma(f'{h["epsilon_0"]:.4f} / {h["epsilon_min"]:.4f}') for h in nets]],
                ['w / q', *[comma(f'{h["w"]:.4f} / {h["q"]:.4f}') for h in nets]],
                ['Lote', *[str(h['batch']) for h in nets]],
                ['Buffer', *[thousands(h['buffer']) for h in nets]],
                ['Sincronización C (pasos)', *[thousands(h['sync_objetivo_C']) for h in nets]],
                ['Recorte del gradiente', *[comma(significant(h['clip_grad'], 4)) for h in nets]],
                ['PBRS κ', *[comma(format(h['pbrs_kappa'], 'g')) if h['pbrs_kappa'] else '0 (sin PBRS)'
                             for h in nets]],
                ['Inicio del aprendizaje f (transiciones)',
                 *[f'{comma(format(h["inicio_aprendizaje_f"], ".4f"))} ({thousands(h["inicio_aprendizaje_transiciones"])})'
                   for h in nets]],
                ['Episodios', *[thousands(h['episodios']) for h in nets]],
                ['Semilla', *[str(h['semilla']) for h in nets]]]
        self.add(*md_table(['Hiperparámetro', 'DQN', 'DQN recurrente', 'Transformer'], 'lrrr', rows), '')
        self.bullets('individuales_notas')
        self.add('')
        self.performance_tables('individual')

    def ensemble_section(self) -> None:
        """Section 5."""
        rows = []
        for pkg, entry in self.data['ensambles'].items():
            definition = entry['definicion']
            parameters = ', '.join(f'{k} = {v}' for k, v in definition['parametros'].items())
            rows.append([f'`{pkg}`', self.names[pkg], definition['regla'], ', '.join(definition['miembros']),
                         parameters, definition['origen_parametros']])
        self.add('## 5. Ensambles', '', '### 5.1 Reglas y parámetros', '')
        self.add(*md_table(['Paquete', 'Nombre en el texto', 'Regla', 'Miembros', 'Parámetros', 'Origen'], 'llllll',
                           rows), '')
        self.bullets('ensambles_notas')
        self.add('')
        self.performance_tables('ensemble')
        decisions = self.data['decisiones_ensambles']
        self.add('### 5.4 Frecuencia con que las reglas fijas cambian la decisión (Tabla `tab:ens-freq`)', '',
                 f'_{decisions["fuente"]} Unidad: {decisions["unidad"]}._', '')
        rows = []
        for pkg in ('pes_ens_trf_guard', 'pes_ens_consensus_prior', WEIGHTED):
            for key, label, _counter, _held in DECISION_EVENTS[pkg]:
                rows.append([self.names[pkg], label, fixed(decisions[pkg][key]['referencia'], 1),
                             fixed(decisions[pkg][key]['generalizacion'], 1)])
        self.add(*md_table(['Ensamble', 'Evento', 'Referencia', 'Generalización'], 'llrr', rows), '')
        self.bullets('decisiones_otros_hechos')
        self.add(f'- {self.note("decisiones_heldout")}', '')

    def scenario_section(self) -> None:
        """Section 6."""
        gen, held = self.ctx['gen'], self.ctx['held']
        rows = [[f'`{scenario}`', entry['familia'], entry['nombre_en_texto'] or '—', entry['descripcion'],
                 str(entry['n_secuencias']), str(entry['n_pasos']), 'sí' if entry['fuera_de_rango'] else '']
                for scenario, entry in self.data['escenarios'].items()]
        self.add(f'## 6. Catálogo de escenarios (1 referencia + {len(gen)} de generalización + {len(held)} réplicas '
                 f'fuera de muestra)', '')
        self.add(*md_table(['Escenario', 'Familia', 'Nombre en el texto', 'Descripción', 'Secuencias', 'Pasos',
                            'Fuera de rango'], 'llllrrc', rows), '')
        self.bullets('catalogo_notas')
        self.add('')

    def matrix(self, title: str, group: str, value) -> None:
        """One model x scenario matrix (§7)."""
        models = self.ctx['suites'][group].models
        rows = [[self.scenario_label(s), *[value(self.data['celdas'][p][s], s) for p in models]]
                for s in self.ctx['main']]
        self.add(title, '', *md_table(['Escenario', *[self.abbr[p] for p in models]], 'l' + 'r' * len(models), rows),
                 '')

    def matrices_section(self) -> None:
        """Section 7."""
        legend = ', '.join(f'{self.abbr[p]} = `{p}` ({self.names[p]})' for s in SUITES
                           for p in self.ctx['suites'][s].models)
        self.add('## 7. Matrices modelo × escenario', '',
                 f'Leyenda de columnas: {legend}. ⚠ = fuera de rango. En negrita, la mayor media de la fila dentro del '
                 f'grupo.', '', self.note('matrices_nota'), '')

        def bold_mean(group: str):
            def cell(entry: dict, scenario: str) -> str:
                top = max(self.data['celdas'][p][scenario]['mu'] for p in self.ctx['suites'][group].models)
                return f'**{entry["mu"]:.3f}**' if entry['mu'] == top else f'{entry["mu"]:.3f}'
            return cell

        def ref_dash(key: str, formatter):
            return lambda e, s: '—' if s == REFERENCE_SCENARIO else formatter(e[key])
        specs = [('7.1', 'Media $\\bar r$ — individuales', 'individual', bold_mean('individual')),
                 ('7.2', 'Media $\\bar r$ — ensambles', 'ensemble', bold_mean('ensemble')),
                 ('7.3', 'Desviación estándar entre secuencias — individuales', 'individual',
                  lambda e, s: f'{e["sd"]:.3f}'),
                 ('7.4', 'Desviación estándar entre secuencias — ensambles', 'ensemble', lambda e, s: f'{e["sd"]:.3f}'),
                 ('7.5', 'Degradación ($\\bar r_{ref} - \\bar r_{esc}$) — individuales', 'individual',
                  lambda e, s: signed(e['deg'], 3)),
                 ('7.6', 'Degradación — ensambles', 'ensemble', lambda e, s: signed(e['deg'], 3)),
                 ('7.7', '$d$ de Cohen vs referencia (positivo = mejor en el escenario) — individuales', 'individual',
                  ref_dash('d', lambda v: signed(v, 2))),
                 ('7.8', '$d$ de Cohen vs referencia — ensambles', 'ensemble', ref_dash('d', lambda v: signed(v, 2))),
                 ('7.9', '$\\log_{10} p$ de Welch vs referencia — individuales', 'individual',
                  ref_dash('log10p', lambda v: fixed(v, 1))),
                 ('7.10', '$\\log_{10} p$ de Welch vs referencia — ensambles', 'ensemble',
                  ref_dash('log10p', lambda v: fixed(v, 1))),
                 ('7.11', 'KL de acciones vs referencia — individuales', 'individual',
                  ref_dash('kl_acc', lambda v: fixed(v, 3))),
                 ('7.12', 'KL de acciones vs referencia — ensambles', 'ensemble',
                  ref_dash('kl_acc', lambda v: fixed(v, 3)))]
        for number, title, group, value in specs:
            self.matrix(f'### {number} {title}', group, value)
        self.add('_Mínimo, máximo, fracción de secuencias en el óptimo (`f_opt`), fracción bajo 0,8 (`f_lt08`) y '
                 'asignación media por celda: ver `celdas` en `mpes_resultados.json`._', '')

    def winners_section(self) -> None:
        """Section 8."""
        ensemble = self.ctx['suites']['ensemble'].models
        gen, scenarios = self.ctx['gen'], self.ctx['main']
        block = self.data['ganadores_por_escenario']

        def who(entry: dict) -> str:
            return f'{", ".join(self.abbr[p] for p in entry["ganador"])} ({entry["media"]:.3f})'
        rows = [[f'`{s}`', who(block[s]['individual']), who(block[s]['ensemble']), who(block[s]['todos']),
                 *[signed(block[s]['ensambles_menos_transformer'][p], 3) for p in ensemble]] for s in scenarios]
        self.add('## 8. Ganadores por escenario y ensambles frente al Transformer', '')
        self.add(*md_table(['Escenario', 'Mejor individual', 'Mejor ensamble', 'Mejor global',
                            *[f'{self.abbr[p]} − TRF' for p in ensemble]], 'llll' + 'r' * len(ensemble), rows), '')
        facts = self.data['hechos_derivados']
        exact = facts['transformer_gana_individuales_exacto']
        two = facts['transformer_primero_o_empatado_a_2_decimales']
        ties = ', '.join(f'`{s}`' for s in two['empates_2_decimales']) or 'ninguno'
        others = [f'`{s}` ({", ".join(self.abbr[p] for p in block[s]["individual"]["ganador"])})'
                  for s in scenarios if TRANSFORMER not in block[s]['individual']['ganador']]
        self.add(f'- El Transformer tiene la mayor media individual (exacta) en **{exact["n"]}** de {len(scenarios)} '
                 f'escenarios; con dos decimales queda primero o empatado en {two["n"]} (empates a 2 decimales: '
                 f'{ties}).',
                 f'- Escenarios donde el mejor individual no es el Transformer (exacto): {", ".join(others)}.')
        for pkg in ensemble:
            beats = facts['escenarios_donde_ensamble_supera_al_transformer'][pkg]
            margin = max(block[s]['ensambles_menos_transformer'][pkg] for s in gen)
            self.add(f'- {self.names[pkg]} supera al Transformer en {beats["n"]} de {len(gen)} escenarios de '
                     f'generalización (margen máximo {signed(margin, 3)}).')
        self.add(f'- Peor escenario del ensamble ponderado − peor escenario del Transformer = '
                 f'{signed(facts["pes_ens_peor_menos_trf_peor"], 3)}.', '')
        comparison = facts['ensambles_vs_transformer_generalizacion']
        rows = [[self.names[p], f'{c["media_generalizacion"]:.3f}', signed(c['diferencia'], 3), signed(c['cohen_d'], 2),
                 f'{c["welch_p"]:.1e}' if c['welch_p'] < 0.01 else f'{c["welch_p"]:.2f}', fixed(c['welch_log10p'], 1),
                 f'{c["escenarios_mayor_media"]} de {c["n_escenarios"]}']
                for p, c in sorted(comparison.items(), key=lambda item: -item[1]['diferencia'])]
        self.add(f'**Cada ensamble frente al Transformer en generalización** (Tabla `tab:trf-vs-ens`, '
                 f'`writings/auxiliar/scripts/trf_vs_ens.py`: diferencia de las medias de los {len(gen)} escenarios; '
                 f'$d$ y Welch sobre todas sus secuencias juntas; positivo = ventaja del ensamble):', '')
        self.add(*md_table(['Ensamble', 'Gen.', 'Diferencia', '$d$', '$p$ Welch', '$\\log_{10} p$', 'Mayor media en'],
                           'lrrrrrr', rows), '')

    def pairs_section(self) -> None:
        """Section 9."""
        self.add(f'## 9. Comparaciones entre pares ({len(self.ctx["gen"])} escenarios juntos)', '',
                 f'### 9.1 Pares citados en la tesis ({self.note("pares_citados_donde")})', '')
        rows = [[label, f'{v["d"]:.2f}', fixed(v['log10p'], 1), f'{v["kl_sym"]:.2f}']
                for label, v in self.data['pares_citados_en_tesis'].items()]
        self.add(*md_table(['Comparación', '$d$', '$\\log_{10} p$', 'KL sim.'], 'lrrr', rows), '')
        for number, group, label in (('9.2', 'individual', 'individuales'), ('9.3', 'ensemble', 'ensambles')):
            pairwise = self.ctx['suites'][group].metrics['pairwise']
            models = pairwise['models']
            self.add(f'### {number} Matrices de pares — {label}', '')
            for key, title, decimals in (('cohen_d', '**$d$ de Cohen** (fila − columna; positivo = gana la fila):', 2),
                                         ('welch_log10_p', '**$\\log_{10} p$ de Welch**:', 1),
                                         ('symmetric_kl', '**KL simetrizada entre histogramas de desempeño**:', 3)):
                rows = [[self.abbr[a], *['—' if i == j else fixed(pairwise[key][i][j], decimals)
                                         for j in range(len(models))]] for i, a in enumerate(models)]
                self.add(title, '', *md_table(['fila \\ columna', *[self.abbr[m] for m in models]],
                                              'l' + 'r' * len(models), rows), '')

    def reference_section(self) -> None:
        """Section 10."""
        all_models = {**self.data['modelos_individuales'], **self.data['ensambles']}
        blocks = len(next(iter(all_models.values()))['referencia_detalle']['media_por_bloque'])
        per_block = self.ctx['lookup']['_']['ref_secuencias_bloque']
        self.add('## 10. Detalle en la referencia (`sev_base`)', '',
                 f'### 10.1 Media por bloque ({blocks} bloques × {per_block} secuencias)', '')
        rows = [[f'`{p}`', *[f'{v:.3f}' for v in e['referencia_detalle']['media_por_bloque']]]
                for p, e in all_models.items()]
        self.add(*md_table(['Paquete', *[f'B{i}' for i in range(1, blocks + 1)]], 'l' + 'r' * blocks, rows), '')
        n_actions = self.ctx['lookup']['_']['n_acciones']
        self.add(f'### 10.2 Distribución de acciones en la referencia (fracción de pasos con cada asignación '
                 f'0..{n_actions - 1})', '')
        rows = [[f'`{p}`', *[f'{v:.3f}' for v in e['referencia_detalle']['distribucion_acciones_0a10']],
                 f'{self.data["celdas"][p][REFERENCE_SCENARIO]["a_media"]:.2f}'] for p, e in all_models.items()]
        self.add(*md_table(['Paquete', *[str(a) for a in range(n_actions)], 'Asignación media'],
                           'l' + 'r' * (n_actions + 1), rows), '',
                 '_Las distribuciones incluyen los pasos sin recursos (acción forzada 0)._', '')
        registry = self.data['confianza_transformer_registro']
        where = 'Apéndice `ap:trf-conf`' if 'ap:trf-conf' in self.ctx['latex']['labels'] else 'no figura en la tesis'
        self.add(f'### 10.3 Registro de confianza del Transformer ({where})', '',
                 f'- {registry["decisiones_con_recursos_referencia"]} decisiones con recursos disponibles en la '
                 f'referencia; confianza media {registry["confianza_media"]:.3f}. {registry["calculo"]}',
                 f'- Fuente: {registry["fuente"]}; figura {registry["figura"]}.', '')

    def heldout_section(self) -> None:
        """Section 10bis."""
        helper = self.ctx['heldout']
        models, reps, sampling = helper['models'], helper['reps'], helper['sampling']
        block = self.data['replicas_fuera_de_muestra']
        self.add('## 10bis. Réplicas fuera de muestra de la referencia (Sección `sec:res-heldout`, Tabla '
                 '`tab:heldout`)', '', self.note('heldout_intro'), '')
        self.bullets('heldout_puntos')
        self.add('', '**Frecuencias de sorteo** (de los CSV de la referencia):', '')
        for key, label in (('severity', 'Severidad inicial'), ('length', 'Longitud')):
            values = sampling[key]
            self.add(*md_table([label, *[str(v) for v in values['values']]], 'l' + 'r' * len(values['values']),
                               [[f'Conteo (n = {values["n"]})', *[str(c) for c in values['counts']]],
                                ['Probabilidad', *[f'{p:.3f}' for p in values['probabilities']]]]), '')
        rows = [[f'`{k}`', str(v['semilla']), str(v['n_secuencias']), str(v['n_pasos'])]
                for k, v in block['replicas'].items()]
        rows.append(['total', '—', str(block['total']['n_secuencias']), str(block['total']['n_pasos'])])
        self.add(*md_table(['Réplica', 'Semilla', 'Secuencias', 'Pasos'], 'lrrr', rows), '')
        self.add(f'**Referencia frente a réplicas** (Tabla `tab:heldout`; ordenado por la referencia; rango entre los '
                 f'{len(models)} modelos):', '')
        rows = [[f'`{k}`', self.names[k], SELECTION_MD[models[k]['selection']], f'{models[k]["reference_mean"]:.3f}',
                 f'{models[k]["heldout_mean"]:.3f}', f'{models[k]["sd_between_replicas"]:.3f}',
                 signed(models[k]['gap'], 3), signed(models[k]['cohen_d'], 2), f'{models[k]["welch_p"]:.2f}',
                 f'{models[k]["rank_reference"]} → {models[k]["rank_heldout"]}'] for k in helper['order']]
        total = block['total']['n_secuencias']
        self.add(*md_table(['Paquete', 'Nombre en el texto', 'Selección', 'Ref.', f'Fuera de muestra ({total})',
                            'SD entre réplicas', 'Caída', '$d$', '$p$ Welch', 'Rango ref. → f. m.'], 'lllrrrrrrl',
                           rows), '')
        rows = [[f'`{k}`', *[f'{models[k]["replica_means"][r]:.3f}' for r in reps]] for k in helper['order']]
        rows.append(['ENS − TRF', *[signed(helper['differences'][r], 3) for r in reps]])
        self.add('**Media por réplica**:', '',
                 *md_table(['Paquete', *[r.replace('heldout_', '') for r in reps]], 'l' + 'r' * len(reps), rows), '')
        self.bullets('heldout_notas')
        self.add('')

    def closing_sections(self) -> None:
        """Sections 11, 12 and 14 (13 is :meth:`latex_section`)."""
        self.add('## 11. Advertencias metodológicas (deben respetarse al redactar)', '')
        self.bullets('advertencias')
        self.add('', '## 12. Puntos de atención detectados en el borrador LaTeX actual', '',
                 self.note('puntos_de_atencion_intro'), '')
        rows = [[str(i), p['severidad'], f'`{p["archivo"]}`', md_cell(p['lugar']), md_cell(p['hallazgo']),
                 md_cell(p['correccion'])] for i, p in enumerate(self.data['puntos_de_atencion_borrador_actual'], 1)]
        self.add(*md_table(['#', 'Severidad', 'Archivo', 'Lugar', 'Hallazgo', 'Corrección sugerida'], 'rlllll', rows),
                 '')
        self.latex_section()
        repro = self.data['reproduccion']
        self.add('## 14. Reproducción', '', f'- Entorno: {repro["entorno"]}',
                 '- Benchmark (desde `h1/`): ' + '; '.join(f'`{c}`' for c in repro['benchmark']),
                 '- Tesis (desde la raíz): ' + '; '.join(f'`{c}`' for c in repro['tesis_desde_raiz']),
                 f'- Réplicas fuera de muestra: {repro["replicas_fuera_de_muestra"]}',
                 f'- Contexto para LLM: {repro["contexto"]}')

    def latex_section(self) -> None:
        """Section 13 (generated from the thesis sources)."""
        document, latex = self.data['documento_latex'], self.ctx['latex']
        audit = document['estado_audit']
        own = audit['recuento_del_generador']
        warnings = '; '.join(audit['avisos']) or 'ninguno'
        orphans = ', '.join(f'`{o}`' + (' (excluido de Main.tex a propósito)' if o in latex['excluded'] else '')
                            for o in audit['tex_huerfanos']) or 'ninguno'
        status = 'Compilación OK' if audit['compilacion_ok'] else 'Compilación FALLIDA'
        same = all(own[k] == audit[k] for k in own)
        self.add('## 13. Mapa del documento LaTeX', '',
                 f'- Estado del último `audit.py` (`{audit["archivo"]}`): '
                 f'{status} ({audit["paginas"]} páginas), {audit["figuras_con_label"]} figuras y '
                 f'{audit["tablas_con_label"]} tablas con label, {audit["referencias_internas"]} referencias internas, '
                 f'{audit["imagenes"]} imágenes; avisos (⚠/❌): {warnings}; `.tex` no incluidos en Main.tex: {orphans}.',
                 f'- Recuento del generador sobre los `.tex` actuales: {own["figuras_con_label"]} figuras y '
                 f'{own["tablas_con_label"]} tablas con label, {own["referencias_internas"]} referencias internas, '
                 f'{own["imagenes"]} imágenes distintas'
                 + (' (coincide con AUDIT.md).' if same else ' (difiere de AUDIT.md: volver a ejecutar `audit.py`).'),
                 '- Estructura en el repositorio (en el chat los archivos están planos):')
        for folder, content in document['estructura_repo'].items():
            if isinstance(content, dict):
                for sub, names in content.items():
                    self.add(f'  - `{folder}{sub}/`: ' + ', '.join(f'`{n}`' for n in names))
            else:
                self.add(f'  - `{folder}`: ' + ', '.join(f'`{n}`' for n in content))
        rows = [[f'`{c["archivo"]}`', c['seccion'], ', '.join(f'`{lab}`' for lab in c['labels']) or '—']
                for c in document['capitulos']]
        self.add('', *md_table(['Archivo', 'Sección (definida en Main.tex)', 'Etiquetas'], 'lll', rows), '')
        self.add('**Secciones y subsecciones** (numeración calculada; etiqueta entre paréntesis):', '')
        for chapter in document['capitulos']:
            if not chapter['secciones'] and not chapter['label_capitulo']:
                continue
            head = chapter['seccion'] + (f' (`{chapter["label_capitulo"]}`)' if chapter['label_capitulo'] else '')
            parts = [f'{s["numero"]} {s["titulo"]}'.strip() + (f' (`{s["label"]}`)' if s['label'] else '')
                     for s in chapter['secciones']]
            self.add(f'- `{chapter["archivo"]}` — {head}' + (': ' + '; '.join(parts) if parts else ''))
        self.add('', '**Figuras de la tesis** (label → archivo en `02_Images/<carpeta>/`; título corto; capítulo):', '')
        for label, info in latex['figures'].items():
            extra = (f'; {render(latex["figure_notes"][label], self.ctx["lookup"], "md")}'
                     if label in latex['figure_notes'] else '')
            self.add(f'- `{label}` → {" + ".join(info["imagenes"])} ({info["titulo"]}; {info["capitulo"]}{extra})')
            for child, child_info in latex['subfigures'].items():
                if child_info['padre'] == label:
                    self.add(f'  - `{child}` → {" + ".join(child_info["imagenes"])} ({child_info["titulo"]})')
        self.add('', '**Tablas de la tesis** (label → título corto; capítulo):', '')
        self.add(*[f'- `{label}` → {info["titulo"]} ({info["capitulo"]})' for label, info in latex['tables'].items()])
        unused = document['figuras_generadas_no_incluidas']
        self.add('', '**Figuras generadas en `h1/general/results/` que la tesis NO usa** (no están en `02_Images/`):', '')
        for folder, stems in unused.items():
            if folder != 'como_incluirlas':
                self.add(f'- {folder}: ' + (', '.join(f'`{s}`' for s in stems) or 'ninguna'))
        self.add(f'- Cómo incluirlas: {unused["como_incluirlas"]}', '',
                 '**Scripts auxiliares** (`writings/auxiliar/scripts/`, primera línea de su docstring):', '')
        self.add(*[f'- `{name}`: {doc}' for name, doc in document['scripts_auxiliares'].items()])
        missing = ', '.join(f'`{k}`' for k in document['bib_faltantes']) or 'ninguna'
        uncited = ', '.join(f'`{k}`' for k in document['bib_no_citadas']) or 'ninguna'
        self.add('', f'**Claves bibliográficas disponibles en `References.bib`** ({len(document["bib_keys"])}; únicas '
                     f'citables sin agregar entradas): ' + ', '.join(f'`{k}`' for k in document['bib_keys']) + '.',
                 f'- Citadas en los capítulos incluidos: {len(document["bib_citadas"])}; sin citar: {uncited}; citadas '
                 f'pero ausentes del `.bib`: {missing}.', '')

    def build(self) -> str:
        """Render the whole document and warn about labels that no longer exist in the thesis."""
        self.header()
        self.summary_sections()
        self.individual_section()
        self.ensemble_section()
        self.scenario_section()
        self.matrices_section()
        self.winners_section()
        self.pairs_section()
        self.reference_section()
        self.heldout_section()
        self.closing_sections()
        text = '\n'.join(self.lines).rstrip('\n') + '\n'
        body = text.split('## 13. Mapa del documento LaTeX', maxsplit=1)[0]
        unknown = sorted({label for label in LABEL_RE.findall(body) if label not in self.ctx['latex']['labels']})
        if unknown:
            print(f'AVISO: etiquetas citadas en el contexto que no existen en la tesis: {unknown}', file=sys.stderr)
        return text


##########################
##  Escritura y control ##
##########################
VOLATILE_MD = re.compile(r'Fecha \(UTC\): [^;]+; commit `[^`]*`(?: \(con cambios sin confirmar\))?')


def normalise(text: str, kind: str) -> str:
    """Remove the timestamp and git state so that two generations can be compared."""
    text = text.replace('\r\n', '\n')
    if kind == 'md':
        return VOLATILE_MD.sub('Fecha (UTC): -; commit `-`', text)
    data = json.loads(text)
    for key in ('generado_utc', 'git'):
        data.get('meta', {}).pop(key, None)
    return json.dumps(data, indent=1, ensure_ascii=False)


def leaves(node, prefix: str = '') -> dict:
    """Flatten a JSON structure to ``{path: value}``."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            out.update(leaves(value, f'{prefix}/{key}' if prefix else str(key)))
        return out
    if isinstance(node, list) and any(isinstance(v, (dict, list)) for v in node):
        out = {}
        for index, value in enumerate(node):
            out.update(leaves(value, f'{prefix}/{index}'))
        return out
    return {prefix: node}


def diff_summary(old: str, new: str, kind: str, limit: int = 40) -> 'list[str]':
    """Human-readable summary of the differences between the file on disk and the regenerated one."""
    if kind == 'json':
        before, after = leaves(json.loads(old)), leaves(json.loads(new))
        changed = [p for p, value in after.items() if p in before and before[p] != value]
        added = [p for p in after if p not in before]
        removed = [p for p in before if p not in after]
        lines = [f'  JSON: {len(changed)} valores distintos, {len(added)} claves nuevas, {len(removed)} eliminadas']
        lines += [f'    ~ {p}: {before[p]!r} -> {after[p]!r}'[:200] for p in changed[:limit]]
        lines += [f'    + {p}' for p in added[:limit]] + [f'    - {p}' for p in removed[:limit]]
        return lines
    old_lines, new_lines = old.splitlines(), new.splitlines()
    headings, heading = [], '(cabecera)'
    for line in new_lines:
        heading = line if line.startswith('#') else heading
        headings.append(heading)
    sections: 'dict[str, int]' = {}
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            where = headings[min(j1, len(headings) - 1)] if headings else '(vacío)'
            sections[where] = sections.get(where, 0) + max(j2 - j1, 1)
    return [f'  MD: {sum(sections.values())} líneas distintas'] + [f'    {n:4d}  {w}' for w, n in sections.items()]


def write_output(path: str, text: str, kind: str, check: bool) -> bool:
    """Write (or compare) one output file; return ``True`` when the file on disk is up to date."""
    current = None
    if os.path.isfile(path):
        with open(path, 'rb') as handle:
            current = handle.read().decode('utf-8')
    same = current is not None and normalise(current, kind) == normalise(text, kind)
    name = relative(path)
    if check:
        print(f'{name}: {"al día" if same else "DESACTUALIZADO"}')
        if not same and current is not None:
            print('\n'.join(diff_summary(normalise(current, kind), normalise(text, kind), kind)))
        return same
    if same:
        print(f'{name}: sin cambios (sólo fecha/commit); no se reescribe')
        return True
    with open(path, 'wb') as handle:
        handle.write(text.replace('\r\n', '\n').replace('\n', '\r\n').encode('utf-8'))
    print(f'{name}: escrito')
    return True


###############
##  Main
###############
def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0])
    parser.add_argument('--decisions', nargs='+', default=[DEFAULT_DECISIONS],
                        help='JSON de ensemble_decisions.py --output (varios se combinan; gana el último)')
    parser.add_argument('--notes', default=DEFAULT_NOTES, help='archivo de notas editables')
    parser.add_argument('--md', default=DEFAULT_MD, help='salida Markdown')
    parser.add_argument('--json', default=DEFAULT_JSON, help='salida JSON')
    parser.add_argument('--check', action='store_true',
                        help='regenera en memoria, compara con los archivos en disco y sale con 1 si difieren')
    args = parser.parse_args()
    notes = Notes(args.notes)
    data, ctx = build_data(args, notes)
    markdown = Markdown(data, ctx, notes).build()
    payload = json.dumps(data, indent=1, ensure_ascii=False) + '\n'
    results = [write_output(args.md, markdown, 'md', args.check), write_output(args.json, payload, 'json', args.check)]
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
