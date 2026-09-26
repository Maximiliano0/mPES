"""Build compact benchmark evidence files for use in an LLM context window.

The compact JSON keeps every per-sequence performance observation, but removes
per-trial actions and repeated execution metadata. A short Markdown companion
contains the descriptive results most useful when discussing the manuscript.

Run from the repository root::

    python utils/scripts/compact_results_context.py
"""

##########################
##  Imports externos    ##
##########################
import argparse
import glob
import json
import os


##########################
##  Configuración       ##
##########################
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_ROOT = os.path.join(ROOT, 'h1', 'general', 'results')
DEFAULT_JSON = os.path.join(RESULTS_ROOT, 'llm_context.json')
DEFAULT_MARKDOWN = os.path.join(RESULTS_ROOT, 'llm_context.md')
SUITES = ('individual', 'ensemble')
CELL_SCHEMA = ('mean', 'std', 'min', 'max', 'per_sequence_perf')


def _load_json(path: str) -> dict:
    """Load one UTF-8 JSON object.

    Parameters
    ----------
    path : str
        JSON file path.

    Returns
    -------
    dict
        Parsed JSON object.
    """
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def _relative(path: str) -> str:
    """Return a repository-relative path with forward slashes.

    Parameters
    ----------
    path : str
        Absolute path below the repository root.

    Returns
    -------
    str
        Portable repository-relative path.
    """
    return os.path.relpath(path, ROOT).replace(os.sep, '/')


def _compact_suite(suite: str) -> tuple[dict, list[str], list[str]]:
    """Compact one benchmark suite without discarding sequence performance.

    Parameters
    ----------
    suite : str
        Benchmark suite identifier.

    Returns
    -------
    tuple
        Compact suite object, ordered scenario families, and source paths.
    """
    suite_root = os.path.join(RESULTS_ROOT, suite)
    summary_path = os.path.join(suite_root, 'summary.json')
    metrics_path = os.path.join(suite_root, 'comparison_metrics.json')
    summary = _load_json(summary_path)
    metrics = _load_json(metrics_path)
    scenarios = summary['scenarios']
    families = [summary['cells'][summary['models'][0]][scenario]['family'] for scenario in scenarios]
    cells = {}
    for model in summary['models']:
        cells[model] = []
        for scenario in scenarios:
            cell = summary['cells'][model][scenario]
            observations = cell['per_sequence_perf']
            if len(observations) != cell['n_sequences']:
                raise ValueError(f'{suite}/{model}/{scenario}: inconsistent sequence count')
            cells[model].append([
                cell['global_mean_perf'],
                cell['std_perf'],
                cell['min_perf'],
                cell['max_perf'],
                observations,
            ])
    compact = {
        'reference_scenario': summary['reference_scenario'],
        'models': summary['models'],
        'aggregates': metrics['models'],
        'cells': cells,
    }
    return compact, families, [_relative(summary_path), _relative(metrics_path)]


def _canonical_best_params() -> tuple[dict, list[str]]:
    """Load canonical package-level ``best_params.json`` files.

    Returns
    -------
    tuple
        Mapping by package name and repository-relative source paths.
    """
    pattern = os.path.join(ROOT, 'h1', '*', '*', 'inputs', 'best_params.json')
    params = {}
    sources = []
    for path in sorted(glob.glob(pattern)):
        package = os.path.basename(os.path.dirname(os.path.dirname(path)))
        params[package] = _load_json(path)
        sources.append(_relative(path))
    return params, sources


def build_context() -> dict:
    """Build the compact evidence object.

    Returns
    -------
    dict
        Compact benchmark evidence and canonical tuned parameters.
    """
    suites = {}
    sources = []
    scenarios = None
    scenario_families = None
    for suite in SUITES:
        compact, families, suite_sources = _compact_suite(suite)
        summary_path = os.path.join(RESULTS_ROOT, suite, 'summary.json')
        suite_scenarios = _load_json(summary_path)['scenarios']
        if scenarios is None:
            scenarios = suite_scenarios
            scenario_families = families
        elif suite_scenarios != scenarios or families != scenario_families:
            raise ValueError(f'{suite}: scenario definitions differ between suites')
        suites[suite] = compact
        sources.extend(suite_sources)

    best_params, param_sources = _canonical_best_params()
    sources.extend(param_sources)
    return {
        'schema_version': 1,
        'purpose': 'Compact, lossless-per-sequence evidence for manuscript discussion with an LLM.',
        'omitted': ['per_trial_actions', 'action_distribution', 'execution metadata', 'file paths'],
        'cell_schema': list(CELL_SCHEMA),
        'scenarios': scenarios,
        'scenario_families': scenario_families,
        'suites': suites,
        'best_params': best_params,
        'sources': sources,
    }


def _format_number(value: float) -> str:
    """Format a descriptive metric for the Markdown summary.

    Parameters
    ----------
    value : float
        Metric value.

    Returns
    -------
    str
        Value rounded to four decimal places.
    """
    return f'{value:.4f}'


def build_markdown(context: dict, json_path: str) -> str:
    """Build a concise human-readable companion to the compact JSON.

    Parameters
    ----------
    context : dict
        Object returned by :func:`build_context`.
    json_path : str
        Output path of the compact JSON.

    Returns
    -------
    str
        Markdown summary suitable for pasting into an LLM session.
    """
    lines = [
        '# Evidencia compacta de resultados mPES',
        '',
        'Fuente detallada: `' + _relative(json_path) + '`. El JSON conserva todos los rendimientos por escenario y modelo,',
        'pero omite acciones por ensayo y metadatos operativos. No contiene una ablación sin ponderación por incertidumbre.',
        '',
        '| Grupo | Modelo | Referencia | Generalización | Peor escenario | Degradación máxima |',
        '|---|---|---:|---:|---|---:|',
    ]
    for suite in SUITES:
        for model in context['suites'][suite]['models']:
            metrics = context['suites'][suite]['aggregates'][model]
            lines.append(
                f"| {suite} | {model} | {_format_number(metrics['reference_mean'])} | "
                f"{_format_number(metrics['stress_mean'])} | {metrics['worst_scenario']} | "
                f"{_format_number(metrics['worst_degradation'])} |"
            )
    lines.extend([
        '',
        '## Lectura del esquema JSON',
        '',
        '- `scenarios` y `scenario_families` comparten índice.',
        '- Cada lista de `cells[modelo]` comparte índice con `scenarios`.',
        '- Cada celda sigue `cell_schema`: media, desviación, mínimo, máximo y rendimientos por secuencia.',
        '- `aggregates` conserva las métricas descriptivas por modelo; `best_params` reúne los archivos canónicos disponibles.',
        '- La comparación estadística directa entre grupos debe calcularse desde `per_sequence_perf`; no se incluye como resultado preexistente.',
        '',
    ])
    return '\n'.join(lines)


def main() -> None:
    """Write compact JSON and Markdown evidence files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--json-output', default=DEFAULT_JSON, help='compact JSON output path')
    parser.add_argument('--markdown-output', default=DEFAULT_MARKDOWN, help='Markdown summary output path')
    args = parser.parse_args()

    context = build_context()
    with open(args.json_output, 'w', encoding='utf-8') as handle:
        json.dump(context, handle, ensure_ascii=True, separators=(',', ':'), allow_nan=False)
    with open(args.markdown_output, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write(build_markdown(context, args.json_output))
    print(f'JSON: {_relative(args.json_output)} ({os.path.getsize(args.json_output):,} bytes)')
    print(f'Markdown: {_relative(args.markdown_output)} ({os.path.getsize(args.markdown_output):,} bytes)')


if __name__ == '__main__':
    main()