"""Result formatter, mirrored from ``ml/pes_dqn/src/result_formatter.py``.

Unlike the ``ml/`` original, ``file_prefix`` is a parameter instead of a
module-level import from ``ml.pes_dqn``, so each ensemble can label its own
output files (``PES_ENS_SPRB_``, ``PES_ENS_ACCQ_``, ...).
"""

import json
import os
from datetime import datetime

import numpy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def generate_results_report(subject_id, outputs_path, performances, all_performances,
                            resources_data=None, file_prefix='PES_ENS_'):
    """
    Generate JSON and PNG report files for experiment results.

    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., ``"001_TEST"``).
    outputs_path : str
        Path to the outputs directory.
    performances : list
        List of performance metrics per sequence.
    all_performances : list
        2-D array of performances per block/session.
    resources_data : dict, optional
        Additional resource allocation data.
    file_prefix : str, optional
        Prefix used for the generated JSON/PNG filenames.

    Returns
    -------
    tuple
        ``(json_filepath, png_filepath)`` paths of the generated reports.
    """
    stats = _calculate_statistics(performances, all_performances)
    json_filepath = _save_json_report(subject_id, outputs_path, stats, resources_data, file_prefix)
    png_filepath = _save_png_plots(subject_id, outputs_path, performances, all_performances, stats, file_prefix)
    return json_filepath, png_filepath


def _calculate_statistics(performances, all_performances):
    """
    Calculate comprehensive statistical summaries from performance data.

    Parameters
    ----------
    performances : array-like
        List of performance scores for each sequence (individual values)
    all_performances : array-like
        2D array where each row contains performance scores for sequences in a block

    Returns
    -------
    dict
        Overall and per-block statistics (mean, median, std, quartiles, improvement).
    """
    performances = numpy.array(performances)
    all_performances = numpy.array(all_performances)

    stats = {
        'overall_mean': float(numpy.mean(performances)),
        'overall_median': float(numpy.median(performances)),
        'overall_std': float(numpy.std(performances)),
        'overall_min': float(numpy.min(performances)),
        'overall_max': float(numpy.max(performances)),
        'total_sequences': len(performances),
        'first_block_mean': float(numpy.mean(performances[:8])) if len(performances) >= 8 else None,
        'last_block_mean': float(numpy.mean(performances[-8:])) if len(performances) >= 8 else None,
        'improvement': float(numpy.mean(performances[-8:]) - numpy.mean(performances[:8])) if len(performances) >= 16 else None,
        'percentile_25': float(numpy.percentile(performances, 25)),
        'percentile_75': float(numpy.percentile(performances, 75)),
    }

    block_stats = []
    for block_idx, block_perf in enumerate(all_performances):
        block_perf = numpy.array(block_perf)
        block_stats.append({
            'block_number': block_idx + 1,
            'mean': float(numpy.mean(block_perf)),
            'median': float(numpy.median(block_perf)),
            'std': float(numpy.std(block_perf)),
            'min': float(numpy.min(block_perf)),
            'max': float(numpy.max(block_perf)),
            'num_sequences': len(block_perf),
        })
    stats['per_block_statistics'] = block_stats

    return stats


def _save_json_report(subject_id, outputs_path, stats, resources_data, file_prefix):
    """
    Export performance statistics to a structured JSON report file.

    Parameters
    ----------
    subject_id : str
        Unique identifier for the subject/session.
    outputs_path : str
        Directory path where the JSON file will be saved.
    stats : dict
        Statistics dictionary from :func:`_calculate_statistics`.
    resources_data : dict or None
        Configuration and resource information.
    file_prefix : str
        Prefix used for the generated filename.

    Returns
    -------
    str
        Absolute path to the generated JSON file.
    """
    report = {
        'metadata': {
            'subject_id': subject_id,
            'timestamp': datetime.now().isoformat(),
            'report_type': 'PES_Experiment_Results_v2',
            'model_type': resources_data.get('agent_type', 'Unknown') if resources_data else 'Unknown'
        },
        'configuration': {
            'total_resources_per_sequence': resources_data.get('total_resources_per_sequence') if resources_data else None,
            'num_blocks': resources_data.get('num_blocks') if resources_data else None,
            'num_sequences': resources_data.get('num_sequences') if resources_data else None,
            'total_sessions': resources_data.get('total_trials') if resources_data else None,
        },
        'performance_statistics': stats,
        'resources_allocation': resources_data if resources_data else {},
        'file_references': {
            'results_file': f'{file_prefix}results_{subject_id}.png',
            'this_file': f'{file_prefix}results_{subject_id}.json'
        }
    }

    json_filename = f'{file_prefix}results_{subject_id}.json'
    json_filepath = os.path.join(outputs_path, json_filename)

    with open(json_filepath, 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)

    return json_filepath


def _save_png_plots(subject_id, outputs_path, performances, all_performances, stats, file_prefix):
    """
    Generate and save multi-panel performance visualization to PNG file.

    Parameters
    ----------
    subject_id : str
        Subject/session identifier used in filename.
    outputs_path : str
        Directory path where the PNG file will be saved.
    performances : array-like
        Individual performance scores for each sequence.
    all_performances : array-like
        2D array of block performances.
    stats : dict
        Statistics dictionary from :func:`_calculate_statistics`.
    file_prefix : str
        Prefix used for the generated filename and used in the plot title.

    Returns
    -------
    str
        Absolute path to the generated PNG file.
    """
    try:
        performances = numpy.array(performances, dtype=float)

        normalized_all_perf = []
        if isinstance(all_performances, (list, numpy.ndarray)):
            for item in all_performances:
                if isinstance(item, (list, numpy.ndarray)):
                    item_list = [float(x) for x in item]
                    if item_list:
                        normalized_all_perf.append(item_list)

        if not normalized_all_perf:
            normalized_all_perf = [[float(x) for x in performances]]

        fig = plt.figure(figsize=(16, 10))
        grid = gridspec.GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.35,
                                 height_ratios=[1.2, 1.2, 0.6])

        ax1 = fig.add_subplot(grid[0, :2])
        ax1.plot(range(1, len(performances) + 1), performances, 'b-o', linewidth=2, markersize=4, alpha=0.7)
        ax1.axhline(
            y=stats['overall_mean'],
            color='r',
            linestyle='--',
            linewidth=2,
            label=f"Mean: {stats['overall_mean']:.3f}")
        ax1.fill_between(range(1, len(performances) + 1),
                         stats['overall_mean'] - stats['overall_std'],
                         stats['overall_mean'] + stats['overall_std'],
                         alpha=0.2, color='red')
        ax1.set_xlabel('Sequence Number', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Performance', fontsize=11, fontweight='bold')
        ax1.set_title('Performance Over All Sequences', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_ylim((0, 1.05))

        ax2 = fig.add_subplot(grid[0, 2])
        ax2.hist(performances, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.axvline(x=stats['overall_mean'], color='red', linestyle='--', linewidth=2, label='Mean')
        ax2.axvline(x=stats['overall_median'], color='green', linestyle='--', linewidth=2, label='Median')
        ax2.set_xlabel('Performance', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax2.set_title('Distribution', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)

        ax3 = fig.add_subplot(grid[1, 0])
        if len(normalized_all_perf) > 0:
            box = ax3.boxplot(normalized_all_perf, tick_labels=[f'B{i + 1}' for i in range(len(normalized_all_perf))],
                              patch_artist=True)
            for patch in box['boxes']:
                patch.set_facecolor('lightblue')
        ax3.set_ylabel('Performance', fontsize=10, fontweight='bold')
        ax3.set_title('Performance by Block', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim((0, 1.05))

        ax4 = fig.add_subplot(grid[1, 1])
        cumulative_mean = numpy.cumsum(performances) / numpy.arange(1, len(performances) + 1)
        ax4.plot(range(1, len(cumulative_mean) + 1), cumulative_mean, 'g-o', linewidth=2, markersize=4)
        ax4.set_xlabel('Sequence Number', fontsize=10, fontweight='bold')
        ax4.set_ylabel('Cumulative Mean', fontsize=10, fontweight='bold')
        ax4.set_title('Cumulative Mean Performance', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim((0, 1.05))

        ax5 = fig.add_subplot(grid[1, 2])
        block_means = [numpy.mean(block) for block in normalized_all_perf]
        if len(block_means) > 0:
            ax5.bar(range(1, len(block_means) + 1), block_means, color='steelblue', alpha=0.7)
        ax5.set_xlabel('Block Number', fontsize=10, fontweight='bold')
        ax5.set_ylabel('Mean Performance', fontsize=10, fontweight='bold')
        ax5.set_title('Block-wise Mean Performance', fontsize=11, fontweight='bold')
        ax5.set_ylim((0, 1.05))
        ax5.grid(True, alpha=0.3, axis='y')

        ax6 = fig.add_subplot(grid[2, :])
        ax6.axis('tight')
        ax6.axis('off')

        summary_data = [
            ['Metric', 'Value', 'Metric', 'Value'],
            ['Mean', f"{stats['overall_mean']:.4f}", 'Median', f"{stats['overall_median']:.4f}"],
            ['Std Dev', f"{stats['overall_std']:.4f}", 'Min', f"{stats['overall_min']:.4f}"],
            ['Max', f"{stats['overall_max']:.4f}", 'Q1', f"{stats['percentile_25']:.4f}"],
            ['Q3', f"{stats['percentile_75']:.4f}", 'N Seq', f"{stats['total_sequences']}"],
        ]

        if stats['first_block_mean'] is not None:
            summary_data.append(['First', f"{stats['first_block_mean']:.4f}",
                                'Last', f"{stats['last_block_mean']:.4f}"])
            summary_data.append(['Improvement', f"{stats['improvement']:.4f}", '', ''])

        table = ax6.table(cellText=summary_data, cellLoc='center', loc='center',
                          colWidths=[0.2, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.3)

        for i in range(len(summary_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white', fontsize=8)

        for i in range(1, len(summary_data)):
            for j in range(len(summary_data[0])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        fig.suptitle(f'PES Experiment Results - Subject {subject_id}\nEnsemble Performance Analysis',
                     fontsize=16, fontweight='bold', y=0.995)

        png_filename = f'{file_prefix}results_{subject_id}.png'
        png_filepath = os.path.join(outputs_path, png_filename)

        plt.savefig(png_filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return png_filepath

    except Exception:
        plt.close('all')
        raise
