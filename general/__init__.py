"""mPES OOD benchmark harness.

This package generalises and benchmarks all trained mPES agents (every
package except ``tabular/pes_base``) under a matrix of out-of-distribution
scenarios (severity / length / structural perturbations).

Entry points
------------
* :mod:`general.scenarios` -- scenario taxonomy and CSV synthesisers.
* :mod:`general.runner`    -- executes ONE (model, scenario) cell.
* :mod:`general.orchestrate` -- iterates the full Cartesian product.
* :mod:`general.aggregate` -- raw JSONs -> matrices + statistics.
* :mod:`general.plot_matrix` -- heatmaps and per-sequence histograms.
* :mod:`general.report`    -- writes ``benchmark_report.md``.

See ``general/README.md`` for the full workflow.
"""
