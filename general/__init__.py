"""mPES OOD benchmark harness.

This package generalises and benchmarks all trained mPES agents (every
package except ``tabular/pes_base``) under a matrix of out-of-distribution
scenarios (severity / length / structural perturbations).

Entry points (all under :mod:`general.scripts`)
-----------------------------------------------
* :mod:`general.scripts.scenarios`   -- scenario taxonomy and CSV synthesisers.
* :mod:`general.scripts.runner`      -- executes ONE (model, scenario) cell.
* :mod:`general.scripts.orchestrate` -- iterates the full Cartesian product.
* :mod:`general.scripts.aggregate`   -- raw JSONs -> matrices + statistics.
* :mod:`general.scripts.plot_matrix` -- heatmaps and per-sequence histograms.
* :mod:`general.scripts.report`      -- writes ``benchmark_report.md``.
* :mod:`general.scripts.progress`    -- live progress bars + ETA.

See ``general/README.md`` for the full workflow.
"""
