"""mPES Under Stress Experiments harness.

This package generalises and benchmarks all trained mPES agents (every
package except ``tabular/pes_base``) under a matrix of stress-test
scenarios (severity / length / structural perturbations).

Entry points (all under :mod:`general.scripts`)
-----------------------------------------------
* :mod:`general.scripts.scenarios`   -- scenario taxonomy and CSV synthesisers.
* :mod:`general.scripts.scenarios` -- perturbation catalogue and CSV synthesis.
* :mod:`general.scripts.benchmark` -- cell execution, sweep driver and progress.
* :mod:`general.scripts.analysis`  -- matrices, statistics and Markdown report.
* :mod:`general.scripts.plotting`  -- shared figure primitives and statistics.
* :mod:`general.scripts.figures`   -- every benchmark figure.

See ``general/README.md`` for the full workflow.
"""
