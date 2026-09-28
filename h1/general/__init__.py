"""mPES Under Stress Experiments harness.

This package generalises and benchmarks every trained mPES agent -- the
seven individual packages (including ``tabular/pes_base``, the
``REFERENCE_MODEL``) and the six ensemble variants -- under a matrix of
stress-test scenarios (severity / length / joint / structural
perturbations).

Entry points (all under :mod:`general.scripts`)
-----------------------------------------------
* :mod:`general.scripts.scenarios`       -- scenario taxonomy, perturbation catalogue and CSV synthesis.
* :mod:`general.scripts.benchmark`       -- cell execution, sweep driver and progress.
* :mod:`general.scripts.analysis`        -- matrices, statistics and Markdown report.
* :mod:`general.scripts.plotting`        -- shared figure primitives and statistics.
* :mod:`general.scripts.figures`         -- every benchmark figure.
* :mod:`general.scripts.random_baseline` -- random-player baseline figures.
* :mod:`general.scripts.agent_internals` -- agent-internals figures of ``pes_trf``.

See ``general/README.md`` for the full workflow.
"""
