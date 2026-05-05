# mPES OOD Benchmark Harness (`general/`)

> **Purpose** — Generalise and benchmark all trained mPES agents
> (every package except `tabular/pes_base`) under a 22-scenario matrix
> of severity / length / structural perturbations to expose each
> model's limitations and identify the most robust one.

## Scope

| Aspect | Value |
|---|---|
| Models evaluated | 7 (`pes_ql`, `pes_dql`, `pes_dqn`, `pes_rdqn`, `pes_a2c`, `pes_trf`, `pes_ens`) |
| Scenarios | 22 (1 baseline + 9 severity + 5 length + 4 joint + 3 structural) |
| Cells | 7 × 22 = **154** |
| `n` per cell | 64 sequences (single seed = 42) |
| Retraining | **None** — pure inference on existing artefacts |

The benchmark **does not modify** any package's source code beyond a
small "BENCHMARK OVERRIDE HOOK" block in each
`<group>/<pkg>/__init__.py` that lets the harness redirect
`OUTPUTS_PATH`, `NUM_BLOCKS` and `NUM_SEQUENCES` via env vars.
Input CSVs are swapped in-place under each package's `inputs/`
directory and atomically restored via `try/finally` (back-up
files use the `.bench_stash` extension).

## Workflow

```powershell
# 0. Activate venv (Windows / Linux):
win_mpes_env\Scripts\Activate.ps1
# source linux_mpes_env/bin/activate

# 1. (One-time) ensure every benchmarked model has been trained -- the
#    harness reads ``<pkg>/inputs/*.keras`` (or ``q.npy`` for tabular)
#    and the empirical CSV baselines from each package.

# 2. Run the full sweep (resumable; cells whose JSON exists are skipped).
python -m general.scripts.orchestrate

# 3. Aggregate raw cells into matrices + statistics.
python -m general.scripts.aggregate

# 4. Produce heatmaps + per-scenario histograms.
python -m general.scripts.plot_matrix

# 5. Compose the executive Markdown report.
python -m general.scripts.report

# (anytime) live progress snapshot during a sweep:
python -m general.scripts.progress             # one-shot
python -m general.scripts.progress --watch     # refresh every 30 s
```

### Single-cell debug runs

```powershell
python -m general.scripts.runner --pkg pes_dqn --scenario sev_empirical
python -m general.scripts.runner --pkg pes_dqn --scenario all --force
```

## Output layout

```
general/
├── README.md                        # this file
├── __init__.py
├── scripts/                         # all executable harness modules
│   ├── __init__.py
│   ├── scenarios.py                 # taxonomy + CSV synthesisers
│   ├── runner.py                    # one (model, scenario) cell
│   ├── orchestrate.py               # full Cartesian product
│   ├── progress.py                  # live progress bars + ETA
│   ├── aggregate.py                 # raw -> matrices + Welch / Cohen / KL
│   ├── plot_matrix.py               # heatmaps + per-scenario histograms
│   └── report.py                    # benchmark_report.md
├── colab/                           # Colab Pro+ packaging
│   ├── colab_bench.ipynb
│   └── run_colab_bench.sh
├── work/                            # runtime intermediates (per cell)
│   └── <pkg>/
│       ├── scenarios/<sid>/         # synthesised input CSVs
│       └── outputs/<sid>/           # subprocess outputs + log
└── results/
    ├── raw/<pkg>__<sid>.json        # one cell payload
    ├── matrix_global_mean.csv
    ├── matrix_std.csv
    ├── matrix_min.csv
    ├── matrix_max.csv
    ├── matrix_ood_degradation.csv   # baseline_mean - cell_mean
    ├── matrix_welch_p.csv           # Welch t two-sided p (raw)
    ├── matrix_welch_logp.csv        # log10(p) via t.logsf (no underflow)
    ├── matrix_cohen_d.csv           # Cohen's d effect size
    ├── matrix_action_kl.csv         # KL(action_dist || in-dist policy)
    ├── matrix_summary.json          # machine-readable consolidation
    ├── heatmap_global_mean.{png,pdf}
    ├── heatmap_ood_degradation.{png,pdf}
    ├── heatmap_welch_logp.{png,pdf}     # log10(p) clipped to [-10, 0]
    ├── heatmap_action_kl.{png,pdf}      # log-scale KL
    ├── per_sequence_histograms/<sid>.{png,pdf}
    └── benchmark_report.md
```

## Scenario catalogue

| Family | Scenario ID | Description |
|---|---|---|
| baseline | `sev_empirical` | Empirical training distribution (in-distribution). |
| severity | `sev_uniform` | Uniform U(0, 9). |
| severity | `sev_gauss_low` | Truncated N(2, 1.5). |
| severity | `sev_gauss_mid` | Truncated N(4.5, 2.0). |
| severity | `sev_gauss_high` | Truncated N(7, 1.5). |
| severity | `sev_weibull` | Weibull(k=1.5), heavy upper tail. |
| severity | `sev_beta_lowskew` | Beta(2, 5)·9 — skewed low. |
| severity | `sev_beta_highskew` | Beta(5, 2)·9 — skewed high. |
| severity | `sev_bimodal` | 0.5 N(2,1) + 0.5 N(7,1). |
| severity | `sev_extrapolate_high` | OOD U(10, 12). |
| length | `len_all_short` | Every sequence length 3. |
| length | `len_all_long` | Every sequence length 10. |
| length | `len_geometric` | Geom(p=0.2) clipped [3,10]. |
| length | `len_poisson` | Poisson(λ=5) clipped [3,10]. |
| length | `len_extrapolate_long` | OOD U{11..20}. |
| joint | `joint_high_long` | Gauss(7,1.5) × all-long. |
| joint | `joint_low_short` | Gauss(2,1.5) × all-short. |
| joint | `joint_uniform_geom` | Uniform × geometric. |
| joint | `joint_extrap_both` | OOD severity × OOD length. |
| structural | `struct_few_long_blocks` | 4 blocks × 16 sequences. |
| structural | `struct_many_short_blocks` | 16 blocks × 4 sequences. |
| structural | `struct_more_total` | 8 blocks × 16 sequences (n=128). |

## Heatmaps (publication quality)

All four heatmaps are written as both **`.png`** (raster, 300 dpi) and
**`.pdf`** (vector, TrueType-embedded) for direct inclusion in papers.
Cells are normalised to fixed colour-scale limits so figures from
different sweeps are directly comparable; clipped values are flagged
in-cell (e.g. `≤-10` in the Welch heatmap).

| Metric | Heatmap | Colour map | Scale |
|---|---|---|---|
| Global mean performance | [heatmap_global_mean.png](results/heatmap_global_mean.png) | `viridis` | auto-bounded |
| OOD degradation (Δ vs baseline) | [heatmap_ood_degradation.png](results/heatmap_ood_degradation.png) | `RdBu_r` (diverging) | symmetric around 0 |
| Welch t-test, log₁₀(p) | [heatmap_welch_logp.png](results/heatmap_welch_logp.png) | `magma_r` | `[-10, 0]`, α-tick marks |
| Action-distribution KL | [heatmap_action_kl.png](results/heatmap_action_kl.png) | `cividis` | `LogNorm` |

![global mean](results/heatmap_global_mean.png)
![ood degradation](results/heatmap_ood_degradation.png)
![welch log10 p](results/heatmap_welch_logp.png)
![action KL](results/heatmap_action_kl.png)

## Metrics per cell

For each `(model, scenario)`:

* `per_sequence_perf` — vector of length `n_sequences` parsed from the
  package's `Sequence X: Performance = Y.YYYY` stdout lines.
* `global_mean_perf`, `std_perf`, `min_perf`, `max_perf`.
* `ood_degradation` = `baseline_mean - cell_mean` (per model).
* `welch_t`, `welch_p` — Welch two-sample t vs the model's own `sev_empirical` baseline.
* `cohen_d` — pooled-SD effect size vs baseline.
* `action_distribution` — empirical pmf over actions {0, …, 10} from the package's `responses_*.txt`.
* `kl_action_drift` — `KL(cell_action_dist || baseline_action_dist)`.

## Compute notes

* **Local first** — full sweep on Windows CPU (`win_mpes_env`). pes_ens
  is the slowest member.
* **Colab Pro+** — open `general/colab/colab_bench.ipynb` in Colab, run
  all cells, then **Runtime → Manage sessions → Background execution**.
  The launcher script `general/colab/run_colab_bench.sh` mirrors trained
  artefacts from Drive, runs `orchestrate → aggregate → plot → report`
  and syncs `general/results/` back to
  `/content/drive/MyDrive/mPES/_benchmark/results/`. Resumable: cells
  whose JSON already exists in Drive are skipped on restart.

## Reproducibility

* Single seed (`42`) for all CSV synthesis ensures all 7 models see the
  exact same severity / length sequences within a scenario.
* Each cell's JSON records the workspace-relative paths to the
  subprocess log, the package's results JSON, and the responses file.
* Empty CSV-swap stash files (`*.bench_stash`) are restored even on
  subprocess failure.
