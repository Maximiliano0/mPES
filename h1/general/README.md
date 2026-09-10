<div align="center">

# 📈 mPES Under Stress Experiments — `general/`

**Cross-model under-stress evaluation under 22 perturbation scenarios.**

[![Models](https://img.shields.io/badge/models-12-blue.svg)](#scope)
[![Scenarios](https://img.shields.io/badge/scenarios-22-blueviolet.svg)](#scenario-catalogue)
[![Cells](https://img.shields.io/badge/cells-264-success.svg)](#scope)
[![Output](https://img.shields.io/badge/figures-PNG-orange.svg)](#heatmaps-publication-quality)

</div>

> **Purpose** — Generalise and benchmark six individual mPES agents and six
> ensemble variants under a 22-scenario matrix of severity / length / joint /
> structural perturbations to expose each model's limitations and identify
> the most robust one within each suite.

The benchmark stores two comparable suites: `individual` contains the six
individual agents and `ensemble` contains the six ensemble variants. Both
suites use the same scenario catalogue and seed. All six ensemble variants —
including `pes_ens` and `pes_ens_consensus_prior` — are part of the active
benchmark; `pes_ens` is the best-performing ensemble in the current results.

---

## Scope

| Aspect | Value |
|---|---|
| Models evaluated | 12: 6 individual + 6 ensemble models |
| Scenarios | 22 (1 baseline + 9 severity + 5 length + 4 joint + 3 structural) |
| Cells | 12 × 22 = **264** |
| `n` per cell | 64 sequences (single seed = 42) |
| Retraining | **None** — pure inference on existing artefacts |

The benchmark **does not modify** any package's source code beyond the
"BENCHMARK OVERRIDE HOOK" block in each
`<group>/<pkg>/__init__.py` that lets the harness redirect
`OUTPUTS_PATH`, `NUM_BLOCKS` and `NUM_SEQUENCES` via env vars.
Input CSVs are swapped in-place under each package's `inputs/`
directory and atomically restored via `try/finally` (back-up
files use the `.bench_stash` extension).

## Workflow

```powershell
# 0. From the repository root: activate the virtual environment, then move
#    into h1/ (it is a plain directory, not a Python package).
win_mpes_env\Scripts\Activate.ps1
cd h1

# 1. (One-time) ensure every benchmarked model has been trained -- the
#    harness reads ``<pkg>/inputs/*.keras`` (or ``q.npy`` for tabular)
#    and the empirical CSV baselines from each package.

# 2. Run the full sweep for individual and ensemble suites
#    (resumable; cells whose JSON exists are skipped).
python -m general.scripts.benchmark run --suite both

# 3. Aggregate each suite into matrices, summary and Markdown report.
python -m general.scripts.analysis

# 4. Render every figure of both suites.
python -m general.scripts.figures

# (anytime) live progress snapshot during a sweep:
python -m general.scripts.benchmark progress --suite individual
python -m general.scripts.benchmark progress --suite individual --watch
```

### Single-cell debug runs

```powershell
python -m general.scripts.benchmark run --pkg pes_dqn --scenario sev_base
python -m general.scripts.benchmark run --pkg pes_dqn --force
```

## Output layout

```
general/
├── README.md                        # this file
├── __init__.py
├── scripts/                         # six harness modules
│   ├── __init__.py
│   ├── scenarios.py                 # perturbation catalogue + CSV synthesis
│   ├── benchmark.py                 # cell execution + sweep + progress
│   ├── analysis.py                  # matrices + statistics + Markdown report
│   ├── plotting.py                  # shared figure primitives + statistics
│   └── figures.py                   # every benchmark figure
├── work/                            # runtime intermediates (per cell)
│   └── <pkg>/
│       ├── scenarios/<sid>/         # synthesised input CSVs
│       └── outputs/<sid>/           # subprocess outputs + log
└── results/
    └── <suite>/                     # individual | ensemble
        ├── cells/<model>__<sid>.json    # one payload per benchmark cell
        ├── matrices/<metric>.csv        # model x scenario matrices
        ├── figures/                     # 01..14 PNG
        │   ├── histogramas/<sid>.*      # per-scenario distributions
        │   └── recompensa/<sid>.*       # cumulative + running-mean reward
        ├── summary.json                 # machine-readable consolidation
        ├── comparison_metrics.json      # pairwise Welch / Cohen / KL
        └── report.md                    # executive summary
```

## Scenario catalogue

| Family | Scenario ID | Description |
|---|---|---|
| baseline | `sev_base` | Empirical training distribution (baseline). |
| severity | `sev_uniform` | Uniform U(0, 9). |
| severity | `sev_gauss_low` | Truncated N(2, 1.5). |
| severity | `sev_gauss_mid` | Truncated N(4.5, 2.0). |
| severity | `sev_gauss_high` | Truncated N(7, 1.5). |
| severity | `sev_weibull` | Weibull(k=1.5), heavy upper tail. |
| severity | `sev_beta_lowskew` | Beta(2, 5)·9 — skewed low. |
| severity | `sev_beta_highskew` | Beta(5, 2)·9 — skewed high. |
| severity | `sev_bimodal` | 0.5 N(2,1) + 0.5 N(7,1). |
| severity | `sev_extrapolate_high` | Under-stress U(10, 12). |
| length | `len_all_short` | Every sequence length 3. |
| length | `len_all_long` | Every sequence length 10. |
| length | `len_geometric` | Geom(p=0.2) clipped [3,10]. |
| length | `len_poisson` | Poisson(λ=5) clipped [3,10]. |
| length | `len_extrapolate_long` | Under-stress U{11..20}. |
| joint | `joint_high_long` | Gauss(7,1.5) × all-long. |
| joint | `joint_low_short` | Gauss(2,1.5) × all-short. |
| joint | `joint_uniform_geom` | Uniform × geometric. |
| joint | `joint_extrap_both` | Under-stress severity × under-stress length. |
| structural | `struct_few_long_blocks` | 4 blocks × 16 sequences. |
| structural | `struct_many_short_blocks` | 16 blocks × 4 sequences. |
| structural | `struct_more_total` | 8 blocks × 16 sequences (n=128). |

## Heatmaps (publication quality)

> **Note**: The heatmaps, matrices, pairwise comparisons, generalisation
> profiles and reports are generated separately for the individual and
> ensemble suites. Re-run the workflow above to regenerate results after any
> configuration change.

All four heatmaps are written as **`.png`** (raster, 300 dpi) for direct
inclusion in papers.
Cells are normalised to fixed colour-scale limits so figures from
different sweeps are directly comparable; clipped values are flagged
in-cell (e.g. `≤-10` in the Welch heatmap).

### Figure conventions

All figures share the publication style defined in `plotting.py`
(`PUB_RC`):

* **Fixed colour per model** — `MODEL_COLOURS` assigns each package one hue
  that is kept across every figure. The best model of each suite
  (`pes_trf`, `pes_ens`) uses the warm accent `#c44e52`; the remaining
  models share a blue–green–amber ramp consistent with the heatmap palettes.
* **Best-model emphasis** — in any figure that overlays several models
  (05, 06, 11, histograms, reward curves) the model with the highest mean
  over the 21 perturbation scenarios is drawn with a thick stroke
  (`BEST_LINEWIDTH = 3.0` vs `BASE_LINEWIDTH = 1.5`) and named in the
  super-title ("trazo grueso = …"). In the ranking (08) the same model is
  outlined in black and its label is bold.
* **Legends** — a single shared legend row below the panels (05, 06 for the
  ensemble suite, 11, reward curves) or below the axis (08, 09). In 06 for
  the individual suite the model pair differs per panel, so each panel keeps
  its own legend in the lower-right corner.
* **Figure 06** — the individual suite contrasts `pes_trf` with one partner
  per panel (`pes_dql`, `pes_a2c`, `pes_dqn`); the ensemble suite draws all
  six variants. Dotted horizontal lines mark each model's mean.

| Figure | Output | Interpretation |
|---|---|---|
| Mean performance | `figures/01_desempeno_por_escenario` | Normalised performance per model and scenario |
| Degradation | `figures/02_degradacion_por_escenario` | `mean(sev_base) - mean(cell)`; positive = loss |
| Welch | `figures/03_welch_logp_por_escenario` | `log10(p)`; lower = stronger evidence |
| Action KL | `figures/04_kl_acciones_por_escenario` | Policy drift vs the reference condition |
| Family curves | `figures/05_curvas_por_familia` | Sorted per-sequence performance by perturbation family; best model in thick stroke |
| Extrapolation curves | `figures/06_curvas_estresores_universales` | Behaviour under extrapolated severity and length; dotted line = mean |
| Effect size | `figures/07_cohen_d_por_escenario` | Standardised change vs the reference condition |
| Ranking | `figures/08_ranking_desempeno` | Reference vs perturbed mean per model; best model outlined |
| Family sensitivity | `figures/09_degradacion_por_familia` | Mean degradation per perturbation family |
| Stability | `figures/10_desempeno_vs_estabilidad` | Mean performance vs dispersion, one colour per model |
| Generalisation | `figures/11_perfiles_generalizacion` | Response profile across each family; best model in thick stroke |
| Pairwise contrasts | `figures/12_pares_welch_logp`, `13_pares_cohen_d`, `14_pares_kl` | Model-versus-model comparison |

## Metrics per cell

For each `(model, scenario)`:

* `per_sequence_perf` — vector of length `n_sequences` parsed from the
  package's `Sequence X: Performance = Y.YYYY` stdout lines, or from the
  `*performances*.npy` artefact written by the ensemble evaluators.
* `global_mean_perf`, `std_perf`, `min_perf`, `max_perf`.
* `action_distribution` — empirical PMF over the 11 allocation actions.
* Matrices in `matrices/` add `stress_degradation`, `welch_p`, `welch_logp`,
  `cohen_d` and `action_kl`, always against the model's own `sev_base`
  reference condition.
* Pairwise `Welch`, `Cohen d` and symmetric `KL` are calculated by
  `figures.py` over scenarios common to all models in a suite; KL uses
  common 20-bin performance histograms in `[0, 1]`.

## Compute notes

* **Local execution** — the sweep runs on Windows CPU in `win_mpes_env`.

## Reproducibility

* Single seed (`42`) for all CSV synthesis ensures all 12 models see the
  exact same severity / length sequences within a scenario.
* Each cell's JSON records the workspace-relative paths to the
  subprocess log, the package's results JSON, and the responses file.
* Empty CSV-swap stash files (`*.bench_stash`) are restored even on
  subprocess failure.
