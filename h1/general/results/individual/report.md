# mPES Under Stress Experiments — individual

Generated: 2026-09-29T00:18:40.180265+00:00

**Reference condition:** `sev_base`

**Models:** 7 — pes_base, pes_ql, pes_dql, pes_dqn, pes_rdqn, pes_a2c, pes_trf
**Scenarios:** 27

## 0. Baseline definition

The **baseline** is the scenario `sev_base`: the empirical training distribution, i.e. the unperturbed `initial_severity.csv` and `sequence_lengths.csv` of the reference package (`--reference-pkg`, default `pes_dqn`), copied into every package so that all models face the same sequences ("normal" conditions). Every stress scenario is compared against it.

**Mean degradation** is the signed mean drop in normalized performance relative to that baseline, `mean_s(baseline - perf_s)` over the stress scenarios. Positive = loss under stress; negative = the model performs better under stress than at baseline.

The held-out replicas (`heldout_s*`) are fresh draws from the baseline distribution; they are excluded from sections 1-3 and reported in section 4.

## 1. Per-model best / worst

| Model | Reference | Best scenario | Best | Worst scenario | Worst | Mean degradation |
|---|---:|---|---:|---|---:|---:|
| pes_base | 0.8706 | `joint_low_short` | 0.9674 | `sev_extrapolate_high` | 0.6269 | +0.0200 |
| pes_ql | 0.8866 | `len_all_short` | 0.9426 | `sev_extrapolate_high` | 0.7623 | +0.0152 |
| pes_dql | 0.8963 | `sev_beta_highskew` | 0.9238 | `sev_extrapolate_high` | 0.7854 | +0.0191 |
| pes_dqn | 0.8937 | `joint_low_short` | 0.9877 | `len_extrapolate_long` | 0.8412 | -0.0050 |
| pes_rdqn | 0.8987 | `sev_gauss_high` | 0.9319 | `sev_extrapolate_high` | 0.8311 | +0.0099 |
| pes_a2c | 0.8872 | `joint_low_short` | 0.9877 | `len_extrapolate_long` | 0.8257 | -0.0090 |
| pes_trf | 0.9272 | `joint_extrap_both` | 0.9969 | `len_extrapolate_long` | 0.8597 | -0.0025 |

## 2. Mean degradation by scenario family

| Family | # scenarios | Mean degradation |
|---|---:|---:|
| Severidad | 9 | +0.0119 |
| Longitud | 5 | +0.0095 |
| Conjunta | 4 | -0.0028 |
| Estructura | 3 | +0.0000 |

## 3. Most degraded cells

| Rank | Model | Scenario | Reference | Cell | Degradation |
|---:|---|---|---:|---:|---:|
| 1 | pes_base | `sev_extrapolate_high` | 0.8706 | 0.6269 | +0.2437 |
| 2 | pes_base | `joint_extrap_both` | 0.8706 | 0.7050 | +0.1656 |
| 3 | pes_ql | `sev_extrapolate_high` | 0.8866 | 0.7623 | +0.1244 |
| 4 | pes_dql | `sev_extrapolate_high` | 0.8963 | 0.7854 | +0.1110 |
| 5 | pes_dql | `joint_uniform_geom` | 0.8963 | 0.8163 | +0.0800 |

## 4. Held-out replicas of the reference distribution

Replicas: `heldout_s1`, `heldout_s2`, `heldout_s3`, `heldout_s4`, `heldout_s5`. Gap = reference - pooled held-out mean (positive = the reference score is optimistic); d and Welch compare the pooled held-out sequences with the reference sequences.

| Model | Reference | Held-out mean | SD between replicas | Gap | d | log10 p |
|---|---:|---:|---:|---:|---:|---:|
| pes_base | 0.8706 | 0.8598 | 0.0096 | +0.0108 | -0.15 | -0.54 |
| pes_ql | 0.8866 | 0.8813 | 0.0079 | +0.0054 | -0.08 | -0.28 |
| pes_dql | 0.8963 | 0.9002 | 0.0055 | -0.0038 | +0.07 | -0.25 |
| pes_dqn | 0.8937 | 0.8915 | 0.0062 | +0.0022 | -0.04 | -0.11 |
| pes_rdqn | 0.8987 | 0.9029 | 0.0075 | -0.0042 | +0.08 | -0.27 |
| pes_a2c | 0.8872 | 0.8869 | 0.0060 | +0.0004 | -0.01 | -0.02 |
| pes_trf | 0.9272 | 0.9286 | 0.0034 | -0.0015 | +0.03 | -0.09 |

## 5. Artefacts

* Matrices: `matrices/*.csv`
* Figures: `figures/*.png`
* Cells: `cells/<model>__<scenario>.json`
