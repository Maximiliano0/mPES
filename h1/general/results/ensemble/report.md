# mPES Under Stress Experiments — ensemble

Generated: 2026-09-29T00:18:40.467291+00:00

**Reference condition:** `sev_base`

**Models:** 6 — pes_ens, pes_ens_sprb, pes_ens_accq, pes_ens_consensus, pes_ens_consensus_prior, pes_ens_trf_guard
**Scenarios:** 27

## 0. Baseline definition

The **baseline** is the scenario `sev_base`: the empirical training distribution, i.e. the unperturbed `initial_severity.csv` and `sequence_lengths.csv` of the reference package (`--reference-pkg`, default `pes_dqn`), copied into every package so that all models face the same sequences ("normal" conditions). Every stress scenario is compared against it.

**Mean degradation** is the signed mean drop in normalized performance relative to that baseline, `mean_s(baseline - perf_s)` over the stress scenarios. Positive = loss under stress; negative = the model performs better under stress than at baseline.

The held-out replicas (`heldout_s*`) are fresh draws from the baseline distribution; they are excluded from sections 1-3 and reported in section 4.

## 1. Per-model best / worst

| Model | Reference | Best scenario | Best | Worst scenario | Worst | Mean degradation |
|---|---:|---|---:|---|---:|---:|
| pes_ens | 0.9373 | `sev_extrapolate_high` | 1.0000 | `len_extrapolate_long` | 0.9000 | -0.0021 |
| pes_ens_sprb | 0.9142 | `sev_gauss_high` | 0.9423 | `sev_extrapolate_high` | 0.8604 | +0.0118 |
| pes_ens_accq | 0.9142 | `sev_gauss_high` | 0.9418 | `sev_extrapolate_high` | 0.8591 | +0.0130 |
| pes_ens_consensus | 0.8893 | `joint_low_short` | 0.9877 | `len_extrapolate_long` | 0.8259 | -0.0152 |
| pes_ens_consensus_prior | 0.9177 | `joint_extrap_both` | 0.9684 | `len_extrapolate_long` | 0.8703 | -0.0017 |
| pes_ens_trf_guard | 0.9279 | `joint_extrap_both` | 0.9969 | `len_extrapolate_long` | 0.8613 | -0.0026 |

## 2. Mean degradation by scenario family

| Family | # scenarios | Mean degradation |
|---|---:|---:|
| Severidad | 9 | -0.0012 |
| Longitud | 5 | +0.0152 |
| Conjunta | 4 | -0.0136 |
| Estructura | 3 | +0.0000 |

## 3. Most degraded cells

| Rank | Model | Scenario | Reference | Cell | Degradation |
|---:|---|---|---:|---:|---:|
| 1 | pes_ens_trf_guard | `len_extrapolate_long` | 0.9279 | 0.8613 | +0.0666 |
| 2 | pes_ens_consensus | `len_extrapolate_long` | 0.8893 | 0.8259 | +0.0634 |
| 3 | pes_ens_accq | `sev_extrapolate_high` | 0.9142 | 0.8591 | +0.0552 |
| 4 | pes_ens_sprb | `sev_extrapolate_high` | 0.9142 | 0.8604 | +0.0538 |
| 5 | pes_ens_accq | `joint_uniform_geom` | 0.9142 | 0.8622 | +0.0521 |

## 4. Held-out replicas of the reference distribution

Replicas: `heldout_s1`, `heldout_s2`, `heldout_s3`, `heldout_s4`, `heldout_s5`. Gap = reference - pooled held-out mean (positive = the reference score is optimistic); d and Welch compare the pooled held-out sequences with the reference sequences.

| Model | Reference | Held-out mean | SD between replicas | Gap | d | log10 p |
|---|---:|---:|---:|---:|---:|---:|
| pes_ens | 0.9373 | 0.9388 | 0.0028 | -0.0015 | +0.04 | -0.12 |
| pes_ens_sprb | 0.9142 | 0.9170 | 0.0077 | -0.0028 | +0.06 | -0.18 |
| pes_ens_accq | 0.9142 | 0.9162 | 0.0094 | -0.0020 | +0.04 | -0.12 |
| pes_ens_consensus | 0.8893 | 0.8931 | 0.0069 | -0.0037 | +0.06 | -0.18 |
| pes_ens_consensus_prior | 0.9177 | 0.9148 | 0.0071 | +0.0029 | -0.07 | -0.21 |
| pes_ens_trf_guard | 0.9279 | 0.9276 | 0.0043 | +0.0003 | -0.01 | -0.02 |

## 5. Artefacts

* Matrices: `matrices/*.csv`
* Figures: `figures/*.png`
* Cells: `cells/<model>__<scenario>.json`
