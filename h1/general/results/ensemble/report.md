# mPES Under Stress Experiments — ensemble

_Generated: 2026-09-07T18:36:17.455854+00:00_

**Reference condition:** `sev_empirical`

**Models:** 6 — pes_ens, pes_ens_sprb, pes_ens_accq, pes_ens_consensus, pes_ens_consensus_prior, pes_ens_trf_guard
**Scenarios:** 22

## 1. Per-model best / worst

| Model | Reference | Best scenario | Best | Worst scenario | Worst | Mean degradation |
|---|---:|---|---:|---|---:|---:|
| pes_ens | 0.9373 | `sev_extrapolate_high` | 1.0000 | `len_extrapolate_long` | 0.9000 | -0.0021 |
| pes_ens_sprb | 0.9142 | `sev_gauss_high` | 0.9423 | `sev_extrapolate_high` | 0.8604 | +0.0118 |
| pes_ens_accq | 0.9142 | `sev_gauss_high` | 0.9418 | `sev_extrapolate_high` | 0.8591 | +0.0130 |
| pes_ens_consensus | 0.8935 | `joint_low_short` | 0.9783 | `len_extrapolate_long` | 0.8308 | -0.0101 |
| pes_ens_consensus_prior | 0.9180 | `joint_extrap_both` | 0.9684 | `len_extrapolate_long` | 0.8758 | -0.0049 |
| pes_ens_trf_guard | 0.9279 | `joint_extrap_both` | 0.9969 | `len_extrapolate_long` | 0.8613 | -0.0026 |

## 2. Mean degradation by scenario family

| Family | # scenarios | Mean degradation |
|---|---:|---:|
| Severidad | 9 | -0.0009 |
| Longitud | 5 | +0.0158 |
| Conjunta | 4 | -0.0134 |
| Estructura | 3 | +0.0000 |

## 3. Most degraded cells

| Rank | Model | Scenario | Reference | Cell | Degradation |
|---:|---|---|---:|---:|---:|
| 1 | pes_ens_trf_guard | `len_extrapolate_long` | 0.9279 | 0.8613 | +0.0666 |
| 2 | pes_ens_consensus | `len_extrapolate_long` | 0.8935 | 0.8308 | +0.0627 |
| 3 | pes_ens_accq | `sev_extrapolate_high` | 0.9142 | 0.8591 | +0.0552 |
| 4 | pes_ens_sprb | `sev_extrapolate_high` | 0.9142 | 0.8604 | +0.0538 |
| 5 | pes_ens_accq | `joint_uniform_geom` | 0.9142 | 0.8622 | +0.0521 |

## 4. Artefacts

* Matrices: `matrices/*.csv`
* Figures: `figures/*.png`
* Cells: `cells/<model>__<scenario>.json`
