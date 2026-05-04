# mPES OOD Benchmark Report

_Generated: 2026-05-04T21:30:00.917194Z_

**Reference baseline scenario:** `sev_empirical`

**Models evaluated:** 7 -- pes_ql, pes_dql, pes_dqn, pes_rdqn, pes_a2c, pes_trf, pes_ens
**Scenarios evaluated:** 22

## 1. Per-model best / worst

| Model | Baseline mean | Best scenario | Best mean | Worst scenario | Worst mean | Mean degradation |
|---|---:|---|---:|---|---:|---:|
| pes_ql | 0.8866 | `len_all_short` | 0.9426 | `sev_extrapolate_high` | 0.7623 | +0.0152 |
| pes_dql | 0.8963 | `sev_beta_highskew` | 0.9238 | `sev_extrapolate_high` | 0.7854 | +0.0191 |
| pes_dqn | 0.8937 | `joint_low_short` | 0.9877 | `len_extrapolate_long` | 0.8412 | -0.0050 |
| pes_rdqn | 0.8987 | `sev_gauss_high` | 0.9319 | `sev_extrapolate_high` | 0.8311 | +0.0099 |
| pes_a2c | 0.8872 | `joint_low_short` | 0.9877 | `len_extrapolate_long` | 0.8257 | -0.0090 |
| pes_trf | 0.9272 | `joint_extrap_both` | 0.9969 | `len_extrapolate_long` | 0.8597 | -0.0025 |
| pes_ens | 0.9373 | `sev_extrapolate_high` | 1.0000 | `len_extrapolate_long` | 0.9000 | -0.0021 |

## 2. Mean degradation by scenario family

| Family | # scenarios | Mean degradation across models |
|---|---:|---:|
| severity | 9 | +0.0062 |
| length | 5 | +0.0115 |
| joint | 4 | -0.0090 |
| structural | 3 | +0.0000 |

## 3. Top 5 most-degraded cells

| Rank | Model | Scenario | Baseline | Cell | Degradation |
|---:|---|---|---:|---:|---:|
| 1 | pes_ql | `sev_extrapolate_high` | 0.8866 | 0.7623 | +0.1244 |
| 2 | pes_dql | `sev_extrapolate_high` | 0.8963 | 0.7854 | +0.1110 |
| 3 | pes_dql | `joint_uniform_geom` | 0.8963 | 0.8163 | +0.0800 |
| 4 | pes_ql | `joint_extrap_both` | 0.8866 | 0.8073 | +0.0794 |
| 5 | pes_dql | `joint_extrap_both` | 0.8963 | 0.8286 | +0.0678 |

## 4. Figures

* ![heatmap_global_mean.png](heatmap_global_mean.png)
* ![heatmap_ood_degradation.png](heatmap_ood_degradation.png)
* ![heatmap_welch_logp.png](heatmap_welch_logp.png)
* ![heatmap_action_kl.png](heatmap_action_kl.png)

Per-scenario histograms in `per_sequence_histograms/`.
