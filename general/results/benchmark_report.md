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

## 5. CSV matrix distribution analysis

Eight `matrix_*.csv` artefacts (rows = 7 models, columns = 22 scenarios) crystallise
the sweep. Reading them jointly yields the following picture:

### 5.1 `matrix_global_mean.csv` — central tendency
* Row order by mean across all 22 scenarios: **pes_ens (0.937) > pes_trf (0.927) > pes_rdqn (0.899) ≈ pes_dqn (0.894) ≈ pes_dql (0.893) > pes_a2c (0.887) ≈ pes_ql (0.887)**.
* `pes_ens` is the only model that stays ≥ 0.90 in every column; `pes_trf` is the only single model achieving the same on 19/22.
* Worst single cell of the whole sweep: `pes_ql @ sev_extrapolate_high = 0.7623` (−12.4 pp vs its own baseline). Tabular Q‑Learning collapses when severity is pushed beyond the empirical support.
* Best single cell: `pes_ens @ {sev_extrapolate_high, joint_extrap_both} = 1.0000` — soft voting is the only way to *gain* under the most aggressive extrapolation.

### 5.2 `matrix_ood_degradation.csv` — sign and magnitude of shift
Positive = degradation vs `sev_empirical` baseline, negative = improvement.

* **Most damaging axis**: `sev_extrapolate_high` (mean +0.046 across models, dominated by tabular: +0.124 ql, +0.111 dql).
* **Most rewarding axis**: `joint_extrap_both` is bipolar — catastrophic for tabular (+0.07/+0.08) but a clear win for the deep family (`pes_a2c` −0.061, `pes_trf` −0.070, `pes_ens` −0.063).
* `len_*` family is the second toughest (mean +0.012); the long-tail length shift `len_extrapolate_long` degrades **all** models (all six positive).
* Structural columns (`struct_*`) are exactly 0 because they alter trial counts, not the per-sequence distribution; they confirm metric stability rather than measure transfer.

### 5.3 `matrix_std.csv` — within-scenario reliability
* `pes_ens` has the lowest mean σ (0.038); it touches σ = 0 on the two `extrapolate` cells, i.e. perfect inter-sequence consistency.
* `pes_ql` and `pes_rdqn` show the largest σ (≈0.08–0.13 in joint scenarios), confirming wider sequence-level dispersion when faced with shift.
* High σ co-occurs with low `min_perf` → dispersion comes from worst-case sequences, not noisy averages.

### 5.4 `matrix_min.csv` / `matrix_max.csv` — tail behaviour
* Most models reach the 1.0 ceiling on at least one sequence per scenario, so `max` is poorly discriminating.
* `min` is the more informative end:
  * `pes_ql` collapses to **0.168** on a single `joint_low_short` sequence; `pes_dql` to 0.286 there. The same scenario also exposes the deep-family ensemble (0.71) but `pes_dqn`/`pes_a2c` keep a 0.93 floor.
  * `pes_ens` keeps a worst-sequence floor ≥ 0.57 across the whole 22-scenario × 64-sequence grid; `pes_trf` ≥ 0.55. These are the two safest models for tail-risk applications.

### 5.5 `matrix_cohen_d.csv` — standardised effect size
Magnitudes follow the Cohen convention (|d| ≥ 0.8 large, ≥ 1.5 very large).

* Largest *negative* effects (real degradation): `pes_dql @ sev_extrapolate_high` d = −2.87, `pes_ql @ sev_extrapolate_high` d = −2.71, `pes_dql @ joint_extrap_both` d = −1.99, `pes_ql @ joint_extrap_both` d = −1.83.
* Largest *positive* effects (real improvement): `pes_ens @ {sev_extrapolate_high, joint_extrap_both}` d = +2.52, `pes_trf @ joint_extrap_both` d = +2.14, `pes_trf @ sev_extrapolate_high` d = +2.08, `pes_a2c @ joint_extrap_both` d = +1.36.
* Practical reading: tabular Q-Learning is *very-large-effect* worse than its baseline whenever extrapolation is involved; the transformer and the ensemble are *very-large-effect* better. The DQN family sits in the middle.

### 5.6 `matrix_welch_p.csv` — statistical significance
* Three columns are uniformly significant at p < 0.001 across **all** seven models: `sev_extrapolate_high`, `len_extrapolate_long`, `joint_extrap_both`. These are the three scenarios that genuinely stress every architecture.
* Three structural columns (`struct_*`) return p = 1.0 across the board → the metric is invariant under block-count rearrangement (sanity check passes).
* Mid-significance pockets (0.01 < p < 0.05): `sev_uniform`, `sev_weibull`, `sev_beta_lowskew` for tabular only — distributional reshapings of the severity prior hurt tabular but are absorbed by deep models.

### 5.7 `matrix_action_kl.csv` — policy distribution shift
KL is computed between per-trial action histograms vs the baseline.

* `pes_dqn` exhibits very high KL on several severity perturbations (3.14 on `sev_gauss_low`, 2.25 on `sev_beta_lowskew`) yet keeps performance high → it changes its policy aggressively but coherently.
* `pes_a2c` shows **zero** KL on every severity scenario: its policy is invariant to severity input. Combined with its competitive performance, this signals partial policy collapse onto a robust default action sequence rather than true context-conditioning.
* `pes_ens` and `pes_trf` show moderate KL (0.05–1.5) — adaptive but smooth.
* All structural columns return KL = 0 (action histogram is structure-invariant, as expected).

## 6. Heatmap conclusions

The four PNGs visualise the matrices above with model rows ordered exactly as in §1.

### 6.1 `heatmap_global_mean.png`
* Bottom two rows (`pes_trf`, `pes_ens`) form a uniformly bright band → robust dominance across **every** column.
* Top two rows (`pes_ql`, `pes_dql`) carry the only visible dark patch on the extreme-right side of the severity block (`sev_extrapolate_high`) and on `joint_extrap_both`.
* Structural columns are visually flat (each model's own baseline) — confirming they should be read as control columns.

### 6.2 `heatmap_ood_degradation.png` (diverging palette around 0)
* Two vertical "danger bars" stand out in red: `sev_extrapolate_high` and `joint_extrap_both` for the tabular rows.
* The same columns appear **green** for `pes_trf` and `pes_ens` → these models do not just resist shift, they *exploit* it (the harder distributions are easier for them because the ensemble/transformer policy is closer to optimal there).
* `len_extrapolate_long` is uniformly orange → the only column that hurts every model.
* Structural columns are pure white (≈ 0) → confirms invariance.

### 6.3 `heatmap_welch_logp.png` (brighter = stronger evidence of shift)
* Three bright vertical bands match the three "extrapolate" scenarios identified in §5.6.
* Tabular rows (`pes_ql`, `pes_dql`) light up on additional severity columns (`sev_uniform`, `sev_weibull`, `sev_beta_lowskew`) — they detect shift where deep models do not.
* Three black columns (`struct_*`) are the no-shift control band.

### 6.4 `heatmap_action_kl.png` (brighter = larger policy divergence)
* `pes_dqn` row is the brightest overall on the severity block → most reactive policy under severity perturbation.
* `pes_a2c` row is almost entirely dark on severity → flat policy regardless of severity context (collapse signal).
* `pes_trf` and `pes_ens` rows show medium intensity, broadly distributed → adaptive but stable.
* Length and joint columns are uniformly bright across all rows → length perturbations force every model to change actions.
* Structural columns are pure black → no policy change, as expected.

## 7. Overall resolutions

1. **Best generalist**: `pes_ens` — highest mean (0.937), tightest dispersion (σ̄ = 0.038), highest worst-sequence floor (0.57), most positive Cohen's d under the toughest extrapolations.
2. **Best single model**: `pes_trf` — second on every metric and the only non-ensemble that *improves* under `sev_extrapolate_high` and `joint_extrap_both` with a very-large positive effect size.
3. **Most fragile**: `pes_dql` then `pes_ql` — highest mean degradation, only models with d < −1.5 on multiple scenarios, lowest min-sequence performance (0.17 floor for `pes_ql` on `joint_low_short`).
4. **Hidden caveat**: `pes_a2c` reaches competitive performance with **zero severity action-KL** — it has likely collapsed to a context-insensitive policy that happens to be near-optimal. Suspect under further OOD pressure not covered here.
5. **Three universal stressors**: `sev_extrapolate_high`, `len_extrapolate_long`, `joint_extrap_both`. They are the only scenarios that produce p < 0.001 across all seven models and should be treated as the headline benchmark cells.
6. **Three control columns**: `struct_few_long_blocks`, `struct_many_short_blocks`, `struct_more_total` show 0 degradation, p = 1.0, and KL = 0 — the metric and pipeline are invariant to block-count rearrangements, which validates the harness rather than measuring transfer.
7. **Family takeaways**:
   * Severity family: deep models > tabular by a wide margin on the extrapolative tails, comparable on in-support reshapings.
   * Length family: hardest for everyone; long-tail extrapolation dominates the cost.
   * Joint family: bimodal — catastrophic for tabular, beneficial for transformer/ensemble.
   * Structural family: control band; useful only as a sanity check.
