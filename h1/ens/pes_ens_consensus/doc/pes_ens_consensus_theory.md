# pes_ens_consensus — Fundamentos teóricos

> Paquete: `ens.pes_ens_consensus`  
> Enfoque: **consenso con acuerdo/desacuerdo**

---

## 1. Motivación

Cuando varios modelos del proyecto aportan información distinta, la decisión del ensemble debe equilibrar dos objetivos:

- maximizar la concordancia entre miembros,
- evitar que un modelo dominante imponga una decisión sin respaldo global.

---

## 2. Agregación de consenso

La función de decisión combina el voto ponderado, un bono por acuerdo, una
penalización por desacuerdo y un término de Q normalizado ponderado por
confianza:

$$
S(a) = \sum_m w_m\, C_m^{\,p}\, \mathbf{1}[a = a_m]
+ \beta_a \!\!\sum_{m:\, a_m = a} \!\! C_m
- \beta_d \!\!\sum_{m:\, a_m \neq a} \!\! C_m
+ \sum_m \tilde{Q}_m(a)\, C_m
$$

Donde:

- $a_m$ es la acción propuesta por el miembro $m$ y $w_m$ su peso,
- $C_m$ es la confianza del miembro (entropía inversa normalizada); el
  exponente $p$ (`confidence_power`) solo afecta al primer término (voto
  ponderado), mientras que los demás términos usan $C_m$ sin exponente,
- $\beta_a$ es el bono por acuerdo (`agreement_bonus`, por defecto $0.5$),
- $\beta_d$ es la penalización por desacuerdo (`disagreement_penalty`, por
  defecto $0.1$); ambos ponderan **confianza**, no un conteo de pares,
- $\tilde{Q}_m(a)$ es el Q normalizado (z-score sobre las acciones factibles)
  del miembro $m$; para el actor A2C se normalizan sus probabilidades de
  política.

La acción elegida es $\hat{a} = \arg\max_{a \le R_t} S(a)$, con $R_t$ los
recursos disponibles.

---

## 3. Ventaja

Este enfoque favorece decisiones que son coherentes con la mayoría o con los miembros con mayor confianza, reduciendo la variabilidad y la fragilidad del ensemble ante decisiones aisladas muy extremas.

---

## 4. Relación con otros ensembles

`pes_ens_consensus` es una variante alternativa a `pes_ens_sprb` y `pes_ens_accq`: mantiene la idea de combinación de múltiples modelos, pero reorienta la agregación hacia la concordancia más que hacia la probabilidad o el valor absoluto.

---

## 5. Referencia práctica

Para la guía de uso, consulte `pes_ens_consensus_explained.md`.

## Referencias

- Dietterich, T. G. (2000). Ensemble methods in machine learning. In *Multiple Classifier Systems* (pp. 1–15). Springer.
- Kuncheva, L. I. (2004). *Combining Pattern Classifiers: Methods and Algorithms*. Wiley.
