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

La función de decisión puede entenderse como una combinación de apoyo positivo y penalización negativa:

$$
S(a) = \sum_m w_m \cdot \mathbf{1}[a = a_m] - \lambda \sum_{m \neq n} \mathbf{1}[a_m \neq a_n]
$$

Donde:

- $a_m$ es la acción propuesta por el miembro $m$,
- $w_m$ es su peso de confianza,
- $\lambda$ penaliza la discordancia entre miembros.

---

## 3. Ventaja

Este enfoque favorece decisiones que son coherentes con la mayoría o con los miembros con mayor confianza, reduciendo la variabilidad y la fragilidad del ensemble ante decisiones aisladas muy extremas.

---

## 4. Relación con otros ensembles

`pes_ens_consensus` es una variante alternativa a `pes_ens_sprb` y `pes_ens_accq`: mantiene la idea de combinación de múltiples modelos, pero reorienta la agregación hacia la concordancia más que hacia la probabilidad o el valor absoluto.

---

## 5. Referencia práctica

Para la guía de uso, consulte `pes_ens_consensus_explained.md`.
