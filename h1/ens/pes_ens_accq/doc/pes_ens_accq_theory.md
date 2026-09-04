# pes_ens_accq — Fundamentos teóricos

> Paquete: `ens.pes_ens_accq`  
> Enfoque: **votación de acciones con desempate por Q-value**

---

## 1. Objetivo

La idea principal es combinar la evidencia de varios modelos sin depender exclusivamente de una suma de probabilidades. Cuando varios miembros proponen acciones distintas, se prioriza la acción que reúne mayor apoyo y, en caso de empate, la que tenga mayor valor estimado de la Q-function.

---

## 2. Agregación

La decisión del ensemble puede describirse como:

$$
\hat{a} = \arg\max_a \left( \sum_m w_m \cdot \mathbf{1}[a = a_m] + \lambda \cdot Q_m(a) \right)
$$

donde:

- $a_m$ es la acción propuesta por el miembro $m$,
- $w_m$ es el peso asociado a la confianza del miembro,
- $Q_m(a)$ representa el valor estimado para esa acción,
- $\lambda$ ajusta la importancia del valor relativo en el desempate.

---

## 3. Ventaja del enfoque

Esta estrategia conserva una decisión discreta y clara, pero evita que una votación simple se vuelva frágil ante empates o distribuciones muy similares. El desempate basado en Q-value incorpora más información de utilidad que la mera cuenta de votos.

---

## 4. Relación con los otros ensembles

`pes_ens_accq` es una variante distinta de `pes_ens_sprb`: el primero enfatiza la acción y el valor, mientras que el segundo prioriza la agregación probabilística suave.

---

## 5. Referencia práctica

Para la guía de uso, consulte `pes_ens_accq_explained.md`.
