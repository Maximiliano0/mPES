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
\text{score}(a) = \sum_m w_m\, C_m^{\,p}\, \mathbf{1}[a = a_m],
\qquad
\hat{a} = \arg\max_{a \in \mathcal{C}} \sum_m \tilde{Q}_m(a)
$$

donde:

- $a_m = \arg\max_a Q_m(a)$ es la acción propuesta por el miembro $m$ sobre
  las acciones factibles,
- $w_m$ es el peso del miembro y $C_m$ su confianza (entropía inversa
  normalizada), con exponente $p$ (`confidence_power`),
- $\mathcal{C} = \arg\max_a \text{score}(a)$ es el conjunto de acciones más
  votadas,
- $\tilde{Q}_m(a)$ es el Q normalizado del miembro $m$, usado **solo para
  desempatar** dentro de $\mathcal{C}$.

No existe un coeficiente $\lambda$: el Q normalizado no se suma al puntaje de
votos, solo resuelve empates entre las acciones más votadas.

---

## 3. Ventaja del enfoque

Esta estrategia conserva una decisión discreta y clara, pero evita que una votación simple se vuelva frágil ante empates o distribuciones muy similares. El desempate basado en Q-value incorpora más información de utilidad que la mera cuenta de votos.

---

## 4. Relación con los otros ensembles

`pes_ens_accq` es una variante distinta de `pes_ens_sprb`: el primero enfatiza la acción y el valor, mientras que el segundo prioriza la agregación probabilística suave.

---

## 5. Referencia práctica

Para la guía de uso, consulte `pes_ens_accq_explained.md`.

## Referencias

- Dietterich, T. G. (2000). Ensemble methods in machine learning. In *Multiple Classifier Systems* (pp. 1–15). Springer.
- Van Hasselt, H. (2010). Double Q-learning. In *Advances in Neural Information Processing Systems*, 23.
