# pes_ens_consensus_prior — Fundamentos teóricos

> Paquete: `ens.pes_ens_consensus_prior`  
> Enfoque: **consenso con prioridad informada por severidad**

---

## 1. Motivación

Cuando varios modelos aportan decisiones distintas, la combinación del ensemble no debe depender únicamente de la frecuencia o del valor bruto de salida del modelo. En este paquete la decisión final integra tres ideas principales:

1. los miembros más confiables deben influir más,
2. la concordancia entre miembros debe recompensarse,
3. la severidad del estado debe introducir una prior informativa útil para la acción.

Esto permite un arreglo más robusto en situaciones con riesgo o incertidumbre sostenida.

---

## 2. Confianza del miembro

Cada modelo produce una distribución de acciones sobre las opciones factibles. A partir de esa distribución se calcula la entropía:

$$
H(p) = -\sum_a p(a)\log p(a)
$$

y la confianza se define como una versión normalizada de la incertidumbre inversa:

$$
C = 1 - \frac{H}{\log(|\mathcal{A}|)}
$$

donde $|\mathcal{A}|$ es el número de acciones factibles. Esta métrica queda acotada en $[0,1]$ y aumenta cuando la distribución es más concentrada.

---

## 3. Agregación basada en consenso

Para cada acción $a$, el ensemble acumula una puntuación:

$$
S(a) = \sum_m w_m \, C_m^{\gamma} \, p_m(a)
+ \beta_a \sum_{m: a_m = a} C_m
- \beta_d \sum_{m: a_m \neq a} C_m
$$

donde:

- $m$ es un miembro del ensemble,
- $w_m$ es el peso del modelo,
- $C_m$ es la confianza del modelo,
- $\gamma$ es `confidence_power`,
- $p_m(a)$ es la probabilidad de acción $a$ emitida por el modelo,
- $\beta_a$ es el bono por acuerdo (`agreement_bonus`),
- $\beta_d$ es la penalización por desacuerdo (`disagreement_penalty`).

La idea es que una decisión apoyada por varios modelos confiables reciba una recompensa, mientras que una acción minoritaria y poco acorde con el resto sea penalizada.

---

## 4. Prior de severidad

El ensemble añade un prior basado en la severidad del estado actual:

$$
P_{prior}(a) \propto \exp\left(-\frac{(a - s)^2}{2\sigma^2}\right)
$$

donde:

- $s$ es la severidad estimada del estado,
- $\sigma$ es `prior_sigma`,
- la fuerza del prior está controlada por `prior_weight`.

La puntuación final se combina de la siguiente forma:

$$
S_{final}(a) = (1 - \lambda) S(a) + \lambda P_{prior}(a)
$$

con $\lambda = \text{prior\_weight}$. De este modo, la decisión se sale del mero consenso y incorpora contexto clínico o de riesgo del estado.

---

## 5. Ventaja principal

La ventaja del enfoque es que combina:

- precisión local de cada modelo,
- estabilidad de la decisión por consenso,
- orientación contextual por severidad.

En escenarios de alta incertidumbre, esta mezcla tiende a ser más robusta que una simple votación o que la agregación por probabilidad sin contexto.

---

## 6. Relación con otros ensembles

`pes_ens_consensus_prior` generaliza la idea de `pes_ens_consensus` añadiendo un prior informado por severidad. Mientras que el consenso base recompensa concordancia y penaliza divergencia, esta variante incorpora además la estructura del estado para reordenar la decisión final.

---

## 7. Referencia práctica

Para la guía de uso, consulte `pes_ens_consensus_prior_explained.md`.

## Referencias

- Dietterich, T. G. (2000). Ensemble methods in machine learning. In *Multiple Classifier Systems* (pp. 1–15). Springer.
- Kuncheva, L. I. (2004). *Combining Pattern Classifiers: Methods and Algorithms*. Wiley.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
