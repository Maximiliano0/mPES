# pes_ens_sprb — Fundamentos teóricos

> Paquete: `ens.pes_ens_sprb`  
> Enfoque: **soft voting ponderado**

---

## 1. Idea central

Cada miembro del ensemble produce una distribución sobre las acciones factibles
$a \le R_t$ (recursos disponibles):

$$
\pi_m(a \mid s) =
\begin{cases}
\operatorname{softmax}\bigl(Q_m(s, \cdot)/\tau\bigr)_a & \text{miembros Q (DQN, RDQN, TRF)} \\
\text{probabilidades del actor A2C, renormalizadas} & \text{miembro de política (A2C)}
\end{cases}
$$

La agregación final toma la forma:

$$
S(a) = \sum_m w_m\, C_m^{\,p}\, \pi_m(a \mid s),
\qquad
\hat{a} = \arg\max_{a \le R_t} S(a)
$$

donde:

- $w_m \ge 0$ es el peso del miembro,
- $C_m = 1 - H(\pi_m)/\log |\mathcal{A}_t|$ es su confianza (entropía inversa
  normalizada sobre las $|\mathcal{A}_t|$ acciones factibles); para los miembros
  Q se calcula sobre $\operatorname{softmax}(Q_m/\tau)$, por lo que la
  temperatura $\tau$ afecta tanto a la distribución como a la confianza,
- $p$ es el exponente de confianza (`confidence_power`) y $\tau$ la temperatura
  (`temperature`), ambos optimizados con Optuna junto con los pesos.

La normalización por $\sum_m w_m$ no altera el $\arg\max$; solo se usa para la
confianza reportada del ensemble, $\operatorname{clip}\bigl(\sum_a S(a) / \sum_m w_m,\, 0,\, 1\bigr)$.

---

## 2. Por qué soft voting

La votación dura fuerza una unica acción y puede producir empates o decisiones demasiado abruptas. La votación suave mantiene la incertidumbre distribuida y permite que:

- los miembros más seguros dominen la decisión,
- la mezcla sea más estable ante ruido o decisiones de baja confianza,
- se califique la decisión final mediante entropía o funciones de confianza.

---

## 3. Factibilidad y regularización

Antes de combinar distribuciones, el paquete descarta las acciones no factibles por recursos disponibles. Esto evita que la distribución agregada asigne masa a decisiones imposibles y mantiene la decisión dentro del espacio operativo real del escenario.

---

## 4. Relación con el benchmark

La variante `pes_ens_sprb` está pensada para ser comparada con otros ensembles activos del proyecto y con los modelos individuales bajo la matriz de estrés del benchmark general.

---

## 5. Referencia práctica

Para la guía de operación, consulte `pes_ens_sprb_explained.md`.

## Referencias

- Dietterich, T. G. (2000). Ensemble methods in machine learning. In *Multiple Classifier Systems* (pp. 1–15). Springer.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. In *Advances in Neural Information Processing Systems*, 30.
