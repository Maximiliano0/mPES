# pes_ens_trf_guard — Fundamentos teóricos

> Paquete: `ens.pes_ens_trf_guard`  
> Enfoque: **guardia por confianza con prioridad Transformer**

---

## 1. Motivación

Los modelos de secuencia (como `pes_trf`) suelen aportar mejor capacidad de representación cuando la estructura temporal es relevante. Sin embargo, también pueden volverse poco fiables bajo condiciones de estrés. La estrategia de guardia consiste en evitar aceptar una decisión de Transformer cuando la confianza no es suficiente.

---

## 2. Regla de decisión

La política puede describirse como:

$$
\text{si } C_{trf}(s) \ge \theta \quad \Rightarrow \quad a = a_{trf}
$$

y en caso contrario:

$$
 a = \begin{cases}
 a_{trf} & \text{si } C_{trf}(s) \ge \theta \\
 \arg\max_a \sum_m w_m \cdot \pi_m(a \mid s) & \text{si } C_{trf}(s) < \theta
 \end{cases}
$$

donde $C_{trf}(s)$ es la confianza del Transformer y $\theta$ es el umbral
configurado. La compuerta real es suave,
$\text{gate} = \sigma\bigl(\kappa\,(C_{trf} - \theta)\bigr)$, y se decide por
$\text{gate} \ge 0.5$, lo que equivale a $C_{trf} \ge \theta$. En el ramo de
*fallback* la suma ponderada recorre **todos** los miembros del ensemble
(el Transformer incluido), no solo $m \neq trf$.

---

## 3. Beneficio

Esta regla mejora la robustez del ensemble en escenarios fuera de distribución, donde los modelos basados en patrones complejos pueden estar sobreconfiados o producir decisiones demasiado extremas.

---

## 4. Referencia práctica

Para la guía de uso, consulte `pes_ens_trf_guard_explained.md`.

## Referencias

- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning* (pp. 1321–1330). PMLR.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. In *Advances in Neural Information Processing Systems*, 30.
