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
 a = \arg\max_{m \neq trf} \left( w_m \cdot \pi_m(a \mid s) \right)
$$

donde $C_{trf}(s)$ es la confianza del Transformer y $\theta$ es el umbral configurado.

---

## 3. Beneficio

Esta regla mejora la robustez del ensemble en escenarios fuera de distribución, donde los modelos basados en patrones complejos pueden estar sobreconfiados o producir decisiones demasiado extremas.

---

## 4. Referencia práctica

Para la guía de uso, consulte `pes_ens_trf_guard_explained.md`.
