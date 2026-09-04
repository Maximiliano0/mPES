# pes_ens_sprb — Fundamentos teóricos

> Paquete: `ens.pes_ens_sprb`  
> Enfoque: **soft voting ponderado**

---

## 1. Idea central

Cada miembro del ensemble produce una distribución sobre las acciones posibles:

$$
\pi_m(a \mid s)
$$

La agregación final toma la forma:

$$
\pi_{ens}(a \mid s) = \frac{\sum_m w_m \pi_m(a \mid s)}{\sum_m w_m}
$$

donde $w_m$ es un peso de confianza y/o robustez del miembro.

---

## 2. Por qué soft voting

La votación dura fuerza una unica acción y puede producir empates o decisiones demasiado abruptas. La votación suave mantiene la incertidumbre distribuida y permite que:

- los miembros más seguros dominen la decisión,
- la mezcla sea más estable ante ruido o decisiones de baja confianza,
- se califique la decisión final mediante entropía o funciones de confianza.

---

## 3. Factibilidad y regularización

Antes de combinar distribuciones, el paquete suele descartar acciones no factibles por recursos disponibles. Esto evita que la distribución agregada asigne masa a decisiones imposibles y mantiene la decisión dentro del espacio operativo real del escenario.

---

## 4. Relación con el benchmark

La variante `pes_ens_sprb` está pensada para ser comparada con otros ensembles activos del proyecto y con los modelos individuales bajo la matriz de estrés del benchmark general.

---

## 5. Referencia práctica

Para la guía de operación, consulte `pes_ens_sprb_explained.md`.
