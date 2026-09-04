# pes_base — Fundamentos teóricos

> Paquete: `tabular.pes_base`  
> Base teórica: **Q-Learning tabular**

---

## 1. Marco del problema

El escenario se modela como un MDP:

$$
(S, A, P, R, \gamma)
$$

donde:

- $S$: estado del sistema, formado por recursos disponibles, número de trial y severidad.
- $A$: acciones discretas de asignación de recursos, entre `0` y `10`.
- $P$: transiciones deterministas del entorno.
- $R$: recompensa asociada a la severidad acumulada.
- $\gamma$: factor de descuento.

---

## 2. Regla de actualización de Q-Learning

El algoritmo aprende una tabla $Q(s, a)$ por la regla:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]
$$

con:

- $\alpha$: tasa de aprendizaje,
- $\gamma$: descuento de recompensas futuras,
- $r$: recompensa instantánea,
- $s'$: siguiente estado.

Esto permite que el agente aprenda a asignar recursos minimizando la severidad total acumulada.

---

## 3. Política

La política implícita del agente es:

$$
\pi(s) = \arg\max_{a} Q(s, a)
$$

y el objetivo es maximizar la recompensa esperada acumulada a lo largo de la secuencia.

---

## 4. Especialización del escenario mPES

En `Pandemic` el estado se representa mediante una tripleta:

$$
(s_t) = [\text{resources\_left}, \text{trial\_no}, \text{severity}]
$$

y cada decisión establece cuántos recursos se asignan a la ciudad activa. La recompensa se deriva de la severidad acumulada tras la transición del entorno.

---

## 5. Relación con otras líneas

`pes_base` es la referencia mínima, mientras que:

- `pes_ql` incorpora optimización Bayesiana de hiperparámetros,
- `pes_dql` usa doble estimación y warm-up por epsilon decay,
- `pes_dqn`, `pes_rdqn`, `pes_a2c` y `pes_trf` usan aproximadores neuronales.

La línea base es útil como control para medir el valor añadido de cada mejora algorítmica.

---

## 6. Notas de documentación

Esta teoría es la base formal del paquete. Para la descripción práctica del entrenamiento y la ejecución, consulte `pes_base_explained.md`.
