# `pes_dqn` — Fundamento teórico

> Paquete: `ml.pes_dqn`
> Documento de teoría para el agente Deep Q-Network del Pandemic Scenario.

---

## 1. Aproximación de la función de valor con redes neuronales

El Q-Learning clásico almacena $Q(s, a)$ en una tabla y la actualiza
mediante la ecuación de Bellman muestral (Sutton & Barto, 2018):

$$
Q(s, a) \leftarrow Q(s, a) + \alpha\bigl[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\bigr].
$$

Esto exige enumerar todos los pares $(s, a)$. En el Pandemic Scenario el
estado vive en $[0, 1]^3$:

$$
s = \bigl[r/30,\; t/10,\; \sigma/9\bigr],
$$

con $r \in \{0,\dots,30\}$ recursos *libres* (39 totales menos 9
pre-asignados a las dos ciudades iniciales), $t \in \{0,\dots,10\}$ y
$\sigma \in \{0,\dots,9\}$ discreto. Una tabla obliga a tratar cada
severidad como independiente, lo que produce dos
patologías:

1. **Maldición de la dimensionalidad**: si discretizamos $\sigma$ en 100
   bins, hay $40 \cdot 10 \cdot 100 = 40{,}000$ celdas; muchas nunca se
   visitan en 175 000 episodios.
2. **Sin generalización**: los estados $\sigma=0.51$ y $\sigma=0.52$ se
   tratan como *independientes*, aunque la política óptima sea idéntica.

DQN (Mnih et al., 2015) sustituye la tabla por una red neuronal
$Q_\theta(s, a)$. En este paquete:

```
Input  s ∈ ℝ³
   │
   ▼
Dense(64, ReLU)  ───►  Dense(64, ReLU)  ───►  Dense(11, lineal)
                                                  │
                                                  ▼
                                       Q-values para 11 acciones (0..10)
```

construida en [ext/dqn_model.py](../ext/dqn_model.py) por
`build_q_network(state_dim=3, action_dim=11, hidden_units=[64, 64])`. La
red **interpola**: dos estados próximos producen valores próximos, lo que
acelera el aprendizaje y elimina la necesidad de visitar exhaustivamente
el espacio.

---

## 2. *Experience Replay* — descorrelacionar las muestras

Las redes neuronales entrenadas con SGD asumen muestras **i.i.d.** Pero
las transiciones consecutivas de un episodio están **fuertemente
correlacionadas**: $s_{t+1}$ depende de $s_t$. Si entrenásemos online, la
red sobreajustaría a la trayectoria reciente y olvidaría experiencias
pasadas (*catastrophic forgetting*).

La solución, propuesta por Lin (1992) y popularizada por Mnih et al.
(2015), es el **Experience Replay Buffer**:

$$
\mathcal{D} = \{(s_i, a_i, r_i, s'_i, d_i)\}_{i=1}^{N}, \quad N = 20{,}000.
$$

En cada paso de gradiente se muestrean $B = 128$ transiciones
**uniformemente** de $\mathcal{D}$. Beneficios:

- **Decorrelación**: las muestras del minibatch provienen de episodios
  distintos.
- **Reutilización**: cada transición se usa muchas veces (eficiencia
  estadística).
- **Estabilidad**: el gradiente esperado es más cercano al gradiente
  poblacional.

En el código, `ReplayBuffer` (en `dqn_model.py`) se implementa como una
cola circular `collections.deque(maxlen=20000)` con muestreo
`random.sample`.

---

## 3. *Target Network* — estabilidad del semi-gradiente

La actualización TD es un **semi-gradiente**: el objetivo
$y = r + \gamma \max_a Q_\theta(s', a)$ depende también de $\theta$. Si
actualizamos $\theta$ y *al instante* recalculamos $y$, perseguimos un
blanco en movimiento y el entrenamiento diverge.

DQN introduce una **red objetivo** $Q_{\theta^-}$ con parámetros
**congelados** $\theta^-$ que se sincronizan cada $C$ pasos:

$$
y = r + \gamma \max_a Q_{\theta^-}(s', a), \quad
\theta^- \leftarrow \theta \text{ cada } C = 1\,000 \text{ pasos.}
$$

En el código:

```python
sync_target_network(q_online, q_target)   # copia pesos
```

llamado cada `DQN_TARGET_SYNC_FREQ = 1 000` pasos en `train_dqn.py`. Esto
estabiliza el aprendizaje al ritmo de un Q-Learning casi *off-policy*
puro entre sincronizaciones.

---

## 4. *Double DQN* — sesgo de maximización

El operador $\max_a Q(s', a)$ está **sesgado al alza**: si los $Q$
estimados tienen ruido, $\max$ amplifica ese ruido y sobreestima
sistemáticamente. Hasselt et al. (2016) demostraron que el sesgo se
elimina **desacoplando** la selección de la evaluación:

$$
\boxed{\;y_{\text{DDQN}} = r + \gamma \cdot Q_{\theta^-}\!\bigl(s',\; \arg\max_a Q_\theta(s', a)\bigr)\;}
$$

- La red **online** $Q_\theta$ elige la acción $a^*$.
- La red **objetivo** $Q_{\theta^-}$ evalúa $Q(s', a^*)$.

`train_step_dqn(...)` implementa exactamente esa fórmula. En este
proyecto la mejora vs. DQN simple fue ~1.5 puntos porcentuales en
`raw_mean_perf` durante las pruebas preliminares.

---

## 5. Pérdida de Huber — robustez ante *outliers*

La pérdida cuadrática $\tfrac{1}{2}(y - Q)^2$ tiene gradiente proporcional
al error. Si una transición produce un error TD enorme (severidad
explotando, recompensa muy negativa), el gradiente domina y desestabiliza.

La **pérdida de Huber** $L_\delta$ se comporta como cuadrática para
errores pequeños y lineal para errores grandes:

$$
L_\delta(e) = \begin{cases}
\tfrac{1}{2} e^2 & |e| \le \delta, \\
\delta\bigl(|e| - \tfrac{1}{2}\delta\bigr) & |e| > \delta.
\end{cases}
$$

Con $\delta = 1$ (valor por defecto en
`tf.keras.losses.Huber()`), los gradientes están acotados y el
entrenamiento es robusto a *outliers* del *replay*.

---

## 6. Política ε-greedy con decaimiento + *warm-up*

La exploración sigue:

$$
\pi_\varepsilon(a \mid s) = \begin{cases}
1 - \varepsilon + \tfrac{\varepsilon}{|\mathcal{A}|} & a = \arg\max_a Q_\theta(s, a), \\
\tfrac{\varepsilon}{|\mathcal{A}|} & \text{en otro caso.}
\end{cases}
$$

con dos fases:

- **Warm-up**: durante los primeros episodios $\varepsilon = 1$ (todo
  aleatorio) para llenar el `ReplayBuffer` con datos diversos antes del
  primer paso de gradiente.
- **Decaimiento exponencial**: $\varepsilon_{t+1} = \max(\varepsilon_{\min},\, \lambda\,\varepsilon_t)$
  con $\varepsilon_{\min} \approx 0.069$ (`DQN_EPSILON_MIN`) y $\lambda$
  derivado de los ratios `DQN_WARMUP_RATIO` / `DQN_TARGET_RATIO` (ambos
  ajustables por Optuna).

Esto compone el típico equilibrio **exploración–explotación** y respeta
las condiciones de convergencia de Robbins–Monro adaptadas (Sutton &
Barto, 2018, cap. 6).

---

## 7. Optimización bayesiana con TPE

La calidad final de DQN es muy sensible a $\alpha$, $\gamma$, tamaño de
red, $C$ y $\lambda$. Una búsqueda en cuadrícula sobre 6 dimensiones es
prohibitiva. Optuna (Akiba et al., 2019) usa
**Tree-structured Parzen Estimator (TPE)**:

1. Mantiene dos densidades sobre el espacio de hiperparámetros:
   $\ell(x)$ para los *trials* "buenos" (mejor cuantil $\gamma_q$) y
   $g(x)$ para el resto.
2. Propone $x^* = \arg\max_x \ell(x)/g(x)$ — equivalentemente, maximiza
   el *Expected Improvement* bajo un modelo no paramétrico.

En este proyecto:

- Storage SQLite persistente.
- *Pruner* `MedianPruner` para abortar *trials* claramente malos.
- Mejor configuración hallada en §4 de `pes_dqn_explained.md`.

---

## 8. Anclaje al código

| Concepto teórico | Implementación |
|---|---|
| Red Q $Q_\theta$ | `build_q_network()` en `dqn_model.py` |
| Replay buffer $\mathcal{D}$ | Clase `ReplayBuffer` |
| Sincronización de target | `sync_target_network()` cada `DQN_TARGET_SYNC_FREQ` |
| Objetivo Double DQN | Bloque `tf.GradientTape` en `train_step_dqn()` |
| Pérdida Huber | `tf.keras.losses.huber()` (δ=1) |
| ε-greedy | Selección inline en el bucle de episodios de `DQNTraining` (`pandemic.py`) |
| TPE bayesiano | `optuna.create_study(sampler=TPESampler)` en `optimize_dqn.py` |

---

## Referencias

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A
next-generation hyperparameter optimization framework. En *Proceedings of
the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data
Mining* (pp. 2623–2631). ACM.

Hasselt, H. van, Guez, A., & Silver, D. (2016). Deep reinforcement
learning with double Q-learning. En *Proceedings of the Thirtieth AAAI
Conference on Artificial Intelligence* (pp. 2094–2100). AAAI Press.

Lin, L.-J. (1992). Self-improving reactive agents based on reinforcement
learning, planning and teaching. *Machine Learning, 8*(3–4), 293–321.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare,
M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G.,
Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran,
D., Wierstra, D., Legg, S., & Hassabis, D. (2015). Human-level control
through deep reinforcement learning. *Nature, 518*(7540), 529–533.

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An
introduction* (2nd ed.). MIT Press.
