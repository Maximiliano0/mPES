# `pes_rdqn` — Fundamento teórico

> Paquete: `ml.pes_rdqn`
> Documento de teoría para el agente Recurrent Deep Q-Network del
> Pandemic Scenario.

---

## 1. Observabilidad parcial: del MDP al POMDP

Un **Proceso de Decisión de Markov (MDP)** asume que el estado $s_t$
contiene **toda** la información necesaria para predecir el futuro:

$$
P(s_{t+1}\mid s_t, a_t, s_{t-1}, \dots, s_0) = P(s_{t+1}\mid s_t, a_t).
$$

En la práctica, muchas tareas son **POMDP** (*Partially Observable
MDP*): el agente recibe una observación $o_t$ que es función ruidosa o
incompleta de un estado oculto $s_t^{\text{real}}$. Sutton & Barto
(2018, cap. 17) muestran que la política óptima en un POMDP depende del
**historial** completo $h_t = (o_1, a_1, o_2, a_2, \dots, o_t)$, no de
una observación aislada.

En el Pandemic Scenario, formalmente $s_t = [r_t, t_t, \sigma_t]$ es
markoviano para una **única** secuencia, pero al evaluar bloques de 8
secuencias con severidades iniciales heterogéneas y longitudes
variables, aparecen factores latentes (carga acumulada del bloque,
variabilidad entre secuencias) que el estado puntual no codifica. Tratar
el problema como POMDP es una aproximación más fiel.

---

## 2. La necesidad de memoria

Una red *feedforward* $Q_\theta(s_t, a)$ no tiene estado interno: cada
predicción depende exclusivamente de $s_t$. Para aproximar la política
óptima en un POMDP necesitamos una arquitectura que **recuerde**.

Hay dos vías clásicas:

1. **Ventana finita de observaciones**: $\tilde s_t = (o_{t-L+1}, \dots, o_t)$.
   Equivale a un MDP de orden $L$. Eficiente pero limitado a memoria
   corta.
2. **Red recurrente**: $h_t = f_\theta(h_{t-1}, o_t)$. Capacidad de
   memoria teóricamente ilimitada (en la práctica acotada por el
   gradiente).

`pes_rdqn` combina ambas: usa una **ventana** $L$ explícita
(`HistoryDeque`) **y** procesa esa ventana con un **LSTM** para extraer
una representación compacta antes de decidir.

---

## 3. LSTM: puertas, estado oculto y celda

La **Long Short-Term Memory** (Hochreiter & Schmidhuber, 1997) resuelve
el problema del *vanishing gradient* de las RNN clásicas mediante una
arquitectura con tres puertas:

$$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) & \text{(forget gate)} \\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) & \text{(input gate)} \\
\tilde c_t &= \tanh(W_c [h_{t-1}, x_t] + b_c) & \text{(candidate)} \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde c_t & \text{(cell state)} \\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) & \text{(output gate)} \\
h_t &= o_t \odot \tanh(c_t) & \text{(hidden state)}
\end{aligned}
$$

donde $\sigma$ es la sigmoide y $\odot$ el producto Hadamard. Las
propiedades clave son:

- **Estado de celda** $c_t$ con conexión aditiva (no multiplicativa),
  evita que el gradiente se desvanezca al retropropagar.
- **Forget gate** $f_t$ aprende **qué olvidar** del pasado.
- **Input gate** $i_t$ aprende **qué incorporar** del presente.

En este paquete se usa una sola capa con `RDQN_LSTM_UNITS = 64` unidades
ocultas (`tf.keras.layers.LSTM(64)`), y solo se conserva la salida final
$h_T$ (no la secuencia entera) como entrada al MLP que produce
$Q(s_{t-L+1:t}, a)$.

---

## 4. Q-Learning recurrente en POMDPs

Hausknecht & Stone (2015) propusieron el **Deep Recurrent Q-Network
(DRQN)** como extensión natural de DQN para POMDPs. La idea: sustituir
$Q_\theta(s, a)$ por $Q_\theta(h_{1:t}, a)$ donde $h_{1:t}$ se procesa
con un LSTM. La actualización TD se conserva:

$$
y_t = r_t + \gamma \cdot \max_a Q_{\theta^-}(h_{1:t+1}, a),
$$

pero ahora cada minibatch contiene **secuencias** $(h_{1:t}, a_t, r_t, h_{1:t+1})$
en vez de transiciones puntuales.

`pes_rdqn` adopta una variante **truncada**: en vez de propagar el
estado oculto del LSTM a lo largo de todo un episodio, reinicia el LSTM
en cada *forward* y procesa la ventana fija de tamaño $L = 6$. Esto
sacrifica memoria infinita por **estabilidad y eficiencia** del replay
(las secuencias tienen forma constante `(L, 3)`), siguiendo la
recomendación práctica de Hausknecht & Stone (2015).

---

## 5. La ventana de historia como estado aproximado

Con la ventana $\tilde s_t = (s_{t-L+1}, \dots, s_t)$ y un LSTM, el
agente aprende implícitamente una **representación de creencia**
$b_\theta(\tilde s_t)$ que aproxima la distribución posterior sobre
estados ocultos. Sutton & Barto (2018, §17.3) describen este enfoque
como **belief-state methods**.

Ventajas frente a apilar simplemente las observaciones (lo que haría una
red densa con entrada $L \times 3$):

- El LSTM aprende **qué pasos del historial pesan más** mediante sus
  puertas, en vez de tratar todas las posiciones por igual.
- Compresión: el estado oculto $h_t \in \mathbb{R}^{64}$ es un resumen
  fijo, independiente de $L$.
- Mejor generalización a secuencias de longitud variable (entre 3 y 10
  trials en este escenario).

---

## 6. Objetivo Double DQN aplicado a redes recurrentes

El objetivo Double DQN (Hasselt et al., 2016) se transfiere sin cambios
a la versión recurrente:

$$
y = r + \gamma \cdot Q_{\theta^-}\!\bigl(\tilde s_{t+1},\; \arg\max_a Q_\theta(\tilde s_{t+1}, a)\bigr),
$$

donde $\tilde s_{t+1}$ es la **ventana actualizada** tras hacer
`history.append_step(s_{t+1})`. Las dos redes (online y target) son LSTMs
idénticas; la sincronización se hace cada `RDQN_TARGET_SYNC_FREQ` pasos
copiando todos los pesos (incluidos los del LSTM).

Beneficio adicional en RDQN: los LSTMs son notoriamente sensibles al
*bootstrapping* inestable; la combinación target-network + Double DQN
amortigua tanto el sesgo de maximización como las oscilaciones del
gradiente recurrente.

---

## 7. *Experience Replay* para datos secuenciales

El replay clásico (Lin, 1992; Mnih et al., 2015) muestrea **transiciones
puntuales** uniformemente. Para RDQN hay tres variantes posibles:

1. **Replay de secuencias completas**: almacenar episodios enteros y
   muestrear ventanas contiguas. Caro en memoria.
2. **Replay de ventanas fijas**: almacenar tuplas
   $(\tilde s_t, a_t, r_t, \tilde s_{t+1}, d_t)$ donde
   $\tilde s_t \in \mathbb{R}^{L \times 3}$. Cada empuje al replay es
   independiente. **Variante usada en `pes_rdqn`**.
3. **Bootstrapped Random Updates** (Hausknecht & Stone, 2015): muestrear
   subsegmentos aleatorios de un episodio y propagar el estado oculto
   del LSTM dentro del segmento.

La opción (2) preserva la decorrelación del replay clásico al precio de
"resetear" el LSTM cada *forward* — un compromiso aceptable dado que la
ventana $L = 6$ ya cubre la mayoría de las dependencias relevantes en el
Pandemic Scenario.

---

## 8. Convergencia y estabilidad de RL recurrente

Tres dificultades específicas del RL con LSTM y sus mitigaciones en este
proyecto:

1. **Gradientes ruidosos del LSTM**: la pérdida **Huber**
   ($L_\delta$, $\delta = 1$) acota la magnitud del gradiente para
   errores grandes (Mnih et al., 2015).
2. **Inestabilidad por blanco móvil**: la *target network* congelada
   reduce la varianza del objetivo (igual que en DQN).
3. **Sobreajuste a la ventana inicial de padding**: durante los primeros
   pasos $\tilde s$ contiene muchos ceros; el agente podría aprender una
   política trivial. Mitigado con un **warm-up** de
   $\varepsilon = 1$ que llena el replay con datos diversos antes de la
   primera actualización de gradiente.

Empíricamente la curva de aprendizaje de RDQN es **más lenta** que la de
DQN durante los primeros 5–10 k episodios pero **converge a un valor
ligeramente superior** a partir de los 30 k.

---

## 9. Anclaje al código

| Concepto teórico | Implementación |
|---|---|
| Ventana de historia $\tilde s_t$ | Clase `HistoryDeque` en `rdqn_model.py` |
| Red recurrente $Q_\theta(\tilde s, a)$ | `build_q_network()` (`Input → LSTM → Dense → Dense`) |
| Replay de ventanas | `ReplayBuffer` con tuplas `(L,3)`-shaped |
| Sincronización de target | `sync_target_network()` cada `RDQN_TARGET_SYNC_FREQ` |
| Objetivo Double DQN recurrente | `train_step_rdqn()` con `tf.GradientTape` |
| Pérdida Huber | `tf.keras.losses.Huber()` |
| ε-greedy con warm-up | `select_action(...)` en `train_rdqn.py` |
| TPE bayesiano sobre `history_len` | `optimize_rdqn.py` con `optuna.create_study` |

---

## Referencias

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A
next-generation hyperparameter optimization framework. En *Proceedings of
the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data
Mining* (pp. 2623–2631). ACM.

Hasselt, H. van, Guez, A., & Silver, D. (2016). Deep reinforcement
learning with double Q-learning. En *Proceedings of the Thirtieth AAAI
Conference on Artificial Intelligence* (pp. 2094–2100). AAAI Press.

Hausknecht, M., & Stone, P. (2015). Deep recurrent Q-learning for
partially observable MDPs. En *AAAI Fall Symposium on Sequential Decision
Making for Intelligent Agents*. AAAI.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural
Computation, 9*(8), 1735–1780.

Lin, L.-J. (1992). Self-improving reactive agents based on reinforcement
learning, planning and teaching. *Machine Learning, 8*(3–4), 293–321.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare,
M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G.,
Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran,
D., Wierstra, D., Legg, S., & Hassabis, D. (2015). Human-level control
through deep reinforcement learning. *Nature, 518*(7540), 529–533.

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An
introduction* (2nd ed.). MIT Press.
