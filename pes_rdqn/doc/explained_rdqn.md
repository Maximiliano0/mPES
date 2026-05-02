<!-- markdownlint-disable MD051 -->
# Recurrent Deep Q-Network (RDQN) — Teoría, Arquitectura e Implementación

> Documento técnico del paquete **`pes_rdqn`** del proyecto mPES.
> Relaciona cada concepto teórico con el código fuente correspondiente
> y compara con los algoritmos anteriores (`pes_ql`, `pes_dql`, `pes_a2c`).

---

## Índice

1. [Introducción](#1-introducción)
2. [Componentes del MDP](#2-componentes-del-mdp)
3. [¿Por qué RDQN en lugar de Q-Learning Tabular?](#3-por-qué-rdqn-en-lugar-de-q-learning-tabular)
4. [Arquitectura de la Red Q](#4-arquitectura-de-la-red-q)
5. [Teoría del RDQN](#5-teoría-del-rdqn)
6. [Función de Pérdida — Huber Loss](#6-función-de-pérdida--huber-loss)
7. [Exploración — ε-Greedy Exponencial con Warm-up](#7-exploración--ε-greedy-exponencial-con-warm-up)
8. [Bucle de Entrenamiento](#8-bucle-de-entrenamiento)
9. [Optimización Bayesiana de Hiperparámetros](#9-optimización-bayesiana-de-hiperparámetros)
10. [Inferencia en Tiempo de Experimento](#10-inferencia-en-tiempo-de-experimento)
11. [Comparación con DQL (Double Q-Learning)](#11-comparación-con-dql-double-q-learning)
12. [Comparación con A2C (Advantage Actor-Critic)](#12-comparación-con-a2c-advantage-actor-critic)
13. [Comparación General entre Algoritmos](#13-comparación-general-entre-algoritmos)
14. [Optimizaciones para CPU](#14-optimizaciones-para-cpu)
15. [Reproducibilidad Cross-Platform (Colab → PC Local)](#15-reproducibilidad-cross-platform-colab--pc-local)

---

## 1. Introducción

> ⚠️ **Diferencia clave con `pes_dqn`** — `pes_rdqn` es la variante
> **Recurrente** del DQN (Hausknecht & Stone, 2015, *Deep Recurrent
> Q-Learning for Partially Observable MDPs*).  En lugar de un MLP
> alimentado con un único estado normalizado $s_t \in \mathbb{R}^3$, el
> trunk de la red es una **LSTM** que consume una **ventana deslizante**
> de los últimos `RDQN_HISTORY_LEN` estados, $(s_{t-T+1}, \dots, s_t)$,
> con padding cero a la izquierda al comienzo de cada secuencia y una
> capa `Masking(0.0)` que evita que dichos pasos contribuyan al estado
> interno.  Esto convierte al MDP parcialmente observable (donde
> $s_t$ ignora la trayectoria de severidades de ensayos previos) en un
> POMDP tratable mediante memoria recurrente.
>
> **Hiperparámetros nuevos:**
> - `RDQN_HISTORY_LEN` (por defecto `6`) — longitud de la ventana.
> - `RDQN_LSTM_UNITS` (por defecto `64`) — ancho del estado oculto LSTM.
>
> Tanto el `ReplayBuffer` como el bucle de entrenamiento almacenan y
> muestrean **ventanas 2-D** $(T, d)$ en lugar de estados 1-D $(d,)$;
> ver `pes_rdqn/ext/rdqn_model.py` y `RDQNTraining` en
> `pes_rdqn/ext/pandemic.py`.  Durante la inferencia
> (`provide_rdqn_agent_response` en `pes_rdqn/src/pygameMediator.py`)
> un `HistoryDeque` cacheado por par `(session_no, sequence_no)`
> reconstruye la ventana ensayo a ensayo.

**Recurrent Deep Q-Network (RDQN)** (Mnih et al., 2015) es el primer algoritmo de
reinforcement learning profundo que logró rendimiento humano en tareas de
alta dimensionalidad (Atari 2600).  Su idea central es reemplazar la tabla
Q finita del Q-Learning clásico por una **red neuronal** que aproxima la
función $Q(s, a)$, permitiendo generalización sobre estados similares.

En el contexto del proyecto **mPES**, `pes_rdqn` toma la misma dinámica
pandémica que los paquetes anteriores (`pes_base`, `pes_ql`, `pes_dql`)
pero sustituye la tabla Q de dimensión $31 \times 11 \times 10 \times 11
= 37\,510$ entradas por una red neuronal con **≈ 5 131 parámetros**
(configuración por defecto `[64, 64]`, mejor trial #41 / 2026-04-23),
ganando capacidad de generalización y escalabilidad.

### Archivos clave del paquete

| Archivo | Descripción |
|---------|-------------|
| `config/CONFIG.py` | Hiperparámetros RDQN (red, replay, target sync, ε) |
| `ext/rdqn_model.py` | Red Q, normalización, ReplayBuffer, `train_step_rdqn` |
| `ext/pandemic.py` | Entorno Gymnasium, `RDQNTraining`, `run_experiment` |
| `ext/train_rdqn.py` | Pipeline de entrenamiento + evaluación |
| `ext/optimize_rdqn.py` | Optimización Bayesiana (Optuna) |
| `__main__.py` | Ejecución del experimento (8 bloques × 8 secuencias) |
| `src/pygameMediator.py` | Integración con UI Pygame |

---

## 2. Componentes del MDP

El entorno pandémico se modela como un **Proceso de Decisión de Markov
(MDP)** idéntico al de los paquetes anteriores:

| Componente | Definición | Implementación |
|------------|-----------|----------------|
| **Estado** $s$ | $(r, t, v)$ — recursos restantes, nº de trial, severidad | `[available_resources, trial_no, severity]` |
| **Espacio de estados** | $31 \times 11 \times 10 = 3\,410$ estados posibles | `Pandemic.observation_shape` |
| **Acción** $a$ | Recursos a asignar: $\{0, 1, \ldots, 10\}$ | `Pandemic.action_space = Discrete(11)` |
| **Recompensa** $r$ | $-\sum_{i} \text{severity}_i$ (penaliza severidades altas) | `reward = (-1) * numpy.sum(env.severities)` |
| **Transición** | $v' = \max(0,\; \beta \cdot v - \alpha \cdot a)$ | `get_updated_severity()` |
| **Parámetros** | $\alpha = 0.4,\; \beta = 1.4$ | `CONFIG.PANDEMIC_PARAMETER` |
| **Descuento** $\gamma$ | $0.9634$ (por defecto, mejor trial) | `CONFIG.RDQN_DISCOUNT` |

### Normalización del estado

A diferencia del Q-Learning tabular (que usa índices enteros), RDQN requiere
un vector numérico continuo.  La función `normalize_state()` escala cada
componente al rango $[0, 1]$:

$$s_{\text{norm}} = \left(\frac{r}{r_{\max}},\; \frac{t}{t_{\max}},\; \frac{v}{v_{\max}}\right)$$

```python
# ext/rdqn_model.py
def normalize_state(state, max_resources, max_trials, max_severity):
    return numpy.array([
        state[0] / max(max_resources, 1),
        state[1] / max(max_trials, 1),
        state[2] / max(max_severity, 1),
    ], dtype=numpy.float32)
```

Valores por defecto: $r_{\max} = 30$, $t_{\max} = 10$, $v_{\max} = 9$.

---

## 3. ¿Por qué RDQN en lugar de Q-Learning Tabular?

| Aspecto | Q-Learning Tabular | RDQN |
|---------|--------------------|-----|
| **Representación** | Tabla de $|S| \times |A|$ entradas | Red neuronal parametrizada por $\theta$ |
| **Generalización** | Ninguna — cada estado es independiente | Comparte pesos → generaliza sobre estados similares |
| **Escalabilidad** | Exponencial en dimensión del estado | Lineal en parámetros de la red |
| **Memoria** | 37 510 escalares (Q-table) | ~5 131 parámetros (red [64,64]) |
| **Estabilidad** | Convergencia garantizada (Robbins-Monro) | Requiere replay buffer + target network |
| **Complejidad** | Mínima | Mayor (backpropagation, batching) |

En este MDP con solo 3 410 estados, la tabla Q es perfectamente viable.
Sin embargo, RDQN ofrece:

- **Generalización**: estados cercanos (p.ej. severidad 3 vs 4) comparten
  información a través de los pesos de la red.
- **Extensibilidad**: si se ampliase el espacio de estados (más ciudades,
  más dimensiones), RDQN escalaría sin cambios.
- **Comparabilidad**: permite evaluar empíricamente cuánto aporta la
  aproximación funcional frente al método tabular.

---

## 4. Arquitectura de la Red Q

### 4.1 Topología

```
Q-Network:  Input(3) → Dense(64, ReLU) → Dense(64, ReLU) → Dense(11, linear)
```

```python
# ext/rdqn_model.py
def build_q_network(state_dim, action_dim, hidden_units, seed=None):
    model = tf.keras.Sequential(name="Q_Network")
    model.add(tf.keras.layers.Input(shape=(int(state_dim),)))
    for idx, units in enumerate(hidden_units):
        # Per-layer GlorotUniform with a deterministic seed offset
        init = (tf.keras.initializers.GlorotUniform(seed=int(seed) + idx)
                if seed is not None else 'glorot_uniform')
        model.add(tf.keras.layers.Dense(
            int(units), activation="relu",
            kernel_initializer=init, name=f"q_hidden_{idx}"))
    out_init = (tf.keras.initializers.GlorotUniform(seed=int(seed) + len(hidden_units))
                if seed is not None else 'glorot_uniform')
    model.add(tf.keras.layers.Dense(
        int(action_dim), activation="linear",
        kernel_initializer=out_init, name="q_values"))
    return model
```

El parámetro `seed` siembra un `GlorotUniform(seed=seed+idx)` distinto en
cada capa, produciendo una inicialización totalmente determinista entre
llamadas a `RDQNTraining` con el mismo `SEED`.

### 4.2 Entrada y salida

| Capa | Forma | Descripción |
|------|-------|-------------|
| Input | $(3,)$ | Estado normalizado $(r, t, v) \in [0,1]^3$ |
| Hidden 0 | $(64,)$ | $\text{ReLU}(W_0 \cdot s + b_0)$ |
| Hidden 1 | $(64,)$ | $\text{ReLU}(W_1 \cdot h_0 + b_1)$ |
| Output | $(11,)$ | $Q(s, a)$ para cada $a \in \{0, \ldots, 10\}$ (activación lineal) |

La activación de salida es **lineal** porque los Q-values pueden ser
negativos (la recompensa es $-\sum \text{severities}$).

### 4.3 Recuento de parámetros

Con la configuración por defecto `RDQN_HIDDEN_UNITS = [64, 64]`
(valor del mejor trial de la optimización bayesiana, trial #41 / 2026-04-23):

| Capa | Pesos | Biases | Total |
|------|-------|--------|-------|
| `q_hidden_0` | $3 \times 64 = 192$ | $64$ | $256$ |
| `q_hidden_1` | $64 \times 64 = 4\,096$ | $64$ | $4\,160$ |
| `q_values` | $64 \times 11 = 704$ | $11$ | $715$ |
| **Total** | | | **5 131** |

La búsqueda bayesiana explora hasta 3 capas ocultas con tamaños en
$\{32, 64, 96, 128\}$ — ver [Sección 9.1](#91-espacio-de-búsqueda).

---

## 5. Teoría del RDQN

### 5.1 Ecuación de Bellman

El valor óptimo de acción $Q^*(s, a)$ satisface la ecuación de Bellman:

$$Q^*(s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right]$$

RDQN aproxima $Q^*(s, a) \approx Q_\theta(s, a)$ con una red neuronal
parametrizada por $\theta$.

### 5.2 TD Targets, Target Network y Double RDQN

El **target** para la actualización usa el esquema **Double RDQN**
(van Hasselt et al., 2016) con enmascaramiento de acciones infactibles:

$$a^{\*}_{i} = \arg\max_{a' \in \mathcal{F}(s'_i)} Q_\theta(s'_i, a')
\qquad
y_i = r_i + \gamma \, Q_{\theta^-}(s'_i, a^{\*}_i) \cdot (1 - d_i)$$

donde:

- $Q_\theta$ (online network) **selecciona** la acción del bootstrap.
- $Q_{\theta^-}$ (target network) **evalúa** esa acción.
- $\mathcal{F}(s'_i) = \{a : a \le r'_i\}$ es el conjunto de acciones
  factibles dado el número de recursos restantes en $s'_i$ (los recursos
  son la primera componente del estado normalizado, recuperada con
  `tf.round(next_states[:, 0] * max_resources)`).
- $d_i \in \{0, 1\}$ es el indicador de episodio terminado.

Esta separación entre selección y evaluación elimina el sesgo de
**sobreestimación** del operador $\max$ del RDQN vanilla, y la máscara de
factibilidad evita que la red bootstrappee desde acciones que no podrían
haberse tomado.

```python
# ext/rdqn_model.py — train_step_rdqn()  (Double RDQN + feasibility mask)
next_q_online = online_net(next_states, training=False)            # selección
next_q_target = target_net(next_states, training=False)            # evaluación

# Reconstruir recursos restantes desde el estado normalizado
resources_left = tf.round(next_states[:, 0] * max_resources)        # (B,)
actions_idx    = tf.range(action_dim, dtype=tf.float32)             # (A,)
feasible       = actions_idx[None, :] <= resources_left[:, None]    # (B, A)
next_q_online_masked = tf.where(feasible, next_q_online, -1e9)

best_actions = tf.argmax(next_q_online_masked, axis=1, output_type=tf.int32)
max_next_q   = tf.gather(next_q_target, best_actions, batch_dims=1)

td_targets = rewards + discount * max_next_q * (1.0 - dones)
```

Sin la target network los targets cambiarían con cada actualización de
$\theta$, creando un problema de **moving targets** que desestabiliza el
aprendizaje.

### 5.3 Experience Replay

El **replay buffer** almacena transiciones $(s, a, r, s', d)$ en un buffer
circular de capacidad fija y las muestrea uniformemente en mini-batches:

```python
# ext/rdqn_model.py
class ReplayBuffer:
    def __init__(self, capacity, seed=None):
        self._buffer = deque(maxlen=capacity)
        self._rng = python_random.Random(seed)

    def push(self, state, action, reward, next_state, done):
        self._buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = self._rng.sample(list(self._buffer), batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (numpy.array(states, ...), ...)
```

**¿Por qué es necesario?**

1. **Rompe la correlación temporal**: las muestras consecutivas de un
   episodio están altamente correlacionadas.  El sampling aleatorio
   diversifica los mini-batches.
2. **Reutilización de datos**: cada transición se usa múltiples veces,
   mejorando la eficiencia muestral frente al Q-Learning tabular que usa
   cada transición una sola vez.

Configuración por defecto: `RDQN_REPLAY_BUFFER_SIZE = 20_000`.

### 5.4 Sincronización de la Target Network

La target network se actualiza mediante **hard sync** (copia completa de
pesos) cada `RDQN_TARGET_SYNC_FREQ = 1_000` steps:

```python
# ext/rdqn_model.py
def sync_target_network(online_net, target_net):
    target_net.set_weights(online_net.get_weights())
```

Alternativas no implementadas:

- **Soft update** (Polyak): $\theta^- \leftarrow \tau \theta + (1-\tau) \theta^-$
  — más suave pero más costoso por step.

---

## 6. Función de Pérdida — Huber Loss

RDQN utiliza **Huber loss** (también llamada smooth L1 loss) en lugar de MSE:

$$\mathcal{L}_\delta(e) = \begin{cases}
\frac{1}{2} e^2 & \text{si } |e| \leq \delta \\
\delta \cdot (|e| - \frac{1}{2}\delta) & \text{si } |e| > \delta
\end{cases}$$

con $e = Q_\theta(s_i, a_i) - y_i$ (error TD).

La pérdida completa del mini-batch:

$$\mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} \text{Huber}\bigl(Q_\theta(s_i, a_i) - y_i\bigr)$$

```python
# ext/rdqn_model.py — train_step_rdqn()
with tf.GradientTape() as tape:
    q_values = online_net(states, training=True)
    action_mask = tf.one_hot(actions, depth=tf.shape(q_values)[1])
    predicted_q = tf.reduce_sum(q_values * action_mask, axis=1)
    loss = tf.reduce_mean(tf.keras.losses.huber(td_targets, predicted_q))
```

**¿Por qué Huber en lugar de MSE?**

- **MSE** ($e^2$) amplifica errores grandes, causando gradientes explosivos
  ante outliers en las recompensas.
- **Huber** es cuadrática cerca de 0 (buen gradiente para errores pequeños)
  pero lineal para errores grandes (robusta a outliers).

### Gradient Clipping

Adicionalmente, se aplica **clipping de norma global** al gradiente:

$$\hat{g} = \frac{g}{\max(1,\; \|g\| / c)}$$

con $c = 1.0$ (por defecto):

```python
grads = tape.gradient(loss, online_net.trainable_variables)
grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
optimizer.apply_gradients(zip(grads, online_net.trainable_variables))
```

---

## 7. Exploración — ε-Greedy Exponencial con Warm-up

RDQN usa una política ε-greedy con **warm-up + decaimiento exponencial**
(adaptado de `pes_a2c` / `pes_dql`), en dos fases:

1. **Fase 1 (warm-up):** durante los primeros $W = r_{\text{warm}} \cdot N$
   episodios, $\varepsilon$ se mantiene constante en $\varepsilon_0$
   (exploración pura).
2. **Fase 2 (decay exponencial):** $\varepsilon$ decae exponencialmente
   hasta alcanzar $\varepsilon_{\min}$ exactamente en el episodio
   $T = r_{\text{target}} \cdot N$:

$$\varepsilon_t = \begin{cases}
\varepsilon_0 & \text{si } t < W \\
\max\left(\varepsilon_{\min},\; \varepsilon_0 \cdot \lambda^{t - W}\right) & \text{si } t \geq W
\end{cases}$$

donde la tasa de decaimiento $\lambda$ se calcula automáticamente:

$$\lambda = \left(\frac{\varepsilon_{\min}}{\varepsilon_0}\right)^{\frac{1}{(r_{\text{target}} - r_{\text{warm}}) \cdot N}}$$

```python
# ext/pandemic.py — RDQNTraining()
epsilon_initial = epsilon
warmup_episodes = int(warmup_ratio * episodes)
resolved_decay_rate = (min_eps / max(epsilon, 1e-8)) ** (
    1.0 / max(1, int((target_ratio - warmup_ratio) * episodes))
)

for i in range(episodes):
    # ... training loop ...

    # Exponential ε-decay with warm-up
    if i < warmup_episodes:
        epsilon = epsilon_initial                # Phase 1: pure exploration
    else:
        epsilon = max(min_eps,                   # Phase 2: exponential decay
                      epsilon_initial * (resolved_decay_rate ** (i - warmup_episodes)))
```

| Parámetro | Valor por defecto | Variable |
|-----------|-------------------|----------|
| $\varepsilon_0$ | 0.963 | `RDQN_EPSILON_INITIAL` |
| $\varepsilon_{\min}$ | 0.069 | `RDQN_EPSILON_MIN` |
| $r_{\text{warm}}$ | 0.278 | `RDQN_WARMUP_RATIO` |
| $r_{\text{target}}$ | 0.629 | `RDQN_TARGET_RATIO` |
| $N$ | 175 000 | `RDQN_EPISODES` |

Con los valores por defecto: warm-up de ≈48 600 episodios, luego decay
exponencial hasta el episodio ≈110 100, y $\varepsilon = \varepsilon_{\min}$
durante los ≈64 900 episodios finales.

### Selección de acción con enmascaramiento

Durante la selección greedy, las **acciones infactibles** ($a > r$, donde
$r$ son los recursos restantes) se enmascaran estableciendo su Q-value a
$-10^9$ antes del $\arg\max$.  La rama aleatoria de $\varepsilon$-greedy
**también** se restringe a las acciones factibles, y el sorteo se hace con
un `numpy.random.Generator` **dedicado** sembrado con `SEED`, de manera que
el RNG global de NumPy (que alimenta `env.random_sequence()`) no se vea
desplazado:

```python
# ext/pandemic.py — RDQNTraining()  (paso de selección de acción)
eps_rng = numpy.random.default_rng(seed)        # RNG dedicado para ε-greedy
feasible = numpy.arange(action_dim) <= state[0]
q_vals = online_net(norm_state[numpy.newaxis], training=False).numpy()[0]

if eps_rng.random() < epsilon:
    feasible_actions = numpy.flatnonzero(feasible)
    action = int(eps_rng.choice(feasible_actions))   # exploración enmascarada
else:
    q_masked = numpy.where(feasible, q_vals, -1e9)
    action = int(numpy.argmax(q_masked))             # explotación enmascarada
```

### Comparación con otros esquemas de decaimiento

| Esquema | Fórmula | Paquete |
|---------|---------|---------|
| Exponencial con warm-up (RDQN) | $\varepsilon_t = \max(\varepsilon_{\min},\; \varepsilon_0 \cdot \lambda^{t - W})$ | `pes_rdqn` |
| Exponencial con warm-up (DQL) | $\varepsilon_t = \max(\varepsilon_{\min},\; \varepsilon_0 \cdot \lambda^{t - W})$ | `pes_dql` |
| Exponencial con warm-up (A2C) | $\varepsilon_t = \max(\varepsilon_{\min},\; \varepsilon_0 \cdot \lambda^{t - W})$ | `pes_a2c` |

Los tres paquetes (`pes_dql`, `pes_rdqn`, `pes_a2c`) ahora comparten el
mismo esquema de dos fases: warm-up constante seguido de decay
exponencial.  La ventaja frente al decay lineal anterior es que la
exploración pura inicial llena el replay buffer con experiencias diversas
antes de empezar a explotar, y el decay exponencial permite una transición
más suave hacia la explotación.

---

## 8. Bucle de Entrenamiento

### 8.1 Pseudocódigo

```
Inicializar online_net θ, target_net θ⁻ ← θ
Inicializar replay_buffer (capacidad = 20 000)
global_step ← 0
warmup_episodes ← warmup_ratio × N
λ ← (ε_min / ε₀) ^ (1 / ((target_ratio − warmup_ratio) × N))

PARA episodio i = 1 hasta N:
    env.random_sequence()
    estado ← env.reset()

    MIENTRAS no terminado:
        s_norm ← normalize_state(estado)
        Φ(s) ← −Σ max(0, severidades)   // PBRS: potencial ANTES del step

        SI random() < ε:
            acción ← random
        SINO:
            Q ← online_net(s_norm)
            enmascarar acciones infactibles
            acción ← argmax(Q)

        estado', recompensa, terminado ← env.step(acción)
        Φ(s') ← −Σ max(0, severidades')  si NO terminado, sino 0
        recompensa ← recompensa + β·(γ·Φ(s') − Φ(s))   // reward shaping
        s'_norm ← normalize_state(estado')

        replay_buffer.push(s_norm, acción, recompensa, s'_norm, terminado)

        SI |replay_buffer| ≥ batch_size:
            batch ← replay_buffer.sample(batch_size)
            train_step_rdqn(online_net, target_net, optimizer, batch, γ, max_grad_norm)

        global_step += 1

        SI global_step % target_sync_freq == 0:
            target_net.set_weights(online_net.get_weights())

        estado ← estado'

    // Exponential ε-decay with warm-up
    SI i < warmup_episodes:
        ε ← ε₀                           // Phase 1: pure exploration
    SINO:
        ε ← max(ε_min, ε₀ · λ^(i − warmup_episodes))  // Phase 2: exponential decay
```

### 8.2 Implementación completa

La función `RDQNTraining()` en `ext/pandemic.py` implementa el bucle
completo.  Sus componentes clave:

```python
# ext/pandemic.py — RDQNTraining()
def RDQNTraining(env, learning_rate, discount, epsilon, min_eps, episodes,
                hidden_units=None, batch_size=64, buffer_size=50_000,
                target_sync_freq=1_000, max_grad_norm=1.0, seed=None,
                penalty_coeff=0.0, compute_confidence=True,
                pruning_callback=None,
                warmup_ratio=0.05, target_ratio=0.60,
                learning_starts=None):

    # 1. Construir redes online y target (inicialización sembrada)
    ...
```

> **Nota:** Tanto `train_rdqn.py` como `optimize_rdqn.py` llaman a
> `RDQNTraining()` con `compute_confidence=False` para evitar consumir
> números aleatorios de `numpy.random` que desplacen el RNG
> (ver [Sección 14.2](#142-eliminación-del-forward-pass-de-confianza)).
    online_net = build_q_network(state_dim, action_dim, hidden_units, seed=seed)
    target_net = build_q_network(state_dim, action_dim, hidden_units, seed=seed)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    sync_target_network(online_net, target_net)   # θ⁻ ← θ

    # 2. Inicializar replay buffer
    replay_buffer = ReplayBuffer(buffer_size, seed=seed)

    # 3. Compilar train_step con tf.function (reduce_retracing minimiza recompilaciones)
    compiled_train_step = tf.function(train_step_rdqn, reduce_retracing=True)
    discount_t        = tf.constant(discount, dtype=tf.float32)
    max_grad_norm_t   = tf.constant(max_grad_norm, dtype=tf.float32)
    max_resources_t   = tf.constant(env.max_resources, dtype=tf.float32)

    # 4. Warm-up del replay buffer (Double RDQN sin datos suficientes es ruido)
    if learning_starts is None:
        learning_starts = max(10 * batch_size, buffer_size // 10)
    # 5. Exponential ε-decay with warm-up
    eps_rng = numpy.random.default_rng(seed)      # RNG dedicado para ε-greedy
    epsilon_initial = epsilon
    warmup_episodes = int(warmup_ratio * episodes)
    resolved_decay_rate = (min_eps / max(epsilon, 1e-8)) ** (
        1.0 / max(1, int((target_ratio - warmup_ratio) * episodes))
    )
    global_step = 0

    for i in range(episodes):
        env.random_sequence()
        state, _ = env.reset()
        done = False

        while not done:
            # ε-greedy con enmascaramiento (rama aleatoria también enmascarada)
            ...
            replay_buffer.push(norm_state, action, reward, norm_state2, done)

            # Actualizar sólo cuando el buffer ha calentado
            if len(replay_buffer) >= learning_starts:
                s_b, a_b, r_b, ns_b, d_b = replay_buffer.sample(batch_size)
                compiled_train_step(online_net, target_net, optimizer,
                    tf.constant(s_b), tf.constant(a_b),
                    tf.constant(r_b), tf.constant(ns_b),
                    tf.constant(d_b),
                    discount_t, max_grad_norm_t, max_resources_t)

            global_step += 1
            if global_step % target_sync_freq == 0:
                sync_target_network(online_net, target_net)

        # Exponential ε-decay with warm-up
        if i < warmup_episodes:
            epsilon = epsilon_initial
        else:
            epsilon = max(min_eps,
                          epsilon_initial * (resolved_decay_rate ** (i - warmup_episodes)))

    return ave_reward_list, online_net, conf_list
```

### 8.3 Hiperparámetros por defecto

Los valores actuales en `config/CONFIG.py` provienen del mejor trial de la
búsqueda bayesiana (trial #41, 2026-04-23) y se usan tanto como defaults de
`train_rdqn.py` como warm-start de la próxima optimización:

| Hiperparámetro | Valor | Variable (`CONFIG.py`) |
|----------------|-------|------------------------|
| Learning rate (Adam) | $0.001508$ | `RDQN_LEARNING_RATE` |
| Discount $\gamma$ | $0.9634$ | `RDQN_DISCOUNT` |
| $\varepsilon_0$ | $0.9627$ | `RDQN_EPSILON_INITIAL` |
| $\varepsilon_{\min}$ | $0.0691$ | `RDQN_EPSILON_MIN` |
| Warmup ratio $r_{\text{warm}}$ | $0.2779$ | `RDQN_WARMUP_RATIO` |
| Target ratio $r_{\text{target}}$ | $0.6290$ | `RDQN_TARGET_RATIO` |
| Episodios | $175\,000$ | `RDQN_EPISODES` |
| Hidden units | $[64,\; 64]$ | `RDQN_HIDDEN_UNITS` |
| Batch size | $128$ | `RDQN_BATCH_SIZE` |
| Replay buffer | $20\,000$ | `RDQN_REPLAY_BUFFER_SIZE` |
| Target sync freq | $1\,000$ | `RDQN_TARGET_SYNC_FREQ` |
| Gradient clipping | $3.953$ | `RDQN_MAX_GRAD_NORM` |
| PBRS $\beta$ | $0.02258$ | `RDQN_PENALTY_COEFF` |
| Learning starts (warm-up del buffer) | fracción $0.1615$ del buffer | `RDQN_LEARNING_STARTS_FRAC` |
| Seed | $42$ | `SEED` |

---

## 9. Optimización Bayesiana de Hiperparámetros

`ext/optimize_rdqn.py` utiliza **Optuna** (TPE sampler) para buscar
hiperparámetros óptimos del RDQN.

### 9.1 Espacio de Búsqueda

| Parámetro | Rango | Tipo |
|-----------|-------|------|
| `learning_rate` | $[10^{-4},\; 5 \cdot 10^{-3}]$ | log-uniforme |
| `discount_factor` | $[0.92,\; 0.995]$ | uniforme |
| `epsilon_initial` | $[0.80,\; 1.0]$ | uniforme |
| `epsilon_min` | $[0.01,\; 0.20]$ | uniforme |
| `num_episodes` | $[40\,000,\; 100\,000]$ | entero (paso 20k, sólo opt) |
| `hidden_layer_size` | $\{32, 64, 96, 128\}$ | categórico |
| `num_hidden_layers` | $\{1, 2, 3\}$ | entero |
| `batch_size` | $\{32, 64, 128, 256\}$ | categórico |
| `buffer_size` | $[20\,000,\; 100\,000]$ | entero (paso 10k) |
| `target_sync_freq` | $[500,\; 5\,000]$ | entero (paso 500) |
| `max_grad_norm` | $[0.5,\; 5.0]$ | uniforme |
| `use_pbrs` | $\{\text{True}, \text{False}\}$ | categórico |
| `penalty_coeff` | $[10^{-4},\; 0.1]$ | log-uniforme (sólo si `use_pbrs=True`) |
| `warmup_ratio` | $[0.05,\; 0.30]$ | uniforme (ε-warmup) |
| `target_ratio` | $[0.50,\; 0.95]$ | uniforme (ε-decay target) |
| `learning_starts_frac` | $[0.05,\; 0.25]$ | uniforme (warm-up del buffer) |

```python
# ext/optimize_rdqn.py — objective()
learning_rate    = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
discount_factor  = trial.suggest_float('discount_factor', 0.92, 0.995)
epsilon_initial  = trial.suggest_float('epsilon_initial', 0.80, 1.0)
epsilon_min      = trial.suggest_float('epsilon_min', 0.01, 0.20)
num_episodes     = trial.suggest_int('num_episodes', 40_000, 100_000, step=20_000)
hidden_layer_size = trial.suggest_categorical('hidden_layer_size', [32, 64, 96, 128])
num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 3)
batch_size       = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
buffer_size      = trial.suggest_int('buffer_size', 20_000, 100_000, step=10_000)
target_sync_freq = trial.suggest_int('target_sync_freq', 500, 5_000, step=500)
max_grad_norm    = trial.suggest_float('max_grad_norm', 0.5, 5.0)
use_pbrs         = trial.suggest_categorical('use_pbrs', [True, False])
penalty_coeff    = (trial.suggest_float('penalty_coeff', 1e-4, 0.1, log=True)
                    if use_pbrs else 0.0)
warmup_ratio     = trial.suggest_float('warmup_ratio', 0.05, 0.30)
target_ratio     = trial.suggest_float('target_ratio', 0.50, 0.95)
learning_starts_frac = trial.suggest_float('learning_starts_frac', 0.05, 0.25)
```

**Diferencias frente al espacio de A2C:**

- RDQN tiene **un único learning rate** (para la Q-network) vs. dos en A2C
  (actor_lr + critic_lr).
- RDQN incluye `batch_size`, `buffer_size` y `target_sync_freq` — conceptos
  exclusivos de off-policy con replay buffer.
- A2C incluye `entropy_coeff`, `gae_lambda`, `lr_min_ratio`
  — conceptos exclusivos del actor-critic.
- Ambos comparten `penalty_coeff` (PBRS), `max_grad_norm`,
  `warmup_ratio` y `target_ratio`.

### 9.2 Función Objetivo

Se entrena un agente RDQN con los hiperparámetros sugeridos y se evalúa
en las **64 secuencias fijas** (las mismas que usa `__main__.py`).  El score
reportado a Optuna es el **rendimiento normalizado medio** (a maximizar),
calculado con `calculate_normalised_final_severity_performance_metric()`.
Se aplica enmascaramiento de acciones infactibles (`actions > resources_left`)
para que la métrica coincida con el comportamiento del agente en el
experimento.

```python
# ext/optimize_rdqn.py — objective()
def qf(_env, state, _seqid):
    norm_s = normalize_state(state, max_res, max_seq, max_sev)
    q_vals = model(norm_s[numpy.newaxis], training=False).numpy()[0].copy()
    response, _conf, _rt_h, _rt_r = rdqn_agent_meta_cognitive(
        q_vals, state[0], 10000
    )
    return response

_, perfs, _ = run_experiment(env_eval, qf, False, _trials_per_sequence, _sevs)
mean_perf = float(numpy.mean(perfs))
```

### 9.3 Persistencia del Mejor Modelo

A diferencia del re-entrenamiento al final (que sería lossy), se preservan
los **pesos del mejor modelo** encontrado durante la optimización:

```python
if mean_perf > _best_artifacts['value']:
    _best_artifacts['weights'] = model.get_weights()
    _best_artifacts['rewards'] = list(rewards)
    _best_artifacts['value'] = mean_perf
    _best_artifacts['hidden_units'] = hidden_units
```

Esto se serializa a disco como **NPZ + JSON sidecar** (no pickle, CWE-502)
para que `--resume` pueda recuperarlo sin reentrenar.

### 9.4 Poda Temprana (MedianPruner)

El estudio incorpora un **MedianPruner** que descarta trials cuyo reward
intermedio (reportado cada 10 000 episodios) es inferior a la mediana de
trials previos.  Esto ahorra un ~40-60 % del tiempo total al no completar
trials prometedoramente malos:

```python
study = optuna.create_study(
    ...
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
)
```

El callback de poda se pasa a `RDQNTraining` vía `pruning_callback`:

```python
def _pruning_cb(episode_idx, avg_reward):
    trial.report(avg_reward, step_counter)
    return trial.should_prune()
```

### 9.5 Warm-Start

La búsqueda comienza con un **trial semilla** usando los valores de
`CONFIG.py`, asegurando que al menos un trial alcance un rendimiento
razonable y sirva de referencia para el pruner:

```python
if len(study.trials) == 0:
    study.enqueue_trial({
        'learning_rate': RDQN_LEARNING_RATE,
        'discount_factor': RDQN_DISCOUNT, ...
        'max_grad_norm': RDQN_MAX_GRAD_NORM,
        'penalty_coeff': RDQN_PENALTY_COEFF,
        'warmup_ratio': RDQN_WARMUP_RATIO,
        'target_ratio': RDQN_TARGET_RATIO,
    })
```

### 9.6 PBRS (Potential-Based Reward Shaping)

Siguiendo a Ng et al. (1999), se añade una señal de reward shaping:

$$r' = r + \beta \cdot (\gamma \cdot \Phi(s') - \Phi(s))$$

donde $\Phi(s) = -\sum_i \max(0, S_i)$ y $\Phi(s_{\text{terminal}}) = 0$.

Esta forma **telescópica** garantiza que la política óptima es invariable
respecto a $\beta$.  El coeficiente $\beta$ (`penalty_coeff`) se optimiza
en $[10^{-4}, 0.1]$ (escala logarítmica) y se activa con la categórica
`use_pbrs ∈ {True, False}`: cuando `use_pbrs=False` el shaping se desactiva
(equivalente a $\beta = 0$), permitiendo a Optuna comparar directamente la
utilidad del shaping frente a su ausencia.

### 9.7 Optimización de Velocidad

Durante la optimización, se desactiva el cálculo de confianza
meta-cognitiva (`compute_confidence=False`), ahorrando ~33 % de tiempo
de forward-pass por step.

### 9.8 SQLite + Resume

El estudio de Optuna se almacena en una base de datos SQLite:

```python
db_path = os.path.join(opt_dir, f'optuna_study_{opt_date}.db')
storage = f'sqlite:///{db_path}'
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    storage=storage,
    load_if_exists=True,
)
```

Para reanudar una optimización previa:

```
python -m pes_rdqn.ext.optimize_rdqn 100 --resume 2026-03-14
```

---

## 10. Inferencia en Tiempo de Experimento

### 10.1 Carga del Modelo

`src/pygameMediator.py` → `provide_rdqn_agent_response()`:

```python
model = tf.keras.models.load_model(model_path)
```

Se carga la **Q-network completa** (un solo modelo). A diferencia de A2C
(que solo carga el Actor, descartando el Critic), en RDQN toda la inferencia
se hace con la misma red que se entrenó.

### 10.2 Selección de Acción

```python
# src/pygameMediator.py — provide_rdqn_agent_response()
# Los denominadores se leen del propio entorno para que coincidan SIEMPRE
# con los usados durante el entrenamiento, eliminando el riesgo de drift
# si AVAILABLE_RESOURCES_PER_SEQUENCE cambiara en CONFIG.
from ..ext.pandemic import Pandemic, rdqn_agent_meta_cognitive
_env_for_norm = Pandemic()
max_res = _env_for_norm.max_resources
state = normalize_state([resources_val, city_val, sever_val],
                        max_res, NUM_MAX_TRIALS, MAX_SEVERITY)
q_values = rdqn_model(tf.expand_dims(state, axis=0),
                     training=False).numpy().flatten()       # shape (11,)
resp, confidence, rt_hold, rt_release = rdqn_agent_meta_cognitive(
    q_values, resources_left, RESPONSE_TIMEOUT)
resp = int(numpy.clip(resp, 0, int(resources_left)))
```

La función `rdqn_agent_meta_cognitive` aplica internamente el
enmascaramiento de acciones infactibles (Q-values con $a > r$ se ponen a
$-10^9$) antes del $\arg\max_a Q_\theta(s, a)$, y devuelve además la
confianza basada en entropía y tiempos de respuesta simulados.

> **Nota de normalización:** Los denominadores de `normalize_state` se
> leen directamente de `env.max_resources`, `env.max_seq_length` y
> `env.max_severity`, exactamente las mismas magnitudes usadas en
> `RDQNTraining` durante el entrenamiento.  Con la configuración actual
> ello equivale a $r_{\max} = 30$ (recursos asignables tras la
> pre-asignación), $t_{\max} = 10$ y $v_{\max} = 9$.

### 10.3 Confianza Meta-Cognitiva

El vector de Q-values (11 valores) se transforma en una distribución de
pseudo-probabilidades para calcular la entropía:

$$H(Q) = -\sum_{a=0}^{10} p_a \log p_a$$

donde $p_a$ es la probabilidad derivada de los Q-values (vía softmax
implícita o normalización en `entropy_from_pdf`).

$$\text{confidence} = \frac{H - H_{\max}}{H_{\min} - H_{\max}}$$

```python
# ext/pandemic.py — rdqn_agent_meta_cognitive()
dec_entropy = entropy_from_pdf(options)
M_entropy = entropy_from_pdf(M_entropy)   # entropía máxima (distribución uniforme)
m_entropy = entropy_from_pdf(m_entropy)   # entropía mínima (distribución delta)
confidence = (1. / (m_entropy - M_entropy)) * (dec_entropy - M_entropy)
```

**Nota importante:** A diferencia de A2C donde $\pi_\theta(a|s)$ es una
distribución de probabilidad genuina, los Q-values de RDQN **no son
probabilidades**.  La "confianza" aquí es una **heurística** basada en la
dispersión de los Q-values — no tiene la misma justificación teórica que
en A2C.

---

## 11. Comparación con DQL (Double Q-Learning)

`pes_dql` implementa **Double Q-Learning** (Hasselt, 2010) — un método
**tabular** que aborda el sesgo de **sobreestimación** del Q-Learning
estándar usando dos tablas Q independientes.

### 11.1 Sobreestimación: el mismo problema, distintas soluciones

El operador $\max$ introduce un **sesgo positivo** en los TD targets:

$$\mathbb{E}[\max_a Q(s', a)] \geq \max_a \mathbb{E}[Q(s', a)]$$

| Algoritmo | Estrategia contra sobreestimación |
|-----------|-----------------------------------|
| **DQL** (pes_dql) | Dos Q-tables $Q_A$, $Q_B$. Selección con una, evaluación con la otra: $Q_A(s', \arg\max_{a'} Q_B(s', a'))$ |
| **RDQN** (pes_rdqn) | Target network $Q_{\theta^-}$ proporciona targets estables (no elimina el sesgo, pero lo reduce al congelar $\theta^-$) |

### 11.2 Arquitectura comparada

| Componente | DQL (`pes_dql`) | RDQN (`pes_rdqn`) |
|------------|-----------------|-----------------|
| Modelo | 2 tablas Q: $Q_A$, $Q_B$ ∈ $\mathbb{R}^{31 \times 11 \times 10 \times 11}$ | 2 redes: online $Q_\theta$, target $Q_{\theta^-}$ (≈5 131 params cada una con `[64,64]`) |
| Actualización | $Q_A(s,a) \leftarrow Q_A + \alpha[r + \gamma Q_B(s', \arg\max Q_A) - Q_A]$ | Huber loss + Adam entre $Q_\theta(s,a)$ y $r + \gamma \max Q_{\theta^-}(s', \cdot)$ |
| Datos por update | 1 transición → 1 actualización | Batch de replay buffer → 1 paso de gradiente |
| Exploración (ε) | Exponencial con warm-up | Exponencial con warm-up |
| PBRS | ✓ ($\Phi(s) = -\sum \max(0, S_i)$, $\beta = 0.1$) | ✓ ($\Phi(s) = -\sum \max(0, S_i)$, $\beta$ optimizable) |
| Convergencia | Garantizada (bajo condiciones de Robbins-Monro) | Sin garantías teóricas (aprox. funcional no-lineal) |

### 11.3 Ventajas y desventajas relativas

**DQL es mejor si:**
- El espacio de estados es pequeño (como en este MDP: 3 410 estados).
- Se desea convergencia garantizada.
- Los recursos computacionales son limitados (no requiere GPU/backprop).

**RDQN es mejor si:**
- Se necesita generalización entre estados.
- Se planea escalar a espacios de estado mayores.
- Se quiere reutilizar datos eficientemente (replay buffer).

---

## 12. Comparación con A2C (Advantage Actor-Critic)

`pes_a2c` implementa **Advantage Actor-Critic** — un método
**on-policy** con dos redes separadas (Actor + Critic) y una política
explícita $\pi_\theta(a|s)$.

### 12.1 Diferencias fundamentales

| Aspecto | RDQN (`pes_rdqn`) | A2C (`pes_a2c`) |
|---------|-----------------|-----------------|
| **Tipo de política** | Implícita ($\arg\max Q$) | Explícita ($\pi_\theta$, softmax) |
| **On/Off-policy** | Off-policy (replay buffer) | On-policy (batch de episodio) |
| **Modelos** | 1 Q-network (+ target) | Actor (491 params) + Critic (321 params) |
| **Params totales** | ≈5 131 (default `[64,64]`) | 812 |
| **Actualización** | TD targets + Huber loss | Policy gradient + MSE del Critic |
| **Replay buffer** | ✓ (20 000 transiciones) | ✗ |
| **Entropía** | Heurística (Q-values) | Teórica ($\pi_\theta$ es PDF) |
| **GAE(λ)** | ✗ | ✓ ($\lambda = 0.95$) |
| **Cosine LR** | ✗ | ✓ |
| **PBRS** | ✓ ($\Phi(s) = -\sum \max(0, S_i)$) | ✓ ($\Phi(s) = -\sum \max(0, S_i)$) |
| **ε-decay** | Exponencial con warm-up | Exponencial con warm-up |

### 12.2 Eficiencia muestral y estabilidad

- **RDQN** es más **eficiente en datos** porque reutiliza transiciones del
  replay buffer.  Cada transición se muestrea ~$\frac{\text{buffer\_size}}{\text{batch\_size}}$
  veces en promedio.
- **A2C** es más **eficiente en parámetros** (812 vs 5 131) pero requiere
  que cada episodio genere datos frescos (on-policy), desperdiciando
  experiencia pasada.
- **A2C** tiene mayor estabilidad de entrenamiento gracias a GAE(λ),
  normalización de advantage, cosine LR y entropy bonus.

### 12.3 Calidad de la confianza

| Propiedad | RDQN | A2C |
|-----------|-----|-----|
| Vector de entrada | $Q(s,\cdot)$ (11 Q-values) | $\pi_\theta(\cdot|s)$ (11 probabilidades) |
| Tipo | Heurística | Teórica |
| Interpretación | Dispersión de Q-values | Entropía de la distribución de política |
| $\sum = 1$? | No | Sí (softmax) |
| Justificación | Razonable pero ad-hoc | Fundamentada en teoría de la información |

---

## 13. Comparación General entre Algoritmos

| Componente | `pes_base` (Q-tabular) | `pes_dql` (Double Q) | `pes_rdqn` (RDQN) | `pes_a2c` (A2C) |
|------------|------------------------|----------------------|------------------|------------------|
| Modelo | `numpy.ndarray` (q.npy) | 2 × Q-table (.npy) | Red ≈5 131 params (.keras, `[64,64]`) | Actor 491 + Critic 321 params (.keras) |
| Update | $Q + \alpha[r + \gamma \max Q - Q]$ | Doble tabla: selección/evaluación separadas | Huber loss + replay | Policy gradient + MSE + entropía |
| Datos | 1 paso → 1 update | 1 paso → 1 update | Replay buffer → mini-batch | Batch de episodio → 1 update |
| Política | Implícita ($\arg\max Q$) | Implícita ($\arg\max (Q_A + Q_B)$) | Implícita ($\arg\max Q_\theta$) | Explícita ($\pi_\theta$) |
| Confianza | Entropía de Q (heurística) | Entropía de Q (heurística) | Entropía de Q (heurística) | Entropía de $\pi$ (teórica) |
| On/Off-policy | — | — | Off-policy | On-policy |
| Target estable | — | 2ª tabla Q | Target network | — (baseline Critic) |
| PBRS | ✗ | ✓ | ✓ | ✓ |
| ε-decay | Lineal | Exp. con warm-up | Exp. con warm-up | Exp. con warm-up |
| Episodios típicos | 900 000 | 250 000 | 175 000 | 250 000 |

---

## 14. Optimizaciones para CPU

### 14.1 `tf.function` por Trial (JIT Compilado, `reduce_retracing=True`)

`train_step_rdqn` se envuelve con `tf.function` **localmente** dentro de
cada llamada a `RDQNTraining`, creando un grafo JIT-compilado fresco por
trial de Optuna:

```python
compiled_train_step = tf.function(train_step_rdqn, reduce_retracing=True)
```

Esto elimina el overhead de eager mode (significativo dado que cada step
del replay buffer implica una pasada de forward + backward) y a la vez
evita conflictos de `tf.Variable` entre trials sucesivos.  El argumento
`reduce_retracing=True` minimiza recompilaciones cuando la forma o el
tipo de los argumentos varían ligeramente entre llamadas.

Los hiperparámetros escalares (`discount`, `max_grad_norm`,
`max_resources`) se convierten a `tf.constant` antes del loop para evitar
retrazado por cambio de valores.

Además, al finalizar cada `objective()` se libera memoria explícitamente:

```python
del online_net, target_net, optimizer, replay_buffer
tf.keras.backend.clear_session()
gc.collect()
```

Esto evita que el grafo y los `tf.Variable` del trial anterior
permanezcan vivos durante toda la búsqueda bayesiana.

### 14.2 Eliminación del Forward Pass de Confianza

El cálculo de meta-cognición (`rdqn_agent_meta_cognitive`) durante
entrenamiento requiere un **forward pass adicional** de la online network
(separado del forward pass de selección de acción).  Tanto `train_rdqn.py`
como `optimize_rdqn.py` pasan `compute_confidence=False` a `RDQNTraining()`
para desactivar este cálculo, ahorrando ~33 % de tiempo de forward-pass.

> **Nota de reproducibilidad:**  `compute_confidence=True` (el default)
> consume números aleatorios adicionales de `numpy.random` en cada step
> (para `rt_hold` y `rt_release`), desplazando el estado del RNG global.
> Dado que el mismo RNG alimenta a `env.random_sequence()`, activar la
> confianza durante el entrenamiento produce secuencias de episodios
> distintas a las de la optimización, y el modelo resultante difiere del
> encontrado por Optuna.  Por tanto, `compute_confidence=False` es
> **obligatorio** para reproducir los hiperparámetros de la optimización
> bayesiana.

### 14.3 Configuración de Hilos TensorFlow

Al importar `ext/rdqn_model.py` se fija un único hilo intra/inter-op y se
activa el modo determinista para que cada step produzca exactamente la
misma salida ante la misma semilla:

```python
if not tf.config.list_physical_devices('GPU'):
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
```

Esto sacrifica un pequeño margen de throughput a cambio de
reproducibilidad bit-a-bit entre el entrenamiento y la búsqueda
bayesiana, requisito esencial para que `train_rdqn.py` reconstruya con
fidelidad el mejor modelo encontrado por Optuna.

---

## 15. Reproducibilidad Cross-Platform (Colab → PC Local)

### 15.1 Motivación

La optimización bayesiana de RDQN es costosa: 60 trials × 40k–100k
episodios consumen ≈12–20 horas en CPU.  El flujo recomendado es
**optimizar en Colab Pro+ con GPU** (donde un trial dura minutos en
lugar de horas) y **reentrenar el ganador localmente en CPU** para
producir el modelo de inferencia final.

Las diferencias numéricas entre CPU y GPU — inherentes al orden de
reducciones en cuDNN — obligan a esta separación: la búsqueda explora
el espacio de hiperparámetros (donde la GPU brilla por throughput) y el
reentrenamiento final fija un modelo determinista bit-a-bit
reproducible (donde CPU + `enable_op_determinism` brilla).

### 15.2 Switch `MPES_USE_GPU`

`pes_rdqn/__init__.py`, `ext/train_rdqn.py` y `ext/optimize_rdqn.py`
consultan la variable de entorno `MPES_USE_GPU` **antes** de importar
submódulos de TensorFlow:

```python
# pes_rdqn/__init__.py
import os
if os.environ.get('MPES_USE_GPU', '0') != '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''        # pin CPU
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
```

El mismo código corre en CPU determinista o en GPU multi-hilo sólo
cambiando una variable de entorno; no hay branches específicos por
dispositivo en el código de modelo.

### 15.3 Persistencia sin `pickle` (CWE-502)

`_save_best_artifacts()` serializa el mejor trial como **NPZ + JSON
sidecar** en `inputs/<DATE>_BAYESIAN_OPT/`:

| Archivo | Formato | Contenido |
|---------|---------|-----------|
| `_best_artifacts.npz` | NumPy compressed | Pesos de la Q-network del mejor trial |
| `_best_artifacts.json` | JSON UTF-8 | Metadatos internos (valor, hidden_units, seed, n_weights) |
| `best_params_<date>.json` | JSON UTF-8 | Hiperparámetros, seed, fecha, `mean_perf` (sidecar público) |
| `rdqn_best_<date>.keras` | Keras nativo | Modelo completo (arquitectura + pesos) |

Esto evita la deserialización insegura de `pickle` (CWE-502 — ejecución
arbitraria de código al cargar artefactos no confiables) sin perder
ninguna información necesaria para reproducir el modelo.

### 15.4 Bloque “CONFIG.PY SNIPPET” auto-generado

Al finalizar la optimización, `_save_report()` (`optimize_rdqn.py`)
emite un bloque copy-paste-ready en `optimization_results_<date>.txt`:

```python
# CONFIG.PY SNIPPET (copy-paste into pes_rdqn/config/CONFIG.py)
RDQN_LEARNING_RATE        = best.params['learning_rate']
RDQN_DISCOUNT             = best.params['discount']
# ...
RDQN_LEARNING_STARTS_FRAC = best.params.get('learning_starts_frac', 0.1)
RDQN_EPISODES             = full_episodes   # NOT best.params['num_episodes']
```

La línea crítica es la última: durante la optimización,
`num_episodes ∈ [40k, 100k]` para mantener el coste bajo, pero el
reentrenamiento final usa `RDQN_EPISODES = 175 000` (el valor canon de
`CONFIG.py`) para no sacrificar calidad.  El snippet reescribe esa
longitud automáticamente para que el usuario no copie un valor
truncado.

### 15.5 Recuperación local con `--from-best`

`train_rdqn.py` acepta una fecha YYYY-MM-DD apuntando a un directorio
`_BAYESIAN_OPT/` y reconstruye el modelo en local:

```bash
python -m pes_rdqn.ext.train_rdqn --from-best 2026-04-20
```

El flujo interno es:

1. `_load_best_trial(date)` lee `best_params_<date>.json` (no `.npz`).
2. Sobrescribe los hiperparámetros de `CONFIG.py` en memoria.
3. Llama a `RDQNTraining()` con `RDQN_EPISODES` completo y
   `compute_confidence=False`.
4. Guarda el modelo final en `inputs/rdqn_model.keras` (canon usado por
   `__main__.py`).

Si no se especifica `--from-best`, `train_rdqn.py` busca
automáticamente el directorio `_BAYESIAN_OPT/` más reciente y lo
offrece (con un aviso).  Esto evita que el usuario olvide pasar la
fecha y reentrene con valores obsoletos.
