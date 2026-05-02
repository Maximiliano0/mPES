<!-- markdownlint-disable MD051 -->
# Transformer Deep Q-Network (TRF) â€” TeorÃ­a, Arquitectura e ImplementaciÃ³n

> Documento tÃ©cnico del paquete **`pes_trf`** del proyecto mPES.
> Relaciona cada concepto teÃ³rico con el cÃ³digo fuente correspondiente
> y compara con los algoritmos anteriores (`pes_ql`, `pes_dql`, `pes_a2c`).

---

## Ãndice

1. [IntroducciÃ³n](#1-introducciÃ³n)
2. [Componentes del MDP](#2-componentes-del-mdp)
3. [Â¿Por quÃ© TRF en lugar de Q-Learning Tabular?](#3-por-quÃ©-trf-en-lugar-de-q-learning-tabular)
4. [Arquitectura de la Red Q](#4-arquitectura-de-la-red-q)
5. [TeorÃ­a del TRF](#5-teorÃ­a-del-trf)
6. [FunciÃ³n de PÃ©rdida â€” Huber Loss](#6-funciÃ³n-de-pÃ©rdida--huber-loss)
7. [ExploraciÃ³n â€” Îµ-Greedy Exponencial con Warm-up](#7-exploraciÃ³n--Îµ-greedy-exponencial-con-warm-up)
8. [Bucle de Entrenamiento](#8-bucle-de-entrenamiento)
9. [OptimizaciÃ³n Bayesiana de HiperparÃ¡metros](#9-optimizaciÃ³n-bayesiana-de-hiperparÃ¡metros)
10. [Inferencia en Tiempo de Experimento](#10-inferencia-en-tiempo-de-experimento)
11. [ComparaciÃ³n con DQL (Double Q-Learning)](#11-comparaciÃ³n-con-dql-double-q-learning)
12. [ComparaciÃ³n con A2C (Advantage Actor-Critic)](#12-comparaciÃ³n-con-a2c-advantage-actor-critic)
13. [ComparaciÃ³n General entre Algoritmos](#13-comparaciÃ³n-general-entre-algoritmos)
14. [Optimizaciones para CPU](#14-optimizaciones-para-cpu)
15. [Reproducibilidad Cross-Platform (Colab â†’ PC Local)](#15-reproducibilidad-cross-platform-colab--pc-local)

---

## 1. IntroducciÃ³n

> âš ï¸ **Diferencia clave con `pes_dqn` y `pes_rdqn`** â€” `pes_trf` es una
> variante **Transformer-DQN** que sustituye el MLP de `pes_dqn` y la
> LSTM de `pes_rdqn` por una pila de **bloques de codificador
> Transformer causal** (Vaswani et al., 2017, *Attention Is All You
> Need*).  La red consume una **ventana deslizante** de los Ãºltimos
> `TRF_HISTORY_LEN` estados normalizados,
> $(s_{t-T+1}, \dots, s_t) \in \mathbb{R}^{T \times 3}$, con padding
> cero a la izquierda al comienzo de cada secuencia y una capa
> `Masking(0.0)` que marca esos pasos como ausentes.  Una proyecciÃ³n
> lineal eleva cada token a `TRF_D_MODEL` dimensiones, se le suma un
> *positional embedding* aprendido, y `TRF_NUM_LAYERS` bloques (MHSA
> causal + FFN, cada uno con conexiÃ³n residual + LayerNorm) resumen la
> secuencia.  SÃ³lo el **Ãºltimo token** alimenta la cabeza Q.
>
> **HiperparÃ¡metros nuevos (todos en `config/CONFIG.py`):**
> - `TRF_HISTORY_LEN` (por defecto `6`) â€” longitud de la ventana.
> - `TRF_D_MODEL` (por defecto `32`) â€” ancho del *residual stream*.
> - `TRF_NUM_HEADS` (por defecto `4`) â€” cabezas de atenciÃ³n por bloque.
> - `TRF_KEY_DIM` (por defecto `16`) â€” dimensiÃ³n por cabeza.
> - `TRF_FF_DIM` (por defecto `64`) â€” ancho del FFN posiciÃ³n-a-posiciÃ³n.
> - `TRF_NUM_LAYERS` (por defecto `2`) â€” bloques apilados.
> - `TRF_DROPOUT` (por defecto `0.0`) â€” *dropout* en MHSA y FFN.
>
> Tanto el `ReplayBuffer` como el bucle de entrenamiento almacenan y
> muestrean **ventanas 2-D** $(T, d)$ en lugar de estados 1-D $(d,)$;
> ver `pes_trf/ext/transformer_model.py` y `TRFTraining` en
> `pes_trf/ext/pandemic.py`.  Durante la inferencia
> (`provide_trf_agent_response` en `pes_trf/src/pygameMediator.py`)
> un `HistoryDeque` cacheado por par `(session_no, sequence_no)`
> reconstruye la ventana ensayo a ensayo.

**Transformer Deep Q-Network (TRF)** (Mnih et al., 2015) es el primer algoritmo de
reinforcement learning profundo que logrÃ³ rendimiento humano en tareas de
alta dimensionalidad (Atari 2600).  Su idea central es reemplazar la tabla
Q finita del Q-Learning clÃ¡sico por una **red neuronal** que aproxima la
funciÃ³n $Q(s, a)$, permitiendo generalizaciÃ³n sobre estados similares.

En el contexto del proyecto **mPES**, `pes_trf` toma la misma dinÃ¡mica
pandÃ©mica que los paquetes anteriores (`pes_base`, `pes_ql`, `pes_dql`)
pero sustituye la tabla Q de dimensiÃ³n $31 \times 11 \times 10 \times 11
= 37\,510$ entradas por una red neuronal **basada en Transformer** que
combina una pila de bloques de codificador causal con una cabeza Q densa
(la configuraciÃ³n por defecto `TRF_NUM_LAYERS = 2`, `TRF_D_MODEL = 32`,
`TRF_NUM_HEADS = 4`, `TRF_FF_DIM = 64` y `TRF_HIDDEN_UNITS = [64]`
ronda los pocos miles de parÃ¡metros), ganando capacidad de
generalizaciÃ³n y memoria de trayectoria gracias a la atenciÃ³n sobre la
ventana de los Ãºltimos `TRF_HISTORY_LEN` estados.

### Archivos clave del paquete

| Archivo | DescripciÃ³n |
|---------|-------------|
| `config/CONFIG.py` | HiperparÃ¡metros TRF (red, replay, target sync, Îµ) |
| `ext/transformer_model.py` | Red Q, normalizaciÃ³n, ReplayBuffer, `train_step_trf` |
| `ext/pandemic.py` | Entorno Gymnasium, `TRFTraining`, `run_experiment` |
| `ext/train_transformer.py` | Pipeline de entrenamiento + evaluaciÃ³n |
| `ext/optimize_tr.py` | OptimizaciÃ³n Bayesiana (Optuna) |
| `__main__.py` | EjecuciÃ³n del experimento (8 bloques Ã— 8 secuencias) |
| `src/pygameMediator.py` | IntegraciÃ³n con UI Pygame |

---

## 2. Componentes del MDP

El entorno pandÃ©mico se modela como un **Proceso de DecisiÃ³n de Markov
(MDP)** idÃ©ntico al de los paquetes anteriores:

| Componente | DefiniciÃ³n | ImplementaciÃ³n |
|------------|-----------|----------------|
| **Estado** $s$ | $(r, t, v)$ â€” recursos restantes, nÂº de trial, severidad | `[available_resources, trial_no, severity]` |
| **Espacio de estados** | $31 \times 11 \times 10 = 3\,410$ estados posibles | `Pandemic.observation_shape` |
| **AcciÃ³n** $a$ | Recursos a asignar: $\{0, 1, \ldots, 10\}$ | `Pandemic.action_space = Discrete(11)` |
| **Recompensa** $r$ | $-\sum_{i} \text{severity}_i$ (penaliza severidades altas) | `reward = (-1) * numpy.sum(env.severities)` |
| **TransiciÃ³n** | $v' = \max(0,\; \beta \cdot v - \alpha \cdot a)$ | `get_updated_severity()` |
| **ParÃ¡metros** | $\alpha = 0.4,\; \beta = 1.4$ | `CONFIG.PANDEMIC_PARAMETER` |
| **Descuento** $\gamma$ | $0.9634$ (por defecto, mejor trial) | `CONFIG.TRF_DISCOUNT` |

### NormalizaciÃ³n del estado

A diferencia del Q-Learning tabular (que usa Ã­ndices enteros), TRF requiere
un vector numÃ©rico continuo.  La funciÃ³n `normalize_state()` escala cada
componente al rango $[0, 1]$:

$$s_{\text{norm}} = \left(\frac{r}{r_{\max}},\; \frac{t}{t_{\max}},\; \frac{v}{v_{\max}}\right)$$

```python
# ext/transformer_model.py
def normalize_state(state, max_resources, max_trials, max_severity):
    return numpy.array([
        state[0] / max(max_resources, 1),
        state[1] / max(max_trials, 1),
        state[2] / max(max_severity, 1),
    ], dtype=numpy.float32)
```

Valores por defecto: $r_{\max} = 30$, $t_{\max} = 10$, $v_{\max} = 9$.

---

## 3. Â¿Por quÃ© TRF en lugar de Q-Learning Tabular?

| Aspecto | Q-Learning Tabular | TRF |
|---------|--------------------|-----|
| **RepresentaciÃ³n** | Tabla de $|S| \times |A|$ entradas | Red neuronal parametrizada por $\theta$ |
| **GeneralizaciÃ³n** | Ninguna â€” cada estado es independiente | Comparte pesos â†’ generaliza sobre estados similares |
| **Escalabilidad** | Exponencial en dimensiÃ³n del estado | Lineal en parÃ¡metros de la red |
| **Memoria** | 37 510 escalares (Q-table) | pocos miles de parÃ¡metros (Transformer encoder + cabeza Q `[64]`) |
| **Estabilidad** | Convergencia garantizada (Robbins-Monro) | Requiere replay buffer + target network |
| **Complejidad** | MÃ­nima | Mayor (backpropagation, batching) |

En este MDP con solo 3 410 estados, la tabla Q es perfectamente viable.
Sin embargo, TRF ofrece:

- **GeneralizaciÃ³n**: estados cercanos (p.ej. severidad 3 vs 4) comparten
  informaciÃ³n a travÃ©s de los pesos de la red.
- **Extensibilidad**: si se ampliase el espacio de estados (mÃ¡s ciudades,
  mÃ¡s dimensiones), TRF escalarÃ­a sin cambios.
- **Comparabilidad**: permite evaluar empÃ­ricamente cuÃ¡nto aporta la
  aproximaciÃ³n funcional frente al mÃ©todo tabular.

---

## 4. Arquitectura de la Red Q

### 4.1 TopologÃ­a

A diferencia de `pes_dqn` (MLP de 1-3 capas) y `pes_rdqn` (LSTM), la red
Q de `pes_trf` consume una **ventana deslizante** de los Ãºltimos
`TRF_HISTORY_LEN` estados normalizados $(s_{t-T+1}, \dots, s_t)$ con
padding cero a la izquierda al comienzo de cada secuencia, una `Masking`
para descartar el prefijo, una proyecciÃ³n lineal a `TRF_D_MODEL`
dimensiones, un *positional embedding* aprendido y `TRF_NUM_LAYERS`
**bloques de codificador Transformer causal** (MHSA + FFN, residual +
LayerNorm).  SÃ³lo el **Ãºltimo token** alimenta una pequeÃ±a cabeza Q
densa (`TRF_HIDDEN_UNITS`) y la salida lineal de 11 Q-values:

```
Q-Network:  Input(history_len, 3)
              -> Masking(0.0)
              -> Dense(d_model)  (token embedding) + PosEmb
              -> [ MultiHeadAttention(num_heads, key_dim, causal)
                   -> Add & LayerNorm
                   -> FFN(ff_dim, ReLU) -> Dense(d_model)
                   -> Add & LayerNorm ] x num_layers
              -> last token
              -> Dense(64, ReLU)        # TRF_HIDDEN_UNITS = [64]
              -> Dense(11, linear)
```

```python
# ext/transformer_model.py
def build_q_network(state_dim, action_dim, hidden_units,
                    history_len=6, d_model=32, num_heads=4,
                    key_dim=16, ff_dim=64, num_layers=2,
                    dropout=0.0, seed=None):
    inputs = tf.keras.layers.Input(shape=(history_len, state_dim))
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.Dense(d_model, name="token_embed")(x)
    pos = tf.keras.layers.Embedding(history_len, d_model)(tf.range(history_len))
    x = x + pos
    for blk in range(num_layers):
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=dropout
        )(x, x, use_causal_mask=True)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ffn = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
        ffn = tf.keras.layers.Dense(d_model)(ffn)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    last_token = tf.keras.layers.Lambda(lambda t: t[:, -1, :])(x)
    h = last_token
    for units in hidden_units:
        h = tf.keras.layers.Dense(units, activation="relu")(h)
    outputs = tf.keras.layers.Dense(action_dim, activation="linear")(h)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
```

El parÃ¡metro `seed` siembra un `GlorotUniform(seed=seed+idx)` distinto en
cada capa, produciendo una inicializaciÃ³n totalmente determinista entre
llamadas a `TRFTraining` con el mismo `SEED`.

### 4.2 Entrada y salida

| Capa | Forma | DescripciÃ³n |
|------|-------|-------------|
| Input | $(T, 3)$ | Ventana de los Ãºltimos $T = $ `TRF_HISTORY_LEN` estados normalizados $(r, t, v)$, zero-padded al comienzo del episodio |
| Masking | $(T, 3)$ | Marca como ausentes los pasos con todos los componentes a 0 |
| Token embed | $(T, d_{\text{model}})$ | ProyecciÃ³n lineal de cada estado a la dimensiÃ³n del *residual stream* |
| + PosEmb | $(T, d_{\text{model}})$ | Embedding posicional aprendido sumado por token |
| Encoder $\times$ N | $(T, d_{\text{model}})$ | $N = $ `TRF_NUM_LAYERS` bloques (MHSA causal + FFN + residual + LN) |
| Last token | $(d_{\text{model}},)$ | Se descarta toda la secuencia salvo el paso $t$ |
| Hidden | $(64,)$ | Cabeza densa con `TRF_HIDDEN_UNITS = [64]` (ReLU) |
| Output | $(11,)$ | $Q(s, a)$ para cada $a \in \{0, \ldots, 10\}$ (activaciÃ³n lineal) |

La activaciÃ³n de salida es **lineal** porque los Q-values pueden ser
negativos (la recompensa es $-\sum \text{severities}$).

### 4.3 HiperparÃ¡metros de la red (defaults de `CONFIG.py`)

| Variable | Valor | DescripciÃ³n |
|----------|-------|-------------|
| `TRF_HISTORY_LEN` | $6$ | Longitud de la ventana deslizante |
| `TRF_D_MODEL` | $32$ | Ancho del *residual stream* (token embedding) |
| `TRF_NUM_HEADS` | $4$ | Cabezas de atenciÃ³n por bloque |
| `TRF_KEY_DIM` | $16$ | DimensiÃ³n por cabeza (query/key) |
| `TRF_FF_DIM` | $64$ | Ancho del FFN posiciÃ³n-a-posiciÃ³n |
| `TRF_NUM_LAYERS` | $2$ | Bloques de codificador apilados |
| `TRF_DROPOUT` | $0.0$ | Dropout dentro de MHSA / FFN |
| `TRF_HIDDEN_UNITS` | $[64]$ | Anchuras de la cabeza Q tras el codificador |

La bÃºsqueda bayesiana explora el espacio Transformer completo
(`history_len`, `d_model`, `num_heads`, `key_dim`, `ff_dim`,
`num_layers`, `dropout`) ademÃ¡s de las anchuras y profundidad de la
cabeza densa â€” ver [SecciÃ³n 9.1](#91-espacio-de-bÃºsqueda).

---

## 5. TeorÃ­a del TRF

### 5.1 EcuaciÃ³n de Bellman

El valor Ã³ptimo de acciÃ³n $Q^*(s, a)$ satisface la ecuaciÃ³n de Bellman:

$$Q^*(s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\right]$$

TRF aproxima $Q^*(s, a) \approx Q_\theta(s, a)$ con una red neuronal
parametrizada por $\theta$.

### 5.2 TD Targets, Target Network y Double TRF

El **target** para la actualizaciÃ³n usa el esquema **Double TRF**
(van Hasselt et al., 2016) con enmascaramiento de acciones infactibles:

$$a^{\*}_{i} = \arg\max_{a' \in \mathcal{F}(s'_i)} Q_\theta(s'_i, a')
\qquad
y_i = r_i + \gamma \, Q_{\theta^-}(s'_i, a^{\*}_i) \cdot (1 - d_i)$$

donde:

- $Q_\theta$ (online network) **selecciona** la acciÃ³n del bootstrap.
- $Q_{\theta^-}$ (target network) **evalÃºa** esa acciÃ³n.
- $\mathcal{F}(s'_i) = \{a : a \le r'_i\}$ es el conjunto de acciones
  factibles dado el nÃºmero de recursos restantes en $s'_i$ (los recursos
  son la primera componente del *Ãºltimo* token de la ventana
  normalizada, recuperada con
  `tf.round(next_states[:, -1, 0] * max_resources)`).
- $d_i \in \{0, 1\}$ es el indicador de episodio terminado.

Esta separaciÃ³n entre selecciÃ³n y evaluaciÃ³n elimina el sesgo de
**sobreestimaciÃ³n** del operador $\max$ del TRF vanilla, y la mÃ¡scara de
factibilidad evita que la red bootstrappee desde acciones que no podrÃ­an
haberse tomado.

```python
# ext/transformer_model.py â€” train_step_trf()  (Double TRF + feasibility mask)
next_q_online = online_net(next_states, training=False)            # selecciÃ³n
next_q_target = target_net(next_states, training=False)            # evaluaciÃ³n

# Reconstruir recursos restantes desde el Ãºltimo paso de la ventana normalizada.
# La ventana es 3-D: (batch, history_len, state_dim); el Ã­ndice -1 toma el
# token mÃ¡s reciente, y el Ã­ndice 0 sobre state_dim corresponde a `resources_left`.
resources_left = tf.round(next_states[:, -1, 0] * max_resources)    # (B,)
actions_idx    = tf.range(action_dim, dtype=tf.float32)             # (A,)
feasible       = actions_idx[None, :] <= resources_left[:, None]    # (B, A)
next_q_online_masked = tf.where(feasible, next_q_online, -1e9)

best_actions = tf.argmax(next_q_online_masked, axis=1, output_type=tf.int32)
max_next_q   = tf.gather(next_q_target, best_actions, batch_dims=1)

td_targets = rewards + discount * max_next_q * (1.0 - dones)
```

Sin la target network los targets cambiarÃ­an con cada actualizaciÃ³n de
$\theta$, creando un problema de **moving targets** que desestabiliza el
aprendizaje.

### 5.3 Experience Replay

El **replay buffer** almacena transiciones $(s, a, r, s', d)$ en un buffer
circular de capacidad fija y las muestrea uniformemente en mini-batches:

```python
# ext/transformer_model.py
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

**Â¿Por quÃ© es necesario?**

1. **Rompe la correlaciÃ³n temporal**: las muestras consecutivas de un
   episodio estÃ¡n altamente correlacionadas.  El sampling aleatorio
   diversifica los mini-batches.
2. **ReutilizaciÃ³n de datos**: cada transiciÃ³n se usa mÃºltiples veces,
   mejorando la eficiencia muestral frente al Q-Learning tabular que usa
   cada transiciÃ³n una sola vez.

ConfiguraciÃ³n por defecto: `TRF_REPLAY_BUFFER_SIZE = 20_000`.

### 5.4 SincronizaciÃ³n de la Target Network

La target network se actualiza mediante **hard sync** (copia completa de
pesos) cada `TRF_TARGET_SYNC_FREQ = 1_000` steps:

```python
# ext/transformer_model.py
def sync_target_network(online_net, target_net):
    target_net.set_weights(online_net.get_weights())
```

Alternativas no implementadas:

- **Soft update** (Polyak): $\theta^- \leftarrow \tau \theta + (1-\tau) \theta^-$
  â€” mÃ¡s suave pero mÃ¡s costoso por step.

---

## 6. FunciÃ³n de PÃ©rdida â€” Huber Loss

TRF utiliza **Huber loss** (tambiÃ©n llamada smooth L1 loss) en lugar de MSE:

$$\mathcal{L}_\delta(e) = \begin{cases}
\frac{1}{2} e^2 & \text{si } |e| \leq \delta \\
\delta \cdot (|e| - \frac{1}{2}\delta) & \text{si } |e| > \delta
\end{cases}$$

con $e = Q_\theta(s_i, a_i) - y_i$ (error TD).

La pÃ©rdida completa del mini-batch:

$$\mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} \text{Huber}\bigl(Q_\theta(s_i, a_i) - y_i\bigr)$$

```python
# ext/transformer_model.py â€” train_step_trf()
with tf.GradientTape() as tape:
    q_values = online_net(states, training=True)
    action_mask = tf.one_hot(actions, depth=tf.shape(q_values)[1])
    predicted_q = tf.reduce_sum(q_values * action_mask, axis=1)
    loss = tf.reduce_mean(tf.keras.losses.huber(td_targets, predicted_q))
```

**Â¿Por quÃ© Huber en lugar de MSE?**

- **MSE** ($e^2$) amplifica errores grandes, causando gradientes explosivos
  ante outliers en las recompensas.
- **Huber** es cuadrÃ¡tica cerca de 0 (buen gradiente para errores pequeÃ±os)
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

## 7. ExploraciÃ³n â€” Îµ-Greedy Exponencial con Warm-up

TRF usa una polÃ­tica Îµ-greedy con **warm-up + decaimiento exponencial**
(adaptado de `pes_a2c` / `pes_dql`), en dos fases:

1. **Fase 1 (warm-up):** durante los primeros $W = r_{\text{warm}} \cdot N$
   episodios, $\varepsilon$ se mantiene constante en $\varepsilon_0$
   (exploraciÃ³n pura).
2. **Fase 2 (decay exponencial):** $\varepsilon$ decae exponencialmente
   hasta alcanzar $\varepsilon_{\min}$ exactamente en el episodio
   $T = r_{\text{target}} \cdot N$:

$$\varepsilon_t = \begin{cases}
\varepsilon_0 & \text{si } t < W \\
\max\left(\varepsilon_{\min},\; \varepsilon_0 \cdot \lambda^{t - W}\right) & \text{si } t \geq W
\end{cases}$$

donde la tasa de decaimiento $\lambda$ se calcula automÃ¡ticamente:

$$\lambda = \left(\frac{\varepsilon_{\min}}{\varepsilon_0}\right)^{\frac{1}{(r_{\text{target}} - r_{\text{warm}}) \cdot N}}$$

```python
# ext/pandemic.py â€” TRFTraining()
epsilon_initial = epsilon
warmup_episodes = int(warmup_ratio * episodes)
resolved_decay_rate = (min_eps / max(epsilon, 1e-8)) ** (
    1.0 / max(1, int((target_ratio - warmup_ratio) * episodes))
)

for i in range(episodes):
    # ... training loop ...

    # Exponential Îµ-decay with warm-up
    if i < warmup_episodes:
        epsilon = epsilon_initial                # Phase 1: pure exploration
    else:
        epsilon = max(min_eps,                   # Phase 2: exponential decay
                      epsilon_initial * (resolved_decay_rate ** (i - warmup_episodes)))
```

| ParÃ¡metro | Valor por defecto | Variable |
|-----------|-------------------|----------|
| $\varepsilon_0$ | 0.963 | `TRF_EPSILON_INITIAL` |
| $\varepsilon_{\min}$ | 0.069 | `TRF_EPSILON_MIN` |
| $r_{\text{warm}}$ | 0.278 | `TRF_WARMUP_RATIO` |
| $r_{\text{target}}$ | 0.629 | `TRF_TARGET_RATIO` |
| $N$ | 175 000 | `TRF_EPISODES` |

Con los valores por defecto: warm-up de â‰ˆ48 600 episodios, luego decay
exponencial hasta el episodio â‰ˆ110 100, y $\varepsilon = \varepsilon_{\min}$
durante los â‰ˆ64 900 episodios finales.

### SelecciÃ³n de acciÃ³n con enmascaramiento

Durante la selecciÃ³n greedy, las **acciones infactibles** ($a > r$, donde
$r$ son los recursos restantes) se enmascaran estableciendo su Q-value a
$-10^9$ antes del $\arg\max$.  La rama aleatoria de $\varepsilon$-greedy
**tambiÃ©n** se restringe a las acciones factibles, y el sorteo se hace con
un `numpy.random.Generator` **dedicado** sembrado con `SEED`, de manera que
el RNG global de NumPy (que alimenta `env.random_sequence()`) no se vea
desplazado:

```python
# ext/pandemic.py â€” TRFTraining()  (paso de selecciÃ³n de acciÃ³n)
eps_rng = numpy.random.default_rng(seed)        # RNG dedicado para Îµ-greedy
feasible = numpy.arange(action_dim) <= state[0]
q_vals = online_net(norm_state[numpy.newaxis], training=False).numpy()[0]

if eps_rng.random() < epsilon:
    feasible_actions = numpy.flatnonzero(feasible)
    action = int(eps_rng.choice(feasible_actions))   # exploraciÃ³n enmascarada
else:
    q_masked = numpy.where(feasible, q_vals, -1e9)
    action = int(numpy.argmax(q_masked))             # explotaciÃ³n enmascarada
```

### ComparaciÃ³n con otros esquemas de decaimiento

| Esquema | FÃ³rmula | Paquete |
|---------|---------|---------|
| Exponencial con warm-up (TRF) | $\varepsilon_t = \max(\varepsilon_{\min},\; \varepsilon_0 \cdot \lambda^{t - W})$ | `pes_trf` |
| Exponencial con warm-up (DQL) | $\varepsilon_t = \max(\varepsilon_{\min},\; \varepsilon_0 \cdot \lambda^{t - W})$ | `pes_dql` |
| Exponencial con warm-up (A2C) | $\varepsilon_t = \max(\varepsilon_{\min},\; \varepsilon_0 \cdot \lambda^{t - W})$ | `pes_a2c` |

Los tres paquetes (`pes_dql`, `pes_trf`, `pes_a2c`) ahora comparten el
mismo esquema de dos fases: warm-up constante seguido de decay
exponencial.  La ventaja frente al decay lineal anterior es que la
exploraciÃ³n pura inicial llena el replay buffer con experiencias diversas
antes de empezar a explotar, y el decay exponencial permite una transiciÃ³n
mÃ¡s suave hacia la explotaciÃ³n.

---

## 8. Bucle de Entrenamiento

### 8.1 PseudocÃ³digo

```
Inicializar online_net Î¸, target_net Î¸â» â† Î¸
Inicializar replay_buffer (capacidad = 20 000)
global_step â† 0
warmup_episodes â† warmup_ratio Ã— N
Î» â† (Îµ_min / Îµâ‚€) ^ (1 / ((target_ratio âˆ’ warmup_ratio) Ã— N))

PARA episodio i = 1 hasta N:
    env.random_sequence()
    estado â† env.reset()

    MIENTRAS no terminado:
        s_norm â† normalize_state(estado)
        Î¦(s) â† âˆ’Î£ max(0, severidades)   // PBRS: potencial ANTES del step

        SI random() < Îµ:
            acciÃ³n â† random
        SINO:
            Q â† online_net(s_norm)
            enmascarar acciones infactibles
            acciÃ³n â† argmax(Q)

        estado', recompensa, terminado â† env.step(acciÃ³n)
        Î¦(s') â† âˆ’Î£ max(0, severidades')  si NO terminado, sino 0
        recompensa â† recompensa + Î²Â·(Î³Â·Î¦(s') âˆ’ Î¦(s))   // reward shaping
        s'_norm â† normalize_state(estado')

        replay_buffer.push(s_norm, acciÃ³n, recompensa, s'_norm, terminado)

        SI |replay_buffer| â‰¥ batch_size:
            batch â† replay_buffer.sample(batch_size)
            train_step_trf(online_net, target_net, optimizer, batch, Î³, max_grad_norm)

        global_step += 1

        SI global_step % target_sync_freq == 0:
            target_net.set_weights(online_net.get_weights())

        estado â† estado'

    // Exponential Îµ-decay with warm-up
    SI i < warmup_episodes:
        Îµ â† Îµâ‚€                           // Phase 1: pure exploration
    SINO:
        Îµ â† max(Îµ_min, Îµâ‚€ Â· Î»^(i âˆ’ warmup_episodes))  // Phase 2: exponential decay
```

### 8.2 ImplementaciÃ³n completa

La funciÃ³n `TRFTraining()` en `ext/pandemic.py` implementa el bucle
completo.  Sus componentes clave:

```python
# ext/pandemic.py â€” TRFTraining()
def TRFTraining(env, learning_rate, discount, epsilon, min_eps, episodes,
                hidden_units=None, batch_size=64, buffer_size=50_000,
                target_sync_freq=1_000, max_grad_norm=1.0, seed=None,
                penalty_coeff=0.0, compute_confidence=True,
                pruning_callback=None,
                warmup_ratio=0.05, target_ratio=0.60,
                learning_starts=None,
                history_len=6, d_model=32, num_heads=4, key_dim=16,
                ff_dim=64, num_layers=2, dropout=0.0):

    # 1. Construir redes online y target (inicializaciÃ³n sembrada)
    ...
```

> **Nota:** Tanto `train_transformer.py` como `optimize_tr.py` llaman a
> `TRFTraining()` con `compute_confidence=False` para evitar consumir
> nÃºmeros aleatorios de `numpy.random` que desplacen el RNG
> (ver [SecciÃ³n 14.2](#142-eliminaciÃ³n-del-forward-pass-de-confianza)).
    online_net = build_q_network(state_dim, action_dim, hidden_units, seed=seed)
    target_net = build_q_network(state_dim, action_dim, hidden_units, seed=seed)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    sync_target_network(online_net, target_net)   # Î¸â» â† Î¸

    # 2. Inicializar replay buffer
    replay_buffer = ReplayBuffer(buffer_size, seed=seed)

    # 3. Compilar train_step con tf.function (reduce_retracing minimiza recompilaciones)
    compiled_train_step = tf.function(train_step_trf, reduce_retracing=True)
    discount_t        = tf.constant(discount, dtype=tf.float32)
    max_grad_norm_t   = tf.constant(max_grad_norm, dtype=tf.float32)
    max_resources_t   = tf.constant(env.max_resources, dtype=tf.float32)

    # 4. Warm-up del replay buffer (Double TRF sin datos suficientes es ruido)
    if learning_starts is None:
        learning_starts = max(10 * batch_size, buffer_size // 10)
    # 5. Exponential Îµ-decay with warm-up
    eps_rng = numpy.random.default_rng(seed)      # RNG dedicado para Îµ-greedy
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
            # Îµ-greedy con enmascaramiento (rama aleatoria tambiÃ©n enmascarada)
            ...
            replay_buffer.push(norm_state, action, reward, norm_state2, done)

            # Actualizar sÃ³lo cuando el buffer ha calentado
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

        # Exponential Îµ-decay with warm-up
        if i < warmup_episodes:
            epsilon = epsilon_initial
        else:
            epsilon = max(min_eps,
                          epsilon_initial * (resolved_decay_rate ** (i - warmup_episodes)))

    return ave_reward_list, online_net, conf_list
```

### 8.3 HiperparÃ¡metros por defecto

Los valores actuales en `config/CONFIG.py` provienen del mejor trial de la
bÃºsqueda bayesiana (trial #41, 2026-04-23) y se usan tanto como defaults de
`train_transformer.py` como warm-start de la prÃ³xima optimizaciÃ³n:

| HiperparÃ¡metro | Valor | Variable (`CONFIG.py`) |
|----------------|-------|------------------------|
| Learning rate (Adam) | $0.001508$ | `TRF_LEARNING_RATE` |
| Discount $\gamma$ | $0.9634$ | `TRF_DISCOUNT` |
| $\varepsilon_0$ | $0.9627$ | `TRF_EPSILON_INITIAL` |
| $\varepsilon_{\min}$ | $0.0691$ | `TRF_EPSILON_MIN` |
| Warmup ratio $r_{\text{warm}}$ | $0.2779$ | `TRF_WARMUP_RATIO` |
| Target ratio $r_{\text{target}}$ | $0.6290$ | `TRF_TARGET_RATIO` |
| Episodios | $175\,000$ | `TRF_EPISODES` |
| Hidden units (cabeza Q) | $[64]$ | `TRF_HIDDEN_UNITS` |
| Batch size | $128$ | `TRF_BATCH_SIZE` |
| Replay buffer | $20\,000$ | `TRF_REPLAY_BUFFER_SIZE` |
| Target sync freq | $1\,000$ | `TRF_TARGET_SYNC_FREQ` |
| Gradient clipping | $3.953$ | `TRF_MAX_GRAD_NORM` |
| PBRS $\beta$ | $0.02258$ | `TRF_PENALTY_COEFF` |
| Learning starts (warm-up del buffer) | fracciÃ³n $0.1615$ del buffer | `TRF_LEARNING_STARTS_FRAC` |
| History length | $6$ | `TRF_HISTORY_LEN` |
| $d_{\text{model}}$ | $32$ | `TRF_D_MODEL` |
| Cabezas de atenciÃ³n | $4$ | `TRF_NUM_HEADS` |
| DimensiÃ³n por cabeza | $16$ | `TRF_KEY_DIM` |
| Ancho FFN | $64$ | `TRF_FF_DIM` |
| Bloques de codificador | $2$ | `TRF_NUM_LAYERS` |
| Dropout | $0.0$ | `TRF_DROPOUT` |
| Seed | $42$ | `SEED` |

---

## 9. OptimizaciÃ³n Bayesiana de HiperparÃ¡metros

`ext/optimize_tr.py` utiliza **Optuna** (TPE sampler) para buscar
hiperparÃ¡metros Ã³ptimos del TRF.

### 9.1 Espacio de BÃºsqueda

| ParÃ¡metro | Rango | Tipo |
|-----------|-------|------|
| `learning_rate` | $[10^{-4},\; 5 \cdot 10^{-3}]$ | log-uniforme |
| `discount_factor` | $[0.92,\; 0.995]$ | uniforme |
| `epsilon_initial` | $[0.80,\; 1.0]$ | uniforme |
| `epsilon_min` | $[0.01,\; 0.20]$ | uniforme |
| `num_episodes` | $[20\,000,\; 60\,000]$ | entero (paso 10k, sÃ³lo opt) |
| `hidden_layer_size` | $\{32, 64, 96, 128\}$ | categÃ³rico |
| `num_hidden_layers` | $\{1, 2, 3\}$ | entero |
| `batch_size` | $\{32, 64, 128, 256\}$ | categÃ³rico |
| `buffer_size` | $[20\,000,\; 100\,000]$ | entero (paso 10k) |
| `target_sync_freq` | $[500,\; 5\,000]$ | entero (paso 500) |
| `max_grad_norm` | $[0.5,\; 5.0]$ | uniforme |
| `use_pbrs` | $\{\text{True}, \text{False}\}$ | categÃ³rico |
| `penalty_coeff` | $[10^{-4},\; 0.1]$ | log-uniforme (sÃ³lo si `use_pbrs=True`) |
| `warmup_ratio` | $[0.05,\; 0.30]$ | uniforme (Îµ-warmup) |
| `target_ratio` | $[0.50,\; 0.95]$ | uniforme (Îµ-decay target) |
| `learning_starts_frac` | $[0.05,\; 0.25]$ | uniforme (warm-up del buffer) |
| `history_len` | $[3,\; 10]$ | entero (longitud de la ventana) |
| `d_model` | $\{16, 32, 64, 128\}$ | categÃ³rico |
| `num_heads` | $\{2, 4, 8\}$ | categÃ³rico |
| `key_dim` | $\{8, 16, 32\}$ | categÃ³rico |
| `ff_dim` | $\{32, 64, 128, 256\}$ | categÃ³rico |
| `num_layers` | $[1,\; 4]$ | entero (bloques de codificador) |
| `dropout` | $[0.0,\; 0.3]$ | uniforme (MHSA / FFN) |

```python
# ext/optimize_tr.py â€” objective()
learning_rate    = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
discount_factor  = trial.suggest_float('discount_factor', 0.92, 0.995)
epsilon_initial  = trial.suggest_float('epsilon_initial', 0.80, 1.0)
epsilon_min      = trial.suggest_float('epsilon_min', 0.01, 0.20)
num_episodes     = trial.suggest_int('num_episodes', 20_000, 60_000, step=10_000)
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
# Transformer-specific knobs
history_len      = trial.suggest_int('history_len', 3, 10)
d_model          = trial.suggest_categorical('d_model', [16, 32, 64, 128])
num_heads        = trial.suggest_categorical('num_heads', [2, 4, 8])
key_dim          = trial.suggest_categorical('key_dim', [8, 16, 32])
ff_dim           = trial.suggest_categorical('ff_dim', [32, 64, 128, 256])
num_layers       = trial.suggest_int('num_layers', 1, 4)
dropout          = trial.suggest_float('dropout', 0.0, 0.3)
```

**Diferencias frente al espacio de A2C:**

- TRF tiene **un Ãºnico learning rate** (para la Q-network) vs. dos en A2C
  (actor_lr + critic_lr).
- TRF incluye `batch_size`, `buffer_size` y `target_sync_freq` â€” conceptos
  exclusivos de off-policy con replay buffer.
- A2C incluye `entropy_coeff`, `gae_lambda`, `lr_min_ratio`
  â€” conceptos exclusivos del actor-critic.
- Ambos comparten `penalty_coeff` (PBRS), `max_grad_norm`,
  `warmup_ratio` y `target_ratio`.

### 9.2 FunciÃ³n Objetivo

Se entrena un agente TRF con los hiperparÃ¡metros sugeridos y se evalÃºa
en las **64 secuencias fijas** (las mismas que usa `__main__.py`).  El score
reportado a Optuna es el **rendimiento normalizado medio** (a maximizar),
calculado con `calculate_normalised_final_severity_performance_metric()`.
Se aplica enmascaramiento de acciones infactibles (`actions > resources_left`)
para que la mÃ©trica coincida con el comportamiento del agente en el
experimento.

```python
# ext/optimize_tr.py â€” objective()
def qf(_env, state, _seqid):
    norm_s = normalize_state(state, max_res, max_seq, max_sev)
    q_vals = model(norm_s[numpy.newaxis], training=False).numpy()[0].copy()
    response, _conf, _rt_h, _rt_r = trf_agent_meta_cognitive(
        q_vals, state[0], 10000
    )
    return response

_, perfs, _ = run_experiment(env_eval, qf, False, _trials_per_sequence, _sevs)
mean_perf = float(numpy.mean(perfs))
```

### 9.3 Persistencia del Mejor Modelo

A diferencia del re-entrenamiento al final (que serÃ­a lossy), se preservan
los **pesos del mejor modelo** encontrado durante la optimizaciÃ³n:

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
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=1, interval_steps=1),
)
```

El callback de poda se pasa a `TRFTraining` vÃ­a `pruning_callback`:

```python
def _pruning_cb(episode_idx, avg_reward):
    trial.report(avg_reward, step_counter)
    return trial.should_prune()
```

### 9.5 Warm-Start

La bÃºsqueda comienza con un **trial semilla** usando los valores de
`CONFIG.py`, asegurando que al menos un trial alcance un rendimiento
razonable y sirva de referencia para el pruner:

```python
if len(study.trials) == 0:
    study.enqueue_trial({
        'learning_rate': TRF_LEARNING_RATE,
        'discount_factor': TRF_DISCOUNT, ...
        'max_grad_norm': TRF_MAX_GRAD_NORM,
        'penalty_coeff': TRF_PENALTY_COEFF,
        'warmup_ratio': TRF_WARMUP_RATIO,
        'target_ratio': TRF_TARGET_RATIO,
    })
```

### 9.6 PBRS (Potential-Based Reward Shaping)

Siguiendo a Ng et al. (1999), se aÃ±ade una seÃ±al de reward shaping:

$$r' = r + \beta \cdot (\gamma \cdot \Phi(s') - \Phi(s))$$

donde $\Phi(s) = -\sum_i \max(0, S_i)$ y $\Phi(s_{\text{terminal}}) = 0$.

Esta forma **telescÃ³pica** garantiza que la polÃ­tica Ã³ptima es invariable
respecto a $\beta$.  El coeficiente $\beta$ (`penalty_coeff`) se optimiza
en $[10^{-4}, 0.1]$ (escala logarÃ­tmica) y se activa con la categÃ³rica
`use_pbrs âˆˆ {True, False}`: cuando `use_pbrs=False` el shaping se desactiva
(equivalente a $\beta = 0$), permitiendo a Optuna comparar directamente la
utilidad del shaping frente a su ausencia.

### 9.7 OptimizaciÃ³n de Velocidad

Durante la optimizaciÃ³n, se desactiva el cÃ¡lculo de confianza
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

Para reanudar una optimizaciÃ³n previa:

```
python -m pes_trf.ext.optimize_tr 100 --resume 2026-03-14
```

---

## 10. Inferencia en Tiempo de Experimento

### 10.1 Carga del Modelo

`src/pygameMediator.py` â†’ `provide_trf_agent_response()`:

```python
model = tf.keras.models.load_model(model_path)
```

Se carga la **Q-network completa** (un solo modelo). A diferencia de A2C
(que solo carga el Actor, descartando el Critic), en TRF toda la inferencia
se hace con la misma red que se entrenÃ³.

### 10.2 SelecciÃ³n de AcciÃ³n

```python
# src/pygameMediator.py â€” provide_trf_agent_response()
# Los denominadores se leen del propio entorno para que coincidan SIEMPRE
# con los usados durante el entrenamiento, eliminando el riesgo de drift
# si AVAILABLE_RESOURCES_PER_SEQUENCE cambiara en CONFIG.
from ..ext.pandemic import Pandemic, trf_agent_meta_cognitive
_env_for_norm = Pandemic()
max_res = _env_for_norm.max_resources
state = normalize_state([resources_val, city_val, sever_val],
                        max_res, NUM_MAX_TRIALS, MAX_SEVERITY)

# Mantener una ventana deslizante de los últimos TRF_HISTORY_LEN
# estados normalizados, cacheada por par (session_no, sequence_no).
cache_key = (int(session_no), int(sequence_no))
history = _history_cache.get(cache_key)
if history is None:
    history = HistoryDeque(TRF_HISTORY_LEN, 3)
    _history_cache[cache_key] = history
history.append_step(state)
window = history.current_window()                              # (T, 3)
state_tensor = window[numpy.newaxis]                           # (1, T, 3)
q_values = trf_model(state_tensor, training=False).numpy().flatten()  # (11,)

resp, confidence, rt_hold, rt_release = trf_agent_meta_cognitive(
    q_values, resources_left, RESPONSE_TIMEOUT)
resp = int(numpy.clip(resp, 0, int(resources_left)))
```

La funciÃ³n `trf_agent_meta_cognitive` aplica internamente el
enmascaramiento de acciones infactibles (Q-values con $a > r$ se ponen a
$-10^9$) antes del $\arg\max_a Q_\theta(s, a)$, y devuelve ademÃ¡s la
confianza basada en entropÃ­a y tiempos de respuesta simulados.

> **Nota de normalizaciÃ³n:** Los denominadores de `normalize_state` se
> leen directamente de `env.max_resources`, `env.max_seq_length` y
> `env.max_severity`, exactamente las mismas magnitudes usadas en
> `TRFTraining` durante el entrenamiento.  Con la configuraciÃ³n actual
> ello equivale a $r_{\max} = 30$ (recursos asignables tras la
> pre-asignaciÃ³n), $t_{\max} = 10$ y $v_{\max} = 9$.

### 10.3 Confianza Meta-Cognitiva

El vector de Q-values (11 valores) se transforma en una distribuciÃ³n de
pseudo-probabilidades para calcular la entropÃ­a:

$$H(Q) = -\sum_{a=0}^{10} p_a \log p_a$$

donde $p_a$ es la probabilidad derivada de los Q-values (vÃ­a softmax
implÃ­cita o normalizaciÃ³n en `entropy_from_pdf`).

$$\text{confidence} = \frac{H - H_{\max}}{H_{\min} - H_{\max}}$$

```python
# ext/pandemic.py â€” trf_agent_meta_cognitive()
dec_entropy = entropy_from_pdf(options)
M_entropy = entropy_from_pdf(M_entropy)   # entropÃ­a mÃ¡xima (distribuciÃ³n uniforme)
m_entropy = entropy_from_pdf(m_entropy)   # entropÃ­a mÃ­nima (distribuciÃ³n delta)
confidence = (1. / (m_entropy - M_entropy)) * (dec_entropy - M_entropy)
```

**Nota importante:** A diferencia de A2C donde $\pi_\theta(a|s)$ es una
distribuciÃ³n de probabilidad genuina, los Q-values de TRF **no son
probabilidades**.  La "confianza" aquÃ­ es una **heurÃ­stica** basada en la
dispersiÃ³n de los Q-values â€” no tiene la misma justificaciÃ³n teÃ³rica que
en A2C.

---

## 11. ComparaciÃ³n con DQL (Double Q-Learning)

`pes_dql` implementa **Double Q-Learning** (Hasselt, 2010) â€” un mÃ©todo
**tabular** que aborda el sesgo de **sobreestimaciÃ³n** del Q-Learning
estÃ¡ndar usando dos tablas Q independientes.

### 11.1 SobreestimaciÃ³n: el mismo problema, distintas soluciones

El operador $\max$ introduce un **sesgo positivo** en los TD targets:

$$\mathbb{E}[\max_a Q(s', a)] \geq \max_a \mathbb{E}[Q(s', a)]$$

| Algoritmo | Estrategia contra sobreestimaciÃ³n |
|-----------|-----------------------------------|
| **DQL** (pes_dql) | Dos Q-tables $Q_A$, $Q_B$. SelecciÃ³n con una, evaluaciÃ³n con la otra: $Q_A(s', \arg\max_{a'} Q_B(s', a'))$ |
| **TRF** (pes_trf) | **Double DQN** sobre la red Transformer: la online network selecciona la acciÃ³n bootstrap (con mÃ¡scara de factibilidad) y la target network la evalÃºa, eliminando el sesgo del operador $\max$ y dando targets estables |

### 11.2 Arquitectura comparada

| Componente | DQL (`pes_dql`) | TRF (`pes_trf`) |
|------------|-----------------|-----------------|
| Modelo | 2 tablas Q: $Q_A$, $Q_B$ âˆˆ $\mathbb{R}^{31 \times 11 \times 10 \times 11}$ | 2 redes: online $Q_\theta$, target $Q_{\theta^-}$ (encoder Transformer con cabeza densa `TRF_HIDDEN_UNITS = [64]`) |
| ActualizaciÃ³n | $Q_A(s,a) \leftarrow Q_A + \alpha[r + \gamma Q_B(s', \arg\max Q_A) - Q_A]$ | Huber loss + Adam entre $Q_\theta(s,a)$ y $r + \gamma \max Q_{\theta^-}(s', \cdot)$ |
| Datos por update | 1 transiciÃ³n â†’ 1 actualizaciÃ³n | Batch de replay buffer â†’ 1 paso de gradiente |
| ExploraciÃ³n (Îµ) | Exponencial con warm-up | Exponencial con warm-up |
| PBRS | âœ“ ($\Phi(s) = -\sum \max(0, S_i)$, $\beta = 0.1$) | âœ“ ($\Phi(s) = -\sum \max(0, S_i)$, $\beta$ optimizable) |
| Convergencia | Garantizada (bajo condiciones de Robbins-Monro) | Sin garantÃ­as teÃ³ricas (aprox. funcional no-lineal) |

### 11.3 Ventajas y desventajas relativas

**DQL es mejor si:**
- El espacio de estados es pequeÃ±o (como en este MDP: 3 410 estados).
- Se desea convergencia garantizada.
- Los recursos computacionales son limitados (no requiere GPU/backprop).

**TRF es mejor si:**
- Se necesita generalizaciÃ³n entre estados.
- Se planea escalar a espacios de estado mayores.
- Se quiere reutilizar datos eficientemente (replay buffer).

---

## 12. ComparaciÃ³n con A2C (Advantage Actor-Critic)

`pes_a2c` implementa **Advantage Actor-Critic** â€” un mÃ©todo
**on-policy** con dos redes separadas (Actor + Critic) y una polÃ­tica
explÃ­cita $\pi_\theta(a|s)$.

### 12.1 Diferencias fundamentales

| Aspecto | TRF (`pes_trf`) | A2C (`pes_a2c`) |
|---------|-----------------|-----------------|
| **Tipo de polÃ­tica** | ImplÃ­cita ($\arg\max Q$) | ExplÃ­cita ($\pi_\theta$, softmax) |
| **On/Off-policy** | Off-policy (replay buffer) | On-policy (batch de episodio) |
| **Modelos** | 1 Q-network (+ target) | Actor (491 params) + Critic (321 params) |
| **Params totales** | encoder Transformer + cabeza densa `[64]` (defaults de `CONFIG.py`) | 812 |
| **ActualizaciÃ³n** | TD targets + Huber loss | Policy gradient + MSE del Critic |
| **Replay buffer** | âœ“ (20 000 transiciones) | âœ— |
| **EntropÃ­a** | HeurÃ­stica (Q-values) | TeÃ³rica ($\pi_\theta$ es PDF) |
| **GAE(Î»)** | âœ— | âœ“ ($\lambda = 0.95$) |
| **Cosine LR** | âœ— | âœ“ |
| **PBRS** | âœ“ ($\Phi(s) = -\sum \max(0, S_i)$) | âœ“ ($\Phi(s) = -\sum \max(0, S_i)$) |
| **Îµ-decay** | Exponencial con warm-up | Exponencial con warm-up |

### 12.2 Eficiencia muestral y estabilidad

- **TRF** es mÃ¡s **eficiente en datos** porque reutiliza transiciones del
  replay buffer.  Cada transiciÃ³n se muestrea ~$\frac{\text{buffer\_size}}{\text{batch\_size}}$
  veces en promedio.
- **A2C** es mÃ¡s **eficiente en parÃ¡metros** (812 params totales) pero requiere
  que cada episodio genere datos frescos (on-policy), desperdiciando
  experiencia pasada.
- **A2C** tiene mayor estabilidad de entrenamiento gracias a GAE(Î»),
  normalizaciÃ³n de advantage, cosine LR y entropy bonus.

### 12.3 Calidad de la confianza

| Propiedad | TRF | A2C |
|-----------|-----|-----|
| Vector de entrada | $Q(s,\cdot)$ (11 Q-values) | $\pi_\theta(\cdot|s)$ (11 probabilidades) |
| Tipo | HeurÃ­stica | TeÃ³rica |
| InterpretaciÃ³n | DispersiÃ³n de Q-values | EntropÃ­a de la distribuciÃ³n de polÃ­tica |
| $\sum = 1$? | No | SÃ­ (softmax) |
| JustificaciÃ³n | Razonable pero ad-hoc | Fundamentada en teorÃ­a de la informaciÃ³n |

---

## 13. ComparaciÃ³n General entre Algoritmos

| Componente | `pes_base` (Q-tabular) | `pes_dql` (Double Q) | `pes_trf` (TRF) | `pes_a2c` (A2C) |
|------------|------------------------|----------------------|------------------|------------------|
| Modelo | `numpy.ndarray` (q.npy) | 2 Ã— Q-table (.npy) | Red Transformer encoder + cabeza densa `[64]` (.keras) | Actor 491 + Critic 321 params (.keras) |
| Update | $Q + \alpha[r + \gamma \max Q - Q]$ | Doble tabla: selecciÃ³n/evaluaciÃ³n separadas | Huber loss + replay | Policy gradient + MSE + entropÃ­a |
| Datos | 1 paso â†’ 1 update | 1 paso â†’ 1 update | Replay buffer â†’ mini-batch | Batch de episodio â†’ 1 update |
| PolÃ­tica | ImplÃ­cita ($\arg\max Q$) | ImplÃ­cita ($\arg\max (Q_A + Q_B)$) | ImplÃ­cita ($\arg\max Q_\theta$) | ExplÃ­cita ($\pi_\theta$) |
| Confianza | EntropÃ­a de Q (heurÃ­stica) | EntropÃ­a de Q (heurÃ­stica) | EntropÃ­a de Q (heurÃ­stica) | EntropÃ­a de $\pi$ (teÃ³rica) |
| On/Off-policy | â€” | â€” | Off-policy | On-policy |
| Target estable | â€” | 2Âª tabla Q | Target network | â€” (baseline Critic) |
| PBRS | âœ— | âœ“ | âœ“ | âœ“ |
| Îµ-decay | Lineal | Exp. con warm-up | Exp. con warm-up | Exp. con warm-up |
| Episodios tÃ­picos | 900 000 | 250 000 | 175 000 | 250 000 |

---

## 14. Optimizaciones para CPU

### 14.1 `tf.function` por Trial (JIT Compilado, `reduce_retracing=True`)

`train_step_trf` se envuelve con `tf.function` **localmente** dentro de
cada llamada a `TRFTraining`, creando un grafo JIT-compilado fresco por
trial de Optuna:

```python
compiled_train_step = tf.function(train_step_trf, reduce_retracing=True)
```

Esto elimina el overhead de eager mode (significativo dado que cada step
del replay buffer implica una pasada de forward + backward) y a la vez
evita conflictos de `tf.Variable` entre trials sucesivos.  El argumento
`reduce_retracing=True` minimiza recompilaciones cuando la forma o el
tipo de los argumentos varÃ­an ligeramente entre llamadas.

Los hiperparÃ¡metros escalares (`discount`, `max_grad_norm`,
`max_resources`) se convierten a `tf.constant` antes del loop para evitar
retrazado por cambio de valores.

AdemÃ¡s, al finalizar cada `objective()` se libera memoria explÃ­citamente:

```python
del online_net, target_net, optimizer, replay_buffer
tf.keras.backend.clear_session()
gc.collect()
```

Esto evita que el grafo y los `tf.Variable` del trial anterior
permanezcan vivos durante toda la bÃºsqueda bayesiana.

### 14.2 EliminaciÃ³n del Forward Pass de Confianza

El cÃ¡lculo de meta-cogniciÃ³n (`trf_agent_meta_cognitive`) durante
entrenamiento requiere un **forward pass adicional** de la online network
(separado del forward pass de selecciÃ³n de acciÃ³n).  Tanto `train_transformer.py`
como `optimize_tr.py` pasan `compute_confidence=False` a `TRFTraining()`
para desactivar este cÃ¡lculo, ahorrando ~33 % de tiempo de forward-pass.

> **Nota de reproducibilidad:**  `compute_confidence=True` (el default)
> consume nÃºmeros aleatorios adicionales de `numpy.random` en cada step
> (para `rt_hold` y `rt_release`), desplazando el estado del RNG global.
> Dado que el mismo RNG alimenta a `env.random_sequence()`, activar la
> confianza durante el entrenamiento produce secuencias de episodios
> distintas a las de la optimizaciÃ³n, y el modelo resultante difiere del
> encontrado por Optuna.  Por tanto, `compute_confidence=False` es
> **obligatorio** para reproducir los hiperparÃ¡metros de la optimizaciÃ³n
> bayesiana.

### 14.3 ConfiguraciÃ³n de Hilos TensorFlow

Al importar `ext/transformer_model.py` se fija un Ãºnico hilo intra/inter-op y se
activa el modo determinista para que cada step produzca exactamente la
misma salida ante la misma semilla:

```python
if not tf.config.list_physical_devices('GPU'):
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
```

Esto sacrifica un pequeÃ±o margen de throughput a cambio de
reproducibilidad bit-a-bit entre el entrenamiento y la bÃºsqueda
bayesiana, requisito esencial para que `train_transformer.py` reconstruya con
fidelidad el mejor modelo encontrado por Optuna.

---

## 15. Reproducibilidad Cross-Platform (Colab â†’ PC Local)

### 15.1 MotivaciÃ³n

La optimizaciÃ³n bayesiana de TRF es costosa: 60 trials Ã— 40kâ€“100k
episodios consumen â‰ˆ12â€“20 horas en CPU.  El flujo recomendado es
**optimizar en Colab Pro+ con GPU** (donde un trial dura minutos en
lugar de horas) y **reentrenar el ganador localmente en CPU** para
producir el modelo de inferencia final.

Las diferencias numÃ©ricas entre CPU y GPU â€” inherentes al orden de
reducciones en cuDNN â€” obligan a esta separaciÃ³n: la bÃºsqueda explora
el espacio de hiperparÃ¡metros (donde la GPU brilla por throughput) y el
reentrenamiento final fija un modelo determinista bit-a-bit
reproducible (donde CPU + `enable_op_determinism` brilla).

### 15.2 Switch `MPES_USE_GPU`

`pes_trf/__init__.py`, `ext/train_transformer.py` y `ext/optimize_tr.py`
consultan la variable de entorno `MPES_USE_GPU` **antes** de importar
submÃ³dulos de TensorFlow:

```python
# pes_trf/__init__.py
import os
if os.environ.get('MPES_USE_GPU', '0') != '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''        # pin CPU
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
```

El mismo cÃ³digo corre en CPU determinista o en GPU multi-hilo sÃ³lo
cambiando una variable de entorno; no hay branches especÃ­ficos por
dispositivo en el cÃ³digo de modelo.

### 15.3 Persistencia sin `pickle` (CWE-502)

`_save_best_artifacts()` serializa el mejor trial como **NPZ + JSON
sidecar** en `inputs/<DATE>_BAYESIAN_OPT/`:

| Archivo | Formato | Contenido |
|---------|---------|-----------|
| `_best_artifacts.npz` | NumPy compressed | Pesos de la Q-network del mejor trial |
| `_best_artifacts.json` | JSON UTF-8 | Metadatos internos (valor, hidden_units, seed, n_weights) |
| `best_params_<date>.json` | JSON UTF-8 | HiperparÃ¡metros, seed, fecha, `mean_perf` (sidecar pÃºblico) |
| `trf_best_<date>.keras` | Keras nativo | Modelo completo (arquitectura + pesos) |

Esto evita la deserializaciÃ³n insegura de `pickle` (CWE-502 â€” ejecuciÃ³n
arbitraria de cÃ³digo al cargar artefactos no confiables) sin perder
ninguna informaciÃ³n necesaria para reproducir el modelo.

### 15.4 Bloque â€œCONFIG.PY SNIPPETâ€ auto-generado

Al finalizar la optimizaciÃ³n, `_save_report()` (`optimize_tr.py`)
emite un bloque copy-paste-ready en `optimization_results_<date>.txt`:

```python
# CONFIG.PY SNIPPET (copy-paste into pes_trf/config/CONFIG.py)
TRF_LEARNING_RATE        = best.params['learning_rate']
TRF_DISCOUNT             = best.params['discount']
# ...
TRF_LEARNING_STARTS_FRAC = best.params.get('learning_starts_frac', 0.1)
TRF_EPISODES             = full_episodes   # NOT best.params['num_episodes']
```

La lÃ­nea crÃ­tica es la Ãºltima: durante la optimizaciÃ³n,
`num_episodes âˆˆ [40k, 100k]` para mantener el coste bajo, pero el
reentrenamiento final usa `TRF_EPISODES = 175 000` (el valor canon de
`CONFIG.py`) para no sacrificar calidad.  El snippet reescribe esa
longitud automÃ¡ticamente para que el usuario no copie un valor
truncado.

### 15.5 RecuperaciÃ³n local con `--from-best`

`train_transformer.py` acepta una fecha YYYY-MM-DD apuntando a un directorio
`_BAYESIAN_OPT/` y reconstruye el modelo en local:

```bash
python -m pes_trf.ext.train_transformer --from-best 2026-04-20
```

El flujo interno es:

1. `_load_best_trial(date)` lee `best_params_<date>.json` (no `.npz`).
2. Sobrescribe los hiperparÃ¡metros de `CONFIG.py` en memoria.
3. Llama a `TRFTraining()` con `TRF_EPISODES` completo y
   `compute_confidence=False`.
4. Guarda el modelo final en `inputs/trf_model.keras` (canon usado por
   `__main__.py`).

Si no se especifica `--from-best`, `train_transformer.py` busca
automÃ¡ticamente el directorio `_BAYESIAN_OPT/` mÃ¡s reciente y lo
offrece (con un aviso).  Esto evita que el usuario olvide pasar la
fecha y reentrene con valores obsoletos.
