# `pes_trf` — Guía de uso e implementación

> **Algoritmo**: Causal Transformer encoder + Double DQN
> **Tipo**: DQN con memoria por atención sobre historia de trials
> **Última actualización**: 2026-05-02

---

## 1. ¿Qué es el Transformer DQN?

`pes_trf` reemplaza la red densa de un DQN clásico por un **encoder
Transformer con atención causal** (Vaswani et al., 2017) que procesa una
ventana deslizante de los últimos `history_len` estados normalizados del
escenario *Pandemic*. La salida del encoder alimenta una cabeza Q-lineal que
estima $Q(s_{t-h:t}, a)$ para las 11 acciones posibles.

### ¿Por qué Transformers para RL secuencial?

En el escenario *Pandemic*, la decisión óptima en el trial $t$ depende del
**patrón de severidades** de los trials previos (no solo del estado actual).
Las arquitecturas con memoria son superiores a las redes sin memoria:

| Arquitectura | Mecanismo de memoria | Limitación |
|---|---|---|
| **MLP (DQN)** | Ninguno | Solo estado actual $s_t$ |
| **LSTM (RDQN)** | Estado oculto recurrente | Cuello de botella secuencial |
| **Transformer** | Atención sobre toda la ventana | Atención paralela y posicional |

El Transformer tiene dos ventajas clave sobre el LSTM:

1. **Atención global**: cada posición de la ventana puede atender directamente
   a cualquier otra posición, sin información comprimida en un único vector
   oculto.
2. **Paralelismo**: el cómputo se vectoriza sobre toda la ventana, no se
   procesa paso a paso.

Empíricamente, en *Pandemic*, el Transformer alcanza
**0.927 de rendimiento normalizado** (n=64), superando a RDQN (0.899)
y al resto de agentes individuales.

---

## 2. Comandos de uso

### Ejecutar el experimento completo

```powershell
.\win_mpes_env\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"
$env:TF_ENABLE_ONEDNN_OPTS = "0"
python -m ml.pes_trf
```

### Entrenar el modelo

```powershell
python -m ml.pes_trf.ext.train_transformer            # reproduce el modelo desplegado (TRF_EPISODES = 30 000)
```

Salida principal: `ml/pes_trf/inputs/trf_model.keras`.

### Optimización Bayesiana

```powershell
python -m ml.pes_trf.ext.optimize_tr                  # trials por defecto
python -m ml.pes_trf.ext.optimize_tr 100              # 100 trials Optuna
```

---

## 3. Mecanismo de atención causal

### 3.1. Atención dot-product escalada

Dada una ventana de $h$ vectores de dimensión $d$, el bloque de atención
calcula:

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

donde $Q, K, V \in \mathbb{R}^{h \times d_k}$ son las proyecciones lineales
del input. Cada fila de la salida es una **mezcla ponderada** de los $V_i$,
con pesos dados por la similitud entre la query y todas las claves.

### 3.2. Máscara causal

Para que el agente en el trial $t$ no atienda a estados **futuros** (que no
existen aún en el momento de la decisión), se aplica una **máscara causal**
triangular antes del softmax:

$$\mathrm{mask}_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

$$\mathrm{Attention}_\mathrm{causal}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + \mathrm{mask}\right) V$$

Esto garantiza que la representación de la posición $i$ solo dependa de
posiciones $\{0, 1, \dots, i\}$.

### 3.3. Multi-head attention

Se ejecutan $H$ atenciones independientes en paralelo (cada una con sus
propias proyecciones $W_Q^h, W_K^h, W_V^h$) y se concatenan:

$$\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_H) W^O$$

En `pes_trf`: $H = 4$ cabezas (`TRF_NUM_HEADS`), $d_\mathrm{model} = 32$, elegidos
mediante exploraciones *ad hoc* (§5).

---

## 4. Pipeline de entrenamiento (`train_transformer.py`)

1. **Carga de configuración**: lee `config/CONFIG.py` y, si existe,
   `inputs/best_params.json`.
2. **Construcción del entorno**: `PandemicEnv` desde `ext/pandemic.py`.
3. **Construcción del modelo Transformer**: `build_q_network()` en
   `ext/transformer_model.py`.
4. **Construcción del target network**: copia con pesos sincronizados.
5. **Inicialización del replay buffer**: deque de tuplas
   $(\mathrm{history}_t, a_t, r_t, \mathrm{history}_{t+1}, d_t)$.
6. **Bucle de episodios**:
   - Reset del entorno y de la `HistoryDeque`.
   - Por cada trial:
     1. Empuja el estado actual a la `HistoryDeque`.
     2. Inferencia: $Q(\mathrm{history}, \cdot) \to a$ con
        $\varepsilon$-greedy y máscara de factibilidad.
     3. Step en el entorno: $(s', r, d)$.
     4. Almacena la transición en el buffer.
     5. Muestrea minibatch del buffer y ejecuta `train_step_trf()`.
   - Cada `TRF_TARGET_SYNC_FREQ` pasos: sincroniza el target network.
   - Guarda checkpoint cada N episodios.
7. **Logging dual-stream**: consola + `outputs/PES_TRF_log_<fecha>.txt`.

### `train_step_trf()` — un paso de gradiente

```python
def train_step_trf(model, target_model, optimizer, batch, gamma):
    states, actions, rewards, next_states, dones = batch

    # 1. Acción óptima del modelo en línea (Double DQN)
    next_q_online = model(next_states)                 # (B, 11)
    next_q_online = mask_infeasible(next_q_online, ...)
    a_star = tf.argmax(next_q_online, axis=1)

    # 2. Valor según target network
    next_q_target = target_model(next_states)          # (B, 11)
    q_target_a_star = gather(next_q_target, a_star)

    # 3. Target Bellman
    td_target = rewards + gamma * q_target_a_star * (1 - dones)

    # 4. Pérdida Huber sobre la acción tomada
    with tf.GradientTape() as tape:
        q_pred = model(states)
        q_pred_a = gather(q_pred, actions)
        loss = huber(td_target, q_pred_a)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

---

## 5. Optimización Bayesiana (`optimize_tr.py`)

`ext/optimize_tr.py` usa **Optuna** con muestreador TPE sobre los
hiperparámetros de entrenamiento. La arquitectura no se busca: se eligió
mediante exploraciones *ad hoc* y está fijada en `config/CONFIG.py`, porque
optimizarla con la búsqueda bayesiana era demasiado costoso para los recursos
de cómputo disponibles.

| Parámetro | Rango de la búsqueda | Modelo desplegado |
|---|---|---|
| `history_len` | — (no se busca) | 6 (`TRF_HISTORY_LEN`, *ad hoc*) |
| `d_model` | — (no se busca) | 32 (`TRF_D_MODEL`, *ad hoc*) |
| `num_heads` | — (no se busca) | 4 (`TRF_NUM_HEADS`, *ad hoc*) |
| `key_dim` | — (no se busca) | 16 (`TRF_KEY_DIM`, *ad hoc*) |
| `ff_dim` | — (no se busca) | 64 (`TRF_FF_DIM`, *ad hoc*) |
| `num_layers` | — (no se busca) | 2 (`TRF_NUM_LAYERS`, *ad hoc*) |
| `dropout` | — (no se busca) | 0 (`TRF_DROPOUT`, *ad hoc*) |
| cabeza densa | — (no se busca) | `[32]` (`TRF_HIDDEN_UNITS`, *ad hoc*) |
| `learning_rate` | $[10^{-4}, 5\times10^{-3}]$ log | 0.000215 (optimización bayesiana) |
| `discount_factor` ($\gamma$) | $[0.92, 0.995]$ | 0.9234 (optimización bayesiana) |
| `num_episodes` | $[20\,000, 60\,000]$ | 30 000 (optimización bayesiana) |

`train_transformer.py` toma de `inputs/best_params.json` los hiperparámetros
de entrenamiento y la semilla (45); toda la arquitectura sale siempre de
`config/CONFIG.py`. El archivo actual todavía conserva claves de arquitectura
(`history_len`, `d_model`, `hidden_units`, ...) que el entrenamiento no usa.
Su `mean_perf` (0.9245) es el puntaje registrado por la búsqueda, no el del
modelo desplegado. Los valores de `CONFIG.py` coinciden con los que usa el entrenamiento, así que
la configuración por defecto reproduce el modelo desplegado (27 019
parámetros, 0.927 en la referencia).

Cada trial entrena desde cero con su propio `num_episodes` y devuelve el
rendimiento medio sobre las 64 secuencias fijas.

### Almacenamiento

- **Estudio Optuna**: `inputs/<fecha>_BAYESIAN_OPT/optuna_study_<fecha>.db`.
- **Mejores parámetros**: `inputs/best_params.json`.
- **Logs**: `outputs/PES_TRF_log_<fecha>_BAYESIAN_OPT.txt`.

```powershell
# desde h1/
optuna-dashboard sqlite:///ml/pes_trf/inputs/<fecha>_BAYESIAN_OPT/optuna_study_<fecha>.db
```

---

## 6. Estructura del código

### `ext/transformer_model.py`

Contiene:

- **Enmascarado causal**: se aplica con `use_causal_mask=True` sobre la capa
  `MultiHeadAttention` de cada bloque; no existe una función `causal_mask`.
- **Vector posicional fijo**: el código crea un `tf.keras.layers.Embedding`
  (`name="pos_embed"`), pero lo evalúa sobre posiciones constantes al
  construir el modelo, de modo que su salida queda como una constante
  inicializada con Glorot uniforme a partir de la semilla y **no se entrena**
  (no forma parte de los 27 019 parámetros entrenables). La variante
  sinusoidal se menciona en el documento teórico pero no está implementada.
- **Bloques encoder inline**: cada bloque Pre-LN aplica MHA causal + FFN con
  conexiones residuales, ensamblado dentro de `build_q_network`.
- **`build_q_network(state_dim, action_dim, history_len, d_model,
  num_heads, key_dim, ff_dim, num_layers, ...)`**: ensambla
  `Input → Masking → Dense(d_model) → + posición → N × bloque encoder (causal)
   → Lambda(last_token) → Dense(32, ReLU) → Dense(11)` (last-token pooling, no global avg).
- **`train_step_trf(...)`**: paso Double DQN con Huber loss.
- **`Lambda` de pooling**: el pooling de último token usa una
  `tf.keras.layers.Lambda` (`name="last_token"`), por lo que al cargar el
  modelo se requiere `safe_mode=False`:

```python
model = tf.keras.models.load_model("trf_model.keras", safe_mode=False)
```

### `ext/pandemic.py`

Mismo `PandemicEnv` Gymnasium-compatible (resources=39, severity=...) que
otros paquetes.

### `ext/train_transformer.py`

Bucle de entrenamiento, ε-decay, replay buffer, sync de target network.

### `ext/optimize_tr.py`

Define `objective(trial)` y `main(n_trials)` para Optuna.

### `__main__.py`

Carga `trf_model.keras`, mantiene una `HistoryDeque` por secuencia, ejecuta
los 8×8×(3..10) trials del escenario y persiste resultados.

---

## 7. `HistoryDeque`: ventana deslizante

Es una cola circular FIFO de longitud fija `TRF_HISTORY_LEN`:

```python
class HistoryDeque:
    def __init__(self, history_len: int, state_dim: int):
        self.history_len = history_len
        self.state_dim   = state_dim
        self._buffer     = collections.deque(maxlen=history_len)

    def reset(self):
        self._buffer.clear()

    def append_step(self, state):
        self._buffer.append(numpy.asarray(state, dtype=numpy.float32))

    def current_window(self):
        # Forma: (history_len, 3), con padding de ceros a la izquierda
        window = numpy.zeros((self.history_len, self.state_dim), dtype=numpy.float32)
        items = list(self._buffer)[-self.history_len:]
        if items:
            window[-len(items):] = numpy.asarray(items, dtype=numpy.float32)
        return window
```

- Al iniciar una **nueva secuencia** se llama a `reset()`: la historia
  se vacía y `current_window()` devuelve ceros (estado neutro).
- En cada trial se hace `append_step(state_normalizado)` y se pasa
  `current_window()` a la red.
- Forma del input al modelo: `(batch, history_len, 3)`.

---

## 8. Archivos de entrada/salida

### Entradas (`ml/pes_trf/inputs/`)

| Archivo | Descripción |
|---|---|
| `trf_model.keras` | Modelo Transformer entrenado (cargar con `safe_mode=False`) |
| `best_params.json` | Mejores hiperparámetros del estudio Optuna |
| `initial_severity.csv` | Severidades iniciales por secuencia |
| `sequence_lengths.csv` | Número de trials por secuencia |
| `rewards.npy` | Curva de recompensas del entrenamiento |
| `<fecha>_TRF_TRAIN/` | Modelo, recompensas, confianzas y figuras fechadas del entrenamiento |
| `<fecha>_BAYESIAN_OPT/` | Bases de datos Optuna |

### Salidas (`ml/pes_trf/outputs/`)

| Archivo / carpeta | Descripción |
|---|---|
| `PES_TRF_log_<fecha>.txt` | Log dual-stream del experimento |
| `<fecha>_TRF_AGENT/` | CSV + figuras de resultados |

---

## 9. Resultados de rendimiento

Evaluación del 2 de mayo de 2026 sobre las $n=64$ secuencias fijas de
referencia (el benchmark de `general/` obtiene 0.927189):

| Métrica | Valor |
|---|---|
| **Rendimiento medio normalizado** | **0.927180** |
| Desviación estándar | 0.045469 |
| Tamaño de muestra | 64 |

**El Transformer es el mejor agente individual del workspace.**

### Comparación global (referencia `sev_base` del benchmark)

| Agente | Algoritmo | Rendimiento medio | $\sigma$ |
|---|---|---|---|
| `pes_base` | Q-Learning tabular | 0.871 | 0.074 |
| `pes_ql` | Q-Learning + Optuna | 0.887 | 0.061 |
| `pes_dql` | Double Q-Learning + PBRS | 0.896 | 0.048 |
| `pes_a2c` | A2C | 0.887 | 0.063 |
| `pes_dqn` | DQN | 0.894 | 0.055 |
| `pes_rdqn` | Recurrent DQN (LSTM) | 0.899 | 0.049 |
| **`pes_trf`** | **Causal Transformer** | **0.927** | **0.045** |
| `pes_ens` | Ensamble ponderado con *prior* de severidad | 0.937 | 0.035 |

Observaciones clave:

- El Transformer tiene la menor desviación estándar de los modelos
  individuales, lo que indica decisiones **más consistentes** entre
  secuencias.
- El DQN recurrente recibe la misma ventana de 6 estados y queda por debajo,
  de modo que la ventaja no se debe sólo a ver la historia; los resultados no
  identifican su causa.
- El ensamble `pes_ens` mejora al Transformer en la referencia (0.937 frente
  a 0.927) y en los escenarios de generalización (0.939 frente a 0.930).

---

## 10. Comparación con RDQN y métodos tabulares

### vs RDQN (Recurrent DQN, LSTM)

| Aspecto | RDQN | Transformer |
|---|---|---|
| Memoria | Estado oculto $h_t$ | Atención sobre $h$ posiciones |
| Procesamiento | Secuencial $O(h)$ | Paralelo $O(h^2)$ pero vectorizado |
| Cuello de botella | Vector oculto único | Ninguno |
| Sensibilidad a $h$ largo | Olvido gradual | Atención decreciente pero accesible |
| Rendimiento (Pandemic) | 0.899 | **0.927** |

### vs métodos tabulares (`pes_base`, `pes_ql`, `pes_dql`)

- Los métodos tabulares **discretizan** el estado y construyen una tabla
  $Q[s, a]$. No pueden generalizar a estados no vistos.
- El Transformer aprende una representación **continua y contextual** del
  estado, generalizando sobre patrones no exactamente repetidos.
- Diferencia de rendimiento en la referencia: **+5.7 puntos** sobre
  `pes_base` (0.927 frente a 0.871) y +4.1 sobre `pes_ql`.

### vs DQN (sin memoria)

DQN ve solo el estado actual. El Transformer integra la trayectoria reciente,
lo que le permite **anticipar** patrones de severidad y **distribuir mejor**
los recursos restantes.

---

## 11. Notas operativas importantes

- **Carga del modelo**: la capa `Lambda` de pooling (`last_token`) obliga a
  pasar `safe_mode=False`:

  ```python
  model = tf.keras.models.load_model(
      "ml/pes_trf/inputs/trf_model.keras",
      safe_mode=False
  )
  ```

- **Reset de la historia**: indispensable al inicio de cada nueva secuencia
  (pasa de un escenario al siguiente).
- **Variables de entorno**: `PYTHONIOENCODING=utf-8` y
  `TF_ENABLE_ONEDNN_OPTS=0` deben estar definidas antes de ejecutar
  cualquier script bajo Windows.
- **GPU**: el modelo entrena correctamente en CPU; en GPU es ~5× más rápido.

---

## 12. Referencias

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,
  A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need.
  *Advances in Neural Information Processing Systems, 30*.
- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare,
  M. G., Graves, A., Riedmiller, M., et al. (2015). Human-level control
  through deep reinforcement learning. *Nature, 518*(7540), 529–533.
- Hasselt, H. van, Guez, A., & Silver, D. (2016). Deep reinforcement learning
  with double Q-learning. *Proceedings of the AAAI Conference on Artificial
  Intelligence, 30*(1), 2094–2100.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural
  Computation, 9*(8), 1735–1780.
- Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel,
  P., Srinivas, A., & Mordatch, I. (2021). Decision transformer:
  Reinforcement learning via sequence modeling. *Advances in Neural
  Information Processing Systems, 34*.
- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A
  next-generation hyperparameter optimization framework. *Proceedings of the
  25th ACM SIGKDD International Conference on Knowledge Discovery & Data
  Mining*, 2623–2631.
