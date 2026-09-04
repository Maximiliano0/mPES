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
**0.927 de rendimiento normalizado** (n=64), superando a RDQN ($\approx 0.91$)
y al resto de agentes individuales.

---

## 2. Comandos de uso

### Ejecutar el experimento completo

```powershell
# Windows
.\win_mpes_env\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"
$env:TF_ENABLE_ONEDNN_OPTS = "0"
python -m ml.pes_trf
```

```bash
# Linux
source linux_mpes_env/bin/activate
export PYTHONIOENCODING=utf-8
export TF_ENABLE_ONEDNN_OPTS=0
python -m ml.pes_trf
```

### Entrenar el modelo

```powershell
python -m ml.pes_trf.ext.train_transformer            # episodios por defecto
python -m ml.pes_trf.ext.train_transformer 50000      # personalizado
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

En `pes_trf` por defecto: $H = 4$ cabezas (`TRF_NUM_HEADS`), $d_\mathrm{model} = 32$.
El espacio de búsqueda Optuna admite $H \in \{2, 4, 8\}$.

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
   - Cada `TRF_TARGET_UPDATE_FREQ` episodios: sincroniza el target network.
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

`ext/optimize_tr.py` usa **Optuna** con muestreador TPE para buscar:

| Hiperparámetro | Rango | Óptimo (2026-05-02) |
|---|---|---|
| `history_len` (Optuna) | $[3, 10]$ entero | **6** (`TRF_HISTORY_LEN`) |
| `d_model` (Optuna) | $\{16, 32, 64, 128\}$ | **32** (`TRF_D_MODEL`) |
| `num_heads` (Optuna) | $\{2, 4, 8\}$ | **4** (`TRF_NUM_HEADS`) |
| `TRF_N_HEADS` | $\{1, 2, 4\}$ | **2** |
| `TRF_N_LAYERS` | $\{1, 2, 3\}$ | **2** |
| `TRF_FF_DIM` | $\{32, 64, 128\}$ | **64** |
| `TRF_LEARNING_RATE` | $[10^{-5}, 10^{-3}]$ log | $\approx 5\times10^{-4}$ |
| `TRF_DISCOUNT` ($\gamma$) | $[0.90, 0.99]$ | $\approx 0.97$ |

Cada trial entrena una versión reducida y devuelve el rendimiento medio sobre
64 evaluaciones independientes.

### Almacenamiento

- **Estudio Optuna**: `inputs/<fecha>_BAYESIAN_OPT/study.db`.
- **Mejores parámetros**: `inputs/best_params.json`.
- **Logs**: `outputs/PES_TRF_log_<fecha>_BAYESIAN_OPT.txt`.

```powershell
.\utils\win\optuna_dashboard.ps1 ml\pes_trf\inputs\<fecha>_BAYESIAN_OPT\study.db
```

---

## 6. Estructura del código

### `ext/transformer_model.py`

Contiene:

- **`causal_mask(seq_len)`**: genera la máscara triangular como `tf.constant`.
- **`PositionalEncoding`**: capa Keras con embedding posicional aprendido (o
  Solo la variante **aprendida** está implementada (`Embedding`); la
  sinusoidal se menciona en el documento teórico pero no en el código.
- **`TransformerEncoderLayer`**: bloque Pre-LN con MHA + FFN + residual.
- **`build_q_network(state_dim, action_dim, history_len, d_model,
  num_heads, num_layers, ...)`**: ensambla
  `Input → Dense(d_model) → + LearnedPosEmbed → N × EncoderLayer (causal)
   → Lambda(last token) → Dense(11)` (last-token pooling, no global avg).
- **`train_step_trf(...)`**: paso Double DQN con Huber loss.
- **`Lambda` causal**: la máscara se inyecta vía una `tf.keras.layers.Lambda`,
  por lo que al cargar el modelo se requiere `safe_mode=False`:

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
    def __init__(self, history_len: int, state_dim: int = 3):
        self.history_len = history_len
        self.state_dim   = state_dim
        self.buffer = numpy.zeros((history_len, state_dim), dtype=numpy.float32)

    def reset(self):
        self.buffer.fill(0.0)

    def push(self, state):
        self.buffer = numpy.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = state

    def get(self):
        # Forma: (history_len, 3)
        return self.buffer.copy()
```

- Al iniciar una **nueva secuencia** se llama a `reset()`: la historia
  se rellena con ceros (estado neutro).
- En cada trial se hace `push(state_normalizado)` y se pasa `get()` a la red.
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
| `<fecha>_BAYESIAN_OPT/` | Bases de datos Optuna |

### Salidas (`ml/pes_trf/outputs/`)

| Archivo / carpeta | Descripción |
|---|---|
| `PES_TRF_log_<fecha>.txt` | Log dual-stream del experimento |
| `<fecha>_TRF_AGENT/` | CSV + figuras de resultados |

---

## 9. Resultados de rendimiento

Evaluación del 2 de mayo de 2026 sobre $n=64$ ejecuciones independientes:

| Métrica | Valor |
|---|---|
| **Rendimiento medio normalizado** | **0.927180** |
| Desviación estándar | 0.045469 |
| Tamaño de muestra | 64 |

**El Transformer es el mejor agente individual del workspace.**

### Comparación global

| Agente | Algoritmo | Rendimiento medio | $\sigma$ |
|---|---|---|---|
| `pes_base` | Q-Learning tabular | $\approx 0.65$ | 0.10 |
| `pes_ql` | Q-Learning + Optuna | $\approx 0.78$ | 0.08 |
| `pes_dql` | Double Q-Learning + PBRS | $\approx 0.83$ | 0.07 |
| `pes_a2c` | A2C | 0.887 | 0.063 |
| `pes_dqn` | DQN | $\approx 0.89$ | 0.06 |
| `pes_rdqn` | Recurrent DQN (LSTM) | $\approx 0.91$ | 0.05 |
| **`pes_trf`** | **Causal Transformer** | **0.927** | **0.045** |
| `pes_ens` | Ensemble (votación blanda) | $\approx 0.93$ | 0.04 |

Observaciones clave:

- El Transformer reduce la desviación estándar respecto a RDQN, lo que indica
  decisiones **más consistentes**.
- Su superioridad frente a RDQN se atribuye a la atención sobre **toda** la
  historia disponible (en lugar del cuello de botella del estado oculto del
  LSTM).
- El ensemble (`pes_ens`) apenas mejora sobre `pes_trf`, lo que confirma que
  la cabeza Transformer ya captura la mayor parte de la información útil.

---

## 10. Comparación con RDQN y métodos tabulares

### vs RDQN (Recurrent DQN, LSTM)

| Aspecto | RDQN | Transformer |
|---|---|---|
| Memoria | Estado oculto $h_t$ | Atención sobre $h$ posiciones |
| Procesamiento | Secuencial $O(h)$ | Paralelo $O(h^2)$ pero vectorizado |
| Cuello de botella | Vector oculto único | Ninguno |
| Sensibilidad a $h$ largo | Olvido gradual | Atención decreciente pero accesible |
| Rendimiento (Pandemic) | $\approx 0.91$ | **0.927** |

### vs métodos tabulares (`pes_base`, `pes_ql`, `pes_dql`)

- Los métodos tabulares **discretizan** el estado y construyen una tabla
  $Q[s, a]$. No pueden generalizar a estados no vistos.
- El Transformer aprende una representación **continua y contextual** del
  estado, generalizando sobre patrones no exactamente repetidos.
- Diferencia de rendimiento: **+30 puntos porcentuales** sobre `pes_base`.

### vs DQN (sin memoria)

DQN ve solo el estado actual. El Transformer integra la trayectoria reciente,
lo que le permite **anticipar** patrones de severidad y **distribuir mejor**
los recursos restantes.

---

## 11. Notas operativas importantes

- **Carga del modelo**: la capa `Lambda` para la máscara causal obliga a
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
