# `pes_rdqn` — Guía de uso e implementación

> Paquete: `ml.pes_rdqn`
> Algoritmo: **Recurrent Deep Q-Network (RDQN)** — DQN con LSTM sobre la
> historia de trials.
> Última actualización: 2026-04-30

---

## 1. ¿Qué es RDQN y por qué importa el contexto temporal?

El **Recurrent DQN** (Hausknecht & Stone, 2015) sustituye la red
*feedforward* del DQN por una red recurrente (LSTM) que recibe una
**ventana de estados pasados** en vez del estado actual aislado. La
intuición:

- En el Pandemic Scenario el "estado oficial"
  $s_t = [R_t/30,\, t_t/10,\, S_t/9]$ no codifica **cómo** llegamos a
  esa severidad: ¿el agente acaba de gastar 5 recursos seguidos? ¿Lleva
  3 trials sin asignar nada?
- Esa información es relevante para predecir la dinámica
  $\sigma_{t+1} = \max(0, 1.4\sigma_t - 0.4 a_t)$ cuando combinamos
  múltiples secuencias en un mismo bloque.
- El LSTM resume las últimas $L$ observaciones en su estado oculto y
  produce $Q(s_{t-L+1:t}, a)$.

Resultado en este proyecto:
**`raw_mean_perf = 0.8987 ± 0.0488` (n=64)** medido el 2026-04-30 — una
mejora pequeña pero consistente sobre `pes_dqn` (0.8937).

---

## 2. Comandos CLI

Activación del entorno:

**Windows (PowerShell):**

```powershell
win_mpes_env\Scripts\Activate.ps1
```

| Modo | Comando |
|---|---|
| Experimento completo | `python -m ml.pes_rdqn` |
| Entrenamiento RDQN | `python -m ml.pes_rdqn.ext.train_rdqn [num_episodes]` |
| Optimización bayesiana | `python -m ml.pes_rdqn.ext.optimize_rdqn [n_trials]` |
| Evaluación de un modelo ya entrenado | `python -m ml.pes_rdqn.ext.eval_model [ruta.keras]` |

Ejemplos:

```powershell
python -m ml.pes_rdqn.ext.train_rdqn         # reproduce el modelo desplegado
python -m ml.pes_rdqn.ext.optimize_rdqn 80
python -m ml.pes_rdqn
```

Sin argumento, `train_rdqn` usa `RDQN_EPISODES = 30 000`, el valor del mejor
ensayo.

`eval_model` carga `inputs/rdqn_model.keras` (o el `.keras` indicado) y lo
evalúa con el mismo protocolo de 64 secuencias fijas que `optimize_rdqn`;
imprime media, desviación, mínimo, máximo y el vector por secuencia, sin
entrenar ni tocar el estudio Optuna.

> Recuerda exportar `PYTHONIOENCODING=utf-8`,
> `TF_ENABLE_ONEDNN_OPTS=0` y `VIRTUAL_ENV` antes de lanzar procesos
> redirigidos a archivo (especialmente en Windows).

---

## 3. La ventana deslizante `HistoryDeque`

El componente clave que diferencia este paquete del DQN simple es la
clase `HistoryDeque`, definida en
[ext/rdqn_model.py](../ext/rdqn_model.py):

```python
class HistoryDeque:
    """Per-episode sliding window of normalised states, left-padded with zeros."""
    def __init__(self, history_len: int, state_dim: int):
        self._history_len = history_len
        self._state_dim   = state_dim
        self._buffer      = collections.deque(maxlen=history_len)

    def append_step(self, state):
        self._buffer.append(numpy.asarray(state, dtype=numpy.float32))

    def reset(self):
        self._buffer.clear()

    def current_window(self):
        window = numpy.zeros((self._history_len, self._state_dim),
                             dtype=numpy.float32)
        items = list(self._buffer)[-self._history_len:]
        if items:
            window[-len(items):] = numpy.asarray(items, dtype=numpy.float32)
        return window           # shape (L, 3)  — batch axis added by caller
```

### Comportamiento

- Al iniciar un episodio se llama `reset()`: la ventana se vacía y
  `current_window()` devolverá ceros (padding) hasta acumular pasos.
- Cada `step` del entorno se hace `append_step(normalize_state(s))`. La
  cola conserva las últimas $L$ observaciones normalizadas.
- Antes de pedir Q-values al modelo, el llamador añade el eje de batch:
  `q_values = q_online(window[numpy.newaxis, :, :])`, produciendo un
  tensor con forma `(1, history_len, 3)`, exactamente la entrada esperada
  por la capa LSTM.

### Diagrama

```
t=1: [0,0,0] [0,0,0] [0,0,0] [0,0,0] [0,0,0] [s₁]
t=2: [0,0,0] [0,0,0] [0,0,0] [0,0,0] [s₁]   [s₂]
...
t=k: [s_{k-5}] [s_{k-4}] [s_{k-3}] [s_{k-2}] [s_{k-1}] [s_k]    (L=6)
```

---

## 4. Pipeline de entrenamiento con LSTM

Implementado en [ext/train_rdqn.py](../ext/train_rdqn.py).

### 4.1 Arquitectura (modelo desplegado, 34 027 parámetros)

La arquitectura se eligió mediante exploraciones *ad hoc* y está fijada en
`config/CONFIG.py`: optimizarla con la búsqueda bayesiana era demasiado
costoso para los recursos de cómputo disponibles.

```
Input (6, 3)       ← RDQN_HISTORY_LEN
        │
        ▼
   LSTM(64)        ← RDQN_LSTM_UNITS
        │
        ▼
   Dense(96, ReLU) ← RDQN_HIDDEN_UNITS
        │
        ▼
   Dense(96, ReLU)
        │
        ▼
   Dense(11, lineal) → Q(s_{t-L+1:t}, a)
```

Construida por `build_q_network(state_dim, action_dim, hidden_units,
history_len, lstm_units)` en `rdqn_model.py`.

### 4.2 Bucle por episodio

1. `env.reset()` → estado inicial; `history.reset()`;
   `history.append_step(s₀)`.
2. Para cada paso:
   - `window = history.current_window()`
   - `q_values = q_online(window[numpy.newaxis, :, :])`
   - Selección ε-greedy de acción.
   - `step` → `(r, s_{t+1}, done)`; normalizar y `append_step`.
   - Guardar la **secuencia completa** (no solo el último estado) en el
     replay: tupla `(history_seq, a, r, history_seq_next, done)`.
3. Si el buffer tiene ≥ `batch_size` secuencias, ejecutar
   `train_step_rdqn(batch)`:
   - Forma del minibatch: `(B, L, 3)` para estados y `(B, L, 3)` para
     siguientes estados.
   - Objetivo Double DQN idéntico al de `pes_dqn`, pero los Q-values se
     calculan a partir de **secuencias** completas.
4. Sincronizar la red objetivo cada `RDQN_TARGET_SYNC_FREQ` pasos.
5. Decaer `ε`.

### 4.3 Salida

Modelo guardado en `inputs/rdqn_model.keras` y curva de recompensas en
`inputs/rewards.npy`.

---

## 5. Optimización bayesiana

[ext/optimize_rdqn.py](../ext/optimize_rdqn.py) define un estudio Optuna
con TPE sobre los hiperparámetros de entrenamiento. La arquitectura no se
busca: se toma de `config/CONFIG.py` (§4.1), porque optimizarla era demasiado
costoso para los recursos de cómputo disponibles.

### 5.1 Espacio de búsqueda

| Hiperparámetro (Optuna) | Tipo | Rango |
|---|---|---|
| `learning_rate` | log-float | 1e-4 … 5e-3 |
| `discount_factor` | float | 0.92 … 0.995 |
| `num_episodes` | int (step=10 000) | 20 000 … 60 000 |
| `batch_size` | categórico | 32, 64, 128, 256 |
| `buffer_size` | int (step=10 000) | 20 000 … 100 000 |
| `target_sync_freq` | int (step=500) | 500 … 5 000 |
| `max_grad_norm` | float | 0.5 … 5.0 |
| `use_pbrs` / `penalty_coeff` | categórico / log-float | True, False / 1e-4 … 0.1 |
| `epsilon_initial` / `epsilon_min` / `warmup_ratio` / `target_ratio` / `learning_starts_frac` | float | (ver `optimize_rdqn.py`) |

### 5.2 Objetivo

`raw_mean_perf` evaluado sobre las 64 secuencias fijas del experimento;
el estudio se crea con `direction='maximize'`, por lo que Optuna
**maximiza** `mean_perf` directamente (no se invierte el signo).

### 5.3 Hiperparámetros de entrenamiento

Valores de `inputs/best_params.json` (semilla 57) que usa el modelo
desplegado:

```json
{
  "learning_rate":        0.002415187706879797,
  "discount_factor":      0.971927101925925,
  "epsilon_initial":      0.8882905282719732,
  "epsilon_min":          0.08869362616978604,
  "batch_size":           64,
  "buffer_size":          60000,
  "target_sync_freq":     2000,
  "max_grad_norm":        3.4745683846777657,
  "penalty_coeff":        0.002630939227551139,
  "warmup_ratio":         0.12367091251239982,
  "target_ratio":         0.5990470166095718,
  "learning_starts_frac": 0.20214436623673876,
  "num_episodes":         30000
}
```

`train_rdqn.py` toma de este archivo los hiperparámetros de entrenamiento y
la semilla; toda la arquitectura (`RDQN_HISTORY_LEN` = 6, `RDQN_LSTM_UNITS` =
64 y `RDQN_HIDDEN_UNITS` = `[96, 96]`) sale siempre de `config/CONFIG.py`. El
archivo actual todavía conserva claves de arquitectura (`history_len`,
`lstm_units`, `hidden_units`, ...) que el entrenamiento no usa. Su `mean_perf`
(0.9260) es el puntaje registrado por la búsqueda, no el del modelo
desplegado, cuya arquitectura es la de `CONFIG.py`. Los valores de
`CONFIG.py` coinciden con los que usa el entrenamiento, así que la
configuración por defecto reproduce el modelo desplegado (0.8987).

Con $L = 6$ el LSTM ve los últimos 6 trials.

---

## 6. Estructura del código

```
ml/pes_rdqn/
├── __init__.py
├── __main__.py
├── config/CONFIG.py        # Constantes RDQN_*
├── ext/
│   ├── pandemic.py         # PandemicEnv compartido
│   ├── rdqn_model.py       # build_q_network, HistoryDeque, ReplayBuffer, train_step_rdqn
│   ├── train_rdqn.py       # Bucle de entrenamiento RDQNTraining
│   ├── optimize_rdqn.py    # Estudio Optuna
│   ├── eval_model.py       # Evaluación aislada de un .keras ya entrenado
│   └── tools.py            # Utilidades (entropía, conversión de secuencias, plots)
├── inputs/
│   ├── rdqn_model.keras
│   ├── best_params.json
│   ├── initial_severity.csv
│   ├── sequence_lengths.csv
│   ├── rewards.npy
│   └── <fecha>_RDQN_TRAIN/  # Artefactos fechados del entrenamiento
└── outputs/<fecha>_RDQN_AGENT/
```

### Funciones clave

| Símbolo | Archivo | Rol |
|---|---|---|
| `build_q_network(state_dim, action_dim, hidden_units, history_len, lstm_units)` | `rdqn_model.py` | Modelo Keras `Input → LSTM → Dense → Dense`. |
| `HistoryDeque` | `rdqn_model.py` | Ventana deslizante (§3). |
| `ReplayBuffer` | `rdqn_model.py` | Cola circular de tuplas con secuencias. |
| `train_step_rdqn(batch, ...)` | `rdqn_model.py` | Paso Double DQN sobre secuencias. |
| `sync_target_network(...)` | `rdqn_model.py` | Copia de pesos cada $C$ pasos. |

---

## 7. Archivos de entrada / salida

| Archivo | Rol |
|---|---|
| `inputs/rdqn_model.keras` | Modelo recurrente entrenado. |
| `inputs/best_params.json` | Mejor combinación TPE. |
| `inputs/initial_severity.csv` | Severidades iniciales. |
| `inputs/sequence_lengths.csv` | Longitudes por secuencia. |
| `inputs/rewards.npy` | Curva de recompensas del entrenamiento. |
| `outputs/<fecha>_RDQN_AGENT/PES_RDQN_log_*.txt` | Log dual. |
| `outputs/<fecha>_RDQN_AGENT/*.png` | Gráficos (severidad media, distribución de acciones). |

---

## 8. Resultados y comparación con DQN

| Paquete | Algoritmo | `raw_mean_perf` | Std | n |
|---|---|---|---|---|
| `pes_dqn` | DQN Double + Replay | 0.8937 | 0.0552 | 64 |
| **`pes_rdqn`** | **RDQN (LSTM, L=6)** | **0.8987** | **0.0488** | 64 |

Mejora absoluta: **+0.005** en media y **−0.007** en desviación
estándar en la referencia. En los 21 escenarios de generalización del
benchmark la relación se invierte (0.889 frente a 0.899 de DQN): el DQN
recurrente es el mejor modelo con secuencias de 11 a 20 pasos (0.889), pero
cae a 0.831 con severidades de 10 a 12.

---

## 9. ¿Cuándo usar RDQN vs DQN?

**Usa `pes_dqn` cuando:**

- Necesites entrenamiento más rápido (LSTM duplica el coste por paso).
- El espacio de estados ya sea totalmente observable y markoviano.
- Quieras un agente más fácil de inspeccionar/depurar.

**Usa `pes_rdqn` cuando:**

- Sospeches **observabilidad parcial**: variables ocultas que solo se
  pueden inferir mirando la trayectoria (p. ej., un sesgo dinámico que
  no aparece en $s_t$).
- Las secuencias sean largas y la dinámica no lineal.
- Busques el último punto porcentual de desempeño sin importar el coste
  computacional.

Para el Pandemic Scenario, RDQN es la opción ligeramente preferida si se
dispone de tiempo de cómputo; en otro caso, DQN es prácticamente
equivalente.

---

## Referencias

Hausknecht, M., & Stone, P. (2015). Deep recurrent Q-learning for
partially observable MDPs. En *AAAI Fall Symposium on Sequential Decision
Making for Intelligent Agents*. AAAI.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare,
M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G.,
Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran,
D., Wierstra, D., Legg, S., & Hassabis, D. (2015). Human-level control
through deep reinforcement learning. *Nature, 518*(7540), 529–533.

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A
next-generation hyperparameter optimization framework. En *Proceedings of
the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data
Mining* (pp. 2623–2631). ACM.
