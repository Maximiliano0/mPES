# `pes_dqn` — Guía de uso e implementación

> Paquete: `ml.pes_dqn`
> Algoritmo: **Deep Q-Network (DQN)** con *Experience Replay*, *Target Network* y *Double DQN*
> Última actualización: 2026-04-30

---

## 1. ¿Qué es DQN y por qué supera al Q-Learning tabular?

El **Deep Q-Network** (Mnih et al., 2015) reemplaza la tabla `Q[s, a]` del
Q-Learning tabular por una **red neuronal** $Q_\theta(s, a)$ que aproxima la
función de valor de acción. En el Pandemic Scenario, el estado normalizado
es continuo:

$$
s = \left[\frac{\text{recursos}}{39},\; \frac{\text{trial}}{10},\; \frac{\text{severidad}}{1}\right] \in [0,1]^3
$$

Una tabla discretiza ese espacio en celdas y sufre dos problemas graves:

1. **Explosión combinatoria**: con discretización fina el número de estados
   crece exponencialmente y la mayoría nunca se visita.
2. **Falta de generalización**: dos estados muy cercanos (`severidad=0.51`
   vs `0.52`) se tratan como independientes; el agente no transfiere lo
   aprendido.

DQN resuelve ambos: la red interpola en el espacio continuo y comparte
parámetros entre estados similares. El resultado en este proyecto es un
salto en desempeño desde valores tabulares (~0.83–0.86) a
**`raw_mean_perf = 0.8937 ± 0.0552` (n=64)** medido el 2026-04-30.

---

## 2. Comandos CLI

Activa primero el entorno virtual:

**Windows (PowerShell):**
```powershell
win_mpes_env\Scripts\Activate.ps1
```

**Linux:**
```bash
source linux_mpes_env/bin/activate
```

Luego ejecuta cualquiera de los tres modos:

| Modo | Comando |
|------|---------|
| Experimento completo (8 bloques × 8 sec.) | `python -m ml.pes_dqn` |
| Entrenamiento del agente | `python -m ml.pes_dqn.ext.train_dqn [num_episodes]` |
| Optimización bayesiana | `python -m ml.pes_dqn.ext.optimize_dqn [n_trials]` |

Ejemplos:

```powershell
# Entrenar 175 000 episodios (valor por defecto, `DQN_EPISODES`)
python -m ml.pes_dqn.ext.train_dqn 175000

# Buscar hiperparámetros con 100 trials Optuna
python -m ml.pes_dqn.ext.optimize_dqn 100

# Lanzar el experimento que carga inputs/dqn_model.keras
python -m ml.pes_dqn
```

> **Tip**: Si lanzas en background o con redirección de stdout, exporta
> antes `PYTHONIOENCODING=utf-8` y `TF_ENABLE_ONEDNN_OPTS=0`, y asegúrate
> de que `VIRTUAL_ENV` apunta al venv activo.

---

## 3. Pipeline de entrenamiento

El entrenamiento vive en [ml/pes_dqn/ext/train_dqn.py](../ext/train_dqn.py)
y la red en [ml/pes_dqn/ext/dqn_model.py](../ext/dqn_model.py).

### 3.1 Inicialización

1. Se construye el entorno `PandemicEnv` (gymnasium), con
   `AVAILABLE_RESOURCES_PER_SEQUENCE = 39`, severidades iniciales leídas de
   `inputs/initial_severity.csv` y longitudes de secuencia de
   `inputs/sequence_lengths.csv`.
2. Se crean **dos** redes Q idénticas: `q_online` y `q_target`, ambas vía
   `build_q_network(state_dim=3, action_dim=11, hidden_units=[64, 64])`.
3. Se instancia el `ReplayBuffer(max_size=20000)` (`DQN_REPLAY_BUFFER_SIZE`).
4. Se fija `epsilon = DQN_EPSILON_INITIAL ≈ 0.963` (valor mejor de Optuna).

### 3.2 Bucle por episodio

Para cada episodio (1…`DQN_EPISODES = 175 000`):

1. **Reset** del entorno → `state ∈ ℝ³` normalizado por
   `normalize_state(state, max_resources, max_trials, max_severity)`,
   donde `max_resources = AVAILABLE_RESOURCES_PER_SEQUENCE - 9 = 30`,
   `max_trials = 10`, `max_severity = 9`.
2. Para cada paso del episodio:
   - Selecciona acción ε-greedy: con prob. `ε` aleatoria, en otro caso
     `argmax_a Q_online(state, a)`.
   - Ejecuta paso → obtiene `(reward, next_state, done)`.
   - Almacena la transición en el `ReplayBuffer`.
   - Si el buffer tiene ≥ `DQN_BATCH_SIZE = 128` transiciones, llama a
     `train_step_dqn(batch)`.
   - Cada `DQN_TARGET_SYNC_FREQ = 1 000` pasos invoca
     `sync_target_network(q_online, q_target)`.
3. Actualiza `ε ← max(ε_min, ε · decay)` con decaimiento exponencial y
   *warm-up* inicial (ver §6 del documento de teoría).

### 3.3 Cálculo del objetivo (Double DQN)

Dentro de `train_step_dqn`:

$$
a^* = \arg\max_a Q_{\text{online}}(s', a)\quad\quad
y = r + \gamma \cdot Q_{\text{target}}(s', a^*) \cdot (1 - d)
$$

Pérdida **Huber** entre $y$ y $Q_{\text{online}}(s, a)$, optimizada con
**Adam** (`learning_rate ≈ 0.001`).

### 3.4 Salida

Al terminar, el modelo se guarda como `inputs/dqn_model.keras` y los
*rewards* por episodio en `inputs/rewards.npy`.

---

## 4. Optimización bayesiana

Implementada en
[ml/pes_dqn/ext/optimize_dqn.py](../ext/optimize_dqn.py) usando **Optuna**
con el muestreador TPE (Akiba et al., 2019).

### 4.1 Espacio de búsqueda

La función `objective()` muestrea **16 hiperparámetros** (no 6); los
principales aparecen abajo — ver [ext/optimize_dqn.py](../ext/optimize_dqn.py)
líneas ≈268-285 para la lista completa:

| Hiperparámetro | Tipo | Rango |
|---|---|---|
| `learning_rate` | log-float | 1e-4 … 5e-3 |
| `discount_factor` (γ) | float | 0.92 … 0.995 |
| `hidden_layer_size` | categórico | 32, 64, 96, 128 |
| `num_hidden_layers` | int | 1 … 3 |
| `batch_size` | categórico | 32, 64, 128, 256 |
| `buffer_size` | categórico | 10 000, 20 000, 50 000 |
| `target_sync_freq` | int (step=500) | 500 … 5 000 |
| `epsilon_initial` | float | 0.5 … 1.0 |
| `epsilon_min` | log-float | 0.01 … 0.10 |
| `warmup_ratio` | float | 0.02 … 0.20 |
| `target_ratio` | float | 0.40 … 0.80 |
| `num_episodes` | int (step=25 000) | 50 000 … 200 000 |
| `max_grad_norm` | log-float | 0.5 … 10.0 |
| `use_pbrs` | categórico | True / False |
| `penalty_coeff` | log-float | 1e-4 … 0.30 |
| `learning_starts_frac` | float | 0.05 … 0.30 |

Nota: la velocidad de decaimiento $\lambda$ **no** se muestrea
directamente; se deriva en tiempo de entrenamiento a partir de
`epsilon_initial`, `epsilon_min`, `warmup_ratio` y `target_ratio`.

### 4.2 Función objetivo

Cada *trial* entrena un agente DQN con un subconjunto reducido de episodios
(*budget* limitado), evalúa `raw_mean_perf` sobre el conjunto fijo de
secuencias y devuelve **`mean_perf`** directamente; el estudio se crea con
`direction='maximize'`, por lo que Optuna **maximiza** el desempeño.

### 4.3 Persistencia y resultados

- Storage: SQLite en `inputs/<fecha>_BAYESIAN_OPT/optuna_study_<fecha>.db`.
- Mejores parámetros: `inputs/best_params.json` (espejo) y
  `inputs/<fecha>_BAYESIAN_OPT/best_params_<fecha>.json` (original).
- Mejor modelo: `inputs/<fecha>_BAYESIAN_OPT/dqn_best_<fecha>.keras` y
  `inputs/dqn_model.keras` (espejo).
- *Dashboard* en vivo: `utils/win/optuna_dashboard.ps1` (o `.sh` en Linux).

### 4.4 Mejores hiperparámetros encontrados (snapshot CONFIG.py)

```json
{
  "DQN_LEARNING_RATE":      0.0015083436603048935,
  "DQN_DISCOUNT":           0.9634244388615337,
  "DQN_HIDDEN_UNITS":       [64, 64],
  "DQN_BATCH_SIZE":         128,
  "DQN_REPLAY_BUFFER_SIZE": 20000,
  "DQN_TARGET_SYNC_FREQ":   1000,
  "DQN_EPSILON_INITIAL":    0.9627337198502147,
  "DQN_EPSILON_MIN":        0.06914686776995618,
  "DQN_EPISODES":           175000
}
```

---

## 5. Estructura del código

```
ml/pes_dqn/
├── __init__.py            # Re-exporta CONFIG, fija seeds
├── __main__.py            # Carga modelo y corre bloques/secuencias/trials
├── config/CONFIG.py       # Constantes DQN_*
├── ext/
│   ├── pandemic.py        # PandemicEnv (gymnasium)
│   ├── dqn_model.py       # build_q_network, ReplayBuffer, sync_target_network
│   ├── train_dqn.py       # Bucle de entrenamiento, train_step_dqn
│   └── optimize_dqn.py    # Estudio Optuna
├── inputs/
│   ├── dqn_model.keras    # Pesos finales
│   ├── best_params.json   # Resultado de la optimización
│   ├── initial_severity.csv
│   ├── sequence_lengths.csv
│   └── rewards.npy
└── outputs/<fecha>_DQN_AGENT/   # Logs y gráficos
```

### Funciones clave

| Función / Clase | Archivo | Rol |
|---|---|---|
| `build_q_network(input_dim, output_dim, hidden_layers)` | `dqn_model.py` | Construye la red Q como Keras `Sequential`. |
| `ReplayBuffer(max_size)` | `dqn_model.py` | Cola circular de tuplas `(s, a, r, s', d)` con muestreo uniforme. |
| `sync_target_network(q_online, q_target)` | `dqn_model.py` | Copia `q_online.get_weights()` a `q_target`. |
| `train_step_dqn(batch, q_online, q_target, optimizer, gamma)` | `train_dqn.py` | Un paso de gradiente con pérdida Huber sobre el objetivo Double DQN. |
| `normalize_state(state, max_resources, max_trials, max_severity)` | `pandemic.py` | Devuelve $s/[\max_r, \max_t, \max_\sigma]$. |

---

## 6. Archivos de entrada / salida

### Entradas (`inputs/`)

| Archivo | Contenido |
|---|---|
| `dqn_model.keras` | Red Q entrenada cargada por `__main__.py`. |
| `best_params.json` | Hiperparámetros TPE-óptimos. |
| `initial_severity.csv` | Severidades iniciales por bloque/secuencia. |
| `sequence_lengths.csv` | Número de trials por secuencia (3–10). |
| `rewards.npy` | Curva de recompensas por episodio. |

### Salidas (`outputs/<fecha>_DQN_AGENT/`)

- Log dual (consola + archivo): `PES_DQN_log_<fecha>_DQN_AGENT.txt`.
- Gráficos: severidad media por bloque, distribución de acciones,
  desempeño normalizado.

---

## 7. Resultados y su interpretación

Resultados oficiales (2026-04-30, 64 secuencias):

| Métrica | Valor |
|---|---|
| `raw_mean_perf` | **0.8937** |
| Desviación estándar | 0.0552 |
| `n` | 64 |

`raw_mean_perf` es el desempeño normalizado: 0 = peor agente posible
(deja explotar todas las ciudades) y 1 = óptimo teórico (severidad final
mínima alcanzable dada la asignación de recursos). Un valor de 0.89
significa que el agente recupera el **89 %** del margen entre la peor y
la mejor política sobre las 64 secuencias evaluadas.

La desviación de 0.055 muestra robustez: las peores secuencias siguen
por encima de 0.83, no hay colapsos catastróficos (típicos de DQN sin
*target network*).

---

## 8. Comparación con métodos tabulares

| Paquete | Algoritmo | `raw_mean_perf` | Notas |
|---|---|---|---|
| `pes_base` | Q-Learning tabular | ~0.83 | Sin generalización; muchos estados no visitados. |
| `pes_ql` | QL + Optuna | ~0.85 | Hiperparámetros óptimos pero sigue siendo tabla. |
| `pes_dql` | Double Q-Learning + PBRS | ~0.87 | Reduce sesgo de maximización. |
| **`pes_dqn`** | **DQN Double + Replay + Target** | **0.8937** | Generalización en $\mathbb{R}^3$. |

Conclusión: para un espacio de estados continuo de baja dimensión pero
con dinámica no lineal (transición $\sigma' = \max(0,\,1.4\sigma-0.4 a)$),
DQN ofrece una mejora consistente de 4–6 puntos porcentuales sobre los
métodos tabulares.

---

## Referencias

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare,
M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G.,
Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran,
D., Wierstra, D., Legg, S., & Hassabis, D. (2015). Human-level control
through deep reinforcement learning. *Nature, 518*(7540), 529–533.

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A
next-generation hyperparameter optimization framework. En *Proceedings of
the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data
Mining* (pp. 2623–2631). ACM.
