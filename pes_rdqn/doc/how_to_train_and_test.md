# Cómo Entrenar y Testear — `pes_rdqn` (Recurrent Deep Q-Network)

> Guía práctica para entrenar, optimizar y ejecutar el agente RDQN del
> paquete `pes_rdqn` en el proyecto mPES.

> **Nota sobre la variante recurrente.** A diferencia de `pes_dqn`, la
> red Q de `pes_rdqn` consume una **ventana** de los últimos
> `RDQN_HISTORY_LEN` estados (por defecto `6`) a través de una capa
> `LSTM(RDQN_LSTM_UNITS)` (por defecto `64`).  Estos dos hiperparámetros
> también forman parte del espacio de búsqueda de
> `optimize_rdqn.objective()`.

---

## Índice

1. [Prerrequisitos](#1-prerrequisitos)
2. [Quick Start](#2-quick-start)
3. [Pipeline de Entrenamiento](#3-pipeline-de-entrenamiento)
4. [Hiperparámetros por Defecto](#4-hiperparámetros-por-defecto)
5. [Tiempo de Entrenamiento](#5-tiempo-de-entrenamiento)
6. [Optimización Bayesiana (Optuna)](#6-optimización-bayesiana-optuna)
7. [Ejecución del Experimento (Testing)](#7-ejecución-del-experimento-testing)
8. [Reproducibilidad Cross-Platform y GPU (Colab → PC)](#8-reproducibilidad-cross-platform-y-gpu-colab--pc)
9. [Troubleshooting](#9-troubleshooting)
10. [Referencia Rápida](#10-referencia-rápida)

---

## 1. Prerrequisitos

### 1.1 Entorno virtual

| OS | Comando |
|----|---------|
| Windows (PowerShell) | `win_mpes_env\Scripts\Activate.ps1` |
| Linux / macOS | `source linux_mpes_env/bin/activate` |

### 1.2 Variables de entorno necesarias

```powershell
# PowerShell
$env:VIRTUAL_ENV = "$PWD\win_mpes_env"
$env:PYTHONIOENCODING = "utf-8"
$env:TF_ENABLE_ONEDNN_OPTS = "0"
```

```bash
# Bash
export VIRTUAL_ENV="$PWD/linux_mpes_env"
export PYTHONIOENCODING=utf-8
export TF_ENABLE_ONEDNN_OPTS=0
```

> `VIRTUAL_ENV` suprime el prompt interactivo de `__init__.py`.
> `PYTHONIOENCODING` evita `UnicodeEncodeError` en Windows.
> `TF_ENABLE_ONEDNN_OPTS` suprime mensajes de oneDNN.

**Variable opcional `MPES_USE_GPU`:**  Por defecto `pes_rdqn` fija TF a
CPU (único hilo + `enable_op_determinism`) para garantizar
reproducibilidad bit-a-bit.  Para habilitar GPU (p. ej. en Colab Pro+):

```bash
export MPES_USE_GPU=1
```

Ver [Sección 8](#8-reproducibilidad-cross-platform-y-gpu-colab--pc) para
el pipeline completo Colab → PC.

### 1.3 Dependencias clave

| Paquete | Versión |
|---------|---------|
| TensorFlow | 2.21.0 |
| Keras | 3.13.2 |
| numpy | 2.4.3 |
| gymnasium | 1.2.3 |
| optuna | 4.7.0 |
| matplotlib | 3.10.8 |

### 1.4 Archivos de entrada requeridos

```
pes_rdqn/inputs/
├── initial_severity.csv      # Severidades iniciales de las ciudades
└── sequence_lengths.csv      # Nº de trials por secuencia (64 secuencias)
```

---

## 2. Quick Start

```bash
# 1. Entrenar el agente RDQN (175 000 episodios por defecto)
python -m pes_rdqn.ext.train_rdqn

# 2. Ejecutar el experimento (8 bloques × 8 secuencias)
python -m pes_rdqn
```

Para entrenar con un número personalizado de episodios:

```bash
python -m pes_rdqn.ext.train_rdqn 200000
```

---

## 3. Pipeline de Entrenamiento

`ext/train_rdqn.py` ejecuta el pipeline completo en 5 etapas:

### Etapa 1 — Carga de datos

Lee `initial_severity.csv` y `sequence_lengths.csv` desde `inputs/`.
Calcula las distribuciones de probabilidad de severidades y longitudes
de secuencia para la generación aleatoria de episodios.

### Etapa 2 — Baseline aleatorio

Ejecuta 64 secuencias con acciones aleatorias como referencia.
Genera gráficas de severidad y rendimiento normalizado.

### Etapa 3 — Entrenamiento RDQN

Llama a `RDQNTraining()` con los hiperparámetros de `CONFIG.py`
(o los del mejor trial de la optimización bayesiana, si ya se ejecutó):

- Construye red online y target network (inicialización `GlorotUniform`
  sembrada con `SEED`)
- Inicializa replay buffer (20 000 transiciones por defecto,
  `RDQN_REPLAY_BUFFER_SIZE`)
- Espera a que el buffer acumule
  `learning_starts = max(int(RDQN_LEARNING_STARTS_FRAC · buffer_size), batch_size)`
  transiciones antes del primer paso de gradiente (warm-up del buffer);
  con los valores por defecto (`RDQN_LEARNING_STARTS_FRAC ≈ 0.1615`,
  `buffer_size = 20 000`) son **3 230** transiciones
- Aplica el target **Double RDQN** con enmascaramiento de acciones
  infactibles en el bootstrap
- Entrena con ε-greedy exponencial con warm-up (ε constante durante
  `WARMUP_RATIO` × episodios, luego decay exponencial hasta `ε_min`
  en `TARGET_RATIO` × episodios).  La rama aleatoria de ε-greedy se
  enmascara y usa un `numpy.random.Generator` dedicado (sembrado con
  `SEED`) para no desplazar el RNG global de NumPy.
- Publica el promedio de reward sobre una ventana móvil de los últimos
  10 000 episodios cada 10 000 episodios

> **Importante:** `train_rdqn.py` pasa `compute_confidence=False` a
> `RDQNTraining()` para reproducir exactamente los resultados de la
> optimización bayesiana.  Si se activa la confianza durante el
> entrenamiento, el RNG de NumPy se desplaza y el modelo resultante
> difiere del óptimo encontrado por Optuna.

### Etapa 4 — Persistencia

Guarda los artefactos en un directorio con timestamp:

```
pes_rdqn/inputs/<fecha>_RDQN_TRAIN/
├── rdqn_model_<fecha>.keras            # Modelo entrenado
├── rewards_<fecha>.npy                # Historial de rewards
└── training_config_<fecha>.txt        # Configuración del entrenamiento
```

Y copia el modelo a la ubicación canónica (`inputs/rdqn_model.keras`)
para que `__main__.py` lo encuentre.

### Etapa 5 — Evaluación

Ejecuta las **64 secuencias fijas** con el modelo entrenado y genera
7 visualizaciones PNG:

| Archivo | Contenido |
|---------|-----------|
| `random_player_sequence_performance_*.png` | Severidad por secuencia (baseline) |
| `random_player_normalised_performance_*.png` | Rendimiento normalizado (baseline) |
| `rdqn_agent_rewards_vs_episodes_*.png` | Curva de aprendizaje (reward vs episodios) |
| `rdqn_agent_sequence_performance_*.png` | Severidad por secuencia (RDQN) |
| `rdqn_agent_normalised_performance_*.png` | Rendimiento normalizado (RDQN) |
| `rdqn_agent_cumulative_performance_*.png` | Rendimiento acumulado (RDQN) |
| `rdqn_agent_confidences_*.png` | Confianza meta-cognitiva por decisión |

---

## 4. Hiperparámetros por Defecto

Los valores actuales en `config/CONFIG.py` provienen del mejor trial de
la optimización bayesiana (trial #41, 2026-04-23, `mean_perf = 0.893729`
sobre 64 secuencias fijas) y se usan tanto como defaults de
`train_rdqn.py` como warm-start de la próxima optimización.

| Hiperparámetro | Valor | Config variable |
|----------------|-------|-----------------|
| Learning rate (Adam) | 0.001508 | `RDQN_LEARNING_RATE` |
| Discount factor (γ) | 0.9634 | `RDQN_DISCOUNT` |
| Epsilon inicial (ε₀) | 0.9627 | `RDQN_EPSILON_INITIAL` |
| Epsilon mínimo (ε_min) | 0.0691 | `RDQN_EPSILON_MIN` |
| Warmup ratio | 0.2779 | `RDQN_WARMUP_RATIO` |
| Target ratio | 0.6290 | `RDQN_TARGET_RATIO` |
| Episodios | 175 000 | `RDQN_EPISODES` |
| Hidden units | [64] | `RDQN_HIDDEN_UNITS` |
| Batch size | 128 | `RDQN_BATCH_SIZE` |
| Replay buffer size | 20 000 | `RDQN_REPLAY_BUFFER_SIZE` |
| Target sync frequency | 1 000 steps | `RDQN_TARGET_SYNC_FREQ` |
| Max gradient norm | 3.9529 | `RDQN_MAX_GRAD_NORM` |
| PBRS β (penalty_coeff) | 0.02258 | `RDQN_PENALTY_COEFF` |
| Learning starts (fracción) | 0.16155 (→ 3 230 transiciones) | `RDQN_LEARNING_STARTS_FRAC` |
| Random seed | 42 | `SEED` |

> Tras cada nueva optimización bayesiana, los valores óptimos se copian
> manualmente a `CONFIG.py` para que `train_rdqn.py` reproduzca el mejor
> modelo encontrado.

---

## 5. Tiempo de Entrenamiento

Estimaciones aproximadas (CPU, Intel i7/i5):

| Episodios | Tiempo |
|-----------|--------|
| 100 000 | ~10-20 min |
| 175 000 (default) | ~20-35 min |
| 250 000 | ~30-50 min |

La optimización Bayesiana (60 trials × 40k-100k episodios/trial, con poda
temprana) puede tardar **12-20 horas** en CPU.  El ganador se reentrena
automáticamente al final con `RDQN_EPISODES = 175 000` para no sacrificar
calidad.

### Optimizaciones de CPU aplicadas

- `tf.function(..., reduce_retracing=True)` compila el training step (JIT)
- Threading: `intra_op=1`, `inter_op=1` + `tf.config.experimental.enable_op_determinism()`
  (single-thread y orden determinista para reproducibilidad)
- `CUDA_VISIBLE_DEVICES=-1` fuerza CPU (evita overhead de inicialización GPU)
- `tf.keras.backend.clear_session()` + `gc.collect()` al final de cada
  trial de Optuna para liberar el grafo y evitar fragmentación

---

## 6. Optimización Bayesiana (Optuna)

### 6.1 Ejecución

```bash
# Ejecutar 60 trials (por defecto)
python -m pes_rdqn.ext.optimize_rdqn

# Ejecutar 100 trials
python -m pes_rdqn.ext.optimize_rdqn 100

# Reanudar una optimización previa
python -m pes_rdqn.ext.optimize_rdqn 100 --resume 2026-03-14
```

### 6.2 Espacio de búsqueda (16 parámetros)

| Parámetro | Rango | Escala |
|-----------|-------|--------|
| `learning_rate` | [1e-4, 5e-3] | log |
| `discount_factor` | [0.92, 0.995] | uniforme |
| `epsilon_initial` | [0.80, 1.0] | uniforme |
| `epsilon_min` | [0.01, 0.20] | uniforme |
| `num_episodes` | [40k, 100k] | paso 20k (sólo opt; el ganador se reentrena a `RDQN_EPISODES`) |
| `hidden_layer_size` | {32, 64, 96, 128} | categórico |
| `num_hidden_layers` | {1, 2, 3} | categórico |
| `batch_size` | {32, 64, 128, 256} | categórico |
| `buffer_size` | [20k, 100k] | paso 10k |
| `target_sync_freq` | [500, 5 000] | paso 500 |
| `max_grad_norm` | [0.5, 5.0] | uniforme |
| `use_pbrs` | {True, False} | categórico |
| `penalty_coeff` | [1e-4, 0.1] | log (sólo si `use_pbrs=True`) |
| `warmup_ratio` | [0.05, 0.30] | uniforme (ε-warmup) |
| `target_ratio` | [0.50, 0.95] | uniforme (ε-decay target) |
| `learning_starts_frac` | [0.05, 0.25] | uniforme (warm-up del buffer) |

### 6.3 Métrica objetivo

**Maximizar** el rendimiento normalizado medio sobre las 64 secuencias
fijas de evaluación:

$$\text{score} = \frac{1}{64} \sum_{i=1}^{64} \text{normalised\_perf}_i$$

### 6.4 Infraestructura

| Componente | Detalle |
|------------|---------|
| Sampler | TPE (Tree-structured Parzen Estimator), seed=42 |
| Pruner | MedianPruner (`n_startup_trials=10`, `n_warmup_steps=2`) |
| Warm-start | Trial semilla con valores de `CONFIG.py` |
| PBRS | Reward shaping potencial ($\beta$ optimizable) |
| Velocidad | `compute_confidence=False` durante optimización (~33 % ahorro) |
| Storage | SQLite (`optuna_study_<fecha>.db`) + sidecar JSON (`best_params_<fecha>.json`) |
| Resume | `--resume YYYY-MM-DD` (lee la DB existente) |
| Persistencia | Mejor modelo en `.keras`, hiperparámetros y `trial_seed` en JSON sidecar (sin pickle, CWE-502) |
| Notificaciones | Push cada 10 trials (vía `utils.notify`, configurar `MPES_NTFY_TOPIC`) |

### 6.5 Archivos de salida

```
pes_rdqn/inputs/<fecha>_BAYESIAN_OPT/
├── rdqn_best_<fecha>.keras                 # Mejor modelo
├── rewards_best_<fecha>.npy               # Rewards del mejor trial
├── best_params_<fecha>.json               # Sidecar: params + trial_seed + mean_perf
├── optimization_results_<fecha>.txt       # Reporte completo
├── optimization_history_<fecha>.png       # Gráfica de convergencia
├── hyperparameter_importances_<fecha>.png # Importancia de parámetros
└── optuna_study_<fecha>.db                # Base de datos Optuna (fallback)
```

`train_rdqn.py --from-best YYYY-MM-DD` prefiere el JSON sidecar (no requiere
Optuna instalado) y cae a la base SQLite si el JSON falta.

> **Auto-load:** Si se omite `--from-best`, `train_rdqn.py` busca
> automáticamente el directorio `<fecha>_BAYESIAN_OPT/` más reciente bajo
> `inputs/` y reproduce su mejor trial.  Esto unifica el comportamiento
> con `pes_ql` y `pes_dql` (basta con `python -m pes_rdqn.ext.train_rdqn`
> tras una optimización).  Para usar los valores fijos de `CONFIG.py`,
> elimina o renombra el directorio `_BAYESIAN_OPT/`.

### 6.6 Después de la optimización

1. Consultar el reporte `optimization_results_*.txt` para ver el mejor
   trial y sus hiperparámetros.
2. (Opcional) Actualizar `config/CONFIG.py` con los valores óptimos si se
   quiere fijarlos como nuevo baseline.
3. Reproducir el mejor trial localmente con
   `python -m pes_rdqn.ext.train_rdqn --from-best YYYY-MM-DD`
   (lee `best_params_<fecha>.json` del directorio `_BAYESIAN_OPT/`).
4. (Opcional) Copiar `rdqn_best_*.keras` a `inputs/rdqn_model.keras`.
5. Ejecutar `python -m pes_rdqn` para validar en el experimento completo.

---

## 7. Ejecución del Experimento (Testing)

### 7.1 Prerrequisito

El modelo entrenado debe existir en `pes_rdqn/inputs/`:

```
pes_rdqn/inputs/
├── rdqn_model.keras      # Q-network entrenada
└── rewards.npy           # Historial de rewards (para visualización)
```

### 7.2 Ejecución

```bash
python -m pes_rdqn
```

### 7.3 Ciclo de vida del experimento

`__main__.py` ejecuta:

1. **Validación**: Verifica que `rdqn_model.keras` y `rewards.npy` existan.
2. **Estructura**: 8 bloques × 8 secuencias × 3-10 trials por secuencia.
3. **Inferencia**: Para cada trial:
   - Lee `max_resources`, `max_seq_length` y `max_severity` directamente
     de una instancia fresca de `Pandemic()` (esto garantiza que la
     normalización coincida con la usada en entrenamiento sin depender
     de constantes duplicadas en `CONFIG`).
   - Normaliza el estado con `normalize_state(raw_state, max_res, max_seq, max_sev)`.
   - Forward pass por la Q-network.
   - Enmascara acciones infactibles ($a > r$) con $-10^9$.
   - Selecciona $\arg\max_a Q(s, a)$.
   - Calcula confianza meta-cognitiva.
4. **Resultados**: JSON con respuestas + PNG con visualizaciones.

### 7.4 Archivos de salida

```
pes_rdqn/outputs/<fecha>_RDQN_AGENT/
├── sequence_<BB>_<SS>.json        # Respuestas por secuencia
├── block_summary_<BB>.json        # Resumen por bloque
└── experiment_summary.json        # Resumen global
```

### 7.5 Verificación de reproducibilidad (`raw_mean_perf`)

Tras guardar el JSON de resultados, `__main__.py` imprime una línea de
consistencia que compara el mean performance recién calculado contra el
valor reportado por el optimizador bayesiano:

```text
raw_mean_perf = 0.898652  (std=0.045213, n=64)
best_params.json mean_perf = 0.925985  |Δ| = 0.027333
```

`raw_mean_perf` es la media de `MyPerformances` sobre la sesión completa
de `NUM_BLOCKS × NUM_SEQUENCES`. `best_params.json['mean_perf']` es el
valor que Optuna registró para el mejor trial (se lee desde
`inputs/best_params.json`). Un `|Δ|` pequeño (< 1e-3) confirma
reproducibilidad bit-exacta. Para Recurrent DQN, una diferencia
**≤ 5e-2** es esperable cuando la optimización se realizó en GPU (kernel
cuDNN LSTM) y la verificación se ejecuta en CPU (kernel genérico): los
resultados difieren en el último decimal por cada paso recurrente y el
error se acumula a lo largo del rollout. Para confirmar localmente sin
correr el experimento completo, usar `python -m pes_rdqn.ext.eval_model`,
que imprime exactamente el mismo `|Δ|`. Véase § 8 sobre el contrato
Colab → PC.

---

## 8. Reproducibilidad Cross-Platform y GPU (Colab → PC)

El flujo recomendado es **optimizar en Colab Pro+ (GPU)** y
**reentrenar localmente (CPU)** con los hiperparámetros ganadores.
Esta sección describe el contrato que conecta ambas etapas.

### 8.1 Variable `MPES_USE_GPU`

`pes_rdqn/__init__.py`, `train_rdqn.py` y `optimize_rdqn.py` consultan la
variable `MPES_USE_GPU` **antes** de inicializar TensorFlow:

| Valor | Comportamiento |
|-------|----------------|
| no definida o `0` | `CUDA_VISIBLE_DEVICES=''` (CPU forzada, 1 hilo, determinista) |
| `1` | GPU habilitada, multi-hilo, sin `enable_op_determinism` |

El mismo código corre idénticamente en ambos modos; sólo cambia el
dispositivo de cómputo.

### 8.2 Artefactos generados por la optimización

Al final de cada `objective()` ganador, `optimize_rdqn.py` escribe a
`inputs/<DATE>_BAYESIAN_OPT/`:

| Archivo | Contenido |
|---------|-----------|
| `rdqn_best_<date>.keras` | Q-network completa del mejor trial |
| `rewards_best_<date>.npy` | Curva de reward del mejor trial |
| `best_params_<date>.json` | Sidecar JSON con hiperparámetros + `trial_seed` + `mean_perf` |
| `_best_artifacts.npz` | Pesos del mejor trial serializados (NPZ, sin `pickle`) |
| `_best_artifacts.json` | Metadatos del mejor trial (valor, hidden_units, seed, n_weights) |
| `optimization_results_<date>.txt` | Reporte humano-legible con **CONFIG.PY snippet** copy-paste |
| `optimization_history_<date>.png` | Gráfica de convergencia |
| `hyperparameter_importances_<date>.png` | Importancia de parámetros |
| `optuna_study_<date>.db` | Base SQLite (permite `--resume`) |

El par **NPZ + JSON sidecar** sustituye al uso anterior de `pickle`
(mitiga CWE-502).  `_save_best_artifacts()` y `_load_best_artifacts()`
gestionan ambos formatos.

### 8.3 Bloque “CONFIG.PY SNIPPET”

`optimization_results_<date>.txt` incluye un bloque listo para pegar:

```text
CONFIG.PY SNIPPET (copy-paste into pes_rdqn/config/CONFIG.py)
----------------------------------------------------------------
RDQN_LEARNING_RATE        = 0.001508
RDQN_DISCOUNT             = 0.9634
RDQN_EPSILON_INITIAL      = 0.9627
RDQN_EPSILON_MIN          = 0.0691
RDQN_HIDDEN_UNITS         = [64]
...
RDQN_LEARNING_STARTS_FRAC = 0.16155
RDQN_EPISODES             = 175000   # full retrain length
```

Nota: `RDQN_EPISODES` se reemplaza por la longitud de **reentrenamiento
completo** (no la usada en optimización, que es más corta para acelerar
la búsqueda).

### 8.4 Pipeline recomendado

```text
  +----------------+        +-------------------+        +-----------------+
  | Colab Pro+ GPU |  --->  | Drive _BAYESIAN_  |  --->  | PC local CPU    |
  | optimize_rdqn   |        |   OPT/<date>/     |        | train_rdqn       |
  | MPES_USE_GPU=1 |        | (NPZ+JSON+keras)  |        | --from-best     |
  +----------------+        +-------------------+        +-----------------+
```

1. **Colab**:

   ```bash
   export MPES_USE_GPU=1
   python -m pes_rdqn.ext.optimize_rdqn 60
   ```

2. **Sincronizar** el directorio `inputs/<date>_BAYESIAN_OPT/` al PC
   local vía Drive / `rsync`.
3. **PC local**:

   ```bash
   python -m pes_rdqn.ext.train_rdqn --from-best <date>
   ```

   `train_rdqn.py` lee `best_params_<date>.json`, fija el seed y
   reentrena con `RDQN_EPISODES` completo en CPU determinista.

### 8.5 Garantías de reproducibilidad

- Mismo seed (`SEED = 42`) en NumPy, TensorFlow y Optuna.
- `tf.config.experimental.enable_op_determinism()` en CPU.
- `compute_confidence=False` durante entrenamiento (ver
  `explained_rdqn.md` §14.2).
- Las diferencias numéricas entre CPU y GPU son inherentes al hardware
  (orden de reducciones en cuDNN); por eso el modelo “de producción”
  siempre se reentrena en la máquina objetivo (CPU local) tras la
  búsqueda.

> **Nota:** Los denominadores de normalización se obtienen del propio
> entorno (`env.max_resources`, `env.max_seq_length`, `env.max_severity`),
> que son exactamente los valores usados durante `RDQNTraining`.  Con la
> configuración actual ello equivale a $r_{\max} = 30$, $t_{\max} = 10$
> y $v_{\max} = 9$.  Ver `src/pygameMediator.py`.

---

## 9. Troubleshooting

### No se encuentra `rdqn_model.keras`

```
FileNotFoundError: rdqn_model.keras not found
```

**Solución**: Entrenar primero con `python -m pes_rdqn.ext.train_rdqn`.

### `UnicodeEncodeError` en Windows

```
UnicodeEncodeError: 'charmap' codec can't encode characters
```

**Solución**: Establecer `$env:PYTHONIOENCODING = "utf-8"` antes de
ejecutar.

### TensorFlow muestra warnings excesivos

**Solución**: El paquete ya configura `TF_CPP_MIN_LOG_LEVEL=3` y
`TF_ENABLE_ONEDNN_OPTS=0` en `__init__.py`.  Si persisten, verificar que
las variables de entorno estén establecidas.

### OOM durante optimización Bayesiana

Con buffers grandes (100k) y redes anchas (3 capas × 128 unidades),
el consumo de memoria puede crecer.  El propio `objective()` libera el
grafo TF al final de cada trial (`tf.keras.backend.clear_session()` +
`gc.collect()`) para mitigar la fragmentación, pero ante OOM se puede:

- Reducir `buffer_size` máximo en el espacio de búsqueda.
- Reducir el máximo de `hidden_layer_size` (× `num_hidden_layers`).
- Ejecutar menos trials o en máquinas con más RAM.

### El prompt "Press ENTER" aparece al importar

**Solución**: Establecer `VIRTUAL_ENV` antes de importar:

```powershell
$env:VIRTUAL_ENV = "$PWD\win_mpes_env"
```

### Resultados irreproducibles

RDQN usa `SEED = 42` para NumPy, Python `random`, TF y —adicionalmente—
para un `numpy.random.Generator` **dedicado** a la rama aleatoria de
ε-greedy.  Además, `rdqn_model.py` fija un único hilo intra/inter-op y
activa `tf.config.experimental.enable_op_determinism()` al importarse,
por lo que la ejecución en CPU es bit-a-bit reproducible.

> **Atención:** Si `RDQNTraining()` se llama con `compute_confidence=True`
> (el default), las llamadas a `numpy.random.normal()` dentro de
> `rdqn_agent_meta_cognitive` consumen ≈1.5 M de números aleatorios
> adicionales durante 175 000 episodios, desplazando el RNG global.
> Esto hace que `env.random_sequence()` genere episodios distintos a los
> de la optimización (que usa `compute_confidence=False`), resultando en
> un modelo diferente aunque el seed sea el mismo.
> **Solución:** Asegurar que `train_rdqn.py` pase `compute_confidence=False`.

### Normalización inconsistente entre entrenamiento e inferencia

```
Q-values muy similares entre sí (rango -75 a -113), agente siempre asigna 9-10
```

**Causa histórica**: en versiones anteriores de `pygameMediator.py` los
denominadores de `normalize_state()` se leían de constantes de `CONFIG`
que podían diverger de los valores efectivos del entorno.

**Solución actual**: el módulo crea una instancia fresca de `Pandemic()`
y lee `env.max_resources`, `env.max_seq_length` y `env.max_severity`
directamente, garantizando que coincidan **siempre** con los usados en
`RDQNTraining`:

| Componente | Entrenamiento (`pandemic.py`) | Inferencia (`pygameMediator.py`) |
|---|---|---|
| `max_resources` | `env.max_resources` = 30 | `AVAILABLE_RESOURCES_PER_SEQUENCE - 9` = 30 |
| `max_trials` | `env.max_seq_length` = 10 | `NUM_MAX_TRIALS` = 10 |
| `max_severity` | `env.max_severity` = 9 | `MAX_SEVERITY` = 9 |

---

## 10. Referencia Rápida

| Tarea | Comando |
|-------|---------|
| Entrenar (175k episodios) | `python -m pes_rdqn.ext.train_rdqn` |
| Entrenar (N episodios) | `python -m pes_rdqn.ext.train_rdqn N` |
| Optimización Bayesiana (60 trials) | `python -m pes_rdqn.ext.optimize_rdqn` |
| Optimización (100 trials) | `python -m pes_rdqn.ext.optimize_rdqn 100` |
| Reanudar optimización | `python -m pes_rdqn.ext.optimize_rdqn 100 --resume YYYY-MM-DD` |
| Ejecutar experimento | `python -m pes_rdqn` |
