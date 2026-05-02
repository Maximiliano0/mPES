# CÃ³mo Entrenar y Testear â€” `pes_trf` (Transformer Deep Q-Network)

> GuÃ­a prÃ¡ctica para entrenar, optimizar y ejecutar el agente TRF del
> paquete `pes_trf` en el proyecto mPES.

> **Nota sobre la variante Transformer.** A diferencia de `pes_dqn`
> (MLP) y `pes_rdqn` (LSTM), la red Q de `pes_trf` consume una
> **ventana** de los Ãºltimos `TRF_HISTORY_LEN` estados (por defecto
> `6`) a travÃ©s de una pila de bloques **codificador Transformer
> causal** con `TRF_NUM_LAYERS` capas, `TRF_NUM_HEADS` cabezas de
> atenciÃ³n por bloque, dimensiÃ³n por cabeza `TRF_KEY_DIM`, ancho del
> *residual stream* `TRF_D_MODEL` y FFN `TRF_FF_DIM`.  Todos estos
> hiperparÃ¡metros forman parte del espacio de bÃºsqueda de
> `optimize_tr.objective()`.

---

## Ãndice

1. [Prerrequisitos](#1-prerrequisitos)
2. [Quick Start](#2-quick-start)
3. [Pipeline de Entrenamiento](#3-pipeline-de-entrenamiento)
4. [HiperparÃ¡metros por Defecto](#4-hiperparÃ¡metros-por-defecto)
5. [Tiempo de Entrenamiento](#5-tiempo-de-entrenamiento)
6. [OptimizaciÃ³n Bayesiana (Optuna)](#6-optimizaciÃ³n-bayesiana-optuna)
7. [EjecuciÃ³n del Experimento (Testing)](#7-ejecuciÃ³n-del-experimento-testing)
8. [Reproducibilidad Cross-Platform y GPU (Colab â†’ PC)](#8-reproducibilidad-cross-platform-y-gpu-colab--pc)
9. [Troubleshooting](#9-troubleshooting)
10. [Referencia RÃ¡pida](#10-referencia-rÃ¡pida)

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

**Variable opcional `MPES_USE_GPU`:**  Por defecto `pes_trf` fija TF a
CPU (Ãºnico hilo + `enable_op_determinism`) para garantizar
reproducibilidad bit-a-bit.  Para habilitar GPU (p. ej. en Colab Pro+):

```bash
export MPES_USE_GPU=1
```

Ver [SecciÃ³n 8](#8-reproducibilidad-cross-platform-y-gpu-colab--pc) para
el pipeline completo Colab â†’ PC.

### 1.3 Dependencias clave

| Paquete | VersiÃ³n |
|---------|---------|
| TensorFlow | 2.21.0 |
| Keras | 3.13.2 |
| numpy | 2.4.3 |
| gymnasium | 1.2.3 |
| optuna | 4.7.0 |
| matplotlib | 3.10.8 |

### 1.4 Archivos de entrada requeridos

```
pes_trf/inputs/
â”œâ”€â”€ initial_severity.csv      # Severidades iniciales de las ciudades
â””â”€â”€ sequence_lengths.csv      # NÂº de trials por secuencia (64 secuencias)
```

---

## 2. Quick Start

```bash
# 1. Entrenar el agente TRF (175 000 episodios por defecto)
python -m pes_trf.ext.train_transformer

# 2. Ejecutar el experimento (8 bloques Ã— 8 secuencias)
python -m pes_trf
```

Para entrenar con un nÃºmero personalizado de episodios:

```bash
python -m pes_trf.ext.train_transformer 200000
```

---

## 3. Pipeline de Entrenamiento

`ext/train_transformer.py` ejecuta el pipeline completo en 5 etapas:

### Etapa 1 â€” Carga de datos

Lee `initial_severity.csv` y `sequence_lengths.csv` desde `inputs/`.
Calcula las distribuciones de probabilidad de severidades y longitudes
de secuencia para la generaciÃ³n aleatoria de episodios.

### Etapa 2 â€” Baseline aleatorio

Ejecuta 64 secuencias con acciones aleatorias como referencia.
Genera grÃ¡ficas de severidad y rendimiento normalizado.

### Etapa 3 â€” Entrenamiento TRF

Llama a `TRFTraining()` con los hiperparÃ¡metros de `CONFIG.py`
(o los del mejor trial de la optimizaciÃ³n bayesiana, si ya se ejecutÃ³):

- Construye red online y target network (inicializaciÃ³n `GlorotUniform`
  sembrada con `SEED`)
- Inicializa replay buffer (20 000 transiciones por defecto,
  `TRF_REPLAY_BUFFER_SIZE`)
- Espera a que el buffer acumule
  `learning_starts = max(int(TRF_LEARNING_STARTS_FRAC Â· buffer_size), batch_size)`
  transiciones antes del primer paso de gradiente (warm-up del buffer);
  con los valores por defecto (`TRF_LEARNING_STARTS_FRAC â‰ˆ 0.1615`,
  `buffer_size = 20 000`) son **3 230** transiciones
- Aplica el target **Double TRF** con enmascaramiento de acciones
  infactibles en el bootstrap
- Entrena con Îµ-greedy exponencial con warm-up (Îµ constante durante
  `WARMUP_RATIO` Ã— episodios, luego decay exponencial hasta `Îµ_min`
  en `TARGET_RATIO` Ã— episodios).  La rama aleatoria de Îµ-greedy se
  enmascara y usa un `numpy.random.Generator` dedicado (sembrado con
  `SEED`) para no desplazar el RNG global de NumPy.
- Publica el promedio de reward sobre una ventana mÃ³vil de los Ãºltimos
  10 000 episodios cada 10 000 episodios

> **Importante:** `train_transformer.py` pasa `compute_confidence=False` a
> `TRFTraining()` para reproducir exactamente los resultados de la
> optimizaciÃ³n bayesiana.  Si se activa la confianza durante el
> entrenamiento, el RNG de NumPy se desplaza y el modelo resultante
> difiere del Ã³ptimo encontrado por Optuna.

### Etapa 4 â€” Persistencia

Guarda los artefactos en un directorio con timestamp:

```
pes_trf/inputs/<fecha>_TRF_TRAIN/
â”œâ”€â”€ trf_model_<fecha>.keras            # Modelo entrenado
â”œâ”€â”€ rewards_<fecha>.npy                # Historial de rewards
â””â”€â”€ training_config_<fecha>.txt        # ConfiguraciÃ³n del entrenamiento
```

Y copia el modelo a la ubicaciÃ³n canÃ³nica (`inputs/trf_model.keras`)
para que `__main__.py` lo encuentre.

### Etapa 5 â€” EvaluaciÃ³n

Ejecuta las **64 secuencias fijas** con el modelo entrenado y genera
7 visualizaciones PNG:

| Archivo | Contenido |
|---------|-----------|
| `random_player_sequence_performance_*.png` | Severidad por secuencia (baseline) |
| `random_player_normalised_performance_*.png` | Rendimiento normalizado (baseline) |
| `trf_agent_rewards_vs_episodes_*.png` | Curva de aprendizaje (reward vs episodios) |
| `trf_agent_sequence_performance_*.png` | Severidad por secuencia (TRF) |
| `trf_agent_normalised_performance_*.png` | Rendimiento normalizado (TRF) |
| `trf_agent_cumulative_performance_*.png` | Rendimiento acumulado (TRF) |
| `trf_agent_confidences_*.png` | Confianza meta-cognitiva por decisiÃ³n |

---

## 4. HiperparÃ¡metros por Defecto

Los valores actuales en `config/CONFIG.py` provienen del mejor trial de
la optimizaciÃ³n bayesiana (trial #41, 2026-04-23, `mean_perf = 0.893729`
sobre 64 secuencias fijas) y se usan tanto como defaults de
`train_transformer.py` como warm-start de la prÃ³xima optimizaciÃ³n.

| HiperparÃ¡metro | Valor | Config variable |
|----------------|-------|-----------------|
| Learning rate (Adam) | 0.001508 | `TRF_LEARNING_RATE` |
| Discount factor (Î³) | 0.9634 | `TRF_DISCOUNT` |
| Epsilon inicial (Îµâ‚€) | 0.9627 | `TRF_EPSILON_INITIAL` |
| Epsilon mÃ­nimo (Îµ_min) | 0.0691 | `TRF_EPSILON_MIN` |
| Warmup ratio | 0.2779 | `TRF_WARMUP_RATIO` |
| Target ratio | 0.6290 | `TRF_TARGET_RATIO` |
| Episodios | 175 000 | `TRF_EPISODES` |
| Hidden units (cabeza Q) | [64] | `TRF_HIDDEN_UNITS` |
| Batch size | 128 | `TRF_BATCH_SIZE` |
| Replay buffer size | 20 000 | `TRF_REPLAY_BUFFER_SIZE` |
| Target sync frequency | 1 000 steps | `TRF_TARGET_SYNC_FREQ` |
| Max gradient norm | 3.9529 | `TRF_MAX_GRAD_NORM` |
| PBRS Î² (penalty_coeff) | 0.02258 | `TRF_PENALTY_COEFF` |
| Learning starts (fracciÃ³n) | 0.16155 (â†’ 3 230 transiciones) | `TRF_LEARNING_STARTS_FRAC` |
| Random seed | 42 | `SEED` |

> Tras cada nueva optimizaciÃ³n bayesiana, los valores Ã³ptimos se copian
> manualmente a `CONFIG.py` para que `train_transformer.py` reproduzca el mejor
> modelo encontrado.

---

## 5. Tiempo de Entrenamiento

Estimaciones aproximadas (CPU, Intel i7/i5):

| Episodios | Tiempo |
|-----------|--------|
| 100 000 | ~10-20 min |
| 175 000 (default) | ~20-35 min |
| 250 000 | ~30-50 min |

La optimizaciÃ³n Bayesiana (60 trials Ã— 40k-100k episodios/trial, con poda
temprana) puede tardar **12-20 horas** en CPU.  El ganador se reentrena
automÃ¡ticamente al final con `TRF_EPISODES = 175 000` para no sacrificar
calidad.

### Optimizaciones de CPU aplicadas

- `tf.function(..., reduce_retracing=True)` compila el training step (JIT)
- Threading: `intra_op=1`, `inter_op=1` + `tf.config.experimental.enable_op_determinism()`
  (single-thread y orden determinista para reproducibilidad)
- `CUDA_VISIBLE_DEVICES=-1` fuerza CPU (evita overhead de inicializaciÃ³n GPU)
- `tf.keras.backend.clear_session()` + `gc.collect()` al final de cada
  trial de Optuna para liberar el grafo y evitar fragmentaciÃ³n

---

## 6. OptimizaciÃ³n Bayesiana (Optuna)

### 6.1 EjecuciÃ³n

```bash
# Ejecutar 60 trials (por defecto)
python -m pes_trf.ext.optimize_tr

# Ejecutar 100 trials
python -m pes_trf.ext.optimize_tr 100

# Reanudar una optimizaciÃ³n previa
python -m pes_trf.ext.optimize_tr 100 --resume 2026-03-14
```

### 6.2 Espacio de bÃºsqueda (23 parÃ¡metros)

| ParÃ¡metro | Rango | Escala |
|-----------|-------|--------|
| `learning_rate` | [1e-4, 5e-3] | log |
| `discount_factor` | [0.92, 0.995] | uniforme |
| `epsilon_initial` | [0.80, 1.0] | uniforme |
| `epsilon_min` | [0.01, 0.20] | uniforme |
| `num_episodes` | [20k, 60k] | paso 10k (sÃ³lo opt; el ganador se reentrena a `TRF_EPISODES`) |
| `hidden_layer_size` | {32, 64, 96, 128} | categÃ³rico (cabeza Q) |
| `num_hidden_layers` | {1, 2, 3} | categÃ³rico (cabeza Q) |
| `batch_size` | {32, 64, 128, 256} | categÃ³rico |
| `buffer_size` | [20k, 100k] | paso 10k |
| `target_sync_freq` | [500, 5 000] | paso 500 |
| `max_grad_norm` | [0.5, 5.0] | uniforme |
| `use_pbrs` | {True, False} | categÃ³rico |
| `penalty_coeff` | [1e-4, 0.1] | log (sÃ³lo si `use_pbrs=True`) |
| `warmup_ratio` | [0.05, 0.30] | uniforme (Îµ-warmup) |
| `target_ratio` | [0.50, 0.95] | uniforme (Îµ-decay target) |
| `learning_starts_frac` | [0.05, 0.25] | uniforme (warm-up del buffer) |
| `history_len` | [3, 10] | entero (longitud de la ventana Transformer) |
| `d_model` | {16, 32, 64, 128} | categÃ³rico (token-embedding / residual stream) |
| `num_heads` | {2, 4, 8} | categÃ³rico (cabezas de atenciÃ³n por bloque) |
| `key_dim` | {8, 16, 32} | categÃ³rico (dimensiÃ³n por cabeza) |
| `ff_dim` | {32, 64, 128, 256} | categÃ³rico (FFN posiciÃ³n-a-posiciÃ³n) |
| `num_layers` | [1, 4] | entero (bloques de codificador apilados) |
| `dropout` | [0.0, 0.3] | uniforme (MHSA / FFN / cabeza Q) |

### 6.3 MÃ©trica objetivo

**Maximizar** el rendimiento normalizado medio sobre las 64 secuencias
fijas de evaluaciÃ³n:

$$\text{score} = \frac{1}{64} \sum_{i=1}^{64} \text{normalised\_perf}_i$$

### 6.4 Infraestructura

| Componente | Detalle |
|------------|---------|
| Sampler | TPE (Tree-structured Parzen Estimator), seed=42 |
| Pruner | MedianPruner (`n_startup_trials=5`, `n_warmup_steps=1`, `interval_steps=1`) |
| Warm-start | Trial semilla con valores de `CONFIG.py` |
| PBRS | Reward shaping potencial ($\beta$ optimizable) |
| Velocidad | `compute_confidence=False` durante optimizaciÃ³n (~33 % ahorro) |
| Storage | SQLite (`optuna_study_<fecha>.db`) + sidecar JSON (`best_params_<fecha>.json`) |
| Resume | `--resume YYYY-MM-DD` (lee la DB existente) |
| Persistencia | Mejor modelo en `.keras`, hiperparÃ¡metros y `trial_seed` en JSON sidecar (sin pickle, CWE-502) |
| Notificaciones | Push cada 10 trials (vÃ­a `utils.notify`, configurar `MPES_NTFY_TOPIC`) |

### 6.5 Archivos de salida

```
pes_trf/inputs/<fecha>_BAYESIAN_OPT/
â”œâ”€â”€ trf_best_<fecha>.keras                 # Mejor modelo
â”œâ”€â”€ rewards_best_<fecha>.npy               # Rewards del mejor trial
â”œâ”€â”€ best_params_<fecha>.json               # Sidecar: params + trial_seed + mean_perf
â”œâ”€â”€ optimization_results_<fecha>.txt       # Reporte completo
â”œâ”€â”€ optimization_history_<fecha>.png       # GrÃ¡fica de convergencia
â”œâ”€â”€ hyperparameter_importances_<fecha>.png # Importancia de parÃ¡metros
â””â”€â”€ optuna_study_<fecha>.db                # Base de datos Optuna (fallback)
```

`train_transformer.py --from-best YYYY-MM-DD` prefiere el JSON sidecar (no requiere
Optuna instalado) y cae a la base SQLite si el JSON falta.

> **Auto-load:** Si se omite `--from-best`, `train_transformer.py` busca
> automÃ¡ticamente el directorio `<fecha>_BAYESIAN_OPT/` mÃ¡s reciente bajo
> `inputs/` y reproduce su mejor trial.  Esto unifica el comportamiento
> con `pes_ql` y `pes_dql` (basta con `python -m pes_trf.ext.train_transformer`
> tras una optimizaciÃ³n).  Para usar los valores fijos de `CONFIG.py`,
> elimina o renombra el directorio `_BAYESIAN_OPT/`.

### 6.6 DespuÃ©s de la optimizaciÃ³n

1. Consultar el reporte `optimization_results_*.txt` para ver el mejor
   trial y sus hiperparÃ¡metros.
2. (Opcional) Actualizar `config/CONFIG.py` con los valores Ã³ptimos si se
   quiere fijarlos como nuevo baseline.
3. Reproducir el mejor trial localmente con
   `python -m pes_trf.ext.train_transformer --from-best YYYY-MM-DD`
   (lee `best_params_<fecha>.json` del directorio `_BAYESIAN_OPT/`).
4. (Opcional) Copiar `trf_best_*.keras` a `inputs/trf_model.keras`.
5. Ejecutar `python -m pes_trf` para validar en el experimento completo.

---

## 7. EjecuciÃ³n del Experimento (Testing)

### 7.1 Prerrequisito

El modelo entrenado debe existir en `pes_trf/inputs/`:

```
pes_trf/inputs/
â”œâ”€â”€ trf_model.keras      # Q-network entrenada
â””â”€â”€ rewards.npy           # Historial de rewards (para visualizaciÃ³n)
```

### 7.2 EjecuciÃ³n

```bash
python -m pes_trf
```

### 7.3 Ciclo de vida del experimento

`__main__.py` ejecuta:

1. **ValidaciÃ³n**: Verifica que `trf_model.keras` y `rewards.npy` existan.
2. **Estructura**: 8 bloques Ã— 8 secuencias Ã— 3-10 trials por secuencia.
3. **Inferencia**: Para cada trial:
   - Lee `max_resources`, `max_seq_length` y `max_severity` directamente
     de una instancia fresca de `Pandemic()` (esto garantiza que la
     normalizaciÃ³n coincida con la usada en entrenamiento sin depender
     de constantes duplicadas en `CONFIG`).
   - Normaliza el estado con `normalize_state(raw_state, max_res, max_seq, max_sev)`.
   - Forward pass por la Q-network.
   - Enmascara acciones infactibles ($a > r$) con $-10^9$.
   - Selecciona $\arg\max_a Q(s, a)$.
   - Calcula confianza meta-cognitiva.
4. **Resultados**: JSON con respuestas + PNG con visualizaciones.

### 7.4 Archivos de salida

```
pes_trf/outputs/<fecha>_TRF_AGENT/
â”œâ”€â”€ sequence_<BB>_<SS>.json        # Respuestas por secuencia
â”œâ”€â”€ block_summary_<BB>.json        # Resumen por bloque
â””â”€â”€ experiment_summary.json        # Resumen global
```
### 7.5 Verificación de reproducibilidad (`raw_mean_perf`)

Tras guardar el JSON de resultados, `__main__.py` imprime una línea de
consistencia que compara el mean performance recién calculado contra el
valor reportado por el optimizador bayesiano:

```text
raw_mean_perf = 0.898652  (std=0.045213, n=64)
best_params.json mean_perf = 0.898652  |Δ| = 0.000000
```

`raw_mean_perf` es la media de `MyPerformances` sobre la sesión completa
de `NUM_BLOCKS × NUM_SEQUENCES`. `best_params.json['mean_perf']` es el
valor que Optuna registró para el mejor trial (se lee desde
`inputs/best_params.json`). Un `|Δ|` pequeño (< 1e-3) confirma
reproducibilidad total; una diferencia mayor (hasta ~5e-2) suele indicar
otro `SEED`, otro CSV de entrada, u otro dispositivo de cómputo (por
ejemplo GPU vs CPU). Véase § 8 sobre el contrato Colab → PC.
---

## 8. Reproducibilidad Cross-Platform y GPU (Colab â†’ PC)

El flujo recomendado es **optimizar en Colab Pro+ (GPU)** y
**reentrenar localmente (CPU)** con los hiperparÃ¡metros ganadores.
Esta secciÃ³n describe el contrato que conecta ambas etapas.

### 8.1 Variable `MPES_USE_GPU`

`pes_trf/__init__.py`, `train_transformer.py` y `optimize_tr.py` consultan la
variable `MPES_USE_GPU` **antes** de inicializar TensorFlow:

| Valor | Comportamiento |
|-------|----------------|
| no definida o `0` | `CUDA_VISIBLE_DEVICES=''` (CPU forzada, 1 hilo, determinista) |
| `1` | GPU habilitada, multi-hilo, sin `enable_op_determinism` |

El mismo cÃ³digo corre idÃ©nticamente en ambos modos; sÃ³lo cambia el
dispositivo de cÃ³mputo.

### 8.2 Artefactos generados por la optimizaciÃ³n

Al final de cada `objective()` ganador, `optimize_tr.py` escribe a
`inputs/<DATE>_BAYESIAN_OPT/`:

| Archivo | Contenido |
|---------|-----------|
| `trf_best_<date>.keras` | Q-network completa del mejor trial |
| `rewards_best_<date>.npy` | Curva de reward del mejor trial |
| `best_params_<date>.json` | Sidecar JSON con hiperparÃ¡metros + `trial_seed` + `mean_perf` |
| `_best_artifacts.npz` | Pesos del mejor trial serializados (NPZ, sin `pickle`) |
| `_best_artifacts.json` | Metadatos del mejor trial (valor, hidden_units, seed, n_weights) |
| `optimization_results_<date>.txt` | Reporte humano-legible con **CONFIG.PY snippet** copy-paste |
| `optimization_history_<date>.png` | GrÃ¡fica de convergencia |
| `hyperparameter_importances_<date>.png` | Importancia de parÃ¡metros |
| `optuna_study_<date>.db` | Base SQLite (permite `--resume`) |

El par **NPZ + JSON sidecar** sustituye al uso anterior de `pickle`
(mitiga CWE-502).  `_save_best_artifacts()` y `_load_best_artifacts()`
gestionan ambos formatos.

### 8.3 Bloque â€œCONFIG.PY SNIPPETâ€

`optimization_results_<date>.txt` incluye un bloque listo para pegar:

```text
CONFIG.PY SNIPPET (copy-paste into pes_trf/config/CONFIG.py)
----------------------------------------------------------------
TRF_LEARNING_RATE        = 0.001508
TRF_DISCOUNT             = 0.9634
TRF_EPSILON_INITIAL      = 0.9627
TRF_EPSILON_MIN          = 0.0691
TRF_HIDDEN_UNITS         = [64]
...
TRF_LEARNING_STARTS_FRAC = 0.16155
TRF_EPISODES             = 175000   # full retrain length
```

Nota: `TRF_EPISODES` se reemplaza por la longitud de **reentrenamiento
completo** (no la usada en optimizaciÃ³n, que es mÃ¡s corta para acelerar
la bÃºsqueda).

### 8.4 Pipeline recomendado

```text
  +----------------+        +-------------------+        +-----------------+
  | Colab Pro+ GPU |  --->  | Drive _BAYESIAN_  |  --->  | PC local CPU    |
  | optimize_tr   |        |   OPT/<date>/     |        | train_trf       |
  | MPES_USE_GPU=1 |        | (NPZ+JSON+keras)  |        | --from-best     |
  +----------------+        +-------------------+        +-----------------+
```

1. **Colab**:

   ```bash
   export MPES_USE_GPU=1
   python -m pes_trf.ext.optimize_tr 60
   ```

2. **Sincronizar** el directorio `inputs/<date>_BAYESIAN_OPT/` al PC
   local vÃ­a Drive / `rsync`.
3. **PC local**:

   ```bash
   python -m pes_trf.ext.train_transformer --from-best <date>
   ```

   `train_transformer.py` lee `best_params_<date>.json`, fija el seed y
   reentrena con `TRF_EPISODES` completo en CPU determinista.

### 8.5 GarantÃ­as de reproducibilidad

- Mismo seed (`SEED = 42`) en NumPy, TensorFlow y Optuna.
- `tf.config.experimental.enable_op_determinism()` en CPU.
- `compute_confidence=False` durante entrenamiento (ver
  `explained_trf.md` Â§14.2).
- Las diferencias numÃ©ricas entre CPU y GPU son inherentes al hardware
  (orden de reducciones en cuDNN); por eso el modelo â€œde producciÃ³nâ€
  siempre se reentrena en la mÃ¡quina objetivo (CPU local) tras la
  bÃºsqueda.

> **Nota:** Los denominadores de normalizaciÃ³n se obtienen del propio
> entorno (`env.max_resources`, `env.max_seq_length`, `env.max_severity`),
> que son exactamente los valores usados durante `TRFTraining`.  Con la
> configuraciÃ³n actual ello equivale a $r_{\max} = 30$, $t_{\max} = 10$
> y $v_{\max} = 9$.  Ver `src/pygameMediator.py`.

---

## 9. Troubleshooting

### No se encuentra `trf_model.keras`

```
FileNotFoundError: trf_model.keras not found
```

**SoluciÃ³n**: Entrenar primero con `python -m pes_trf.ext.train_transformer`.

### `UnicodeEncodeError` en Windows

```
UnicodeEncodeError: 'charmap' codec can't encode characters
```

**SoluciÃ³n**: Establecer `$env:PYTHONIOENCODING = "utf-8"` antes de
ejecutar.

### TensorFlow muestra warnings excesivos

**SoluciÃ³n**: El paquete ya configura `TF_CPP_MIN_LOG_LEVEL=3` y
`TF_ENABLE_ONEDNN_OPTS=0` en `__init__.py`.  Si persisten, verificar que
las variables de entorno estÃ©n establecidas.

### OOM durante optimizaciÃ³n Bayesiana

Con buffers grandes (100k) y redes anchas (3 capas Ã— 128 unidades),
el consumo de memoria puede crecer.  El propio `objective()` libera el
grafo TF al final de cada trial (`tf.keras.backend.clear_session()` +
`gc.collect()`) para mitigar la fragmentaciÃ³n, pero ante OOM se puede:

- Reducir `buffer_size` mÃ¡ximo en el espacio de bÃºsqueda.
- Reducir el mÃ¡ximo de `hidden_layer_size` (Ã— `num_hidden_layers`).
- Ejecutar menos trials o en mÃ¡quinas con mÃ¡s RAM.

### El prompt "Press ENTER" aparece al importar

**SoluciÃ³n**: Establecer `VIRTUAL_ENV` antes de importar:

```powershell
$env:VIRTUAL_ENV = "$PWD\win_mpes_env"
```

### Resultados irreproducibles

TRF usa `SEED = 42` para NumPy, Python `random`, TF y â€”adicionalmenteâ€”
para un `numpy.random.Generator` **dedicado** a la rama aleatoria de
Îµ-greedy.  AdemÃ¡s, `transformer_model.py` fija un Ãºnico hilo intra/inter-op y
activa `tf.config.experimental.enable_op_determinism()` al importarse,
por lo que la ejecuciÃ³n en CPU es bit-a-bit reproducible.

> **AtenciÃ³n:** Si `TRFTraining()` se llama con `compute_confidence=True`
> (el default), las llamadas a `numpy.random.normal()` dentro de
> `trf_agent_meta_cognitive` consumen â‰ˆ1.5 M de nÃºmeros aleatorios
> adicionales durante 175 000 episodios, desplazando el RNG global.
> Esto hace que `env.random_sequence()` genere episodios distintos a los
> de la optimizaciÃ³n (que usa `compute_confidence=False`), resultando en
> un modelo diferente aunque el seed sea el mismo.
> **SoluciÃ³n:** Asegurar que `train_transformer.py` pase `compute_confidence=False`.

### NormalizaciÃ³n inconsistente entre entrenamiento e inferencia

```
Q-values muy similares entre sÃ­ (rango -75 a -113), agente siempre asigna 9-10
```

**Causa histÃ³rica**: en versiones anteriores de `pygameMediator.py` los
denominadores de `normalize_state()` se leÃ­an de constantes de `CONFIG`
que podÃ­an diverger de los valores efectivos del entorno.

**SoluciÃ³n actual**: el mÃ³dulo crea una instancia fresca de `Pandemic()`
y lee `env.max_resources`, `env.max_seq_length` y `env.max_severity`
directamente, garantizando que coincidan **siempre** con los usados en
`TRFTraining`:

| Componente | Entrenamiento (`pandemic.py`) | Inferencia (`pygameMediator.py`) |
|---|---|---|
| `max_resources` | `env.max_resources` = 30 | `AVAILABLE_RESOURCES_PER_SEQUENCE - 9` = 30 |
| `max_trials` | `env.max_seq_length` = 10 | `NUM_MAX_TRIALS` = 10 |
| `max_severity` | `env.max_severity` = 9 | `MAX_SEVERITY` = 9 |

---

## 10. Referencia RÃ¡pida

| Tarea | Comando |
|-------|---------|
| Entrenar (175k episodios) | `python -m pes_trf.ext.train_transformer` |
| Entrenar (N episodios) | `python -m pes_trf.ext.train_transformer N` |
| OptimizaciÃ³n Bayesiana (60 trials) | `python -m pes_trf.ext.optimize_tr` |
| OptimizaciÃ³n (100 trials) | `python -m pes_trf.ext.optimize_tr 100` |
| Reanudar optimizaciÃ³n | `python -m pes_trf.ext.optimize_tr 100 --resume YYYY-MM-DD` |
| Ejecutar experimento | `python -m pes_trf` |
