# pes_ql — Guía de Uso e Implementación

> Q-Learning tabular con optimización Bayesiana de hiperparámetros (Optuna/TPE)
> aplicado al **Pandemic Experiment Scenario**.

---

## Índice

1. [Descripción del paquete](#1-descripción-del-paquete)
2. [Estructura de directorios](#2-estructura-de-directorios)
3. [Comandos de ejecución](#3-comandos-de-ejecución)
4. [El entorno `Pandemic`](#4-el-entorno-pandemic)
5. [Estructura del experimento (bloques / secuencias / trials)](#5-estructura-del-experimento-bloques--secuencias--trials)
6. [Algoritmo Q-Learning implementado](#6-algoritmo-q-learning-implementado)
7. [Optimización Bayesiana (`optimize_rl.py`)](#7-optimización-bayesiana-optimize_rlpy)
8. [Entrenamiento (`train_rl.py`)](#8-entrenamiento-train_rlpy)
9. [Ejecución del experimento (`__main__.py`)](#9-ejecución-del-experimento-__main__py)
10. [Archivos de entrada y salida](#10-archivos-de-entrada-y-salida)
11. [Resultados de rendimiento](#11-resultados-de-rendimiento)
12. [Reproducibilidad](#12-reproducibilidad)
13. [Solución de problemas](#13-solución-de-problemas)

---

## 1. Descripción del paquete

`tabular/pes_ql` es la variante **Q-Learning tabular con búsqueda Bayesiana de
hiperparámetros** del proyecto mPES. Comparte el mismo entorno de simulación
(`Pandemic`) y la misma métrica de evaluación que `pes_base`, pero añade una
fase previa de optimización con **Optuna** que selecciona los cinco
hiperparámetros principales de Q-Learning antes de entrenar el agente final.

El pipeline completo se compone de tres pasos secuenciales:

| Paso | Comando | Salida |
|------|---------|--------|
| 1 | `python -m tabular.pes_ql.ext.optimize_rl [N]` | `best_params.json`, `q.npy`, `rewards.npy` |
| 2 | `python -m tabular.pes_ql.ext.train_rl [num_episodes]` | `q.npy`, `rewards.npy` (re-entrenado y verificado) |
| 3 | `python -m tabular.pes_ql` | Resultados del experimento (JSON, PNG, log) |

> El paso 2 **carga automáticamente** los parámetros del último directorio
> `inputs/<fecha>_BAYESIAN_OPT/` mediante [ext/repro.py](../ext/repro.py); no
> es necesario copiar archivos a mano.

---

## 2. Estructura de directorios

```
tabular/pes_ql/
├── __init__.py              # Re-exporta CONFIG, ANSI, INPUTS_PATH, etc.
├── __main__.py              # Punto de entrada del experimento
├── config/
│   └── CONFIG.py            # Todos los parámetros tunables
├── doc/
│   ├── pes_ql_explained.md  # Este documento
│   └── pes_ql_theory.md     # Fundamentos teóricos
├── ext/
│   ├── pandemic.py          # Entorno Gymnasium + QLearning() + run_experiment()
│   ├── train_rl.py          # Pipeline de entrenamiento
│   ├── optimize_rl.py       # Estudio Optuna (TPE + MedianPruner)
│   ├── repro.py             # Carga/validación de fingerprint reproducibilidad
│   ├── tools.py             # Helpers (entropía, conversión de secuencias, plots)
│   └── recover_optimization.py  # Re-genera reports desde la BD SQLite
├── inputs/
│   ├── q.npy                        # Q-table entrenada (cache)
│   ├── rewards.npy                  # Historial de recompensas
│   ├── best_params.json             # Hiperparámetros óptimos
│   ├── initial_severity.csv         # Severidades iniciales fijas (64 sec.)
│   ├── sequence_lengths.csv         # Longitud de cada secuencia
│   ├── <fecha>_BAYESIAN_OPT/        # Salida completa de optimize_rl
│   └── <fecha>_RL_TRAIN/            # Salida completa de train_rl
├── outputs/
│   └── <fecha>_QL_AGENT/            # Resultados del experimento
└── src/
    ├── exp_utils.py                 # Cálculo de severidades y métricas
    ├── log_utils.py                 # Logging dual (consola + archivo)
    ├── pygameMediator.py            # Bridge con la UI Pygame
    ├── result_formatter.py          # Plots de resultados con matplotlib
    └── terminal_utils.py            # Salida enriquecida (header, info, ...)
```

---

## 3. Comandos de ejecución

Todos los comandos asumen el directorio raíz del workspace
(`Win_mPES/`) y un entorno virtual activo
(`win_mpes_env\Scripts\Activate.ps1` en Windows o
`source linux_mpes_env/bin/activate` en Linux).

### 3.1. Optimización Bayesiana

```powershell
python -m tabular.pes_ql.ext.optimize_rl 100
# Reanudar un estudio existente:
python -m tabular.pes_ql.ext.optimize_rl 200 --resume 2026-04-22
# Directorio de salida personalizado (útil en Colab / runs paralelos):
python -m tabular.pes_ql.ext.optimize_rl 100 --out-dir /custom/path
# Backend de almacenamiento Optuna personalizado:
python -m tabular.pes_ql.ext.optimize_rl 100 --storage sqlite:////custom/study.db
```

### 3.2. Entrenamiento del agente

```powershell
# Auto-carga los hiperparámetros del último BAYESIAN_OPT:
python -m tabular.pes_ql.ext.train_rl
# Sobre-escribe num_episodes:
python -m tabular.pes_ql.ext.train_rl 1000000
```

### 3.3. Ejecución del experimento

```powershell
python -m tabular.pes_ql
```

---

## 4. El entorno `Pandemic`

Definido en [ext/pandemic.py](../ext/pandemic.py) como subclase de
`gymnasium.Env`. Modela un escenario de asignación de recursos limitados a
ciudades infectadas:

- **Estado** (`observation`): vector entero
  $s = [\text{available\_resources},\ \text{trial\_no},\ \text{severity}]$.
- **Acción**: número discreto de recursos a asignar en el trial actual,
  $a \in \{0, 1, \ldots, 10\}$ (`MAX_ALLOCATABLE_RESOURCES = 10`).
- **Recompensa**: $r = -\sum_i \text{severity}_i$ (negativo de la suma de
  severidades de todas las ciudades).
- **Transición de severidad**:

$$
\text{severity}' = \max\!\left(0,\; \beta \cdot \text{severity} - \alpha \cdot a\right)
$$

con $\alpha = 0.4$ (`PANDEMIC_PARAMETER`) y $\beta = 1 + \alpha = 1.4$.

### Forma de la Q-table

$$
Q \in \mathbb{R}^{31 \times 11 \times 10 \times 11} \;\;=\;\; 37{,}510 \text{ celdas}
$$

| Dimensión | Significado | Rango |
|-----------|-------------|-------|
| 0 | Recursos disponibles | 0 – 30 (el entorno reserva 9) |
| 1 | Número de trial dentro de la secuencia | 0 – 10 |
| 2 | Severidad observada | 0 – 9 (`MAX_SEVERITY = 9`) |
| 3 | Acción (recursos a asignar) | 0 – 10 |

> **Nota**: aunque `AVAILABLE_RESOURCES_PER_SEQUENCE = 39`, el entorno
> pre-asigna 9 recursos al iniciar la secuencia, dejando 30 controlables por
> el agente. De ahí el `31` (= 30 + 1) en la primera dimensión.

---

## 5. Estructura del experimento (bloques / secuencias / trials)

Configurada en [config/CONFIG.py](../config/CONFIG.py):

```
Experimento (1)
└── Bloque (8)         ← NUM_BLOCKS
    └── Secuencia (8)  ← NUM_SEQUENCES
        └── Trial (3-10) ← NUM_MIN_TRIALS .. NUM_MAX_TRIALS
            └── Decisión de recursos (0-10)
```

- 8 bloques × 8 secuencias por bloque × ~5.6 trials promedio ≈ **360 trials totales**.
- `TOTAL_NUM_TRIALS_IN_BLOCK = 45` impone que la suma de longitudes por
  bloque sea exactamente 45 (`NUM_ATTEMPTS_TO_ASSIGN_SEQ = 8` reintentos).
- Cuando `USE_FIXED_BLOCK_SEQUENCES = True`, las longitudes se leen de
  `inputs/sequence_lengths.csv` para garantizar reproducibilidad entre
  ejecuciones.

---

## 6. Algoritmo Q-Learning implementado

La función `QLearning` en [ext/pandemic.py](../ext/pandemic.py) sigue la
formulación clásica de Watkins (1989):

```python
delta = learning * (reward + discount * numpy.max(Q[s2]) - Q[s, a])
Q[s, a] += delta
```

es decir,

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \bigl[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \bigr].
$$

### Detalles de implementación

1. **Inicialización aleatoria**: `Q = numpy.random.uniform(-1, 1, shape)`.
   Romper simetría evita que `argmax` sea determinista al comienzo.
2. **Política ε-greedy**:

   ```python
   if numpy.random.random() < 1 - epsilon:
       action = numpy.argmax(Q[s])
   else:
       action = numpy.random.randint(0, env.action_space.n)
   ```
3. **Decaimiento lineal de ε**:
   $\epsilon \leftarrow \epsilon - \frac{\epsilon_0 - \epsilon_{\min}}{\text{episodes}}$
   tras cada episodio.
4. **Estado terminal**: $Q(s,a) \leftarrow r$ (sin término futuro).
5. **Reporte de progreso**: cada 10 000 episodios se imprime la recompensa
   media; `progress_callback` (usado por Optuna) puede abortar el entrenamiento.
6. **Reproducibilidad**: si `seed` no es `None`, se siembran `numpy.random`,
   `random` y `env.action_space`.

### Métrica de meta-cognición

`rl_agent_meta_cognitive` calcula una **confianza basada en entropía** de los
Q-valores (no afecta a la selección ni a la actualización). Se utiliza solo
para reportes humano-vs-agente en `__main__.py`. Durante optimización y
entrenamiento se desactiva con `track_confidence=False` para reproducir
exactamente los trials de Optuna y ahorrar tiempo de cómputo.

---

## 7. Optimización Bayesiana (`optimize_rl.py`)

### 7.1. Espacio de búsqueda

```python
learning_rate    = trial.suggest_float('learning_rate', 0.05, 0.40, log=True)
discount_factor  = trial.suggest_float('discount_factor', 0.85, 0.999)
epsilon_initial  = trial.suggest_float('epsilon_initial', 0.50, 1.00)
epsilon_min      = trial.suggest_float('epsilon_min', 0.01, 0.15)
num_episodes     = trial.suggest_int('num_episodes', 500_000, 1_200_000, step=50_000)
```

### 7.2. Configuración del estudio

```python
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=4),
    storage=f'sqlite:///{db_path}',
    load_if_exists=True,
)
```

- **Sampler**: TPE (Tree-structured Parzen Estimator) sembrado con `SEED = 42`.
- **Pruner**: `MedianPruner` aborta trials cuya recompensa media intermedia
  está por debajo de la mediana histórica tras 4 reportes (40 000 episodios).
- **Storage**: SQLite (`optuna_study_<fecha>.db`) → permite **reanudar** con
  `--resume YYYY-MM-DD`.

### 7.3. Función objetivo

Cada trial:

1. Muestrea hiperparámetros.
2. Crea un `Pandemic()`, lo configura con las distribuciones empíricas.
3. Llama a `QLearning(..., seed=SEED + trial.number + 1, track_confidence=False)`.
4. Evalúa la política voraz (con **enmascaramiento de acciones inviables** —
   acciones `> resources_left` reciben sentinela `-1e9` antes de `argmax`)
   sobre las **64 secuencias fijas**.
5. Retorna `mean_perf` clipeado a $[0, 1]$.

> Las semillas por trial son distintas (`SEED + trial.number + 1`) para que
> Optuna pueda estimar el ruido estocástico del objetivo.

### 7.4. Artefactos generados

En `inputs/<fecha>_BAYESIAN_OPT/`:

| Archivo | Descripción |
|---------|-------------|
| `q_best_<fecha>.npy` | Q-table del mejor trial |
| `rewards_best_<fecha>.npy` | Historial de recompensa del mejor trial |
| `best_params_<fecha>.json` | Hiperparámetros + `mean_perf` + `trial_seed` |
| `repro_fingerprint_<fecha>.json` | Hash NumPy/Python/CSV/Git/SEED |
| `_best_artifacts.npz` / `.json` | Backup pickle-free para `--resume` |
| `optimization_results_<fecha>.txt` | Reporte humano-legible |
| `optimization_history_<fecha>.png` | Curva de convergencia |
| `hyperparameter_importances_<fecha>.png` | Importancias fANOVA |
| `optuna_study_<fecha>.db` | Base de datos SQLite del estudio |

Adicionalmente, `q.npy`, `rewards.npy` y `best_params.json` se **espejan** a
`inputs/` para consumo directo.

---

## 8. Entrenamiento (`train_rl.py`)

El script [ext/train_rl.py](../ext/train_rl.py) implementa el siguiente
pipeline:

1. **Carga de datos**: `initial_severity.csv` y `sequence_lengths.csv`.
2. **Baseline aleatorio**: ejecuta `run_experiment()` con un agente que toma
   acciones uniformemente al azar; guarda dos PNGs.
3. **Auto-carga de hiperparámetros**: `repro.find_latest_artifacts()` busca
   el `<fecha>_BAYESIAN_OPT/` más reciente y carga `best_params*.json` +
   `repro_fingerprint*.json`.
4. **Verificación de fingerprint**: compara versiones de NumPy/Python, hash
   de los CSV, commit de Git y `SEED`. Cualquier divergencia se reporta como
   advertencia (no aborta).
5. **Fallback**: si no hay artefactos, usa `_FALLBACK_PARAMS` (trial #40
   histórico).
6. **Entrenamiento**: invoca `QLearning(..., seed=trial_seed,
   track_confidence=False)` con la misma semilla del trial Optuna ⇒ Q-table
   bit-a-bit idéntica.
7. **Evaluación**: política voraz con enmascaramiento sobre las 64 secuencias.
   Compara `local_mean_perf` contra `expected_perf` (de `best_params.json`).
8. **Persistencia**: escribe `q_<fecha>.npy`, `rewards_<fecha>.npy`,
   `training_config_<fecha>.txt` y plots en `inputs/<fecha>_RL_TRAIN/`.
   Sobrescribe `inputs/q.npy` y `inputs/rewards.npy`.

> El `track_confidence=False` es **obligatorio** para reproducibilidad: dejar
> el valor por defecto consume samples adicionales del RNG global y diverge
> el Q-table.

---

## 9. Ejecución del experimento (`__main__.py`)

Una vez generados `q.npy`, `rewards.npy` y `best_params.json`:

```powershell
python -m tabular.pes_ql
```

`__main__.py` orquesta:

- Validación de archivos de entrenamiento.
- Creación de sesión con logging dual (consola + `outputs/PES_QL_log_<fecha>.txt`).
- Asignación de bloques/secuencias/trials con la estructura de §5.
- Recogida de decisiones del agente RL vía `pygameMediator`.
- Cálculo de severidades actualizadas y métricas normalizadas.
- Generación de reportes JSON/PNG en `outputs/<fecha>_QL_AGENT/`.

---

## 10. Archivos de entrada y salida

### 10.1. Entradas (`inputs/`)

| Archivo | Contenido | Generador |
|---------|-----------|-----------|
| `initial_severity.csv` | 360 valores flotantes (severidades iniciales 0–9) | Manual / `pes_base` |
| `sequence_lengths.csv` | 64 enteros (3–10) con la longitud de cada secuencia | Manual / `pes_base` |
| `q.npy` | Q-table entrenada `(31, 11, 10, 11)` | `train_rl.py` / `optimize_rl.py` |
| `rewards.npy` | Recompensa promedio cada 10 000 episodios | `train_rl.py` / `optimize_rl.py` |
| `best_params.json` | Hiperparámetros óptimos + métricas | `optimize_rl.py` |

### 10.2. Salidas (`outputs/<fecha>_QL_AGENT/`)

- `PES_QL_log_<fecha>.txt` — log completo de la sesión.
- `PES_QL_results_<fecha>.json` — métricas por bloque/secuencia/trial.
- Plots de severidad final, performance normalizada y confianza
  (generados por `src/result_formatter.py`).

---

## 11. Resultados de rendimiento

La métrica principal es la **performance final normalizada por secuencia**,
calculada por
`calculate_normalised_final_severity_performance_metric()` en
[src/exp_utils.py](../src/exp_utils.py):

$$
P = \frac{\text{WorstCase} - \text{Achieved}}{\text{WorstCase} - \text{BestCase}} \in [0, 1]
$$

donde $1$ = óptimo, $0$ = peor caso (no asignar recursos).

### Mejor resultado (2026-04-30)

| Métrica | Valor |
|---------|-------|
| `raw_mean_perf` | **0.886640** |
| `std_perf` | 0.060781 |
| Secuencias evaluadas | 64 |
| `learning_rate` | ≈ 0.286 |
| `discount_factor` | ≈ 0.859 |
| `epsilon_initial` | ≈ 0.681 |
| `epsilon_min` | ≈ 0.044 |
| `num_episodes` | 550 000 |
| `trial_seed` | 106 (= 42 + 63 + 1) |

> Comparado con el baseline aleatorio (~0.50), Q-Learning con búsqueda
> Bayesiana mejora la performance en **+0.39** (≈78 % de mejora relativa).

### Cómo interpretar

- $P > 0.85$: política casi óptima en la mayoría de secuencias.
- $P \in [0.7, 0.85]$: política buena con ocasionales sub-asignaciones.
- $P < 0.5$: política deficiente; revisar si la Q-table convergió.

---

## 12. Reproducibilidad

El paquete implementa un **fingerprint** de entorno
([ext/repro.py](../ext/repro.py)) que registra:

- Versión exacta de Python y NumPy.
- Hash SHA-256 de `initial_severity.csv` y `sequence_lengths.csv`.
- Commit Git actual del repositorio.
- Valor de `SEED`.

Cuando `train_rl.py` carga un sidecar de Optuna, compara este fingerprint
contra el guardado y reporta cualquier diferencia. La igualdad bit-a-bit
de `mean_perf` requiere:

1. Mismo `SEED`.
2. Mismo `trial_seed` (= `SEED + trial_number + 1`).
3. `track_confidence=False`.
4. Mismos CSVs de entrada (mismo hash).
5. Misma versión de NumPy/Python.

---

## 13. Solución de problemas

| Síntoma | Causa probable | Solución |
|---------|----------------|----------|
| `mean_perf` diverge de `expected_mean_perf` | Diferencia de fingerprint | Revisar advertencias amarillas de `train_rl.py` |
| `FileNotFoundError: best_params.json` | No se ha corrido `optimize_rl.py` | Ejecutar el paso 1 o usar `_FALLBACK_PARAMS` |
| `optuna_study_*.db locked` | Dos procesos escribiendo simultáneamente | Cerrar el otro proceso o usar `--storage` distinto |
| Q-table con valores NaN | Descomposición por reward extremo | Verificar `PANDEMIC_PARAMETER` y rango de severidades |
| `UnicodeEncodeError` en stdout | Windows cp1252 | `set PYTHONIOENCODING=utf-8` antes de ejecutar |
| Entrenamiento muy lento | `track_confidence=True` por error | Asegurar `False` en optimización |

---

## Referencias internas

- [config/CONFIG.py](../config/CONFIG.py) — todas las constantes.
- [ext/pandemic.py](../ext/pandemic.py) — entorno + `QLearning()` + `run_experiment()`.
- [ext/optimize_rl.py](../ext/optimize_rl.py) — estudio Optuna.
- [ext/train_rl.py](../ext/train_rl.py) — pipeline de entrenamiento.
- [ext/repro.py](../ext/repro.py) — fingerprint + auto-load de artefactos.
- [src/exp_utils.py](../src/exp_utils.py) — métricas y dinámica de severidad.

Para los fundamentos teóricos, ver
[pes_ql_theory.md](pes_ql_theory.md).
