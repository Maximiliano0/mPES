# pes_dql — Guía de uso e implementación

> Paquete: **`tabular.pes_dql`** — Double Q-Learning con calentamiento (warm-up)
> exponencial de ε y *Potential-Based Reward Shaping* (PBRS).
>
> Última actualización: 2026-05-04

---

## Índice

1. [Qué hace el paquete](#1-qué-hace-el-paquete)
2. [Cómo usarlo (CLI)](#2-cómo-usarlo-cli)
3. [Cómo entrenar el modelo](#3-cómo-entrenar-el-modelo)
4. [Cómo optimizar el modelo (Bayesiano)](#4-cómo-optimizar-el-modelo-bayesiano)
5. [Referencias de código](#5-referencias-de-código)
6. [Estructura de directorios](#6-estructura-de-directorios)
7. [Archivos de entrada y salida](#7-archivos-de-entrada-y-salida)
8. [Resultados de rendimiento](#8-resultados-de-rendimiento)
9. [Diferencias frente a `pes_ql`](#9-diferencias-frente-a-pes_ql)

---

## 1. Qué hace el paquete

`pes_dql` implementa la versión mejorada del agente tabular de Q-Learning
para el escenario "Pandemic". Combina **tres mejoras algorítmicas** sobre
el Q-Learning estándar de `pes_base` y `pes_ql`:

1. **Double Q-Learning** (Van Hasselt, 2010) — dos tablas Q
   independientes (`Q_A`, `Q_B`) que eliminan el sesgo de maximización del
   operador `max`.
2. **ε-decay con warm-up** — ε se mantiene constante en su valor inicial
   ε₀ durante una fracción inicial (`warmup_ratio`) de los episodios y
   después decae exponencialmente hasta ε_min. Evita explotar prematuramente
   antes de que las Q-tables tengan estimaciones razonables.
3. **PBRS — Potential-Based Reward Shaping** (Ng, Harada, & Russell, 1999) —
   suma un término de modelado de recompensa
   $F(s, s') = \beta\,(\gamma\,\Phi(s') - \Phi(s))$ con potencial
   $\Phi(s) = -\sum_i s_i$. Acelera la convergencia sin alterar la política
   óptima.

El entorno **Pandemic** se modela como un MDP discreto en
[`tabular/pes_dql/ext/pandemic.py`](../ext/pandemic.py):

| Componente | Detalle |
|---|---|
| Estados | `(31, 11, 10)` = 3 410 estados |
| Acciones | 11 (asignar 0–10 recursos) |
| Recompensa | $r_t = -\sum_i s_i$ |
| Transición | $s'_i = \max(0,\; 1.4\,s_i - 0.4\,a_i)$ |
| Recursos por secuencia | 39 (`AVAILABLE_RESOURCES_PER_SEQUENCE`) |
| Estructura | 8 bloques × 8 secuencias × 3–10 trials |

La forma de la Q-table es `(31, 11, 10, 11)` = 37 510 celdas por tabla
(≈ 293 KB en `float64`). Con Double Q-Learning se almacenan dos tablas en
memoria durante el entrenamiento (≈ 586 KB).

---

## 2. Cómo usarlo (CLI)

Todos los comandos asumen que la raíz del workspace
(`c:\Users\maxvega\Documents\Win_mPES`) es el directorio actual y que el
entorno virtual está activado.

### 2.1 Activar el entorno virtual

**Windows (PowerShell):**

```powershell
win_mpes_env\Scripts\Activate.ps1
```

**Linux:**

```bash
source linux_mpes_env/bin/activate
```

### 2.2 Comandos principales

| Acción | Comando |
|---|---|
| Ejecutar el experimento completo (8 × 8 = 64 secuencias) | `python -m tabular.pes_dql` |
| Entrenar el agente (Double Q-Learning) | `python -m tabular.pes_dql.ext.train_rl` |
| Entrenar con N episodios personalizados | `python -m tabular.pes_dql.ext.train_rl 1000000` |
| Lanzar la optimización bayesiana (50 trials por defecto) | `python -m tabular.pes_dql.ext.optimize_rl` |
| Lanzar la optimización con N trials | `python -m tabular.pes_dql.ext.optimize_rl 100` |
| Reanudar una optimización previa | `python -m tabular.pes_dql.ext.optimize_rl 100 --resume 2026-04-21` |

### 2.3 Variables de entorno recomendadas

Para evitar errores de codificación cuando la salida se redirige a fichero:

**Windows (PowerShell):**

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:VIRTUAL_ENV      = "$PWD\win_mpes_env"
$env:TF_ENABLE_ONEDNN_OPTS = "0"
```

**Linux:**

```bash
export PYTHONIOENCODING=utf-8
export VIRTUAL_ENV="$PWD/linux_mpes_env"
export TF_ENABLE_ONEDNN_OPTS=0
```

---

## 3. Cómo entrenar el modelo

El script de entrenamiento es
[`tabular/pes_dql/ext/train_rl.py`](../ext/train_rl.py). Se invoca con:

```bash
python -m tabular.pes_dql.ext.train_rl [num_episodes]
```

### 3.1 Etapas del *pipeline* (`train_rl.py`)

`train_rl.py` ejecuta secuencialmente las siguientes fases (cada fase aparece
delimitada en el log con cabeceras de `terminal_utils`):

1. **Resolución de hiperparámetros**
   (`_load_best_params()`).
   Orden de precedencia:
   1. `inputs/best_params.json` (mirroreado por `optimize_rl.py`).
   2. El `best_params_*.json` más reciente bajo
      `inputs/<fecha>_BAYESIAN_OPT/`.
   3. El diccionario `_DEFAULT_HYPERPARAMS` codificado en el script.

2. **Carga de datos**
   Lee `INITIAL_SEVERITY_FILE` (severidades iniciales) y
   `SEQ_LENGTHS_FILE` (longitudes de secuencia) desde `inputs/`.

3. **Baseline aleatorio**
   Ejecuta las 64 secuencias con un agente que asigna recursos
   uniformemente al azar. Sirve como referencia inferior.

4. **Entrenamiento Double Q-Learning**
   Llama a `QLearning(env, learning, discount, epsilon, min_eps,
   episodes, warmup_ratio, target_ratio, double_q=True,
   penalty_coeff=...)` desde
   [`pandemic.py`](../ext/pandemic.py). Por defecto entrena durante
   **860 000 episodios** con `SEED=42`.

5. **Evaluación dual**
   Ejecuta dos pasadas sobre las 64 secuencias fijas:
   - **Determinista** (`qf_eval`) — replica bit a bit la función
     `objective` de `optimize_rl.py` (mismo enmascaramiento `-1e9`,
     mismo *clamping*). Su `mean_perf` debe coincidir con el valor
     guardado en el mejor trial bayesiano (`abs(delta) < 1e-9`).
   - **Estocástica** (`qf`) — usa `rl_agent_meta_cognitive` para recoger
     entropías y tiempos de respuesta para los plots.

6. **Persistencia de artefactos**
   Escribe la Q-table, el historial de recompensas, el reporte de
   configuración y todas las visualizaciones en
   `inputs/<YYYY-MM-DD>_RL_TRAIN/`. Adicionalmente *mirrorea* la mejor
   Q-table y `rewards.npy` a `inputs/q.npy` y `inputs/rewards.npy`,
   que son los ficheros que consume `python -m tabular.pes_dql`.

### 3.2 Hiperparámetros por defecto

Los valores codificados en `_DEFAULT_HYPERPARAMS` (resultado de la
optimización bayesiana del 2026-04-21):

| Parámetro | Valor | Descripción |
|---|---|---|
| `learning_rate` (α) | 0.2593 | Tasa de aprendizaje |
| `discount_factor` (γ) | 0.9806 | Factor de descuento |
| `epsilon_initial` (ε₀) | 0.8392 | Exploración inicial |
| `epsilon_min` (ε_min) | 0.0799 | Exploración mínima |
| `warmup_ratio` | 0.0240 | Fracción de episodios en warm-up |
| `target_ratio` | 0.5174 | Fracción de episodios para alcanzar ε_min |
| `penalty_coeff` (β) | 0.2177 | Coeficiente PBRS |
| `num_episodes` | 860 000 | Episodios totales |
| `seed` | 42 | Semilla global |

### 3.3 Salidas del entrenamiento

Bajo `tabular/pes_dql/inputs/<YYYY-MM-DD>_RL_TRAIN/`:

| Fichero | Descripción |
|---|---|
| `q_<date>.npy` | Q-table entrenada `(31, 11, 10, 11)` |
| `rewards_<date>.npy` | Recompensa media cada 10 000 episodios |
| `training_config_<date>.txt` | Resumen de hiperparámetros |
| `confsrl_<date>.npy` | Confianzas de la pasada estocástica |
| `random_player_*_<date>.png` | Plots del baseline aleatorio |
| `rl_agent_rewards_vs_episodes_<date>.png` | Curva de aprendizaje |
| `rl_agent_sequence_performance_<date>.png` | Severidad por secuencia |
| `rl_agent_normalised_performance_<date>.png` | Rendimiento normalizado |
| `rl_agent_cumulative_performance_<date>.png` | Rendimiento acumulado |
| `rl_agent_confidences_<date>.png` | Confianza por trial |
| `rl_agent_remapped_confidences_<date>.png` | Confianza re-escalada |

Y los enlaces canónicos para el experimento:

- `tabular/pes_dql/inputs/q.npy`
- `tabular/pes_dql/inputs/rewards.npy`

---

## 4. Cómo optimizar el modelo (Bayesiano)

El script de optimización es
[`tabular/pes_dql/ext/optimize_rl.py`](../ext/optimize_rl.py). Usa
**Optuna 4.7.0** con el sampler `TPESampler` (Akiba et al., 2019).

### 4.1 Invocación

```bash
python -m tabular.pes_dql.ext.optimize_rl [n_trials] [--resume YYYY-MM-DD]
                                          [--out-dir PATH] [--storage URL]
```

Argumentos:

- `n_trials` — número de trials (por defecto 50).
- `--resume YYYY-MM-DD` — reanuda un estudio Optuna previo.
- `--out-dir PATH` — sobrescribe el directorio
  `inputs/<fecha>_BAYESIAN_OPT/`.
- `--storage URL` — sobrescribe el SQLite por defecto
  (`sqlite:///<opt_dir>/optuna_study_<date>.db`).

### 4.2 Espacio de búsqueda

Definido en la función `objective(trial)`:

| Parámetro | Rango | Escala |
|---|---|---|
| `learning_rate` | `[0.05, 0.30]` | log |
| `discount_factor` | `[0.90, 0.999]` | lineal |
| `epsilon_initial` | `[0.50, 1.00]` | lineal |
| `epsilon_min` | `[0.01, 0.10]` | log |
| `num_episodes` | `[150 000, 500 000]` | lineal (paso 10 000) |
| `warmup_ratio` | `[0.02, 0.15]` | lineal |
| `target_ratio` | `[0.40, 0.80]` | lineal |
| `penalty_coeff` (β) | `[1e-4, 0.30]` | log |

El sampler se configura como
`TPESampler(seed=SEED, n_startup_trials=10, multivariate=True, group=True)`.

Cada trial se entrena con la semilla `SEED + trial.number + 1`, lo que
asegura **reproducibilidad individual** de cada trial sin colapsar todos a
la misma trayectoria estocástica.

### 4.3 Función objetivo

Para cada trial:

1. Se entrena `QLearning(...)` con los hiperparámetros propuestos,
   `track_confidence=False` (más rápido) y `double_q=True`.
2. Se evalúa la Q-table sobre las 64 secuencias fijas con
   enmascaramiento de acciones inviables.
3. Se devuelve el `mean_perf` (rendimiento normalizado promedio) que
   Optuna intenta **maximizar**.
4. La mejor Q-table y su historial de recompensas se conservan en
   memoria (`_best_artifacts`) para evitar un re-entrenamiento "lossy".

### 4.4 Salidas de la optimización

Bajo `tabular/pes_dql/inputs/<YYYY-MM-DD>_BAYESIAN_OPT/`:

| Fichero | Descripción |
|---|---|
| `optuna_study_<date>.db` | Base de datos SQLite (resumible) |
| `q_best_<date>.npy` | Q-table del mejor trial |
| `rewards_best_<date>.npy` | Recompensas del mejor trial |
| `best_params_<date>.json` | Hiperparámetros + `trial_seed` + `mean_perf` |
| `optimization_results_<date>.txt` | Reporte completo (índices 1-based) |
| `optimization_history_<date>.png` | Curva de convergencia |
| `hyperparameter_importances_<date>.png` | Importancia de parámetros |

Adicionalmente se mirrorea `best_params_<date>.json` →
`inputs/best_params.json` para que `train_rl.py` lo recoja en su próxima
ejecución.

---

## 5. Referencias de código

### 5.1 Funciones públicas en `pandemic.py`

| Símbolo | Descripción |
|---|---|
| `class Pandemic(Env)` | Entorno Gymnasium (espacios + dinámica). |
| `Pandemic.step(action)` | Transición $s \to s'$ y recompensa. |
| `Pandemic.reset(seed, options)` | Reinicia la secuencia. |
| `QLearning(env, learning, discount, epsilon, min_eps, episodes, warmup_ratio, target_ratio, double_q, penalty_coeff, ...)` | Entrenamiento (con o sin Double Q, warm-up y PBRS). |
| `run_experiment(env, actionfunction, ...)` | Evaluación greedy sobre las 64 secuencias. |
| `rl_agent_meta_cognitive(options, resources_left, response_timeout)` | Estimación de confianza por entropía. |

### 5.2 Constantes clave (`config/CONFIG.py`)

| Constante | Valor | Uso |
|---|---|---|
| `AVAILABLE_RESOURCES_PER_SEQUENCE` | 39 | Presupuesto por secuencia |
| `MAX_ALLOCATABLE_RESOURCES` | 10 | Acción máxima |
| `MAX_SEVERITY` | 9 | Estado de severidad máximo |
| `NUM_BLOCKS` | 8 | Bloques del experimento |
| `NUM_SEQUENCES` | 8 | Secuencias por bloque |
| `NUM_MIN_TRIALS` / `NUM_MAX_TRIALS` | 3 / 10 | Trials por secuencia |
| `PANDEMIC_PARAMETER` | 0.4 | α (efectividad de recursos) |
| `SEED` | 42 | Semilla global |
| `OUTPUT_FILE_PREFIX` | `'PES_DQL_'` | Prefijo de logs |

### 5.3 Módulos de soporte (`src/`)

- `exp_utils.py` — `get_updated_severity()`,
  `calculate_normalised_final_severity_performance_metric()`.
- `log_utils.py` — duplicación de `stdout` a fichero.
- `pygameMediator.py` — interfaz visual (no usada en modo agente).
- `result_formatter.py` — generación de plots con Matplotlib.
- `terminal_utils.py` — `header`, `section`, `info`, `success`,
  `list_item`.

---

## 6. Estructura de directorios

```
tabular/pes_dql/
├── __init__.py            # Re-exporta CONFIG y constantes ANSI
├── __main__.py            # Entry point del experimento
├── config/
│   └── CONFIG.py          # Todos los parámetros tuneables
├── doc/
│   ├── how_to_train_and_test.md     # Guía operativa
│   ├── mejoras_qlearning.md         # Teoría de las 3 mejoras
│   ├── pes_dql_explained.md         # Este documento
│   └── pes_dql_theory.md            # Documento teórico complementario
├── ext/
│   ├── pandemic.py        # Env Gym + QLearning() (Double Q + PBRS)
│   ├── train_rl.py        # Pipeline de entrenamiento
│   ├── optimize_rl.py     # Optimización bayesiana (Optuna)
│   ├── repro.py           # Fingerprints de reproducibilidad
│   └── tools.py           # Utilidades (entropy_from_pdf, plots)
├── inputs/
│   ├── q.npy              # Q-table canónica (consumida por __main__)
│   ├── rewards.npy        # Recompensas canónicas
│   ├── best_params.json   # Hiperparámetros bayesianos
│   ├── initial_severity.csv
│   ├── sequence_lengths.csv
│   ├── <date>_RL_TRAIN/   # Salidas datadas de train_rl.py
│   └── <date>_BAYESIAN_OPT/  # Salidas datadas de optimize_rl.py
├── outputs/
│   └── <date>_DQL_AGENT/  # Logs y resultados del experimento
└── src/
    ├── exp_utils.py
    ├── log_utils.py
    ├── pygameMediator.py
    ├── result_formatter.py
    └── terminal_utils.py
```

---

## 7. Archivos de entrada y salida

### 7.1 Entradas requeridas

| Fichero | Generado por | Forma |
|---|---|---|
| `inputs/initial_severity.csv` | Manual / experimento | Matriz de severidades iniciales |
| `inputs/sequence_lengths.csv` | Manual / experimento | Longitudes de secuencia |
| `inputs/q.npy` | `train_rl.py` u `optimize_rl.py` | `(31, 11, 10, 11)` |
| `inputs/rewards.npy` | Idem | Vector |
| `inputs/best_params.json` | `optimize_rl.py` | JSON con hiperparámetros |

### 7.2 Estructura de `best_params.json`

```json
{
    "hyperparameters": {
        "learning_rate":   0.2593,
        "discount_factor": 0.9806,
        "epsilon_initial": 0.8392,
        "epsilon_min":     0.0799,
        "num_episodes":    860000,
        "warmup_ratio":    0.0240,
        "target_ratio":    0.5174,
        "penalty_coeff":   0.2177
    },
    "trial_seed":       143,
    "track_confidence": false,
    "double_q":         true,
    "mean_perf":        0.896344,
    "trial_number":     100
}
```

### 7.3 Salidas del experimento (`__main__.py`)

`python -m tabular.pes_dql` produce:

- `outputs/<date>_DQL_AGENT/` con resultados por secuencia (JSON/TXT)
  y plots agregados.
- `outputs/PES_DQL_log_<date>_DQL_AGENT.txt` con el log completo.

---

## 8. Resultados de rendimiento

La métrica de rendimiento es la **severidad final normalizada** (más alto =
mejor; 1.0 sería un agente perfecto que reduce toda la severidad a cero):

$$\text{perf}(\text{seq}) = 1 - \frac{\text{severidad\_final}}{\text{severidad\_máxima\_posible}}$$

### 8.1 Última corrida (2026-04-30)

| Métrica | Valor |
|---|---|
| `raw_mean_perf` | **0.896344** |
| `std` | 0.047708 |
| `n` (secuencias) | 64 |
| Episodios | 860 000 |
| Semilla | 143 (`SEED + best_trial_number + 1`) |

### 8.2 Interpretación

- Un valor de **0.896** indica que el agente reduce la severidad final
  al ~10.4 % del peor caso posible. Comparado con `pes_ql` (Q-Learning
  estándar con la misma optimización bayesiana), Double Q + PBRS aporta
  una mejora estadísticamente consistente atribuible a:
  - Eliminación del sesgo de maximización (Double Q).
  - Mejor distribución temporal de exploración (warm-up).
  - Convergencia más rápida sin alterar la política óptima (PBRS).
- La desviación estándar (0.048) sobre 64 secuencias indica buena
  consistencia entre mapas de distinta longitud.

---

## 9. Diferencias frente a `pes_ql`

| Aspecto | `pes_ql` | `pes_dql` |
|---|---|---|
| Algoritmo | Q-Learning estándar | **Double** Q-Learning |
| Q-tables | 1 | 2 (`Q_A`, `Q_B`) |
| ε-decay | Lineal o exponencial simple | Exponencial **con warm-up** |
| Reward shaping | Ninguno | **PBRS** con Φ(s) = -Σ s_i |
| Hiperparámetros bayesianos | 5 (α, γ, ε₀, ε_min, episodes) | **8** (+ `warmup_ratio`, `target_ratio`, `penalty_coeff`) |
| Sesgo de maximización | Presente (sobreestima Q) | Eliminado |
| Convergencia empírica | Lenta y ruidosa al inicio | Más rápida (PBRS) y estable (Double Q) |
| Política óptima | Idéntica (PBRS preserva π*) | Idéntica (garantía de Ng et al., 1999) |

### Por qué cada mejora

1. **Double Q-Learning (Van Hasselt, 2010)** corrige que el operador `max`
   en Q-Learning estándar acopla *selección* y *evaluación* en la misma
   tabla ruidosa, lo que produce Q-valores inflados (sesgo positivo).
   Dos tablas independientes desacoplan ambas operaciones y cancelan el
   sesgo en promedio.

2. **Warm-up de ε** evita un problema observado en `pes_ql`: cuando ε
   empieza a decaer desde el episodio 1, el agente comienza a explotar
   estimaciones Q que aún son esencialmente ruido aleatorio. Mantener
   ε₀ durante un 2–5 % inicial garantiza que las primeras explotaciones
   se basen en algo aprendido.

3. **PBRS (Ng et al., 1999)** acelera la convergencia sin cambiar la
   política óptima: la teoría demuestra que cualquier shaping de la forma
   $F(s, s') = \gamma\,\Phi(s') - \Phi(s)$ es **invariante de política**.
   El potencial elegido $\Phi(s) = -\sum_i s_i$ guía al agente hacia
   estados con menor severidad acumulada.

Para el detalle teórico completo, véase
[`pes_dql_theory.md`](./pes_dql_theory.md) y la documentación previa
[`mejoras_qlearning.md`](./mejoras_qlearning.md).
