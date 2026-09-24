# dql_conf — Guía de uso e implementación

> Paquete suspendido en `h2/tabular_conf/dql_conf`. Este documento describe lo
> que realmente está implementado en el código actual de `__main__.py`,
> `config/CONFIG.py`, `ext/pandemic.py`, `ext/tools.py`, `ext/train_rl.py` y
> `ext/optimize_rl.py`.

## 1. Qué implementa este paquete

`dql_conf` es una variante experimental de Q-Learning tabular para el entorno
`Pandemic` con estos elementos activos:

- `Double Q-Learning` sobre dos tablas independientes `Q_A` y `Q_B`.
- `Warm-up` inicial y decaimiento exponencial de `epsilon`.
- `PBRS` sobre un potencial basado en la severidad acumulada.
- `Exploración modular por confianza` según la incertidumbre de los valores Q
  de las acciones factibles.

El paquete no forma parte del benchmark activo; su propósito es dejar un
referente reproducible para la línea experimental `h2/`.

## 2. Estructura real del paquete

```text
h2/tabular_conf/dql_conf/
├── __init__.py
├── __main__.py
├── config/
│   └── CONFIG.py
├── doc/
│   ├── dql_conf_explained.md     # documento de referencia del paquete
│   └── dql_conf_theory.md        # teoría matemática del agente
├── ext/
│   ├── pandemic.py               # entorno y actualización de Q
│   ├── tools.py                  # confianza y utilidades de evaluación
│   ├── optimize_rl.py            # Optuna / TPE
│   ├── train_rl.py               # entrenamiento y persistencia
│   ├── repro.py                  # reproducibilidad / validación
│   └── recover_optimization.py   # recuperación de optimizaciones
├── inputs/
├── outputs/
└── src/
    ├── exp_utils.py
    ├── log_utils.py
    ├── pygameMediator.py
    ├── result_formatter.py
    └── terminal_utils.py
```

## 3. Comandos reales

Se ejecutan desde la raíz del repositorio con el entorno virtual de Windows
activo:

```powershell
cd h2
python -m tabular_conf.dql_conf
python -m tabular_conf.dql_conf.ext.train_rl
python -m tabular_conf.dql_conf.ext.train_rl 1000000
python -m tabular_conf.dql_conf.ext.optimize_rl
python -m tabular_conf.dql_conf.ext.optimize_rl 100
python -m tabular_conf.dql_conf.ext.optimize_rl 100 --resume 2026-09-07
```

`optimize_rl.py` admite además `--out-dir PATH` y `--storage URL`.

## 4. Puntos relevantes del código

| Módulo | Función principal |
|---|---|
| `__main__.py` | Entrada del experimento y validación del artefacto `q.npy` |
| `config/CONFIG.py` | Parámetros de entrenamiento, semilla y configuraciones del entorno |
| `ext/pandemic.py` | Entorno `Pandemic`, pasos de decisión y actualización de Q |
| `ext/tools.py` | Cálculo de `confidence_from_q_values` y utilidades auxiliares |
| `ext/train_rl.py` | Entrenamiento y persistencia de `q.npy`, `rewards.npy` y artefactos |
| `ext/optimize_rl.py` | Optimización bayesiana con Optuna |

## 5. Qué no debe leerse en este documento

- No son resultados activos de benchmark ni métricas certificadas.
- No hay un flujo de comparación en `h2/general` que se use para reportar
  resultados oficiales.
- La documentación de esta línea es de referencia y no reemplaza la
  validación del benchmark activo en `h1/`.

## 6. Resumen breve

La variante `dql_conf` combina Double Q-Learning, warm-up, PBRS y una política
`epsilon-greedy` ajustada por confianza para rebalancear exploración y
explotación. El paquete queda documentado como trabajo experimental suspendido,
no como ejecución activa del proyecto.

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
[`ext/pandemic.py`](../ext/pandemic.py):

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

Todos los comandos se ejecutan desde `h2/`, con el entorno virtual activado
desde la raíz del repositorio.

### 2.1 Activar el entorno virtual

**Windows (PowerShell):**

```powershell
win_mpes_env\Scripts\Activate.ps1
```

### 2.2 Comandos principales

| Acción | Comando |
|---|---|
| Ejecutar el experimento completo (8 × 8 = 64 secuencias) | `python -m tabular_conf.dql_conf` |
| Entrenar el agente (Double Q-Learning) | `python -m tabular_conf.dql_conf.ext.train_rl` |
| Entrenar con N episodios personalizados | `python -m tabular_conf.dql_conf.ext.train_rl 1000000` |
| Lanzar la optimización bayesiana (50 trials por defecto) | `python -m tabular_conf.dql_conf.ext.optimize_rl` |
| Lanzar la optimización con N trials | `python -m tabular_conf.dql_conf.ext.optimize_rl 100` |
| Reanudar una optimización previa | `python -m tabular_conf.dql_conf.ext.optimize_rl 100 --resume 2026-09-07` |

### 2.3 Variables de entorno recomendadas

Para evitar errores de codificación cuando la salida se redirige a fichero:

**Windows (PowerShell):**

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:VIRTUAL_ENV      = "$PWD\win_mpes_env"
$env:TF_ENABLE_ONEDNN_OPTS = "0"
```

---

## 3. Cómo entrenar el modelo

El script de entrenamiento es
[`ext/train_rl.py`](../ext/train_rl.py). Se invoca con:

```powershell
python -m tabular_conf.dql_conf.ext.train_rl [num_episodes]
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
   [`pandemic.py`](../ext/pandemic.py). Con el `inputs/best_params.json`
   actual entrena **320 000 episodios** con semilla 83; sin ese archivo cae
   a 860 000 episodios con `SEED=42`.

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
   que son los ficheros que consume `python -m tabular_conf.dql_conf`.

### 3.2 Hiperparámetros

`inputs/best_params.json` (optimización del 2026-09-07, ensayo #40,
`mean_perf` = 0.8769) tiene prioridad. `_DEFAULT_HYPERPARAMS` es sólo el
respaldo heredado cuando no hay ningún `best_params`:

| Parámetro | `best_params.json` | `_DEFAULT_HYPERPARAMS` | Descripción |
|---|---:|---:|---|
| `learning_rate` (α) | 0.1779 | 0.2593 | Tasa de aprendizaje |
| `discount_factor` (γ) | 0.9390 | 0.9806 | Factor de descuento |
| `epsilon_initial` (ε₀) | 0.5127 | 0.8392 | Exploración inicial |
| `epsilon_min` (ε_min) | 0.0665 | 0.0799 | Exploración mínima |
| `warmup_ratio` | 0.1221 | 0.0240 | Fracción de episodios en warm-up |
| `target_ratio` | 0.6665 | 0.5174 | Fracción de episodios para alcanzar ε_min |
| `penalty_coeff` (β) | 0.00028 | 0.2177 | Coeficiente PBRS |
| `confidence_exploration_strength` | 0.1047 | 0.0 | Intensidad de la exploración por confianza |
| `confidence_exploration_exponent` | 0.5838 | 1.0 | Exponente de la exploración por confianza |
| `num_episodes` | 320 000 | 860 000 | Episodios totales |
| `seed` | 83 | 42 | Semilla |

### 3.3 Salidas del entrenamiento

Bajo `h2/tabular_conf/dql_conf/inputs/<YYYY-MM-DD>_RL_TRAIN/`:

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

- `h2/tabular_conf/dql_conf/inputs/q.npy`
- `h2/tabular_conf/dql_conf/inputs/rewards.npy`

---

## 4. Cómo optimizar el modelo (Bayesiano)

El script de optimización es
[`ext/optimize_rl.py`](../ext/optimize_rl.py). Usa
**Optuna 4.7.0** con el sampler `TPESampler` (Akiba et al., 2019).

### 4.1 Invocación

```powershell
python -m tabular_conf.dql_conf.ext.optimize_rl [n_trials] [--resume YYYY-MM-DD]
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
| `confidence_exploration_strength` | `[0.0, 0.60]` | lineal |
| `confidence_exploration_exponent` | `[0.50, 3.0]` | lineal |

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

Bajo `h2/tabular_conf/dql_conf/inputs/<YYYY-MM-DD>_BAYESIAN_OPT/`:

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
tabular_conf/dql_conf/
├── __init__.py            # Re-exporta CONFIG y constantes ANSI
├── __main__.py            # Entry point del experimento
├── config/
│   └── CONFIG.py          # Todos los parámetros tuneables
├── doc/
│   ├── dql_conf_explained.md        # Este documento
│   └── dql_conf_theory.md           # Documento teórico complementario
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
│   └── <date>_RL_AGENT/  # Logs y resultados del experimento
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
  "opt_date": "2026-09-07",
  "best_trial_number": 40,
  "mean_perf": 0.8769217142772177,
  "std_perf": 0.07137749680630046,
  "min_perf": 0.6475238443963114,
  "max_perf": 1.0,
  "trial_seed": 83,
  "seed": 83,
  "track_confidence": false,
  "double_q": true,
  "hyperparameters": {
    "learning_rate": 0.17794514190358618,
    "discount_factor": 0.9389572361956726,
    "epsilon_initial": 0.5127360505349272,
    "epsilon_min": 0.0664517056611224,
    "num_episodes": 320000,
    "warmup_ratio": 0.12211509070207466,
    "target_ratio": 0.6664534124314193,
    "penalty_coeff": 0.00027782590226948346,
    "confidence_exploration_strength": 0.10468365524328493,
    "confidence_exploration_exponent": 0.5837669008883881
  }
}
```

### 7.3 Salidas del experimento (`__main__.py`)

`python -m tabular_conf.dql_conf` produce:

- `outputs/<date>_RL_AGENT/` con resultados por secuencia (JSON/TXT)
  y plots agregados.
- `outputs/<date>_RL_AGENT/PES_DQL_log_<date>_RL_AGENT.txt` con el log completo.

---

## 8. Resultados de rendimiento

La métrica es el **rendimiento normalizado** de
[`exp_utils.py`](../src/exp_utils.py) (más alto = mejor):

$$\text{perf} = \frac{S_{\text{peor}} - S_{\text{final}}}{S_{\text{peor}} - S_{\text{mejor}}}$$

donde $S_{\text{peor}}$ es la severidad total sin asignar recursos y
$S_{\text{mejor}}$ la de la asignación factible óptima (programación
dinámica exacta, `_best_feasible_sequence_severity`). 1.0 equivale a
jugar esa política óptima.

### 8.1 Última corrida (2026-09-07)

Fuente: `outputs/2026-09-07_RL_AGENT/PES_DQL_results_2026-09-07_RL_AGENT.json`.

| Métrica | Valor |
|---|---|
| `overall_mean` | **0.876922** |
| `overall_std` | 0.071377 |
| `n` (secuencias) | 64 |
| Episodios | 320 000 |
| Semilla | 83 (`SEED + best_trial_number + 1`) |

### 8.2 Interpretación

- El valor coincide con el `mean_perf` del mejor ensayo (#40) de la
  optimización del 2026-09-07 (55 ensayos: 12 completos, 42 podados,
  1 fallido).
- Es inferior al `pes_dql` activo de `h1/` (0.896 en la referencia).

---

## 9. Diferencias frente a la configuración Q-Learning anterior

| Aspecto | Q-Learning estándar | `tabular_conf.dql_conf` |
|---|---|---|
| Algoritmo | Q-Learning estándar | **Double** Q-Learning |
| Q-tables | 1 | 2 (`Q_A`, `Q_B`) |
| ε-decay | Lineal o exponencial simple | Exponencial **con warm-up** |
| Reward shaping | Ninguno | **PBRS** con Φ(s) = -Σ s_i |
| Hiperparámetros bayesianos | 5 (α, γ, ε₀, ε_min, episodes) | **10** (+ `warmup_ratio`, `target_ratio`, `penalty_coeff`, `confidence_exploration_strength`, `confidence_exploration_exponent`) |
| Sesgo de maximización | Presente (sobreestima Q) | Eliminado |
| Convergencia empírica | Lenta y ruidosa al inicio | Más rápida (PBRS) y estable (Double Q) |
| Política óptima | Idéntica (PBRS preserva π*) | Idéntica (garantía de Ng et al., 1999) |

### Por qué cada mejora

1. **Double Q-Learning (Van Hasselt, 2010)** corrige que el operador `max`
   en Q-Learning estándar acopla *selección* y *evaluación* en la misma
   tabla ruidosa, lo que produce Q-valores inflados (sesgo positivo).
   Dos tablas independientes desacoplan ambas operaciones y cancelan el
   sesgo en promedio.

2. **Warm-up de ε** evita que, cuando ε
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
[`dql_conf_theory.md`](./dql_conf_theory.md).
-->
