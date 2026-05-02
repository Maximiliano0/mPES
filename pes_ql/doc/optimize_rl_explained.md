# Optimización Bayesiana de Hiperparámetros Q-Learning

## Descripción general

`optimize_rl.py` automatiza la búsqueda de los mejores hiperparámetros para el algoritmo
Q-Learning del escenario pandémico.  En lugar de probar combinaciones a mano o recorrer
una grilla exhaustiva, usa **optimización Bayesiana** (librería Optuna) para elegir de
forma inteligente qué combinación de hiperparámetros evaluar en cada paso.

---

## 1. Fundamento teórico

### 1.1 El problema

Q-Learning depende de cinco hiperparámetros cuyo valor óptimo se desconoce a priori:

| Símbolo | Parámetro | Rango explorado | Escala |
|---------|-----------|-----------------|--------|
| $\alpha$ | `learning_rate` | $[0.05,\; 0.40]$ | logarítmica |
| $\gamma$ | `discount_factor` | $[0.85,\; 0.999]$ | lineal |
| $\varepsilon_0$ | `epsilon_initial` | $[0.50,\; 1.00]$ | lineal |
| $\varepsilon_{\min}$ | `epsilon_min` | $[0.01,\; 0.15]$ | lineal |
| $N$ | `num_episodes` | $[500\,000,\; 1\,200\,000]$ | paso = 50 000 |

Probar todas las combinaciones (grid search) requiere un número exponencial de
evaluaciones.  Cada evaluación implica **entrenar una Q-table completa** (con
una semilla específica del trial, derivada de `SEED` de `CONFIG.py`) y después
evaluar sobre 64 secuencias fijas, lo que puede tardar varios minutos por
combinación dependiendo del `num_episodes` muestreado.

### 1.2 Optimización Bayesiana

La optimización Bayesiana resuelve el problema de "caja negra":

$$\boldsymbol{\theta}^{*} = \arg\max_{\boldsymbol{\theta}\in\Theta}\; f(\boldsymbol{\theta})$$

donde $\boldsymbol{\theta} = (\alpha,\gamma,\varepsilon_0,\varepsilon_{\min},N)$ y
$f(\boldsymbol{\theta})$ es el rendimiento medio normalizado sobre 64 secuencias.

El algoritmo sigue un ciclo de tres pasos en cada *trial* $t$:

1. **Modelo sustituto (surrogate):** Construir un modelo probabilístico
   $p(f \mid \boldsymbol{\theta})$ a partir de los resultados de los trials
   $1,\dots,t-1$ anteriores.
2. **Función de adquisición:** Elegir el $\boldsymbol{\theta}_t$ que maximice la
   probabilidad de mejorar el mejor valor conocido (*expected improvement*).
3. **Evaluación:** Ejecutar $f(\boldsymbol{\theta}_t)$ (entrenar + evaluar) y agregar
   el resultado al historial.

Optuna usa el **muestreador TPE (Tree-structured Parzen Estimator)** como modelo
sustituto.  En vez de modelar $p(f \mid \boldsymbol{\theta})$ directamente, modela
dos distribuciones:

$$\ell(\boldsymbol{\theta}) = p(\boldsymbol{\theta} \mid f > f^{*}), \qquad
  g(\boldsymbol{\theta}) = p(\boldsymbol{\theta} \mid f \leq f^{*})$$

y elige el $\boldsymbol{\theta}$ que maximice la razón $\ell(\boldsymbol{\theta}) / g(\boldsymbol{\theta})$.

Esto permite concentrar la exploración en zonas del espacio que estadísticamente
producen mejor rendimiento, necesitando muchos menos trials que grid search.

### 1.3 Función objetivo

El valor que se maximiza es el **rendimiento medio normalizado** de la Q-table
entrenada, evaluado sobre las 64 secuencias fijas que constituyen el *benchmark*
del escenario pandémico:

$$f(\boldsymbol{\theta}) = \frac{1}{64}\sum_{i=1}^{64}\;
  \text{perf}\!\left(S^{(i)}_{\text{final}},\; S^{(i)}_{\text{inicial}}\right)$$

donde $\text{perf}$ es la métrica de severidad final normalizada
(`calculate_normalised_final_severity_performance_metric`), definida en
`exp_utils.py` e invocada desde `run_experiment` en `pandemic.py`.  Un valor
cercano a $1.0$ indica que el agente redujo la severidad al mínimo posible con
los recursos disponibles.

---

## 2. Estructura del código

El archivo `optimize_rl.py` tiene cuatro secciones principales:

```
optimize_rl.py
├── _best_artifacts             # Almacena la mejor Q-table en memoria
├── _load_evaluation_data()     # Datos de evaluación (carga una vez)
├── objective(trial)            # Función objetivo que Optuna llama
├── _save_report(study, ...)    # Reportes y gráficos
└── main()                      # Orquestación: CLI, estudio, guardado Q-table
```

### 2.1 Carga de datos de evaluación

```python
def _load_evaluation_data():
    global _trials_per_sequence, _sevs, _number_cities_prob, _severity_prob

    _trials_per_sequence = numpy.loadtxt(
        os.path.join(INPUTS_PATH, 'sequence_lengths.csv'), delimiter=','
    )
    all_severities = numpy.loadtxt(
        os.path.join(INPUTS_PATH, 'initial_severity.csv'), delimiter=','
    )
    _sevs = convert_globalseq_to_seqs(_trials_per_sequence, all_severities)

    val_cities, count_cities = numpy.unique(
        _trials_per_sequence, return_counts=True)
    _number_cities_prob = numpy.asarray(
        (val_cities, count_cities / len(_trials_per_sequence))).T

    val_severity, count_severity = numpy.unique(
        all_severities, return_counts=True)
    _severity_prob = numpy.asarray(
        (val_severity, count_severity / len(all_severities))).T
```

**Propósito:** Cargar los CSV de longitudes y severidades una sola vez al inicio del
programa y precalcular las distribuciones de probabilidad.  Estos cuatro arrays globales
se reutilizan en cada trial sin re-leerlos del disco.

| Variable | Contenido |
|----------|-----------|
| `_trials_per_sequence` | Vector con la longitud (número de ciudades) de cada una de las 64 secuencias. |
| `_sevs` | Lista de arrays con las severidades iniciales de cada ciudad en cada secuencia. |
| `_number_cities_prob` | Distribución empírica de longitudes de secuencia (para generar secuencias aleatorias durante el entrenamiento). |
| `_severity_prob` | Distribución empírica de severidades iniciales (ídem). |

**Relación con la teoría:** Para que $f(\boldsymbol{\theta})$ sea comparable entre
trials, la evaluación debe hacerse siempre sobre el mismo benchmark; por eso se
cargan secuencias fijas.  Las distribuciones se usan durante el *entrenamiento* para
que el agente vea secuencias representativas del mismo dominio.

### 2.2 Función objetivo

```python
def objective(trial: optuna.Trial) -> float:
    # (1) Muestrear hiperparámetros
    learning_rate    = trial.suggest_float('learning_rate',    0.05,    0.40, log=True)
    discount_factor  = trial.suggest_float('discount_factor',  0.85,    0.999)
    epsilon_initial  = trial.suggest_float('epsilon_initial',  0.50,    1.00)
    epsilon_min      = trial.suggest_float('epsilon_min',      0.01,    0.15)
    num_episodes     = trial.suggest_int('num_episodes',       500_000, 1_200_000, step=50_000)

    # (2) Entrenar Q-table (semilla única por trial → réplicas independientes)
    env = Pandemic()
    env.number_cities_prob = _number_cities_prob
    env.severity_prob      = _severity_prob
    env.verbose = False

    trial_seed = SEED + int(trial.number) + 1

    # Optuna intermediate-value reporting for MedianPruner
    _pruned = {'flag': False}

    def _progress(avg_reward: float, step: int) -> bool:
        trial.report(avg_reward, step)
        if trial.should_prune():
            _pruned['flag'] = True
            return True
        return False

    rewards, Q, _ = QLearning(
        env, learning_rate, discount_factor,
        epsilon_initial, epsilon_min, num_episodes,
        seed=trial_seed,
        progress_callback=_progress,
        track_confidence=False,   # acelera 5–10× (no afecta a mean_perf)
    )

    if _pruned['flag']:
        raise optuna.TrialPruned()

    # (3) Evaluar sobre 64 secuencias fijas
    env_eval = Pandemic()
    env_eval.verbose = False

    def qf(_env, state, _seqid):
        s0 = min(int(state[0]), Q.shape[0] - 1)
        s1 = min(int(state[1]), Q.shape[1] - 1)
        s2 = min(int(state[2]), Q.shape[2] - 1)
        # Mask infeasible actions con sentinela muy negativo (los Q-valores son
        # negativos, así que un sentinela positivo dominaría argmax).
        options = Q[s0, s1, s2].copy()
        o = numpy.arange(len(options), dtype=numpy.float32)
        options[o > state[0]] = -1e9
        return numpy.argmax(options)

    _, perfs, _ = run_experiment(
        env_eval, qf, False, _trials_per_sequence, _sevs)
    mean_perf = float(numpy.mean(perfs))

    # (4) Sanitizar el objetivo (NaN/Inf de secuencias degeneradas → 0,
    # luego clamp a [0,1] para no envenenar el modelo TPE).
    if not numpy.isfinite(mean_perf):
        mean_perf = 0.0
    mean_perf = float(numpy.clip(mean_perf, 0.0, 1.0))

    # (5) Guardar métricas auxiliares
    trial.set_user_attr('mean_perf', mean_perf)
    trial.set_user_attr('std_perf',  float(numpy.std(perfs)))
    trial.set_user_attr('min_perf',  float(numpy.min(perfs)))
    trial.set_user_attr('max_perf',  float(numpy.max(perfs)))

    # (6) Preservar la mejor Q-table en memoria y en disco
    global _best_artifacts
    if mean_perf > _best_artifacts['value']:
        _best_artifacts['Q'] = Q.copy()
        _best_artifacts['rewards'] = list(rewards)
        _best_artifacts['value'] = mean_perf
        if _opt_dir:
            _save_best_artifacts(_opt_dir, _best_artifacts)

    return mean_perf
```

**Correspondencia teoría ↔ código:**

| Concepto teórico | Implementación |
|------------------|----------------|
| $\boldsymbol{\theta}$ muestreado por TPE | `trial.suggest_*()` — Optuna aplica el modelo sustituto para decidir qué valores probar. |
| Evaluación $f(\boldsymbol{\theta})$ | Entrenar Q-table + evaluar con `run_experiment()` sobre 64 secuencias fijas. |
| Resultado devuelto a Optuna | `return mean_perf` — Optuna lo usa para actualizar su modelo sustituto y elegir el próximo $\boldsymbol{\theta}$. |

#### Detalle de `trial.suggest_*`

- `suggest_float('learning_rate', 0.05, 0.40, log=True)`:
  Muestrea $\alpha$ en escala logarítmica dentro del rango $0.05$–$0.40$.
  El límite inferior bajo permite descubrir tasas de aprendizaje conservadoras
  que evitan divergencia tabular.

- `suggest_float('discount_factor', 0.85, 0.999)`:
  Permite valores hasta $0.999$ para que la señal de recompensa pueda
  propagarse a lo largo de secuencias completas (hasta 10 trials).

- `suggest_float('epsilon_min', 0.01, 0.15)`:
  Rango amplio para que Optuna decida entre explotación casi-pura
  ($\varepsilon_{\min}=0.01$) o exploración residual ($0.15$).

- `suggest_int('num_episodes', 500_000, 1_200_000, step=50_000)`:
  Discretiza en múltiplos de 50 000, ofreciendo 15 valores posibles en
  el rango — suficiente granularidad sin explotar la dimensionalidad.

**Nota sobre la semilla:** Cada trial usa una semilla única
`trial_seed = SEED + trial.number + 1` (donde `SEED = 42` proviene de
`CONFIG.py`).  Así cada trial es una *réplica estocástica independiente*: si
Optuna re-muestrea hiperparámetros similares, los resultados no son
clónicos deterministas, lo que permite al estimador TPE estimar la
varianza intrínseca del entrenamiento.  El re-entrenamiento final
(sección 2.5) sí usa `SEED` fijo para reproducibilidad.

**Nota sobre `track_confidence=False`:** `rl_agent_meta_cognitive`
(entropy + softmax + muestreo gaussiano) se invocaba en cada paso de
cada episodio para llenar `conf_list`, una lista que la optimización no
consume. Como tampoco influye sobre la actualización Q ni la selección
de acción, gatear esa llamada con `track_confidence=False` reduce el
coste por trial entre 5 y 10 × sin alterar la Q-table resultante ni
`mean_perf`. El entrenamiento local (`train_rl.py`) y el experimento
(`__main__.py`) mantienen el default `True` porque sí consumen la
confianza para los gráficos y el aggregator multi-jugador.

**Nota sobre la sanitización del objetivo:** La métrica
$\mathrm{perf} = (W - F) / (W - B)$ puede generar `NaN` si una secuencia es
degenerada ($W = B$, división por cero) o valores fuera de $[0, 1]$ si la
Q-table apenas-inicializada asigna peor que la política de cero recursos.
El bloque `if not numpy.isfinite(mean_perf): mean_perf = 0.0` seguido de
`numpy.clip(..., 0.0, 1.0)` evita que estos valores atípicos contaminen el
modelo sustituto del TPE.

#### Función `qf` (política greedy con masking)

Dentro de `objective`, se define una política greedy con enmascaramiento de acciones
infactibles, consistente con `rl_agent_meta_cognitive` en `pandemic.py`
(y, por extensión, con la versión en `pygameMediator.py` usada al ejecutar el
experimento con `python3 -m pes_ql`):

```python
def qf(_env, state, _seqid):
    s0 = min(int(state[0]), Q.shape[0] - 1)
    s1 = min(int(state[1]), Q.shape[1] - 1)
    s2 = min(int(state[2]), Q.shape[2] - 1)
    options = Q[s0, s1, s2].copy()
    o = numpy.arange(len(options), dtype=numpy.float32)
    options[o > state[0]] = -1e9
    return numpy.argmax(options)
```

Esta función se pasa a `run_experiment()` como la *action function*.  Evalúa la
Q-table **sin epsilon**: siempre elige la acción con mayor Q-valor entre las
acciones factibles (las que no exceden los recursos disponibles `state[0]`).

**Masking de acciones infactibles:** Las acciones cuyo índice supera los recursos
disponibles se sustituyen por el sentinela $-10^{9}$.  Como las recompensas son
$r = -\sum \text{severidades}$, los Q-valores aprendidos son siempre **negativos**;
un sentinela positivo (p. ej. $0.00001$) dominaría cualquier Q-valor legítimo y
forzaría a `argmax` a elegir un índice infactible que el entorno luego *clamparia*
silenciosamente a "gastar todos los recursos restantes".  El sentinela muy
negativo garantiza que las acciones infactibles pierdan `argmax` en cualquier
escenario.  Esto replica el comportamiento de `rl_agent_meta_cognitive` y
asegura que la métrica obtenida durante la optimización sea **consistente** con
el rendimiento observado al ejecutar `python3 -m pes_ql`.

Los `min(...)` aseguran que los índices de estado no excedan las dimensiones de la
Q-table `(31, 11, 10, 11)` = (recursos, trial, severidad, acciones).

#### Preservación de la mejor Q-table

Al final de cada trial, si el `mean_perf` supera el mejor valor previo, se guarda
una copia de la Q-table en `_best_artifacts`.  Esto evita el problema de re-entrenar
desde cero al final: Q-Learning es estocástico (inicialización aleatoria,
exploración $\varepsilon$-greedy, secuencias de entrenamiento aleatorias), por lo que
un re-entrenamiento con los mismos hiperparámetros puede producir una Q-table diferente
y potencialmente peor.

```python
    global _best_artifacts
    if mean_perf > _best_artifacts['value']:
        _best_artifacts['Q'] = Q.copy()
        _best_artifacts['rewards'] = list(rewards)
        _best_artifacts['value'] = mean_perf
        if _opt_dir:
            _save_best_artifacts(_opt_dir, _best_artifacts)
```

La pareja `_best_artifacts.npz` (Q-table + rewards) y
`_best_artifacts.json` (metadatos) se reescribe atomicamente después de
cada mejora, **sin usar `pickle`** (CWE-502 hardening). Esto permite
reanudar una corrida y recuperar el mejor trial sin reentrenar.

### 2.3 Persistencia con SQLite

```python
db_path = os.path.join(opt_dir, f'optuna_study_{opt_date}.db')
storage = f'sqlite:///{db_path}'

study = optuna.create_study(
    direction='maximize',
    study_name=f'qlearning_opt_{opt_date}',
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=4),
    storage=storage,
    load_if_exists=True,
)
```

Cada trial completado se persiste en un archivo SQLite. Si el proceso se interrumpe
(suspensión de la máquina, error, etc.), al re-ejecutar el mismo comando Optuna detecta
los trials ya completados y continúa desde donde se detuvo:

```python
completed = len([t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE])
remaining = max(0, n_trials - completed)
```

**MedianPruner:** El estudio se crea con
`MedianPruner(n_startup_trials=5, n_warmup_steps=4)`. Una vez completados
los primeros cinco trials, cada nuevo trial reporta su reward medio cada
10 000 episodios y Optuna lo aborta si está por debajo de la mediana de
los rewards intermedios reportados por trials previos en el mismo *step*.
Esto suele recortar el 50 % o más del tiempo de cómputo en
configuraciones poco prometedoras.

**Relación con la teoría:** El modelo sustituto del TPE se reconstruye a partir del
historial almacenado; no hay pérdida de información respecto a correr todo de corrido.

### 2.4 Protección contra underflow numérico

```python
_prev_err = numpy.seterr(under='ignore')
try:
    study.optimize(objective, n_trials=remaining,
                   callbacks=[_progress_callback])
finally:
    numpy.seterr(**_prev_err)
```

El módulo `pes_ql` configura `numpy.seterr(all='raise', under='ignore')` en
su `__init__.py` para detectar errores numéricos durante la simulación.  Aunque el
*underflow* ya se ignora a nivel de paquete, el muestreador TPE calcula internamente
`numpy.exp(x)` donde $x$ puede ser muy negativo ($x < -700$), produciendo un
*underflow* a $0.0$ que es matemáticamente inofensivo (simplemente indica
probabilidades ínfimas).

El `try/finally` funciona como protección adicional (*belt-and-suspenders*): captura
el estado actual de `numpy.seterr` antes de la optimización y lo restaura al
terminar, asegurando robustez incluso si la configuración de errores cambiara en
versiones futuras del paquete.

### 2.5 Guardado de Q-table y reportes

Una vez completada la optimización, `main()` realiza dos pasos finales:

1. **Usar la Q-table preservada** — Si `_best_artifacts` contiene la Q-table del
   mejor trial (corrida completa sin interrupción), se usa directamente sin
   re-entrenar.  Si se reanudó un estudio previo y la Q-table no está en
   memoria se intenta cargar `_best_artifacts.npz/.json` desde disco; sólo
   si tampoco están disponibles se aplica un *replay* del mejor trial usando
   la misma semilla por trial (`SEED + best.number + 1`) y `track_confidence=False`,
   lo que reproduce *bit a bit* la Q-table original:

```python
if _best_artifacts['Q'] is not None and _best_artifacts['value'] >= best.value:
    best_Q = _best_artifacts['Q']
    best_rewards = numpy.array(_best_artifacts['rewards'])
else:
    # Replay determinista del trial ganador (solo si se resumió y no hay caché)
    replay_seed = SEED + int(best.number) + 1
    best_rewards, best_Q, _ = QLearning(
        env_final, bp['learning_rate'], bp['discount_factor'],
        bp['epsilon_initial'], bp['epsilon_min'], bp['num_episodes'],
        seed=replay_seed, track_confidence=False,
    )
```

2. **Generar reportes** mediante `_save_report()`:

| Archivo generado | Contenido |
|------------------|-----------|
| `optimization_results_<fecha>.txt` | Tabla de todos los trials ordenados por rendimiento, mejores hiperparámetros, estadísticas. |
| `optimization_history_<fecha>.png` | Gráfico de convergencia: rendimiento de cada trial y curva de mejor acumulado (*running best*). |
| `hyperparameter_importances_<fecha>.png` | Gráfico de barras con la importancia relativa de cada hiperparámetro (calculada por Optuna con *fANOVA*). |
| `q_best_<fecha>.npy` | Q-table entrenada con los mejores hiperparámetros. |
| `rewards_best_<fecha>.npy` | Historia de recompensas del mejor entrenamiento. |

El gráfico de importancia responde directamente a la pregunta: *¿cuál hiperparámetro
afecta más al rendimiento?*  Esto orienta futuros ajustes manuales o refinamientos
del espacio de búsqueda.

### 2.6 Notificaciones push

El módulo incluye soporte opcional de **notificaciones push** a través de
`utils.notify`.  Si el módulo `utils` está disponible en `sys.path`, se envían
notificaciones en dos momentos:

1. **Progreso cada 10 trials** — El callback `_progress_callback` invoca
   `notify()` cada vez que `done % 10 == 0`, informando trials completados,
   mejor valor hasta el momento, tiempo transcurrido y rendimiento del último
   trial.

2. **Error durante la optimización** — El bloque `if __name__ == '__main__'`
   envuelve `main()` en un `try/except`: si ocurre cualquier excepción, se
   envía una notificación con prioridad `urgent` y el traceback completo antes
   de re-lanzar la excepción.

```python
# Notificación de progreso (cada 10 trials, robusta a trial.value None
# y a la fase inicial sin trials COMPLETE).
try:
    best_val = study.best_value
    best_str = f"{best_val:.4f}"
except ValueError:
    best_val = None
    best_str = "  n/a  "
value_str = f"{trial.value:.4f}" if trial.value is not None else "   n/a"
if done > 0 and done % 10 == 0:
    best_msg = f"{best_val:.6f}" if best_val is not None else "n/a"
    notify(
        f"[{_PKG_NAME}] {done}/{n_trials} trials",
        f"Se completaron {done} de {n_trials} trials.\n"
        f"Mejor valor hasta ahora: {best_msg}\n"
        f"Último trial: value={value_str}\n"
        f"Tiempo transcurrido: {elapsed:.0f}s ({elapsed / 60:.1f} min)",
        tags="chart_with_upwards_trend"
    )
```

Si `utils.notify` no está disponible (por ejemplo, al ejecutar fuera del
workspace mPES), el import falla silenciosamente y `notify` se reemplaza
por un *no-op* tipado:

```python
try:
    from utils.scripts.notify import notify
except ImportError:
    def notify(*_args, **_kwargs):
        """No-op fallback when ``utils.scripts.notify`` is not importable."""
        return None
```

---

## 3. Diagrama de flujo

```
main()
  │
  ├─ parsear argumentos: n_trials, --resume YYYY-MM-DD
  ├─ _load_evaluation_data()
  │
  ├─ crear/cargar estudio Optuna (SQLite)
  │       │
  │       │  ┌──────────────────────────────────────────┐
  │       └──►  study.optimize(objective, n_trials)     │
  │           │                                         │
  │           │  ┌─ trial n ────────────────────────┐   │
  │           │  │ TPE elige θ = (α, γ, ε₀, ε_min, N) │
  │           │  │ QLearning(env, θ) → Q-table      │   │
  │           │  │ run_experiment(Q, 64 seqs) → perf│   │
  │           │  │ return mean(perf)                │   │
  │           │  └─────────────────────────────────┘   │
  │           │  repetir hasta completar n_trials       │
  │           └──────────────────────────────────────────┘
  │
  ├─ imprimir mejores hiperparámetros
  ├─ usar Q-table preservada (o retrain si --resume)
  └─ _save_report() → .txt, .png, .npy
```

---

## 4. Uso

### 4.1 Ejecución básica

```bash
cd /home/mecatronica/Documentos/maximiliano/mPES
source linux_mpes_env/bin/activate

# 100 trials (valor por defecto)
python3 -m pes_ql.ext.optimize_rl

# 200 trials
python3 -m pes_ql.ext.optimize_rl 200
```

### 4.2 Ejecución en segundo plano (recomendada)

```bash
nohup python3 -m pes_ql.ext.optimize_rl 100 \
  > pes_ql/inputs/bayesian_opt.log 2>&1 &
```

Para evitar suspensión de la máquina ver
[bayesian_optimization_guide.md](bayesian_optimization_guide.md) § 2.

### 4.3 Reanudar una corrida interrumpida

**Mismo día** — re-ejecutar el mismo comando; Optuna detecta los trials previos en el
SQLite y continúa:

```bash
python3 -m pes_ql.ext.optimize_rl 100
```

**Día diferente** — usar `--resume` con la fecha original del SQLite:

```bash
python3 -m pes_ql.ext.optimize_rl 100 --resume 2026-02-12
```

### 4.4 Salida esperada

```
══════════════════════════════════════════════════════════════════════
  BAYESIAN OPTIMISATION — Q-LEARNING HYPERPARAMETERS
══════════════════════════════════════════════════════════════════════

ℹ Output directory: .../2026-02-12_BAYESIAN_OPT
ℹ Target number of trials: 100

── Loading Evaluation Data ──
  ● Sequence lengths shape: (64,)
  ● Sequences loaded: 64

── Running Bayesian Optimisation ──
ℹ Search space:
  ● learning_rate    ∈ [0.05, 0.40]      (log scale)
  ● discount_factor  ∈ [0.85, 0.999]
  ● epsilon_initial  ∈ [0.50, 1.00]
  ● epsilon_min      ∈ [0.01, 0.15]
  ● num_episodes     ∈ [500000, 1200000]  (step=50000)

  Trial   1/100  |  value=0.7563  |  best=0.7563  |  elapsed=19s
  Trial   2/100  |  value=0.6941  |  best=0.7563  |  elapsed=76s
  ...
  Trial 100/100  |  value=0.8201  |  best=0.8482  |  elapsed=2934s

✓ Optimisation finished in 2934.2s (48.9 min)

── Best Hyperparameters Found ──
  ● learning_rate             = 0.3597
  ● discount_factor           = 0.8651
  ● epsilon_initial           = 0.6791
  ● epsilon_min               = 0.0848
  ● num_episodes              = 900000
ℹ Mean normalised performance: 0.848200
```

### 4.5 Archivos de salida

Todos se guardan en `pes_ql/inputs/<FECHA>_BAYESIAN_OPT/`:

| Archivo | Descripción |
|---------|-------------|
| `optuna_study_<fecha>.db` | Base de datos SQLite con todo el historial de Optuna (permite reanudar). |
| `q_best_<fecha>.npy` | Q-table `(31,11,10,11)` del mejor trial de la optimización. |
| `rewards_best_<fecha>.npy` | Historia de recompensas promedio del mejor entrenamiento (cada 10 000 episodios). |
| `best_params_<fecha>.json` | Hiperparámetros ganadores + `best_trial_number`, `mean_perf` y `trial_seed` (= `SEED + trial_number + 1`). Consumido automáticamente por `train_rl.py`. |
| `repro_fingerprint_<fecha>.json` | Snapshot del entorno (versiones de numpy/Python, plataforma, hash SHA-256 de los CSV, commit `git`, `SEED`) para validación de reproducibilidad. |
| `_best_artifacts.npz` / `_best_artifacts.json` | Caché de la Q-table + metadatos del mejor trial; se reescribe atómicamente tras cada mejora y permite reanudar sin reentrenar. Formato sin `pickle` (CWE-502). |
| `optimization_results_<fecha>.txt` | Reporte textual completo: mejores parámetros, estadísticas, tabla de todos los trials (numeración 1-based). |
| `optimization_history_<fecha>.png` | Gráfico de convergencia (rendimiento por trial + curva de mejor acumulado). |
| `hyperparameter_importances_<fecha>.png` | Importancia relativa de cada hiperparámetro (fANOVA). |

Además, al finalizar la corrida `optimize_rl.py` **espeja** automáticamente
`q_best_<fecha>.npy`, `rewards_best_<fecha>.npy` y `best_params_<fecha>.json`
a `pes_ql/inputs/q.npy`, `pes_ql/inputs/rewards.npy` y
`pes_ql/inputs/best_params.json`, de modo que `train_rl.py` y el
experimento (`python3 -m pes_ql`) los consumen sin pasos manuales.

---

## 5. Relación con Q-Learning

La ecuación de actualización de Q-Learning usada en `pandemic.py` es:

$$Q(s, a) \;\leftarrow\; Q(s, a) + \alpha\left[r + \gamma\max_{a'}Q(s', a') - Q(s, a)\right]$$

donde:

- $s = (\text{recursos},\; \text{trial},\; \text{severidad})$ — estado discretizado.
- $a \in \{0, 1, \ldots, 10\}$ — recursos asignados a la ciudad actual.
- $r$ — recompensa inmediata del entorno.
- $\alpha$ — tasa de aprendizaje (`learning_rate`).
- $\gamma$ — factor de descuento (`discount_factor`).

La política de exploración es $\varepsilon$-greedy con decaimiento lineal:

$$\varepsilon_t = \max\!\left(\varepsilon_{\min},\;\; \varepsilon_0 - t\cdot\frac{\varepsilon_0 - \varepsilon_{\min}}{N}\right)$$

La optimización Bayesiana actúa **un nivel por encima** de Q-Learning: no modifica
el algoritmo, sino que busca los valores de $(\alpha, \gamma, \varepsilon_0,
\varepsilon_{\min}, N)$ que maximicen el rendimiento de la política resultante.
Es decir, es una **meta-optimización** (optimización de la optimización).

Todos los trials de entrenamiento usan **semillas distintas pero
reproducibles**: cada trial recibe `trial_seed = SEED + trial.number + 1`
(con `SEED = 42` de `CONFIG.py`), de modo que dos trials con
hiperparámetros similares son réplicas estocásticas independientes y
Optuna puede estimar la varianza intrínseca del entrenamiento. La misma
semilla por trial queda registrada en `best_params_<fecha>.json` para
que `train_rl.py` reproduzca *bit a bit* la Q-table del mejor trial.

```
┌──────────────────────────────────────────────────┐
│  Nivel externo: Optimización Bayesiana (Optuna)    │
│  Decide θ = (α, γ, ε₀, ε_min, N)                  │
│                                                    │
│  ┌─────────────────────────────────────────────┐  │
│  │  Nivel interno: Q-Learning                    │  │
│  │  seed = SEED + trial.number + 1               │  │
│  │  Entrena Q-table con θ durante N episodios     │  │
│  │  Q(s,a) ← Q(s,a) + α[r + γ·max Q - Q]         │  │
│  └───────────────────────────────────────────────┘  │
│                                                    │
│  Evaluar Q-table → f(θ) = rendimiento medio        │
│  Devolver f(θ) a Optuna                            │
└────────────────────────────────────────────────────┘
```

---

## 6. Flujo de trabajo completo

La optimización Bayesiana es el **primer paso** de un flujo de tres etapas
totalmente automatizado (sin transferencia manual de hiperparámetros ni
copiado manual de archivos):

```
1. Optimización Bayesiana         python3 -m pes_ql.ext.optimize_rl [N]
   └─ Buscar mejores (α, γ, ε₀, ε_min, N)
   └─ Guardar Q-table, best_params_<fecha>.json y repro_fingerprint_<fecha>.json
      en inputs/<fecha>_BAYESIAN_OPT/
   └─ Espejar q.npy, rewards.npy y best_params.json a inputs/

2. Entrenamiento definitivo       python3 -m pes_ql.ext.train_rl [episodes]
   └─ Auto-carga el último best_params_<fecha>.json vía repro.find_latest_artifacts()
   └─ Verifica el repro_fingerprint contra el entorno actual
   └─ Reentrena la Q-table con la misma trial_seed y track_confidence=False
      (reproducibilidad bit-a-bit del trial ganador)
   └─ Guarda Q-table y gráficos en inputs/<fecha>_RL_TRAIN/ y reescribe
      inputs/q.npy + inputs/rewards.npy

3. Experimento                    python3 -m pes_ql
   └─ Lee inputs/q.npy y inputs/rewards.npy
   └─ Ejecuta el agente RL sobre 8 bloques × 8 secuencias
```

### 6.1 Transferencia automática de hiperparámetros

A partir de 2026-04, **`train_rl.py` ya no contiene hiperparámetros
codificados a mano**. Lee automáticamente el último
`best_params_<fecha>.json` disponible bajo
`pes_ql/inputs/<fecha>_BAYESIAN_OPT/` mediante
`repro.find_latest_artifacts()` y extrae:

- `hyperparameters` (`learning_rate`, `discount_factor`, `epsilon_initial`,
  `epsilon_min`, `num_episodes`).
- `best_trial_number` (0-based) y `trial_seed = SEED + best_trial_number + 1`,
  para reproducir bit-a-bit la Q-table del mejor trial.
- `mean_perf`, comparado al final contra la `mean_perf` calculada localmente
  para detectar divergencias.

Si no existe ningún directorio `_BAYESIAN_OPT/`, `train_rl.py` cae en
`_FALLBACK_PARAMS` (referencia histórica, trial #40):

| Parámetro | Valor del fallback |
|-----------|--------------------|
| `learning_rate` | 0.35965545888114453 |
| `discount_factor` | 0.8650520580454709 |
| `epsilon_initial` | 0.6791201210873763 |
| `epsilon_min` | 0.08483331103075126 |
| `num_episodes` | 900 000 |
| `trial_number` | 39 (0-based; 1-based = 40) |
| `training_seed` | `SEED + 39 + 1 = 82` |

### 6.2 Transferencia automática de la Q-table al experimento

No hace falta copiar manualmente la Q-table: tanto `optimize_rl.py`
(al finalizar la optimización) como `train_rl.py` (al finalizar el
entrenamiento) escriben/sobrescriben `inputs/q.npy` y `inputs/rewards.npy`
con la versión más reciente. `__main__.py` valida la existencia de ambos
archivos antes de iniciar el experimento; si no los encuentra, muestra un
error indicando que se debe ejecutar `train_rl.py` primero.
