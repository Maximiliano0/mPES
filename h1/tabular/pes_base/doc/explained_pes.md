# PES: Explicación Detallada del Funcionamiento del Proyecto

## 1. Introducción

**PES (Pandemic Experiment Scenario)** es un paquete del workspace **mPES** que
simula escenarios de respuesta a pandemias, donde un agente de
**Reinforcement Learning (Q-Learning tabular)** aprende a optimizar la
asignación limitada de recursos para minimizar la severidad de enfermedades en
múltiples ciudades.

### Objetivo Principal

Entrenar y ejecutar un agente inteligente que tome decisiones estratégicas sobre
distribución de recursos médicos/sanitarios bajo restricciones de disponibilidad.

---

## 2. Estructura Jerárquica del Experimento

El experimento sigue una estructura anidada estricta definida en `__main__.py`:

```
1 EXPERIMENTO
├─ NUM_BLOCKS = 8 BLOQUES
│  ├─ NUM_SEQUENCES = 8 SECUENCIAS por bloque
│  │  ├─ NUM_MIN_TRIALS a NUM_MAX_TRIALS (3–10) TRIALS por secuencia
│  │  │  └─ 1 ACCIÓN DE ASIGNACIÓN DE RECURSOS (0–10)
```

### Desglose de Números

- **Total de bloques**: 8
- **Total de secuencias**: 64 (8 bloques × 8 secuencias)
- **Trials por bloque**: 45 (`TOTAL_NUM_TRIALS_IN_BLOCK`)
- **Trials totales**: ~360 (8 bloques × 45 trials)
- **Total de decisiones**: ~360

### Configuración en `CONFIG.py`

```python
NUM_BLOCKS = 8                      # Número de bloques experimentales
NUM_SEQUENCES = 8                   # Secuencias por bloque
NUM_MIN_TRIALS = 3                  # Trials mínimos por secuencia
NUM_MAX_TRIALS = 10                 # Trials máximos por secuencia
TOTAL_NUM_TRIALS_IN_BLOCK = 45      # Suma exacta de trials en un bloque
AVAILABLE_RESOURCES_PER_SEQUENCE = 39  # Presupuesto total por secuencia
INIT_NO_OF_CITIES = 2               # Ciudades visibles al inicio
```

### Presupuesto de Recursos

| Concepto | Valor |
|----------|-------|
| Recursos totales por secuencia | 39 (`AVAILABLE_RESOURCES_PER_SEQUENCE`) |
| Recursos pre-asignados (2 ciudades iniciales, seed fija) | 9 (3 + 6) |
| Recursos disponibles para el agente | 30 (= 39 − 9) |
| Rango de asignación por trial | 0–10 (`MIN/MAX_ALLOCATABLE_RESOURCES`) |

> **Nota**: Las 2 ciudades iniciales siempre reciben asignaciones de 3 y 6
> recursos respectivamente, porque `numpy.random.seed(3)` con
> `INIT_NO_OF_CITIES = 2` produce esos valores determinísticamente.

---

## 3. Modelo Dinámico del Escenario Pandémico

### 3.1 La Fórmula de Progresión de Severidad

Implementada en `src/exp_utils.py` → `get_updated_severity()`:

```
new_severity = max(0, β × initial_severity − α × allocated_resources)
```

Donde:

- **β** (`SEVERITY_MULTIPLIER`) = 1 + `PANDEMIC_PARAMETER` = **1.4**
  - Representa el crecimiento natural de la pandemia sin intervención.
  - Tasa de crecimiento: 40 % por paso temporal.

- **α** (`RESPONSE_MULTIPLIER`) = `PANDEMIC_PARAMETER` = **0.4**
  - Representa la efectividad de los recursos asignados.
  - Cada unidad de recurso reduce 0.4 puntos de severidad.

Ambas constantes se derivan en `__init__.py`:

```python
RESPONSE_MULTIPLIER = PANDEMIC_PARAMETER        # α = 0.4
SEVERITY_MULTIPLIER = 1 + PANDEMIC_PARAMETER     # β = 1.4
```

### 3.2 Ejemplo Numérico (una sola ciudad)

```
Entrada: severidad_inicial = 4, recursos_asignados = 5, α = 0.4, β = 1.4

Paso 1: new_sev = 1.4 × 4   − 0.4 × 5 = 5.6  − 2.0 = 3.6
Paso 2: new_sev = 1.4 × 3.6 − 0.4 × 5 = 5.04 − 2.0 = 3.04
Paso 3: new_sev = 1.4 × 3.04− 0.4 × 5 = 4.256− 2.0 = 2.256
```

La severidad evoluciona a través de los trials, y el efecto de los recursos se
acumula secuencialmente.

### 3.3 Evolución Temporal en una Secuencia Completa

La función `get_array_of_sequence_severities_from_allocations()` en
`src/exp_utils.py` calcula la evolución de severidad de **todas** las ciudades
en una secuencia. En cada trial:

1. Se añade una nueva ciudad con su severidad inicial.
2. Se aplica la fórmula de actualización a **todas** las ciudades visibles
   (nuevas y previas).
3. El vector de severidades se actualiza con los nuevos valores.

**Ejemplo con 3 ciudades** (α = 0.4, β = 1.4):

```
Severidades iniciales: [3, 4, 8]
Asignaciones:          [5, 6, 4]

Trial 0 (entra Ciudad 1, severidad=3, alloc=5):
  Ciudad 1: 1.4×3 − 0.4×5 = 2.20

Trial 1 (entra Ciudad 2, severidad=4, alloc=6):
  Ciudad 1: 1.4×2.20 − 0.4×5 = 1.08
  Ciudad 2: 1.4×4    − 0.4×6 = 3.20

Trial 2 (entra Ciudad 3, severidad=8, alloc=4):
  Ciudad 1: 1.4×1.08 − 0.4×5 = 0    (clipeado a 0)
  Ciudad 2: 1.4×3.20 − 0.4×6 = 2.08
  Ciudad 3: 1.4×8    − 0.4×4 = 9.60

Severidades finales: [0, 2.08, 9.60]
```

> **Clave**: La asignación de recursos a una ciudad se aplica en **todos** los
> pasos subsiguientes, no solo en el trial en que se asigna. Esto crea una
> dinámica temporal compuesta donde las decisiones tempranas tienen mayor
> impacto acumulativo.

---

## 4. Flujo Principal del Experimento

### 4.1 Inicialización (`__main__.py`)

```python
def main():
    # 1. Validar archivos del RL-Agent
    if PLAYER_TYPE == 'RL_AGENT':
        q_file = os.path.join(INPUTS_PATH, 'q.npy')
        rewards_file = os.path.join(INPUTS_PATH, 'rewards.npy')

        if not os.path.isfile(q_file):
            terminal_utils.error("Q-Table file not found!")
            return

        # Validar carga exitosa
        Q = numpy.load(q_file)
        rewards = numpy.load(rewards_file)
```

**Pasos**:

1. Verifica que `inputs/q.npy` (Q-table entrenada) exista.
2. Verifica que `inputs/rewards.npy` (histórico de recompensas) exista.
3. Carga ambos archivos y valida dimensiones/tipo de datos.
4. Si alguno falta o falla, sugiere ejecutar `python3 -m tabular.pes_base.ext.train_rl`.

> **Nota sobre archivos de entrenamiento**: El pipeline de entrenamiento
> (`train_rl.py`) guarda los archivos en `inputs/<fecha>_RL_TRAIN/q_<fecha>.npy`.
> El experimento busca `inputs/q.npy`. El usuario debe copiar manualmente la
> Q-table entrenada al directorio raíz de `inputs/`.

### 4.2 Creación de Sesión

```python
experiment_date = datetime.date.today().strftime("%Y-%m-%d")
MySubjectId = f"{experiment_date}_{PLAYER_TYPE}"
# Ejemplo: "2026-02-26_RL_AGENT"

session_outputs_path = os.path.join(OUTPUTS_PATH, MySubjectId)
os.makedirs(session_outputs_path, exist_ok=True)

log_utils.create_ConsoleLog_filehandle_singleton(MySubjectId)
# Crea: outputs/PES_log_2026-02-26_RL_AGENT.txt
```

Se guardan dos archivos al inicio de la sesión:

- **SubjectInfo**: `PES__<SubjectId>.txt` — parámetros de configuración.
- **Responses**: `PES_responses_<SubjectId>.txt` — decisiones trial a trial.

### 4.3 Asignación de Mapas y Secuencias

**Estructuras de datos principales**:

```python
NumTrials__blocks_x_sequences__2darray  = numpy.zeros((NUM_BLOCKS, NUM_SEQUENCES))
MapIndices__blocks_x_sequences__2darray = numpy.zeros((NUM_BLOCKS, NUM_SEQUENCES))
```

**Lógica de asignación de índices de mapa**:

```python
for blk in range(NUM_BLOCKS):
    numpy.random.seed(100 + blk)          # Seed reproducible por bloque
    for seq in range(NUM_SEQUENCES):
        counter_seq = NUM_ATTEMPTS_TO_ASSIGN_SEQ

        while counter_seq > 0:
            b = numpy.random.randint(0, 9)     # Índice aleatorio 0–8
            if b not in MapIndices__blocks_x_sequences__2darray[blk, :]:
                MapIndices__blocks_x_sequences__2darray[blk, seq] = b
                break                          # Asignación exitosa, avanzar
            counter_seq -= 1
```

> **Nota de implementación**: la semilla `numpy.random.seed(100 + blk)` se
> establece una vez por bloque (fuera del loop de secuencias), de modo que la
> secuencia pseudoaleatoria avanza continuamente entre secuencias. El `break`
> tras cada asignación exitosa garantiza que cada slot reciba el primer índice
> válido sin sobrescrituras. Esto produce 8 índices únicos por bloque de un
> pool de 9 posibles (0–8), con un patrón determinístico y específico por
> bloque.

**Asignación de número de trials por secuencia**:

```python
if USE_FIXED_BLOCK_SEQUENCES:
    # Cargar desde sequence_lengths.csv (64 valores pre-definidos)
    NumTrials__blocks_x_sequences__2darray[blk, :] = exp_utils.next_seq_length(
        blk * NUM_SEQUENCES, NUM_SEQUENCES
    )
else:
    # Generar aleatoriamente respetando constraint de 45 trials/bloque
    NumTrials__blocks_x_sequences__2darray[blk, :] = exp_utils.sampler(
        NUM_SEQUENCES, TOTAL_NUM_TRIALS_IN_BLOCK,
        [NUM_MIN_TRIALS, NUM_MAX_TRIALS], rn=blk
    )
```

### 4.4 Severidades Iniciales

```python
if RANDOM_INITIAL_SEVERITY:
    first_severity = exp_utils.random_severity_generator(
        int(numpy.sum(NumTrials__blocks_x_sequences__2darray)), 2, 9
    )
else:
    first_severity = numpy.loadtxt(os.path.join(INPUTS_PATH, INITIAL_SEVERITY_FILE))
    first_severity = first_severity[0 : int(numpy.sum(NumTrials__blocks_x_sequences__2darray))]
```

El array `first_severity` es un vector plano con ~360 valores que se indexa
secuencialmente a lo largo del experimento mediante `AbsoluteTrialIndex`.

### 4.5 Ejecución de Bloques, Secuencias y Trials

```python
for CurrentBlockIndex in range(NUM_BLOCKS):
    for CurrentSequenceIndex, CurrentSequenceMapIndex in enumerate(CurrentBlockMapIndices):

        # Inicializar recursos y ciudades iniciales
        resources_to_allocate = AVAILABLE_RESOURCES_PER_SEQUENCE  # 39
        numpy.random.seed(3)  # → severidades iniciales [4, 3], allocs [3, 6]

        for c in range(INIT_NO_OF_CITIES):
            init_severity.append(numpy.random.randint(MIN_INIT_SEVERITY, 1 + MAX_INIT_SEVERITY))
            ResourceAllocationsAtCurrentlyVisibleCities.append(
                numpy.random.randint(MIN_INIT_RESOURCES, 1 + MAX_INIT_RESOURCES)
            )

        resources_left = resources_to_allocate - numpy.sum(
            ResourceAllocationsAtCurrentlyVisibleCities
        )  # 39 − 9 = 30

        # Actualizar severidades de ciudades iniciales
        SeveritiesOfCurrentlyVisibleCities = exp_utils.get_updated_severity(
            INIT_NO_OF_CITIES,
            ResourceAllocationsAtCurrentlyVisibleCities,
            init_severity
        )

        # Loop de trials
        for trial_no in range(int(NumTrials__blocks_x_sequences__2darray[blk, seq])):
            # Consultar al agente RL
            (pc, r, rt_h, rt_rel, mov) = pygameMediator.provide_rl_agent_response(
                ResourceAllocationsAtCurrentlyVisibleCities,
                resources_left,
                CurrentBlockIndex,
                CurrentSequenceIndex,
                trial_no
            )
            # Actualizar severidades, registrar respuesta, reducir recursos
            ...
```

### 4.6 Consulta al Agente RL (`src/pygameMediator.py`)

La función `provide_rl_agent_response()` es la interfaz principal:

1. **Carga** la Q-table desde `inputs/q.npy`.
2. **Obtiene** la severidad de la ciudad actual a partir de las longitudes de
   secuencia y el array global de severidades.
3. **Construye** el estado: `[resources_left, trial_no, severity]`.
4. **Indexa** la Q-table: `Q[resources_idx, city_idx, sever_idx]`.
5. **Llama** a `rl_agent_meta_cognitive()` para obtener la acción (argmax),
   confianza (entropía) y tiempos de reacción simulados.
6. **Retorna** `(confidence, response, rt_hold, rt_release, movement)`.

> **Nota**: La Q-table se carga desde disco en **cada** llamada a
> `provide_rl_agent_response()`. Esto es ineficiente pero funcional.

---

## 5. Cálculo de Severidades y Performance

### 5.1 Severidad Final por Secuencia

Implementado en `get_sequence_severity_from_allocations()`:

```python
def get_sequence_severity_from_allocations(Allocations, InitialSeverities):
    return numpy.sum(
        get_array_of_sequence_severities_from_allocations(Allocations, InitialSeverities)
    )
```

### 5.2 Métrica de Performance Normalizado

Implementada en `calculate_normalised_final_severity_performance_metric()`:

```python
FinalSequenceSeverity     = numpy.sum(SeveritiesFromSequence)

WorstCaseAllocations      = numpy.full_like(SeveritiesFromSequence, MIN_ALLOCATABLE_RESOURCES)
# El "best case" es la asignación factible óptima bajo el presupuesto
# real del agente (DP por presupuesto), no "max_alloc en cada trial".
BestCaseSequenceSeverity  = _best_feasible_sequence_severity(InitialSequenceSeverities)
WorstCaseSequenceSeverity = get_sequence_severity_from_allocations(WorstCaseAllocations, InitialSequenceSeverities)

Performance = (WorstCaseSequenceSeverity - FinalSequenceSeverity) / \
              (WorstCaseSequenceSeverity - BestCaseSequenceSeverity)
```

**Interpretación**:

| Performance | Significado |
|-------------|-------------|
| 0.0 | Resultado igual al peor caso (sin recursos) |
| 0.5 | Resultado intermedio |
| 1.0 | Resultado óptimo factible (oráculo DP bajo el presupuesto real) |

> **Nota técnica**: El "best case" se calcula con un DP de mochila acotada
> sobre asignaciones enteras por ciudad, sumando como mucho
> `_FEASIBLE_BUDGET_PER_SEQUENCE = AVAILABLE_RESOURCES_PER_SEQUENCE - 9`
> (=30 con la configuración por defecto). Una política factible puede
> efectivamente alcanzar `Performance = 1.0`.

### 5.3 Ejemplo Numérico

```
Severidades iniciales: [3, 4, 8]
Asignaciones reales:   [5, 6, 4]
Severidades finales:   [0, 2.08, 9.60]  → FinalSeverity = 11.68

Worst case (alloc = [0, 0, 0]):
  Trial 0: [4.20]
  Trial 1: [5.88, 5.60]
  Trial 2: [8.232, 7.84, 11.20]  → WorstCase = 27.272

Best case (alloc = [10, 10, 10]):
  Trial 0: [0.20]
  Trial 1: [0, 1.60]
  Trial 2: [0, 0, 7.20]  → BestCase = 7.20

Performance = (27.272 − 11.68) / (27.272 − 7.20) = 15.592 / 20.072 ≈ 0.777
```

---

## 6. Módulo de Confianza Meta-cognitiva

### 6.1 `rl_agent_meta_cognitive()`

Esta función existe en **dos ubicaciones** con implementaciones similares:

- `src/pygameMediator.py`: usada durante la **ejecución** del experimento
  (`python3 -m tabular.pes_base`).
- `ext/pandemic.py`: usada durante el **entrenamiento** (`python3 -m tabular.pes_base.ext.train_rl`).

**Algoritmo**:

```python
def rl_agent_meta_cognitive(options, resources_left, response_timeout):
    # 1. Calcular entropías de referencia
    m_entropy = entropy_from_pdf([1, 0, 0, ..., 0])   # Mínima (determinística)
    M_entropy = entropy_from_pdf([1, 1, 1, ..., 1])   # Máxima (uniforme)

    # 2. Filtrar opciones infactibles (acción > recursos disponibles)
    options[acción > resources_left] = 0.00001

    # 3. Calcular entropía de las opciones filtradas
    dec_entropy = entropy_from_pdf(options)

    # 4. Normalizar confianza a [0, 1]
    confidence = (dec_entropy - M_entropy) / (m_entropy - M_entropy)

    # 5. Seleccionar acción (greedy)
    response = numpy.argmax(options)

    # 6. Mapear confianza a tiempos de reacción
    map_to_response_time = lambda x: x * (-2) + 1
    mu = int(map_to_response_time(confidence) * 10)
    rt_hold    = numpy.clip(numpy.random.normal(mu, 3, 1)[0], 0, response_timeout/1000)
    rt_release = numpy.clip(rt_hold + numpy.random.normal(mu, 1, 1)[0], 0, response_timeout/1000)

    return response, confidence, rt_hold, rt_release
```

**Propósito**: Simular un agente que no solo toma decisiones óptimas, sino que
refleja incertidumbre (baja confianza) con tiempos de reacción más largos.

### 6.2 Entropía (`entropy_from_pdf()` en `ext/tools.py`)

```python
def entropy_from_pdf(pdf):
    pdf = pdf + numpy.abs(numpy.min(pdf))     # Desplazar a positivos
    p = pdf / numpy.sum(pdf)                  # Normalizar a probabilidad
    p[p == 0] += 0.000001                     # Evitar log(0)
    H = -numpy.dot(p, numpy.log2(p))          # Entropía de Shannon (bits)
    return H
```

| Distribución | Entropía (11 acciones) | Confianza |
|-------------|------------------------|-----------|
| Determinística `[1,0,...,0]` | ≈ 0 bits | ≈ 1.0 (alta) |
| Uniforme `[1,1,...,1]` | ≈ 3.46 bits | ≈ 0.0 (baja) |

### 6.3 Diferencias entre las dos implementaciones

| Aspecto | `pygameMediator.py` | `pandemic.py` |
|---------|---------------------|---------------|
| Tamaño vectores referencia | Fijo: `numpy.zeros((11,))` | Dinámico: `numpy.zeros((len(options),))` |
| Clampeo de respuesta | `numpy.clip(response, 0, resources_left)` | Sin clampeo explícito |
| Logging | Sí (`log_utils.tee()`) | No |
| Uso | Ejecución del experimento | Entrenamiento y evaluación |

---

## 7. Agregación de Decisiones (Multi-participante)

### 7.1 Métodos Disponibles

Seleccionados en `CONFIG.py`:

```python
AGGREGATION_METHOD = {
    1: 'confidence_weighted_median',    # Robusto a outliers
    2: 'confidence_weighted_mean',      # Promedio ponderado estándar
    3: 'confidence_weighted_mode'       # No implementado (raises NotImplementedError)
}[2]  # ← Selección activa: método 2
```

> **Nota sobre agente único**: Cuando el experimento se ejecuta con un solo
> agente RL (configuración `PLAYER_TYPE = 'RL_AGENT'`), `AllMessages` contiene
> un único participante, por lo que la agregación es trivial (el resultado es
> idéntico al del agente). Estas funciones se conservan por compatibilidad con
> escenarios multi-jugador futuros.

### 7.2 `get_confidence_weighted_mean()`

Implementada en `src/exp_utils.py`. Para cada trial:

```python
TrialResponses   = all_messages[:, t, 0]     # Asignaciones de todos los participantes
TrialConfidences = all_messages[:, t, 1]     # Sus confianzas

# Filtrar respuestas inválidas (confidence == -1)
TrialResponses   = TrialResponses[TrialConfidences != -1]
TrialConfidences = TrialConfidences[TrialConfidences != -1]

# Si todas las confianzas son 0, asignar peso igual
if numpy.sum(TrialConfidences) == 0:
    TrialConfidences[:] = 1.0

# Promedio ponderado
ConfidenceWeightedMean = numpy.average(TrialResponses, weights=TrialConfidences)
```

### 7.3 `get_confidence_weighted_median()`

Implementada usando `statsmodels.stats.weightstats.DescrStatsW`:

```python
from statsmodels.stats.weightstats import DescrStatsW as WeightedStats

# Para cada trial:
TrialResponses   = TrialResponses[TrialConfidences != -1]
TrialConfidences = TrialConfidences[TrialConfidences != -1]

# Si solo una respuesta válida, se duplica para que WeightedStats funcione
if numpy.size(TrialResponses) == 1:
    TrialResponses   = numpy.repeat(TrialResponses, 2)
    TrialConfidences = numpy.repeat(TrialConfidences, 2)

ConfidenceWeightedMedian = WeightedStats(
    data=TrialResponses,
    weights=TrialConfidences
).quantile(probs=[0.5], return_pandas=False)[0]
```

---

## 8. Generación de Reportes

### 8.1 `result_formatter.py`

La función `generate_results_report()` produce dos archivos:

**1. JSON** (`PES_results_<SubjectId>.json`):

```json
{
  "metadata": {"subject_id": "...", "timestamp": "...", "report_type": "PES_Experiment_Results_v2"},
  "configuration": {"total_resources_per_sequence": 39, "num_blocks": 8, "num_sequences": 8},
  "performance_statistics": {
    "overall_mean": 0.72,
    "overall_median": 0.74,
    "overall_std": 0.08,
    "first_block_mean": 0.68,
    "last_block_mean": 0.76,
    "improvement": 0.08,
    "per_block_statistics": [{"block_number": 1, "mean": 0.68, "std": 0.05, ...}, ...]
  }
}
```

**2. PNG multi-panel** (`PES_results_<SubjectId>.png`) con 6 subplots:

| Panel | Contenido |
|-------|-----------|
| 1 | Tendencia de performance por secuencia + media + ±1σ |
| 2 | Histograma de distribución de performance |
| 3 | Box plot de performance por bloque |
| 4 | Media acumulativa de performance |
| 5 | Comparación de medias por bloque (barras) |
| 6 | Tabla resumen de estadísticas |

### 8.2 Archivos Generados por Experimento

**En `outputs/<SubjectId>/`**:

| Archivo | Contenido |
|---------|-----------|
| `PES__<SubjectId>.txt` | Parámetros de configuración del experimento |
| `PES_responses_<SubjectId>.txt` | CSV: InitialSeverity, Response, Confidence, PressEvent_s, ReleaseEvent_s |
| `PES_results_<SubjectId>.json` | Estadísticas calculadas (media, mediana, std, por bloque) |
| `PES_results_<SubjectId>.png` | Visualización multi-panel (6 subplots) |
| `PES_movement_log_<SubjectId>.npy` | Datos de movimiento (dict de bloques → secuencias → trials) |

**En `outputs/`**:

| Archivo | Contenido |
|---------|-----------|
| `PES_log_<SubjectId>.txt` | Log dual (timestamps UTC + mensajes sin color ANSI) |

---

## 9. Entrada de Datos

### 9.1 Archivos en `inputs/`

**`initial_severity.csv`**

- Formato: CSV con un valor de severidad por línea.
- Total: ~360 valores (uno por trial del experimento completo).
- Rango: típicamente 2–9 (enteros).
- Cargado en: `__main__.py` vía `numpy.loadtxt()`.

**`sequence_lengths.csv`**

- Formato: CSV con el número de trials por secuencia.
- Total: 64 valores (8 bloques × 8 secuencias).
- Rango: 3–10, con cada grupo de 8 sumando 45.
- Cargado en: `src/exp_utils.py` → `next_seq_length()`.

### 9.2 Archivos del Modelo Entrenado

**`q.npy`** (requerido para ejecutar el experimento):

| Propiedad | Valor |
|-----------|-------|
| Shape | `(31, 11, 10, 11)` |
| Dimensiones | (recursos_disponibles, trial_no, severidad, acciones) |
| Tipo | `float64` |
| Tamaño | 37,510 entradas |
| Rango de recursos | 0–30 (39 total − 9 pre-asignados) |

**`rewards.npy`**:

| Propiedad | Valor |
|-----------|-------|
| Shape | `(N / 10000,)` donde N = episodios de entrenamiento |
| Contenido | Recompensa promedio cada 10,000 episodios |
| Uso | Visualización de curva de aprendizaje |

### 9.3 Archivos de Entrenamiento (en `inputs/<fecha>_RL_TRAIN/`)

El pipeline `train_rl.py` genera:

| Archivo | Descripción |
|---------|-------------|
| `q_<fecha>.npy` | Q-table entrenada |
| `rewards_<fecha>.npy` | Histórico de recompensas promedio |
| `training_config_<fecha>.txt` | Hiperparámetros y metadatos |
| `confsrl_<fecha>.npy` | Confianza por trial durante evaluación |
| `*.png` (×8) | Plots: baseline aleatorio, performance, confianza |

> **Workflow de deployment**: Tras el entrenamiento, copiar
> `inputs/<fecha>_RL_TRAIN/q_<fecha>.npy` como `inputs/q.npy` y
> `inputs/<fecha>_RL_TRAIN/rewards_<fecha>.npy` como `inputs/rewards.npy`
> para que el experimento pueda ejecutarse.

---

## 10. Tabla de Referencia: Código y Experimento

| Componente | Archivo | Función / Sección | Funcionalidad |
|------------|---------|-------------------|---------------|
| Inicialización | `__main__.py` | `main()` inicio | Validación Q-table |
| Creación sesión | `__main__.py` | `main()` sesión | ID único, logging |
| Asignación mapas | `__main__.py` | `main()` asignación | Seeds reproducibles |
| Asignación trials | `__main__.py` | `main()` asignación | Constraint 45/bloque |
| Loop bloques | `__main__.py` | `main()` loop principal | Iteración 8 bloques |
| Loop secuencias | `__main__.py` | `main()` loop interno | Iteración 8 secuencias |
| Loop trials | `__main__.py` | `main()` loop trials | Iteración 3–10 trials |
| Consulta Q-table | `src/pygameMediator.py` | `provide_rl_agent_response()` | Obtener acción + confianza |
| Update severidad | `src/exp_utils.py` | `get_updated_severity()` | Aplicar fórmula β×sev − α×alloc |
| Evolución secuencia | `src/exp_utils.py` | `get_array_of_sequence_severities_from_allocations()` | Severidades finales |
| Calc performance | `src/exp_utils.py` | `calculate_normalised_...()` | Normalizar [0, 1] |
| Confianza | `src/pygameMediator.py` | `rl_agent_meta_cognitive()` | Entropía meta-cognitiva |
| Entropía | `ext/tools.py` | `entropy_from_pdf()` | Shannon entropy (bits) |
| Reportes | `src/result_formatter.py` | `generate_results_report()` | JSON + PNG |
| Logging | `src/log_utils.py` | `tee()`, `create_ConsoleLog_...()` | Dual terminal + archivo |
| Terminal UI | `src/terminal_utils.py` | `header()`, `section()`, ... | Formato consola ANSI |
| Env Gymnasium | `ext/pandemic.py` | `Pandemic(Env)` | Ambiente Gymnasium |
| Entrenamiento QL | `ext/pandemic.py` | `QLearning()` | Q-Learning tabular |
| Evaluación | `ext/pandemic.py` | `run_experiment()` | Ejecutar secuencias en env |
| Baseline aleatorio | `ext/train_rl.py` | `random_qf()` | Política de comparación |

---

## 11. Flujo Temporal Resumido

```
Inicio
  │
  ├─ Cargar inputs/q.npy e inputs/rewards.npy
  ├─ Inicializar logging → "PES_log_<fecha>_RL_AGENT.txt"
  ├─ Guardar configuración → "PES__<fecha>_RL_AGENT.txt"
  │
  ├─ FOR bloque = 0 to 7:
  │   ├─ Asignar 8 índices de mapa (0–8, sin repetición intra-bloque)
  │   ├─ Asignar 8 cantidades de trials (sum = 45)
  │   │
  │   ├─ FOR secuencia = 0 to 7:
  │   │   ├─ Inicializar 2 ciudades (seed=3 → sev=[4,3], alloc=[3,6])
  │   │   ├─ resources_left = 39 − 9 = 30
  │   │   │
  │   │   ├─ FOR trial = 0 to num_trials:
  │   │   │   ├─ Obtener severidad_nueva desde first_severity[abs_idx]
  │   │   │   ├─ Llamar pygameMediator.provide_rl_agent_response()
  │   │   │   │   ├─ Cargar Q-table desde inputs/q.npy
  │   │   │   │   ├─ state = [resources_left, trial_no, severity]
  │   │   │   │   ├─ Q_values = Q[state[0], state[1], state[2], :]
  │   │   │   │   ├─ action = argmax(Q_values), clampear a resources_left
  │   │   │   │   ├─ confidence = entropy_meta_cognitive(Q_values)
  │   │   │   │   └─ Retornar (confidence, action, rt_hold, rt_release, [])
  │   │   │   ├─ Registrar response, confidence, tiempos en responses.txt
  │   │   │   ├─ Actualizar severidades: get_updated_severity()
  │   │   │   └─ resources_left -= action
  │   │   │
  │   │   ├─ FIN trials
  │   │   ├─ perf = (worst − actual) / (worst − best) → MyPerformances[]
  │   │   ├─ Agregar decisiones (confidence_weighted_mean/median)
  │   │   └─ Loguear performance de secuencia
  │   │
  │   └─ FIN secuencias
  │
  ├─ FIN bloques
  │
  ├─ Generar PES_results_<id>.json + PES_results_<id>.png
  ├─ Guardar PES_movement_log_<id>.npy
  ├─ Cerrar archivos y logging
  │
  └─ Fin
```

---

## 12. Reproducibilidad

### 12.1 Seeds Reproducibles (solo ejecución)

```python
# En asignación de mapas (__main__.py)
numpy.random.seed(100 + blk)

# En inicialización de ciudades (__main__.py)
numpy.random.seed(3)  # → siempre produce severidades [4, 3] y allocs [3, 6]
```

### 12.2 Datos Fijos

```python
USE_FIXED_BLOCK_SEQUENCES = True    # Cargar trials/secuencia desde CSV
RANDOM_INITIAL_SEVERITY = False     # Cargar severidades desde CSV
```

### 12.3 Entrenamiento NO Reproducible

> **Advertencia de reproducibilidad**: El pipeline de entrenamiento
> (`train_rl.py` / `QLearning()`) **no fija ninguna semilla** de números
> aleatorios antes de entrenar. Esto afecta a:
>
> - La **inicialización** de la Q-table (`numpy.random.uniform`).
> - La **exploración ε-greedy** (`numpy.random.random`).
> - La **generación de secuencias** aleatorias durante el entrenamiento
>   (`numpy.random.choice`, `random.randrange`).
>
> En consecuencia, **cada ejecución de entrenamiento produce una Q-table
> distinta** y, por tanto, un agente con comportamiento diferente. Para
> reproducir un resultado específico sería necesario agregar llamadas
> explícitas a `numpy.random.seed()` y `random.seed()` al inicio de
> `QLearning()` o de `train_rl.main()`. Mientras esto no se implemente,
> la única forma de preservar un resultado concreto es conservar los
> archivos `.npy` generados en `inputs/<fecha>_RL_TRAIN/`.

### 12.4 Configuración Registrada

Cada experimento guarda su archivo de configuración:

```
PES__<SubjectId>.txt
```

Contenido: todos los parámetros `CONFIG.*` usados en la ejecución, con formato
tabular (nombre, valor).

---

## 13. Estructura de Archivos del Paquete

```
pes_base/
├── __init__.py          # Config loading, paths, ANSI, env setup, 36 exports
├── __main__.py          # Experiment lifecycle (main function)
├── config/
│   └── CONFIG.py        # Todos los parámetros tunables (10 secciones)
├── doc/
│   ├── explained_pes.md # Este documento
│   ├── explained_rl.md  # Mapeo teoría RL ↔ implementación
│   ├── theory_rl.md     # Teoría de RL para científicos de datos
│   └── PES.__doc__      # Resumen técnico del paquete
├── ext/
│   ├── pandemic.py      # Gymnasium Env (Pandemic), QLearning, run_experiment, meta_cognitive
│   ├── tools.py         # entropy_from_pdf, convert_globalseq_to_seqs, plot_confidences
│   └── train_rl.py      # Pipeline de entrenamiento RL (baseline → train → eval → plots)
├── inputs/
│   ├── initial_severity.csv
│   ├── sequence_lengths.csv
│   ├── q.npy            # Q-table para ejecución (copiar desde train output)
│   ├── rewards.npy      # Rewards para ejecución (copiar desde train output)
│   └── <fecha>_RL_TRAIN/  # Output del entrenamiento
├── outputs/
│   ├── PES_log_*.txt    # Logs globales
│   └── <fecha>_RL_AGENT/  # Output del experimento
└── src/
    ├── exp_utils.py       # Severity, performance, aggregation, sampling (11 funciones)
    ├── log_utils.py       # Dual-stream logging con singleton (5 funciones)
    ├── pygameMediator.py  # Interfaz Q-table → response (2 funciones)
    ├── result_formatter.py # JSON + PNG reports (5 funciones)
    └── terminal_utils.py  # Formato consola con ANSI (12 funciones)
```

---

## Conclusión

El experimento PES implementa un ciclo estructurado y repetible:

1. **Configuración** → Define parámetros fijos en `CONFIG.py`.
2. **Asignación** → Mapea índices de mapa y cantidades de trials a bloques/secuencias.
3. **Ejecución** → Consulta la Q-table entrenada para cada trial vía `pygameMediator`.
4. **Cálculo** → Actualiza severidades con la fórmula dinámica y calcula performance normalizado.
5. **Logging** → Registra todas las decisiones, confianzas y tiempos en archivos duales.
6. **Reportes** → Genera estadísticas JSON y visualizaciones PNG multi-panel.

Este diseño permite reproducir el mismo experimento múltiples veces y comparar
variaciones de agentes o parámetros de forma sistemática.
