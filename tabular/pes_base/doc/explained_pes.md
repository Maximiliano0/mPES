# PES: ExplicaciÃ³n Detallada del Funcionamiento del Proyecto

## 1. IntroducciÃ³n

**PES (Pandemic Experiment Scenario)** es un paquete del workspace **mPES** que
simula escenarios de respuesta a pandemias, donde un agente de
**Reinforcement Learning (Q-Learning tabular)** aprende a optimizar la
asignaciÃ³n limitada de recursos para minimizar la severidad de enfermedades en
mÃºltiples ciudades.

### Objetivo Principal

Entrenar y ejecutar un agente inteligente que tome decisiones estratÃ©gicas sobre
distribuciÃ³n de recursos mÃ©dicos/sanitarios bajo restricciones de disponibilidad.

---

## 2. Estructura JerÃ¡rquica del Experimento

El experimento sigue una estructura anidada estricta definida en `__main__.py`:

```
1 EXPERIMENTO
â”œâ”€ NUM_BLOCKS = 8 BLOQUES
â”‚  â”œâ”€ NUM_SEQUENCES = 8 SECUENCIAS por bloque
â”‚  â”‚  â”œâ”€ NUM_MIN_TRIALS a NUM_MAX_TRIALS (3â€“10) TRIALS por secuencia
â”‚  â”‚  â”‚  â””â”€ 1 ACCIÃ“N DE ASIGNACIÃ“N DE RECURSOS (0â€“10)
```

### Desglose de NÃºmeros

- **Total de bloques**: 8
- **Total de secuencias**: 64 (8 bloques Ã— 8 secuencias)
- **Trials por bloque**: 45 (`TOTAL_NUM_TRIALS_IN_BLOCK`)
- **Trials totales**: ~360 (8 bloques Ã— 45 trials)
- **Total de decisiones**: ~360

### ConfiguraciÃ³n en `CONFIG.py`

```python
NUM_BLOCKS = 8                      # NÃºmero de bloques experimentales
NUM_SEQUENCES = 8                   # Secuencias por bloque
NUM_MIN_TRIALS = 3                  # Trials mÃ­nimos por secuencia
NUM_MAX_TRIALS = 10                 # Trials mÃ¡ximos por secuencia
TOTAL_NUM_TRIALS_IN_BLOCK = 45      # Suma exacta de trials en un bloque
AVAILABLE_RESOURCES_PER_SEQUENCE = 39  # Presupuesto total por secuencia
INIT_NO_OF_CITIES = 2               # Ciudades visibles al inicio
```

### Presupuesto de Recursos

| Concepto | Valor |
|----------|-------|
| Recursos totales por secuencia | 39 (`AVAILABLE_RESOURCES_PER_SEQUENCE`) |
| Recursos pre-asignados (2 ciudades iniciales, seed fija) | 9 (3 + 6) |
| Recursos disponibles para el agente | 30 (= 39 âˆ’ 9) |
| Rango de asignaciÃ³n por trial | 0â€“10 (`MIN/MAX_ALLOCATABLE_RESOURCES`) |

> **Nota**: Las 2 ciudades iniciales siempre reciben asignaciones de 3 y 6
> recursos respectivamente, porque `numpy.random.seed(3)` con
> `INIT_NO_OF_CITIES = 2` produce esos valores determinÃ­sticamente.

---

## 3. Modelo DinÃ¡mico del Escenario PandÃ©mico

### 3.1 La FÃ³rmula de ProgresiÃ³n de Severidad

Implementada en `src/exp_utils.py` â†’ `get_updated_severity()`:

```
new_severity = max(0, Î² Ã— initial_severity âˆ’ Î± Ã— allocated_resources)
```

Donde:

- **Î²** (`SEVERITY_MULTIPLIER`) = 1 + `PANDEMIC_PARAMETER` = **1.4**
  - Representa el crecimiento natural de la pandemia sin intervenciÃ³n.
  - Tasa de crecimiento: 40 % por paso temporal.

- **Î±** (`RESPONSE_MULTIPLIER`) = `PANDEMIC_PARAMETER` = **0.4**
  - Representa la efectividad de los recursos asignados.
  - Cada unidad de recurso reduce 0.4 puntos de severidad.

Ambas constantes se derivan en `__init__.py`:

```python
RESPONSE_MULTIPLIER = PANDEMIC_PARAMETER        # Î± = 0.4
SEVERITY_MULTIPLIER = 1 + PANDEMIC_PARAMETER     # Î² = 1.4
```

### 3.2 Ejemplo NumÃ©rico (una sola ciudad)

```
Entrada: severidad_inicial = 4, recursos_asignados = 5, Î± = 0.4, Î² = 1.4

Paso 1: new_sev = 1.4 Ã— 4   âˆ’ 0.4 Ã— 5 = 5.6  âˆ’ 2.0 = 3.6
Paso 2: new_sev = 1.4 Ã— 3.6 âˆ’ 0.4 Ã— 5 = 5.04 âˆ’ 2.0 = 3.04
Paso 3: new_sev = 1.4 Ã— 3.04âˆ’ 0.4 Ã— 5 = 4.256âˆ’ 2.0 = 2.256
```

La severidad evoluciona a travÃ©s de los trials, y el efecto de los recursos se
acumula secuencialmente.

### 3.3 EvoluciÃ³n Temporal en una Secuencia Completa

La funciÃ³n `get_array_of_sequence_severities_from_allocations()` en
`src/exp_utils.py` calcula la evoluciÃ³n de severidad de **todas** las ciudades
en una secuencia. En cada trial:

1. Se aÃ±ade una nueva ciudad con su severidad inicial.
2. Se aplica la fÃ³rmula de actualizaciÃ³n a **todas** las ciudades visibles
   (nuevas y previas).
3. El vector de severidades se actualiza con los nuevos valores.

**Ejemplo con 3 ciudades** (Î± = 0.4, Î² = 1.4):

```
Severidades iniciales: [3, 4, 8]
Asignaciones:          [5, 6, 4]

Trial 0 (entra Ciudad 1, severidad=3, alloc=5):
  Ciudad 1: 1.4Ã—3 âˆ’ 0.4Ã—5 = 2.20

Trial 1 (entra Ciudad 2, severidad=4, alloc=6):
  Ciudad 1: 1.4Ã—2.20 âˆ’ 0.4Ã—5 = 1.08
  Ciudad 2: 1.4Ã—4    âˆ’ 0.4Ã—6 = 3.20

Trial 2 (entra Ciudad 3, severidad=8, alloc=4):
  Ciudad 1: 1.4Ã—1.08 âˆ’ 0.4Ã—5 = 0    (clipeado a 0)
  Ciudad 2: 1.4Ã—3.20 âˆ’ 0.4Ã—6 = 2.08
  Ciudad 3: 1.4Ã—8    âˆ’ 0.4Ã—4 = 9.60

Severidades finales: [0, 2.08, 9.60]
```

> **Clave**: La asignaciÃ³n de recursos a una ciudad se aplica en **todos** los
> pasos subsiguientes, no solo en el trial en que se asigna. Esto crea una
> dinÃ¡mica temporal compuesta donde las decisiones tempranas tienen mayor
> impacto acumulativo.

---

## 4. Flujo Principal del Experimento

### 4.1 InicializaciÃ³n (`__main__.py`)

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
2. Verifica que `inputs/rewards.npy` (histÃ³rico de recompensas) exista.
3. Carga ambos archivos y valida dimensiones/tipo de datos.
4. Si alguno falta o falla, sugiere ejecutar `python3 -m tabular.pes_base.ext.train_rl`.

> **Nota sobre archivos de entrenamiento**: El pipeline de entrenamiento
> (`train_rl.py`) guarda los archivos en `inputs/<fecha>_RL_TRAIN/q_<fecha>.npy`.
> El experimento busca `inputs/q.npy`. El usuario debe copiar manualmente la
> Q-table entrenada al directorio raÃ­z de `inputs/`.

### 4.2 CreaciÃ³n de SesiÃ³n

```python
experiment_date = datetime.date.today().strftime("%Y-%m-%d")
MySubjectId = f"{experiment_date}_{PLAYER_TYPE}"
# Ejemplo: "2026-02-26_RL_AGENT"

session_outputs_path = os.path.join(OUTPUTS_PATH, MySubjectId)
os.makedirs(session_outputs_path, exist_ok=True)

log_utils.create_ConsoleLog_filehandle_singleton(MySubjectId)
# Crea: outputs/PES_log_2026-02-26_RL_AGENT.txt
```

Se guardan dos archivos al inicio de la sesiÃ³n:

- **SubjectInfo**: `PES__<SubjectId>.txt` â€” parÃ¡metros de configuraciÃ³n.
- **Responses**: `PES_responses_<SubjectId>.txt` â€” decisiones trial a trial.

### 4.3 AsignaciÃ³n de Mapas y Secuencias

**Estructuras de datos principales**:

```python
NumTrials__blocks_x_sequences__2darray  = numpy.zeros((NUM_BLOCKS, NUM_SEQUENCES))
MapIndices__blocks_x_sequences__2darray = numpy.zeros((NUM_BLOCKS, NUM_SEQUENCES))
```

**LÃ³gica de asignaciÃ³n de Ã­ndices de mapa**:

```python
for blk in range(NUM_BLOCKS):
    numpy.random.seed(100 + blk)          # Seed reproducible por bloque
    for seq in range(NUM_SEQUENCES):
        counter_seq = NUM_ATTEMPTS_TO_ASSIGN_SEQ

        while counter_seq > 0:
            b = numpy.random.randint(0, 9)     # Ãndice aleatorio 0â€“8
            if b not in MapIndices__blocks_x_sequences__2darray[blk, :]:
                MapIndices__blocks_x_sequences__2darray[blk, seq] = b
                break                          # AsignaciÃ³n exitosa, avanzar
            counter_seq -= 1
```

> **Nota de implementaciÃ³n**: la semilla `numpy.random.seed(100 + blk)` se
> establece una vez por bloque (fuera del loop de secuencias), de modo que la
> secuencia pseudoaleatoria avanza continuamente entre secuencias. El `break`
> tras cada asignaciÃ³n exitosa garantiza que cada slot reciba el primer Ã­ndice
> vÃ¡lido sin sobrescrituras. Esto produce 8 Ã­ndices Ãºnicos por bloque de un
> pool de 9 posibles (0â€“8), con un patrÃ³n determinÃ­stico y especÃ­fico por
> bloque.

**AsignaciÃ³n de nÃºmero de trials por secuencia**:

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

### 4.5 EjecuciÃ³n de Bloques, Secuencias y Trials

```python
for CurrentBlockIndex in range(NUM_BLOCKS):
    for CurrentSequenceIndex, CurrentSequenceMapIndex in enumerate(CurrentBlockMapIndices):

        # Inicializar recursos y ciudades iniciales
        resources_to_allocate = AVAILABLE_RESOURCES_PER_SEQUENCE  # 39
        numpy.random.seed(3)  # â†’ severidades iniciales [4, 3], allocs [3, 6]

        for c in range(INIT_NO_OF_CITIES):
            init_severity.append(numpy.random.randint(MIN_INIT_SEVERITY, 1 + MAX_INIT_SEVERITY))
            ResourceAllocationsAtCurrentlyVisibleCities.append(
                numpy.random.randint(MIN_INIT_RESOURCES, 1 + MAX_INIT_RESOURCES)
            )

        resources_left = resources_to_allocate - numpy.sum(
            ResourceAllocationsAtCurrentlyVisibleCities
        )  # 39 âˆ’ 9 = 30

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

La funciÃ³n `provide_rl_agent_response()` es la interfaz principal:

1. **Carga** la Q-table desde `inputs/q.npy`.
2. **Obtiene** la severidad de la ciudad actual a partir de las longitudes de
   secuencia y el array global de severidades.
3. **Construye** el estado: `[resources_left, trial_no, severity]`.
4. **Indexa** la Q-table: `Q[resources_idx, city_idx, sever_idx]`.
5. **Llama** a `rl_agent_meta_cognitive()` para obtener la acciÃ³n (argmax),
   confianza (entropÃ­a) y tiempos de reacciÃ³n simulados.
6. **Retorna** `(confidence, response, rt_hold, rt_release, movement)`.

> **Nota**: La Q-table se carga desde disco en **cada** llamada a
> `provide_rl_agent_response()`. Esto es ineficiente pero funcional.

---

## 5. CÃ¡lculo de Severidades y Performance

### 5.1 Severidad Final por Secuencia

Implementado en `get_sequence_severity_from_allocations()`:

```python
def get_sequence_severity_from_allocations(Allocations, InitialSeverities):
    return numpy.sum(
        get_array_of_sequence_severities_from_allocations(Allocations, InitialSeverities)
    )
```

### 5.2 MÃ©trica de Performance Normalizado

Implementada en `calculate_normalised_final_severity_performance_metric()`:

```python
FinalSequenceSeverity     = numpy.sum(SeveritiesFromSequence)

WorstCaseAllocations      = numpy.full_like(SeveritiesFromSequence, MIN_ALLOCATABLE_RESOURCES)
# El "best case" es la asignaciÃ³n factible Ã³ptima bajo el presupuesto
# real del agente (DP por presupuesto), no "max_alloc en cada trial".
BestCaseSequenceSeverity  = _best_feasible_sequence_severity(InitialSequenceSeverities)
WorstCaseSequenceSeverity = get_sequence_severity_from_allocations(WorstCaseAllocations, InitialSequenceSeverities)

Performance = (WorstCaseSequenceSeverity - FinalSequenceSeverity) / \
              (WorstCaseSequenceSeverity - BestCaseSequenceSeverity)
```

**InterpretaciÃ³n**:

| Performance | Significado |
|-------------|-------------|
| 0.0 | Resultado igual al peor caso (sin recursos) |
| 0.5 | Resultado intermedio |
| 1.0 | Resultado Ã³ptimo factible (orÃ¡culo DP bajo el presupuesto real) |

> **Nota tÃ©cnica**: El "best case" se calcula con un DP de mochila acotada
> sobre asignaciones enteras por ciudad, sumando como mucho
> `_FEASIBLE_BUDGET_PER_SEQUENCE = AVAILABLE_RESOURCES_PER_SEQUENCE - 9`
> (=30 con la configuraciÃ³n por defecto). Una polÃ­tica factible puede
> efectivamente alcanzar `Performance = 1.0`.

### 5.3 Ejemplo NumÃ©rico

```
Severidades iniciales: [3, 4, 8]
Asignaciones reales:   [5, 6, 4]
Severidades finales:   [0, 2.08, 9.60]  â†’ FinalSeverity = 11.68

Worst case (alloc = [0, 0, 0]):
  Trial 0: [4.20]
  Trial 1: [5.88, 5.60]
  Trial 2: [8.232, 7.84, 11.20]  â†’ WorstCase = 27.272

Best case (Ã³ptimo factible vÃ­a DP de mochila acotada;
            con budget=30 y 3 ciudades coincide con alloc = [10, 10, 10]):
  Trial 0: [0.20]
  Trial 1: [0, 1.60]
  Trial 2: [0, 0, 7.20]  â†’ BestCase = 7.20

Performance = (27.272 âˆ’ 11.68) / (27.272 âˆ’ 7.20) = 15.592 / 20.072 â‰ˆ 0.777
```

---

## 6. MÃ³dulo de Confianza Meta-cognitiva

### 6.1 `rl_agent_meta_cognitive()`

Esta funciÃ³n existe en **dos ubicaciones** con implementaciones similares:

- `src/pygameMediator.py`: usada durante la **ejecuciÃ³n** del experimento
  (`python3 -m tabular.pes_base`).
- `ext/pandemic.py`: usada durante el **entrenamiento** (`python3 -m tabular.pes_base.ext.train_rl`).

**Algoritmo**:

```python
def rl_agent_meta_cognitive(options, resources_left, response_timeout):
    # 1. Calcular entropÃ­as de referencia
    m_entropy = entropy_from_pdf([1, 0, 0, ..., 0])   # MÃ­nima (determinÃ­stica)
    M_entropy = entropy_from_pdf([1, 1, 1, ..., 1])   # MÃ¡xima (uniforme)

    # 2. Filtrar opciones infactibles (acciÃ³n > recursos disponibles)
    #    Se asigna un valor centinela muy negativo (-1e9) para que esas
    #    acciones nunca ganen el argmax, incluso cuando los Q-values
    #    aprendidos sean tÃ­picamente negativos.
    options[acciÃ³n > resources_left] = -1e9

    # 3. Calcular entropÃ­a de las opciones filtradas
    dec_entropy = entropy_from_pdf(options)

    # 4. Normalizar confianza a [0, 1]
    confidence = (dec_entropy - M_entropy) / (m_entropy - M_entropy)

    # 5. Seleccionar acciÃ³n (greedy)
    response = numpy.argmax(options)

    # 6. Mapear confianza a tiempos de reacciÃ³n
    map_to_response_time = lambda x: x * (-2) + 1
    mu = int(map_to_response_time(confidence) * 10)
    rt_hold    = numpy.clip(numpy.random.normal(mu, 3, 1)[0], 0, response_timeout/1000)
    rt_release = numpy.clip(rt_hold + numpy.random.normal(mu, 1, 1)[0], 0, response_timeout/1000)

    return response, confidence, rt_hold, rt_release
```

**PropÃ³sito**: Simular un agente que no solo toma decisiones Ã³ptimas, sino que
refleja incertidumbre (baja confianza) con tiempos de reacciÃ³n mÃ¡s largos.

### 6.2 EntropÃ­a (`entropy_from_pdf()` en `ext/tools.py`)

```python
def entropy_from_pdf(pdf):
    pdf = pdf + numpy.abs(numpy.min(pdf))     # Desplazar a positivos
    p = pdf / numpy.sum(pdf)                  # Normalizar a probabilidad
    p[p == 0] += 0.000001                     # Evitar log(0)
    H = -numpy.dot(p, numpy.log2(p))          # EntropÃ­a de Shannon (bits)
    return H
```

| DistribuciÃ³n | EntropÃ­a (11 acciones) | Confianza |
|-------------|------------------------|-----------|
| DeterminÃ­stica `[1,0,...,0]` | â‰ˆ 0 bits | â‰ˆ 1.0 (alta) |
| Uniforme `[1,1,...,1]` | â‰ˆ 3.46 bits | â‰ˆ 0.0 (baja) |

### 6.3 Diferencias entre las dos implementaciones

| Aspecto | `pygameMediator.py` | `pandemic.py` |
|---------|---------------------|---------------|
| TamaÃ±o vectores referencia | Fijo: `numpy.zeros((11,))` | DinÃ¡mico: `numpy.zeros((len(options),))` |
| Clampeo de respuesta | `numpy.clip(response, 0, resources_left)` | Sin clampeo explÃ­cito |
| Logging | SÃ­ (`log_utils.tee()`) | No |
| Uso | EjecuciÃ³n del experimento | Entrenamiento y evaluaciÃ³n |

---

## 7. AgregaciÃ³n de Decisiones (Multi-participante)

### 7.1 MÃ©todos Disponibles

Seleccionados en `CONFIG.py`:

```python
AGGREGATION_METHOD = {
    1: 'confidence_weighted_median',    # Robusto a outliers
    2: 'confidence_weighted_mean',      # Promedio ponderado estÃ¡ndar
    3: 'confidence_weighted_mode'       # No implementado (raises NotImplementedError)
}[2]  # â† SelecciÃ³n activa: mÃ©todo 2
```

> **Nota sobre agente Ãºnico**: Cuando el experimento se ejecuta con un solo
> agente RL (configuraciÃ³n `PLAYER_TYPE = 'RL_AGENT'`), `AllMessages` contiene
> un Ãºnico participante, por lo que la agregaciÃ³n es trivial (el resultado es
> idÃ©ntico al del agente). Estas funciones se conservan por compatibilidad con
> escenarios multi-jugador futuros.

### 7.2 `get_confidence_weighted_mean()`

Implementada en `src/exp_utils.py`. Para cada trial:

```python
TrialResponses   = all_messages[:, t, 0]     # Asignaciones de todos los participantes
TrialConfidences = all_messages[:, t, 1]     # Sus confianzas

# Filtrar respuestas invÃ¡lidas (confidence == -1)
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

# Si solo una respuesta vÃ¡lida, se duplica para que WeightedStats funcione
if numpy.size(TrialResponses) == 1:
    TrialResponses   = numpy.repeat(TrialResponses, 2)
    TrialConfidences = numpy.repeat(TrialConfidences, 2)

ConfidenceWeightedMedian = WeightedStats(
    data=TrialResponses,
    weights=TrialConfidences
).quantile(probs=[0.5], return_pandas=False)[0]
```

---

## 8. GeneraciÃ³n de Reportes

### 8.1 `result_formatter.py`

La funciÃ³n `generate_results_report()` produce dos archivos:

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
| 1 | Tendencia de performance por secuencia + media + Â±1Ïƒ |
| 2 | Histograma de distribuciÃ³n de performance |
| 3 | Box plot de performance por bloque |
| 4 | Media acumulativa de performance |
| 5 | ComparaciÃ³n de medias por bloque (barras) |
| 6 | Tabla resumen de estadÃ­sticas |

### 8.2 Archivos Generados por Experimento

**En `outputs/<SubjectId>/`**:

| Archivo | Contenido |
|---------|-----------|
| `PES__<SubjectId>.txt` | ParÃ¡metros de configuraciÃ³n del experimento |
| `PES_responses_<SubjectId>.txt` | CSV: InitialSeverity, Response, Confidence, PressEvent_s, ReleaseEvent_s |
| `PES_results_<SubjectId>.json` | EstadÃ­sticas calculadas (media, mediana, std, por bloque) |
| `PES_results_<SubjectId>.png` | VisualizaciÃ³n multi-panel (6 subplots) |
| `PES_movement_log_<SubjectId>.npy` | Datos de movimiento (dict de bloques â†’ secuencias â†’ trials) |

**En `outputs/`**:

| Archivo | Contenido |
|---------|-----------|
| `PES_log_<SubjectId>.txt` | Log dual (timestamps UTC + mensajes sin color ANSI) |

---

## 9. Entrada de Datos

### 9.1 Archivos en `inputs/`

**`initial_severity.csv`**

- Formato: CSV con un valor de severidad por lÃ­nea.
- Total: ~360 valores (uno por trial del experimento completo).
- Rango: tÃ­picamente 2â€“9 (enteros).
- Cargado en: `__main__.py` vÃ­a `numpy.loadtxt()`.

**`sequence_lengths.csv`**

- Formato: CSV con el nÃºmero de trials por secuencia.
- Total: 64 valores (8 bloques Ã— 8 secuencias).
- Rango: 3â€“10, con cada grupo de 8 sumando 45.
- Cargado en: `src/exp_utils.py` â†’ `next_seq_length()`.

### 9.2 Archivos del Modelo Entrenado

**`q.npy`** (requerido para ejecutar el experimento):

| Propiedad | Valor |
|-----------|-------|
| Shape | `(31, 11, 10, 11)` |
| Dimensiones | (recursos_disponibles, trial_no, severidad, acciones) |
| Tipo | `float64` |
| TamaÃ±o | 37,510 entradas |
| Rango de recursos | 0â€“30 (39 total âˆ’ 9 pre-asignados) |

**`rewards.npy`**:

| Propiedad | Valor |
|-----------|-------|
| Shape | `(N / 10000,)` donde N = episodios de entrenamiento |
| Contenido | Recompensa promedio cada 10,000 episodios |
| Uso | VisualizaciÃ³n de curva de aprendizaje |

### 9.3 Archivos de Entrenamiento (en `inputs/<fecha>_RL_TRAIN/`)

El pipeline `train_rl.py` genera:

| Archivo | DescripciÃ³n |
|---------|-------------|
| `q_<fecha>.npy` | Q-table entrenada |
| `rewards_<fecha>.npy` | HistÃ³rico de recompensas promedio |
| `training_config_<fecha>.txt` | HiperparÃ¡metros y metadatos |
| `confsrl_<fecha>.npy` | Confianza por trial durante evaluaciÃ³n |
| `*.png` (Ã—8) | Plots: baseline aleatorio, performance, confianza |

> **Workflow de deployment**: Tras el entrenamiento, copiar
> `inputs/<fecha>_RL_TRAIN/q_<fecha>.npy` como `inputs/q.npy` y
> `inputs/<fecha>_RL_TRAIN/rewards_<fecha>.npy` como `inputs/rewards.npy`
> para que el experimento pueda ejecutarse.

---

## 10. Tabla de Referencia: CÃ³digo y Experimento

| Componente | Archivo | FunciÃ³n / SecciÃ³n | Funcionalidad |
|------------|---------|-------------------|---------------|
| InicializaciÃ³n | `__main__.py` | `main()` inicio | ValidaciÃ³n Q-table |
| CreaciÃ³n sesiÃ³n | `__main__.py` | `main()` sesiÃ³n | ID Ãºnico, logging |
| AsignaciÃ³n mapas | `__main__.py` | `main()` asignaciÃ³n | Seeds reproducibles |
| AsignaciÃ³n trials | `__main__.py` | `main()` asignaciÃ³n | Constraint 45/bloque |
| Loop bloques | `__main__.py` | `main()` loop principal | IteraciÃ³n 8 bloques |
| Loop secuencias | `__main__.py` | `main()` loop interno | IteraciÃ³n 8 secuencias |
| Loop trials | `__main__.py` | `main()` loop trials | IteraciÃ³n 3â€“10 trials |
| Consulta Q-table | `src/pygameMediator.py` | `provide_rl_agent_response()` | Obtener acciÃ³n + confianza |
| Update severidad | `src/exp_utils.py` | `get_updated_severity()` | Aplicar fÃ³rmula Î²Ã—sev âˆ’ Î±Ã—alloc |
| EvoluciÃ³n secuencia | `src/exp_utils.py` | `get_array_of_sequence_severities_from_allocations()` | Severidades finales |
| Calc performance | `src/exp_utils.py` | `calculate_normalised_...()` | Normalizar [0, 1] |
| Confianza | `src/pygameMediator.py` | `rl_agent_meta_cognitive()` | EntropÃ­a meta-cognitiva |
| EntropÃ­a | `ext/tools.py` | `entropy_from_pdf()` | Shannon entropy (bits) |
| Reportes | `src/result_formatter.py` | `generate_results_report()` | JSON + PNG |
| Logging | `src/log_utils.py` | `tee()`, `create_ConsoleLog_...()` | Dual terminal + archivo |
| Terminal UI | `src/terminal_utils.py` | `header()`, `section()`, ... | Formato consola ANSI |
| Env Gymnasium | `ext/pandemic.py` | `Pandemic(Env)` | Ambiente Gymnasium |
| Entrenamiento QL | `ext/pandemic.py` | `QLearning()` | Q-Learning tabular |
| EvaluaciÃ³n | `ext/pandemic.py` | `run_experiment()` | Ejecutar secuencias en env |
| Baseline aleatorio | `ext/train_rl.py` | `random_qf()` | PolÃ­tica de comparaciÃ³n |

---

## 11. Flujo Temporal Resumido

```
Inicio
  â”‚
  â”œâ”€ Cargar inputs/q.npy e inputs/rewards.npy
  â”œâ”€ Inicializar logging â†’ "PES_log_<fecha>_RL_AGENT.txt"
  â”œâ”€ Guardar configuraciÃ³n â†’ "PES__<fecha>_RL_AGENT.txt"
  â”‚
  â”œâ”€ FOR bloque = 0 to 7:
  â”‚   â”œâ”€ Asignar 8 Ã­ndices de mapa (0â€“8, sin repeticiÃ³n intra-bloque)
  â”‚   â”œâ”€ Asignar 8 cantidades de trials (sum = 45)
  â”‚   â”‚
  â”‚   â”œâ”€ FOR secuencia = 0 to 7:
  â”‚   â”‚   â”œâ”€ Inicializar 2 ciudades (seed=3 â†’ sev=[4,3], alloc=[3,6])
  â”‚   â”‚   â”œâ”€ resources_left = 39 âˆ’ 9 = 30
  â”‚   â”‚   â”‚
  â”‚   â”‚   â”œâ”€ FOR trial = 0 to num_trials:
  â”‚   â”‚   â”‚   â”œâ”€ Obtener severidad_nueva desde first_severity[abs_idx]
  â”‚   â”‚   â”‚   â”œâ”€ Llamar pygameMediator.provide_rl_agent_response()
  â”‚   â”‚   â”‚   â”‚   â”œâ”€ Cargar Q-table desde inputs/q.npy
  â”‚   â”‚   â”‚   â”‚   â”œâ”€ state = [resources_left, trial_no, severity]
  â”‚   â”‚   â”‚   â”‚   â”œâ”€ Q_values = Q[state[0], state[1], state[2], :]
  â”‚   â”‚   â”‚   â”‚   â”œâ”€ action = argmax(Q_values), clampear a resources_left
  â”‚   â”‚   â”‚   â”‚   â”œâ”€ confidence = entropy_meta_cognitive(Q_values)
  â”‚   â”‚   â”‚   â”‚   â””â”€ Retornar (confidence, action, rt_hold, rt_release, [])
  â”‚   â”‚   â”‚   â”œâ”€ Registrar response, confidence, tiempos en responses.txt
  â”‚   â”‚   â”‚   â”œâ”€ Actualizar severidades: get_updated_severity()
  â”‚   â”‚   â”‚   â””â”€ resources_left -= action
  â”‚   â”‚   â”‚
  â”‚   â”‚   â”œâ”€ FIN trials
  â”‚   â”‚   â”œâ”€ perf = (worst âˆ’ actual) / (worst âˆ’ best) â†’ MyPerformances[]
  â”‚   â”‚   â”œâ”€ Agregar decisiones (confidence_weighted_mean/median)
  â”‚   â”‚   â””â”€ Loguear performance de secuencia
  â”‚   â”‚
  â”‚   â””â”€ FIN secuencias
  â”‚
  â”œâ”€ FIN bloques
  â”‚
  â”œâ”€ Generar PES_results_<id>.json + PES_results_<id>.png
  â”œâ”€ Guardar PES_movement_log_<id>.npy
  â”œâ”€ Cerrar archivos y logging
  â”‚
  â””â”€ Fin
```

---

## 12. Reproducibilidad

### 12.1 Seeds Reproducibles (solo ejecuciÃ³n)

```python
# En asignaciÃ³n de mapas (__main__.py)
numpy.random.seed(100 + blk)

# En inicializaciÃ³n de ciudades (__main__.py)
numpy.random.seed(3)  # â†’ siempre produce severidades [4, 3] y allocs [3, 6]
```

### 12.2 Datos Fijos

```python
USE_FIXED_BLOCK_SEQUENCES = True    # Cargar trials/secuencia desde CSV
RANDOM_INITIAL_SEVERITY = False     # Cargar severidades desde CSV
```

### 12.3 Entrenamiento Reproducible (semilla fija)

La funciÃ³n `QLearning()` (en `ext/pandemic.py`) acepta un parÃ¡metro
`seed`. Cuando se proporciona, fija las semillas de `numpy.random` y
`random` antes de inicializar la Q-table, por lo que afecta a:

- La **inicializaciÃ³n** de la Q-table (`numpy.random.uniform`).
- La **exploraciÃ³n Îµ-greedy** (`numpy.random.random` /
  `numpy.random.randint`).
- La **generaciÃ³n de secuencias** aleatorias durante el entrenamiento
  (`numpy.random.choice`, `random.randrange`).

El pipeline `train_rl.py` invoca `QLearning(..., seed=SEED)` con
`SEED = 42` (definido en `config/CONFIG.py`), por lo que **dos
ejecuciones consecutivas con los mismos hiperparÃ¡metros producen la
misma Q-table**.

Para obtener entrenamientos no determinÃ­sticos (p. ej. para promediar
resultados sobre mÃºltiples semillas), basta con cambiar `SEED` en
`config/CONFIG.py` o llamar manualmente a `QLearning(..., seed=None)`.

### 12.4 ConfiguraciÃ³n Registrada

Cada experimento guarda su archivo de configuraciÃ³n:

```
PES__<SubjectId>.txt
```

Contenido: todos los parÃ¡metros `CONFIG.*` usados en la ejecuciÃ³n, con formato
tabular (nombre, valor).

---

## 13. Estructura de Archivos del Paquete

```
pes_base/
â”œâ”€â”€ __init__.py          # Config loading, paths, ANSI, env setup, 36 exports
â”œâ”€â”€ __main__.py          # Experiment lifecycle (main function)
â”œâ”€â”€ config/
â”‚   â””â”€â”€ CONFIG.py        # Todos los parÃ¡metros tunables (10 secciones)
â”œâ”€â”€ doc/
â”‚   â”œâ”€â”€ explained_pes.md # Este documento
â”‚   â”œâ”€â”€ explained_rl.md  # Mapeo teorÃ­a RL â†” implementaciÃ³n
â”‚   â”œâ”€â”€ theory_rl.md     # TeorÃ­a de RL para cientÃ­ficos de datos
â”‚   â””â”€â”€ pes.__doc__      # Resumen tÃ©cnico del paquete
â”œâ”€â”€ ext/
â”‚   â”œâ”€â”€ pandemic.py      # Gymnasium Env (Pandemic), QLearning, run_experiment, meta_cognitive
â”‚   â”œâ”€â”€ tools.py         # entropy_from_pdf, convert_globalseq_to_seqs, plot_confidences
â”‚   â””â”€â”€ train_rl.py      # Pipeline de entrenamiento RL (baseline â†’ train â†’ eval â†’ plots)
â”œâ”€â”€ inputs/
â”‚   â”œâ”€â”€ initial_severity.csv
â”‚   â”œâ”€â”€ sequence_lengths.csv
â”‚   â”œâ”€â”€ q.npy            # Q-table para ejecuciÃ³n (copiar desde train output)
â”‚   â”œâ”€â”€ rewards.npy      # Rewards para ejecuciÃ³n (copiar desde train output)
â”‚   â””â”€â”€ <fecha>_RL_TRAIN/  # Output del entrenamiento
â”œâ”€â”€ outputs/
â”‚   â”œâ”€â”€ PES_log_*.txt    # Logs globales
â”‚   â””â”€â”€ <fecha>_RL_AGENT/  # Output del experimento
â””â”€â”€ src/
    â”œâ”€â”€ exp_utils.py       # Severity, performance, aggregation, sampling (11 funciones)
    â”œâ”€â”€ log_utils.py       # Dual-stream logging con singleton (5 funciones)
    â”œâ”€â”€ pygameMediator.py  # Interfaz Q-table â†’ response (2 funciones)
    â”œâ”€â”€ result_formatter.py # JSON + PNG reports (5 funciones)
    â””â”€â”€ terminal_utils.py  # Formato consola con ANSI (12 funciones)
```

---

## ConclusiÃ³n

El experimento PES implementa un ciclo estructurado y repetible:

1. **ConfiguraciÃ³n** â†’ Define parÃ¡metros fijos en `CONFIG.py`.
2. **AsignaciÃ³n** â†’ Mapea Ã­ndices de mapa y cantidades de trials a bloques/secuencias.
3. **EjecuciÃ³n** â†’ Consulta la Q-table entrenada para cada trial vÃ­a `pygameMediator`.
4. **CÃ¡lculo** â†’ Actualiza severidades con la fÃ³rmula dinÃ¡mica y calcula performance normalizado.
5. **Logging** â†’ Registra todas las decisiones, confianzas y tiempos en archivos duales.
6. **Reportes** â†’ Genera estadÃ­sticas JSON y visualizaciones PNG multi-panel.

Este diseÃ±o permite reproducir el mismo experimento mÃºltiples veces y comparar
variaciones de agentes o parÃ¡metros de forma sistemÃ¡tica.
