# Mapeo TeorÃ­a RL â†” ImplementaciÃ³n en pes

## 1. IntroducciÃ³n

Este documento conecta la **teorÃ­a de Reinforcement Learning (RL)** con su
**implementaciÃ³n concreta** en el paquete pes. Para cada concepto teÃ³rico se
indica la variable, funciÃ³n o lÃ­nea de cÃ³digo correspondiente.

Para la teorÃ­a pura, consultar `theory_rl.md`.
Para la descripciÃ³n funcional del experimento, consultar `explained_pes.md`.

---

## 2. Componentes del MDP (Markov Decision Process)

### 2.1 DefiniciÃ³n Formal

Un MDP se define como la tupla $(S, A, P, R, \gamma)$:

| Componente | SÃ­mbolo | ImplementaciÃ³n pes |
|------------|---------|-------------------|
| Estados | $S$ | `(resources_left, trial_no, severity)` â€” 3 dimensiones discretas |
| Acciones | $A$ | `{0, 1, 2, ..., 10}` â€” recursos a asignar |
| Transiciones | $P(s' \mid s, a)$ | DeterminÃ­sticas: `env.step(action)` en `pandemic.py` |
| Recompensas | $R(s, a)$ | $-\sum_{i} \text{severities}_i$ â€” negativo de suma de severidades |
| Factor de descuento | $\gamma$ | `discount_factor = 0.9` en `train_rl.py` |

### 2.2 Espacio de Estados

Implementado en `ext/pandemic.py` â†’ `Pandemic.__init__()`:

```python
# Recursos disponibles: 0 a 30 (39 total - 9 pre-asignados)
self.max_resources = AVAILABLE_RESOURCES_PER_SEQUENCE - 9   # = 30
self.available_resources_states = self.max_resources + 1     # = 31

# NÃºmero de trial: 0 a 10
self.max_seq_length = NUM_MAX_TRIALS                         # = 10
self.trial_no_states = self.max_seq_length + 1               # = 11

# Severidad: 0 a 9
self.max_severity = MAX_SEVERITY                             # = 9
self.severity_states = self.max_severity + 1                 # = 10
```

**Cardinalidad del espacio de estados**:

$$|S| = 31 \times 11 \times 10 = 3{,}410 \text{ estados}$$

**Significado de cada dimensiÃ³n**:

| DimensiÃ³n | Rango | InterpretaciÃ³n |
|-----------|-------|---------------|
| `resources_left` | 0â€“30 | Recursos restantes en la secuencia actual |
| `trial_no` | 0â€“10 | Ãndice del trial actual dentro de la secuencia |
| `severity` | 0â€“9 | Severidad inicial de la ciudad entrante (entera) |

### 2.3 Espacio de Acciones

```python
self.action_space = spaces.Discrete(self.max_allocation + 1)  # Discrete(11)
```

Cada acciÃ³n $a \in \{0, 1, ..., 10\}$ representa la cantidad de recursos a
asignar a la ciudad actual. Si `a > resources_left`, se clampea:

```python
# En Pandemic.step():
if (self.available_resources - action) <= 0:
    action = self.available_resources
```

### 2.4 FunciÃ³n de TransiciÃ³n

La transiciÃ³n es **determinÃ­stica** (dado el estado y la acciÃ³n, el siguiente
estado es Ãºnico):

$$s_{t+1} = f(s_t, a_t)$$

Implementada en `Pandemic.step()`:

```python
self.available_resources -= action               # Reducir recursos
self.resources.append(action)                    # Registrar acciÃ³n
self.severities = get_updated_severity(...)      # Actualizar severidades
self.iteration += 1                              # Avanzar trial

# Nuevo estado
if self.iteration == self.seq_length:
    done = True
    new_severity = 0
else:
    new_severity = self.new_city()
    self.severities.append(new_severity)

return [self.available_resources, self.iteration, int(new_severity)], reward, done, False, {}
```

### 2.5 FunciÃ³n de Recompensa

```python
# En Pandemic.step():
reward = (-1) * numpy.sum(self.severities)
```

La recompensa es el **negativo de la suma de las severidades** de todas las
ciudades visibles en el estado actual, despuÃ©s de aplicar la actualizaciÃ³n. Esto
incentiva al agente a minimizar la severidad total.

**Caso terminal**: Cuando `done == True`, la recompensa es la severidad final
total (negativa) de toda la secuencia.

**Caso no-terminal**: La recompensa intermedia refleja el costo acumulado de
severidad en cada paso, guiando al agente paso a paso.

---

## 3. Q-Learning: TeorÃ­a e ImplementaciÃ³n

### 3.1 EcuaciÃ³n de ActualizaciÃ³n

**FÃ³rmula teÃ³rica (Bellman temporal-difference)**:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

**ImplementaciÃ³n** en `ext/pandemic.py` â†’ `QLearning()`:

```python
# Estado no-terminal:
delta = learning * (reward +
                    discount * numpy.max(Q[state2_idx[0],
                                           state2_idx[1],
                                           state2_idx[2]]) -
                    Q[state_idx[0], state_idx[1], state_idx[2], action])
Q[state_idx[0], state_idx[1], state_idx[2], action] += delta

# Estado terminal (done == True):
Q[state_idx[0], state_idx[1], state_idx[2], action] = reward
```

> **Nota**: En el estado terminal, el Q-value se asigna directamente como la
> recompensa (sin tÃ©rmino $\gamma \max Q(s')$), ya que no hay estado futuro.

### 3.2 HiperparÃ¡metros

Definidos en `ext/train_rl.py`:

| SÃ­mbolo | Nombre | Valor | Variable |
|---------|--------|-------|----------|
| $\alpha$ | Learning rate | 0.2 | `learning_rate` |
| $\gamma$ | Discount factor | 0.9 | `discount_factor` |
| $\varepsilon_0$ | Epsilon inicial | 0.8 | `epsilon_initial` |
| $\varepsilon_{\min}$ | Epsilon mÃ­nimo | 0.0 | `epsilon_min` |
| $N$ | Episodios | 1,000,000 | `num_episodes` |

```python
learning_rate = 0.2
discount_factor = 0.9
epsilon_initial = 0.8
epsilon_min = 0
num_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 1000000
```

> **Nota**: El nÃºmero de episodios por defecto es **1,000,000** (un millÃ³n).
> Se puede sobreescribir pasando un argumento por lÃ­nea de comandos:
> `python3 -m tabular.pes_base.ext.train_rl 500000`.

### 3.3 InicializaciÃ³n de la Q-Table

```python
Q = numpy.random.uniform(low=-1, high=1,
                          size=(env.available_resources_states,   # 31
                                env.trial_no_states,              # 11
                                env.severity_states,              # 10
                                env.action_space.n))              # 11
```

- **Shape**: `(31, 11, 10, 11)` = **37,510 entradas**
- **InicializaciÃ³n**: Valores aleatorios uniformes en $[-1, 1]$
- **Tipo**: `float64`

La inicializaciÃ³n aleatoria promueve exploraciÃ³n inicial al dar Q-values
optimistas o pesimistas para diferentes pares estado-acciÃ³n.

### 3.4 Clipping de Ãndices

El estado observado por el agente se clampea a los lÃ­mites de la Q-table para
evitar errores de Ã­ndice:

```python
state_idx = [min(int(state[0]), env.available_resources_states - 1),
             min(int(state[1]), env.trial_no_states - 1),
             min(int(state[2]), env.severity_states - 1)]
```

Esto es necesario porque las severidades y recursos podrÃ­an exceder los
rangos discretos en casos extremos.

---

## 4. ExploraciÃ³n vs. ExplotaciÃ³n

### 4.1 PolÃ­tica Îµ-Greedy

**FÃ³rmula**:

$$
\pi(a|s) = \begin{cases}
\arg\max_a Q(s, a) & \text{con probabilidad } 1 - \varepsilon \\
a \sim \text{Uniform}(0, 10) & \text{con probabilidad } \varepsilon
\end{cases}
$$

**ImplementaciÃ³n**:

```python
if numpy.random.random() < 1 - epsilon and state[0] is not None:
    action = numpy.argmax(Q[state_idx[0], state_idx[1], state_idx[2]])
else:
    action = numpy.random.randint(0, env.action_space.n)
```

### 4.2 Decaimiento Lineal de Îµ

```python
reduction = (epsilon - min_eps) / episodes
# = (0.8 - 0.0) / 1,000,000 = 0.0000008 por episodio

# Al final de cada episodio:
if epsilon > min_eps:
    epsilon -= reduction
```

**Tabla de exploraciÃ³n durante el entrenamiento** (1,000,000 episodios):

| Episodio | Îµ | Comportamiento |
|----------|---|---------------|
| 1 | 0.800 | 80 % exploraciÃ³n |
| 100,000 | 0.720 | 72 % exploraciÃ³n |
| 250,000 | 0.600 | 60 % exploraciÃ³n |
| 500,000 | 0.400 | 40 % exploraciÃ³n |
| 750,000 | 0.200 | 20 % exploraciÃ³n |
| 1,000,000 | 0.000 | 100 % explotaciÃ³n |

El decaimiento es lento y lineal. La transiciÃ³n de exploraciÃ³n a explotaciÃ³n es
gradual, permitiendo que el agente explore ampliamente durante la primera mitad
del entrenamiento y explote progresivamente en la segunda mitad.

---

## 5. Estructura de un Episodio de Entrenamiento

### 5.1 Flujo de un Episodio

```python
for i in range(episodes):            # 1,000,000 episodios
    done = False
    tot_reward, reward = 0, 0
    env.random_sequence()             # Generar secuencia aleatoria
    state, _ = env.reset()               # â†’ [max_resources, 0, severity_0]

    while done != True:
        # 1. SelecciÃ³n de acciÃ³n (Îµ-greedy)
        if numpy.random.random() < 1 - epsilon:
            action = numpy.argmax(Q[s0, s1, s2])
        else:
            action = numpy.random.randint(0, 11)

        # 2. Ejecutar acciÃ³n
        state2, reward, done, _, _ = env.step(action)

        # 3. Actualizar Q-table
        if done:
            Q[s0, s1, s2, action] = reward
        else:
            Q[s0, s1, s2, action] += Î± * (r + Î³ * max(Q[s'0, s'1, s'2]) - Q[s0, s1, s2, action])

        # 4. Avanzar estado
        tot_reward += reward
        state = state2

    # 5. Decaer epsilon
    epsilon -= reduction
```

### 5.2 GeneraciÃ³n de Secuencias Aleatorias

Durante el entrenamiento, las secuencias se generan aleatoriamente usando las
distribuciones de probabilidad aprendidas de los datos:

```python
# En Pandemic.random_sequence():
self.seq_length = numpy.random.choice(
    number_cities_prob[:, 0],
    p=number_cities_prob[:, 1]
)  # Muestrea longitud de secuencia (3â€“10 trials)

self.initial_severities = numpy.random.choice(
    severity_prob[:, 0],
    size=(self.seq_length,),
    p=severity_prob[:, 1]
)  # Muestrea severidades iniciales
```

Estas distribuciones se calculan a partir de los datos reales en
`initial_severity.csv` y `sequence_lengths.csv`, asegurando que el
entrenamiento refleje la distribuciÃ³n del experimento.

### 5.3 Tracking de Progreso

```python
if (i + 1) % 10000 == 0:
    ave_reward = numpy.mean(reward_list)
    ave_reward_list.append(ave_reward)
    reward_list = []
    print(f'Episode {i+1} Average Reward: {ave_reward}')
```

El reward promedio se calcula cada 10,000 episodios y se almacena en
`ave_reward_list`, que luego se guarda como `rewards_<fecha>.npy`.

---

## 6. Pipeline Completo de Entrenamiento (`train_rl.py`)

### 6.1 Etapas del Pipeline

```
python3 -m tabular.pes_base.ext.train_rl [num_episodes]
```

| Etapa | DescripciÃ³n | Output |
|-------|-------------|--------|
| 1. Carga datos | Lee CSV de severidades y longitudes | Arrays numpy |
| 2. Baseline aleatorio | Ejecuta 64 secuencias con polÃ­tica random | 2 PNGs |
| 3. Entrenamiento Q-Learning | 1,000,000 episodios (defecto) | Q-table, rewards |
| 4. Guardar artefactos | Q-table, rewards, config en directorio fechado | 3 archivos |
| 5. EvaluaciÃ³n | Ejecuta 64 secuencias con polÃ­tica entrenada | seqs, perfs, confs |
| 6. VisualizaciÃ³n | 6 plots de performance, severidad y confianza | 6 PNGs |

### 6.2 Baseline Aleatorio

```python
def random_qf(env, state, seqid):
    """Return a uniformly random action (baseline policy)."""
    return env.sample()

seqs1, perfs1, _ = run_experiment(env, random_qf, False, trials_per_sequence, sevs)
```

El baseline utiliza asignaciones generadas aleatoriamente por el entorno
(las severidades y longitudes de secuencia sÃ­ provienen de los archivos CSV),
proporcionando un punto de comparaciÃ³n.

### 6.3 EvaluaciÃ³n Post-Entrenamiento

```python
def eval_qf(env, state, seqid):
    """Select an action from the trained Q-table with confidence tracking."""
    response, confidence, rt_hold, rt_release = rl_agent_meta_cognitive(
        Q[state[0], state[1], int(state[2])], state[0], 10000
    )
    if state[0] == 0:
        confidence = -1.0    # Sin recursos â†’ confidence invÃ¡lida
    confsrl.append(confidence)
    return response

seqs, perfs, _ = run_experiment(env, eval_qf, False, trials_per_sequence, sevs)
```

La evaluaciÃ³n usa `rl_agent_meta_cognitive()` para obtener tanto la acciÃ³n
(greedy, sin exploraciÃ³n) como la confianza del agente.

### 6.4 Archivos de Salida

Directorio: `inputs/<fecha>_RL_TRAIN/`

| Archivo | Contenido |
|---------|-----------|
| `q_<fecha>.npy` | Q-table `(31, 11, 10, 11)` float64 |
| `rewards_<fecha>.npy` | Rewards promedio cada 10k episodios |
| `training_config_<fecha>.txt` | HiperparÃ¡metros, shapes, metadatos |
| `confsrl_<fecha>.npy` | Confianzas durante evaluaciÃ³n |
| `random_player_sequence_performance_<fecha>.png` | Severidad por secuencia (baseline) |
| `random_player_normalised_performance_<fecha>.png` | Performance normalizado (baseline) |
| `rl_agent_rewards_vs_episodes_<fecha>.png` | Curva de aprendizaje |
| `rl_agent_sequence_performance_<fecha>.png` | Severidad por secuencia (agente) |
| `rl_agent_normalised_performance_<fecha>.png` | Performance normalizado (agente) |
| `rl_agent_cumulative_performance_<fecha>.png` | Performance acumulativo |
| `rl_agent_confidences_<fecha>.png` | Confianza raw del agente |
| `rl_agent_remapped_confidences_<fecha>.png` | Confianza renormalizada [0, 1] |

---

## 7. FunciÃ³n de EvaluaciÃ³n: `run_experiment()`

### 7.1 Firma

```python
def run_experiment(env, actionfunction, RandomSequences=True,
                   trials_per_sequence=None, sevs=None,
                   AssignAllocations=False, allocs=None,
                   NumberOfIterations=64):
```

### 7.2 Flujo

```
Para cada secuencia (0 a NumberOfIterations-1):
â”‚
â”œâ”€ Configurar secuencia (random o fija desde CSV)
â”œâ”€ Reset environment â†’ stateâ‚€
â”‚
â”œâ”€ While not done:
â”‚   â”œâ”€ action = actionfunction(env, state, seqid)
â”‚   â”œâ”€ state', reward, done, truncated, info = env.step(action)
â”‚   â””â”€ state = state'
â”‚
â”œâ”€ Al terminar secuencia:
â”‚   â”œâ”€ seqs.append(sum(severities))
â”‚   â”œâ”€ perfs.append(normalised_performance)
â”‚   â””â”€ seq_ev.append(severity_evolution)
â”‚
â””â”€ Retornar (seqs, perfs, seq_ev)
```

### 7.3 MÃ©tricas Colectadas

- **`seqs`**: Suma de severidades finales por secuencia (menor = mejor).
- **`perfs`**: Performance normalizado $\in [0, 1]$ (mayor = mejor).
- **`seq_ev`**: Matriz de evoluciÃ³n temporal de severidades por ciudad.

---

## 8. ConexiÃ³n Q-Table â†’ Experimento

### 8.1 Workflow de Deployment

```
Entrenamiento                         Experimento
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€                         â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
train_rl.py                           __main__.py
    â”‚                                     â”‚
    â”œâ”€ QLearning() â†’ Q                   â”œâ”€ Cargar inputs/q.npy
    â”œâ”€ numpy.save(                       â”œâ”€ Cargar inputs/rewards.npy
    â”‚   'inputs/<fecha>_RL_TRAIN/        â”‚
    â”‚    q_<fecha>.npy', Q)              â”œâ”€ pygameMediator
    â”‚                                    â”‚   .provide_rl_agent_response()
    â””â”€ (copiar manualmente)  â”€â”€â”€â”€â”€â†’      â”‚      â””â”€ numpy.load('inputs/q.npy')
       q_<fecha>.npy â†’ q.npy            â”‚      â””â”€ Q[res, trial, sev, :] â†’ argmax
                                         â”‚
                                         â””â”€ result_formatter
                                              .generate_results_report()
```

> **Paso manual requerido**: DespuÃ©s de entrenar, copiar los archivos:
>
> ```bash
> cp inputs/<fecha>_RL_TRAIN/q_<fecha>.npy inputs/q.npy
> cp inputs/<fecha>_RL_TRAIN/rewards_<fecha>.npy inputs/rewards.npy
> ```

### 8.2 IndexaciÃ³n de la Q-Table en EjecuciÃ³n

En `src/pygameMediator.py` â†’ `provide_rl_agent_response()`:

```python
Q = numpy.load(os.path.join(INPUTS_PATH, 'q.npy'))

# state = [resources_left, trial_no, severity]
options = Q[state[0], state[1], state[2], :]    # Vector de 11 Q-values

response, confidence, rt_hold, rt_release = rl_agent_meta_cognitive(
    options, state[0], RESPONSE_TIMEOUT
)
```

La Q-table se indexa con las 3 primeras dimensiones del estado para obtener
un vector de 11 Q-values (uno por acciÃ³n posible). La acciÃ³n es `argmax`.

---

## 9. Reproducibilidad del Entrenamiento

La funciÃ³n `QLearning()` en `ext/pandemic.py` acepta un parÃ¡metro
`seed`. Cuando es distinto de `None`, ejecuta:

```python
if seed is not None:
    numpy.random.seed(seed)
    random.seed(seed)
```

antes de inicializar la Q-table. Esto fija el estado de los dos
generadores que el algoritmo consume:

- `numpy.random.uniform` para la **inicializaciÃ³n** de la Q-table.
- `numpy.random.random` / `numpy.random.randint` para la **exploraciÃ³n
  Îµ-greedy**.
- `numpy.random.choice` y `random.randrange` (usados internamente por
  `Pandemic.random_sequence()`) para la **generaciÃ³n de secuencias**.

El pipeline de entrenamiento `train_rl.py` lee `SEED` desde
`config/CONFIG.py` (`SEED = 42` por defecto) y lo pasa a `QLearning`:

```python
from ..config.CONFIG import SEED
rewards, Q, confsrl = QLearning(env, learning_rate, discount_factor,
                                epsilon_initial, epsilon_min,
                                num_episodes, seed=SEED)
```

Como consecuencia, **dos ejecuciones consecutivas de
`python3 -m tabular.pes_base.ext.train_rl` con los mismos hiperparÃ¡metros
producen exactamente la misma Q-table** y, por tanto, exactamente el
mismo agente.

Para obtener trayectorias de entrenamiento independientes (p. ej. para
estimar varianza entre semillas), basta con modificar `SEED` en
`config/CONFIG.py` o invocar `QLearning(..., seed=None)` manualmente,
lo que restaura el comportamiento no determinÃ­stico clÃ¡sico.

---

## 10. Convergencia y Aprendizaje

### 10.1 Indicadores de Convergencia

1. **Reward promedio estabilizado**: La curva `rewards_<fecha>.npy` debe
   aplanarse en los Ãºltimos episodios.
2. **Performance en evaluaciÃ³n**: `perfs` cercano a 1.0 en la mayorÃ­a de
   secuencias indica polÃ­tica bien aprendida.
3. **Confianza alta**: DistribuciÃ³n de `confsrl` concentrada cerca de 1.0
   indica Q-values bien diferenciados.

### 10.2 Factores que Afectan la Convergencia

| Factor | Efecto positivo | Efecto negativo |
|--------|----------------|----------------|
| MÃ¡s episodios (1M) | Mejor convergencia | Mayor tiempo computacional |
| Î± = 0.2 (moderado) | Balanceo estabilidad/velocidad | â€” |
| Î³ = 0.9 | Considera futuro | Puede propagar errores |
| Îµ decay lento (lineal 1M) | ExploraciÃ³n exhaustiva | Convergencia lenta |
| Estado discreto | Tabla exacta | Pierde granularidad en severidad |

### 10.3 GarantÃ­as TeÃ³ricas

Q-Learning tabular converge a la polÃ­tica Ã³ptima $Q^*$ bajo las condiciones:

1. **Visitas infinitas**: Todos los pares $(s, a)$ se visitan infinitamente.
   Con 1M episodios y Îµ-greedy decayendo lentamente, la cobertura es amplia.
2. **Condiciones de Robbins-Monro**: $\sum \alpha_t = \infty$ y
   $\sum \alpha_t^2 < \infty$. Con Î± constante = 0.2, esta condiciÃ³n no se
   cumple estrictamente, pero en la prÃ¡ctica la convergencia es suficiente.
3. **MDP estacionario**: El entorno no cambia entre episodios (cumplido: las
   distribuciones de probabilidad de severidad y longitud son fijas).

---

## 11. Resumen de Correspondencias

| Concepto TeÃ³rico | ImplementaciÃ³n pes |
|------------------|--------------------|
| Estado $s$ | `[resources_left, trial_no, severity]` |
| AcciÃ³n $a$ | `action âˆˆ {0, ..., 10}` |
| Recompensa $r$ | `-numpy.sum(env.severities)` |
| Q-table $Q(s,a)$ | `numpy.ndarray` shape `(31, 11, 10, 11)` |
| ActualizaciÃ³n TD | `Q[s,a] += Î±(r + Î³ max Q[s'] - Q[s,a])` |
| PolÃ­tica Îµ-greedy | `random < 1-Îµ â†’ argmax Q ; else random` |
| Episodio | Una secuencia completa (3â€“10 trials) |
| Ambiente | `Pandemic(gymnasium.Env)` |
| Paso | `env.step(action)` â†’ sev update |
| Terminal | `iteration == seq_length` |
| InicializaciÃ³n Q | `uniform(-1, 1)` |
| Decaimiento Îµ | Lineal: `Îµ -= (Îµâ‚€-Îµ_min)/N` |
| Tracking | Reward promedio cada 10,000 episodios |
| EvaluaciÃ³n | `run_experiment(env, eval_qf, ...)` 64 secuencias |
| Confianza | EntropÃ­a de Shannon normalizada sobre Q-values |

---

## 12. EjecuciÃ³n PrÃ¡ctica

### 12.1 Entrenamiento

```bash
# Con episodios por defecto (1,000,000):
source linux_mpes_env/bin/activate
python3 -m tabular.pes_base.ext.train_rl

# Con episodios personalizados:
python3 -m tabular.pes_base.ext.train_rl 500000
```

### 12.2 Deployment

```bash
# Copiar modelo entrenado
cp pes_base/inputs/<fecha>_RL_TRAIN/q_<fecha>.npy pes_base/inputs/q.npy
cp pes_base/inputs/<fecha>_RL_TRAIN/rewards_<fecha>.npy pes_base/inputs/rewards.npy
```

### 12.3 EjecuciÃ³n del Experimento

```bash
python3 -m tabular.pes_base
```

### 12.4 VerificaciÃ³n

- Revisar `outputs/PES_log_<fecha>_RL_AGENT.txt` para el log completo.
- Revisar `outputs/<fecha>_RL_AGENT/PES_results_<id>.json` para estadÃ­sticas.
- Revisar `outputs/<fecha>_RL_AGENT/PES_results_<id>.png` para visualizaciones.
