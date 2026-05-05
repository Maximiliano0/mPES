# pes_ens — Guía de uso e implementación

> Última actualización: 2026-05-04
> Paquete: `ml/pes_ens`
> Algoritmo: Ensemble por *soft voting* de DQN + RDQN + Transformer (+ A2C opcional)

---

## 1. ¿Qué es `pes_ens` y por qué un ensemble?

`pes_ens` es el **agente compuesto** del proyecto mPES. En lugar de entrenar un
nuevo modelo desde cero, **combina las predicciones de varios agentes ya
entrenados** (los miembros del ensemble) para tomar una decisión final más
robusta y precisa en la tarea Pandemic Scenario.

La idea es sencilla: si tres expertos diferentes coinciden en una respuesta,
esa respuesta tiene mayor probabilidad de ser correcta que la de cualquier
experto individual. Formalmente, los ensembles **reducen la varianza** del
predictor agregado y, cuando los miembros cometen errores no correlacionados,
también **reducen el sesgo efectivo** de la decisión final (ver
[pes_ens_theory.md](pes_ens_theory.md)).

En mPES, los miembros son agentes con arquitecturas muy distintas:

- **DQN** (denso, sin memoria) — captura patrones locales del estado actual.
- **RDQN** (LSTM recurrente) — modela dependencias temporales cortas.
- **Transformer** (atención causal) — modela dependencias temporales largas.
- **A2C** (actor-crítico) — opcionalmente, política estocástica.

Cada uno tiene **fortalezas y debilidades complementarias**, por lo que la
combinación supera a cualquier miembro individual.

---

## 2. Cómo usarlo

A diferencia del resto de paquetes del workspace, `pes_ens` **no requiere
ninguna fase de entrenamiento ni de optimización bayesiana**. Solo carga los
modelos `.keras` de los paquetes hermanos y los combina en tiempo de
inferencia.

### Comando único

**Linux / macOS:**
```bash
source linux_mpes_env/bin/activate
python -m ml.pes_ens
```

**Windows (PowerShell):**
```powershell
win_mpes_env\Scripts\Activate.ps1
python -m ml.pes_ens
```

Eso es todo. El script ejecuta los bloques/secuencias/ensayos definidos en
`CONFIG.py`, registrando logs en `ml/pes_ens/outputs/` y resultados gráficos
con el formateador estándar del proyecto.

### Variables de entorno recomendadas

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:TF_ENABLE_ONEDNN_OPTS = "0"
$env:VIRTUAL_ENV = "$PWD\win_mpes_env"
```

---

## 3. Prerrequisitos: modelos miembro entrenados

`pes_ens` **no entrena nada**: necesita los `.keras` ya generados por los
paquetes hermanos. Antes de ejecutarlo, asegúrate de tener:

| Miembro | Archivo requerido | Cómo generarlo |
|---------|-------------------|----------------|
| DQN | `ml/pes_dqn/inputs/dqn_model.keras` | `python -m ml.pes_dqn` (tras `optimize_dqn.py` + `train_dqn.py`) |
| RDQN | `ml/pes_rdqn/inputs/rdqn_model.keras` | `python -m ml.pes_rdqn` |
| Transformer | `ml/pes_trf/inputs/trf_model.keras` | `python -m ml.pes_trf` |
| A2C *(opcional, deshabilitado por defecto)* | `ml/pes_a2c/inputs/ac_actor.keras` | `python -m ml.pes_a2c` |

Si falta alguno de los modelos habilitados, el agente lanzará un error claro
durante la inicialización indicando la ruta esperada.

---

## 4. Estructura del código

```
ml/pes_ens/
├── __init__.py              # Re-exports de CONFIG y setup numpy/TF
├── __main__.py              # Entry point: ejecuta bloques/secuencias/ensayos
├── config/
│   └── CONFIG.py            # ENS_MEMBER_MODELS, pesos, temperatura, prior
├── doc/                     # Documentación (este archivo y compañeros)
├── ext/
│   ├── ensemble_model.py    # EnsembleAgent, normalize_state, HistoryDeque
│   └── pandemic.py          # Entorno Gymnasium compartido
├── inputs/                  # (Vacío: los modelos están en paquetes hermanos)
├── outputs/                 # Logs y resultados
└── src/                     # Helpers compartidos del proyecto
```

### `ext/ensemble_model.py`

Tres componentes centrales:

#### `normalize_state(state) -> numpy.ndarray`
Escala el vector de estado al rango usado durante el entrenamiento de los
miembros (típicamente `[0, 1]`). Garantiza que cada miembro reciba el mismo
formato de entrada que cuando fue entrenado.

#### `HistoryDeque`
Búfer FIFO de tamaño fijo (`history_len`) por episodio y por miembro
recurrente. Almacena los últimos *k* estados normalizados que necesitan los
modelos LSTM y Transformer para condicionar su salida.

- Se reinicia al inicio de cada episodio.
- Hace *padding* con ceros mientras no haya suficientes pasos.

#### `EnsembleAgent`
Clase principal. Responsabilidades:

1. **Carga perezosa** de los modelos `.keras` de los miembros habilitados.
   La llamada usa `tf.keras.models.load_model(path, safe_mode=False)`; el
   flag es **obligatorio** para el modelo de `pes_trf`, que contiene una
   capa `Lambda` (`t[:, -1, :]`) que Keras 3 rechaza con `safe_mode=True`
   por mitigación CWE-502.
2. **Mantiene un `HistoryDeque` por miembro recurrente y por episodio**
   (clave `(session_no, sequence_no)` en `_history_caches`).
3. Implementa
   `predict(state_norm, resources_left, session_no, sequence_no)` que
   devuelve la tupla `(ensemble_probs, per_member_probs)` aplicando el
   pipeline completo descrito en §6.

---

## 5. Configuración (`config/CONFIG.py`)

Las constantes clave del ensemble son:

```python
ENS_MEMBER_MODELS = [
    {'name': 'dqn',  'role': 'q_dense',     'path': '../pes_dqn/inputs/dqn_model.keras',
     'history_len': 1, 'weight': 0.18, 'enabled': True},
    {'name': 'a2c',  'role': 'actor',       'path': '../pes_a2c/inputs/ac_actor.keras',
     'history_len': 1, 'weight': 1.00, 'enabled': False},
    {'name': 'rdqn', 'role': 'q_recurrent', 'path': '../pes_rdqn/inputs/rdqn_model.keras',
     'history_len': 6, 'weight': 0.90, 'enabled': True},
    {'name': 'trf',  'role': 'q_recurrent', 'path': '../pes_trf/inputs/trf_model.keras',
     'history_len': 6, 'weight': 5.00, 'enabled': True},
]

ENS_SOFTMAX_TEMPERATURE   = 15.0   # Suaviza/agudiza las distribuciones de Q
ENS_SEVERITY_PRIOR_WEIGHT = 0.17   # Mezcla con prior gaussiano
ENS_SEVERITY_PRIOR_SIGMA  = 3.0    # Desviación del prior
```

> **Formato**: `ENS_MEMBER_MODELS` es una **lista de diccionarios** (no
> un dict por nombre); el orden refleja el orden de iteración. Cada
> entrada admite las claves `name`, `role`, `path`, `history_len`,
> `weight`, `enabled`.

### Roles soportados

| Rol | Significado | Procesado |
|-----|-------------|-----------|
| `q_dense` | Modelo Q sin memoria (entrada: estado actual) | softmax(Q / T) |
| `q_recurrent` | Modelo Q con historia (entrada: ventana `history_len`) | softmax(Q / T) |
| `actor` | Política estocástica (salida ya en simplex) | usar tal cual |

### Pesos

Los `weight` son **relativos**: el agente los normaliza internamente. La
filosofía es sencilla — *miembros más precisos reciben mayor peso*. La
configuración actual refleja los resultados empíricos: el Transformer es
claramente el mejor miembro y por eso domina (`weight=5.0`), seguido por el
RDQN (`0.9`) y el DQN (`0.18`). El A2C se ha dejado **deshabilitado** por su
desempeño débil en este entorno.

---

## 6. Cómo funciona el *soft voting* paso a paso

Para una llamada
`agent.predict(state_norm, resources_left, session_no, sequence_no)`:

### Paso 1 — Inferencia por miembro
Para cada miembro habilitado se obtiene una distribución cruda sobre
las 11 acciones:
```python
for m in self.members:
    if   m['role'] == 'q_dense':
        q     = m['model'](state_norm[None, :]).numpy().flatten()
        probs = softmax(q / ENS_SOFTMAX_TEMPERATURE)        # (Q-net)
    elif m['role'] == 'q_recurrent':
        history.append_step(state_norm)
        window = history.current_window()                   # (T, 3)
        q      = m['model'](window[None, ...]).numpy().flatten()
        probs  = softmax(q / ENS_SOFTMAX_TEMPERATURE)
    elif m['role'] == 'actor':
        probs  = m['model'](state_norm[None, :]).numpy().flatten()
        probs  = clip_and_renormalise(probs)                # ya está en simplex
```

### Paso 2 — Máscara de factibilidad **por miembro**
Antes de mezclar, cada distribución individual se enmascara y
renormaliza con `max_feasible = max(0, resources_left)`:
```python
masked[max_feasible + 1:] = 0.0
masked /= masked.sum()           # cada miembro vota solo sobre acciones válidas
```

### Paso 3 — Voto dinámico ponderado por confianza
La confianza de un miembro se mide vía la entropía **normalizada**
(en $[0, 1]$) de su distribución factible:
```python
H_norm     = -sum(p * log2(p)) / log2(ACTION_DIM)
confidence = 1.0 - H_norm
dyn_weight = weight_norm * (0.1 + confidence)
ensemble  += dyn_weight * masked
```
El término constante `0.1` evita que un miembro completamente incierto
quede totalmente silenciado.

### Paso 4 — Penalización de la acción nula
Si quedan recursos disponibles, la acción "no responder" se castiga:
```python
if max_feasible > 0:
    ensemble[0] *= 0.3
```
(El registro experimental marca `r==0` como *no response*
— `confidence = -1` — desperdiciando el trial.)

### Paso 5 — Renormalización del ensemble
```python
ensemble /= ensemble.sum()
```

### Paso 6 — Mezcla con prior gaussiano de severidad
```python
severity_raw = state_norm[2] * MAX_SEVERITY
prior        = exp(-((arange(ACTION_DIM) - severity_raw) ** 2) / (2 * SIGMA ** 2))
prior[max_feasible + 1:] = 0.0
prior       /= prior.sum()
ensemble     = (1 - W_PRIOR) * ensemble + W_PRIOR * prior
ensemble    /= ensemble.sum()
```

### Paso 7 — *Severity-floor safety net*
Si la severidad observada es alta y el ensemble propone una asignación
demasiado baja, se fuerza un mínimo:
```python
if severity_raw >= 6.0:
    floor = severity_raw // 2
    if max_feasible >= floor and argmax(ensemble) < floor:
        ensemble = one_hot(floor)
```
Esta única regla eliminó los outliers catastróficos de los bloques
iniciales (mínimos en torno a 0.754).

### Paso 8 — Selección final
El llamador (entry point del paquete) toma
```python
action = int(numpy.argmax(ensemble_probs))
```

---

## 7. El *severity prior* y su efecto

El prior es una **gaussiana centrada en la severidad actual**:

$$
\text{prior}(a) \;=\; \exp\!\left(-\frac{(a - \text{severity})^{2}}{2\sigma^{2}}\right)
$$

Y se mezcla linealmente con la distribución del ensemble:

$$
\text{final}(a) \;=\; (1 - w)\,\text{ensemble}(a) \;+\; w\,\text{prior}(a)
$$

con `w = ENS_SEVERITY_PRIOR_WEIGHT = 0.17` y `σ = 3.0`.

**Efecto práctico**: empuja suavemente al agente a asignar recursos cercanos
a la severidad observada — un sesgo inductivo razonable en este dominio. Con
`w = 0.17` el ensemble sigue dominando la decisión, pero gana estabilidad en
estados ambiguos.

Si pones `w = 0`, el agente se comporta como un ensemble puro. Si pones
`w = 1`, se reduce a una política heurística *"asigna ≈ severidad"*.

---

## 8. Efecto de la temperatura softmax

La salida cruda de los miembros Q es un vector de Q-values que se convierte
en probabilidades con:

$$
p_i \;=\; \frac{\exp(Q_i / T)}{\sum_j \exp(Q_j / T)}
$$

- **T pequeña** (p. ej. 1.0) → distribución muy puntiaguda; el voto del
  miembro se concentra casi en una sola acción (≈ hard voting).
- **T grande** (p. ej. 15.0, valor actual) → distribución suave; el miembro
  expresa *grados de preferencia* y la combinación ponderada tiene más
  matices.

El valor `T = 15.0` se eligió empíricamente para que las diferencias
relativas entre Q-values se traduzcan en probabilidades informativas pero no
explosivas. Con T pequeña, un único miembro con confianza alta podría
sobrevotar al resto.

---

## 9. Archivos de entrada

`pes_ens` no genera ni consume archivos propios en `inputs/`. Lee
exclusivamente los `.keras` de paquetes hermanos:

```
../pes_dqn/inputs/dqn_model.keras
../pes_a2c/inputs/ac_actor.keras   (solo si enabled=True)
../pes_rdqn/inputs/rdqn_model.keras
../pes_trf/inputs/trf_model.keras
```

Las rutas se construyen con `os.path.join` desde `__init__.py` para
compatibilidad cross-platform.

---

## 10. Resultados de desempeño

Evaluación del 2026-05-02 (n = 64 ensayos):

| Modelo | `raw_mean_perf` | Desv. típica |
|--------|-----------------|--------------|
| pes_dqn | ~0.86 | ~0.06 |
| pes_rdqn | ~0.91 | ~0.05 |
| pes_trf | ~0.93 | ~0.04 |
| **pes_ens** | **0.937318** | **0.034937** |

El ensemble **supera a todos los miembros individuales**, incluyendo al
mejor (`pes_trf`), y además presenta **menor varianza** que cualquier
modelo aislado.

---

## 11. ¿Por qué el ensemble supera a todos sus miembros?

Tres razones principales:

1. **Reducción de varianza por promediado.** Si los errores de los miembros
   son parcialmente independientes, el promedio de sus salidas tiene
   varianza menor (ver descomposición sesgo–varianza en
   [pes_ens_theory.md](pes_ens_theory.md)).

2. **Diversidad de arquitecturas.** DQN (sin memoria), RDQN (LSTM) y
   Transformer (atención) capturan **regularidades distintas** del entorno.
   Los errores se distribuyen de forma desigual entre estados — donde uno
   falla, otro suele acertar.

3. **Inyección de conocimiento del dominio** vía severity prior, que
   estabiliza decisiones en estados raros donde *todos* los miembros tienen
   alta incertidumbre.

El resultado es un agente más preciso *y* más consistente, sin entrenar
una sola red adicional.

---

## Ver también

- [pes_ens_theory.md](pes_ens_theory.md) — fundamentos teóricos.
- [explained_ens.md](explained_ens.md) — arquitectura detallada.
- [how_to_train_and_test.md](how_to_train_and_test.md) — guía operativa.