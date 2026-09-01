# Teoría de Reinforcement Learning para Científicos de Datos

## 1. Introducción al Reinforcement Learning

**Reinforcement Learning (RL)** es un paradigma de aprendizaje automático donde
un **agente** aprende a tomar decisiones mediante interacción con un
**ambiente**. A diferencia del aprendizaje supervisado (donde hay etiquetas
correctas) o no supervisado (donde se buscan patrones), en RL el agente aprende
por **prueba y error**, recibiendo **recompensas** o **castigos** como señal de
retroalimentación.

### 1.1 El Problema de RL

El agente observa un **estado** del ambiente, ejecuta una **acción**, recibe
una **recompensa**, y el ambiente transiciona a un nuevo estado. El objetivo es
aprender una **política** que maximice la recompensa acumulada a largo plazo.

```
    ┌──────────┐    acción aₜ     ┌──────────┐
    │          │ ───────────────→ │          │
    │  Agente  │                  │ Ambiente │
    │          │ ←─────────────── │          │
    └──────────┘   sₜ₊₁, rₜ₊₁    └──────────┘
```

### 1.2 Contexto en PES

En el **Pandemic Experiment Scenario (PES)**:

- **Agente**: Algoritmo de Q-Learning que decide asignación de recursos.
- **Ambiente**: Simulación de pandemia (`Pandemic(gymnasium.Env)` en `ext/pandemic.py`).
- **Estado**: `(recursos_disponibles, número_de_trial, severidad)`.
- **Acción**: Cantidad de recursos a asignar (0–10).
- **Recompensa**: Negativo de la suma de severidades de todas las ciudades.
- **Objetivo**: Minimizar la severidad total de la pandemia.

---

## 2. Procesos de Decisión de Markov (MDP)

### 2.1 Definición Formal

Un **MDP** es una tupla $(S, A, P, R, \gamma)$:

- $S$: Conjunto finito de **estados**.
- $A$: Conjunto finito de **acciones**.
- $P(s' | s, a)$: **Función de transición** — Probabilidad de llegar al
  estado $s'$ dado que se está en $s$ y se ejecuta $a$.
- $R(s, a, s')$: **Función de recompensa** — Recompensa obtenida al
  transicionar de $s$ a $s'$ mediante $a$.
- $\gamma \in [0, 1]$: **Factor de descuento** — Importancia relativa de
  recompensas futuras vs. inmediatas.

### 2.2 Propiedad de Markov

Un estado $s_t$ satisface la propiedad de Markov si:

$$P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, s_1, a_1, ..., s_t, a_t)$$

Es decir, el estado futuro depende **únicamente** del estado actual y la acción
actual, no de toda la historia. En PES, el estado `(resources, trial, severity)`
captura toda la información necesaria para predecir el siguiente estado.

### 2.3 Retorno Descontado

El **retorno** $G_t$ es la suma descontada de recompensas futuras:

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

Con $\gamma = 0.9$ (valor en PES):

| Horizonte | Factor | Contribución |
|-----------|--------|-------------|
| Inmediata ($k=0$) | $0.9^0 = 1.0$ | 100 % |
| 1 paso | $0.9^1 = 0.9$ | 90 % |
| 2 pasos | $0.9^2 = 0.81$ | 81 % |
| 5 pasos | $0.9^5 = 0.59$ | 59 % |
| 10 pasos | $0.9^{10} = 0.35$ | 35 % |

> Con $\gamma = 0.9$, las recompensas a 10 pasos valen ~35 % de las inmediatas.
> Esto hace que el agente valore tanto el impacto inmediato de asignar recursos
> como las consecuencias a mediano plazo.

---

## 3. Funciones de Valor

### 3.1 Función de Valor de Estado $V(s)$

La función de valor de estado estima la recompensa esperada estando en $s$ y
siguiendo la política $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \mid s_t = s\right]$$

### 3.2 Función de Valor de Acción $Q(s, a)$

La función de valor de acción estima la recompensa esperada al ejecutar $a$ en
$s$ y luego seguir $\pi$:

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \mid s_t = s, a_t = a\right]$$

### 3.3 Relación entre $V$ y $Q$

$$V^\pi(s) = \sum_a \pi(a|s) \cdot Q^\pi(s, a)$$

Para una política determinística $\pi^*(s) = \arg\max_a Q^*(s, a)$:

$$V^*(s) = \max_a Q^*(s, a)$$

### 3.4 Implementación en PES

PES utiliza una **Q-table** (tabla de Q-values) de la forma:

$$Q : \underbrace{S}_{31 \times 11 \times 10} \times \underbrace{A}_{11} \rightarrow \mathbb{R}$$

```python
Q.shape = (31, 11, 10, 11)   # (resources, trials, severity, actions)
# Total: 37,510 entradas
```

Cada entrada $Q[r, t, s, a]$ almacena el valor estimado de asignar $a$ recursos
cuando quedan $r$ recursos disponibles, se está en el trial $t$, y la severidad
actual es $s$.

---

## 4. Ecuaciones de Bellman

### 4.1 Ecuación de Bellman de Optimalidad

$$Q^*(s, a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma \max_{a'} Q^*(s', a') \right]$$

Para MDP determinísticos (como PES), $P(s'|s,a) = 1$ para un único $s'$:

$$Q^*(s, a) = R(s, a) + \gamma \max_{a'} Q^*(s', a')$$

### 4.2 Diferencia Temporal (TD Error)

El **TD error** mide la discrepancia entre la estimación actual y la
observación:

$$\delta_t = r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)$$

Si $\delta_t > 0$: La recompensa observada fue **mejor** de lo esperado.
Si $\delta_t < 0$: La recompensa observada fue **peor** de lo esperado.
Si $\delta_t = 0$: La estimación es perfecta (convergencia).

---

## 5. Q-Learning

### 5.1 Algoritmo

Q-Learning es un algoritmo **off-policy** y **model-free** que aprende $Q^*$
directamente, sin necesidad de conocer $P(s'|s,a)$:

```
Inicializar Q(s, a) arbitrariamente para todo (s, a)
Para cada episodio:
    Inicializar s
    Repetir (para cada paso del episodio):
        Elegir a desde s usando ε-greedy de Q
        Ejecutar a, observar r, s'
        Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]
        s ← s'
    Hasta que s sea terminal
```

### 5.2 Regla de Actualización

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ \underbrace{r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a')}_{\text{target}} - \underbrace{Q(s_t, a_t)}_{\text{estimación actual}} \right]$$

Los componentes:

| Término | Significado |
|---------|-------------|
| $Q(s_t, a_t)$ | Estimación actual del valor |
| $r_{t+1}$ | Recompensa observada |
| $\gamma \max_{a'} Q(s_{t+1}, a')$ | Valor descontado del mejor futuro estimado |
| $\alpha$ | Tasa de aprendizaje (0.2 en PES) |

### 5.3 Propiedades

| Propiedad | Descripción |
|-----------|-------------|
| **Off-policy** | Aprende sobre la política greedy ($\max Q$) mientras sigue ε-greedy |
| **Model-free** | No necesita conocer las probabilidades de transición |
| **Tabular** | Almacena un valor para cada par $(s, a)$ en una tabla |
| **Online** | Actualiza después de cada transición |
| **Bootstrap** | Usa su propia estimación de $Q(s')$ para actualizar $Q(s)$ |

### 5.4 Reproducibilidad

> **Nota importante**: Si no se fija una semilla (`seed`) antes de
> entrenar, la inicialización aleatoria de la Q-table, las decisiones
> ε-greedy y la generación de secuencias de entrenamiento serán
> diferentes en cada ejecución. Esto implica que **el resultado del
> entrenamiento no es reproducible** a menos que se establezcan
> semillas explícitas con `numpy.random.seed()` y `random.seed()`.
> Sin semilla fija, la única forma de preservar un resultado es
> conservar la Q-table (`.npy`) generada.
>
> En la implementación actual de PES, la función `QLearning()` acepta
> un parámetro `seed` que, cuando se proporciona, fija las semillas
> de `numpy.random` y `random` antes de inicializar la Q-table.
> El pipeline de entrenamiento (`train_rl.py`) pasa `seed=SEED`
> (por defecto `SEED = 42` en `CONFIG.py`), haciendo que el
> entrenamiento sea **reproducible** entre ejecuciones con los
> mismos hiperparámetros.

### 5.5 Convergencia

Q-Learning converge a $Q^*$ bajo las condiciones:

1. Todos los pares $(s, a)$ se visitan infinitamente a menudo.
2. La tasa de aprendizaje $\alpha$ satisface las condiciones de Robbins-Monro:
   - $\sum_{t=0}^{\infty} \alpha_t = \infty$
   - $\sum_{t=0}^{\infty} \alpha_t^2 < \infty$
3. El MDP es ergódico (todos los estados son alcanzables).

> **Nota técnica**: En PES, $\alpha = 0.2$ es constante, lo cual viola
> estrictamente la condición 2. Sin embargo, con 1,000,000 de episodios y
> un espacio de estados relativamente pequeño (3,410 estados), la convergencia
> práctica es suficiente.

---

## 6. Exploración vs. Explotación

### 6.1 El Dilema

- **Exploración**: Probar acciones nuevas para descubrir mejores estrategias.
- **Explotación**: Usar el conocimiento actual para maximizar recompensa.

### 6.2 Política ε-Greedy

La solución más simple al dilema:

$$
a_t = \begin{cases}
\arg\max_a Q(s_t, a) & \text{con probabilidad } 1 - \varepsilon \\
\text{acción aleatoria} & \text{con probabilidad } \varepsilon
\end{cases}
$$

### 6.3 Decaimiento de ε

Para transicionar de exploración a explotación, $\varepsilon$ decrece durante
el entrenamiento:

**Decaimiento lineal** (usado en PES):

$$\varepsilon_{i+1} = \varepsilon_i - \frac{\varepsilon_0 - \varepsilon_{\min}}{N}$$

Con los valores de PES:

$$\varepsilon_{i+1} = \varepsilon_i - \frac{0.8 - 0.0}{1{,}000{,}000} = \varepsilon_i - 0.0000008$$

Otras estrategias de decaimiento (no usadas en PES pero comunes en RL):

- **Exponencial**: $\varepsilon_{i+1} = \varepsilon_i \cdot \lambda$, con $\lambda \in (0.999, 1)$
- **Inversión**: $\varepsilon_i = \frac{1}{1 + i/k}$
- **Step function**: $\varepsilon$ fijo por bloques de episodios

---

## 7. Función de Recompensa

### 7.1 Diseño de la Función de Recompensa

La función de recompensa define qué comportamiento el agente debe aprender.
Un buen diseño es crucial:

| Tipo | Descripción | Ventaja | Desventaja |
|------|-------------|---------|-----------|
| **Densa** | Recompensa en cada paso | Señal frecuente | Puede crear atajos |
| **Dispersa** | Solo en terminal | Objetivo claro | Difícil de aprender |
| **Shaped** | Modificada con heurística | Acelera aprendizaje | Puede sesgar |

### 7.2 Recompensa en PES

PES usa recompensa **densa** (en cada paso):

$$r_t = -\sum_{i=1}^{n_t} \text{severity}_i$$

Donde $n_t$ es el número de ciudades visibles en el paso $t$.

**Propiedades**:

- Siempre negativa (o cero si todas las severidades son 0).
- Más negativa = peor situación → incentiva reducir severidades.
- Densa: proporciona señal en cada trial, no solo al final.
- La suma incluye **todas** las ciudades, no solo la actual.

### 7.3 Caso Terminal

En PES, cuando la secuencia termina:

```python
if done:
    Q[s, a] = reward    # Sin término de descuento futuro
```

El Q-value del estado terminal se fija como la recompensa observada, ya que
no hay estado futuro.

---

## 8. Tabular vs. Aproximación de Funciones

### 8.1 Método Tabular (PES)

| Aspecto | Tabular |
|---------|---------|
| Almacenamiento | Tabla explícita $|S| \times |A|$ |
| Tamaño en PES | $31 \times 11 \times 10 \times 11 = 37{,}510$ |
| Actualización | Directa: `Q[s,a] += ...` |
| Convergencia | Garantizada (bajo condiciones) |
| Generalización | No generaliza entre estados similares |
| Escalabilidad | Crece exponencialmente con dimensiones |

### 8.2 Aproximación de Funciones (alternativa)

Para espacios de estados grandes o continuos, se puede aproximar $Q$ con:

- **Redes neuronales** (Deep Q-Network / DQN)
- **Funciones lineales** con features
- **Tiles coding** (discretización adaptativa)
- **Transformers** (como en `pes_trf`)

### 8.3 ¿Por qué tabular funciona en PES?

El espacio de estados de PES es pequeño:

$$|S| = 31 \times 11 \times 10 = 3{,}410$$
$$|S \times A| = 3{,}410 \times 11 = 37{,}510$$

Con 1,000,000 episodios y secuencias de 3–10 trials:

- Mínimo $\sim 3{,}000{,}000$ transiciones vistas.
- Promedio de visitas por par $(s,a)$: $\sim 73$.
- Suficiente para buena cobertura estadística.

---

## 9. Entropía y Confianza Meta-cognitiva

### 9.1 Entropía de Shannon

Para una distribución de probabilidad $p = (p_1, ..., p_n)$:

$$H(p) = -\sum_{i=1}^{n} p_i \log_2 p_i$$

Propiedades:

- $H \geq 0$ siempre.
- $H = 0$ si y solo si la distribución es determinística (un $p_i = 1$).
- $H = \log_2 n$ si la distribución es uniforme ($p_i = 1/n$ para todo $i$).

### 9.2 Entropía como Medida de Confianza

En PES, los Q-values se interpretan como una distribución sobre acciones:

1. Normalizar Q-values a probabilidades: $p_i = Q_i / \sum Q_j$
2. Calcular entropía: $H(p)$
3. Normalizar a confianza:

$$\text{confidence} = \frac{H(p) - H_{\max}}{H_{\min} - H_{\max}}$$

Donde:

- $H_{\min}$: Entropía de distribución determinística ($\approx 0$)
- $H_{\max}$: Entropía de distribución uniforme ($= \log_2 11 \approx 3.46$)

| Confianza | Interpretación |
|-----------|---------------|
| $\approx 1.0$ | Q-values muy concentrados (una acción claramente dominante) |
| $\approx 0.5$ | Q-values moderadamente diferenciados |
| $\approx 0.0$ | Q-values uniformes (todas las acciones igualmente valoradas) |

### 9.3 Confianza → Tiempo de Reacción

PES simula respuestas humanas mapeando confianza a tiempos de reacción:

$$\mu = \lfloor(-2 \cdot \text{confidence} + 1) \times 10\rfloor$$

$$t_{\text{hold}} \sim \mathcal{N}(\mu, 3), \quad t_{\text{release}} \sim t_{\text{hold}} + \mathcal{N}(\mu, 1)$$

| Confianza | $\mu$ | Interpretación |
|-----------|-------|---------------|
| 1.0 | -10 | Respuesta rápida (decisión segura) |
| 0.5 | 0 | Tiempo medio |
| 0.0 | 10 | Respuesta lenta (decisión incierta) |

---

## 10. Evaluación de Políticas

### 10.1 Performance Normalizado

PES normaliza el resultado de cada secuencia:

$$\text{Perf} = \frac{\text{Sev}_{\text{worst}} - \text{Sev}_{\text{actual}}}{\text{Sev}_{\text{worst}} - \text{Sev}_{\text{best}}}$$

Donde:

- $\text{Sev}_{\text{actual}}$: Severidad final con la política del agente.
- $\text{Sev}_{\text{worst}}$: Severidad final con asignación 0 en todos los trials.
- $\text{Sev}_{\text{best}}$: Severidad final con asignación 10 en todos los trials.

### 10.2 Interpretación

| Performance | Calidad |
|-------------|---------|
| 0.0 | Equivalente a no asignar recursos |
| 0.5 | Mitad del rango posible |
| 1.0 | Equivalente al óptimo factible (DP) bajo el presupuesto real |
| > 1.0 | No posible (cota teórica) |

### 10.3 Baseline de Comparación

El pipeline de entrenamiento ejecuta un **agente aleatorio** como baseline:

- Usa las asignaciones pre-definidas de los datos.
- Proporciona un piso de referencia para el agente entrenado.
- Se espera que el agente Q-Learning supere significativamente al baseline.

---

## 11. Resumen de Fórmulas Clave

| Concepto | Fórmula |
|----------|---------|
| Retorno | $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$ |
| Valor de acción | $Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t=s, a_t=a]$ |
| Bellman óptimo | $Q^*(s,a) = R + \gamma \max_{a'} Q^*(s', a')$ |
| Actualización Q | $Q(s,a) \leftarrow Q(s,a) + \alpha[\delta_t]$ |
| TD error | $\delta_t = r + \gamma \max_{a'} Q(s', a') - Q(s,a)$ |
| ε-greedy | $\pi(a\|s) = (1-\varepsilon)\mathbb{1}[a=a^*] + \varepsilon/\|A\|$ |
| Entropía | $H(p) = -\sum p_i \log_2 p_i$ |
| Confianza | $c = (H - H_{\max}) / (H_{\min} - H_{\max})$ |
| Severidad | $s' = \max(0, \beta \cdot s - \alpha \cdot a)$ |
| Performance | $(W - A) / (W - B)$ |

---

## 12. Glosario

| Término | Definición |
|---------|-----------|
| **Agente** | Entidad que toma decisiones (Q-Learning en PES) |
| **Ambiente** | Sistema con el que interactúa el agente (`Pandemic(Env)`) |
| **Estado** | Observación del agente (`[resources, trial, severity]`) |
| **Acción** | Decisión del agente (recursos a asignar, 0–10) |
| **Recompensa** | Señal numérica de retroalimentación ($-\sum$ severidades) |
| **Episodio** | Una secuencia completa (3–10 trials) |
| **Política** | Regla de decisión $\pi(a\|s)$ |
| **Q-value** | Valor esperado de acción en estado |
| **Q-table** | Tabla de todos los Q-values (`31×11×10×11`) |
| **ε-greedy** | Política que explora con probabilidad ε |
| **Descuento** | Factor $\gamma$ que reduce valor de recompensas futuras |
| **TD error** | Diferencia entre target y estimación actual |
| **Off-policy** | Aprende sobre política diferente a la que sigue |
| **Model-free** | No necesita modelo de transiciones |
| **Convergencia** | Q-values se estabilizan en valores óptimos |
| **MDP** | Proceso de Decisión de Markov (marco formal) |
| **Entropía** | Medida de incertidumbre de una distribución |
| **Meta-cognición** | Auto-evaluación de la confianza del agente |

---

## 13. Referencias

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An
   Introduction* (2nd ed.). MIT Press.
2. Watkins, C. J. C. H., & Dayan, P. (1992). Q-Learning. *Machine Learning*,
   8(3), 279–292.
3. Mnih, V., et al. (2015). Human-level control through deep reinforcement
   learning. *Nature*, 518(7540), 529–533.
4. Shannon, C. E. (1948). A Mathematical Theory of Communication. *Bell System
   Technical Journal*, 27(3), 379–423.
