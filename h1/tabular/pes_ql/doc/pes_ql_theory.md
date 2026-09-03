# pes_ql — Fundamentos Teóricos

> Bases matemáticas del Q-Learning tabular y de la optimización Bayesiana
> con TPE, relacionadas explícitamente con la implementación de
> `tabular/pes_ql`.

---

## Índice

1. [Procesos de Decisión de Markov (MDP)](#1-procesos-de-decisión-de-markov-mdp)
2. [Programación dinámica y ecuaciones de Bellman](#2-programación-dinámica-y-ecuaciones-de-bellman)
3. [El algoritmo Q-Learning](#3-el-algoritmo-q-learning)
4. [Garantías de convergencia](#4-garantías-de-convergencia)
5. [Política ε-greedy y compromiso exploración-explotación](#5-política-ε-greedy-y-compromiso-exploración-explotación)
6. [Optimización Bayesiana de hiperparámetros](#6-optimización-bayesiana-de-hiperparámetros)
7. [Tree-structured Parzen Estimator (TPE)](#7-tree-structured-parzen-estimator-tpe)
8. [Confianza meta-cognitiva basada en entropía](#8-confianza-meta-cognitiva-basada-en-entropía)
9. [Métrica de performance normalizada](#9-métrica-de-performance-normalizada)
10. [Referencias](#10-referencias)

---

## 1. Procesos de Decisión de Markov (MDP)

El entorno `Pandemic` (definido en
[ext/pandemic.py](../ext/pandemic.py)) constituye un **proceso de decisión de
Markov finito** (Bellman, 1957; Sutton & Barto, 2018, cap. 3), formalizado
como la tupla:

$$
\mathcal{M} = \langle \mathcal{S},\; \mathcal{A},\; \mathcal{P},\; \mathcal{R},\; \gamma \rangle
$$

donde:

- $\mathcal{S}$ es el conjunto de estados. En `pes_ql`,
  $\mathcal{S} \subseteq \mathbb{Z}^3$ con cada estado
  $s = (r,\, t,\, \sigma)$, donde $r \in \{0, \ldots, 30\}$ son los recursos
  disponibles, $t \in \{0, \ldots, 10\}$ el número de trial dentro de la
  secuencia, y $\sigma \in \{0, \ldots, 9\}$ la severidad observada.
- $\mathcal{A} = \{0, 1, \ldots, 10\}$ es el conjunto de acciones (recursos
  asignados al trial actual).
- $\mathcal{P}(s' \mid s, a)$ es la distribución de transición. En `pes_ql`
  es **determinista** dada la nueva severidad inicial muestreada al inicio
  del trial; condicional a esa muestra, $s'$ es función determinista de $s$
  y $a$.
- $\mathcal{R}(s, a) = -\sum_i \sigma_i$ es la recompensa inmediata
  (negativa de la suma de severidades de todas las ciudades).
- $\gamma \in [0, 1)$ es el factor de descuento. En `pes_ql` se optimiza
  $\gamma \in [0.85,\, 0.999]$.

### Propiedad de Markov

La suposición central es que la distribución del próximo estado y recompensa
depende **solo** del estado y acción presentes:

$$
\Pr\bigl(s_{t+1} = s',\, r_{t+1} = r \,\bigm|\, s_t, a_t, s_{t-1}, a_{t-1}, \ldots\bigr)
= \Pr\bigl(s_{t+1} = s',\, r_{t+1} = r \,\bigm|\, s_t, a_t\bigr).
$$

En `Pandemic` esta propiedad se cumple porque la dinámica de severidad
$\sigma' = \max(0,\; 1.4\sigma - 0.4 a)$ y el contador de recursos
$r' = r - a$ dependen exclusivamente del estado y acción actuales.

### Política y retorno

Una **política** $\pi: \mathcal{S} \to \mathcal{A}$ define la regla de
decisión. El **retorno** desde el instante $t$ es:

$$
G_t = \sum_{k=0}^{T-t-1} \gamma^k\, r_{t+k+1}
$$

donde $T$ es el número de trials de la secuencia (3 a 10).

---

## 2. Programación dinámica y ecuaciones de Bellman

### Funciones de valor

La **función de valor de estado** bajo $\pi$ es:

$$
V^\pi(s) = \mathbb{E}_\pi\!\left[ G_t \,\bigm|\, s_t = s \right],
$$

y la **función de valor de acción** (Q-function):

$$
Q^\pi(s, a) = \mathbb{E}_\pi\!\left[ G_t \,\bigm|\, s_t = s,\, a_t = a \right].
$$

### Ecuación de Bellman para $Q^\pi$

$$
Q^\pi(s, a) = \sum_{s', r} \mathcal{P}(s', r \mid s, a)\,
\bigl[ r + \gamma\, \mathbb{E}_{a' \sim \pi}\bigl[ Q^\pi(s', a') \bigr] \bigr]
$$

### Ecuación óptima de Bellman

La política óptima $\pi^*$ satisface:

$$
Q^*(s, a) = \mathbb{E}\!\left[ r_{t+1} + \gamma \max_{a'} Q^*(s_{t+1}, a') \,\bigm|\, s_t = s,\, a_t = a \right]
$$

(Bellman, 1957). Esta es la ecuación que Q-Learning aproxima por iteración
estocástica.

---

## 3. El algoritmo Q-Learning

Q-Learning, introducido por Watkins (1989) y formalizado en
Watkins y Dayan (1992), es un método **off-policy, model-free, de diferencia
temporal**. Estima $Q^*$ directamente sin requerir conocimiento de
$\mathcal{P}$ ni $\mathcal{R}$.

### Regla de actualización

Para una transición observada $(s, a, r, s')$:

$$
\boxed{\;
Q(s, a) \;\leftarrow\; Q(s, a) \;+\; \alpha\, \underbrace{\bigl[\, r + \gamma \max_{a'} Q(s', a') - Q(s, a)\, \bigr]}_{\text{TD error}\;\delta_t}
\;}
$$

donde $\alpha \in (0, 1]$ es la **tasa de aprendizaje**.

### Implementación en `pes_ql`

En [ext/pandemic.py](../ext/pandemic.py), líneas 731–741:

```python
delta = learning * (reward
                    + discount * numpy.max(Q[s2_idx[0], s2_idx[1], s2_idx[2]])
                    - Q[s_idx[0], s_idx[1], s_idx[2], action])
Q[s_idx[0], s_idx[1], s_idx[2], action] += delta
```

con la salvedad de que en estados terminales se asigna directamente
$Q(s, a) \leftarrow r$ (sin término de bootstrapping), preservando la
ecuación de Bellman para horizontes finitos.

### Pseudo-código

```
Inicializar Q(s, a) ← Uniform(-1, 1) ∀ s, a
para episodio = 1, …, M:
    s ← reset(env)
    repetir hasta terminal:
        a ← ε-greedy(Q, s)
        (s', r, terminal) ← env.step(a)
        si terminal:
            Q(s, a) ← r
        sino:
            Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') − Q(s, a)]
        s ← s'
    ε ← ε − (ε₀ − ε_min) / M
```

---

## 4. Garantías de convergencia

Watkins y Dayan (1992) demostraron que $Q$ converge a $Q^*$ con probabilidad
1 cuando se cumplen las condiciones de Robbins–Monro sobre las tasas
$\alpha_t(s, a)$:

$$
\sum_{t=0}^{\infty} \alpha_t(s, a) = \infty,
\qquad
\sum_{t=0}^{\infty} \alpha_t^2(s, a) < \infty,
$$

junto con la **visita infinita** de cada par $(s, a)$.

### Limitaciones prácticas en `pes_ql`

- La implementación usa $\alpha$ **constante** (no decreciente), por lo que
  estrictamente las hipótesis de Robbins–Monro no se satisfacen. En la
  práctica, con $\alpha \in [0.05, 0.4]$ y $5 \times 10^5$–$1.2 \times 10^6$
  episodios, $Q$ converge a una vecindad de $Q^*$ suficientemente pequeña
  para producir políticas casi-óptimas (Sutton & Barto, 2018, §6.5).
- La visita infinita se aproxima vía exploración ε-greedy con
  $\epsilon_{\min} > 0$ (en `pes_ql`, $\epsilon_{\min} \in [0.01, 0.15]$).

---

## 5. Política ε-greedy y compromiso exploración-explotación

La política ε-greedy es:

$$
\pi(a \mid s) =
\begin{cases}
\arg\max_{a'} Q(s, a') & \text{con prob. } 1 - \epsilon \\
\text{uniforme sobre } \mathcal{A} & \text{con prob. } \epsilon
\end{cases}
$$

### Decaimiento lineal en `pes_ql`

```python
reduction = (epsilon - min_eps) / episodes
# tras cada episodio:
if epsilon > min_eps:
    epsilon -= reduction
```

es decir,

$$
\epsilon_k = \max\!\left(\epsilon_{\min},\; \epsilon_0 - k \cdot \frac{\epsilon_0 - \epsilon_{\min}}{M}\right).
$$

Este esquema garantiza:

- **Fase exploratoria** inicial donde el agente acumula información sobre
  pares $(s, a)$ raramente visitados.
- **Fase explotativa** final donde el agente refina la estimación de los
  Q-valores cercanos a $Q^*$ siguiendo la política voraz casi siempre.

Sutton y Barto (2018, §2.2) discuten cómo $\epsilon_{\min} > 0$ asegura
exploración persistente, condición necesaria para la convergencia en
entornos no estacionarios o con estados raramente visitados.

---

## 6. Optimización Bayesiana de hiperparámetros

El espacio de hiperparámetros de Q-Learning es:

$$
\boldsymbol{\theta} = (\alpha,\, \gamma,\, \epsilon_0,\, \epsilon_{\min},\, M) \in \Theta \subset \mathbb{R}^5.
$$

La función objetivo $f: \Theta \to [0, 1]$ es la **performance media
normalizada** sobre las 64 secuencias fijas de evaluación. Su evaluación
es:

- **Cara**: cada trial cuesta minutos (millones de pasos de Q-Learning).
- **Estocástica**: depende de la semilla.
- **Sin gradiente**: $f$ no es diferenciable respecto a $\boldsymbol{\theta}$.

Estas tres propiedades hacen a la **optimización Bayesiana** (BO) la
herramienta idónea (Bergstra et al., 2011).

### Modelo conceptual

BO mantiene un modelo probabilístico sustituto $p(f \mid \mathcal{D}_n)$
condicionado a las $n$ evaluaciones previas
$\mathcal{D}_n = \{(\boldsymbol{\theta}_i, y_i)\}_{i=1}^n$. En cada
iteración, selecciona el siguiente $\boldsymbol{\theta}_{n+1}$ que maximiza
una **función de adquisición** $\mathcal{A}$, típicamente la **Expected
Improvement** (EI):

$$
\text{EI}(\boldsymbol{\theta}) = \mathbb{E}\!\left[ \max(0,\; f(\boldsymbol{\theta}) - y^*) \right],
$$

donde $y^* = \max_{i \le n} y_i$.

---

## 7. Tree-structured Parzen Estimator (TPE)

`pes_ql` utiliza el sampler `optuna.samplers.TPESampler` (Akiba et al.,
2019), que implementa el algoritmo TPE de Bergstra et al. (2011).

### Idea central

En lugar de modelar $p(f \mid \boldsymbol{\theta})$ con un proceso
Gaussiano (como GP-EI), TPE invierte el condicionamiento y modela
$p(\boldsymbol{\theta} \mid y)$ usando dos densidades de Parzen
(kernel-density):

$$
p(\boldsymbol{\theta} \mid y) =
\begin{cases}
\ell(\boldsymbol{\theta}) & \text{si } y < y^* \\
g(\boldsymbol{\theta}) & \text{si } y \ge y^*
\end{cases}
$$

donde $y^*$ es un percentil $\gamma$ de los valores observados (por defecto
$\gamma = 0.15$ en Optuna). Bergstra et al. (2011) demuestran que la EI es
**proporcional al cociente** $\ell(\boldsymbol{\theta}) /
g(\boldsymbol{\theta})$, lo que reduce la maximización de la adquisición a
muestrear de $\ell$ y elegir el punto con mayor cociente.

### Configuración en `pes_ql`

```python
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=4),
    storage=f'sqlite:///{db_path}',
    load_if_exists=True,
)
```

- `seed=SEED` (= 42) garantiza secuencias de muestreo reproducibles entre
  ejecuciones independientes del estudio.
- `MedianPruner(n_startup_trials=5, n_warmup_steps=4)` aborta cualquier
  trial cuya recompensa media tras 4 reportes (40 000 episodios) caiga por
  debajo de la mediana histórica de los trials previos en el mismo paso.
- `storage=sqlite://...` persiste el estudio permitiendo `--resume`.

### Espacio de búsqueda

| Hiperparámetro | Distribución | Rango | Justificación |
|----------------|--------------|-------|---------------|
| $\alpha$ | log-uniforme | $[0.05,\, 0.40]$ | Tasas pequeñas estabilizan TD |
| $\gamma$ | uniforme | $[0.85,\, 0.999]$ | Horizontes ≤ 10 trials |
| $\epsilon_0$ | uniforme | $[0.50,\, 1.00]$ | Exploración inicial alta |
| $\epsilon_{\min}$ | uniforme | $[0.01,\, 0.15]$ | Mínima exploración persistente |
| $M$ | entero, paso 50 000 | $[5\!\times\!10^5,\, 1.2\!\times\!10^6]$ | Convergencia empírica |

La elección **log** para $\alpha$ refleja que el efecto de variar $\alpha$
de 0.05 a 0.10 es comparable al de variar de 0.20 a 0.40 (Bergstra et al.,
2011, §4.2).

---

## 8. Confianza meta-cognitiva basada en entropía

`rl_agent_meta_cognitive` (en [ext/pandemic.py](../ext/pandemic.py))
calcula una **medida de confianza** del agente derivada de la entropía de
los Q-valores en el estado actual:

$$
H(s) = -\sum_{a \in \mathcal{A}} p(a \mid s)\, \log p(a \mid s),
$$

con $p(a \mid s)$ obtenido vía softmax sobre las opciones factibles
(acciones $a > r$ se enmascaran con $-10^9$).

### Normalización

$$
\text{confidence}(s) = \frac{H(s) - H_{\max}}{H_{\min} - H_{\max}},
$$

donde:

- $H_{\max} = \log |\mathcal{A}|$ corresponde a una distribución uniforme
  (mínima confianza, máxima incertidumbre).
- $H_{\min} = 0$ corresponde a una delta sobre la acción óptima (máxima
  confianza).

Así $\text{confidence} \in [0, 1]$, con 1 = certeza total.

### Uso

La confianza **no afecta** la actualización de $Q$ ni la selección de
acciones (siempre se usa $\arg\max$). Se utiliza únicamente para:

1. Generar tiempos de respuesta sintéticos (estilo humano) vía mapeo lineal.
2. Producir plots comparativos humano-vs-agente en `__main__.py`.

Por eficiencia, durante optimización y entrenamiento se desactiva con
`track_confidence=False`.

---

## 9. Métrica de performance normalizada

Definida en
`calculate_normalised_final_severity_performance_metric()` en
[src/exp_utils.py](../src/exp_utils.py):

$$
P = \frac{\text{WorstCase} - \text{Achieved}}{\text{WorstCase} - \text{BestCase}}
$$

donde:

- $\text{Achieved}$: suma de severidades finales obtenida por la política.
- $\text{BestCase}$: óptimo teórico (asignación voraz informada).
- $\text{WorstCase}$: peor caso (asignación nula).

$P = 1$ corresponde a la política óptima; $P = 0$ a la peor. La función
maneja secuencias degeneradas ($\text{WorstCase} = \text{BestCase}$)
devolviendo $P = 0$ en lugar de divisiones por cero, y `optimize_rl.py`
adicionalmente aplica `numpy.clip(mean_perf, 0, 1)` para no envenenar el
modelo TPE con valores fuera de rango.

---

## 10. Referencias

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A
next-generation hyperparameter optimization framework. *Proceedings of the
25th ACM SIGKDD International Conference on Knowledge Discovery & Data
Mining*, 2623–2631. https://doi.org/10.1145/3292500.3330701

Bellman, R. (1957). *Dynamic programming*. Princeton University Press.

Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for
hyper-parameter optimization. *Advances in Neural Information Processing
Systems*, 24, 2546–2554.

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An
introduction* (2nd ed.). MIT Press.

Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*,
8(3–4), 279–292. https://doi.org/10.1007/BF00992698
