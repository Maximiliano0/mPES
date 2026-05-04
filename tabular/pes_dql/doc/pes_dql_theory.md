# pes_dql — Fundamento teórico

> Paquete: **`tabular.pes_dql`** — Double Q-Learning con calentamiento
> exponencial de ε y *Potential-Based Reward Shaping* (PBRS).
>
> Este documento desarrolla los fundamentos matemáticos del agente y
> los enlaza con las líneas de código de
> [`tabular/pes_dql/ext/pandemic.py`](../ext/pandemic.py) y
> [`tabular/pes_dql/ext/optimize_rl.py`](../ext/optimize_rl.py).

---

## Índice

1. [Marco MDP y ecuaciones de Bellman](#1-marco-mdp-y-ecuaciones-de-bellman)
2. [Q-Learning clásico](#2-q-learning-clásico)
3. [Double Q-Learning](#3-double-q-learning)
4. [ε-greedy con warm-up](#4-ε-greedy-con-warm-up)
5. [Potential-Based Reward Shaping (PBRS)](#5-potential-based-reward-shaping-pbrs)
6. [Garantías de convergencia](#6-garantías-de-convergencia)
7. [Optimización bayesiana con TPE](#7-optimización-bayesiana-con-tpe)
8. [Referencias APA 7](#8-referencias-apa-7)

---

## 1. Marco MDP y ecuaciones de Bellman

### 1.1 MDP del entorno Pandemic

El entorno se modela como un proceso de decisión de Markov (MDP) finito
$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma\rangle$
(Sutton & Barto, 2018, cap. 3):

- **Estados** $\mathcal{S}$: tuplas
  $s = (\text{recursos\_disponibles},\, \text{trial},\, \text{severidad})$
  con cardinalidad $|\mathcal{S}| = 31 \times 11 \times 10 = 3{,}410$.
- **Acciones** $\mathcal{A} = \{0, 1, \ldots, 10\}$ — recursos asignados al
  trial actual.
- **Transición** $P(s'\mid s, a)$: determinista,
  $s'_i = \max(0,\, \beta\, s_i - \alpha\, a_i)$ con
  $\alpha = 0.4$ (`PANDEMIC_PARAMETER`) y $\beta = 1 + \alpha = 1.4$
  (`SEVERITY_MULTIPLIER`). Implementada en
  `exp_utils.get_updated_severity`.
- **Recompensa** $R(s, a) = -\sum_i s'_i$ — negativo de la severidad
  total tras la transición.
- **Factor de descuento** $\gamma \in (0, 1)$ (típicamente 0.98).

La política óptima $\pi^*$ satisface la **ecuación de Bellman para Q**:

$$Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q^*(s', a').$$

Como la transición es determinista,

$$Q^*(s, a) = R(s, a) + \gamma \max_{a'} Q^*\!\bigl(T(s, a),\, a'\bigr).$$

### 1.2 De Bellman a la actualización TD

La actualización **temporal-difference (TD)** (Sutton & Barto, 2018,
cap. 6) sustituye el valor verdadero por una estimación bootstrap:

$$Q_{t+1}(s, a) \leftarrow Q_t(s, a) + \alpha_t\,\bigl[\underbrace{r + \gamma\,\max_{a'} Q_t(s', a')}_{\text{target}} - Q_t(s, a)\bigr].$$

---

## 2. Q-Learning clásico

Q-Learning (Watkins & Dayan, 1992) es la versión *off-policy* de la
actualización TD: aprende la política greedy $\pi^*$ mientras explora con
una política diferente (típicamente ε-greedy).

### 2.1 Sesgo de maximización

El operador `max` en el target de TD acumula una sobrestimación
sistemática debido a la **desigualdad de Jensen** aplicada al
operador máximo:

$$\mathbb{E}\bigl[\max_{a'} \hat Q(s', a')\bigr] \;\geq\; \max_{a'} \mathbb{E}\bigl[\hat Q(s', a')\bigr].$$

En palabras: el `max` selecciona los Q-valores con error positivo,
inflando sistemáticamente el target. El resultado son políticas que
sobreasignan recursos a acciones con valor aparentemente alto pero
ruidoso.

Este problema se intensifica cuando:
- Hay pocas visitas por par $(s, a)$ (Q-tables ruidosas).
- El número de acciones es grande (más oportunidades de error positivo).
- La recompensa tiene varianza alta.

---

## 3. Double Q-Learning

Van Hasselt (2010) propuso desacoplar **selección** y **evaluación** del
target manteniendo dos estimadores independientes $Q_A$ y $Q_B$:

$$
\begin{aligned}
Q_A(s, a) &\leftarrow Q_A(s, a) + \alpha\,\bigl[r + \gamma\, Q_B\!\bigl(s',\, \arg\max_{a'} Q_A(s', a')\bigr) - Q_A(s, a)\bigr], \\
Q_B(s, a) &\leftarrow Q_B(s, a) + \alpha\,\bigl[r + \gamma\, Q_A\!\bigl(s',\, \arg\max_{a'} Q_B(s', a')\bigr) - Q_B(s, a)\bigr].
\end{aligned}
$$

En cada step se actualiza **una sola** de las dos tablas con probabilidad
0.5. La idea clave: si $Q_A$ sobreestima la acción $a^*$, el target usa
$Q_B(s', a^*)$ —que es **independiente** de $Q_A$— y por tanto no comparte
el sesgo.

### 3.1 Análisis del sesgo

Sea $X_a$ una variable aleatoria con $\mathbb{E}[X_a] = \mu_a$ y
$\mu^* = \max_a \mu_a$. Entonces:

- **Single estimator** (Q-Learning): el estimador
  $\max_a X_a$ tiene sesgo positivo $\mathbb{E}[\max_a X_a] - \mu^* > 0$.
- **Double estimator** (Double Q-Learning): si dividimos las muestras en
  dos conjuntos independientes y usamos un conjunto para seleccionar y
  el otro para evaluar, el estimador resultante tiene sesgo negativo o
  nulo (Van Hasselt, 2010, Lemma 1).

En la práctica, el sesgo negativo de Double Q-Learning es mucho menos
problemático que el positivo de Q-Learning estándar, porque las políticas
greedy aún convergen a $\pi^*$ y los errores se promedian al usar
$(Q_A + Q_B)/2$ en inferencia.

### 3.2 Extensión a Deep RL

Hasselt, Guez y Silver (2016) demostraron que el mismo principio aplica a
DQN: el algoritmo **Double DQN** logra mejoras estadísticamente
significativas en 49 juegos de Atari simplemente desacoplando la red
*online* (para selección) de la red *target* (para evaluación). El
paquete `pes_dqn` del workspace usa esta extensión.

### 3.3 Correspondencia con el código

En [`pandemic.py`](../ext/pandemic.py), dentro de la función
`QLearning(env, learning, discount, epsilon, min_eps, episodes,
warmup_ratio, target_ratio, double_q=True, penalty_coeff, ...)`:

```python
if double_q:
    Q_A = numpy.random.uniform(low=-1, high=1, size=q_shape)
    Q_B = numpy.random.uniform(low=-1, high=1, size=q_shape)
```

Y en el bucle de aprendizaje:

```python
if double_q:
    if numpy.random.random() < 0.5:
        best_action = int(numpy.argmax(q_a_s2))   # selección con Q_A
        target = reward + discount * Q_B[s2 + (best_action,)]   # evaluación con Q_B
        Q_A[s + (action,)] += learning * (target - Q_A[s + (action,)])
    else:
        best_action = int(numpy.argmax(q_b_s2))   # selección con Q_B
        target = reward + discount * Q_A[s2 + (best_action,)]   # evaluación con Q_A
        Q_B[s + (action,)] += learning * (target - Q_B[s + (action,)])
```

La inferencia (política greedy) usa la suma:

```python
q_sel = Q_A[s] + Q_B[s]
action = int(numpy.argmax(q_masked))
```

que es equivalente a $\arg\max_a (Q_A + Q_B)/2$ porque `argmax` es
invariante a escalado positivo.

---

## 4. ε-greedy con warm-up

### 4.1 Compromiso exploración–explotación

La política de comportamiento durante el entrenamiento es
**ε-greedy** (Sutton & Barto, 2018, cap. 2):

$$\pi_\varepsilon(a \mid s) = \begin{cases} 1 - \varepsilon + \dfrac{\varepsilon}{|\mathcal{A}|}, & a = \arg\max_{a'} Q(s, a'), \\ \dfrac{\varepsilon}{|\mathcal{A}|}, & \text{en otro caso}. \end{cases}$$

Para garantizar convergencia (visitar infinitamente cada par $(s, a)$),
$\varepsilon$ debe decaer lentamente (Robbins-Monro). El esquema típico es
exponencial:

$$\varepsilon_t = \max\bigl(\varepsilon_{\min},\, \varepsilon_0\,\lambda^t\bigr).$$

### 4.2 Problema sin warm-up

Si $\varepsilon$ comienza a decaer desde el episodio 1, el agente empieza
a explotar Q-valores que aún son esencialmente ruido (las tablas se
inicializan con $\mathcal{U}(-1, 1)$). Esto causa:

- **Convergencia a políticas subóptimas locales** porque las primeras
  decisiones greedy aleatorias se refuerzan.
- **Ineficiencia de muestreo** porque grandes regiones del espacio de
  estados nunca se visitan.

### 4.3 Esquema con warm-up

`pes_dql` introduce un **calentamiento** controlado por dos parámetros:

- $w \in [0, 1]$ (`warmup_ratio`) — fracción de episodios con
  $\varepsilon = \varepsilon_0$ (exploración pura).
- $\tau \in [0, 1]$ (`target_ratio`) — fracción de episodios al cabo de
  la cual $\varepsilon$ debe alcanzar $\varepsilon_{\min}$.

Con $N$ episodios totales, el cronograma es:

$$\varepsilon_t = \begin{cases}
\varepsilon_0, & t < wN \\
\max\bigl(\varepsilon_{\min},\; \varepsilon_0\,\lambda^{t - wN}\bigr), & t \geq wN
\end{cases}$$

donde $\lambda$ se calcula automáticamente para que
$\varepsilon$ alcance $\varepsilon_{\min}$ exactamente en $t = \tau N$:

$$\lambda = \left(\frac{\varepsilon_{\min}}{\varepsilon_0}\right)^{1 / ((\tau - w) N)}.$$

Tras $\tau N$ episodios, el agente entra en una fase de **explotación
pura** (con ε = ε_min) que ocupa el $(1 - \tau)$ final del entrenamiento.

### 4.4 Justificación

El warm-up garantiza que las primeras políticas explotadas se basen en
estimaciones Q que ya integran al menos una pasada por las regiones
relevantes del MDP. Al combinar warm-up con Double Q, el periodo inicial
de exploración se utiliza para reducir la varianza de ambas tablas
*antes* de que cualquiera empiece a guiar la explotación.

---

## 5. Potential-Based Reward Shaping (PBRS)

### 5.1 Definición

Dado un MDP $\mathcal{M}$, Ng, Harada y Russell (1999) demostraron que
modificar la recompensa con un término de la forma

$$F(s, s') = \gamma\,\Phi(s') - \Phi(s)$$

—donde $\Phi: \mathcal{S} \to \mathbb{R}$ es una **función potencial**
arbitraria— produce un MDP $\mathcal{M}'$ con recompensa
$R'(s, a, s') = R(s, a, s') + F(s, s')$ que comparte la **misma política
óptima** que $\mathcal{M}$.

### 5.2 Teorema de invariancia de política

**Teorema** (Ng et al., 1999, Theorem 1): Sea
$F: \mathcal{S} \times \mathcal{S} \to \mathbb{R}$ una función de
*shaping*. Entonces toda política óptima en $\mathcal{M}'$ también es
óptima en $\mathcal{M}$ (y viceversa) **si y solo si** existe una función
real $\Phi$ tal que para todo $(s, a, s')$:

$$F(s, a, s') = \gamma\,\Phi(s') - \Phi(s).$$

La intuición es que el término $F$ es un **gradiente** sobre la función
potencial $\Phi$, y al sumarse a lo largo de cualquier trayectoria
forma una serie telescópica:

$$\sum_{t=0}^{T-1} \gamma^t F(s_t, s_{t+1}) = \sum_{t=0}^{T-1} \gamma^t\,(\gamma\,\Phi(s_{t+1}) - \Phi(s_t)) = \gamma^T \Phi(s_T) - \Phi(s_0).$$

Es decir, $F$ solo añade una constante al retorno total que depende
únicamente de los estados inicial y final, no de la trayectoria. Por
tanto la política que maximiza $V'$ es la misma que maximiza $V$.

### 5.3 Potencial elegido en `pes_dql`

`pes_dql` define:

$$\Phi(s) = -\sum_i s_i$$

(negativo de la severidad acumulada). Estados con menor severidad tienen
mayor potencial, y la recompensa modelada premia transiciones que
**reducen** la severidad y penaliza las que la dejan crecer. El
coeficiente $\beta$ (`penalty_coeff`) escala el shaping:

$$F(s, s') = \beta\,\bigl(\gamma\,\Phi(s') - \Phi(s)\bigr).$$

El teorema 1 de Ng et al. (1999) sigue aplicando: $\beta\Phi$ es también
un potencial válido, y la política óptima es invariante a $\beta$ para
todo $\beta \geq 0$.

### 5.4 Por qué acelera la convergencia

Sin shaping, el agente solo recibe la señal $r = -\sum_i s_i$ al final
de cada step, agregada sobre todas las ciudades (no localizada por
acción). PBRS añade **gradiente local**: cada transición individual recibe
inmediatamente una recompensa proporcional al cambio en el potencial,
guiando el descenso por gradiente del Q-Learning hacia regiones de
$\mathcal{S}$ con menor severidad.

### 5.5 Correspondencia con el código

En `QLearning(...)`:

```python
phi_s  = -float(numpy.sum(env.severities))      # antes del step
state2, reward, done, _, _ = env.step(action)
phi_s2 = 0.0 if done else -float(numpy.sum(env.severities))
shaped_reward = reward + penalty_coeff * (discount * phi_s2 - phi_s)
```

El shaping se desactiva poniendo `penalty_coeff = 0.0`.

---

## 6. Garantías de convergencia

### 6.1 Q-Learning estándar

Watkins y Dayan (1992) demostraron que Q-Learning converge con
probabilidad 1 a $Q^*$ bajo:

1. Todos los pares $(s, a)$ se visitan infinitas veces.
2. La tasa de aprendizaje satisface las condiciones de Robbins-Monro:
   $\sum_t \alpha_t = \infty$ y $\sum_t \alpha_t^2 < \infty$.
3. Las recompensas tienen varianza acotada.

Con tasa fija ($\alpha$ constante, como en `pes_dql`) la convergencia es a
una **vecindad** de $Q^*$ cuyo radio es $O(\alpha)$.

### 6.2 Double Q-Learning

Van Hasselt (2010, Theorem 1) demostró que Double Q-Learning converge a
$Q^*$ bajo las mismas condiciones de Robbins-Monro, siempre que **ambas**
tablas reciban infinitas actualizaciones (lo que ocurre con probabilidad 1
si se selecciona cuál actualizar mediante un Bernoulli(0.5)).

La ventaja práctica es **menor varianza durante el transitorio**:
aunque ambos métodos convergen al mismo límite, Double Q-Learning lo
hace con menos oscilaciones.

### 6.3 PBRS

Por el teorema de invariancia (Ng et al., 1999), PBRS no afecta las
condiciones de convergencia: si Q-Learning converge en $\mathcal{M}$
también converge en $\mathcal{M}'$, y al mismo $\pi^*$. El shaping solo
modifica la **velocidad** de convergencia (Wiewiora, 2003, completó el
análisis para mostrar que PBRS también preserva la equivalencia entre
políticas Q-greedy intermedias).

---

## 7. Optimización bayesiana con TPE

### 7.1 Planteamiento del problema

El espacio de hiperparámetros tiene 8 dimensiones (ver § 4.2 de
[`pes_dql_explained.md`](./pes_dql_explained.md)). Cada evaluación
requiere entrenar un Double Q-Learning de hasta 500 000 episodios y
evaluar sobre 64 secuencias — una *grid search* exhaustiva sería
prohibitiva.

La optimización bayesiana modela la función objetivo
$f: \mathcal{X} \to \mathbb{R}$ (el `mean_perf` de evaluación) como un
proceso estocástico y elige iterativamente el siguiente punto a
evaluar maximizando una **función de adquisición** que equilibra
exploración (regiones inciertas) y explotación (regiones prometedoras).

### 7.2 TPE (Tree-structured Parzen Estimator)

`pes_dql` usa Optuna (Akiba et al., 2019) con el sampler `TPESampler`,
que es la implementación de referencia del algoritmo TPE de Bergstra et
al. (2011). TPE modela $p(x \mid y)$ en lugar de $p(y \mid x)$:

1. Las observaciones se separan en "buenas" (top-γ%) y "malas" (resto).
2. Se ajustan dos densidades $\ell(x)$ (buenas) y $g(x)$ (malas) usando
   estimación de Parzen.
3. La función de adquisición es proporcional a $\ell(x) / g(x)$, que se
   maximiza para proponer el siguiente punto.

Configuración usada:

```python
TPESampler(seed=SEED, n_startup_trials=10, multivariate=True, group=True)
```

- `n_startup_trials=10` — los primeros 10 trials usan muestreo aleatorio
  para inicializar las densidades.
- `multivariate=True` y `group=True` — el TPE modela conjuntamente las
  variables relacionadas en lugar de marginalmente, lo que es esencial
  para espacios con correlaciones (como `warmup_ratio` y `target_ratio`).

### 7.3 Reproducibilidad por trial

Cada trial se entrena con `seed = SEED + trial.number + 1`. Esto:

- Garantiza que dos trials distintos exploren trayectorias estocásticas
  diferentes (de lo contrario, el ruido de muestreo podría dominar la
  diferencia entre hiperparámetros similares).
- Asegura que cada trial sea **individualmente reproducible**: re-ejecutar
  el trial $k$ con la misma semilla y los mismos hiperparámetros
  reproduce exactamente la misma Q-table y el mismo `mean_perf`.

Esta convención (`SEED + trial.number + 1`) es compartida con `pes_dqn`,
`pes_a2c` y `pes_ql` para mantener consistencia entre paquetes.

### 7.4 Persistencia y reanudación

El estudio se almacena en SQLite:

```
sqlite:///inputs/<date>_BAYESIAN_OPT/optuna_study_<date>.db
```

Esto permite reanudar con `--resume YYYY-MM-DD` o inspeccionar la
historia con `optuna-dashboard` (ver
[`utils/win/optuna_dashboard.ps1`](../../utils/win/optuna_dashboard.ps1) y
[`utils/linux/optuna_dashboard.sh`](../../utils/linux/optuna_dashboard.sh)).

---

## 8. Referencias APA 7

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna:
A next-generation hyperparameter optimization framework. *Proceedings of
the 25th ACM SIGKDD International Conference on Knowledge Discovery &
Data Mining*, 2623–2631. https://doi.org/10.1145/3292500.3330701

Hasselt, H. van, Guez, A., & Silver, D. (2016). Deep reinforcement
learning with double Q-learning. *Proceedings of the 30th AAAI
Conference on Artificial Intelligence*, *30*(1), 2094–2100.
https://doi.org/10.1609/aaai.v30i1.10295

Ng, A. Y., Harada, D., & Russell, S. J. (1999). Policy invariance under
reward transformations: Theory and application to reward shaping.
*Proceedings of the 16th International Conference on Machine Learning*,
278–287.

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An
introduction* (2nd ed.). MIT Press.

Van Hasselt, H. (2010). Double Q-learning. *Advances in Neural
Information Processing Systems*, *23*, 2613–2621.

Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine
Learning*, *8*(3–4), 279–292. https://doi.org/10.1007/BF00992698
