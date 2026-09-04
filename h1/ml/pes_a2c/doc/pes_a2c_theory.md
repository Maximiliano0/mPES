# `pes_a2c` — Fundamentos teóricos

> **Tema**: Métodos de gradiente de política y arquitectura Actor-Critic
> **Última actualización**: 2026-04-30

---

## 1. Métodos de gradiente de política vs métodos basados en valor

El aprendizaje por refuerzo distingue dos grandes familias de algoritmos
(Sutton & Barto, 2018):

### 1.1. Métodos basados en valor

Aprenden una función de valor —típicamente $Q(s,a)$— y derivan la política de
forma indirecta:

$$\pi(s) = \arg\max_a Q(s,a)$$

Ejemplos: Q-Learning tabular (`pes_base`, `pes_ql`), Double Q-Learning
(`pes_dql`), DQN (`pes_dqn`), Recurrent DQN (`pes_rdqn`), Transformer DQN
(`pes_trf`).

**Ventajas**: muestreo eficiente, convergencia bien estudiada en el caso
tabular.
**Desventajas**: política determinista, $\arg\max$ exhaustivo en cada paso,
poco apropiados para acciones continuas o políticas estocásticas.

### 1.2. Métodos de gradiente de política

Parametrizan **directamente la política** $\pi_\theta(a \mid s)$ y optimizan
el rendimiento esperado por gradiente ascendente (Williams, 1992):

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$$

$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$

Ejemplos: REINFORCE, Actor-Critic (A2C, A3C), PPO, TRPO, SAC.

**Ventajas**: políticas estocásticas naturales, manejo de acciones continuas,
exploración explícita por entropía.
**Desventajas**: alta varianza del gradiente, sensibilidad a hiperparámetros.

---

## 2. El teorema del gradiente de política

Sutton & Barto (2018, cap. 13) demuestran que, para una política diferenciable
$\pi_\theta$, el gradiente del objetivo se expresa como:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\Big[\nabla_\theta \log \pi_\theta(a\mid s)\, Q^{\pi}(s,a)\Big]$$

Este resultado es la base de todos los métodos policy-gradient. La esperanza
se aproxima por **muestreo Monte Carlo** sobre trayectorias generadas con la
propia política.

### 2.1. REINFORCE (Williams, 1992)

Estimador Monte Carlo más simple:

$$\nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^{N} \sum_{t=0}^{T_i} \nabla_\theta \log \pi_\theta(a_t^{(i)} \mid s_t^{(i)})\, G_t^{(i)}$$

donde $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ es el retorno descontado desde
el paso $t$.

**Problema**: la varianza de $G_t$ es enorme, lo que hace el aprendizaje muy
inestable.

### 2.2. REINFORCE con línea base

Si se resta una **línea base** $b(s)$ que dependa solo del estado:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a\mid s) (G_t - b(s))\right]$$

el estimador sigue siendo insesgado (porque $\mathbb{E}[\nabla \log \pi \cdot
b(s)] = 0$) pero con **varianza mucho menor**. La elección óptima de la línea
base es $b(s) = V^\pi(s)$.

---

## 3. Arquitectura Actor-Critic

La idea clave de Actor-Critic (Mnih et al., 2016) es **aprender simultáneamente**:

- **Actor**: la política $\pi_\theta(a\mid s)$.
- **Crítico**: la función de valor $V_\phi(s)$ que sirve como línea base.

### 3.1. Función de ventaja

Se define la **ventaja** de tomar la acción $a$ en el estado $s$ como:

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

Mide cuánto **mejor o peor** es la acción $a$ respecto al promedio bajo la
política actual. Es exactamente $G_t - V(s_t)$ con la línea base óptima.

En la práctica se aproxima la ventaja con el **TD-error** (un solo paso):

$$A(s_t, a_t) \approx r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

### 3.2. Pérdidas de A2C

**Pérdida del actor** (a minimizar):

$$\mathcal{L}_\mathrm{actor}(\theta) = -\mathbb{E}\big[\log \pi_\theta(a\mid s) \cdot A(s,a)\big] - \beta\, H(\pi_\theta(\cdot\mid s))$$

**Pérdida del crítico** (a minimizar):

$$\mathcal{L}_\mathrm{critic}(\phi) = \mathbb{E}\Big[\big(r + \gamma V_\phi(s') - V_\phi(s)\big)^2\Big]$$

**Pérdida total** (cuando se entrena con un único optimizador):

$$\mathcal{L} = \mathcal{L}_\mathrm{actor}(\theta) + c_v \cdot \mathcal{L}_\mathrm{critic}(\phi)$$

donde $c_v$ sería el coeficiente del crítico ($\approx 0.5$ en
implementaciones típicas de A2C).

> En `pes_a2c` se usan **dos optimizadores Adam separados** con
> *learning-rates* independientes (`AC_ACTOR_LR`, `AC_CRITIC_LR`), por lo
> que **no existe** un coeficiente $c_v$ ni una pérdida combinada: cada
> red minimiza su propio escalar.

---

## 4. Estimación de la ventaja: detalle

La estimación TD(0) usada en `train_step_a2c()`:

$$\hat A_t = r_t + \gamma V_\phi(s_{t+1}) (1 - d_t) - V_\phi(s_t)$$

donde $d_t \in \{0,1\}$ indica terminación del episodio. El factor
$(1 - d_t)$ asegura que en estados terminales el target es solo $r_t$.

### 4.1. Por qué TD frente a Monte Carlo

| Estimador | Sesgo | Varianza |
|---|---|---|
| Monte Carlo ($G_t$) | 0 | Alta |
| TD(0) ($r + \gamma V$) | Pequeño | Baja |
| GAE ($\lambda$) | Configurable | Configurable |

`pes_a2c` usa TD(0) por simplicidad. Una extensión natural es **Generalized
Advantage Estimation** (GAE), que interpola entre ambos.

### 4.2. `stop_gradient` sobre la ventaja

Es **fundamental** detener el gradiente sobre $A(s,a)$ al calcular la pérdida
del actor:

```python
actor_loss = -mean(log_pi * tf.stop_gradient(advantage))
```

Sin `stop_gradient`, el optimizador del actor intentaría modificar también los
pesos del crítico, mezclando objetivos incompatibles.

---

## 5. Regularización por entropía

La **entropía** de una distribución categórica $\pi(\cdot\mid s)$ con $K$
acciones es:

$$H(\pi(\cdot\mid s)) = -\sum_{a=1}^{K} \pi(a\mid s) \log \pi(a\mid s)$$

Se incluye en la pérdida del actor con peso $\beta$ (`AC_ENTROPY_COEFF`)
**positivo** dentro de la maximización (negativo en la pérdida):

$$\mathcal{L}_\mathrm{actor} = -\mathbb{E}[\log \pi \cdot A] - \beta\, H(\pi)$$

### Efecto:

- Si $\beta$ es **alto**, la política se mantiene cerca de la uniforme →
  exploración elevada, riesgo de no converger.
- Si $\beta$ es **bajo**, la política colapsa rápido en una acción
  determinista → riesgo de quedarse en óptimo local.
- Valor empíricamente óptimo en `pes_a2c`: $\beta \approx 0.01$.

La entropía regularizada se conoce también como **soft policy gradient** y es
la base teórica de Soft Actor-Critic (SAC).

---

## 6. Política con restricciones de factibilidad

En el escenario *Pandemic*, una acción $a$ es factible solo si
$a \le \mathrm{resources\_left}$. Hay dos enfoques:

### 6.1. Penalización en la recompensa

Añadir un castigo grande si el agente elige una acción inválida. Sencillo
pero **lento de aprender** y deja al agente desperdiciar exploración en
acciones imposibles.

### 6.2. Enmascaramiento de la política (usado en `pes_a2c`)

Forzar $\pi(a\mid s) = 0$ para acciones infactibles **antes** del muestreo o
del $\arg\max$:

$$\tilde\pi(a\mid s) = \begin{cases}
\dfrac{\pi(a\mid s)}{\sum_{a' \in \mathcal{F}(s)} \pi(a'\mid s)} & \text{si } a \in \mathcal{F}(s)\\
0 & \text{en otro caso}
\end{cases}$$

donde $\mathcal{F}(s)$ es el conjunto de acciones factibles.

**Implicaciones**:

- En **inferencia**: garantiza que toda decisión es válida.
- En **entrenamiento**: la red aprende a asignar probabilidad cero o casi
  cero a acciones infactibles, ya que las acciones inválidas nunca se
  muestrean y por tanto no contribuyen al gradiente de log-verosimilitud.

---

## 7. Propiedades de convergencia

A2C **no tiene garantías de convergencia global** (problema no convexo). Sin
embargo, se conocen los siguientes resultados teóricos y empíricos:

1. **Bajo aproximación lineal y línea base óptima**, el gradiente de política
   converge a un óptimo local del rendimiento (Sutton et al., 1999).
2. **Reducción de varianza**: el uso de la línea base $V_\phi(s)$ reduce la
   varianza del estimador hasta en un orden de magnitud frente a REINFORCE
   puro (Schulman et al., 2016, GAE).
3. **Entropía regularizada**: garantiza una política estrictamente positiva
   en todos los estados, evitando colapso prematuro.

### Diagnósticos prácticos durante el entrenamiento

| Síntoma | Causa probable | Acción |
|---|---|---|
| Rendimiento estancado en azar | $\beta$ demasiado alto | Reducir entropía |
| Política colapsada en una acción | $\beta$ demasiado bajo | Aumentar entropía |
| Pérdida del crítico no baja | LR del crítico muy bajo o $c_v$ muy bajo | Subir LR o $c_v$ |
| Explosión del gradiente | LR muy alto | Bajar LR, `clipnorm=1.0` |

---

## 8. Diferencias con A3C y PPO

`pes_a2c` implementa la versión **síncrona** y **single-thread** de
Actor-Critic (a veces llamada A2C en oposición al A3C asíncrono original
de Mnih et al., 2016).

| Variante | Paralelismo | Ventaja |
|---|---|---|
| **A3C** | Asíncrono, varios workers | Diversidad de exploración |
| **A2C** | Síncrono, varios workers | Mejor uso de GPU |
| **`pes_a2c`** | Single-thread, batches por episodio | Sencillez, suficiente para 39 recursos |
| **PPO** | Síncrono + clipping de ratio | Mayor estabilidad |

PPO sería una mejora natural; se descartó en `pes_a2c` porque A2C ya alcanza
$\approx 0.887$ y la complejidad adicional no se justifica para el tamaño del
escenario.

---

## 9. Optimización Bayesiana de hiperparámetros

A2C es **muy sensible** a la elección de $\alpha$ (learning rate), $\beta$
(entropía), $\gamma$ (descuento) y $c_v$ (coeficiente del crítico). La
búsqueda manual es impracticable.

`optimize_a2c.py` usa **Optuna** (Akiba et al., 2019) con muestreador
**Tree-structured Parzen Estimator (TPE)**:

$$\text{TPE}: p(\theta \mid y) = \begin{cases} \ell(\theta) & y < y^* \\ g(\theta) & y \ge y^* \end{cases}$$

y propone el siguiente trial maximizando la **expected improvement**:

$$\mathrm{EI}(\theta) \propto \frac{\ell(\theta)}{g(\theta)}$$

### Espacio de búsqueda (claves Optuna reales)

```python
trial.suggest_float ('actor_lr',           1e-4, 1e-2, log=True)
trial.suggest_float ('critic_lr',          1e-4, 1e-2, log=True)
trial.suggest_float ('discount_factor',    0.85, 0.995)
trial.suggest_float ('entropy_coeff',      0.0,  0.1)        # lineal, no log
trial.suggest_categorical('actor_hidden_dim',  [32, 64, 128, 256])
trial.suggest_categorical('critic_hidden_dim', [32, 64, 128, 256])
trial.suggest_int('n_hidden_layers',       1, 3)
trial.suggest_int('num_episodes',          50_000, 250_000, step=25_000)
# Recompensa moldeada (PBRS) y otros:
trial.suggest_float('penalty_coeff',       0.0,  0.3)
trial.suggest_float('gae_lambda',          0.90, 0.99)
trial.suggest_float('max_grad_norm',       0.3,  1.5)
trial.suggest_float('lr_min_ratio',        0.05, 0.25)
trial.suggest_float('spend_cost_coeff',    0.0,  0.05)
trial.suggest_float('last_action_bias',   -2.0,  0.0)
```

### Pruning

Cada trial puede abortarse anticipadamente si su rendimiento intermedio queda
por debajo del cuartil inferior de los trials previos en la misma época
(MedianPruner). Esto reduce drásticamente el coste de la búsqueda.

---

## 10. Comparación teórica con DQN

| Aspecto | A2C | DQN |
|---|---|---|
| Output de la red | $\pi(\cdot\mid s)$ (softmax) | $Q(s,\cdot)$ (lineal) |
| Exploración | Estocástica + entropía | $\varepsilon$-greedy |
| Regla de decisión | $\arg\max_a \pi$ o muestreo | $\arg\max_a Q$ |
| Necesita replay buffer | No (on-policy) | Sí (off-policy) |
| Estabilidad numérica | Sensible a $\beta$ | Estable con target network |
| Complejidad computacional | 2 redes simultáneas | 1 red + target |

A2C y DQN son **complementarios**: el primero es policy-gradient on-policy y
el segundo value-based off-policy. Su diversidad es la justificación principal
de incluir ambos en el ensemble `pes_ens`.

---

## 11. Referencias (APA 7)

- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A
  next-generation hyperparameter optimization framework. *Proceedings of the
  25th ACM SIGKDD International Conference on Knowledge Discovery & Data
  Mining*, 2623–2631.
- Hasselt, H. van, Guez, A., & Silver, D. (2016). Deep reinforcement learning
  with double Q-learning. *Proceedings of the AAAI Conference on Artificial
  Intelligence, 30*(1), 2094–2100.
- Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T.,
  Silver, D., & Kavukcuoglu, K. (2016). Asynchronous methods for deep
  reinforcement learning. *Proceedings of the 33rd International Conference on
  Machine Learning*, 1928–1937.
- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare,
  M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G.,
  Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran,
  D., Wierstra, D., Legg, S., & Hassabis, D. (2015). Human-level control
  through deep reinforcement learning. *Nature, 518*(7540), 529–533.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An
  introduction* (2nd ed.). MIT Press.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for
  connectionist reinforcement learning. *Machine Learning, 8*(3–4), 229–256.
