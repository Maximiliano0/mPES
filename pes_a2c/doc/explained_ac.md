# Mapeo Teoría Actor-Critic (A2C) ↔ Implementación en pes_a2c

## 1. Introducción

Este documento conecta la **teoría de Advantage Actor-Critic (A2C)** con su
**implementación concreta** en el paquete `pes_a2c`.  Para cada
concepto teórico se indica la variable, función o línea de código
correspondiente.

`pes_a2c` reemplaza la red DQN de `pes_dqn` (y la tabla Q tabular
de `pes_base` / `pes_ql`) por **dos redes neuronales separadas**:

- un **Actor** $\pi_\theta(a \mid s)$ que produce directamente una
  distribución de probabilidad sobre las acciones, y
- un **Critic** $V_\phi(s)$ que estima el valor del estado actual.

Conserva el mismo entorno Gymnasium (`Pandemic`), el mismo espacio de estados y
acciones, y la misma integración con la UI de Pygame.

---

## 2. Componentes del MDP (heredados de `pes_base`)

El MDP subyacente es idéntico al del paquete base:

| Componente | Símbolo | Implementación |
|------------|---------|----------------|
| Estados | $S$ | `(resources_left, trial_no, severity)` — 3 dimensiones discretas |
| Acciones | $A$ | $\{0, 1, 2, \dots, 10\}$ — recursos a asignar |
| Transiciones | $P(s' \mid s, a)$ | Determinísticas: `env.step(action)` en `pandemic.py` |
| Recompensas | $R(s, a)$ | $-\sum_{i} \text{severities}_i$ |
| Factor de descuento | $\gamma$ | `AC_DISCOUNT` (por defecto 0.8545 en `CONFIG.py`) |

**Cardinalidad del espacio de estados** (con `MAX_SEVERITY = 9`):

$$|S| = 31 \times 11 \times 10 = 3{,}410 \text{ estados}$$

---

## 3. ¿Por qué Actor-Critic en lugar de DQN?

### 3.1 Filosofía del Enfoque

DQN aprende una **función de valor-acción** $Q(s, a)$ y deriva la política
de forma implícita como $\pi(s) = \arg\max_a Q(s, a)$.  Actor-Critic
aprende **explícitamente** una política $\pi_\theta(a \mid s)$ junto con
una función de valor de estado $V_\phi(s)$, combinando las ventajas de
métodos basados en política y basados en valor.

### 3.2 Comparación

| Aspecto | DQN (`pes_dqn`) | A2C (`pes_a2c`) |
|---------|-----------------|--------------------------|
| Representación de la política | Implícita: $\arg\max_a Q(s, a)$ | Explícita: $\pi_\theta(a \mid s)$ |
| Salida de la red | Q-values para todas las acciones | Actor: probabilidades; Critic: valor escalar |
| Tipo de aprendizaje | Off-policy (replay buffer) | On-policy (sin replay buffer) |
| Exploración | ε-greedy sobre Q-values | Muestreo on-policy de $\pi_\theta$ enmascarada + bono de entropía |
| Estabilidad | Target network | Ventaja centrada (Advantage) |
| Confianza meta-cognitiva | Heurística (entropía de Q-values) | Teóricamente fundamentada (entropía de $\pi$) |
| Nº de redes | 2 (online + target) | 2 (Actor + Critic) |

### 3.3 Ventaja Teórica de la Confianza en A2C

En `pes_dqn`, la "confianza" del agente se calcula como la entropía de
los Q-values normalizados — una heurística, ya que los Q-values **no son
una distribución de probabilidad**.

En `pes_a2c`, la salida del Actor $\pi_\theta(a \mid s)$ **es** una
distribución de probabilidad (softmax), por lo que la entropía tiene un
significado teórico preciso:

$$H(\pi_\theta(\cdot \mid s)) = -\sum_{a} \pi_\theta(a \mid s) \log \pi_\theta(a \mid s)$$

- $H$ bajo → el agente está **seguro** (la distribución se concentra en
  pocas acciones).
- $H$ alto → el agente está **inseguro** (la distribución es casi uniforme).

---

## 4. Arquitectura de las Redes Neuronales

### 4.1 Actor (Red de Política)

Implementada en `ext/ac_model.py` → `build_actor()`:

```
Input  (3)  →  Dense(128, ReLU)  →  Dense(11, softmax)
               hidden_units[0]      action_dim
```

La arquitectura es configurable: `AC_ACTOR_HIDDEN_UNITS` define las capas
ocultas (por defecto `[128]` — una sola capa de 128 neuronas, optimizada en
trial #90 del 2026-04-23, mean_perf = 0.887236).

```python
model = tf.keras.Sequential(name="Actor")
model.add(tf.keras.layers.Input(shape=(int(state_dim),)))
for idx, units in enumerate(hidden_units):
    layer_seed = None if seed is None else int(seed) + idx
    model.add(tf.keras.layers.Dense(
        int(units), activation="relu", name=f"actor_hidden_{idx}",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=layer_seed)))
out_seed = None if seed is None else int(seed) + len(hidden_units)

# Sesgo personalizado: ceros excepto en la última acción ("gastar el máximo").
bias_vec = numpy.zeros((int(action_dim),), dtype=numpy.float32)
if float(last_action_bias) != 0.0:
    bias_vec[-1] = float(last_action_bias)
bias_init = tf.keras.initializers.Constant(bias_vec)

model.add(tf.keras.layers.Dense(
    int(action_dim), activation="softmax", name="policy",
    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=out_seed),
    bias_initializer=bias_init))
```

El parámetro opcional `seed=` propaga una semilla a `GlorotUniform` para
que cada capa se inicialice de forma reproducible (cada capa recibe un
offset distinto).  Esto es independiente del contador global de
operaciones de TensorFlow y garantiza réplicas exactas entre procesos.

El parámetro `last_action_bias` (por defecto $-1.389$ tras trial #90)
inyecta un sesgo logit negativo en la última acción de la cabeza
*policy*, reduciendo la probabilidad inicial de la acción "gastar el
máximo factible" y rompiendo la simetría que conducía al colapso
``argmax → max-feasible`` observado en trials tempranos.

La capa de salida **softmax** garantiza que $\sum_a \pi_\theta(a \mid s) = 1$
y $\pi_\theta(a \mid s) \ge 0$, produciendo una distribución de
probabilidad válida.

**Parámetros** (con `AC_ACTOR_HIDDEN_UNITS = [128]`):

| Capa | Forma | Parámetros |
|------|-------|------------|
| `actor_hidden_0` | 3 → 128 | $3 \times 128 + 128 = 512$ |
| `policy` | 128 → 11 | $128 \times 11 + 11 = 1\,419$ |
| **Total Actor** | | **1 931** |

### 4.2 Critic (Red de Valor de Estado)

Implementada en `ext/ac_model.py` → `build_critic()`:

```
Input  (3)  →  Dense(128, ReLU)  →  Dense(1, linear)
               hidden_units[0]      scalar value
```

```python
model = tf.keras.Sequential(name="Critic")
model.add(tf.keras.layers.Input(shape=(int(state_dim),)))
for idx, units in enumerate(hidden_units):
    layer_seed = None if seed is None else int(seed) + idx
    model.add(tf.keras.layers.Dense(
        int(units), activation="relu", name=f"critic_hidden_{idx}",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=layer_seed)))
out_seed = None if seed is None else int(seed) + len(hidden_units)
model.add(tf.keras.layers.Dense(
    1, activation="linear", name="value",
    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=out_seed)))
```

La capa de salida **lineal** permite que $V_\phi(s)$ tome cualquier valor
real (las recompensas del entorno son negativas).

**Parámetros** (con `AC_CRITIC_HIDDEN_UNITS = [128]`):

| Capa | Forma | Parámetros |
|------|-------|------------|
| `critic_hidden_0` | 3 → 128 | $3 \times 128 + 128 = 512$ |
| `value` | 128 → 1 | $128 \times 1 + 1 = 129$ |
| **Total Critic** | | **641** |

### 4.3 Total de Parámetros

$$|\Theta_\text{total}| = |\Theta_\text{Actor}| + |\Theta_\text{Critic}| = 1\,931 + 641 = 2\,572$$

Con la arquitectura optimizada en trial #90 (una capa oculta de 128
neuronas tanto para Actor como para Critic), A2C usa ~0.50× menos
parámetros que DQN (~5 131).  Además, la mitad (Critic) solo se necesita
durante entrenamiento; en inferencia solo se usa el Actor (1 931
parámetros).

### 4.4 Normalización del Estado

Antes de alimentar las redes, el estado entero se escala a $[0, 1]^3$:

$$\hat{s} = \left(\frac{r}{30},\; \frac{t}{10},\; \frac{v}{9}\right)$$

Implementada en `ext/ac_model.py` → `normalize_state()`:

```python
numpy.array([
    state[0] / max(max_resources, 1),   # / (AVAILABLE_RESOURCES_PER_SEQUENCE - 9) = / 30
    state[1] / max(max_trials, 1),      # / NUM_MAX_TRIALS = / 10
    state[2] / max(max_severity, 1),    # / MAX_SEVERITY = / 9
], dtype=numpy.float32)
```

Los divisores se derivan de `CONFIG.py`:
`AVAILABLE_RESOURCES_PER_SEQUENCE = 39` menos los 9 recursos
pre-asignados al inicio del bloque ⇒ `max_resources = 30`;
`NUM_MAX_TRIALS = 10`; `MAX_SEVERITY = 9`.

---

## 5. Fundamento Teórico: Policy Gradient y Advantage

### 5.1 El Teorema del Gradiente de Política

El objetivo del Actor es maximizar el retorno esperado:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

El **Teorema del Gradiente de Política** (Sutton et al., 1999) establece
que:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \Psi_t \right]$$

donde $\Psi_t$ es una señal de refuerzo genérica.  En REINFORCE,
$\Psi_t = G_t$ (retorno desde $t$), lo que produce alta varianza.

### 5.2 La Función de Ventaja (Advantage)

Actor-Critic reduce la varianza reemplazando $G_t$ por la **ventaja**
(*advantage*):

$$A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$$

La ventaja mide cuánto **mejor (o peor)** fue la acción $a_t$ respecto al
valor promedio del estado $s_t$.  Usando la aproximación TD(0):

$$\hat{A}(s_t, a_t) = r_t + \gamma V_\phi(s_{t+1}) \cdot (1 - d_t) - V_\phi(s_t)$$

donde $d_t \in \{0, 1\}$ indica si el episodio terminó.

Esta señal tiene **media cero** en expectativa, lo que reduce drásticamente
la varianza del gradiente respecto a REINFORCE.

### 5.3 Intuición

- Si $A > 0$: la acción fue **mejor** que el promedio → aumentar
  $\pi_\theta(a_t \mid s_t)$.
- Si $A < 0$: la acción fue **peor** que el promedio → disminuir
  $\pi_\theta(a_t \mid s_t)$.
- Si $A \approx 0$: la acción fue **típica** → cambio mínimo.

---

## 6. Funciones de Pérdida

### 6.1 Pérdida del Critic (MSE sobre TD Target)

El Critic se entrena para minimizar el error entre su predicción y el
*target* de diferencia temporal:

$$y_t = r_t + \gamma \cdot V_\phi(s_{t+1}) \cdot (1 - d_t)$$

$$\mathcal{L}_\text{Critic} = \frac{1}{N} \sum_{t=1}^{N} \left( V_\phi(s_t) - y_t \right)^2$$

**Nota:** En la implementación, tanto `values` como `next_values` se calculan
**dentro** del `GradientTape` del Critic.  Esto significa que el gradiente de
`critic_loss` fluye a través de ambos términos — un enfoque de *semi-gradient*
que funciona bien en la práctica y simplifica el código.  La ventaja
(advantage) se calcula **después** de actualizar el Critic, usando los pesos
recién actualizados para obtener una señal más limpia.

### 6.2 Pérdida del Actor (Policy Gradient + Entropía)

El Actor se actualiza siguiendo el gradiente de política con ventaja,
**más** un bono de entropía que incentiva la exploración:

$$\mathcal{L}_\text{Actor} = -\frac{1}{N} \sum_{t=1}^{N} \left[ \log \pi_\theta(a_t \mid s_t) \cdot \hat{A}_t \right] - c_\text{ent} \cdot H(\pi_\theta)$$

donde:

- $\log \pi_\theta(a_t \mid s_t)$: log-probabilidad de la acción tomada.
- $\hat{A}_t = y_t - V_\phi(s_t)$: ventaja estimada (detached del Actor).
- $c_\text{ent}$: coeficiente de entropía (`AC_ENTROPY_COEFF`, default 0.005278).
- $H(\pi_\theta) = -\sum_a \pi_\theta(a \mid s) \log \pi_\theta(a \mid s)$:
  entropía de la política.

El signo negativo delante del primer término convierte el **ascenso** de
gradiente de política en una **minimización** compatible con
`optimizer.apply_gradients()`.

### 6.3 Implementación

`ext/ac_model.py` → `train_step_actor_critic()`:

```python
def train_step_actor_critic(actor, critic, actor_optimizer, critic_optimizer,
                            states, actions, rewards, next_states, dones,
                            discount, entropy_coeff, max_grad_norm, gae_lambda,
                            masks=None):
    # ---------- Critic update ----------
    with tf.GradientTape() as critic_tape:
        values = tf.squeeze(critic(states, training=True), axis=1)
        next_values = tf.squeeze(critic(next_states, training=False), axis=1)
        td_targets = rewards + discount * next_values * (1.0 - dones)
        critic_loss = tf.reduce_mean(tf.square(td_targets - values))

    critic_grads = critic_tape.gradient(critic_loss, critic.trainable_variables)
    critic_grads, _ = tf.clip_by_global_norm(critic_grads, max_grad_norm)
    critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

    # ---------- GAE(λ) Advantage ----------
    values_updated = tf.squeeze(critic(states, training=False), axis=1)
    next_values_updated = tf.squeeze(critic(next_states, training=False), axis=1)
    deltas = rewards + discount * next_values_updated * (1.0 - dones) - values_updated

    T = tf.shape(deltas)[0]
    gae_buffer = tf.TensorArray(dtype=tf.float32, size=T, dynamic_size=False)
    last_gae = tf.constant(0.0)
    for t in tf.range(T - 1, -1, -1):
        last_gae = deltas[t] + discount * gae_lambda * (1.0 - dones[t]) * last_gae
        gae_buffer = gae_buffer.write(t, last_gae)
    advantages = gae_buffer.stack()

    # Advantage normalisation
    adv_mean = tf.reduce_mean(advantages)
    adv_std = tf.math.reduce_std(advantages) + 1e-8
    advantages = (advantages - adv_mean) / adv_std

    # ---------- Actor update ----------
    with tf.GradientTape() as actor_tape:
        probs = actor(states, training=True)

        # Infeasible-action masking: zero out actions > resources_left
        # and renormalise so that log_probs and entropy reflect only
        # feasible actions — consistent with evaluation and __main__.
        if masks is not None:
            probs = probs * masks
            probs = probs / (tf.reduce_sum(probs, axis=1, keepdims=True) + 1e-8)

        probs_clipped = tf.clip_by_value(probs, 1e-8, 1.0)
        action_mask = tf.one_hot(actions, depth=tf.shape(probs)[1])
        log_probs = tf.reduce_sum(tf.math.log(probs_clipped) * action_mask, axis=1)

        entropy = -tf.reduce_sum(probs_clipped * tf.math.log(probs_clipped), axis=1)
        mean_entropy = tf.reduce_mean(entropy)

        # Policy gradient loss: -E[ log π(a|s) · Â_GAE(s,a) ] - c_ent · H(π)
        actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantages))
        actor_loss = actor_loss - entropy_coeff * mean_entropy

    actor_grads = actor_tape.gradient(actor_loss, actor.trainable_variables)
    actor_grads, _ = tf.clip_by_global_norm(actor_grads, max_grad_norm)
    actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))

    return actor_loss, critic_loss, mean_entropy
```

**Notas de implementación**:

- `train_step_actor_critic` **no** lleva `@tf.function` a nivel de módulo
  porque la optimización bayesiana (Optuna) ejecuta múltiples trials con
  diferentes instancias de modelo y optimizador.  `@tf.function` prohíbe
  crear nuevas `tf.Variable` dentro de un grafo ya trazado, lo que
  provocaría un error al iniciar el segundo trial.
- En su lugar, `A2CTraining()` en `pandemic.py` envuelve la función con
  `compiled_train_step = tf.function(train_step_actor_critic)` de forma
  **local** a cada trial, asegurando que cada grafo sea independiente.
- Los optimizadores se pre-construyen con `optimizer.build()` antes del
  loop de entrenamiento para que sus `tf.Variable` internas se creen
  fuera del `@tf.function`.
- `discount` y `entropy_coeff` se pasan como `tf.constant` (no como
  `float` de Python) para evitar re-tracing cuando sus valores cambian
  entre trials de Optuna.
- Los `td_targets` se calculan **dentro** del `GradientTape` del Critic
  (semi-gradient), lo que simplifica el código y funciona bien en la
  práctica.
- Tras actualizar el Critic, se **recomputan** los valores con los pesos
  nuevos (`values_updated`, `next_values_updated`) para obtener una
  ventaja más limpia.
- `tf.stop_gradient(advantages)` impide que el gradiente del Actor fluya
  hacia el Critic a través de la ventaja.
- `tf.clip_by_value(probs, 1e-8, 1.0)` previene $\log(0)$ en estados
  donde la política es casi determinística.
- Cuando se pasan **masks** (tensor de factibilidad), las probabilidades
  del Actor se multiplican por la máscara y se renormalizan antes de
  calcular `log_probs` y `entropy`, alineando el entrenamiento con la
  inferencia enmascarada.

---

## 7. Exploración: Muestreo On-Policy + Bono de Entropía

### 7.1 Política de Exploración

Durante el entrenamiento, A2C **muestrea acciones directamente** de la
distribución del Actor enmascarada sobre acciones factibles — sin
overlay $\varepsilon$-greedy.  Esto preserva la propiedad on-policy del
gradiente de política: las acciones provienen de $\pi_\theta$, por lo
que la actualización de $J(\theta)$ es no sesgada.

$$a_t \sim \tilde{\pi}_\theta(\cdot \mid s_t),\qquad
\tilde{\pi}_\theta(a \mid s_t) =
\frac{\pi_\theta(a \mid s_t)\, m_a(s_t)}{\sum_{a'}\pi_\theta(a' \mid s_t)\, m_{a'}(s_t)}$$

donde $m_a(s_t) \in \{0, 1\}$ es la máscara binaria de factibilidad
($m_a = 1$ si y sólo si $a \le \text{resources\_left}$).  Si todas las
probabilidades enmascaradas son numéricamente cero (caso degenerado),
se muestrea uniformemente sobre acciones factibles.

La exploración se garantiza por dos mecanismos complementarios:

1. **Aleatoriedad intrínseca de $\pi_\theta$**: la softmax produce una
   distribución con masa positiva en múltiples acciones, especialmente
   al inicio del entrenamiento cuando los logits son pequeños.
2. **Bono de entropía** en la pérdida del Actor: penaliza
   distribuciones puntiagudas a nivel de gradiente, incentivando la
   diversificación.

**Implementación** (`ext/pandemic.py` → `A2CTraining()`):

```python
probs = actor_model(state_norm[numpy.newaxis, :], training=False)[0].numpy()
feasibility_mask = numpy.zeros(action_dim, dtype=numpy.float32)
feasibility_mask[:min(int(state[0]), action_dim - 1) + 1] = 1.0
masked_probs = probs * feasibility_mask
masked_sum = masked_probs.sum()
if masked_sum > 1e-8:
    masked_probs = masked_probs / masked_sum
else:
    # Softmax degenerado → uniforme sobre acciones factibles
    masked_probs = feasibility_mask / feasibility_mask.sum()
action = int(numpy.random.choice(action_dim, p=masked_probs))
```

### 7.2 Parámetros $\varepsilon$ Legacy

Los parámetros `AC_EPSILON_INITIAL`, `AC_EPSILON_MIN`, `AC_WARMUP_RATIO`
y `AC_TARGET_RATIO` se conservan en `CONFIG.py` y en la firma de
`A2CTraining()` por compatibilidad de API, pero **no se usan** en la
selección de acciones bajo el esquema on-policy actual.  El bucle de
entrenamiento sigue computando el cronograma de decaimiento de
$\varepsilon$, pero su valor nunca se aplica a la rama de selección.

En la optimización bayesíana (`optimize_a2c.py`) estos parámetros se
fijan a 0/0/0/1 para no consumir presupuesto de búsqueda.

### 7.3 Bono de Entropía

El término de entropía en $\mathcal{L}_\text{Actor}$ actúa como un
**regularizador de exploración** a nivel de gradiente: si
$c_\text{ent} > 0$, el optimizador penaliza distribuciones demasiado
concentradas, manteniendo la diversidad de acciones a lo largo del
entrenamiento.  El espacio de búsqueda bayesíano ahora permite
$c_\text{ent} = 0$, dejando que Optuna desactive el bono si lo prefiere.

---

## 8. Bucle de Entrenamiento Completo

`ext/pandemic.py` → `A2CTraining()`:

```
Para cada episodio i = 1 … N:
    env.random_sequence()          ← secuencia aleatoria
    state, _ = env.reset()

    batch_states, batch_actions, batch_rewards,
    batch_next_states, batch_dones, batch_masks = [], [], [], [], [], []

    Mientras no done:
        1. Normalizar estado:     s_norm = normalize_state(state, 30, 10, 9)
        2. Selección on-policy con enmascaramiento de acciones infactibles:
              - probs = actor_predict(s_norm)            ← π_θ(·|s)  (tf.function JIT)
              - feasibility_mask ∈ {0, 1}¹¹  con 1 en [0, min(resources_left, 10)]
              - masked_probs = probs · feasibility_mask
              - masked_probs /= masked_probs.sum()       ← renormalizar
              - a ~ Multinomial(masked_probs)            ← numpy.random.choice
           No se usa ε-greedy ni argmax: la exploración proviene de la
           entropía natural de π_θ y del bono H(π_θ) en la pérdida del Actor.
        3. (Opcional) Meta-cognición: confidence = ac_agent_meta_cognitive(π(·|s), ...)
        4. PBRS: calcular Φ(s) = −Σ max(0, sev_i) ANTES del step
        5. Paso del entorno:      s', r, done, truncated, info = env.step(action)
        6. PBRS: r' = r + β·(γ·Φ(s') − Φ(s))  (si penalty_coeff > 0)
        7. Almacenar transición en batch de episodio:
              batch_states.append(s_norm)
              batch_actions.append(action)
              batch_rewards.append(r')
              batch_next_states.append(s'_norm)
              batch_dones.append(done)
              batch_masks.append(feasibility_mask)

    ── Fin del episodio ──
    8. Convertir batch a tensores:
         states_t, actions_t, rewards_t, next_t, dones_t,
         masks_t = cast_to_tensors(batch)
    9. Actualizar Actor y Critic (con GAE(λ), gradient clipping,
       normalización de ventaja y enmascaramiento de acciones infactibles):
         actor_loss, critic_loss, entropy = compiled_train_step(
             actor, critic, actor_opt, critic_opt,
             states_t, actions_t, rewards_t, next_t, dones_t,
             discount_t, entropy_coeff_t, max_grad_norm_t, gae_lambda_t,
             masks_t)
    10. Decaer ε (exponencial con warm-up)

    Cada 10 000 episodios:
        ─ Imprimir recompensa promedio

    Retornar (ave_reward_list, actor_model, conf_list)
```

### 8.1 Diferencia Clave: On-Policy vs. Off-Policy

| Aspecto | DQN (off-policy) | A2C (on-policy) |
|---------|-------------------|-----------------|
| Buffer | Replay buffer de 20 000 transiciones | Sin buffer; batch por episodio |
| Reutilización de datos | Cada transición se muestrea múltiples veces | Cada transición se usa exactamente una vez |
| Actualización | Cada 4 env steps (mini-batch del buffer) | Al final de cada episodio (batch completo) |
| Correlación temporal | Eliminada por muestreo aleatorio | Presente dentro del episodio |
| Eficiencia de datos | Alta (reutilización) | Baja (un solo uso) |
| Estabilidad del gradiente | Target network | Advantage centrada + entropía |

### 8.2 Parámetro `verbose`

`A2CTraining()` acepta un parámetro opcional `verbose` (por defecto `True`).
Cuando está desactivado (`verbose=False`), se suprimen los mensajes
periódicos de progreso que normalmente se imprimen cada 10 000 episodios.
Esto es especialmente útil durante la optimización bayesiana, donde
decenas de trials consecutivos generarían ruido excesivo en la terminal.

De forma análoga, `run_experiment()` también acepta `verbose` (por defecto
`True`).  Cuando `verbose=False`, se omiten las impresiones del estado
inicial y de los valores de severidad por secuencia.

### 8.3 Parámetro `compute_confidence`

Al igual que `DQNTraining()`, `A2CTraining()` acepta `compute_confidence`
(por defecto `False`).  Cuando está desactivado, se omite la llamada a
`ac_agent_meta_cognitive()` durante entrenamiento, ahorrando cómputo.

En A2C la meta-cognición **no requiere un forward pass adicional**: la
distribución $\pi_\theta(a \mid s)$ ya se calculó para el muestreo
on-policy y se reutiliza directamente.  No obstante,
`ac_agent_meta_cognitive()` realiza operaciones adicionales
(enmascaramiento, entropía, normalización) que se pueden omitir durante
entrenamiento intensivo.

### 8.4 Mejoras de Entrenamiento

Las siguientes mejoras se incorporaron al A2C base para aumentar la
estabilidad y convergencia del entrenamiento.  Todas son configurables
mediante `CONFIG.py` y optimizables vía búsqueda Bayesiana.

#### 8.4.1 Gradient Clipping — Recorte de Gradientes

**Teoría**: Los gradientes del Actor y Critic pueden explotar en
episodios con recompensas atípicas.  El recorte por norma global
(Mnih et al., 2016) limita el vector de gradientes completo a una
norma máxima $g_{\max}$, sin alterar la dirección:

$$\hat{g} = g \cdot \frac{g_{\max}}{\max(g_{\max},\; \|g\|_2)}$$

**Código** (`ext/ac_model.py` → `train_step_actor_critic()`):

```python
critic_grads, _ = tf.clip_by_global_norm(critic_grads, max_grad_norm)
critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
# ...
actor_grads, _ = tf.clip_by_global_norm(actor_grads, max_grad_norm)
actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
```

| Parámetro | CONFIG | Valor por defecto |
|-----------|--------|-------------------|
| `AC_MAX_GRAD_NORM` | Umbral de norma global | 1.200 |

#### 8.4.2 Normalización de la Ventaja (Advantage Normalisation)

**Teoría**: La escala de la ventaja varía según la magnitud de las
recompensas.  Normalizar la ventaja a media cero y varianza unitaria
estabiliza los gradientes del Actor independientemente de la escala
de recompensa:

$$\hat{A}_t^{\text{norm}} = \frac{\hat{A}_t - \mu_A}{\sigma_A + 10^{-8}}$$

**Código** (`ext/ac_model.py` → `train_step_actor_critic()`):

```python
adv_mean = tf.reduce_mean(advantages)
adv_std = tf.math.reduce_std(advantages) + 1e-8
advantages = (advantages - adv_mean) / adv_std
```

#### 8.4.3 GAE(λ) — Estimación Generalizada de la Ventaja

**Teoría** (Schulman et al., 2016): GAE interpola entre el estimador
TD(0) de baja varianza/alto sesgo ($\lambda=0$) y los retornos
Monte-Carlo de alta varianza/bajo sesgo ($\lambda=1$):

$$\hat{A}_t^{\text{GAE}} = \sum_{\ell=0}^{T-t-1} (\gamma \lambda)^\ell \delta_{t+\ell}$$

donde $\delta_t = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)$ es el
error TD de un paso.  La acumulación se realiza en reversa para
eficiencia $O(T)$:

$$G_t = \delta_t + \gamma \lambda (1 - d_t) \cdot G_{t+1}, \quad G_T = \delta_T$$

**Código** (`ext/ac_model.py` → `train_step_actor_critic()`):

```python
deltas = rewards + discount * next_values_updated * (1.0 - dones) - values_updated

T = tf.shape(deltas)[0]
gae_buffer = tf.TensorArray(dtype=tf.float32, size=T, dynamic_size=False)
last_gae = tf.constant(0.0)
for t in tf.range(T - 1, -1, -1):
    last_gae = deltas[t] + discount * gae_lambda * (1.0 - dones[t]) * last_gae
    gae_buffer = gae_buffer.write(t, last_gae)
advantages = gae_buffer.stack()
```

| Parámetro | CONFIG | Valor por defecto |
|-----------|--------|-------------------|
| `AC_GAE_LAMBDA` | λ (bias-varianza) | 0.913 |

> **Nota**: El bucle `tf.range` se compila a `tf.while_loop` mediante
> autograph, manteniendo la compatibilidad con `@tf.function`.

#### 8.4.4 Decaimiento Exponencial de ε con Warm-up (Legacy)

Esquema de decaimiento conservado por compatibilidad de API.  El código
continúa actualizando $\varepsilon$ al final de cada episodio, pero el
valor **no se aplica** en la nueva rama de selección on-policy descrita
en la sección 7.1.

#### 8.4.5 PBRS — Reward Shaping Basado en Potencial

**Teoría** (Ng et al., 1999): PBRS añade una recompensa de modelado que
provee señal más densa sin alterar la política óptima.  La función de
potencial $\Phi(s)$ codifica conocimiento del dominio:

$$r'_t = r_t + \beta \left[ \gamma \cdot \Phi(s_{t+1}) - \Phi(s_t) \right]$$

En el escenario Pandemic, usamos $\Phi(s) = -\sum_i \max(0,\; \text{sev}_i)$,
que es mayor (menos negativo) cuando las severidades son bajas — alineado
con el objetivo de minimizar la severidad.

**Código** (`ext/pandemic.py` → `A2CTraining()`, dentro del bucle de
episodio):

```python
# ANTES del step
phi_s = -sum(max(0.0, sv) for sv in env.severities)

state2, reward, done, _trunc, _info = env.step(action)

# DESPUÉS del step
phi_s_prime = 0.0 if done else -sum(max(0.0, sv) for sv in env.severities)
reward += penalty_coeff * (discount * phi_s_prime - phi_s)
```

| Parámetro | CONFIG | Valor por defecto |
|-----------|--------|-------------------|
| `AC_PENALTY_COEFF` | β (coeficiente de shaping) | 0.153 |

> Cuando `AC_PENALTY_COEFF = 0`, PBRS se desactiva completamente.

#### 8.4.6 Enmascaramiento de Acciones Infactibles durante Entrenamiento

**Teoría**: Durante la evaluación e inferencia, las acciones que exceden
los recursos disponibles ($a > \text{resources\_left}$) se enmascaran
estableciendo su probabilidad a cero y renormalizando.  Si el
entrenamiento **no** aplica el mismo enmascaramiento, el gradiente de
política refuerza acciones que nunca se ejecutarán, creando una
discrepancia entrenamiento⟷evaluación.  Enmascarar durante el
entrenamiento alinea ambos procesos y concentra la capacidad del Actor
en el subconjunto de acciones legales.

**Implementación**: En cada transición del bucle de entrenamiento
(`A2CTraining()` en `pandemic.py`) se genera una **máscara de
factibilidad** binaria de dimensión 11 (una posición por acción):

```python
feasibility_mask = numpy.zeros(action_dim, dtype=numpy.float32)
feasibility_mask[:min(int(state[0]), action_dim - 1) + 1] = 1.0
```

Las probabilidades del Actor se multiplican por la máscara y se
renormalizan para obtener $\tilde{\pi}_\theta(\cdot \mid s_t)$, de la
cual se muestrea la acción on-policy.  Si la suma renormalizada es
numéricamente cero (caso degenerado), se aplica un fallback uniforme
sobre acciones factibles.

Todas las máscaras del episodio se acumulan en `ep_masks` y se pasan
como tensor al paso de gradiente.  Dentro de
`train_step_actor_critic()`, la máscara se aplica **antes** de calcular
log-probabilidades y entropía:

```python
if masks is not None:
    probs = probs * masks
    probs = probs / (tf.reduce_sum(probs, axis=1, keepdims=True) + 1e-8)
```

De esta forma, $\log \pi_\theta(a_t \mid s_t)$ y $H(\pi_\theta)$ se
calculan únicamente sobre acciones factibles, y el bono de entropía
penaliza distribuciones uniformes **dentro** del rango factible, no
sobre acciones inválidas.

#### 8.4.7 Cosine Annealing — Programación de Tasa de Aprendizaje

**Teoría**: En lugar de mantener la tasa de aprendizaje constante, se
aplica un decaimiento con curva coseno (Loshchilov & Hutter, 2017):

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

donde $\eta_{\min} = \alpha \cdot \eta_0$ y $T$ = total de episodios.
Esto permite exploración agresiva al inicio (LR alta) y ajuste fino al
final (LR baja).

**Código** (`ext/pandemic.py` → `A2CTraining()`):

```python
actor_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=actor_lr,
    decay_steps=episodes,
    alpha=lr_min_ratio,
)
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_schedule)
```

| Parámetro | CONFIG | Valor por defecto |
|-----------|--------|-------------------|
| `AC_LR_MIN_RATIO` | α (ratio mínimo) | 0.237 |

> Se aplican schedules **independientes** para Actor y Critic, cada uno
> con su propio step counter interno.

---

## 9. Optimización Bayesiana de Hiperparámetros

`ext/optimize_a2c.py` utiliza **Optuna** (TPE sampler) para buscar
hiperparámetros óptimos del A2C.

### 9.1 Espacio de Búsqueda

En modo `full` (por defecto en CONFIG), Optuna muestrea **8 parámetros
base**:

| Parámetro | Rango | Tipo |
|-----------|-------|------|
| `actor_lr` | $[10^{-4},\; 10^{-2}]$ | log-uniforme |
| `critic_lr` | $[10^{-4},\; 10^{-2}]$ | log-uniforme |
| `discount_factor` | $[0.85,\; 0.995]$ | uniforme |
| `entropy_coeff` | $[0.0,\; 0.1]$ | lineal (incluye 0) |
| `num_episodes` | $[50\,000,\; 250\,000]$ | entero (paso 25k) |
| `actor_hidden_dim` | $\{32, 64, 128, 256\}$ | categórico |
| `critic_hidden_dim` | $\{32, 64, 128, 256\}$ | categórico |
| `n_hidden_layers` | $\{1, 2, 3\}$ | entero |

Los parámetros $\varepsilon$ (`epsilon_initial`, `epsilon_min`) ya **no
se muestrean** — se fijan a 0 porque el entrenamiento es puramente
on-policy (ver § 7).

Adicionalmente, se optimizan **6 hiperparámetros de mejora** (siempre
muestreados, tanto en modo `full` como `improvements_only`):

| Parámetro | Rango | Tipo |
|-----------|-------|------|
| `penalty_coeff` | $[0.0,\; 0.3]$ | lineal (incluye 0) |
| `gae_lambda` | $[0.90,\; 0.99]$ | uniforme |
| `max_grad_norm` | $[0.3,\; 1.5]$ | uniforme |
| `lr_min_ratio` | $[0.05,\; 0.25]$ | uniforme |
| `spend_cost_coeff` | $[0.0,\; 0.05]$ | lineal (coste de gasto, solo entrenamiento; 0 desactiva) |
| `last_action_bias` | $[-2.0,\; 0.0]$ | uniforme (sesgo logit inicial de la acción "gastar máximo") |

`warmup_ratio` y `target_ratio` se fijan a 0/1 (no se muestrean) porque
gobiernan el cronograma legacy de $\varepsilon$ que no se aplica en la
rama on-policy.

#### Modos de Optimización

- **`full`** (por defecto en `CONFIG.AC_OPTIMIZE_MODE`): optimiza los
  14 parámetros (8 base + 6 mejora).
- **`improvements_only`**: fija los 8 parámetros base en los valores de
  CONFIG.py y solo optimiza los 6 de mejora, reduciendo el espacio de
  búsqueda de 14D a 6D.  Se selecciona con `AC_OPTIMIZE_MODE` en
  CONFIG.py o `--mode` en la línea de comandos.

**Diferencias frente al espacio de DQN:**

- A2C tiene **dos learning rates** (Actor y Critic) en lugar de uno solo.
- Se añade `entropy_coeff` — ausente en DQN.
- No existen `batch_size`, `replay_buffer_size`, ni `target_sync_freq`
  (conceptos exclusivos de DQN off-policy).

### 9.2 Función Objetivo

Se entrena un agente A2C con los hiperparámetros sugeridos y se evalúa
en las **64 secuencias fijas** (las mismas que usa `__main__.py`).  El score
reportado a Optuna es el **rendimiento normalizado medio** (a maximizar),
calculado con `calculate_normalised_final_severity_performance_metric()`.

A diferencia de una evaluación determinista basada en `argmax`, la
evaluación se realiza de forma **estocástica** con
`n_eval_replicates = 8` réplicas independientes: en cada réplica las
acciones se muestrean de la *softmax enmascarada* (acciones $a >$
recursos disponibles puestas a probabilidad cero y renormalizadas) y se
promedia el rendimiento sobre las 8 réplicas.  Esto:

- Refleja la verdadera distribución de retornos de la política
  estocástica (no solo su moda).
- Inyecta varianza intra-configuración para que el TPE pueda estimar el
  ruido del objetivo.
- Penaliza políticas que colapsan a la heurística trivial
  ``argmax → max-feasible`` cuando esta no es óptima.

Cada réplica utiliza un sub-RNG distinto derivado de la semilla del
trial.

Cada trial usa una **semilla independiente** derivada como
`trial_seed = SEED + trial.number + 1`, de modo que distintos trials
son réplicas estocásticas independientes — sin esto Optuna vería
varianza cero por configuración y no podría estimar el ruido del
objetivo.  La semilla se propaga a `numpy`, `random`, `tf.random` y a
los inicializadores `GlorotUniform` de cada capa, y
`tf.config.experimental.enable_op_determinism()` se invoca al comienzo
de `A2CTraining()` para forzar grafos deterministas en CPU.

Durante la evaluación, tanto `A2CTraining()` como `run_experiment()` se
invocan con `verbose=False` para suprimir las impresiones de consola por
episodio/secuencia, evitando ruido en la salida de terminal durante
decenas de trials consecutivos.

### 9.3 Poda Temprana (MedianPruner)

El estudio de Optuna incorpora un **MedianPruner** configurado con
`n_startup_trials=10` y `n_warmup_steps=2`.  Este pruner descarta trials
cuyo rendimiento intermedio es inferior a la mediana de trials anteriores,
acelerando la búsqueda al evitar completar trials prometedoramente malos:

```python
study = optuna.create_study(
    ...
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    ...
)
```

### 9.4 Warm-Start

La búsqueda comienza con un **trial semilla** usando los valores de CONFIG.py
(correspondientes al mejor trial #90 de la optimización previa,
mean_perf = 0.887236), asegurando
que al menos un trial alcance un rendimiento razonable.  En modo
`improvements_only`, solo se incluyen los 6 parámetros de mejora
sampleados; en modo `full`, se incluyen también los 8 parámetros base.
Las claves del warm-start que ya no se muestrean (p. ej. `epsilon_*`,
`warmup_ratio`, `target_ratio`) son ignoradas silenciosamente por
Optuna:

```python
warmstart: dict = {}
if _optimize_mode == 'full':
    warmstart.update({
        'actor_lr': AC_ACTOR_LR,
        'critic_lr': AC_CRITIC_LR,
        'discount_factor': AC_DISCOUNT,
        'entropy_coeff': AC_ENTROPY_COEFF,
        'epsilon_initial': AC_EPSILON_INITIAL,
        'epsilon_min': AC_EPSILON_MIN,
        'num_episodes': 250000,
        'actor_hidden_dim': AC_ACTOR_HIDDEN_UNITS[0],
        'critic_hidden_dim': AC_CRITIC_HIDDEN_UNITS[0],
        'n_hidden_layers': len(AC_ACTOR_HIDDEN_UNITS),
    })
warmstart.update({
    'warmup_ratio': AC_WARMUP_RATIO,
    'target_ratio': AC_TARGET_RATIO,
    'penalty_coeff': AC_PENALTY_COEFF,
    'gae_lambda': AC_GAE_LAMBDA,
    'max_grad_norm': AC_MAX_GRAD_NORM,
    'lr_min_ratio': AC_LR_MIN_RATIO,
    'spend_cost_coeff': AC_SPEND_COST_COEFF,
    'last_action_bias': AC_LAST_ACTION_BIAS,
})
study.enqueue_trial(warmstart)
```

---

## 10. Inferencia en Tiempo de Experimento

### 10.1 Carga del Modelo Actor

`src/pygameMediator.py` → `provide_ac_agent_response()`:

```python
model = tf.keras.models.load_model(model_path)
```

Solo se carga el **Actor** — el Critic no es necesario en inferencia.

### 10.2 Selección de Acción

```python
s_norm = normalize_state([resources_left, city_trial_no, severity], 30, 10, 9)
policy_probs = model(s_norm[numpy.newaxis, :], training=False)[0].numpy()
action = int(numpy.argmax(policy_probs))
```

A diferencia de DQN (que toma $\arg\max$ de Q-values), aquí se toma
$\arg\max$ de $\pi_\theta(a \mid s)$ — la acción más probable bajo la
política aprendida.  Se aplica enmascaramiento de acciones infactibles
(`action > resources_left`) estableciendo su probabilidad a 0 antes del
argmax.

### 10.3 Confianza Meta-Cognitiva

El vector $\pi_\theta(a \mid s)$ (11 probabilidades) se pasa a
`ac_agent_meta_cognitive()` que calcula entropía y la normaliza:

$$H(\pi_\theta(\cdot \mid s)) = -\sum_{a=0}^{10} \pi_\theta(a \mid s) \log \pi_\theta(a \mid s)$$

$$\text{confidence} = \frac{H - H_\text{max}}{H_\text{min} - H_\text{max}}$$

donde $H_\text{min}$ y $H_\text{max}$ se calculan sobre las acciones
factibles únicamente.

---

## 11. Comparación con Algoritmos Anteriores

| Componente | `pes_base` (Q-tabular) | `pes_dqn` (DQN) | `pes_a2c` (A2C) |
|------------|-------------------|-----------------|--------------------------|
| Modelo | `numpy.ndarray` (q.npy) | Red 5 131 params (.keras) | Actor 1 931 + Critic 641 params |
| Update | $Q(s,a) \leftarrow Q + \alpha[r + \gamma \max Q - Q]$ | Huber loss + replay | Policy gradient + MSE + entropía |
| Datos | Un paso → un update | Replay buffer → mini-batch | Batch de episodio → un update |
| Política | Implícita ($\arg\max Q$) | Implícita ($\arg\max Q$) | Explícita ($\pi_\theta$) |
| Confianza | Entropía de Q (heurística) | Entropía de Q (heurística) | Entropía de $\pi$ (teórica) |
| Exploración | ε-greedy | ε-greedy | Muestreo on-policy de $\pi$ + bono de entropía |
| Episodios típicos | 900 000 | 175 000 | 250 000 (opt. bayesiana: 50k–250k) |

---

## 12. Optimizaciones para CPU

### 12.1 `tf.function` por Trial (JIT Compilado)

`train_step_actor_critic` se envuelve con `tf.function` **localmente**
dentro de cada llamada a `A2CTraining`, creando un grafo JIT-compilado
fresco por trial de Optuna.  Esto elimina el overhead de eager mode
que sería particularmente costoso dado que A2C realiza una actualización
por episodio (vs. cada 4 steps en DQN), y a la vez evita conflictos
de `tf.Variable` entre trials.

Además, los hiperparámetros escalares (`discount`, `entropy_coeff`,
`max_grad_norm`, `gae_lambda`) se convierten a `tf.constant` antes del
loop para que `tf.function` no retrace el grafo en cada trial con valores
distintos.

### 12.2 Reutilización del Forward Pass para Confianza

El parámetro `compute_confidence=False` (por defecto) omite el cálculo
de meta-cognición durante entrenamiento.  En A2C esto **no** ahorra un
forward pass adicional (la distribución del Actor ya se computa para
el muestreo on-policy y simplemente se descarta sin invocar
`ac_agent_meta_cognitive`), pero sí evita el cómputo de entropía,
enmascaramiento y normalización por step.

### 12.3 Configuración de Hilos TensorFlow

Al importar `ext/ac_model.py` se configuran los pools de hilos de TF:

```python
tf.config.threading.set_intra_op_parallelism_threads(0)   # auto-detect
tf.config.threading.set_inter_op_parallelism_threads(2)
```

### 12.4 Solo el Actor en Inferencia

Durante el experimento (`__main__.py`), solo se carga y ejecuta el Actor.
El Critic se descarta tras el entrenamiento, reduciendo la memoria en
inferencia a ~50 % del total.

---

## 13. Estructura de Archivos

```
pes_a2c/
├── __init__.py              # Exporta constantes AC_*; configura variables de entorno TF
├── __main__.py              # Valida ac_actor.keras antes de ejecutar
├── config/CONFIG.py         # 17 constantes AC_* (arquitectura, training, mejoras, optimización)
├── ext/
│   ├── ac_model.py          # build_actor, build_critic, normalize_state,
│   │                        #   train_step_actor_critic (masking + GAE(λ);
│   │                        #   sin @tf.function, se envuelve por trial
│   │                        #   en pandemic.py);
│   │                        #   configura tf.config.threading al importar
│   ├── pandemic.py          # Entorno Gymnasium Pandemic +
│   │                        #   ac_agent_meta_cognitive (entropía → confianza),
│   │                        #   run_experiment (evaluación sobre secuencias) y
│   │                        #   A2CTraining (masking, PBRS, GAE, cosine LR)
│   ├── train_a2c.py         # Pipeline de entrenamiento autónomo
│   ├── optimize_a2c.py       # Búsqueda Bayesiana con Optuna
│   └── tools.py             # Entropía, gráficas (sin cambios)
├── src/
│   ├── pygameMediator.py    # Carga ac_actor.keras, forward pass en experimento
│   ├── exp_utils.py         # Severidades, secuencias
│   ├── log_utils.py         # Logging dual
│   ├── result_formatter.py  # Gráficas matplotlib
│   └── terminal_utils.py    # UI de consola Rich
└── doc/
    ├── explained_ac.md              # ← este documento
    └── how_to_train_and_test.md     # Guía práctica de entrenamiento y pruebas
```

---

## 14. Formulario Resumen de Ecuaciones

| Concepto | Ecuación |
|----------|----------|
| Objetivo del Actor | $J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_t \gamma^t r_t\right]$ |
| Política gradiente | $\nabla_\theta J = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a \mid s) \cdot \hat{A}\right]$ |
| Ventaja (Advantage) | $\hat{A}_t = r_t + \gamma V_\phi(s_{t+1})(1 - d_t) - V_\phi(s_t)$ |
| TD Target | $y_t = r_t + \gamma V_\phi(s_{t+1})(1 - d_t)$ |
| Pérdida del Critic | $\mathcal{L}_\text{C} = \frac{1}{N}\sum_t (V_\phi(s_t) - y_t)^2$ |
| Pérdida del Actor | $\mathcal{L}_\text{A} = -\frac{1}{N}\sum_t [\log \pi_\theta(a_t \mid s_t) \hat{A}_t] - c_\text{ent} H(\pi_\theta)$ |
| Entropía | $H(\pi) = -\sum_a \pi(a \mid s) \log \pi(a \mid s)$ |
| Muestreo on-policy | $a_t \sim \tilde{\pi}_\theta(\cdot \mid s_t)$, con $\tilde{\pi} \propto \pi \cdot m$ |
| GAE(λ) | $\hat{A}_t^{\text{GAE}} = \sum_{\ell} (\gamma\lambda)^\ell \delta_{t+\ell}$ |
| PBRS | $r'_t = r_t + \beta[\gamma \Phi(s') - \Phi(s)]$ |
| Potencial | $\Phi(s) = -\sum_i \max(0, \text{sev}_i)$ |
| Cosine Annealing | $\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_0 - \eta_{\min})(1 + \cos(\pi t / T))$ |
| Gradient Clipping | $\hat{g} = g \cdot \min(1, g_{\max} / \|g\|_2)$ |
| Advantage Norm. | $\hat{A}^{\text{norm}} = (\hat{A} - \mu_A) / (\sigma_A + 10^{-8})$ |
| Enmascaramiento | $\tilde{\pi}(a \mid s) = \pi(a \mid s) \cdot m_a \;/\; \sum_{a'} \pi(a' \mid s) \cdot m_{a'}$ |
| Normalización | $\hat{s} = (r/30, t/10, v/9)$ |

---

## 15. Referencias

1. Sutton, R. S. et al. (1999). *Policy gradient methods for reinforcement
   learning with function approximation*. NeurIPS.
2. Mnih, V. et al. (2016). *Asynchronous methods for deep reinforcement
   learning*. ICML.  (Introduce A3C; A2C es la variante síncrona.)
3. Williams, R. J. (1992). *Simple statistical gradient-following
   algorithms for connectionist reinforcement learning*. Machine Learning,
   8(3-4), 229–256.  (REINFORCE.)
4. Kingma, D. P. & Ba, J. (2015). *Adam: A Method for Stochastic
   Optimization*. ICLR 2015.
5. Akiba, T. et al. (2019). *Optuna: A next-generation hyperparameter
   optimization framework*. KDD '19.
6. Schulman, J. et al. (2016). *High-Dimensional Continuous Control Using
   Generalized Advantage Estimation*. ICLR 2016.
7. Ng, A. Y. et al. (1999). *Policy Invariance Under Reward Transformations:
   Theory and Application to Reward Shaping*. ICML '99.
8. Loshchilov, I. & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent
   with Warm Restarts*. ICLR 2017.
