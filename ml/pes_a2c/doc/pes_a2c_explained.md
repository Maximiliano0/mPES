# `pes_a2c` — Guía de uso e implementación

> **Algoritmo**: Advantage Actor-Critic (A2C)
> **Tipo**: Método de gradiente de política (policy gradient) con línea base
> **Última actualización**: 2026-04-30

---

## 1. ¿Qué es A2C?

**Advantage Actor-Critic (A2C)** es un algoritmo de aprendizaje por refuerzo
profundo que combina dos paradigmas clásicos:

- **Actor (política)**: una red neuronal que produce directamente una
  distribución de probabilidad sobre las acciones, $\pi_\theta(a \mid s)$.
- **Critic (función de valor)**: una segunda red neuronal que estima el valor
  esperado del estado, $V_\phi(s)$, y se utiliza como **línea base** para
  reducir la varianza del gradiente de política.

A diferencia de los métodos basados en valor (como Q-Learning, DQN o RDQN), que
aprenden $Q(s,a)$ y eligen la acción mediante $\arg\max_a Q(s,a)$, A2C aprende
**directamente la política** maximizando el rendimiento esperado mediante el
gradiente:

$$\nabla_\theta J(\theta) = \mathbb{E}\big[\nabla_\theta \log \pi_\theta(a\mid s)\, A(s,a)\big]$$

donde $A(s,a) = r + \gamma V(s') - V(s)$ es la **función de ventaja** estimada
por el crítico.

### Separación actor / crítico

| Componente | Entrada | Salida | Pérdida |
|---|---|---|---|
| **Actor** | estado $s$ normalizado en $[0,1]^3$ | softmax sobre 11 acciones (0–10) | $-\log \pi(a\mid s)\,A(s,a) - \beta H(\pi)$ |
| **Critic** | estado $s$ normalizado en $[0,1]^3$ | escalar $V(s)$ | $\mathrm{MSE}\big(r + \gamma V(s'),\ V(s)\big)$ |

El término $-\beta H(\pi)$ es una **bonificación de entropía** que fomenta la
exploración penalizando políticas demasiado deterministas.

### Ventaja sobre métodos tabulares y DQN

A2C ofrece varias propiedades atractivas para el escenario *Pandemic*:

1. **Política estocástica nativa**: emite probabilidades sobre acciones, lo que
   habilita ensembles por **votación blanda** con otros agentes (ver `pes_ens`).
2. **Confianza interpretable**: $\max_a \pi(a\mid s)$ funciona como medida de
   certeza del agente sobre su decisión.
3. **Generalización continua**: al ser una red neuronal sobre el estado
   normalizado, no sufre la explosión combinatoria del Q-Learning tabular.
4. **Convergencia estable**: el crítico reduce drásticamente la varianza
   respecto a REINFORCE puro.

---

## 2. Comandos de uso

### Ejecutar el experimento completo

```powershell
# Windows
.\win_mpes_env\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"
$env:TF_ENABLE_ONEDNN_OPTS = "0"
python -m ml.pes_a2c
```

```bash
# Linux
source linux_mpes_env/bin/activate
export PYTHONIOENCODING=utf-8
export TF_ENABLE_ONEDNN_OPTS=0
python -m ml.pes_a2c
```

Esto ejecuta `ml/pes_a2c/__main__.py`, que corre los 8 bloques × 8 secuencias
× 3–10 trials del escenario y produce las salidas en `ml/pes_a2c/outputs/`.

### Entrenar el modelo

```powershell
python -m ml.pes_a2c.ext.train_a2c                # número por defecto de episodios
python -m ml.pes_a2c.ext.train_a2c 100000         # personalizado
```

Salida principal: `ml/pes_a2c/inputs/ac_actor.keras`.

### Optimización Bayesiana de hiperparámetros

```powershell
python -m ml.pes_a2c.ext.optimize_a2c             # número por defecto de trials
python -m ml.pes_a2c.ext.optimize_a2c 100         # 100 trials Optuna
```

Genera/actualiza `ml/pes_a2c/inputs/best_params.json` con los mejores
hiperparámetros encontrados.

---

## 3. Pipeline de entrenamiento (`train_a2c.py`)

El script `ext/train_a2c.py` ejecuta las siguientes etapas:

1. **Carga de configuración**: lee `config/CONFIG.py` y, si existe,
   sobrescribe valores con `inputs/best_params.json`.
2. **Inicialización del entorno**: instancia `PandemicEnv` desde
   `ext/pandemic.py` (Gymnasium-compatible).
3. **Construcción de redes**: crea las redes Actor y Critic en
   `ext/ac_model.py` mediante `build_actor()` y `build_critic()`.
4. **Bucle de episodios**: por cada episodio
   - Se ejecuta una secuencia completa (resources=39, severity inicial dada).
   - Se acumulan trayectorias $(s_t, a_t, r_t, s_{t+1})$.
   - Al terminar el episodio se llama a `train_step_a2c()` con el batch
     completo de la trayectoria (Monte Carlo + bootstrap del crítico).
5. **Logging**: dual-stream a consola y a archivo
   `outputs/PES_A2C_log_<fecha>.txt` mediante `src/log_utils.py`.
6. **Guardado periódico**: el actor se persiste como `inputs/ac_actor.keras`
   cada N episodios y al final.

### Función `train_step_a2c()`

Pseudo-código:

```python
def train_step_a2c(states, actions, rewards, next_states, dones):
    # 1. Crítico predice V(s) y V(s')
    V_s    = critic(states)
    V_s_next = critic(next_states)

    # 2. Target y ventaja
    target = rewards + gamma * V_s_next * (1 - dones)
    advantage = target - V_s

    # 3. Pérdida del actor
    log_probs = log(actor(states))[actions]
    actor_loss = -mean(log_probs * stop_gradient(advantage))
    entropy   = -sum(actor(states) * log(actor(states)))
    actor_loss -= beta_entropy * entropy

    # 4. Pérdida del crítico
    critic_loss = MSE(target, V_s) * critic_coef

    # 5. Backprop separado para cada red
    actor_optimizer.minimize(actor_loss, actor.trainable_variables)
    critic_optimizer.minimize(critic_loss, critic.trainable_variables)
```

---

## 4. Optimización Bayesiana (`optimize_a2c.py`)

`ext/optimize_a2c.py` usa **Optuna** (Akiba et al., 2019) con muestreador TPE
para buscar la mejor configuración:

| Hiperparámetro (Optuna) | Rango | Valor óptimo (CONFIG) |
|---|---|---|
| `actor_lr` | $[10^{-4}, 10^{-2}]$ log | $\approx 6{,}5 \times 10^{-4}$ |
| `critic_lr` | $[10^{-4}, 10^{-2}]$ log | $\approx 4{,}3 \times 10^{-3}$ |
| `discount_factor` ($\gamma$) | $[0{,}85, 0{,}995]$ | $\approx 0{,}854$ |
| `entropy_coeff` ($\beta$) | $[0{,}0, 0{,}1]$ lineal | $\approx 0{,}0053$ |
| `actor_hidden_dim` / `critic_hidden_dim` | $\{32, 64, 128, 256\}$ | $128 / 128$ |
| `n_hidden_layers` | $\{1, 2, 3\}$ | $1$ |
| `num_episodes` | $[50\,000, 250\,000]$ paso $25\,000$ | según trial |
| `penalty_coeff` (PBRS) | $[0{,}0, 0{,}3]$ | $\approx 0{,}153$ |
| `gae_lambda` | $[0{,}90, 0{,}99]$ | $\approx 0{,}913$ |
| `max_grad_norm` | $[0{,}3, 1{,}5]$ | $\approx 1{,}20$ |
| `lr_min_ratio` | $[0{,}05, 0{,}25]$ | $\approx 0{,}237$ |
| `spend_cost_coeff` | $[0{,}0, 0{,}05]$ | $\approx 0{,}0104$ |
| `last_action_bias` | $[-2{,}0, 0{,}0]$ | $\approx -1{,}39$ |

Nota: el actor y el crítico tienen **dos optimizadores Adam separados**
con learning-rates independientes (`AC_ACTOR_LR` y `AC_CRITIC_LR`); la
búsqueda los muestrea por separado.

Cada trial entrena un agente reducido (menos episodios) y devuelve el
**rendimiento medio normalizado** sobre 64 evaluaciones independientes.

### Almacenamiento

- **Estudio Optuna**: SQLite en `inputs/<fecha>_BAYESIAN_OPT/study.db`.
- **Mejores parámetros**: `inputs/best_params.json`.
- **Logs**: `outputs/PES_A2C_log_<fecha>_BAYESIAN_OPT.txt`.

Para visualizar el progreso del estudio:

```powershell
.\utils\win\optuna_dashboard.ps1 ml\pes_a2c\inputs\<fecha>_BAYESIAN_OPT\study.db
```

---

## 5. Estructura del código

### `ext/ac_model.py`

Contiene la arquitectura y el paso de entrenamiento.

```python
class Actor(tf.keras.Model):
    """Red densa con softmax sobre 11 acciones."""
    def __init__(self, hidden_units=[128], n_actions=11):   # AC_ACTOR_HIDDEN_UNITS
        ...
    def call(self, state):
        # state: (batch, 3)
        # return: (batch, 11) probabilidades

class Critic(tf.keras.Model):
    """Red densa que estima V(s) escalar."""
    def __init__(self, hidden_units=[128]):                 # AC_CRITIC_HIDDEN_UNITS
        ...
    def call(self, state):
        # state: (batch, 3)
        # return: (batch, 1)

def train_step_a2c(actor, critic, optimizer_a, optimizer_c, batch,
                   gamma, entropy_coeff):
    """Un paso de gradiente actor-crítico (dos optimizadores Adam)."""
    ...
```

### `ext/pandemic.py`

Define `PandemicEnv` (Gymnasium):

- **Estado**: `[resources_left/30, trial_no/10, severity/9]` $\in [0,1]^3$
  (con `max_resources = AVAILABLE_RESOURCES_PER_SEQUENCE − 9 = 30`).
- **Acción**: entero discreto $\in \{0, 1, \dots, 10\}$.
- **Recompensa**: $-\sum \mathrm{severity}_{ciudad}$.
- **Dinámica**: $\mathrm{severity}_{t+1} = \max(0, 1.4 \cdot \mathrm{severity}_t - 0.4 \cdot a_t)$.

### `ext/train_a2c.py`

Orquesta el bucle de entrenamiento, logging, checkpoints y métricas.

### `ext/optimize_a2c.py`

Define `objective(trial)` para Optuna y la función `main(n_trials)`.

### `__main__.py`

Carga el actor entrenado (`ac_actor.keras`), ejecuta los 8×8×(3..10) trials del
escenario *Pandemic* y persiste resultados.

---

## 6. Cómo funciona la inferencia

En `__main__.py`, para cada trial:

1. Se construye el estado normalizado $s$.
2. Se llama al actor: $\pi(\cdot \mid s) = \mathrm{actor}(s)$.
3. Se calcula la **máscara de factibilidad**: cualquier acción
   $a > \mathrm{resources\_left}$ recibe probabilidad cero.
4. Se renormaliza la distribución sobre las acciones factibles.
5. Se elige la acción mediante **argmax** sobre las probabilidades enmascaradas
   (modo determinista en evaluación).
6. La **confianza** se reporta como $\max_a \pi(a\mid s)$ después del
   enmascaramiento.

```python
def select_action(actor, state, resources_left):
    probs = actor(state[None, :]).numpy()[0]   # (11,)
    mask = numpy.arange(11) <= resources_left
    probs = probs * mask
    probs = probs / probs.sum()
    action = int(numpy.argmax(probs))
    confidence = float(probs[action])
    return action, confidence, probs
```

Esta probabilidad enmascarada es la que el módulo `pes_ens` consume para la
**votación blanda** del ensemble.

---

## 7. Archivos de entrada/salida

### Entradas (`ml/pes_a2c/inputs/`)

| Archivo | Descripción |
|---|---|
| `ac_actor.keras` | Red Actor entrenada (única persistida) |
| `best_params.json` | Mejores hiperparámetros del estudio Optuna |
| `initial_severity.csv` | Severidades iniciales por secuencia |
| `sequence_lengths.csv` | Número de trials por secuencia |
| `rewards.npy` | Recompensas históricas (opcional) |
| `<fecha>_BAYESIAN_OPT/` | Bases de datos Optuna por fecha |
| `<fecha>_A2C_TRAIN/` | Checkpoints intermedios del entrenamiento |

> El **crítico no se persiste**: solo es necesario durante el entrenamiento
> para calcular la ventaja. En inferencia basta con el actor.

### Salidas (`ml/pes_a2c/outputs/`)

| Archivo / carpeta | Descripción |
|---|---|
| `PES_A2C_log_<fecha>.txt` | Log dual-stream del experimento |
| `<fecha>_A2C_AGENT/` | Resultados del experimento (CSV + figuras) |

---

## 8. Resultados de rendimiento

Evaluación del 30 de abril de 2026 sobre $n = 64$ ejecuciones independientes
del escenario completo:

| Métrica | Valor |
|---|---|
| **Rendimiento medio normalizado** | **0.887236** |
| Desviación estándar | 0.063162 |
| Tamaño de muestra | 64 |

### Comparación con métodos tabulares

| Agente | Algoritmo | Rendimiento medio |
|---|---|---|
| `pes_base` | Q-Learning tabular | $\approx 0.65$ |
| `pes_ql` | Q-Learning + Optuna | $\approx 0.78$ |
| `pes_dql` | Double Q-Learning + PBRS | $\approx 0.83$ |
| **`pes_a2c`** | **A2C** | **0.887** |
| `pes_dqn` | DQN | $\approx 0.89$ |
| `pes_rdqn` | Recurrent DQN | $\approx 0.91$ |
| `pes_trf` | Causal Transformer | $\approx 0.927$ |

A2C iguala prácticamente al DQN como método sin memoria, demostrando el valor
del paradigma policy-gradient en este escenario. Su mayor utilidad práctica es
como **componente del ensemble** (`pes_ens`), donde la diversidad respecto a
los métodos basados en valor mejora la robustez global.

---

## 9. Ventajas y limitaciones

### Ventajas del policy gradient

- **Distribución sobre acciones**: ideal para ensembles por votación blanda.
- **Política estocástica natural**: útil en escenarios con simetrías o
  multimodalidad.
- **Sin necesidad de $\arg\max$ exhaustivo**: la política es el output directo
  de la red.
- **Bonificación de entropía**: control fino del trade-off
  exploración/explotación.
- **Convergencia más suave**: el crítico actúa como línea base que reduce la
  varianza del gradiente.

### Limitaciones

- **Sin memoria**: A2C en su forma básica no captura dependencias temporales
  más allá del estado actual (ver `pes_rdqn`, `pes_trf` para variantes con
  memoria).
- **Sensibilidad a hiperparámetros**: $\beta$ (entropía) y la relación
  actor/crítico requieren búsqueda cuidadosa (Optuna lo gestiona).
- **Two-network overhead**: dos optimizadores y dos redes que mantener
  sincronizados; mayor coste por episodio que un DQN equivalente.
- **Recompensas dispersas**: la varianza del gradiente puede crecer en
  escenarios con feedback escaso.

---

## 10. Referencias

- Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T.,
  Silver, D., & Kavukcuoglu, K. (2016). Asynchronous methods for deep
  reinforcement learning. *Proceedings of the 33rd International Conference on
  Machine Learning*, 1928–1937.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An
  introduction* (2nd ed.). MIT Press.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for
  connectionist reinforcement learning. *Machine Learning, 8*(3–4), 229–256.
- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A
  next-generation hyperparameter optimization framework. *Proceedings of the
  25th ACM SIGKDD International Conference on Knowledge Discovery & Data
  Mining*, 2623–2631.
