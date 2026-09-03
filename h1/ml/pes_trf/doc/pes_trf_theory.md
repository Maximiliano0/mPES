# `pes_trf` — Fundamentos teóricos

> **Tema**: Transformers causales como Q-network para RL secuencial
> **Última actualización**: 2026-05-02

---

## 1. Arquitectura Transformer: visión general

El **Transformer** (Vaswani et al., 2017) es una arquitectura neuronal
basada exclusivamente en mecanismos de **atención**, sin recurrencia ni
convolución. Su unidad fundamental es la **atención dot-product escalada**:

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

donde:

- $Q \in \mathbb{R}^{n \times d_k}$ son las **queries**.
- $K \in \mathbb{R}^{n \times d_k}$ son las **keys**.
- $V \in \mathbb{R}^{n \times d_v}$ son los **values**.
- $\frac{1}{\sqrt{d_k}}$ es un factor de normalización para estabilizar
  gradientes cuando $d_k$ es grande.

Cada fila de la salida es una **combinación lineal convexa** de las filas de
$V$, donde los pesos provienen del producto escalar entre la query
correspondiente y todas las keys.

### 1.1. Self-attention

En **self-attention**, $Q$, $K$ y $V$ se obtienen de la misma secuencia de
entrada $X \in \mathbb{R}^{n \times d}$ mediante proyecciones aprendidas:

$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

Esto permite que cada posición de la secuencia **atienda** a todas las demás
posiciones, capturando dependencias a cualquier distancia con un único paso
de red (a diferencia de las RNN, que requieren $O(n)$ pasos secuenciales).

---

## 2. Multi-head attention

En lugar de una sola operación de atención, el Transformer ejecuta $H$
**cabezas** en paralelo, cada una con sus propias proyecciones:

$$\mathrm{head}_h = \mathrm{Attention}(X W_h^Q,\ X W_h^K,\ X W_h^V)$$

$$\mathrm{MultiHead}(X) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_H)\, W^O$$

con $W_h^Q, W_h^K, W_h^V \in \mathbb{R}^{d \times d_k}$ y $W^O \in
\mathbb{R}^{H d_k \times d}$.

### Justificación

Cada cabeza aprende a **atender a un aspecto distinto** de las relaciones
entre posiciones. Por ejemplo, una cabeza puede capturar correlaciones a
corto plazo (último trial) y otra a largo plazo (primeros trials de la
secuencia).

En `pes_trf` por defecto: $H = 4$ (el espacio de búsqueda admite
$\{2, 4, 8\}$), $d_\mathrm{model} = 32$, por tanto
$d_k = d_\mathrm{model}/H = 16$.

---

## 3. Atención causal (autoregresiva)

En tareas secuenciales donde se predice el siguiente paso, la posición $i$
**no debe atender a posiciones $j > i$** (futuro). Esto se impone añadiendo
una **máscara triangular** antes del softmax:

$$M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

$$\mathrm{Attention}_\mathrm{causal} = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V$$

Aplicar $\exp(-\infty) = 0$ tras el softmax bloquea totalmente la atención
hacia posiciones futuras.

### Por qué es esencial en `pes_trf`

En el escenario *Pandemic*, la decisión en el trial $t$ debe depender solo
de los trials $\{0, 1, \dots, t\}$ ya observados. Sin máscara causal, la red
podría "filtrar" información futura durante el entrenamiento por padding o
por la propia estructura de la `HistoryDeque`, llevando a un sesgo de
**look-ahead** que invalidaría la política aprendida.

---

## 4. Codificación posicional

La atención por sí sola es **invariante a permutaciones**: el modelo no
distingue el orden de las posiciones. Para inyectar información posicional
se suma al embedding una **codificación posicional** $\mathrm{PE}(t) \in
\mathbb{R}^d$:

$$X' = X + \mathrm{PE}$$

Hay dos variantes:

### 4.1. Sinusoidal (Vaswani et al., 2017)

$$\mathrm{PE}_{(t, 2i)} = \sin\!\left(\frac{t}{10000^{2i/d}}\right)$$
$$\mathrm{PE}_{(t, 2i+1)} = \cos\!\left(\frac{t}{10000^{2i/d}}\right)$$

Determinista, no requiere parámetros entrenables, generaliza a longitudes no
vistas.

### 4.2. Aprendida

$$\mathrm{PE} = \mathrm{Embedding}(\{0, 1, \dots, h-1\})$$

Tabla de parámetros entrenables; mejor capacidad expresiva pero limitada a
la longitud máxima vista en entrenamiento.

`pes_trf` permite ambas; en la configuración por defecto se usa la versión
**aprendida** porque `TRF_HISTORY_LEN = 6` es fijo. La variante
sinusoidal se documenta a título teórico pero **no** está implementada
en el código: `transformer_model.build_q_network()` siempre instancia un
`tf.keras.layers.Embedding` aprendido («pos_embed»).

---

## 5. Bloque Encoder

Cada capa del encoder Transformer aplica:

```
x → LayerNorm → MultiHeadAttention → Dropout → x + residual
  → LayerNorm → FeedForward(d → ff_dim → d) → Dropout → x + residual
```

(Variante **Pre-LN**, más estable para entrenamiento que Post-LN).

El **feed-forward** aplica una MLP punto-a-punto:

$$\mathrm{FFN}(x) = \mathrm{ReLU}(x W_1 + b_1) W_2 + b_2$$

con $W_1 \in \mathbb{R}^{d \times d_\mathrm{ff}}$ y $W_2 \in
\mathbb{R}^{d_\mathrm{ff} \times d}$. En `pes_trf` óptimo:
$d = 32$, $d_\mathrm{ff} = 64$, $N_\mathrm{layers} = 2$.

---

## 6. Transformer como modelo de secuencia para RL

### 6.1. Formulación del Q-network

Sea $h_t = (s_{t-h+1}, \dots, s_t)$ la ventana de los últimos $h$ estados.
La Q-network parametrizada por el Transformer estima:

$$Q_\theta(h_t, a) \in \mathbb{R}^{|\mathcal{A}|}$$

mediante:

$$Q_\theta(h_t, \cdot) = W^Q \cdot \mathrm{Pool}(\mathrm{Encoder}(h_t W^E + \mathrm{PE}))$$

donde:

- $W^E \in \mathbb{R}^{3 \times d}$ proyecta el estado de 3 dimensiones al
  espacio del modelo.
- $\mathrm{Encoder}$ es la pila de $N$ bloques con atención causal.
- $\mathrm{Pool}$ es una agrupación temporal. En `pes_trf` se usa
  **last-token pooling**, no media global: tras el encoder se aplica
  `tf.keras.layers.Lambda(lambda t: t[:, -1, :])` para quedarse con el
  embedding del último paso temporal (el más reciente). El enmascarado
  causal garantiza que ese token resume toda la historia.
- $W^Q \in \mathbb{R}^{d \times 11}$ es la cabeza Q-lineal.

### 6.2. Objetivo Double DQN

El target Bellman con **Double DQN** (Hasselt et al., 2016) desacopla la
selección y la evaluación de la acción:

$$y_t = r_t + \gamma\, Q_{\bar\theta}\!\left(h_{t+1},\ \arg\max_a Q_\theta(h_{t+1}, a)\right)$$

donde:

- $\theta$ son los pesos del **modelo en línea**.
- $\bar\theta$ son los pesos del **target network** (sincronizados
  periódicamente desde $\theta$).

La pérdida de un minibatch es **Huber** (smooth L1) sobre el error TD:

$$\mathcal{L}(\theta) = \mathbb{E}\big[\mathcal{H}(y_t - Q_\theta(h_t, a_t))\big]$$

con $\mathcal{H}(\delta) = \frac{1}{2}\delta^2$ si $|\delta| \le 1$ y
$|\delta| - \frac{1}{2}$ en otro caso. La pérdida Huber es robusta a
outliers, frecuentes en RL por la varianza de los retornos.

### 6.3. Enmascaramiento de factibilidad

Para evitar que el agente proponga acciones $a > \mathrm{resources\_left}$:

$$\tilde Q(h_t, a) = \begin{cases} Q(h_t, a) & a \in \mathcal{F}(s_t) \\ -\infty & \text{en otro caso} \end{cases}$$

$$a_t = \arg\max_a \tilde Q(h_t, a)$$

Se aplica tanto en la **política $\varepsilon$-greedy** durante el
entrenamiento como en el cálculo de $\arg\max$ del target Double DQN.

---

## 7. Por qué los Transformers superan a los LSTM en este escenario

### 7.1. Limitaciones del LSTM

El LSTM (Hochreiter & Schmidhuber, 1997) procesa la secuencia de forma
**secuencial**: cada paso $t$ actualiza un estado oculto $h_t \in
\mathbb{R}^d$ a partir de $(h_{t-1}, x_t)$. Limitaciones:

1. **Cuello de botella**: toda la historia debe comprimirse en un único
   vector $h_t$ de tamaño fijo.
2. **Olvido gradual**: a pesar de las puertas, la información antigua se
   degrada exponencialmente.
3. **Dependencia secuencial**: imposible vectorizar el cómputo a lo largo
   del tiempo.
4. **Gradientes**: aunque mejores que la RNN simple, siguen siendo
   propensos a desvanecerse en horizontes largos.

### 7.2. Ventajas del Transformer causal

1. **Atención directa**: cada posición accede a cualquier otra posición
   anterior en una **única operación**, sin compresión intermedia.
2. **Capacidad uniforme**: la dependencia de la posición $i$ sobre la
   posición $j$ (con $j \le i$) tiene el mismo coste computacional que
   sobre $i-1$.
3. **Paralelismo**: el cómputo de la atención sobre toda la ventana se
   vectoriza en una sola pasada matricial $O(h^2 d)$, eficiente en GPU.
4. **Interpretabilidad**: las matrices de atención $\mathrm{softmax}(QK^\top/\sqrt{d_k})$
   son inspectables y revelan **qué trials previos influyen en la decisión actual**.

### 7.3. Evidencia empírica en `pes_trf`

| Aspecto | RDQN (LSTM) | TRF (Transformer) | Diferencia |
|---|---|---|---|
| Rendimiento medio | 0.91 | **0.927** | +1.7 pts |
| Desviación estándar | 0.05 | **0.045** | $-10\%$ |
| Episodios para converger | $\sim 80\,000$ | $\sim 50\,000$ | $-37\%$ |

---

## 8. Relación entre longitud de ventana y rendimiento

La longitud de la ventana `TRF_HISTORY_LEN` ($h$) es un trade-off:

| $h$ | Pros | Contras |
|---|---|---|
| Pequeño (2–4) | Rápido, poco memory footprint | Pierde patrones de largo plazo |
| **Medio (6–8)** | **Equilibrio óptimo en *Pandemic*** | — |
| Grande (>10) | Captura patrones largos | Atención difusa, sobreajuste, $O(h^2)$ memoria |

El estudio Optuna seleccionó $h = 6$ como óptimo, lo que coincide con la
**duración promedio de las decisiones críticas** en *Pandemic* (3–10 trials
por secuencia, con la media en torno a 6).

### Análisis teórico

Para un escenario con dependencias máximas de orden $k$, el rendimiento
satura cuando $h \ge k$. Aumentar $h$ más allá:

1. No aporta señal nueva.
2. Aumenta el coste $O(h^2)$ de la atención.
3. Diluye la atención softmax sobre más posiciones, lo que puede degradar
   la precisión.

Por eso el rendimiento como función de $h$ es **unimodal** con máximo
alrededor de la longitud característica del proceso de decisión.

---

## 9. Optimización Bayesiana del Transformer

`optimize_tr.py` usa **Optuna** (Akiba et al., 2019) con muestreador TPE
sobre el espacio:

```python
trial.suggest_int        ('history_len',      3, 10)
trial.suggest_categorical('d_model',          [16, 32, 64, 128])
trial.suggest_categorical('num_heads',        [2, 4, 8])
# (la restricción d_model %% num_heads == 0 NO está validada en código;
#  combinaciones inválidas pueden disparar errores en MultiHeadAttention)
trial.suggest_categorical("TRF_N_HEADS",     [1, 2, 4])
trial.suggest_categorical("TRF_N_LAYERS",    [1, 2, 3])
trial.suggest_categorical("TRF_FF_DIM",      [32, 64, 128])
trial.suggest_float      ("TRF_LEARNING_RATE", 1e-5, 1e-3, log=True)
trial.suggest_float      ("TRF_DISCOUNT",      0.90, 0.99)
```

### Restricciones de coherencia

- Debe cumplirse $d_\mathrm{model} \mod n_\mathrm{heads} = 0$ (cada cabeza
  necesita una dimensión entera $d_k = d_\mathrm{model}/n_\mathrm{heads}$).
  Optuna lo valida con `trial.set_user_attr("valid", ...)` y descarta
  combinaciones inválidas con `optuna.TrialPruned`.

### Pruning anticipado

`MedianPruner(n_startup_trials=5, n_warmup_steps=10)` aborta trials cuyo
rendimiento intermedio quede bajo la mediana de los previos. Reduce el coste
total de la búsqueda en $\sim 50\%$.

---

## 10. Conexión con Decision Transformer

Chen et al. (2021) propusieron el **Decision Transformer**, que reformula RL
como un problema de **modelado de secuencias supervisado**:

$$\tau = (\hat R_1, s_1, a_1,\ \hat R_2, s_2, a_2,\ \dots)$$

donde $\hat R_t$ es el *return-to-go*. Se entrena un Transformer
autoregresivo para predecir $a_t$ condicionado a la secuencia previa.

`pes_trf` **no** es un Decision Transformer estricto: usa el Transformer
como Q-network dentro de un esquema **Double DQN**, no como modelo
generativo de acciones. Sin embargo, la motivación es análoga:

> **El historial de decisiones es una secuencia, y los Transformers son la
> arquitectura óptima para procesar secuencias.**

Una extensión natural de `pes_trf` sería migrar a una formulación tipo
Decision Transformer condicionada al rendimiento objetivo.

---

## 11. Limitaciones teóricas

1. **Coste cuadrático**: $O(h^2 d)$ por capa. Para $h$ grande, otras
   alternativas (Performer, Linformer) reducirían este coste.
2. **Dependencia de la longitud máxima**: la codificación posicional
   aprendida no generaliza a $h$ no vistas en entrenamiento.
3. **Sobreajuste con pocos datos**: el Transformer es expresivo y puede
   sobreajustar el replay buffer si éste es pequeño. Solución en
   `pes_trf`: dropout + warmup de aprendizaje.
4. **Estabilidad**: como todo método off-policy con aproximadores no
   lineales, puede divergir si el target network se actualiza demasiado
   rápido o el LR es muy alto.

---

## 12. Referencias (APA 7)

- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A
  next-generation hyperparameter optimization framework. *Proceedings of the
  25th ACM SIGKDD International Conference on Knowledge Discovery & Data
  Mining*, 2623–2631.
- Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel,
  P., Srinivas, A., & Mordatch, I. (2021). Decision transformer:
  Reinforcement learning via sequence modeling. *Advances in Neural
  Information Processing Systems, 34*.
- Hasselt, H. van, Guez, A., & Silver, D. (2016). Deep reinforcement learning
  with double Q-learning. *Proceedings of the AAAI Conference on Artificial
  Intelligence, 30*(1), 2094–2100.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural
  Computation, 9*(8), 1735–1780.
- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare,
  M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G.,
  Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran,
  D., Wierstra, D., Legg, S., & Hassabis, D. (2015). Human-level control
  through deep reinforcement learning. *Nature, 518*(7540), 529–533.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An
  introduction* (2nd ed.). MIT Press.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,
  A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need.
  *Advances in Neural Information Processing Systems, 30*.
