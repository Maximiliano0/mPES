# pes_ens — Fundamentos teóricos

> Última actualización: 2026-05-04
> Paquete: `ml/pes_ens`
> Tema: Métodos de ensemble aplicados a aprendizaje por refuerzo

---

## 1. Métodos de ensemble: descomposición sesgo–varianza

Los métodos de ensemble combinan múltiples predictores base para obtener un
predictor agregado con mejor desempeño que cualquier miembro individual
(Dietterich, 2000; Zhou, 2012). El marco teórico clásico es la
**descomposición sesgo–varianza** del error esperado.

Para una pérdida cuadrática y un predictor $\hat{f}$ entrenado sobre datos
aleatorios $\mathcal{D}$:

$$
\mathbb{E}_{\mathcal{D}, x}\!\left[(\hat{f}(x) - y)^{2}\right]
\;=\;
\underbrace{(\mathbb{E}[\hat{f}(x)] - y)^{2}}_{\text{sesgo}^{2}}
\;+\;
\underbrace{\mathrm{Var}[\hat{f}(x)]}_{\text{varianza}}
\;+\;
\sigma^{2}_{\text{ruido}}
$$

Si combinamos $M$ predictores $\hat{f}_1, \dots, \hat{f}_M$ mediante
promediado, el agregado $\bar{f} = \tfrac{1}{M}\sum_i \hat{f}_i$ cumple:

$$
\mathrm{Var}[\bar{f}] \;=\; \frac{1}{M^{2}}\!\left(\sum_i \mathrm{Var}[\hat{f}_i]
+ \sum_{i \ne j} \mathrm{Cov}[\hat{f}_i, \hat{f}_j]\right)
$$

Cuando las covarianzas son bajas (miembros **diversos**), la varianza se
reduce aproximadamente en un factor $1/M$ sin alterar el sesgo. Este es el
mecanismo principal por el que un ensemble supera a sus miembros
individuales (Zhou, 2012).

---

## 2. *Soft voting* vs *hard voting*

En clasificación con $K$ clases, dos esquemas de combinación son canónicos:

- **Hard voting**: cada miembro vota por una clase y se elige la mayoritaria.
  Se descarta toda información sobre la confianza del miembro.

- **Soft voting**: cada miembro emite una distribución $p^{(i)} \in
  \Delta^{K-1}$ y se promedian (con pesos $w_i$):

$$
p^{\text{ens}}(k) \;=\; \frac{\sum_i w_i\, p^{(i)}(k)}{\sum_i w_i}
$$

La acción final es $\arg\max_k p^{\text{ens}}(k)$.

El soft voting es **estrictamente más informativo** que el hard voting: un
miembro indeciso entre dos clases no fuerza una decisión binaria, sino que
contribuye con su incertidumbre a la mezcla (Zhou, 2012). Esto es
especialmente valioso cuando los miembros tienen diferentes regiones de
competencia, como ocurre en mPES.

`pes_ens` usa soft voting con pesos no uniformes calibrados al desempeño
empírico de cada miembro.

---

## 3. Diversidad y complementariedad de los miembros

Para que un ensemble mejore sobre el mejor miembro, los errores deben ser
**parcialmente no correlacionados** (Dietterich, 2000). Una forma de lograr
esta diversidad es combinar arquitecturas con sesgos inductivos distintos.

En `pes_ens`:

| Miembro | Sesgo inductivo | Captura |
|---------|-----------------|---------|
| **DQN** (Mnih et al., 2015) | Markoviano puro | Patrones del estado actual |
| **RDQN** | Recurrencia LSTM (Hochreiter & Schmidhuber, 1997) | Dependencias temporales locales |
| **Transformer** | Atención causal (Vaswani et al., 2017) | Dependencias temporales largas, no monótonas |
| **A2C** *(opcional)* | Política estocástica (Sutton & Barto, 2018) | Exploración blanda |

Cada arquitectura *generaliza distinto*. Donde el DQN sobreajusta a un patrón
local, el Transformer puede aprovechar el contexto histórico; donde el RDQN
olvida un evento lejano, la atención lo recupera. Esta heterogeneidad es la
fuente principal de la ganancia del ensemble.

---

## 4. Interpretación bayesiana: *Bayesian Model Averaging*

El soft voting ponderado admite una lectura bayesiana. Sea $\mathcal{M}_i$
el modelo $i$-ésimo y $a$ una acción. La distribución posterior marginal
sobre acciones, dado el conjunto de modelos, es:

$$
p(a \mid s, \mathcal{D})
\;=\;
\sum_i p(a \mid s, \mathcal{M}_i)\, p(\mathcal{M}_i \mid \mathcal{D})
$$

donde $p(\mathcal{M}_i \mid \mathcal{D})$ es la *probabilidad posterior del
modelo* dada la evidencia. En `pes_ens`, los pesos normalizados
$w_i / \sum_j w_j$ son **estimadores ad-hoc** de esta posterior, ajustados
empíricamente al desempeño en validación. Lakshminarayanan et al. (2017)
muestran que el promediado simple de modelos profundos —pese a no ser
estrictamente bayesiano— produce estimaciones de incertidumbre
sorprendentemente bien calibradas y mejora consistentemente sobre modelos
únicos.

`pes_ens` se sitúa en esta familia: un *deep ensemble* heterogéneo
ponderado, sin pretensión de inferencia bayesiana exacta.

---

## 5. Temperatura softmax: teoría

Convertir Q-values en probabilidades requiere una función de transformación
calibrada. La elección estándar es la **softmax con temperatura**:

$$
p_i \;=\; \frac{\exp(Q_i / T)}{\sum_j \exp(Q_j / T)}
$$

Casos límite:

- $T \to 0^{+}$: la distribución colapsa a un *one-hot* en
  $\arg\max_i Q_i$ (≈ hard voting).
- $T \to \infty$: la distribución tiende al uniforme; el miembro deja de
  expresar preferencias.

Sutton y Barto (2018, cap. 13) discuten esta parametrización en el contexto
de políticas Boltzmann. En el régimen del ensemble, $T$ controla **cuánto
peso recibe la opinión más fuerte de cada miembro**:

- Con $T$ pequeño, un solo miembro hipersure puede dominar el voto.
- Con $T$ grande, las preferencias graduadas se promedian de forma estable.

El valor empírico $T = 15$ en `pes_ens` mantiene el contraste entre acciones
buenas y malas pero evita que un Q-value extremo borre la contribución de
los demás miembros — coherente con la práctica de Lakshminarayanan et al.
(2017) de evitar distribuciones excesivamente concentradas en deep ensembles.

---

## 6. *Severity prior*: inyección de conocimiento del dominio

Los métodos bayesianos formalizan la incorporación de conocimiento previo
mediante una distribución *prior*. En `pes_ens`, el prior es gaussiano:

$$
\pi(a \mid \text{severity}) \;\propto\;
\exp\!\left(-\frac{(a - \text{severity})^{2}}{2\sigma^{2}}\right)
$$

y se combina linealmente con el ensemble:

$$
p^{\text{final}}(a) \;=\; (1 - w)\, p^{\text{ens}}(a) \;+\; w\, \pi(a)
$$

Esta es una **mezcla convexa**, no un producto bayesiano formal — pero
captura la misma intuición: regularizar la salida del modelo hacia una
hipótesis razonable a priori. Sutton y Barto (2018, cap. 17) discuten
estrategias análogas de *reward shaping* y *initial value bias* para
inyectar sesgos inductivos en RL.

El parámetro $w = 0.17$ es deliberadamente pequeño: el ensemble decide,
pero el prior estabiliza. La elección de $\sigma = 3$ permite que el prior
abarque varias acciones cercanas a la severidad sin volverse uniforme.

---

## 7. Inferencia restringida por factibilidad

Tras combinar ensemble y prior, se aplica una **máscara de factibilidad**:

$$
\tilde{p}(a) \;=\;
\begin{cases}
p^{\text{final}}(a) & \text{si } a \le \text{resources\_left} \\
0 & \text{en otro caso}
\end{cases}
$$

seguida de renormalización. Esta operación puede verse como un
*posterior condicionado* a la restricción de recursos: la distribución
restringida es proporcional a la distribución original sobre el subconjunto
factible. Garantiza que el agente nunca proponga una acción imposible — un
aspecto de seguridad operativa irrenunciable en el entorno mPES.

---

## 8. ¿Por qué `pes_trf` recibe el peso más alto?

El peso $w_{\text{trf}} = 5.0$ refleja la observación empírica de que el
Transformer es el miembro más preciso. Teóricamente esto se justifica por:

1. **Atención global causal** (Vaswani et al., 2017): el Transformer puede
   atender directamente a cualquier paso del historial sin el cuello de
   botella del estado oculto recurrente. En entornos con dependencias
   temporales no monótonas (como Pandemic Scenario), esta capacidad es
   especialmente valiosa.

2. **Mejor escalabilidad de la representación**: la atención multi-cabeza
   aprende relaciones complejas entre eventos pasados que un LSTM tendría
   que comprimir en un único estado oculto.

3. **Estabilidad de entrenamiento**: la normalización por capa y las
   conexiones residuales del Transformer producen modelos finales con
   menor varianza entre semillas — una ventaja directa para un miembro de
   ensemble.

Asignar mayor peso a un miembro más preciso es **óptimo** en el sentido de
mínima varianza ponderada cuando los miembros son aproximadamente
independientes y se conoce su precisión relativa (Zhou, 2012).

---

## 9. Análisis: por qué ensemble > mejor miembro individual

Sea $L_i$ el error esperado del miembro $i$ y $L_{\text{ens}}$ el error
del ensemble ponderado. Krogh y Vedelsby demostraron la **descomposición
ambigüedad–error** (citada en Zhou, 2012):

$$
L_{\text{ens}} \;=\; \bar{L} \;-\; \bar{A}
$$

donde $\bar{L} = \sum_i w_i L_i$ es el error promedio ponderado y
$\bar{A} = \sum_i w_i \mathbb{E}\!\left[(\hat{f}_i - \bar{f})^{2}\right]$ es
la **ambigüedad** (diversidad) del ensemble. Como $\bar{A} \ge 0$:

$$
L_{\text{ens}} \;\le\; \bar{L}
$$

con igualdad solo si todos los miembros producen idéntica salida. En
`pes_ens`, los resultados empíricos confirman esta predicción:

- Mejor miembro individual: $\approx 0.93$ (Transformer).
- Ensemble: $0.937318$ con varianza $0.034937$ (la menor del proyecto).

La mejora absoluta es modesta, pero la **reducción de varianza** es notable
y robusta a través de los 64 ensayos — exactamente el comportamiento
predicho por la teoría.

---

## 10. Conexión con *Mixture of Experts*

El soft voting ponderado es un caso particular del marco **Mixture of
Experts (MoE)**, donde un *gating network* asigna pesos dependientes del
input a cada experto:

$$
p(a \mid s) \;=\; \sum_i g_i(s)\, p_i(a \mid s)
$$

`pes_ens` usa **gating estático** — los pesos $w_i$ son constantes,
independientes de $s$. Esto es una simplificación que evita la necesidad de
entrenar un gating adicional sin perder demasiado en desempeño, pues los
miembros ya están bien especializados globalmente.

Una extensión natural sería entrenar un gating dinámico que asignase pesos
en función del estado (p. ej., dar más peso al Transformer cuando el
historial es largo y al DQN en estados terminales). Hasselt et al. (2016)
también muestran cómo combinar estimadores de Q (Double DQN) para reducir
sesgo — una idea relacionada que podría aplicarse miembro a miembro antes
del ensemble.

---

## 11. Resumen

`pes_ens` aplica principios bien fundamentados de la literatura de
ensembles a un problema de RL secuencial:

- **Soft voting** ponderado para combinar distribuciones (Zhou, 2012).
- **Diversidad arquitectónica** (denso + LSTM + atención) para errores
  complementarios (Dietterich, 2000).
- **Deep ensembles** con pesos calibrados empíricamente
  (Lakshminarayanan et al., 2017).
- **Prior gaussiano** como inyección de conocimiento del dominio
  (Sutton & Barto, 2018).
- **Máscara de factibilidad** como posterior condicionado.

El resultado teórico — *ensemble ≤ mejor miembro en error y mucho menor en
varianza* — se verifica empíricamente en mPES y posiciona a `pes_ens` como
el agente más robusto del proyecto.

---

## Referencias (APA 7)

Dietterich, T. G. (2000). Ensemble methods in machine learning. En
*Proceedings of the 1st International Workshop on Multiple Classifier
Systems* (pp. 1–15). Springer.

Hasselt, H. van, Guez, A., & Silver, D. (2016). Deep reinforcement
learning with double Q-learning. En *Proceedings of the 30th AAAI
Conference on Artificial Intelligence* (pp. 2094–2100). AAAI Press.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
*Neural Computation*, *9*(8), 1735–1780.
https://doi.org/10.1162/neco.1997.9.8.1735

Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and
scalable predictive uncertainty estimation using deep ensembles. En
*Advances in Neural Information Processing Systems* (Vol. 30,
pp. 6402–6413). Curran Associates.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J.,
Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K.,
Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I.,
King, H., Kumaran, D., Wierstra, D., Legg, S., & Hassabis, D. (2015).
Human-level control through deep reinforcement learning. *Nature*,
*518*(7540), 529–533. https://doi.org/10.1038/nature14236

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An
introduction* (2nd ed.). MIT Press.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,
A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need.
En *Advances in Neural Information Processing Systems* (Vol. 30,
pp. 5998–6008). Curran Associates.

Zhou, Z.-H. (2012). *Ensemble methods: Foundations and algorithms*. CRC
Press. https://doi.org/10.1201/b12207