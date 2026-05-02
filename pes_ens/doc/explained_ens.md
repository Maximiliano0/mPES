# pes_ens — Agente Ensamblado por Soft Voting

> Última actualización: 2026-05-02

## Resumen

`pes_ens` es la única variante del *workspace* que **no entrena ningún
modelo propio**. En su lugar, fusiona en tiempo de inferencia los
agentes ya entrenados de los paquetes hermanos:

| Miembro | Paquete | Rol (`role`) | Salida | `weight` (CONFIG) | `enabled` |
|---------|---------|--------------|--------|-------------------|-----------|
| `dqn`  | `pes_dqn`  | `q_dense`     | $Q(s,a)$, shape `(11,)`           | `0.18` | `True`  |
| `a2c`  | `pes_a2c`  | `actor`       | $\pi(a\mid s)$, shape `(11,)`     | `1.00` | `False` |
| `rdqn` | `pes_rdqn` | `q_recurrent` | $Q(s_{t-T:t},a)$, shape `(11,)`   | `0.90` | `True`  |
| `trf`  | `pes_trf`  | `q_recurrent` | $Q(s_{t-T:t},a)$, shape `(11,)`   | `5.00` | `True`  |

Los pesos absolutos se renormalizan internamente a $w_m^{\text{norm}}$
con $\sum_m w_m^{\text{norm}} = 1$. Por defecto `a2c` está
deshabilitado (su política era débil tras el entrenamiento) y el
ensamble vota efectivamente con tres miembros: DQN, RDQN y TRF, con
`trf` dominando el voto.

Cada miembro vota con una distribución de probabilidad sobre las 11
acciones discretas $\{0,1,\ldots,10\}$. La estrategia es **soft
voting** (promedio ponderado de distribuciones) en lugar de hard
voting porque:

- Con varios miembros, el hard voting genera demasiados empates.
- El promedio de distribuciones preserva información de incertidumbre
  que se descarta al colapsar a una acción única.
- Permite calcular una métrica de confianza basada en la entropía de
  la distribución resultante, igual que en los agentes individuales.

## Pipeline de decisión por trial

Sea $s_t = [\text{recursos\_left},\ \text{trial\_no},\
\text{severidad}]$ el estado en el paso $t$, normalizado a $[0,1]^3$
mediante `normalize_state` (los límites son
`AVAILABLE_RESOURCES_PER_SEQUENCE − 9`, `NUM_MAX_TRIALS` y
`MAX_SEVERITY`). Por cada miembro habilitado $m \in \mathcal{M}$:

### 1. Inferencia local

- `q_dense` (DQN) — entrada `(1, 3)`, salida $Q_m(s_t,\cdot)$.
- `q_recurrent` (RDQN, TRF) — entrada `(1, T, 3)` con la ventana
  deslizante actualizada (ver §"Memoria por episodio"), salida
  $Q_m(s_t,\cdot)$.
- `actor` (A2C) — entrada `(1, 3)`, salida $\pi_m(\cdot\mid s_t)$
  (ya es distribución; se hace `clip(0,·)` y se renormaliza por
  defensa).

### 2. Conversión a probabilidad

Para los dos roles tipo Q-network se aplica un softmax con
temperatura $\tau =$ `ENS_SOFTMAX_TEMPERATURE` $= 15.0$ (alta, para
suavizar):

$$
p_m(a \mid s_t) = \frac{\exp\!\big(Q_m(s_t,a)/\tau\big)}
                       {\sum_{a'} \exp\!\big(Q_m(s_t,a')/\tau\big)}
$$

El miembro `actor` ya es distribución, no requiere softmax.

### 3. Máscara de factibilidad **por miembro**

Sea `max_feasible = max(0, resources_left)`. Antes de votar, cada
distribución se trunca y se renormaliza dentro de su soporte
factible:

$$
p_m^{\text{feas}}(a) = \begin{cases}
\dfrac{p_m(a)}{\sum_{a'\le \text{max\_feasible}} p_m(a')}
  & a \le \text{max\_feasible} \\[6pt]
0 & a > \text{max\_feasible}
\end{cases}
$$

Si la suma es nula (todo el soporte cae en acciones inviables) se
fuerza el one-hot en $a=0$. Esto evita que un miembro malo arrastre
masa hacia acciones imposibles.

### 4. Voto ponderado por confianza

Por cada miembro se calcula la entropía normalizada de su
distribución factible:

$$
H_m^{\text{norm}} = \frac{-\sum_a p_m^{\text{feas}}(a)\,
                          \log_2 p_m^{\text{feas}}(a)}{\log_2 11}
\;\in\;[0,1]
$$

y un peso dinámico:

$$
w_m^{\text{dyn}} = w_m^{\text{norm}} \cdot
                   \big(0.1 + (1 - H_m^{\text{norm}})\big)
$$

El término constante `0.1` impide que un miembro perfectamente
incierto (entropía máxima) se anule por completo. La distribución
agregada inicial es:

$$
\bar p(a) = \sum_{m\in\mathcal{M}} w_m^{\text{dyn}}\,
            p_m^{\text{feas}}(a)
$$

### 5. Penalización de la acción "no asignar"

Si todavía hay presupuesto (`max_feasible > 0`), la acción $a=0$ se
penaliza:

$$
\bar p(0) \leftarrow 0.3 \cdot \bar p(0)
$$

Motivo: en el log de respuestas la columna `Confidence` se marca
como `-1` cuando la respuesta es `0`, lo que efectivamente desperdicia
el trial. La penalización empuja al ensamble a "gastar" cuando puede.

A continuación se renormaliza $\bar p$ a una distribución de
probabilidad. Si la masa total cae a cero (caso patológico) se
colapsa a un one-hot en $a=0$.

### 6. Mezcla con prior gaussiano de severidad

Se reconstruye la severidad cruda desde la coordenada normalizada,
$\text{sev} = s_t[2]\cdot\text{MAX\_SEVERITY}$, y se construye un
prior gaussiano sobre las 11 acciones:

$$
\text{prior}(a) = \exp\!\left(-\frac{(a - \text{sev})^2}
                                    {2\sigma^2}\right),
\qquad \sigma = \texttt{ENS\_SEVERITY\_PRIOR\_SIGMA} = 3.0
$$

con la misma máscara de factibilidad y renormalizado a probabilidad.
La distribución final mezcla ensamble y prior:

$$
\bar p_{\text{final}}(a) = (1 - w)\,\bar p(a) + w\,\text{prior}(a),
\qquad w = \texttt{ENS\_SEVERITY\_PRIOR\_WEIGHT} = 0.17
$$

Este paso recupera el conocimiento de dominio "asigna ≈ severidad"
cuando los miembros están inseguros. Si `w = 0` la mezcla queda
desactivada.

### 7. Regla de seguridad: piso por severidad

Para evitar los outliers catastróficos del tipo "severidad 8, voto 0":
si $\text{sev} \ge 6$, se calcula
$\text{floor} = \lfloor \text{sev}/2 \rfloor$ y, siempre que
`max_feasible ≥ floor` y $\arg\max \bar p_{\text{final}} <
\text{floor}$, se sobreescribe la distribución con un one-hot en
`floor`. Esta regla elimina los mínimos de $\sim 0.75$ que
afectaban a los primeros bloques.

### 8. Selección final

$$
a^* = \arg\max_a \bar p_{\text{final}}(a)
$$

Posteriormente, el `pygameMediator` recorta el resultado al rango
factible y deriva la confianza meta-cognitiva (ver §"Confianza
meta-cognitiva").

## Memoria por episodio

Los miembros recurrentes (`rdqn`, `trf`) requieren una ventana
deslizante de los últimos $T$ estados normalizados (con
$T =$ `history_len` $= 6$). `pes_ens` mantiene **una `HistoryDeque`
independiente por miembro y por episodio** $(\text{block},
\text{sequence})$, almacenadas en un caché interno
`_history_caches[member_name][(session_no, sequence_no)]`. La función
`EnsembleAgent.reset_episode(session_no, sequence_no)` se llama en el
primer trial de cada secuencia para limpiar la ventana y evitar fugas
de estado entre episodios.

## Confianza meta-cognitiva

La confianza se deriva de la entropía de la distribución factible
final:

$$
H = -\sum_a \bar p_{\text{final}}(a)\,
            \log_2 \bar p_{\text{final}}(a)
$$

$$
\text{conf} = \frac{H_{\min} - H}{H_{\min} - H_{\max}} \in [0,1]
$$

donde $H_{\min}$ y $H_{\max}$ son las entropías de una delta y de la
uniforme sobre 11 valores, respectivamente. Es la misma fórmula
empleada en los paquetes individuales, lo que mantiene la
comparabilidad de la columna `Confidence` en los logs.

## Configuración

`config/CONFIG.py` expone los hiperparámetros del ensamble:

```python
ENS_MEMBER_MODELS = [
    {'name': 'dqn',  'role': 'q_dense',
     'path': '../pes_dqn/inputs/dqn_model.keras',
     'history_len': 1, 'weight': 0.18, 'enabled': True},
    {'name': 'a2c',  'role': 'actor',
     'path': '../pes_a2c/inputs/ac_actor.keras',
     'history_len': 1, 'weight': 1.0,  'enabled': False},
    {'name': 'rdqn', 'role': 'q_recurrent',
     'path': '../pes_rdqn/inputs/rdqn_model.keras',
     'history_len': 6, 'weight': 0.9,  'enabled': True},
    {'name': 'trf',  'role': 'q_recurrent',
     'path': '../pes_trf/inputs/trf_model.keras',
     'history_len': 6, 'weight': 5.0,  'enabled': True},
]

ENS_SOFTMAX_TEMPERATURE   = 15.0
ENS_SEVERITY_PRIOR_WEIGHT = 0.17   # 0 desactiva la mezcla con prior
ENS_SEVERITY_PRIOR_SIGMA  = 3.0    # ancho del prior gaussiano
```

- Los `path` son relativos a la raíz del paquete `pes_ens` y apuntan
  a los `inputs/` canónicos de los paquetes hermanos (no se duplican
  artefactos `.keras`).
- Se puede deshabilitar un miembro poniendo `enabled: False` para,
  por ejemplo, comparar voting de 3 vs 4 miembros sin tocar el resto.
- Los pesos crudos no necesitan sumar 1; el constructor de
  `EnsembleAgent` los normaliza a $w_m^{\text{norm}}$.

## Notas técnicas

- **Sin imports cruzados entre paquetes.** Cada miembro es un
  artefacto autónomo `.keras`; `pes_ens` lo lee directamente del
  sistema de archivos vía `tf.keras.models.load_model`. Los
  *helpers* locales necesarios (`normalize_state`, `HistoryDeque`)
  están reimplementados *verbatim* en `ext/ensemble_model.py` para
  ser bit-equivalentes a los de los paquetes hermanos.
- **`safe_mode=False`** en `tf.keras.models.load_model` es
  obligatorio para `pes_trf/inputs/trf_model.keras` (capa `Lambda`,
  CWE-502); inocuo para los otros tres miembros. Los artefactos son
  generados por nuestra propia *pipeline*, por lo que la
  deserialización es de confianza.
- **Sin `train_ens.py` ni `optimize_ens.py`**: el ensamble se rige
  íntegramente por los pesos `weight` y por los hiperparámetros del
  prior. Si en el futuro se desean pesos optimizados (p. ej. mediante
  búsqueda Bayesiana sobre `mean_perf`), basta con añadir un script
  en `ext/` que ajuste `ENS_MEMBER_MODELS[i]['weight']` y vuelva a
  ejecutar `python -m pes_ens`.
