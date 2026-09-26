# mPES — Síntesis de métricas, salidas y resultados (contexto para la tesis)

> Generado el 2026-09-24 a partir de `h1/general/results/` (matrices, `comparison_metrics.json`, `cells/`), los `inputs/best_params.json` y `config/CONFIG.py` de cada paquete. Es la **fuente de verdad numérica** para redactar la tesis. Los números usan punto decimal; en LaTeX se escriben con coma (`$0{,}927$`). El archivo `mpes_resultados.json` contiene los mismos datos con más precisión y por celda.

## 0. Resumen ejecutivo (hallazgos verificados)

1. **Transformer (`pes_trf`) = mejor modelo individual.** Referencia 0.927 (σ 0.045, la menor entre individuales), generalización 0.930 (resto de optimizados entre 0.871 y 0.899); mayor media en 16 de 22 escenarios. Reduce la distancia al óptimo de Q-Learning en 36 % en la referencia. d = 0.78 vs Q-Learning, 0.57 vs DQN recurrente, 0.51 vs DQN. → Respalda **H1**.
2. **No es el que menos pierde en su peor caso**: peor escenario `len_extrapolate_long` 0.860 (degradación 0.068) frente a 0.053 de DQN y 0.062 de A2C.
3. **Tabulares colapsan con severidad fuera de rango** (`sev_extrapolate_high`): Q-Learning base 0.627 (degradación 0.244, peor celda del estudio), Q-Learning 0.762, Double Q-Learning 0.785: la tabla recorta S a 9, fila nunca entrenada.
4. **Las redes mantienen el desempeño en ese escenario**: Transformer 0.996 (y 0.997 en la conjunta), A2C 0.929, DQN 0.890; excepción: DQN recurrente 0.831 (su peor escenario). Advertencia: con S = 10..12 asignar mucho acerca al óptimo.
5. **Secuencias largas (`len_extrapolate_long`) = peor escenario de DQN, A2C, Transformer y de 4 ensambles.** Allí el mejor individual es el DQN recurrente (0.889 vs 0.860 del Transformer): único escenario fuera de rango en que el Transformer no es primero.
6. **Escenarios estructurales** reproducen exactamente la referencia en los 13 modelos (control correcto).
7. **Ensamble ponderado (`pes_ens`) = único sistema que supera al Transformer**: referencia 0.937 (σ 0.035, la menor de todo el estudio), generalización 0.939, peor escenario 0.900 (+0.040 sobre el del Transformer), mayor degradación 0.037 (la menor). Δrel vs Q-Learning 45 %. Óptimo en todas las secuencias de `sev_extrapolate_high` y `joint_extrap_both` (1,000; σ 0). d = 0.58 vs consenso y 0.17 vs compuerta.
8. **Pero su acción final difiere de la del Transformer en el 61,2 % (referencia) / 48,1 % (generalización)** de las decisiones; el prior de severidad cambia la acción votada en el 40,9 % / 32,5 %; la cota casi nunca actúa. Con τ = 15 las distribuciones de los miembros son casi planas. La mejora es compatible con prior + temperatura, no con la ponderación por confianza.
9. **Compuerta del Transformer ≈ Transformer**: generalización 0.931 vs 0.930; sigue al Transformer en el 96,9 % / 95,0 % de las decisiones.
10. **Voto suave y voto por acción**: generalización más baja (0.902 / 0.901), peor escenario `sev_extrapolate_high` (0.860 / 0.859), con 41 % / 42 % de secuencias bajo 0,8. Sus pesos dan más peso al DQN recurrente que al Transformer.
11. **Consenso** tiene la referencia más baja de los ensambles (0.889); la variante con prior mejora referencia (0.917) y peor escenario (0.826 → 0.871), aunque su prior sólo cambia el 2,4 % / 3,6 % de las decisiones.
12. **Agente aleatorio**: 0.670 (cota inferior de referencia).

## 1. Definiciones

- **Desempeño normalizado** por secuencia: $\bar r = (S_{peor} - S_{agente}) / (S_{peor} - S_{mejor})$. $0$ = no asignar recursos; $1$ = asignación óptima ($S_{mejor}$ exacto por programación dinámica, mochila acotada con asignaciones 0..10 por ciudad que suman ≤ 30). $\bar r \le 1$. No es una probabilidad.
- **Referencia** = escenario `sev_base` (distribución de entrenamiento), $n = 64$ secuencias (8 bloques × 8), 360 pasos.
- **Generalización** = media de las medias de los **21** escenarios distintos de la referencia (incluye 3 estructurales que reproducen exactamente la referencia y escenarios más fáciles, p. ej. secuencias cortas).
- **Degradación** de una celda = $\bar r_{ref} - \bar r_{esc}$ (positiva = pérdida; negativa = mejora). **Degradación media** = promedio con signo sobre los 21 escenarios. **Mayor degradación** = máximo de esas 21.
- **Por celda frente a la propia referencia**: $d$ de Cohen (positivo = mejor en el escenario), $\log_{10} p$ de Welch (bilateral) y KL de acciones (desplazamiento de la política sobre las 11 acciones).
- **Entre pares de modelos** (mismo grupo): se juntan las secuencias de los 21 escenarios de generalización (sin `sev_base`); $d > 0$ ⇒ gana el modelo de la fila; KL simetrizada entre histogramas de desempeño (20 bins en [0, 1], $\varepsilon = 10^{-9}$).
- **Δrel vs Q-Learning** (Ec. `eq:severity-reduction`): $(\bar r_m - \bar r_{ql}) / (1 - \bar r_{ql})$, reducción relativa de la distancia al óptimo.
- Advertencia estadística: todas las celdas usan las mismas secuencias (semilla 42), no son independientes; no hay corrección por comparaciones múltiples. Los $p$ son exploratorios; la medida principal es $d$.

## 2. Entorno y protocolo

- **estado**: (R, t, S): R en 0..30 recursos restantes, t en 0..10 paso, S en 0..9 severidad de la ciudad actual; 31x11x10 = 3 410 estados (observación parcial).
- **acciones**: 0..10 unidades; una asignación mayor que R se recorta.
- **presupuesto**: 39 recursos por secuencia, 9 preasignados por el entorno -> el agente administra 30.
- **transicion**: S_{i,t+1} = max(0, 1,4·S_{i,t} - 0,4·a_i) para toda ciudad ya incorporada (alpha = 0,4, beta = 1,4); la asignación de cada ciudad se fija al llegar.
- **recompensa**: r_t = -sum_{i<=t} S_{i,t+1}.
- **estructura**: 8 bloques x 8 secuencias, longitudes 3..10, 360 pasos; archivos initial_severity.csv y sequence_lengths.csv idénticos en todos los paquetes.
- **entrenamiento**: cada episodio sortea longitud y severidades de sus distribuciones empíricas en esos archivos: sólo severidades 2..8 (filas S = 0, 1, 9 de las tablas Q quedan aleatorias).
- **entrada_redes**: s = [R/30, t/10, S/9]; A2C y pes_ens recortan S a 9; DQN, DQN recurrente, Transformer y los otros cinco ensambles no recortan (S/9 > 1 en sev_extrapolate_high).
- **evaluacion**: determinista (epsilon = 0, acción factible de mayor valor); sólo inferencia, sin reentrenar.
- **software**: Python 3.12, TensorFlow 2.21, Keras 3.13, NumPy 2.4, Optuna 4.7, Gymnasium 1.2, SciPy 1.17.
- **Escenarios**: generados con semilla 42; dentro de cada escenario todos los modelos enfrentan las mismas secuencias. Los modelos individuales y los ensambles se evalúan como dos grupos (Q-Learning base se agrega como referencia al grupo individual).

## 3. Agente aleatorio (cota inferior de referencia)

- Agente que en cada paso elige a ~ U{0..10}, recortada al presupuesto restante (30), sobre las 64 secuencias de referencia; semilla 42. Verificado con general.scripts.random_baseline.run_random_player.
- $\bar r$: media **0.670**, σ 0.186, rango [0.164; 0.950]; $S_{cruda}$ media 48.73 (σ 51.86). Figura `fig:baseline-random`.

## 4. Modelos individuales

### 4.1 Paquetes, nombres y configuración

| Paquete | Nombre en el texto | Tipo | Arquitectura / tabla | Parámetros | Episodios | Semilla | Origen de hiperparámetros |
|---|---|---|---|---:|---:|---:|---|
| `pes_base` | Q-Learning base | Q-Learning tabular | 31x11x10x11 = 37 510 valores, init U(-1,1) | 37 510 (tabla) | 1000000 | 42 | valores originales de PES (sin optimizar) |
| `pes_ql` | Q-Learning | Q-Learning tabular optimizado | tabla Q 31x11x10x11 = 37 510 valores, init U(-1,1) | 37 510 (tabla) | 550000 | 106 | Optuna/TPE, 100 ensayos, mejor #63 |
| `pes_dql` | Double Q-Learning | Double Q-Learning tabular + PBRS + calentamiento de epsilon | tabla Q 31x11x10x11 = 37 510 valores, init U(-1,1) | 37 510 (tabla) | 360000 | 46 | Optuna/TPE, 100 ensayos (86 podados) |
| `pes_dqn` | DQN | Double DQN, red densa | Input(3) -> Dense(64,ReLU) -> Dense(64,ReLU) -> Dense(11) | 5131 | 40000 | 84 | Optuna/TPE, 47 ensayos, mejor #41 (arquitectura incluida) |
| `pes_rdqn` | DQN recurrente | Double DQN con codificador LSTM (esquema DRQN) | ventana W=6 estados -> LSTM(64) -> Dense(96,ReLU) -> Dense(96,ReLU) -> Dense(11) | 34027 | 30000 | 57 | BO sólo de hiperparámetros de entrenamiento; arquitectura ad hoc |
| `pes_a2c` | A2C | Advantage Actor-Critic (redes separadas) | actor Input(3)->Dense(128,ReLU)->Dense(11,softmax); crítico Input(3)->Dense(128,ReLU)->Dense(1) | — | 125000 | 132 | Optuna/TPE, 100 ensayos, mejor #89 |
| `pes_trf` | Transformer | Double DQN con codificador Transformer causal (Pre-LN) | ventana W=6 -> proyección d_model=32 + vector de posición fijo (Glorot, no entrenado) -> 2 bloques [atención causal 4 cabezas key_dim 16; FFN 64; residual; LayerNorm previa] -> última posición -> Dense(32,ReLU) -> Dense(11); dropout 0 | 27019 | 30000 | 45 | BO sólo de hiperparámetros de entrenamiento; arquitectura ad hoc |

**Tabulares** (Tabla `tab:tabular-hparams`): Q-Learning base α 0,2 · γ 0,9 · ε 0,8→0 lineal · 1 000 000 episodios. Q-Learning α 0,2862 · γ 0,8587 · ε 0,6813→0,0437 lineal · 550 000 episodios. Double Q-Learning α 0,1132 · γ 0,9777 · ε 0,5998→0,0327 exponencial con calentamiento w 0,0260 y fracción q 0,6430 · PBRS κ 0,000392 (Φ(s) = −Σ S_i) · 360 000 episodios. Semilla de entrenamiento = 42 + i + 1 (i = índice del mejor ensayo).

**Tipo DQN** (Tabla `tab:deep-hparams`; blanco Double DQN, enmascarado de acciones no factibles, pérdida de Huber, Adam, Glorot uniforme, recorte de gradiente, ε-greedy con calentamiento y decaimiento exponencial):

| Hiperparámetro | DQN | DQN recurrente | Transformer |
|---|---:|---:|---:|
| Entrada | estado actual | ventana de 6 estados | ventana de 6 estados |
| Parámetros entrenables | 5 131 | 34 027 | 27 019 |
| Tasa de aprendizaje | 0,001508 | 0,002415 | 0,000215 |
| γ | 0,9634 | 0,9719 | 0,9234 |
| ε0 / εmin | 0,9627 / 0,0691 | 0,8883 / 0,0887 | 0,8651 / 0,0838 |
| w / q | 0,2779 / 0,6290 | 0,1237 / 0,5990 | 0,2428 / 0,5333 |
| Lote | 128 | 64 | 128 |
| Buffer | 20 000 | 60 000 | 30 000 |
| Sincronización C (pasos) | 1 000 | 2 000 | 500 |
| Recorte del gradiente | 3,953 | 3,475 | 4,170 |
| PBRS κ | 0,0226 | 0,00263 | 0 (sin PBRS) |
| Inicio del aprendizaje f (transiciones) | 0,1615 (3 230) | 0,2021 (12 128) | 0,1217 (3 650) |
| Episodios | 40 000 | 30 000 | 30 000 |
| Semilla | 84 | 57 | 45 |

- DQN: `Input(3) → Dense(64, ReLU) → Dense(64, ReLU) → Dense(11)`, replay + red objetivo.
- DQN recurrente: ventana W = 6 (relleno con ceros al inicio), LSTM(64), último estado oculto → Dense(96) → Dense(96) → 11 Q; el estado oculto no se conserva entre decisiones.
- Transformer: W = 6, proyección a d_model = 32 + vector de posición fijo (Glorot, no entrenado), 2 bloques Pre-LN (atención causal 4 cabezas de dim. 16 + FFN 64, residual), sin dropout, última posición → Dense(32, ReLU) → 11 Q.
- **RDQN y TRF**: arquitectura elegida por exploración *ad hoc* (optimizarla con BO era demasiado costoso); sólo sus hiperparámetros de entrenamiento vienen de la BO. No citar nº de ensayos, score de Optuna ni los campos de arquitectura de sus `best_params.json`.
- **A2C** (Tabla `tab:a2c-hparams`): actor `Input(3)→Dense(128)→Dense(11, softmax)`, crítico `Input(3)→Dense(128)→Dense(1)`; lr actor 0,000648 / crítico 0,004299, decaimiento coseno hasta 23,75 %, γ 0,8545, β_H 0,00528, GAE λ 0,9135, recorte 1,200, PBRS κ 0,1527, penalización de gasto r ← r − 0,01039·a, sesgo inicial del logit de a = 10: −1,389, 125 000 episodios, semilla 132.
- Q-Learning, Double Q-Learning, DQN y A2C reproducen el score de su mejor ensayo de Optuna (0,8866 / 0,8963 / 0,8937 / 0,8872).

### 4.2 Resumen de desempeño (ordenado por generalización)

| Paquete | Nombre en el texto | Ref. | σ ref. | Gen. (21) | Deg. media | Peor escenario (media) | Mayor deg. | Δrel vs QL ref. |
|---|---|---:|---:|---:|---:|---|---:|---:|
| `pes_trf` | Transformer | 0.927 | 0.045 | 0.930 | -0.0025 | `len_extrapolate_long` (0.860) | 0.068 | 35.8 % |
| `pes_dqn` | DQN | 0.894 | 0.055 | 0.899 | -0.0050 | `len_extrapolate_long` (0.841) | 0.053 | 6.3 % |
| `pes_a2c` | A2C | 0.887 | 0.063 | 0.896 | -0.0090 | `len_extrapolate_long` (0.826) | 0.062 | 0.5 % |
| `pes_rdqn` | DQN recurrente | 0.899 | 0.049 | 0.889 | +0.0099 | `sev_extrapolate_high` (0.831) | 0.068 | 10.6 % |
| `pes_dql` | Double Q-Learning | 0.896 | 0.048 | 0.877 | +0.0191 | `sev_extrapolate_high` (0.785) | 0.111 | 8.6 % |
| `pes_ql` | Q-Learning | 0.887 | 0.061 | 0.871 | +0.0152 | `sev_extrapolate_high` (0.762) | 0.124 | 0.0 % |
| `pes_base` | Q-Learning base | 0.871 | 0.074 | 0.851 | +0.0200 | `sev_extrapolate_high` (0.627) | 0.244 | -14.2 % |

### 4.3 Degradación por familia

| Paquete | Deg. severidad | Deg. longitud | Deg. conjunta | Deg. estructural | Media 18 sin estruct. (derivado) | Media 3 fuera de rango (derivado) |
|---|---:|---:|---:|---:|---:|---:|
| `pes_base` | +0.0316 | +0.0053 | +0.0272 | 0.0000 | 0.847 | 0.716 |
| `pes_ql` | +0.0240 | +0.0098 | +0.0136 | 0.0000 | 0.869 | 0.802 |
| `pes_dql` | +0.0294 | +0.0062 | +0.0265 | 0.0000 | 0.874 | 0.819 |
| `pes_dqn` | -0.0016 | +0.0084 | -0.0330 | 0.0000 | 0.900 | 0.884 |
| `pes_rdqn` | +0.0112 | +0.0089 | +0.0154 | 0.0000 | 0.887 | 0.863 |
| `pes_a2c` | -0.0047 | +0.0046 | -0.0425 | 0.0000 | 0.898 | 0.901 |
| `pes_trf` | -0.0065 | +0.0231 | -0.0271 | 0.0000 | 0.930 | 0.951 |

Promedio del grupo individual: severidad +0.0119, longitud +0.0095, conjunta -0.0028, estructural 0.0000.

## 5. Ensambles

### 5.1 Reglas y parámetros

| Paquete | Nombre en el texto | Regla | Miembros | Parámetros | Origen |
|---|---|---|---|---|---|
| `pes_ens` | Ensamble ponderado | Voto suave ponderado por (0,1 + c_k) sobre DQN, DQN recurrente y Transformer; factor 0,3 a a=0 si R>0; mezcla con prior de severidad gaussiano; argmax; cota de seguridad floor(S/2) si S>=6. | dqn, rdqn, trf | tau = 15.0, w_dqn = 0.18, w_rdqn = 0.9, w_trf = 5.0, peso_normalizado_trf = 0.822, w_prior = 0.17, sigma_prior = 3.0 | valores fijos de config/CONFIG.py (sin optimización) |
| `pes_ens_sprb` | Voto suave | Voto suave: promedio de p_k con pesos efectivos w_k·c_k^rho; argmax. | dqn, rdqn, trf, a2c | rho = 2.68075, tau = 1.219194, w_a2c = 0.007027, w_dqn = 0.270032, w_rdqn = 1.759607, w_trf = 0.895639 | mejor ensayo de Optuna/TPE sobre las 64 secuencias de referencia (inputs/best_params.json) |
| `pes_ens_accq` | Voto por acción | Voto por acción: cada miembro vota su argmax con peso w_k·c_k^rho; desempate por suma de Q estandarizados. | dqn, rdqn, trf, a2c | rho = 2.1972, tau = 1.0, w_a2c = 0.174251, w_dqn = 0.467984, w_rdqn = 2.598528, w_trf = 1.803345 | mejor ensayo de Optuna/TPE sobre las 64 secuencias de referencia (inputs/best_params.json) |
| `pes_ens_consensus` | Consenso | Voto ponderado + beta_a·(confianza de quienes coinciden) - beta_d·(confianza de quienes discrepan) + sum_k Qhat_k(a)·c_k; argmax. | dqn, rdqn, trf, a2c | beta_a = 2.074141, beta_d = 0.121143, rho = 0.275376, tau = 1.0, w_a2c = 1.880651, w_dqn = 1.531684, w_rdqn = 1.931138, w_trf = 2.503556 | mejor ensayo de Optuna/TPE sobre las 64 secuencias de referencia (inputs/best_params.json) |
| `pes_ens_consensus_prior` | Consenso con prior | Como consenso pero con suma de distribuciones; factor 0,3 a a=0 si R>0; mezcla con prior de severidad; cota de seguridad floor(S/2) si S>=6. Confianza calculada aplicando softmax a p_k (reduce c_k). | dqn, rdqn, trf, a2c | beta_a = 1.605238, beta_d = 0.411452, rho = 0.325261, sigma_prior = 1.436415, tau = 1.0, w_a2c = 0.432274, w_dqn = 2.558183, w_prior = 0.118571, w_rdqn = 2.941942, w_trf = 2.995977 | mejor ensayo de Optuna/TPE sobre las 64 secuencias de referencia (inputs/best_params.json) |
| `pes_ens_trf_guard` | Compuerta del Transformer | Compuerta logística g = sigmoid(kappa_g·(c_trf - tau_g)); si g>=0,5 ejecuta la acción del Transformer, si no un voto suave de respaldo. | dqn, rdqn, trf, a2c | kappa_g = 2.397349, rho = 1.451819, tau = 1.0, tau_g = 0.222186, w_a2c = 0.067448, w_dqn = 0.83668, w_rdqn = 2.561269, w_trf = 2.601263 | mejor ensayo de Optuna/TPE sobre las 64 secuencias de referencia (inputs/best_params.json) |

- Todos: $p_k$ = softmax de temperatura τ de los Q (o salida del actor de A2C), renormalizada sobre acciones factibles; confianza $c_k = 1 - H_{norm}(p_k)$; peso efectivo $\tilde w_k = w_k c_k^{\rho}$ (salvo `pes_ens`, que usa $w_k(0{,}1 + c_k)$).
- `pes_ens`: pesos base 0,18 / 0,90 / 5,0 (DQN / DQN recurrente / Transformer) → el Transformer tiene el 82 % del peso normalizado; A2C deshabilitado; recorta S a 9.
- En los cinco ensambles optimizados, el score de Optuna **coincide** con la media del benchmark en la referencia; los cinco optimizan también `w_a2c` (rango 0–3). Sólo el voto suave optimiza τ (1,219); los demás usan τ = 1.

### 5.2 Resumen de desempeño (ordenado por generalización)

| Paquete | Nombre en el texto | Ref. | σ ref. | Gen. (21) | Deg. media | Peor escenario (media) | Mayor deg. | Δrel vs QL ref. |
|---|---|---:|---:|---:|---:|---|---:|---:|
| `pes_ens` | Ensamble ponderado | 0.937 | 0.035 | 0.939 | -0.0021 | `len_extrapolate_long` (0.900) | 0.037 | 44.7 % |
| `pes_ens_trf_guard` | Compuerta del Transformer | 0.928 | 0.046 | 0.931 | -0.0026 | `len_extrapolate_long` (0.861) | 0.067 | 36.4 % |
| `pes_ens_consensus_prior` | Consenso con prior | 0.917 | 0.041 | 0.922 | -0.0048 | `len_extrapolate_long` (0.871) | 0.046 | 26.7 % |
| `pes_ens_consensus` | Consenso | 0.889 | 0.064 | 0.905 | -0.0152 | `len_extrapolate_long` (0.826) | 0.063 | 2.4 % |
| `pes_ens_sprb` | Voto suave | 0.914 | 0.045 | 0.902 | +0.0118 | `sev_extrapolate_high` (0.860) | 0.054 | 24.3 % |
| `pes_ens_accq` | Voto por acción | 0.914 | 0.044 | 0.901 | +0.0130 | `sev_extrapolate_high` (0.859) | 0.055 | 24.3 % |

### 5.3 Degradación por familia

| Paquete | Deg. severidad | Deg. longitud | Deg. conjunta | Deg. estructural | Media 18 sin estruct. (derivado) | Media 3 fuera de rango (derivado) |
|---|---:|---:|---:|---:|---:|---:|
| `pes_ens` | -0.0085 | +0.0192 | -0.0161 | 0.0000 | 0.940 | 0.967 |
| `pes_ens_sprb` | +0.0137 | +0.0163 | +0.0110 | 0.0000 | 0.900 | 0.879 |
| `pes_ens_accq` | +0.0150 | +0.0156 | +0.0150 | 0.0000 | 0.899 | 0.879 |
| `pes_ens_consensus` | -0.0170 | +0.0026 | -0.0447 | 0.0000 | 0.907 | 0.905 |
| `pes_ens_consensus_prior` | -0.0074 | +0.0137 | -0.0253 | 0.0000 | 0.922 | 0.932 |
| `pes_ens_trf_guard` | -0.0067 | +0.0223 | -0.0267 | 0.0000 | 0.931 | 0.951 |

Promedio del grupo de ensambles: severidad -0.0018, longitud +0.0149, conjunta -0.0145, estructural 0.0000.

### 5.4 Frecuencia con que las reglas fijas cambian la decisión (Tabla `tab:ens-freq`)

_Tesis, Tabla tab:ens-freq; generado con writings/auxiliar/scripts/ensemble_decisions.py (salida no persistida en archivo). El replay reproduce las medias del benchmark con diferencia < 0,0001. Unidad: % de decisiones con recursos disponibles (R > 0)._

| Ensamble | Evento | Referencia | Generalización |
|---|---|---:|---:|
| Compuerta del Transformer | Sigue al Transformer (g ≥ 0,5) | 96.9 | 95.0 |
| Compuerta del Transformer | Acción final distinta de la del Transformer | 0.8 | 2.7 |
| Consenso con prior | El prior cambia la acción votada | 2.4 | 3.6 |
| Consenso con prior | La cota cambia la acción | 0.0 | 1.1 |
| Consenso con prior | Acción final distinta de la del Transformer | 63.5 | 57.9 |
| Ensamble ponderado | El prior cambia la acción votada | 40.9 | 32.5 |
| Ensamble ponderado | La cota cambia la acción | 0.0 | 0.3 |
| Ensamble ponderado | Acción final distinta de la del Transformer | 61.2 | 48.1 |

- Con tau = 15 las distribuciones de los tres miembros quedan casi planas: en la referencia la acción más probable de cada uno tiene en promedio probabilidad 0,19 y supera a la segunda por 0,02-0,03.
- El voto (antes del prior) ya difiere de la acción del Transformer en el 31 % de las decisiones de la referencia.
- En sev_extrapolate_high y joint_extrap_both todas las decisiones de pes_ens coinciden con las de su miembro Transformer (que recibe la severidad recortada a 9).

## 6. Catálogo de escenarios (1 referencia + 21 de generalización)

| Escenario | Familia | Nombre en el texto | Descripción | Secuencias | Pasos | Fuera de rango |
|---|---|---|---|---:|---:|:---:|
| `sev_base` | referencia | referencia | Distribución empírica de entrenamiento: initial_severity.csv y sequence_lengths.csv sin perturbar (severidades 2..8, longitudes 3..10). | 64 | 360 |  |
| `sev_uniform` | severidad | — | Severidad U{0..9}; longitudes empíricas. | 64 | 360 |  |
| `sev_gauss_low` | severidad | — | Severidad N(2; 1,5) truncada a [0, 9] (brotes leves); longitudes empíricas. | 64 | 360 |  |
| `sev_gauss_mid` | severidad | — | Severidad N(4,5; 2,0) truncada a [0, 9] (brotes medios); longitudes empíricas. | 64 | 360 |  |
| `sev_gauss_high` | severidad | — | Severidad N(7; 1,5) truncada a [0, 9] (brotes intensos); longitudes empíricas. | 64 | 360 |  |
| `sev_weibull` | severidad | — | Severidad Weibull(k = 1,5), escala para media ~4,5, recortada a [0, 9] (cola superior pesada); longitudes empíricas. | 64 | 360 |  |
| `sev_beta_lowskew` | severidad | — | Severidad 9·Beta(2, 5) (sesgo bajo); longitudes empíricas. | 64 | 360 |  |
| `sev_beta_highskew` | severidad | — | Severidad 9·Beta(5, 2) (sesgo alto); longitudes empíricas. | 64 | 360 |  |
| `sev_bimodal` | severidad | — | Severidad 0,5·N(2, 1) + 0,5·N(7, 1), recortada a [0, 9]; longitudes empíricas. | 64 | 360 |  |
| `sev_extrapolate_high` | severidad | extrapolación de severidad | Severidad U{10, 11, 12}, por ENCIMA de la cota S_max = 9 (fuera de rango); longitudes empíricas. | 64 | 360 | sí |
| `len_all_short` | longitud | — | Todas las secuencias de longitud 3; severidades empíricas. | 64 | 192 |  |
| `len_all_long` | longitud | — | Todas las secuencias de longitud 10; severidades empíricas. | 64 | 640 |  |
| `len_geometric` | longitud | — | Longitud 2 + Geom(p = 0,2) recortada a [3, 10]; severidades empíricas. | 64 | 380 |  |
| `len_poisson` | longitud | — | Longitud Poisson(λ = 5) recortada a [3, 10]; severidades empíricas. | 64 | 333 |  |
| `len_extrapolate_long` | longitud | secuencias largas | Longitud U{11..20}, por ENCIMA de la cota T_max = 10 (fuera de rango); severidades empíricas. | 64 | 1014 | sí |
| `joint_high_long` | conjunta | — | Severidad N(7; 1,5) truncada × todas las secuencias de longitud 10. | 64 | 640 |  |
| `joint_low_short` | conjunta | — | Severidad N(2; 1,5) truncada × todas las secuencias de longitud 3. | 64 | 192 |  |
| `joint_uniform_geom` | conjunta | — | Severidad U{0..9} × longitudes geométricas (p = 0,2). | 64 | 380 |  |
| `joint_extrap_both` | conjunta | extrapolación conjunta | Severidad U{10..12} × longitud U{11..20}: ambas fuera de rango. | 64 | 1014 | sí |
| `struct_few_long_blocks` | estructural | — | 4 bloques × 16 secuencias = 64; misma distribución que la referencia (control). | 64 | 360 |  |
| `struct_many_short_blocks` | estructural | — | 16 bloques × 4 secuencias = 64; misma distribución que la referencia (control). | 64 | 360 |  |
| `struct_more_total` | estructural | — | 8 bloques × 16 secuencias = 128; los CSV empíricos se repiten (las 64 secuencias dos veces), por eso reproduce exactamente la media de la referencia (control). | 128 | 720 |  |

- Familias: severidad 9 · longitud 5 · conjunta 4 · estructural 3. Se descartaron escenarios de severidad constante (S_peor = S_mejor ⇒ métrica indefinida).
- Los 8 escenarios de severidad dentro de rango incluyen S = 0, 1 o 9, que están en el espacio de estados pero no aparecen en el entrenamiento (2..8).

## 7. Matrices modelo × escenario

Leyenda de columnas: QL-base = `pes_base` (Q-Learning base), QL = `pes_ql` (Q-Learning), DQL = `pes_dql` (Double Q-Learning), DQN = `pes_dqn` (DQN), RDQN = `pes_rdqn` (DQN recurrente), A2C = `pes_a2c` (A2C), TRF = `pes_trf` (Transformer), ENS = `pes_ens` (Ensamble ponderado), VS = `pes_ens_sprb` (Voto suave), VA = `pes_ens_accq` (Voto por acción), CONS = `pes_ens_consensus` (Consenso), CONS+P = `pes_ens_consensus_prior` (Consenso con prior), GUARD = `pes_ens_trf_guard` (Compuerta del Transformer). ⚠ = fuera de rango. En negrita, la mayor media de la fila dentro del grupo.

### 7.1 Media $\bar r$ — individuales

| Escenario | QL-base | QL | DQL | DQN | RDQN | A2C | TRF |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sev_base` | 0.871 | 0.887 | 0.896 | 0.894 | 0.899 | 0.887 | **0.927** |
| `sev_uniform` | 0.831 | 0.843 | 0.840 | 0.874 | 0.870 | 0.860 | **0.923** |
| `sev_gauss_low` | 0.881 | 0.895 | 0.890 | 0.895 | 0.905 | 0.860 | **0.914** |
| `sev_gauss_mid` | 0.847 | 0.880 | 0.888 | 0.895 | 0.889 | 0.886 | **0.920** |
| `sev_gauss_high` | 0.898 | 0.909 | 0.916 | 0.930 | 0.932 | 0.941 | **0.949** |
| `sev_weibull` | 0.825 | 0.842 | 0.845 | 0.883 | 0.881 | 0.877 | **0.929** |
| `sev_beta_lowskew` | 0.871 | 0.854 | 0.861 | 0.879 | 0.891 | 0.858 | **0.905** |
| `sev_beta_highskew` | 0.916 | 0.914 | 0.924 | 0.935 | 0.926 | **0.944** | 0.942 |
| `sev_bimodal` | 0.854 | 0.864 | 0.853 | 0.877 | 0.862 | 0.872 | **0.925** |
| `sev_extrapolate_high` ⚠ | 0.627 | 0.762 | 0.785 | 0.890 | 0.831 | 0.929 | **0.996** |
| `len_all_short` | 0.942 | 0.943 | 0.922 | 0.935 | 0.869 | **0.958** | 0.936 |
| `len_all_long` | 0.836 | 0.859 | 0.878 | 0.860 | **0.901** | 0.848 | 0.876 |
| `len_geometric` | 0.859 | 0.869 | 0.898 | 0.888 | 0.895 | 0.884 | **0.916** |
| `len_poisson` | 0.873 | 0.878 | 0.909 | 0.903 | 0.894 | 0.898 | **0.933** |
| `len_extrapolate_long` ⚠ | 0.817 | 0.835 | 0.844 | 0.841 | **0.889** | 0.826 | 0.860 |
| `joint_high_long` | 0.896 | 0.903 | 0.921 | **0.935** | 0.930 | 0.922 | 0.930 |
| `joint_low_short` | 0.967 | 0.936 | 0.914 | **0.988** | 0.895 | **0.988** | 0.969 |
| `joint_uniform_geom` | 0.805 | 0.846 | 0.816 | 0.864 | 0.841 | 0.861 | **0.921** |
| `joint_extrap_both` ⚠ | 0.705 | 0.807 | 0.829 | 0.920 | 0.867 | 0.949 | **0.997** |
| `struct_few_long_blocks` | 0.871 | 0.887 | 0.896 | 0.894 | 0.899 | 0.887 | **0.927** |
| `struct_many_short_blocks` | 0.871 | 0.887 | 0.896 | 0.894 | 0.899 | 0.887 | **0.927** |
| `struct_more_total` | 0.871 | 0.887 | 0.896 | 0.894 | 0.899 | 0.887 | **0.927** |

### 7.2 Media $\bar r$ — ensambles

| Escenario | ENS | VS | VA | CONS | CONS+P | GUARD |
|---|---:|---:|---:|---:|---:|---:|
| `sev_base` | **0.937** | 0.914 | 0.914 | 0.889 | 0.917 | 0.928 |
| `sev_uniform` | **0.937** | 0.883 | 0.884 | 0.874 | 0.911 | 0.923 |
| `sev_gauss_low` | **0.939** | 0.910 | 0.908 | 0.896 | 0.920 | 0.919 |
| `sev_gauss_mid` | **0.938** | 0.902 | 0.900 | 0.890 | 0.906 | 0.921 |
| `sev_gauss_high` | **0.957** | 0.942 | 0.942 | 0.948 | 0.949 | 0.949 |
| `sev_weibull` | **0.938** | 0.901 | 0.897 | 0.890 | 0.914 | 0.929 |
| `sev_beta_lowskew` | **0.917** | 0.894 | 0.898 | 0.888 | 0.898 | 0.907 |
| `sev_beta_highskew` | **0.947** | 0.934 | 0.927 | 0.947 | 0.945 | 0.943 |
| `sev_bimodal` | **0.938** | 0.879 | 0.879 | 0.890 | 0.919 | 0.925 |
| `sev_extrapolate_high` ⚠ | **1.000** | 0.860 | 0.859 | 0.935 | 0.957 | 0.996 |
| `len_all_short` | 0.908 | 0.900 | 0.900 | **0.961** | 0.932 | 0.936 |
| `len_all_long` | **0.921** | 0.893 | 0.892 | 0.853 | 0.886 | 0.881 |
| `len_geometric` | **0.929** | 0.909 | 0.912 | 0.890 | 0.912 | 0.917 |
| `len_poisson` | 0.933 | 0.910 | 0.909 | 0.903 | 0.915 | **0.933** |
| `len_extrapolate_long` ⚠ | **0.900** | 0.878 | 0.880 | 0.826 | 0.871 | 0.861 |
| `joint_high_long` | **0.943** | 0.925 | 0.925 | 0.918 | 0.928 | 0.930 |
| `joint_low_short` | 0.942 | 0.921 | 0.913 | **0.988** | 0.964 | 0.969 |
| `joint_uniform_geom` | **0.928** | 0.867 | 0.862 | 0.878 | 0.908 | 0.922 |
| `joint_extrap_both` ⚠ | **1.000** | 0.900 | 0.897 | 0.953 | 0.968 | 0.997 |
| `struct_few_long_blocks` | **0.937** | 0.914 | 0.914 | 0.889 | 0.917 | 0.928 |
| `struct_many_short_blocks` | **0.937** | 0.914 | 0.914 | 0.889 | 0.917 | 0.928 |
| `struct_more_total` | **0.937** | 0.914 | 0.914 | 0.889 | 0.917 | 0.928 |

### 7.3 Desviación estándar entre secuencias — individuales

| Escenario | QL-base | QL | DQL | DQN | RDQN | A2C | TRF |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sev_base` | 0.074 | 0.061 | 0.048 | 0.055 | 0.049 | 0.063 | 0.045 |
| `sev_uniform` | 0.122 | 0.108 | 0.113 | 0.070 | 0.111 | 0.099 | 0.056 |
| `sev_gauss_low` | 0.084 | 0.087 | 0.077 | 0.082 | 0.074 | 0.108 | 0.074 |
| `sev_gauss_mid` | 0.108 | 0.075 | 0.072 | 0.063 | 0.072 | 0.072 | 0.052 |
| `sev_gauss_high` | 0.072 | 0.050 | 0.067 | 0.039 | 0.058 | 0.027 | 0.024 |
| `sev_weibull` | 0.117 | 0.116 | 0.092 | 0.074 | 0.096 | 0.088 | 0.052 |
| `sev_beta_lowskew` | 0.094 | 0.103 | 0.081 | 0.085 | 0.082 | 0.102 | 0.069 |
| `sev_beta_highskew` | 0.048 | 0.051 | 0.046 | 0.040 | 0.061 | 0.028 | 0.032 |
| `sev_bimodal` | 0.116 | 0.092 | 0.097 | 0.083 | 0.134 | 0.099 | 0.054 |
| `sev_extrapolate_high` ⚠ | 0.047 | 0.021 | 0.026 | 0.025 | 0.112 | 0.016 | 0.008 |
| `len_all_short` | 0.059 | 0.058 | 0.045 | 0.051 | 0.068 | 0.026 | 0.063 |
| `len_all_long` | 0.055 | 0.054 | 0.042 | 0.037 | 0.027 | 0.041 | 0.032 |
| `len_geometric` | 0.082 | 0.066 | 0.052 | 0.058 | 0.051 | 0.065 | 0.049 |
| `len_poisson` | 0.074 | 0.073 | 0.050 | 0.061 | 0.056 | 0.067 | 0.047 |
| `len_extrapolate_long` ⚠ | 0.067 | 0.063 | 0.042 | 0.051 | 0.033 | 0.053 | 0.045 |
| `joint_high_long` | 0.041 | 0.052 | 0.045 | 0.037 | 0.025 | 0.040 | 0.024 |
| `joint_low_short` | 0.054 | 0.131 | 0.099 | 0.018 | 0.114 | 0.018 | 0.041 |
| `joint_uniform_geom` | 0.142 | 0.111 | 0.113 | 0.083 | 0.126 | 0.100 | 0.069 |
| `joint_extrap_both` ⚠ | 0.003 | 0.002 | 0.002 | 0.001 | 0.085 | 0.001 | 0.006 |
| `struct_few_long_blocks` | 0.074 | 0.061 | 0.048 | 0.055 | 0.049 | 0.063 | 0.045 |
| `struct_many_short_blocks` | 0.074 | 0.061 | 0.048 | 0.055 | 0.049 | 0.063 | 0.045 |
| `struct_more_total` | 0.074 | 0.061 | 0.048 | 0.055 | 0.049 | 0.063 | 0.045 |

### 7.4 Desviación estándar entre secuencias — ensambles

| Escenario | ENS | VS | VA | CONS | CONS+P | GUARD |
|---|---:|---:|---:|---:|---:|---:|
| `sev_base` | 0.035 | 0.045 | 0.044 | 0.064 | 0.041 | 0.046 |
| `sev_uniform` | 0.046 | 0.095 | 0.095 | 0.084 | 0.051 | 0.056 |
| `sev_gauss_low` | 0.042 | 0.074 | 0.075 | 0.085 | 0.067 | 0.070 |
| `sev_gauss_mid` | 0.039 | 0.059 | 0.060 | 0.067 | 0.049 | 0.052 |
| `sev_gauss_high` | 0.018 | 0.037 | 0.038 | 0.029 | 0.025 | 0.024 |
| `sev_weibull` | 0.042 | 0.080 | 0.082 | 0.077 | 0.058 | 0.051 |
| `sev_beta_lowskew` | 0.063 | 0.093 | 0.088 | 0.086 | 0.083 | 0.068 |
| `sev_beta_highskew` | 0.032 | 0.045 | 0.058 | 0.026 | 0.030 | 0.031 |
| `sev_bimodal` | 0.043 | 0.130 | 0.131 | 0.084 | 0.057 | 0.054 |
| `sev_extrapolate_high` ⚠ | 0.000 | 0.112 | 0.113 | 0.019 | 0.011 | 0.008 |
| `len_all_short` | 0.064 | 0.059 | 0.059 | 0.021 | 0.057 | 0.063 |
| `len_all_long` | 0.032 | 0.030 | 0.031 | 0.043 | 0.019 | 0.038 |
| `len_geometric` | 0.038 | 0.044 | 0.042 | 0.067 | 0.046 | 0.049 |
| `len_poisson` | 0.041 | 0.042 | 0.044 | 0.065 | 0.042 | 0.048 |
| `len_extrapolate_long` ⚠ | 0.034 | 0.035 | 0.042 | 0.048 | 0.039 | 0.048 |
| `joint_high_long` | 0.018 | 0.023 | 0.025 | 0.038 | 0.020 | 0.023 |
| `joint_low_short` | 0.056 | 0.107 | 0.116 | 0.018 | 0.049 | 0.041 |
| `joint_uniform_geom` | 0.056 | 0.119 | 0.114 | 0.094 | 0.060 | 0.067 |
| `joint_extrap_both` ⚠ | 0.000 | 0.081 | 0.084 | 0.012 | 0.003 | 0.006 |
| `struct_few_long_blocks` | 0.035 | 0.045 | 0.044 | 0.064 | 0.041 | 0.046 |
| `struct_many_short_blocks` | 0.035 | 0.045 | 0.044 | 0.064 | 0.041 | 0.046 |
| `struct_more_total` | 0.035 | 0.045 | 0.044 | 0.064 | 0.041 | 0.046 |

### 7.5 Degradación ($\bar r_{ref} - \bar r_{esc}$) — individuales

| Escenario | QL-base | QL | DQL | DQN | RDQN | A2C | TRF |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sev_base` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `sev_uniform` | +0.039 | +0.044 | +0.056 | +0.020 | +0.029 | +0.027 | +0.004 |
| `sev_gauss_low` | -0.010 | -0.009 | +0.006 | -0.002 | -0.007 | +0.027 | +0.014 |
| `sev_gauss_mid` | +0.024 | +0.007 | +0.008 | -0.001 | +0.010 | +0.001 | +0.007 |
| `sev_gauss_high` | -0.028 | -0.023 | -0.020 | -0.037 | -0.033 | -0.053 | -0.022 |
| `sev_weibull` | +0.045 | +0.045 | +0.051 | +0.011 | +0.018 | +0.010 | -0.002 |
| `sev_beta_lowskew` | 0.000 | +0.032 | +0.036 | +0.014 | +0.008 | +0.029 | +0.022 |
| `sev_beta_highskew` | -0.046 | -0.028 | -0.027 | -0.041 | -0.027 | -0.057 | -0.015 |
| `sev_bimodal` | +0.017 | +0.023 | +0.044 | +0.017 | +0.036 | +0.016 | +0.002 |
| `sev_extrapolate_high` ⚠ | +0.244 | +0.124 | +0.111 | +0.004 | +0.068 | -0.042 | -0.069 |
| `len_all_short` | -0.071 | -0.056 | -0.025 | -0.041 | +0.029 | -0.071 | -0.009 |
| `len_all_long` | +0.035 | +0.028 | +0.019 | +0.034 | -0.002 | +0.039 | +0.052 |
| `len_geometric` | +0.012 | +0.017 | -0.002 | +0.006 | +0.004 | +0.004 | +0.011 |
| `len_poisson` | -0.002 | +0.008 | -0.013 | -0.009 | +0.005 | -0.010 | -0.006 |
| `len_extrapolate_long` ⚠ | +0.054 | +0.052 | +0.052 | +0.053 | +0.009 | +0.062 | +0.068 |
| `joint_high_long` | -0.026 | -0.016 | -0.025 | -0.042 | -0.031 | -0.035 | -0.003 |
| `joint_low_short` | -0.097 | -0.049 | -0.017 | -0.094 | +0.004 | -0.100 | -0.042 |
| `joint_uniform_geom` | +0.066 | +0.040 | +0.080 | +0.030 | +0.058 | +0.027 | +0.007 |
| `joint_extrap_both` ⚠ | +0.166 | +0.079 | +0.068 | -0.026 | +0.031 | -0.061 | -0.070 |
| `struct_few_long_blocks` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `struct_many_short_blocks` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `struct_more_total` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### 7.6 Degradación — ensambles

| Escenario | ENS | VS | VA | CONS | CONS+P | GUARD |
|---|---:|---:|---:|---:|---:|---:|
| `sev_base` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `sev_uniform` | 0.000 | +0.032 | +0.030 | +0.015 | +0.006 | +0.005 |
| `sev_gauss_low` | -0.002 | +0.005 | +0.007 | -0.006 | -0.003 | +0.009 |
| `sev_gauss_mid` | -0.001 | +0.012 | +0.015 | -0.001 | +0.011 | +0.007 |
| `sev_gauss_high` | -0.020 | -0.028 | -0.028 | -0.058 | -0.032 | -0.021 |
| `sev_weibull` | 0.000 | +0.013 | +0.017 | -0.001 | +0.003 | -0.002 |
| `sev_beta_lowskew` | +0.020 | +0.020 | +0.016 | +0.002 | +0.019 | +0.021 |
| `sev_beta_highskew` | -0.010 | -0.020 | -0.013 | -0.057 | -0.028 | -0.015 |
| `sev_bimodal` | -0.001 | +0.035 | +0.036 | 0.000 | -0.002 | +0.003 |
| `sev_extrapolate_high` ⚠ | -0.063 | +0.054 | +0.055 | -0.046 | -0.040 | -0.068 |
| `len_all_short` | +0.029 | +0.014 | +0.014 | -0.072 | -0.016 | -0.008 |
| `len_all_long` | +0.017 | +0.021 | +0.022 | +0.037 | +0.031 | +0.047 |
| `len_geometric` | +0.008 | +0.005 | +0.002 | -0.001 | +0.004 | +0.011 |
| `len_poisson` | +0.005 | +0.004 | +0.005 | -0.014 | +0.002 | -0.006 |
| `len_extrapolate_long` ⚠ | +0.037 | +0.036 | +0.035 | +0.063 | +0.046 | +0.067 |
| `joint_high_long` | -0.006 | -0.011 | -0.010 | -0.028 | -0.012 | -0.003 |
| `joint_low_short` | -0.004 | -0.006 | +0.001 | -0.098 | -0.047 | -0.042 |
| `joint_uniform_geom` | +0.009 | +0.047 | +0.052 | +0.012 | +0.009 | +0.006 |
| `joint_extrap_both` ⚠ | -0.063 | +0.014 | +0.017 | -0.064 | -0.052 | -0.069 |
| `struct_few_long_blocks` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `struct_many_short_blocks` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `struct_more_total` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### 7.7 $d$ de Cohen vs referencia (positivo = mejor en el escenario) — individuales

| Escenario | QL-base | QL | DQL | DQN | RDQN | A2C | TRF |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sev_base` | — | — | — | — | — | — | — |
| `sev_uniform` | -0.38 | -0.49 | -0.64 | -0.32 | -0.34 | -0.32 | -0.08 |
| `sev_gauss_low` | +0.13 | +0.11 | -0.10 | +0.02 | +0.11 | -0.30 | -0.22 |
| `sev_gauss_mid` | -0.25 | -0.10 | -0.14 | +0.02 | -0.16 | -0.02 | -0.14 |
| `sev_gauss_high` | +0.38 | +0.40 | +0.34 | +0.76 | +0.62 | +1.09 | +0.61 |
| `sev_weibull` | -0.46 | -0.48 | -0.69 | -0.17 | -0.23 | -0.13 | +0.04 |
| `sev_beta_lowskew` | 0.00 | -0.38 | -0.53 | -0.20 | -0.11 | -0.34 | -0.38 |
| `sev_beta_highskew` | +0.73 | +0.49 | +0.58 | +0.84 | +0.49 | +1.15 | +0.38 |
| `sev_bimodal` | -0.17 | -0.29 | -0.57 | -0.23 | -0.36 | -0.19 | -0.03 |
| `sev_extrapolate_high` ⚠ | -3.90 | -2.71 | -2.87 | -0.09 | -0.78 | +0.91 | +2.08 |
| `len_all_short` | +1.06 | +0.93 | +0.54 | +0.76 | -0.49 | +1.46 | +0.15 |
| `len_all_long` | -0.53 | -0.48 | -0.41 | -0.71 | +0.05 | -0.73 | -1.30 |
| `len_geometric` | -0.15 | -0.27 | +0.03 | -0.10 | -0.07 | -0.06 | -0.22 |
| `len_poisson` | +0.03 | -0.12 | +0.26 | +0.15 | -0.09 | +0.16 | +0.13 |
| `len_extrapolate_long` ⚠ | -0.75 | -0.83 | -1.15 | -0.98 | -0.22 | -1.04 | -1.48 |
| `joint_high_long` | +0.43 | +0.28 | +0.52 | +0.88 | +0.79 | +0.65 | +0.08 |
| `joint_low_short` | +1.48 | +0.48 | +0.22 | +2.28 | -0.05 | +2.15 | +0.97 |
| `joint_uniform_geom` | -0.58 | -0.44 | -0.92 | -0.42 | -0.60 | -0.31 | -0.11 |
| `joint_extrap_both` ⚠ | -3.14 | -1.83 | -1.99 | +0.66 | -0.45 | +1.36 | +2.14 |
| `struct_few_long_blocks` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `struct_many_short_blocks` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `struct_more_total` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

### 7.8 $d$ de Cohen vs referencia — ensambles

| Escenario | ENS | VS | VA | CONS | CONS+P | GUARD |
|---|---:|---:|---:|---:|---:|---:|
| `sev_base` | — | — | — | — | — | — |
| `sev_uniform` | 0.00 | -0.42 | -0.40 | -0.20 | -0.13 | -0.10 |
| `sev_gauss_low` | +0.04 | -0.08 | -0.11 | +0.08 | +0.06 | -0.15 |
| `sev_gauss_mid` | +0.01 | -0.23 | -0.28 | +0.01 | -0.24 | -0.13 |
| `sev_gauss_high` | +0.72 | +0.68 | +0.67 | +1.17 | +0.94 | +0.58 |
| `sev_weibull` | +0.01 | -0.20 | -0.26 | +0.01 | -0.06 | +0.03 |
| `sev_beta_lowskew` | -0.39 | -0.27 | -0.23 | -0.02 | -0.28 | -0.37 |
| `sev_beta_highskew` | +0.30 | +0.43 | +0.24 | +1.17 | +0.77 | +0.37 |
| `sev_bimodal` | +0.02 | -0.36 | -0.36 | 0.00 | +0.05 | -0.07 |
| `sev_extrapolate_high` ⚠ | +2.52 | -0.62 | -0.64 | +0.97 | +1.31 | +2.05 |
| `len_all_short` | -0.56 | -0.27 | -0.27 | +1.51 | +0.31 | +0.14 |
| `len_all_long` | -0.49 | -0.56 | -0.57 | -0.67 | -0.96 | -1.11 |
| `len_geometric` | -0.23 | -0.12 | -0.05 | +0.02 | -0.10 | -0.22 |
| `len_poisson` | -0.12 | -0.09 | -0.11 | +0.21 | -0.05 | +0.12 |
| `len_extrapolate_long` ⚠ | -1.07 | -0.89 | -0.79 | -1.12 | -1.14 | -1.41 |
| `joint_high_long` | +0.22 | +0.31 | +0.29 | +0.54 | +0.35 | +0.07 |
| `joint_low_short` | +0.09 | +0.08 | -0.02 | +2.09 | +1.04 | +0.95 |
| `joint_uniform_geom` | -0.19 | -0.53 | -0.60 | -0.14 | -0.17 | -0.11 |
| `joint_extrap_both` ⚠ | +2.52 | -0.21 | -0.25 | +1.38 | +1.76 | +2.10 |
| `struct_few_long_blocks` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `struct_many_short_blocks` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `struct_more_total` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

### 7.9 $\log_{10} p$ de Welch vs referencia — individuales

| Escenario | QL-base | QL | DQL | DQN | RDQN | A2C | TRF |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sev_base` | — | — | — | — | — | — | — |
| `sev_uniform` | -1.5 | -2.2 | -3.3 | -1.1 | -1.2 | -1.1 | -0.2 |
| `sev_gauss_low` | -0.3 | -0.3 | -0.2 | -0.1 | -0.3 | -1.0 | -0.7 |
| `sev_gauss_mid` | -0.8 | -0.2 | -0.4 | 0.0 | -0.4 | 0.0 | -0.4 |
| `sev_gauss_high` | -1.5 | -1.6 | -1.3 | -4.4 | -3.2 | -7.7 | -3.1 |
| `sev_weibull` | -2.0 | -2.1 | -3.7 | -0.5 | -0.7 | -0.3 | -0.1 |
| `sev_beta_lowskew` | 0.0 | -1.5 | -2.5 | -0.6 | -0.3 | -1.3 | -1.5 |
| `sev_beta_highskew` | -4.1 | -2.2 | -2.9 | -5.3 | -2.2 | -8.3 | -1.5 |
| `sev_bimodal` | -0.5 | -1.0 | -2.7 | -0.7 | -1.3 | -0.5 | -0.1 |
| `sev_extrapolate_high` ⚠ | -40.9 | -24.5 | -28.7 | -0.2 | -4.5 | -5.6 | -17.2 |
| `len_all_short` | -7.7 | -6.2 | -2.6 | -4.5 | -2.2 | -11.7 | -0.4 |
| `len_all_long` | -2.5 | -2.1 | -1.7 | -4.0 | -0.1 | -4.1 | -10.5 |
| `len_geometric` | -0.4 | -0.9 | -0.1 | -0.3 | -0.2 | -0.1 | -0.7 |
| `len_poisson` | -0.1 | -0.3 | -0.9 | -0.4 | -0.2 | -0.4 | -0.3 |
| `len_extrapolate_long` ⚠ | -4.4 | -5.1 | -8.8 | -6.8 | -0.7 | -7.5 | -13.1 |
| `joint_high_long` | -1.8 | -0.9 | -2.4 | -5.6 | -4.7 | -3.4 | -0.2 |
| `joint_low_short` | -12.8 | -2.1 | -0.7 | -20.0 | -0.1 | -18.5 | -6.7 |
| `joint_uniform_geom` | -2.8 | -1.9 | -5.8 | -1.7 | -2.9 | -1.1 | -0.3 |
| `joint_extrap_both` ⚠ | -25.5 | -14.5 | -16.0 | -3.4 | -1.9 | -9.9 | -17.5 |
| `struct_few_long_blocks` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `struct_many_short_blocks` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `struct_more_total` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### 7.10 $\log_{10} p$ de Welch vs referencia — ensambles

| Escenario | ENS | VS | VA | CONS | CONS+P | GUARD |
|---|---:|---:|---:|---:|---:|---:|
| `sev_base` | — | — | — | — | — | — |
| `sev_uniform` | 0.0 | -1.7 | -1.6 | -0.6 | -0.3 | -0.2 |
| `sev_gauss_low` | -0.1 | -0.2 | -0.3 | -0.2 | -0.1 | -0.4 |
| `sev_gauss_mid` | 0.0 | -0.7 | -0.9 | 0.0 | -0.8 | -0.3 |
| `sev_gauss_high` | -4.0 | -3.7 | -3.6 | -8.6 | -6.2 | -2.9 |
| `sev_weibull` | 0.0 | -0.6 | -0.8 | 0.0 | -0.1 | -0.1 |
| `sev_beta_lowskew` | -1.5 | -0.9 | -0.7 | 0.0 | -0.9 | -1.4 |
| `sev_beta_highskew` | -1.0 | -1.8 | -0.8 | -8.5 | -4.5 | -1.4 |
| `sev_bimodal` | -0.1 | -1.3 | -1.4 | 0.0 | -0.1 | -0.2 |
| `sev_extrapolate_high` ⚠ | -20.6 | -3.2 | -3.3 | -6.2 | -9.7 | -16.9 |
| `len_all_short` | -2.7 | -0.9 | -0.9 | -12.0 | -1.1 | -0.4 |
| `len_all_long` | -2.2 | -2.7 | -2.8 | -3.6 | -6.3 | -8.3 |
| `len_geometric` | -0.7 | -0.3 | -0.1 | 0.0 | -0.2 | -0.7 |
| `len_poisson` | -0.3 | -0.2 | -0.3 | -0.6 | -0.1 | -0.3 |
| `len_extrapolate_long` ⚠ | -7.8 | -5.8 | -4.8 | -8.3 | -8.7 | -12.1 |
| `joint_high_long` | -0.7 | -1.1 | -1.0 | -2.5 | -1.3 | -0.2 |
| `joint_low_short` | -0.2 | -0.2 | 0.0 | -17.8 | -7.4 | -6.4 |
| `joint_uniform_geom` | -0.5 | -2.4 | -2.9 | -0.4 | -0.5 | -0.3 |
| `joint_extrap_both` ⚠ | -20.6 | -0.6 | -0.8 | -10.3 | -13.8 | -17.2 |
| `struct_few_long_blocks` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `struct_many_short_blocks` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `struct_more_total` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### 7.11 KL de acciones vs referencia — individuales

| Escenario | QL-base | QL | DQL | DQN | RDQN | A2C | TRF |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sev_base` | — | — | — | — | — | — | — |
| `sev_uniform` | 0.045 | 0.070 | 0.142 | 1.535 | 0.097 | 0.000 | 0.302 |
| `sev_gauss_low` | 0.080 | 0.126 | 0.168 | 3.144 | 0.366 | 0.000 | 0.331 |
| `sev_gauss_mid` | 0.023 | 0.038 | 0.029 | 0.296 | 0.033 | 0.000 | 0.051 |
| `sev_gauss_high` | 0.103 | 0.022 | 0.066 | 0.000 | 0.385 | 0.000 | 0.280 |
| `sev_weibull` | 0.059 | 0.034 | 0.108 | 1.008 | 0.037 | 0.000 | 0.137 |
| `sev_beta_lowskew` | 0.084 | 0.100 | 0.086 | 2.254 | 0.212 | 0.000 | 0.252 |
| `sev_beta_highskew` | 0.101 | 0.032 | 0.124 | 0.000 | 0.338 | 0.000 | 0.297 |
| `sev_bimodal` | 0.050 | 0.035 | 0.096 | 1.162 | 0.141 | 0.000 | 0.344 |
| `sev_extrapolate_high` ⚠ | 1.092 | 1.147 | 0.697 | 0.000 | 0.723 | 0.000 | 1.064 |
| `len_all_short` | 0.608 | 0.543 | 0.593 | 0.629 | 0.435 | 0.629 | 0.473 |
| `len_all_long` | 0.194 | 0.209 | 0.186 | 0.164 | 0.236 | 0.164 | 0.205 |
| `len_geometric` | 0.034 | 0.024 | 0.013 | 0.003 | 0.025 | 0.003 | 0.021 |
| `len_poisson` | 0.037 | 0.055 | 0.020 | 0.004 | 0.017 | 0.004 | 0.016 |
| `len_extrapolate_long` ⚠ | 0.457 | 0.418 | 0.404 | 0.381 | 0.535 | 0.381 | 0.420 |
| `joint_high_long` | 0.217 | 0.180 | 0.206 | 0.164 | 0.527 | 0.164 | 0.387 |
| `joint_low_short` | 0.468 | 0.385 | 0.425 | 2.819 | 0.628 | 0.629 | 1.025 |
| `joint_uniform_geom` | 0.033 | 0.041 | 0.123 | 1.437 | 0.121 | 0.003 | 0.212 |
| `joint_extrap_both` ⚠ | 0.690 | 0.629 | 0.767 | 0.381 | 0.791 | 0.381 | 0.918 |
| `struct_few_long_blocks` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `struct_many_short_blocks` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `struct_more_total` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### 7.12 KL de acciones vs referencia — ensambles

| Escenario | ENS | VS | VA | CONS | CONS+P | GUARD |
|---|---:|---:|---:|---:|---:|---:|
| `sev_base` | — | — | — | — | — | — |
| `sev_uniform` | 0.096 | 0.190 | 0.122 | 0.127 | 0.278 | 0.309 |
| `sev_gauss_low` | 0.349 | 0.449 | 0.409 | 0.470 | 0.434 | 0.359 |
| `sev_gauss_mid` | 0.041 | 0.028 | 0.035 | 0.070 | 0.060 | 0.171 |
| `sev_gauss_high` | 0.332 | 0.334 | 0.332 | 0.051 | 0.277 | 0.292 |
| `sev_weibull` | 0.047 | 0.097 | 0.065 | 0.150 | 0.279 | 0.159 |
| `sev_beta_lowskew` | 0.204 | 0.275 | 0.227 | 0.416 | 0.311 | 0.319 |
| `sev_beta_highskew` | 0.306 | 0.277 | 0.270 | 0.035 | 0.331 | 0.316 |
| `sev_bimodal` | 0.127 | 0.208 | 0.150 | 0.294 | 0.474 | 0.361 |
| `sev_extrapolate_high` ⚠ | 1.524 | 0.718 | 0.699 | 1.873 | 0.985 | 1.062 |
| `len_all_short` | 0.543 | 0.404 | 0.406 | 0.655 | 0.449 | 0.472 |
| `len_all_long` | 0.234 | 0.279 | 0.257 | 0.169 | 0.245 | 0.194 |
| `len_geometric` | 0.055 | 0.047 | 0.042 | 0.010 | 0.031 | 0.021 |
| `len_poisson` | 0.038 | 0.013 | 0.019 | 0.023 | 0.008 | 0.026 |
| `len_extrapolate_long` ⚠ | 0.595 | 0.558 | 0.532 | 0.388 | 0.552 | 0.416 |
| `joint_high_long` | 0.497 | 0.462 | 0.443 | 0.187 | 0.500 | 0.398 |
| `joint_low_short` | 0.663 | 0.759 | 0.753 | 1.166 | 0.715 | 1.025 |
| `joint_uniform_geom` | 0.070 | 0.167 | 0.146 | 0.171 | 0.259 | 0.255 |
| `joint_extrap_both` ⚠ | 1.367 | 0.812 | 0.761 | 1.082 | 1.035 | 0.905 |
| `struct_few_long_blocks` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `struct_many_short_blocks` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `struct_more_total` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

_Mínimo, máximo, fracción de secuencias en el óptimo (`f_opt`), fracción bajo 0,8 (`f_lt08`) y asignación media por celda: ver `celdas` en `mpes_resultados.json`._

## 8. Ganadores por escenario y ensambles frente al Transformer

| Escenario | Mejor individual | Mejor ensamble | Mejor global | ENS − TRF | VS − TRF | VA − TRF | CONS − TRF | CONS+P − TRF | GUARD − TRF |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `sev_base` | TRF (0.927) | ENS (0.937) | ENS (0.937) | +0.010 | -0.013 | -0.013 | -0.038 | -0.010 | +0.001 |
| `sev_uniform` | TRF (0.923) | ENS (0.937) | ENS (0.937) | +0.014 | -0.040 | -0.039 | -0.048 | -0.012 | 0.000 |
| `sev_gauss_low` | TRF (0.914) | ENS (0.939) | ENS (0.939) | +0.025 | -0.004 | -0.006 | -0.018 | +0.007 | +0.006 |
| `sev_gauss_mid` | TRF (0.920) | ENS (0.938) | ENS (0.938) | +0.018 | -0.018 | -0.021 | -0.030 | -0.014 | +0.001 |
| `sev_gauss_high` | TRF (0.949) | ENS (0.957) | ENS (0.957) | +0.008 | -0.007 | -0.008 | -0.002 | 0.000 | 0.000 |
| `sev_weibull` | TRF (0.929) | ENS (0.938) | ENS (0.938) | +0.008 | -0.028 | -0.032 | -0.039 | -0.015 | 0.000 |
| `sev_beta_lowskew` | TRF (0.905) | ENS (0.917) | ENS (0.917) | +0.013 | -0.011 | -0.006 | -0.017 | -0.007 | +0.002 |
| `sev_beta_highskew` | A2C (0.944) | ENS (0.947) | ENS (0.947) | +0.005 | -0.008 | -0.015 | +0.004 | +0.002 | 0.000 |
| `sev_bimodal` | TRF (0.925) | ENS (0.938) | ENS (0.938) | +0.013 | -0.046 | -0.047 | -0.036 | -0.006 | -0.001 |
| `sev_extrapolate_high` | TRF (0.996) | ENS (1.000) | ENS (1.000) | +0.004 | -0.135 | -0.137 | -0.061 | -0.039 | 0.000 |
| `len_all_short` | A2C (0.958) | CONS (0.961) | CONS (0.961) | -0.028 | -0.036 | -0.036 | +0.026 | -0.003 | 0.000 |
| `len_all_long` | RDQN (0.901) | ENS (0.921) | ENS (0.921) | +0.045 | +0.017 | +0.017 | -0.023 | +0.010 | +0.005 |
| `len_geometric` | TRF (0.916) | ENS (0.929) | ENS (0.929) | +0.012 | -0.007 | -0.004 | -0.026 | -0.004 | +0.001 |
| `len_poisson` | TRF (0.933) | GUARD (0.933) | GUARD (0.933) | 0.000 | -0.023 | -0.024 | -0.030 | -0.019 | 0.000 |
| `len_extrapolate_long` | RDQN (0.889) | ENS (0.900) | ENS (0.900) | +0.040 | +0.018 | +0.020 | -0.034 | +0.011 | +0.002 |
| `joint_high_long` | DQN (0.935) | ENS (0.943) | ENS (0.943) | +0.013 | -0.005 | -0.006 | -0.013 | -0.002 | 0.000 |
| `joint_low_short` | DQN, A2C (0.988) | CONS (0.988) | CONS (0.988) | -0.028 | -0.049 | -0.057 | +0.018 | -0.006 | 0.000 |
| `joint_uniform_geom` | TRF (0.921) | ENS (0.928) | ENS (0.928) | +0.008 | -0.054 | -0.058 | -0.043 | -0.013 | +0.001 |
| `joint_extrap_both` | TRF (0.997) | ENS (1.000) | ENS (1.000) | +0.003 | -0.097 | -0.100 | -0.044 | -0.028 | 0.000 |
| `struct_few_long_blocks` | TRF (0.927) | ENS (0.937) | ENS (0.937) | +0.010 | -0.013 | -0.013 | -0.038 | -0.010 | +0.001 |
| `struct_many_short_blocks` | TRF (0.927) | ENS (0.937) | ENS (0.937) | +0.010 | -0.013 | -0.013 | -0.038 | -0.010 | +0.001 |
| `struct_more_total` | TRF (0.927) | ENS (0.937) | ENS (0.937) | +0.010 | -0.013 | -0.013 | -0.038 | -0.010 | +0.001 |

- El Transformer tiene la mayor media individual (exacta) en **16** de 22 escenarios; con dos decimales queda primero o empatado en 17 (empates a 2 decimales: `sev_gauss_low`, `sev_beta_highskew`).
- Escenarios donde el mejor individual no es el Transformer (exacto): `sev_beta_highskew` (A2C), `len_all_short` (A2C), `len_all_long` (RDQN), `len_extrapolate_long` (RDQN), `joint_high_long` (DQN), `joint_low_short` (DQN, A2C).
- Ensamble ponderado supera al Transformer en 18 de 21 escenarios de generalización (margen máximo +0.045).
- Voto suave supera al Transformer en 2 de 21 escenarios de generalización (margen máximo +0.018).
- Voto por acción supera al Transformer en 2 de 21 escenarios de generalización (margen máximo +0.020).
- Consenso supera al Transformer en 3 de 21 escenarios de generalización (margen máximo +0.026).
- Consenso con prior supera al Transformer en 4 de 21 escenarios de generalización (margen máximo +0.011).
- Compuerta del Transformer supera al Transformer en 15 de 21 escenarios de generalización (margen máximo +0.006).
- Peor escenario del ensamble ponderado − peor escenario del Transformer = +0.040.

## 9. Comparaciones entre pares (21 escenarios juntos)

### 9.1 Pares citados en la tesis (Tablas `tab:pairwise-stats` y `tab:pairwise-ens`)

| Comparación | $d$ | $\log_{10} p$ | KL sim. |
|---|---:|---:|---:|
| Transformer vs Q-Learning | 0.78 | -87.8 | 0.49 |
| Transformer vs DQN recurrente | 0.57 | -48.9 | 0.34 |
| Transformer vs DQN | 0.51 | -39.2 | 0.17 |
| Ensamble ponderado vs Consenso | 0.58 | -48.6 | 0.23 |
| Ensamble ponderado vs Compuerta del Transformer | 0.17 | -5.1 | 0.03 |

### 9.2 Matrices de pares — individuales

**$d$ de Cohen** (fila − columna; positivo = gana la fila):

| fila \ columna | QL-base | QL | DQL | DQN | RDQN | A2C | TRF |
|---|---:|---:|---:|---:|---:|---:|---:|
| QL-base | — | -0.21 | -0.28 | -0.52 | -0.39 | -0.47 | -0.90 |
| QL | 0.21 | — | -0.07 | -0.34 | -0.20 | -0.29 | -0.78 |
| DQL | 0.28 | 0.07 | — | -0.28 | -0.14 | -0.23 | -0.76 |
| DQN | 0.52 | 0.34 | 0.28 | — | 0.12 | 0.04 | -0.51 |
| RDQN | 0.39 | 0.20 | 0.14 | -0.12 | — | -0.08 | -0.57 |
| A2C | 0.47 | 0.29 | 0.23 | -0.04 | 0.08 | — | -0.49 |
| TRF | 0.90 | 0.78 | 0.76 | 0.51 | 0.57 | 0.49 | — |

**$\log_{10} p$ de Welch**:

| fila \ columna | QL-base | QL | DQL | DQN | RDQN | A2C | TRF |
|---|---:|---:|---:|---:|---:|---:|---:|
| QL-base | — | -7.6 | -13.0 | -41.5 | -24.1 | -33.9 | -111.4 |
| QL | -7.6 | — | -1.3 | -18.7 | -7.1 | -13.5 | -87.8 |
| DQL | -13.0 | -1.3 | — | -13.2 | -3.7 | -8.8 | -83.3 |
| DQN | -41.5 | -18.7 | -13.2 | — | -3.0 | -0.5 | -39.2 |
| RDQN | -24.1 | -7.1 | -3.7 | -3.0 | — | -1.5 | -48.9 |
| A2C | -33.9 | -13.5 | -8.8 | -0.5 | -1.5 | — | -37.4 |
| TRF | -111.4 | -87.8 | -83.3 | -39.2 | -48.9 | -37.4 | — |

**KL simetrizada entre histogramas de desempeño**:

| fila \ columna | QL-base | QL | DQL | DQN | RDQN | A2C | TRF |
|---|---:|---:|---:|---:|---:|---:|---:|
| QL-base | — | 0.094 | 0.105 | 0.369 | 0.203 | 0.185 | 0.978 |
| QL | 0.094 | — | 0.018 | 0.121 | 0.081 | 0.070 | 0.490 |
| DQL | 0.105 | 0.018 | — | 0.070 | 0.073 | 0.057 | 0.408 |
| DQN | 0.369 | 0.121 | 0.070 | — | 0.115 | 0.084 | 0.173 |
| RDQN | 0.203 | 0.081 | 0.073 | 0.115 | — | 0.044 | 0.339 |
| A2C | 0.185 | 0.070 | 0.057 | 0.084 | 0.044 | — | 0.199 |
| TRF | 0.978 | 0.490 | 0.408 | 0.173 | 0.339 | 0.199 | — |

### 9.3 Matrices de pares — ensambles

**$d$ de Cohen** (fila − columna; positivo = gana la fila):

| fila \ columna | ENS | VS | VA | CONS | CONS+P | GUARD |
|---|---:|---:|---:|---:|---:|---:|
| ENS | — | 0.59 | 0.60 | 0.58 | 0.36 | 0.17 |
| VS | -0.59 | — | 0.01 | -0.03 | -0.29 | -0.42 |
| VA | -0.60 | -0.01 | — | -0.04 | -0.31 | -0.43 |
| CONS | -0.58 | 0.03 | 0.04 | — | -0.27 | -0.40 |
| CONS+P | -0.36 | 0.29 | 0.31 | 0.27 | — | -0.16 |
| GUARD | -0.17 | 0.42 | 0.43 | 0.40 | 0.16 | — |

**$\log_{10} p$ de Welch**:

| fila \ columna | ENS | VS | VA | CONS | CONS+P | GUARD |
|---|---:|---:|---:|---:|---:|---:|
| ENS | — | -49.4 | -51.4 | -48.6 | -20.1 | -5.1 |
| VS | -49.4 | — | -0.1 | -0.3 | -13.5 | -26.2 |
| VA | -51.4 | -0.1 | — | -0.6 | -14.8 | -27.8 |
| CONS | -48.6 | -0.3 | -0.6 | — | -11.8 | -24.5 |
| CONS+P | -20.1 | -13.5 | -14.8 | -11.8 | — | -4.7 |
| GUARD | -5.1 | -26.2 | -27.8 | -24.5 | -4.7 | — |

**KL simetrizada entre histogramas de desempeño**:

| fila \ columna | ENS | VS | VA | CONS | CONS+P | GUARD |
|---|---:|---:|---:|---:|---:|---:|
| ENS | — | 0.306 | 0.303 | 0.230 | 0.066 | 0.025 |
| VS | 0.306 | — | 0.001 | 0.082 | 0.117 | 0.227 |
| VA | 0.303 | 0.001 | — | 0.084 | 0.119 | 0.225 |
| CONS | 0.230 | 0.082 | 0.084 | — | 0.107 | 0.128 |
| CONS+P | 0.066 | 0.117 | 0.119 | 0.107 | — | 0.030 |
| GUARD | 0.025 | 0.227 | 0.225 | 0.128 | 0.030 | — |

## 10. Detalle en la referencia (`sev_base`)

### 10.1 Media por bloque (8 bloques × 8 secuencias)

| Paquete | B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pes_base` | 0.840 | 0.875 | 0.870 | 0.873 | 0.875 | 0.888 | 0.883 | 0.861 |
| `pes_ql` | 0.872 | 0.850 | 0.876 | 0.891 | 0.919 | 0.879 | 0.906 | 0.898 |
| `pes_dql` | 0.890 | 0.887 | 0.904 | 0.890 | 0.898 | 0.895 | 0.903 | 0.904 |
| `pes_dqn` | 0.875 | 0.884 | 0.885 | 0.901 | 0.912 | 0.874 | 0.905 | 0.913 |
| `pes_rdqn` | 0.910 | 0.883 | 0.865 | 0.900 | 0.906 | 0.904 | 0.912 | 0.909 |
| `pes_a2c` | 0.876 | 0.876 | 0.882 | 0.892 | 0.897 | 0.874 | 0.896 | 0.905 |
| `pes_trf` | 0.926 | 0.912 | 0.902 | 0.938 | 0.953 | 0.920 | 0.932 | 0.935 |
| `pes_ens` | 0.943 | 0.926 | 0.911 | 0.935 | 0.945 | 0.944 | 0.952 | 0.942 |
| `pes_ens_sprb` | 0.932 | 0.907 | 0.881 | 0.911 | 0.922 | 0.907 | 0.933 | 0.922 |
| `pes_ens_accq` | 0.924 | 0.904 | 0.881 | 0.919 | 0.925 | 0.907 | 0.932 | 0.922 |
| `pes_ens_consensus` | 0.874 | 0.875 | 0.880 | 0.892 | 0.897 | 0.894 | 0.900 | 0.903 |
| `pes_ens_consensus_prior` | 0.914 | 0.893 | 0.888 | 0.931 | 0.924 | 0.924 | 0.928 | 0.932 |
| `pes_ens_trf_guard` | 0.929 | 0.912 | 0.902 | 0.938 | 0.953 | 0.920 | 0.935 | 0.935 |

### 10.2 Distribución de acciones en la referencia (fracción de pasos con cada asignación 0..10)

| Paquete | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Asignación media |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pes_base` | 0.272 | 0.075 | 0.033 | 0.039 | 0.025 | 0.036 | 0.061 | 0.031 | 0.014 | 0.181 | 0.233 | 5.19 |
| `pes_ql` | 0.281 | 0.039 | 0.042 | 0.028 | 0.017 | 0.053 | 0.022 | 0.081 | 0.192 | 0.097 | 0.150 | 5.14 |
| `pes_dql` | 0.303 | 0.003 | 0.019 | 0.025 | 0.039 | 0.019 | 0.100 | 0.114 | 0.122 | 0.164 | 0.092 | 5.14 |
| `pes_dqn` | 0.322 | 0.000 | 0.000 | 0.000 | 0.144 | 0.000 | 0.000 | 0.000 | 0.178 | 0.356 | 0.000 | 5.20 |
| `pes_rdqn` | 0.178 | 0.092 | 0.017 | 0.044 | 0.017 | 0.222 | 0.111 | 0.111 | 0.011 | 0.147 | 0.050 | 4.79 |
| `pes_a2c` | 0.322 | 0.000 | 0.000 | 0.144 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.533 | 0.000 | 5.23 |
| `pes_trf` | 0.289 | 0.022 | 0.019 | 0.000 | 0.033 | 0.033 | 0.042 | 0.258 | 0.086 | 0.139 | 0.078 | 5.14 |
| `pes_ens` | 0.192 | 0.033 | 0.069 | 0.014 | 0.044 | 0.206 | 0.078 | 0.053 | 0.136 | 0.108 | 0.067 | 4.99 |
| `pes_ens_sprb` | 0.194 | 0.081 | 0.008 | 0.083 | 0.006 | 0.194 | 0.053 | 0.089 | 0.061 | 0.158 | 0.072 | 4.92 |
| `pes_ens_accq` | 0.208 | 0.067 | 0.014 | 0.081 | 0.008 | 0.175 | 0.056 | 0.094 | 0.061 | 0.164 | 0.072 | 4.92 |
| `pes_ens_consensus` | 0.317 | 0.003 | 0.017 | 0.131 | 0.000 | 0.008 | 0.000 | 0.000 | 0.000 | 0.503 | 0.022 | 5.22 |
| `pes_ens_consensus_prior` | 0.200 | 0.094 | 0.033 | 0.039 | 0.006 | 0.208 | 0.025 | 0.000 | 0.056 | 0.289 | 0.050 | 5.04 |
| `pes_ens_trf_guard` | 0.294 | 0.017 | 0.017 | 0.000 | 0.036 | 0.033 | 0.042 | 0.258 | 0.086 | 0.139 | 0.078 | 5.14 |

_Las distribuciones incluyen los pasos sin recursos (acción forzada 0)._

### 10.3 Registro de confianza del Transformer (Apéndice `ap:trf-conf`)

- 258 decisiones con recursos disponibles en la referencia; confianza media 0.116. Q no factibles -> valor muy negativo; desplazamiento a valores no negativos; normalización; confianza = 1 - H_norm. NO comparable con la confianza de los ensambles (refleja sobre todo cuántas acciones quedan factibles).

## 11. Advertencias metodológicas (deben respetarse al redactar)

- Una única semilla (42) por escenario y un único conjunto de 64 secuencias de referencia: no hay intervalos de confianza del ordenamiento; los p de Welch no son independientes ni corregidos por comparaciones múltiples.
- Hiperparámetros de modelos y ensambles ajustados con las mismas 64 secuencias de la referencia: la referencia puede ser optimista; la generalización está menos expuesta a ese sesgo.
- La media de generalización incluye los 3 escenarios estructurales (idénticos a la referencia) y escenarios más fáciles (p. ej. len_all_short, joint_low_short); leerla junto con el peor escenario.
- Entrenamiento sólo con severidades 2..8: filas S = 0, 1, 9 de las tablas Q quedan con valores iniciales aleatorios; parte de la caída tabular refleja falta de cobertura, no sólo capacidad de generalizar.
- Con severidades 10..12 una política que asigna mucho puede acercarse al óptimo: un rbar alto en sev_extrapolate_high no prueba por sí solo mejor generalización (A2C también supera allí su referencia).
- Arquitecturas de DQN recurrente y Transformer elegidas ad hoc (no optimizadas): las conclusiones se refieren a los modelos finales evaluados; los resultados no identifican la causa de la ventaja del Transformer.
- La confianza 1 - H_norm se calcula sobre softmax de valores Q (no probabilidades aprendidas, salvo A2C): es heurística, no calibrada; no se comprobó que sea mayor en decisiones acertadas.
- El registro de confianza del Transformer (media 0,116) usa otro cálculo que los ensambles: no comparar con tau_g ni con otros umbrales.
- La ventaja del ensamble ponderado es compatible con el prior de severidad y tau = 15, no con la ponderación por confianza; con un único ensamble ganador no se puede separar el aporte de cada componente.
- "Distancia de Lieber" en documentos previos = divergencia de Kullback-Leibler.
- h2/ es una línea suspendida: no citarla como trabajo realizado. La tesis no usa datos humanos (ds004477 sólo como contexto del entorno PES).

## 12. Puntos de atención detectados en el borrador LaTeX actual

Contrastados contra los datos de este archivo (2026-09-24). Corregirlos cuando se edite el archivo afectado.

| # | Severidad | Archivo | Lugar | Hallazgo | Corrección sugerida |
|---:|---|---|---|---|---|
| 1 | mayor | `04Materials.tex` | subsubsection Voto suave (tras eq:ens-sprb) | El texto dice "ajusta la temperatura de la softmax (tau = 0,1044)", pero tab:ens-params y inputs/best_params.json dan tau = 1,219 (temperature = 1.2192). | Reemplazar $\tau = 0{,}1044$ por $\tau = 1{,}219$. |
| 2 | mayor | `04Materials.tex` | subsubsection Consenso con prior (último párrafo) | "Es la única variante en la que se optimizó el peso del actor de A2C; en las demás vale 0,10" es falso: los cinco ensambles optimizables optimizan weight_a2c (rango 0-3) y sus best_params lo contienen (0,0070; 0,174; 1,881; 0,432; 0,0674), como ya muestra tab:ens-params. | Eliminar la oración o reemplazarla por una que diga que los cinco optimizan w_A2C. |
| 3 | menor | `05Results.tex` | sec:res-ensembles, párrafo previo a sec:res-freq | "el voto suave y el voto por acción pierden más con la severidad": por familia pierden algo más con la longitud (degradación media VS 0,016 vs 0,014 en severidad; VA 0,016 vs 0,015); lo que sí es de severidad es su peor escenario (sev_extrapolate_high). | Precisar que su peor escenario es la extrapolación de severidad. |
| 4 | menor | `06Discussion.tex` | sec:ens-best, viñeta voto suave / voto por acción | "quedan por debajo de ambos y sólo superan al DQN recurrente" es ambiguo: en sev_extrapolate_high VS 0,860 y VA 0,859 quedan por debajo de DQN 0,890, A2C 0,929 y Transformer 0,996, y por encima del DQN recurrente 0,831 (y de los tabulares). | Por ejemplo: "entre sus miembros, sólo superan al DQN recurrente". |
| 5 | observación | `00Abstract.tex / 00Abstract_en.tex` | párrafo de resultados | "Los peores resultados fueron los de los métodos tabulares ... (0,76 y 0,79)" se refiere a Q-Learning y Double Q-Learning; Q-Learning base llega a 0,627 (peor celda del estudio). | Explicitar "los métodos tabulares optimizados" o mencionar Q-Learning base. |
| 6 | observación | `04Materials.tex` | Optimización de hiperparámetros | Nº de ensayos (DQN 47, A2C 100, DQL 86/100 podados) no verificable con estos archivos de contexto. | Conservar tal cual salvo indicación del autor. |
| 7 | observación | `audit.py` | criterio 6 (cobertura de paquetes) | Pasa sólo gracias a los nombres de archivo PES_<PKG>_results.png (\texttt{pes\_base} no contiene "pes_base"). Si se eliminan las figuras por modelo, el criterio falla. | Mantener esas figuras o mencionar los paquetes con \verb|pes_base| (el criterio de idioma ignora \verb). |

## 13. Mapa del documento LaTeX

- Estado del último `audit.py` (2026-09-24): Compilación OK (59 páginas), 21 figuras y 13 tablas con label, 88 referencias, 24 imágenes, sin refs rotas, labels duplicados, imágenes huérfanas, citas faltantes, palabras inglesas ni TODO; único aviso: Acknowledgement.tex huérfano.
- Estructura en el repositorio (en el chat los archivos están planos):
  - `writings/00_Main/`: `Main.tex`, `References.bib`, `apa.bst`, `IEEEtran.cls`, `mPES_citation.bib`, `Lakshminarayanan2017.bib`
  - `writings/01_Chapters/`: `000NHH-Frontpage.tex`, `00Abstract.tex`, `00Abstract_en.tex`, `01Introduction.tex`, `02Background.tex`, `03StateOfTheArt.tex`, `04Materials.tex`, `05Results.tex`, `06Discussion.tex`, `07Conclusion.tex`, `Appendix.tex`, `Acknowledgement.tex (huérfano: no incluido por Main.tex)`
  - `writings/02_Images/frontpage/`: `LOGO-ITBA.jpg`
  - `writings/02_Images/baseline/`: `random_player_sequence_performance.png`, `random_player_normalised_performance.png`
  - `writings/02_Images/per_model/`: `PES_A2C_results.png`, `PES_BASE_results.png`, `PES_DQL_results.png`, `PES_DQN_results.png`, `PES_ENS_results.png`, `PES_QL_results.png`, `PES_RDQN_results.png`, `PES_TRF_results.png`
  - `writings/02_Images/individual/`: `ind_01_desempeno_por_escenario.png`, `ind_03_welch_logp_por_escenario.png`, `ind_04_kl_acciones_por_escenario.png`, `ind_05_curvas_por_familia.png`, `ind_13_pares_cohen_d.png`
  - `writings/02_Images/ensemble/`: `ens_01_desempeno_por_escenario.png`, `ens_03_welch_logp_por_escenario.png`, `ens_04_kl_acciones_por_escenario.png`, `ens_06_curvas_extrapolacion.png`, `ens_07_cohen_d_por_escenario.png`, `ens_12_pares_welch_logp.png`, `ens_13_pares_cohen_d.png`
  - `writings/02_Images/agent_internals/`: `trf_agent_confidences.png`
  - `writings/audit/`: `audit.py`, `AUDIT.md`
  - `writings/auxiliar/scripts/`: `sync_figures.py`, `ensemble_decisions.py`, `rebuild_thesis.py`

| Archivo | Sección (definida en Main.tex) | Etiquetas |
|---|---|---|
| `000NHH-Frontpage.tex` | Portada | — |
| `00Abstract.tex` | Resumen (español) | — |
| `00Abstract_en.tex` | Abstract (inglés, dentro de otherlanguage{english}) | — |
| `01Introduction.tex` | 1 Introducción | `sec:motivacion`, `sec:problema`, `sec:hypothesis` |
| `02Background.tex` | 2 Marco Teórico | `sec:background`, `sec:ensembles-bg`, `sec:stats-bg`, `eq:return`, `eq:bellman-opt`, `eq:qlearning`, `eq:double-q`, `eq:dqn-loss`, `eq:ddqn-target`, `eq:lstm`, `eq:pg`, `eq:gae`, `eq:causal-attention`, `eq:shannon`, `eq:shannon-norm`, `eq:ens-softmax`, `eq:soft-voting`, `eq:hard-voting`, `eq:cohen`, `eq:welch`, `eq:kl` |
| `03StateOfTheArt.tex` | 3 Estado de la Cuestión | `sec:soa`, `sec:soa-trf` |
| `04Materials.tex` | 4 Materiales y Métodos | `sec:methods`, `sec:env-dyn`, `sec:metric`, `sec:tabular`, `sec:deep`, `sec:ens-methods`, `sec:scenario-catalogue`, `eq:state-space`, `eq:transicion`, `eq:reward`, `eq:raw-severity-materials`, `eq:normalised-severity-materials`, `eq:tabular-update-materials`, `eq:dql-epsilon-materials`, `eq:dql-pbrs-materials`, `eq:ens-soft`, `eq:ens-prior`, `eq:ens-action`, `eq:ens-sprb`, `eq:ens-accq-vote`, `eq:ens-accq-action`, `eq:ens-consensus`, `eq:ens-guard`, `fig:baseline-random`, `fig:baseline-random-raw`, `fig:baseline-random-normalised`, `tab:packages`, `tab:tabular-hparams`, `tab:deep-hparams`, `tab:a2c-hparams`, `tab:ens-summary`, `tab:ens-params`, `tab:scenarios` |
| `05Results.tex` | 5 Resultados | `sec:results`, `sec:res-individual`, `sec:per-model`, `sec:per-seq-extended`, `sec:res-ensembles`, `sec:res-freq`, `eq:severity-reduction`, `tab:global-mean`, `tab:pairwise-stats`, `tab:ensemble-stress`, `tab:ens-freq`, `tab:pairwise-ens`, `fig:heatmap-global`, `fig:c-base`, `fig:c-ql`, `fig:c-dql`, `fig:c-dqn`, `fig:c-rdqn`, `fig:c-a2c`, `fig:c-trf`, `fig:extra-sev-skew`, `fig:heatmap-ens`, `fig:ensemble-extrapolation`, `fig:c-ens` |
| `06Discussion.tex` | 6 Discusión | `sec:discussion`, `sec:disc-individual`, `sec:ens-best` |
| `07Conclusion.tex` | 7 Conclusiones | `sec:conclusion`, `sec:limitations`, `sec:future` |
| `Appendix.tex` | Apéndice | `ap:repro`, `ap:orchestrator`, `ap:stat-maps`, `ap:trf-conf`, `tab:repro-commands`, `fig:heatmap-welch`, `fig:heatmap-kl`, `fig:pairwise-cohen`, `fig:ensemble-statistical-heatmaps`, `fig:ensemble-pairwise`, `fig:trf-internals` |

**Figuras de la tesis** (label → archivo en `02_Images/<carpeta>/`):

- `fig:baseline-random` → baseline/random_player_sequence_performance.png + baseline/random_player_normalised_performance.png (agente aleatorio)
- `fig:heatmap-global` → individual/ind_01_desempeno_por_escenario.png (media por modelo y escenario)
- `fig:c-base .. fig:c-trf` → per_model/PES_<PKG>_results.png (64 secuencias de la referencia, distribución, estadísticos por bloque, resumen)
- `fig:extra-sev-skew` → individual/ind_05_curvas_por_familia.png (curvas ordenadas en sev_bimodal, sev_gauss_high, sev_beta_highskew, len_poisson, len_extrapolate_long, joint_high_long; trazo grueso = Transformer)
- `fig:heatmap-ens` → ensemble/ens_01_desempeno_por_escenario.png
- `fig:ensemble-extrapolation` → ensemble/ens_06_curvas_extrapolacion.png (3 escenarios fuera de rango; trazo grueso = ensamble ponderado)
- `fig:c-ens` → per_model/PES_ENS_results.png
- `fig:heatmap-welch` → individual/ind_03_welch_logp_por_escenario.png
- `fig:heatmap-kl` → individual/ind_04_kl_acciones_por_escenario.png
- `fig:pairwise-cohen` → individual/ind_13_pares_cohen_d.png
- `fig:ensemble-statistical-heatmaps` → ensemble/ens_03 + ens_07 + ens_04
- `fig:ensemble-pairwise` → ensemble/ens_13 + ens_12
- `fig:trf-internals` → agent_internals/trf_agent_confidences.png

**Figuras generadas en `h1/general/results/` que la tesis NO usa** (no están en `02_Images/`):

- individual: `02_degradacion_por_escenario`, `06_curvas_estresores_universales`, `07_cohen_d_por_escenario`, `08_ranking_desempeno`, `09_degradacion_por_familia`, `10_desempeno_vs_estabilidad`, `11_perfiles_generalizacion`, `12_pares_welch_logp`, `14_pares_kl`
- ensemble: `02_degradacion_por_escenario`, `05_curvas_por_familia`, `08_ranking_desempeno`, `09_degradacion_por_familia`, `10_desempeno_vs_estabilidad`, `11_perfiles_generalizacion`, `14_pares_kl`
- agent_internals: `trf_agent_cumulative_performance`, `trf_agent_normalised_performance`, `trf_agent_remapped_confidences`
- Cómo incluirlas: Agregar el stem a SUITE_FIGURES en writings/auxiliar/scripts/sync_figures.py y ejecutarlo (sync_figures borra de 02_Images todo lo que no esté en su lista); nombre destino ind_<stem>.png o ens_<stem>.png.

**Claves bibliográficas disponibles en `References.bib`** (únicas citables sin agregar entradas): `BCINE2022`, `Towers2024`, `SuttonBarto2018`, `Watkins1992`, `Hasselt2010`, `Ng1999`, `Mnih2015`, `Lin1992`, `Kingma2015`, `Hasselt2016`, `Huber1964`, `Hochreiter1997`, `Hausknecht2015`, `Williams1992`, `Mnih2016`, `Vaswani2017`, `Parisotto2020`, `Chen2021`, `Lakshminarayanan2017`, `Bergstra2011`, `Akiba2019`, `Cohen1988`, `Welch1947`, `Kuhl2021`, `ds004477`, `Nijjar2025`, `Henderson2018`, `Cobbe2020`, `Packer2018`, `Kirk2023`, `mPES2026`, `Glorot2010`, `Shannon1948`, `Schulman2016`, `Wiering2008`, `Parisi2021`.

## 14. Reproducción

- Entorno: win_mpes_env\Scripts\Activate.ps1 desde la raíz; comandos desde h1/; VIRTUAL_ENV, PYTHONIOENCODING=utf-8, TF_ENABLE_ONEDNN_OPTS=0.
- Benchmark (desde `h1/`): `python -m general.scripts.benchmark run --suite both`; `python -m general.scripts.analysis`; `python -m general.scripts.figures`; `python -m general.scripts.random_baseline`; `python -m general.scripts.agent_internals`
- Tesis (desde la raíz): `cd writings`; `python audit\audit.py`; `python audit\audit.py --no-tex`; `python auxiliar\scripts\sync_figures.py`; `python auxiliar\scripts\ensemble_decisions.py`
