# mPES — Síntesis de métricas, salidas y resultados (contexto para la tesis)

> Generado por `writings/auxiliar/scripts/build_results_context.py` — no editar a mano. Fecha (UTC): 2026-09-29T14:06:35Z; commit `3be74eb` (con cambios sin confirmar). Fuentes: `h1/general/results/` (`summary.json`, `comparison_metrics.json`, `matrices/`, `cells/`, `heldout/`), las decisiones del replay de `writings/auxiliar/scripts/ensemble_decisions.py`, los `inputs/best_params.json`, `inputs/*_BAYESIAN_OPT/` y `config/CONFIG.py` de cada paquete, y los `.tex` de `writings/`. Es la **fuente de verdad numérica** para redactar la tesis. Los números usan punto decimal; en LaTeX se escriben con coma (`$0{,}927$`). El archivo `mpes_resultados.json` contiene los mismos datos con más precisión y por celda. El texto redactado a mano vive en `writings/auxiliar/claude_context/mpes_resultados_notas.md`.

## 0. Resumen ejecutivo (hallazgos verificados)

1. **Transformer (`pes_trf`) = mejor modelo individual.** Referencia 0.927 (σ 0.045, la menor entre individuales), generalización 0.930 (resto de optimizados entre 0.871 y 0.899); mayor media en 16 de 22 escenarios. Reduce la distancia al óptimo de Q-Learning en 36 % en la referencia. d = 0.79 vs Q-Learning, 0.57 vs DQN recurrente, 0.50 vs DQN. → Respalda **H1**.
2. **No es el que menos pierde en su peor caso**: peor escenario `len_extrapolate_long` 0.860 (degradación 0.068) frente a 0.053 de DQN y 0.062 de A2C.
3. **Tabulares colapsan con severidad fuera de rango** (`sev_extrapolate_high`): Q-Learning base 0.627 (degradación 0.244, peor celda del estudio), Q-Learning 0.762, Double Q-Learning 0.785: la tabla recorta S a 9, fila nunca entrenada.
4. **Las redes mantienen el desempeño en ese escenario**: Transformer 0.996 (y 0.997 en la conjunta), A2C 0.929, DQN 0.890; excepción: DQN recurrente 0.831 (su peor escenario). Advertencia: con S = 10..12 asignar mucho acerca al óptimo.
5. **Secuencias largas (`len_extrapolate_long`) = peor escenario de DQN, A2C, Transformer y de 4 ensambles.** Allí el mejor individual es el DQN recurrente (0.889 vs 0.860 del Transformer): único escenario fuera de rango en que el Transformer no es primero.
6. **Escenarios estructurales** reproducen exactamente la referencia en 13 de los 13 modelos (control correcto).
7. **Ensamble ponderado (`pes_ens`) = único sistema que supera al Transformer**: referencia 0.937 (σ 0.035, la menor de todo el estudio), generalización 0.939, peor escenario 0.900 (+0.040 sobre el del Transformer), mayor degradación 0.037 (la menor). Δrel vs Q-Learning 45 %. Óptimo en todas las secuencias de `sev_extrapolate_high` y `joint_extrap_both` (1,000; σ 0). d = 0.58 vs consenso y 0.17 vs compuerta; frente al Transformer en generalización d = 0.19 (despreciable, p = 1.2e-06).
8. **Pero su acción final difiere de la del Transformer en el 61,2 % (referencia) / 47,4 % (generalización)** de las decisiones; el prior de severidad cambia la acción votada en el 40,9 % / 32,1 %; la cota casi nunca actúa. Con τ = 15 las distribuciones de los miembros son casi planas. La mejora es compatible con prior + temperatura, no con la ponderación por confianza.
9. **Compuerta del Transformer ≈ Transformer**: generalización 0.931 vs 0.930; sigue al Transformer en el 96,9 % / 94,9 % de las decisiones.
10. **Voto suave y voto por acción**: generalización más baja (0.902 / 0.901), peor escenario `sev_extrapolate_high` (0.860 / 0.859), con 41 % / 42 % de secuencias bajo 0,8. Sus pesos dan más peso al DQN recurrente que al Transformer.
11. **Consenso** tiene la referencia más baja de los ensambles (0.889); la variante con prior mejora referencia (0.918) y peor escenario (0.826 → 0.870), aunque su prior sólo cambia el 4,5 % / 4,6 % de las decisiones.
12. **Agente aleatorio**: 0.670 (cota inferior de referencia).
13. **Réplicas fuera de muestra** (§10bis, Tabla `tab:heldout`): en 320 secuencias nuevas sorteadas de la distribución de la referencia ningún modelo pierde desempeño apreciable: caída (ref. − réplicas) de -0.004 a +0.011 (|d| ≤ 0.15, p ≥ 0.29); en los 11 optimizados con Optuna entre -0.004 y +0.005; la mayor es la de Q-Learning base, que no se optimizó. Spearman ρ = 0.97 entre medias. El ensamble ponderado supera al Transformer por 0.010 (t pareada p < 10⁻⁷) y en 5 de las 5 réplicas. → La selección sobre las 64 secuencias no infló la referencia.

## 1. Definiciones

- **Desempeño normalizado** por secuencia: $\bar r = (S_{peor} - S_{agente}) / (S_{peor} - S_{mejor})$. $0$ = no asignar recursos; $1$ = asignación óptima ($S_{mejor}$ exacto por programación dinámica, mochila acotada con asignaciones 0..10 por ciudad que suman ≤ 30). $\bar r \le 1$. No es una probabilidad.
- **Referencia** = escenario `sev_base` (distribución de entrenamiento), $n = 64$ secuencias (8 bloques × 8), 360 pasos.
- **Generalización** = media de las medias de los **21** escenarios distintos de la referencia, sin las 5 réplicas fuera de muestra (incluye 3 estructurales que reproducen exactamente la referencia y escenarios más fáciles, p. ej. secuencias cortas).
- **Degradación** de una celda = $\bar r_{ref} - \bar r_{esc}$ (positiva = pérdida; negativa = mejora). **Degradación media** = promedio con signo sobre los 21 escenarios. **Mayor degradación** = máximo de esas 21.
- **Por celda frente a la propia referencia**: $d$ de Cohen (positivo = mejor en el escenario), $\log_{10} p$ de Welch (bilateral) y KL de acciones (desplazamiento de la política sobre las 11 acciones).
- **Entre pares de modelos** (mismo grupo): se juntan las secuencias de los 21 escenarios de generalización (sin `sev_base`); $d > 0$ ⇒ gana el modelo de la fila; KL simetrizada entre histogramas de desempeño (20 bins en [0, 1], $\varepsilon = 10^{-9}$).
- **Δrel vs Q-Learning** (fórmula en línea de `sec:res-ensembles`): $(\bar r_m - \bar r_{ql}) / (1 - \bar r_{ql})$, reducción relativa de la distancia al óptimo.
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

- Agente que en cada paso elige a ~ U{0..10}, recortada al presupuesto restante (30), sobre las 64 secuencias de referencia; semilla 42. Calculado con general.scripts.random_baseline.run_random_player (sin escribir figuras).
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
- Transformer: W = 6, proyección a d_model = 32 + vector de posición fijo (Glorot, no entrenado), 2 bloques Post-LN sin las compuertas de Parisotto (atención causal 4 cabezas de dim. 16 + FFN 64, residual), sin dropout, última posición → Dense(32, ReLU) → 11 Q.
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
| `pes_ens` | Ensamble ponderado | Voto suave ponderado por (0,1 + c_k) sobre DQN, DQN recurrente y Transformer; factor 0,3 a a=0 si R>0; mezcla con prior de severidad gaussiano; argmax; cota de seguridad floor(S/2) si S>=6. | dqn, rdqn, trf | tau = 15.0, w_dqn = 0.18, w_rdqn = 0.9, w_trf = 5.0, peso_normalizado_trf = 0.822, w_prior = 0.17, sigma_prior = 3.0 | fijados a mano sobre la referencia (commit d2f0c49, sin búsqueda sistemática); justificados a posteriori en sec:res-posthoc (sensibilidad, óptimo y regla fija) |
| `pes_ens_sprb` | Voto suave | Voto suave: promedio de p_k con pesos efectivos w_k·c_k^rho; argmax. | dqn, rdqn, trf, a2c | rho = 2.68075, tau = 1.219194, w_a2c = 0.007027, w_dqn = 0.270032, w_rdqn = 1.759607, w_trf = 0.895639 | mejor ensayo de Optuna/TPE sobre las 64 secuencias de referencia (inputs/best_params.json) |
| `pes_ens_accq` | Voto por acción | Voto por acción: cada miembro vota su argmax con peso w_k·c_k^rho; desempate por suma de Q estandarizados. | dqn, rdqn, trf, a2c | rho = 2.1972, tau = 1.0, w_a2c = 0.174251, w_dqn = 0.467984, w_rdqn = 2.598528, w_trf = 1.803345 | mejor ensayo de Optuna/TPE sobre las 64 secuencias de referencia (inputs/best_params.json) |
| `pes_ens_consensus` | Consenso | Voto ponderado + beta_a·(confianza de quienes coinciden) - beta_d·(confianza de quienes discrepan) + sum_k Qhat_k(a)·c_k; argmax. | dqn, rdqn, trf, a2c | beta_a = 2.074141, beta_d = 0.121143, rho = 0.275376, tau = 1.0, w_a2c = 1.880651, w_dqn = 1.531684, w_rdqn = 1.931138, w_trf = 2.503556 | mejor ensayo de Optuna/TPE sobre las 64 secuencias de referencia (inputs/best_params.json) |
| `pes_ens_consensus_prior` | Consenso con prior | Como consenso pero con suma de distribuciones; factor 0,3 a a=0 si R>0; mezcla con prior de severidad; cota de seguridad floor(S/2) si S>=6. Confianza c_k = 1 − H_norm(p_k), igual que el resto (corregida el 2026-09-28: antes se aplicaba una segunda softmax a p_k). | dqn, rdqn, trf, a2c | beta_a = 0.2437, beta_d = 0.114644, rho = 1.049975, sigma_prior = 0.909726, tau = 1.0, w_a2c = 0.336563, w_dqn = 2.12233, w_prior = 0.177653, w_rdqn = 2.722707, w_trf = 2.2768 | mejor ensayo de Optuna/TPE sobre las 64 secuencias de referencia (inputs/best_params.json) |
| `pes_ens_trf_guard` | Compuerta del Transformer | Compuerta logística g = sigmoid(kappa_g·(c_trf - tau_g)); si g>=0,5 ejecuta la acción del Transformer, si no un voto suave de respaldo. | dqn, rdqn, trf, a2c | kappa_g = 2.397349, rho = 1.451819, tau = 1.0, tau_g = 0.222186, w_a2c = 0.067448, w_dqn = 0.83668, w_rdqn = 2.561269, w_trf = 2.601263 | mejor ensayo de Optuna/TPE sobre las 64 secuencias de referencia (inputs/best_params.json) |

- Todos: $p_k$ = softmax de temperatura τ de los Q (o salida del actor de A2C), renormalizada sobre acciones factibles; confianza $c_k = 1 - H_{norm}(p_k)$; peso efectivo $\tilde w_k = w_k c_k^{\rho}$ (salvo `pes_ens`, que usa $w_k(0{,}1 + c_k)$).
- `pes_ens`: pesos base 0,18 / 0,90 / 5,0 (DQN / DQN recurrente / Transformer) → el Transformer tiene el 82 % del peso normalizado; A2C deshabilitado; recorta S a 9.
- En los 5 ensambles optimizados, el score de Optuna **coincide** con la media del benchmark en la referencia; los 5 optimizan también `w_a2c` (rango 0–3). Sólo el voto suave optimiza τ (1,219); los demás usan τ = 1.
- **Justificación a posteriori de `pes_ens`** (sec:res-posthoc): τ, w_π, σ y el peso del Transformer tienen su máximo en el valor fijo en la referencia y en las réplicas (máximos angostos: un paso de la grilla cuesta 0,002–0,008); η, el término 0,1 y la cota, en meseta. En las réplicas, sin prior (w_π = 0) supera al Transformer por 0,004; con prior y τ = 1, por 0,001; con ambos, por 0,010 (± 0,002 EE pareado). Supera al Transformer por más de 2 EE con τ 2–30, w_π 0–0,25, σ 3–5, peso TRF 2,5–10.
- **Óptimo** (DP de S_mejor con reconstrucción, 384 secuencias ref. + réplicas; 6 % de ciudades con empates, todas S ≤ 5): con presupuesto nunca asigna 0 y asigna S + 0,7 a S + 1,9; gaussiana centrada en S: σ = 2,7 (≈ 3); centro libre: S + 2,2, σ = 1,4. La cota cambia 2 de 1 726 decisiones; η, el 1,7 % (0,0004 por secuencia); el prior, el 37 % (+0,012 por secuencia).
- **Regla fija sin modelo** a = min(S + k, R) (tab:fixed-rule), ref. / gen. / réplicas: k = 0 0,782 / 0,805 / 0,787; k = 1 0,900 / 0,907 / 0,906; k = 2 0,940 / 0,937 / 0,943; k = 3 0,939 / 0,940 / 0,941 (Transformer 0,927 / 0,930 / 0,929; pes_ens 0,937 / 0,939 / 0,939). Óptimo (DP) = 1 en todas las condiciones. k = 2 frente a los 13 modelos, diferencia en generalización y escenarios de gen. en que el modelo la supera: pes_ens +0,002 (4/21; réplicas −0,004, 1/5), compuerta −0,007 (3), Transformer −0,008 (3), consenso con prior −0,018, consenso −0,033, voto suave −0,035, voto por acción −0,036, DQN −0,039, A2C −0,041, DQN recurrente −0,049, Double QL −0,060, QL −0,066, QL base −0,087 (2 c/u); ningún modelo la supera en la referencia. Los 13 la superan sólo en len_all_short (regla 0,845) y joint_low_short (0,866): le sobra presupuesto. k = 2 supera al Transformer en 18/21 y 5/5 réplicas. Dinámica con a = S + k constante: S_n = max(0, S + k(1 − β^n)) (eq:rule-dynamics); el equilibrio a = S se deduce de la dinámica; k = 2 se eligió a la vista del óptimo con información completa (también es el mejor k en la referencia). **Lectura obligatoria al redactar** (sec:disc-rule): no invalida H1/H2 (comparan modelos entre sí) ni implica que los agentes no aprendieran (sin el óptimo, k = 0 rinde 0,78; los agentes 0,85–0,93 sólo con la recompensa); sí implica que la política óptima de mPES es simple y que en este entorno la ventaja práctica de los modelos frente a una heurística calibrada es nula.

### 5.2 Resumen de desempeño (ordenado por generalización)

| Paquete | Nombre en el texto | Ref. | σ ref. | Gen. (21) | Deg. media | Peor escenario (media) | Mayor deg. | Δrel vs QL ref. |
|---|---|---:|---:|---:|---:|---|---:|---:|
| `pes_ens` | Ensamble ponderado | 0.937 | 0.035 | 0.939 | -0.0021 | `len_extrapolate_long` (0.900) | 0.037 | 44.7 % |
| `pes_ens_trf_guard` | Compuerta del Transformer | 0.928 | 0.046 | 0.931 | -0.0026 | `len_extrapolate_long` (0.861) | 0.067 | 36.4 % |
| `pes_ens_consensus_prior` | Consenso con prior | 0.918 | 0.041 | 0.919 | -0.0017 | `len_extrapolate_long` (0.870) | 0.047 | 27.4 % |
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
| `pes_ens_consensus_prior` | -0.0034 | +0.0152 | -0.0200 | 0.0000 | 0.920 | 0.932 |
| `pes_ens_trf_guard` | -0.0067 | +0.0223 | -0.0267 | 0.0000 | 0.931 | 0.951 |

Promedio del grupo de ensambles: severidad -0.0012, longitud +0.0152, conjunta -0.0136, estructural 0.0000.

### 5.4 Frecuencia con que las reglas fijas cambian la decisión (Tabla `tab:ens-freq`)

_Tesis, Tabla tab:ens-freq; replay de writings/auxiliar/scripts/ensemble_decisions.py --output (h1/general/results/ensemble/ens_decisions.json). Diferencia máxima entre la media del replay y la del benchmark actual: 5.8e-06 (generalización y referencia), 1.1e-05 (réplicas). Unidad: % de decisiones con recursos disponibles (R > 0)._

| Ensamble | Evento | Referencia | Generalización |
|---|---|---:|---:|
| Compuerta del Transformer | Sigue al Transformer (g ≥ 0,5) | 96.9 | 94.9 |
| Compuerta del Transformer | Acción final distinta de la del Transformer | 0.8 | 2.7 |
| Consenso con prior | El prior cambia la acción votada | 4.5 | 4.6 |
| Consenso con prior | La cota cambia la acción | 0.3 | 1.5 |
| Consenso con prior | Acción final distinta de la del Transformer | 62.4 | 58.4 |
| Ensamble ponderado | El prior cambia la acción votada | 40.9 | 32.1 |
| Ensamble ponderado | La cota cambia la acción | 0.0 | 0.3 |
| Ensamble ponderado | Acción final distinta de la del Transformer | 61.2 | 47.4 |

- Con tau = 15 las distribuciones de los tres miembros quedan casi planas: en la referencia la acción más probable de cada uno tiene en promedio probabilidad 0,19 y supera a la segunda por 0,02-0,03.
- El voto (antes del prior) ya difiere de la acción del Transformer en el 31 % de las decisiones de la referencia.
- En sev_extrapolate_high y joint_extrap_both todas las decisiones de pes_ens coinciden con las de su miembro Transformer (que recibe la severidad recortada a 9).
- Réplicas fuera de muestra (`heldout_s1`..`s5`, no incluidas en "Generalización"; replay = benchmark con diferencia máxima 1.1e-05): compuerta sigue al Transformer 94.6 % y difiere 2.8 %; consenso con prior: prior 3.7 %, cota 0.5 %, acción distinta del Transformer 63.4 %; ensamble ponderado: prior 36.4 %, cota 0.1 %, distinta 50.5 %. Quedan dentro del rango referencia–generalización en 4 de 8 eventos; fuera: Compuerta del Transformer: sigue al Transformer (g ≥ 0,5) 94.6 frente a 96.9–94.9; Compuerta del Transformer: acción final distinta de la del Transformer 2.8 frente a 0.8–2.7; Consenso con prior: el prior cambia la acción votada 3.7 frente a 4.5–4.6; Consenso con prior: acción final distinta de la del Transformer 63.4 frente a 62.4–58.4. La tesis no agrega una columna para las réplicas.

## 6. Catálogo de escenarios (1 referencia + 21 de generalización + 5 réplicas fuera de muestra)

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
| `struct_more_total` | estructural | — | 8 bloques × 8 secuencias = 64: las mismas secuencias que la referencia (control; originalmente 8 × 16 = 128, que sólo repetía las 64 secuencias y que los evaluadores de Optuna de los ensambles recortaban a 64). | 64 | 360 |  |
| `heldout_s1` | fuera de muestra | réplica fuera de muestra | Réplica 1 de la referencia: 8 × 8 secuencias nuevas; longitud y severidad inicial sorteadas de forma independiente (i.i.d.) con las frecuencias empíricas de la referencia; semilla 43. Excluida de la generalización. | 64 | 340 |  |
| `heldout_s2` | fuera de muestra | réplica fuera de muestra | Réplica 2 de la referencia: 8 × 8 secuencias nuevas; longitud y severidad inicial sorteadas de forma independiente (i.i.d.) con las frecuencias empíricas de la referencia; semilla 44. Excluida de la generalización. | 64 | 374 |  |
| `heldout_s3` | fuera de muestra | réplica fuera de muestra | Réplica 3 de la referencia: 8 × 8 secuencias nuevas; longitud y severidad inicial sorteadas de forma independiente (i.i.d.) con las frecuencias empíricas de la referencia; semilla 45. Excluida de la generalización. | 64 | 369 |  |
| `heldout_s4` | fuera de muestra | réplica fuera de muestra | Réplica 4 de la referencia: 8 × 8 secuencias nuevas; longitud y severidad inicial sorteadas de forma independiente (i.i.d.) con las frecuencias empíricas de la referencia; semilla 46. Excluida de la generalización. | 64 | 346 |  |
| `heldout_s5` | fuera de muestra | réplica fuera de muestra | Réplica 5 de la referencia: 8 × 8 secuencias nuevas; longitud y severidad inicial sorteadas de forma independiente (i.i.d.) con las frecuencias empíricas de la referencia; semilla 47. Excluida de la generalización. | 64 | 389 |  |

- Familias: severidad 9 · longitud 5 · conjunta 4 · estructural 3 (+ 5 réplicas fuera de muestra, fuera de todos los agregados; ver §10bis). Se descartaron escenarios de severidad constante (S_peor = S_mejor ⇒ métrica indefinida).
- Los 8 escenarios de severidad dentro de rango incluyen S = 0, 1 o 9, que están en el espacio de estados pero no aparecen en el entrenamiento (2..8).

## 7. Matrices modelo × escenario

Leyenda de columnas: QL-base = `pes_base` (Q-Learning base), QL = `pes_ql` (Q-Learning), DQL = `pes_dql` (Double Q-Learning), DQN = `pes_dqn` (DQN), RDQN = `pes_rdqn` (DQN recurrente), A2C = `pes_a2c` (A2C), TRF = `pes_trf` (Transformer), ENS = `pes_ens` (Ensamble ponderado), VS = `pes_ens_sprb` (Voto suave), VA = `pes_ens_accq` (Voto por acción), CONS = `pes_ens_consensus` (Consenso), CONS+P = `pes_ens_consensus_prior` (Consenso con prior), GUARD = `pes_ens_trf_guard` (Compuerta del Transformer). ⚠ = fuera de rango. En negrita, la mayor media de la fila dentro del grupo.

_Réplicas fuera de muestra: los CSV `h1/general/results/{individual,ensemble}/matrices/*.csv` (y `summary.json`) tienen 5 columnas extra al final (`heldout_s1`..`heldout_s5`); se omiten en estas tablas y se resumen en §10bis. Los heatmaps por escenario (01, 02, 03, 04, 07) las muestran a la derecha de una línea vertical._

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
| `sev_base` | **0.937** | 0.914 | 0.914 | 0.889 | 0.918 | 0.928 |
| `sev_uniform` | **0.937** | 0.883 | 0.884 | 0.874 | 0.906 | 0.923 |
| `sev_gauss_low` | **0.939** | 0.910 | 0.908 | 0.896 | 0.917 | 0.919 |
| `sev_gauss_mid` | **0.938** | 0.902 | 0.900 | 0.890 | 0.904 | 0.921 |
| `sev_gauss_high` | **0.957** | 0.942 | 0.942 | 0.948 | 0.951 | 0.949 |
| `sev_weibull` | **0.938** | 0.901 | 0.897 | 0.890 | 0.908 | 0.929 |
| `sev_beta_lowskew` | **0.917** | 0.894 | 0.898 | 0.888 | 0.896 | 0.907 |
| `sev_beta_highskew` | **0.947** | 0.934 | 0.927 | 0.947 | 0.946 | 0.943 |
| `sev_bimodal` | **0.938** | 0.879 | 0.879 | 0.890 | 0.906 | 0.925 |
| `sev_extrapolate_high` ⚠ | **1.000** | 0.860 | 0.859 | 0.935 | 0.957 | 0.996 |
| `len_all_short` | 0.908 | 0.900 | 0.900 | **0.961** | 0.932 | 0.936 |
| `len_all_long` | **0.921** | 0.893 | 0.892 | 0.853 | 0.886 | 0.881 |
| `len_geometric` | **0.929** | 0.909 | 0.912 | 0.890 | 0.913 | 0.917 |
| `len_poisson` | 0.933 | 0.910 | 0.909 | 0.903 | 0.910 | **0.933** |
| `len_extrapolate_long` ⚠ | **0.900** | 0.878 | 0.880 | 0.826 | 0.870 | 0.861 |
| `joint_high_long` | **0.943** | 0.925 | 0.925 | 0.918 | 0.929 | 0.930 |
| `joint_low_short` | 0.942 | 0.921 | 0.913 | **0.988** | 0.949 | 0.969 |
| `joint_uniform_geom` | **0.928** | 0.867 | 0.862 | 0.878 | 0.904 | 0.922 |
| `joint_extrap_both` ⚠ | **1.000** | 0.900 | 0.897 | 0.953 | 0.968 | 0.997 |
| `struct_few_long_blocks` | **0.937** | 0.914 | 0.914 | 0.889 | 0.918 | 0.928 |
| `struct_many_short_blocks` | **0.937** | 0.914 | 0.914 | 0.889 | 0.918 | 0.928 |
| `struct_more_total` | **0.937** | 0.914 | 0.914 | 0.889 | 0.918 | 0.928 |

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
| `sev_gauss_low` | 0.042 | 0.074 | 0.075 | 0.085 | 0.071 | 0.070 |
| `sev_gauss_mid` | 0.039 | 0.059 | 0.060 | 0.067 | 0.049 | 0.052 |
| `sev_gauss_high` | 0.018 | 0.037 | 0.038 | 0.029 | 0.024 | 0.024 |
| `sev_weibull` | 0.042 | 0.080 | 0.082 | 0.077 | 0.061 | 0.051 |
| `sev_beta_lowskew` | 0.063 | 0.093 | 0.088 | 0.086 | 0.083 | 0.068 |
| `sev_beta_highskew` | 0.032 | 0.045 | 0.058 | 0.026 | 0.030 | 0.031 |
| `sev_bimodal` | 0.043 | 0.130 | 0.131 | 0.084 | 0.075 | 0.054 |
| `sev_extrapolate_high` ⚠ | 0.000 | 0.112 | 0.113 | 0.019 | 0.011 | 0.008 |
| `len_all_short` | 0.064 | 0.059 | 0.059 | 0.021 | 0.057 | 0.063 |
| `len_all_long` | 0.032 | 0.030 | 0.031 | 0.043 | 0.019 | 0.038 |
| `len_geometric` | 0.038 | 0.044 | 0.042 | 0.067 | 0.045 | 0.049 |
| `len_poisson` | 0.041 | 0.042 | 0.044 | 0.065 | 0.043 | 0.048 |
| `len_extrapolate_long` ⚠ | 0.034 | 0.035 | 0.042 | 0.048 | 0.037 | 0.048 |
| `joint_high_long` | 0.018 | 0.023 | 0.025 | 0.038 | 0.020 | 0.023 |
| `joint_low_short` | 0.056 | 0.107 | 0.116 | 0.018 | 0.098 | 0.041 |
| `joint_uniform_geom` | 0.056 | 0.119 | 0.114 | 0.094 | 0.058 | 0.067 |
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
| `sev_uniform` | 0.000 | +0.032 | +0.030 | +0.015 | +0.012 | +0.005 |
| `sev_gauss_low` | -0.002 | +0.005 | +0.007 | -0.006 | +0.001 | +0.009 |
| `sev_gauss_mid` | -0.001 | +0.012 | +0.015 | -0.001 | +0.014 | +0.007 |
| `sev_gauss_high` | -0.020 | -0.028 | -0.028 | -0.058 | -0.034 | -0.021 |
| `sev_weibull` | 0.000 | +0.013 | +0.017 | -0.001 | +0.010 | -0.002 |
| `sev_beta_lowskew` | +0.020 | +0.020 | +0.016 | +0.002 | +0.022 | +0.021 |
| `sev_beta_highskew` | -0.010 | -0.020 | -0.013 | -0.057 | -0.029 | -0.015 |
| `sev_bimodal` | -0.001 | +0.035 | +0.036 | 0.000 | +0.012 | +0.003 |
| `sev_extrapolate_high` ⚠ | -0.063 | +0.054 | +0.055 | -0.046 | -0.039 | -0.068 |
| `len_all_short` | +0.029 | +0.014 | +0.014 | -0.072 | -0.015 | -0.008 |
| `len_all_long` | +0.017 | +0.021 | +0.022 | +0.037 | +0.032 | +0.047 |
| `len_geometric` | +0.008 | +0.005 | +0.002 | -0.001 | +0.004 | +0.011 |
| `len_poisson` | +0.005 | +0.004 | +0.005 | -0.014 | +0.007 | -0.006 |
| `len_extrapolate_long` ⚠ | +0.037 | +0.036 | +0.035 | +0.063 | +0.047 | +0.067 |
| `joint_high_long` | -0.006 | -0.011 | -0.010 | -0.028 | -0.011 | -0.003 |
| `joint_low_short` | -0.004 | -0.006 | +0.001 | -0.098 | -0.031 | -0.042 |
| `joint_uniform_geom` | +0.009 | +0.047 | +0.052 | +0.012 | +0.013 | +0.006 |
| `joint_extrap_both` ⚠ | -0.063 | +0.014 | +0.017 | -0.064 | -0.051 | -0.069 |
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
| `sev_uniform` | 0.00 | -0.42 | -0.40 | -0.20 | -0.25 | -0.10 |
| `sev_gauss_low` | +0.04 | -0.08 | -0.11 | +0.08 | -0.02 | -0.15 |
| `sev_gauss_mid` | +0.01 | -0.23 | -0.28 | +0.01 | -0.31 | -0.13 |
| `sev_gauss_high` | +0.72 | +0.68 | +0.67 | +1.17 | +0.99 | +0.58 |
| `sev_weibull` | +0.01 | -0.20 | -0.26 | +0.01 | -0.19 | +0.03 |
| `sev_beta_lowskew` | -0.39 | -0.27 | -0.23 | -0.02 | -0.33 | -0.37 |
| `sev_beta_highskew` | +0.30 | +0.43 | +0.24 | +1.17 | +0.80 | +0.37 |
| `sev_bimodal` | +0.02 | -0.36 | -0.36 | 0.00 | -0.19 | -0.07 |
| `sev_extrapolate_high` ⚠ | +2.52 | -0.62 | -0.64 | +0.97 | +1.29 | +2.05 |
| `len_all_short` | -0.56 | -0.27 | -0.27 | +1.51 | +0.29 | +0.14 |
| `len_all_long` | -0.49 | -0.56 | -0.57 | -0.67 | -0.99 | -1.11 |
| `len_geometric` | -0.23 | -0.12 | -0.05 | +0.02 | -0.10 | -0.22 |
| `len_poisson` | -0.12 | -0.09 | -0.11 | +0.21 | -0.17 | +0.12 |
| `len_extrapolate_long` ⚠ | -1.07 | -0.89 | -0.79 | -1.12 | -1.20 | -1.41 |
| `joint_high_long` | +0.22 | +0.31 | +0.29 | +0.54 | +0.35 | +0.07 |
| `joint_low_short` | +0.09 | +0.08 | -0.02 | +2.09 | +0.41 | +0.95 |
| `joint_uniform_geom` | -0.19 | -0.53 | -0.60 | -0.14 | -0.26 | -0.11 |
| `joint_extrap_both` ⚠ | +2.52 | -0.21 | -0.25 | +1.38 | +1.73 | +2.10 |
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
| `sev_uniform` | 0.0 | -1.7 | -1.6 | -0.6 | -0.8 | -0.2 |
| `sev_gauss_low` | -0.1 | -0.2 | -0.3 | -0.2 | 0.0 | -0.4 |
| `sev_gauss_mid` | 0.0 | -0.7 | -0.9 | 0.0 | -1.1 | -0.3 |
| `sev_gauss_high` | -4.0 | -3.7 | -3.6 | -8.6 | -6.7 | -2.9 |
| `sev_weibull` | 0.0 | -0.6 | -0.8 | 0.0 | -0.5 | -0.1 |
| `sev_beta_lowskew` | -1.5 | -0.9 | -0.7 | 0.0 | -1.2 | -1.4 |
| `sev_beta_highskew` | -1.0 | -1.8 | -0.8 | -8.5 | -4.8 | -1.4 |
| `sev_bimodal` | -0.1 | -1.3 | -1.4 | 0.0 | -0.5 | -0.2 |
| `sev_extrapolate_high` ⚠ | -20.6 | -3.2 | -3.3 | -6.2 | -9.5 | -16.9 |
| `len_all_short` | -2.7 | -0.9 | -0.9 | -12.0 | -1.0 | -0.4 |
| `len_all_long` | -2.2 | -2.7 | -2.8 | -3.6 | -6.6 | -8.3 |
| `len_geometric` | -0.7 | -0.3 | -0.1 | 0.0 | -0.2 | -0.7 |
| `len_poisson` | -0.3 | -0.2 | -0.3 | -0.6 | -0.5 | -0.3 |
| `len_extrapolate_long` ⚠ | -7.8 | -5.8 | -4.8 | -8.3 | -9.4 | -12.1 |
| `joint_high_long` | -0.7 | -1.1 | -1.0 | -2.5 | -1.3 | -0.2 |
| `joint_low_short` | -0.2 | -0.2 | 0.0 | -17.8 | -1.7 | -6.4 |
| `joint_uniform_geom` | -0.5 | -2.4 | -2.9 | -0.4 | -0.9 | -0.3 |
| `joint_extrap_both` ⚠ | -20.6 | -0.6 | -0.8 | -10.3 | -13.6 | -17.2 |
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
| `sev_uniform` | 0.096 | 0.190 | 0.122 | 0.127 | 0.089 | 0.309 |
| `sev_gauss_low` | 0.349 | 0.449 | 0.409 | 0.470 | 0.210 | 0.359 |
| `sev_gauss_mid` | 0.041 | 0.028 | 0.035 | 0.070 | 0.033 | 0.171 |
| `sev_gauss_high` | 0.332 | 0.334 | 0.332 | 0.051 | 0.206 | 0.292 |
| `sev_weibull` | 0.047 | 0.097 | 0.065 | 0.150 | 0.029 | 0.159 |
| `sev_beta_lowskew` | 0.204 | 0.275 | 0.227 | 0.416 | 0.148 | 0.319 |
| `sev_beta_highskew` | 0.306 | 0.277 | 0.270 | 0.035 | 0.181 | 0.316 |
| `sev_bimodal` | 0.127 | 0.208 | 0.150 | 0.294 | 0.102 | 0.361 |
| `sev_extrapolate_high` ⚠ | 1.524 | 0.718 | 0.699 | 1.873 | 1.026 | 1.062 |
| `len_all_short` | 0.543 | 0.404 | 0.406 | 0.655 | 0.459 | 0.472 |
| `len_all_long` | 0.234 | 0.279 | 0.257 | 0.169 | 0.258 | 0.194 |
| `len_geometric` | 0.055 | 0.047 | 0.042 | 0.010 | 0.036 | 0.021 |
| `len_poisson` | 0.038 | 0.013 | 0.019 | 0.023 | 0.007 | 0.026 |
| `len_extrapolate_long` ⚠ | 0.595 | 0.558 | 0.532 | 0.388 | 0.569 | 0.416 |
| `joint_high_long` | 0.497 | 0.462 | 0.443 | 0.187 | 0.446 | 0.398 |
| `joint_low_short` | 0.663 | 0.759 | 0.753 | 1.166 | 0.535 | 1.025 |
| `joint_uniform_geom` | 0.070 | 0.167 | 0.146 | 0.171 | 0.078 | 0.255 |
| `joint_extrap_both` ⚠ | 1.367 | 0.812 | 0.761 | 1.082 | 1.028 | 0.905 |
| `struct_few_long_blocks` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `struct_many_short_blocks` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `struct_more_total` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

_Mínimo, máximo, fracción de secuencias en el óptimo (`f_opt`), fracción bajo 0,8 (`f_lt08`) y asignación media por celda: ver `celdas` en `mpes_resultados.json`._

## 8. Ganadores por escenario y ensambles frente al Transformer

| Escenario | Mejor individual | Mejor ensamble | Mejor global | ENS − TRF | VS − TRF | VA − TRF | CONS − TRF | CONS+P − TRF | GUARD − TRF |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `sev_base` | TRF (0.927) | ENS (0.937) | ENS (0.937) | +0.010 | -0.013 | -0.013 | -0.038 | -0.009 | +0.001 |
| `sev_uniform` | TRF (0.923) | ENS (0.937) | ENS (0.937) | +0.014 | -0.040 | -0.039 | -0.048 | -0.017 | 0.000 |
| `sev_gauss_low` | TRF (0.914) | ENS (0.939) | ENS (0.939) | +0.025 | -0.004 | -0.006 | -0.018 | +0.003 | +0.006 |
| `sev_gauss_mid` | TRF (0.920) | ENS (0.938) | ENS (0.938) | +0.018 | -0.018 | -0.021 | -0.030 | -0.017 | +0.001 |
| `sev_gauss_high` | TRF (0.949) | ENS (0.957) | ENS (0.957) | +0.008 | -0.007 | -0.008 | -0.002 | +0.002 | 0.000 |
| `sev_weibull` | TRF (0.929) | ENS (0.938) | ENS (0.938) | +0.008 | -0.028 | -0.032 | -0.039 | -0.021 | 0.000 |
| `sev_beta_lowskew` | TRF (0.905) | ENS (0.917) | ENS (0.917) | +0.013 | -0.011 | -0.006 | -0.017 | -0.009 | +0.002 |
| `sev_beta_highskew` | A2C (0.944) | ENS (0.947) | ENS (0.947) | +0.005 | -0.008 | -0.015 | +0.004 | +0.004 | 0.000 |
| `sev_bimodal` | TRF (0.925) | ENS (0.938) | ENS (0.938) | +0.013 | -0.046 | -0.047 | -0.036 | -0.019 | -0.001 |
| `sev_extrapolate_high` | TRF (0.996) | ENS (1.000) | ENS (1.000) | +0.004 | -0.135 | -0.137 | -0.061 | -0.039 | 0.000 |
| `len_all_short` | A2C (0.958) | CONS (0.961) | CONS (0.961) | -0.028 | -0.036 | -0.036 | +0.026 | -0.003 | 0.000 |
| `len_all_long` | RDQN (0.901) | ENS (0.921) | ENS (0.921) | +0.045 | +0.017 | +0.017 | -0.023 | +0.010 | +0.005 |
| `len_geometric` | TRF (0.916) | ENS (0.929) | ENS (0.929) | +0.012 | -0.007 | -0.004 | -0.026 | -0.003 | +0.001 |
| `len_poisson` | TRF (0.933) | GUARD (0.933) | GUARD (0.933) | 0.000 | -0.023 | -0.024 | -0.030 | -0.023 | 0.000 |
| `len_extrapolate_long` | RDQN (0.889) | ENS (0.900) | ENS (0.900) | +0.040 | +0.018 | +0.020 | -0.034 | +0.011 | +0.002 |
| `joint_high_long` | DQN (0.935) | ENS (0.943) | ENS (0.943) | +0.013 | -0.005 | -0.006 | -0.013 | -0.001 | 0.000 |
| `joint_low_short` | DQN, A2C (0.988) | CONS (0.988) | CONS (0.988) | -0.028 | -0.049 | -0.057 | +0.018 | -0.020 | 0.000 |
| `joint_uniform_geom` | TRF (0.921) | ENS (0.928) | ENS (0.928) | +0.008 | -0.054 | -0.058 | -0.043 | -0.016 | +0.001 |
| `joint_extrap_both` | TRF (0.997) | ENS (1.000) | ENS (1.000) | +0.003 | -0.097 | -0.100 | -0.044 | -0.029 | 0.000 |
| `struct_few_long_blocks` | TRF (0.927) | ENS (0.937) | ENS (0.937) | +0.010 | -0.013 | -0.013 | -0.038 | -0.009 | +0.001 |
| `struct_many_short_blocks` | TRF (0.927) | ENS (0.937) | ENS (0.937) | +0.010 | -0.013 | -0.013 | -0.038 | -0.009 | +0.001 |
| `struct_more_total` | TRF (0.927) | ENS (0.937) | ENS (0.937) | +0.010 | -0.013 | -0.013 | -0.038 | -0.009 | +0.001 |

- El Transformer tiene la mayor media individual (exacta) en **16** de 22 escenarios; con dos decimales queda primero o empatado en 17 (empates a 2 decimales: `sev_gauss_low`, `sev_beta_highskew`).
- Escenarios donde el mejor individual no es el Transformer (exacto): `sev_beta_highskew` (A2C), `len_all_short` (A2C), `len_all_long` (RDQN), `len_extrapolate_long` (RDQN), `joint_high_long` (DQN), `joint_low_short` (DQN, A2C).
- Ensamble ponderado supera al Transformer en 18 de 21 escenarios de generalización (margen máximo +0.045).
- Voto suave supera al Transformer en 2 de 21 escenarios de generalización (margen máximo +0.018).
- Voto por acción supera al Transformer en 2 de 21 escenarios de generalización (margen máximo +0.020).
- Consenso supera al Transformer en 3 de 21 escenarios de generalización (margen máximo +0.026).
- Consenso con prior supera al Transformer en 5 de 21 escenarios de generalización (margen máximo +0.011).
- Compuerta del Transformer supera al Transformer en 15 de 21 escenarios de generalización (margen máximo +0.006).
- Peor escenario del ensamble ponderado − peor escenario del Transformer = +0.040.

**Cada ensamble frente al Transformer en generalización** (Tabla `tab:trf-vs-ens`, `writings/auxiliar/scripts/trf_vs_ens.py`: diferencia de las medias de los 21 escenarios; $d$ y Welch sobre todas sus secuencias juntas; positivo = ventaja del ensamble):

| Ensamble | Gen. | Diferencia | $d$ | $p$ Welch | $\log_{10} p$ | Mayor media en |
|---|---:|---:|---:|---:|---:|---:|
| Ensamble ponderado | 0.939 | +0.010 | +0.19 | 1.2e-06 | -5.9 | 18 de 21 |
| Compuerta del Transformer | 0.931 | +0.001 | +0.02 | 0.68 | -0.2 | 15 de 21 |
| Consenso con prior | 0.919 | -0.010 | -0.18 | 2.9e-06 | -5.5 | 5 de 21 |
| Consenso | 0.905 | -0.025 | -0.39 | 1.9e-23 | -22.7 | 3 de 21 |
| Voto suave | 0.902 | -0.027 | -0.40 | 3.5e-25 | -24.5 | 2 de 21 |
| Voto por acción | 0.901 | -0.028 | -0.42 | 9.7e-27 | -26.0 | 2 de 21 |

## 9. Comparaciones entre pares (21 escenarios juntos)

### 9.1 Pares citados en la tesis (texto de `sec:res-individual` y `sec:res-ensembles`; mapas en `fig:pairwise-cohen` y `fig:ensemble-pairwise`)

| Comparación | $d$ | $\log_{10} p$ | KL sim. |
|---|---:|---:|---:|
| Transformer vs Q-Learning | 0.79 | -84.4 | 0.50 |
| Transformer vs DQN recurrente | 0.57 | -46.8 | 0.35 |
| Transformer vs DQN | 0.50 | -36.6 | 0.17 |
| Ensamble ponderado vs Consenso | 0.58 | -48.0 | 0.23 |
| Ensamble ponderado vs Compuerta del Transformer | 0.17 | -5.0 | 0.02 |

### 9.2 Matrices de pares — individuales

**$d$ de Cohen** (fila − columna; positivo = gana la fila):

| fila \ columna | QL-base | QL | DQL | DQN | RDQN | A2C | TRF |
|---|---:|---:|---:|---:|---:|---:|---:|
| QL-base | — | -0.21 | -0.28 | -0.53 | -0.39 | -0.48 | -0.90 |
| QL | 0.21 | — | -0.07 | -0.35 | -0.20 | -0.30 | -0.79 |
| DQL | 0.28 | 0.07 | — | -0.30 | -0.14 | -0.24 | -0.77 |
| DQN | 0.53 | 0.35 | 0.30 | — | 0.13 | 0.03 | -0.50 |
| RDQN | 0.39 | 0.20 | 0.14 | -0.13 | — | -0.09 | -0.57 |
| A2C | 0.48 | 0.30 | 0.24 | -0.03 | 0.09 | — | -0.49 |
| TRF | 0.90 | 0.79 | 0.77 | 0.50 | 0.57 | 0.49 | — |

**$\log_{10} p$ de Welch**:

| fila \ columna | QL-base | QL | DQL | DQN | RDQN | A2C | TRF |
|---|---:|---:|---:|---:|---:|---:|---:|
| QL-base | — | -7.2 | -12.2 | -40.7 | -23.1 | -33.6 | -106.7 |
| QL | -7.2 | — | -1.1 | -18.7 | -6.8 | -13.8 | -84.4 |
| DQL | -12.2 | -1.1 | — | -13.7 | -3.7 | -9.4 | -80.6 |
| DQN | -40.7 | -18.7 | -13.7 | — | -3.2 | -0.4 | -36.6 |
| RDQN | -23.1 | -6.8 | -3.7 | -3.2 | — | -1.8 | -46.8 |
| A2C | -33.6 | -13.8 | -9.4 | -0.4 | -1.8 | — | -34.6 |
| TRF | -106.7 | -84.4 | -80.6 | -36.6 | -46.8 | -34.6 | — |

**KL simetrizada entre histogramas de desempeño**:

| fila \ columna | QL-base | QL | DQL | DQN | RDQN | A2C | TRF |
|---|---:|---:|---:|---:|---:|---:|---:|
| QL-base | — | 0.096 | 0.106 | 0.385 | 0.202 | 0.196 | 1.000 |
| QL | 0.096 | — | 0.017 | 0.126 | 0.084 | 0.077 | 0.498 |
| DQL | 0.106 | 0.017 | — | 0.076 | 0.075 | 0.064 | 0.417 |
| DQN | 0.385 | 0.126 | 0.076 | — | 0.115 | 0.089 | 0.169 |
| RDQN | 0.202 | 0.084 | 0.075 | 0.115 | — | 0.045 | 0.346 |
| A2C | 0.196 | 0.077 | 0.064 | 0.089 | 0.045 | — | 0.201 |
| TRF | 1.000 | 0.498 | 0.417 | 0.169 | 0.346 | 0.201 | — |

### 9.3 Matrices de pares — ensambles

**$d$ de Cohen** (fila − columna; positivo = gana la fila):

| fila \ columna | ENS | VS | VA | CONS | CONS+P | GUARD |
|---|---:|---:|---:|---:|---:|---:|
| ENS | — | 0.59 | 0.60 | 0.58 | 0.39 | 0.17 |
| VS | -0.59 | — | 0.01 | -0.03 | -0.25 | -0.42 |
| VA | -0.60 | -0.01 | — | -0.04 | -0.27 | -0.43 |
| CONS | -0.58 | 0.03 | 0.04 | — | -0.23 | -0.40 |
| CONS+P | -0.39 | 0.25 | 0.27 | 0.23 | — | -0.20 |
| GUARD | -0.17 | 0.42 | 0.43 | 0.40 | 0.20 | — |

**$\log_{10} p$ de Welch**:

| fila \ columna | ENS | VS | VA | CONS | CONS+P | GUARD |
|---|---:|---:|---:|---:|---:|---:|
| ENS | — | -48.9 | -50.8 | -48.0 | -22.6 | -5.0 |
| VS | -48.9 | — | -0.1 | -0.3 | -10.2 | -26.2 |
| VA | -50.8 | -0.1 | — | -0.6 | -11.3 | -27.8 |
| CONS | -48.0 | -0.3 | -0.6 | — | -8.6 | -24.5 |
| CONS+P | -22.6 | -10.2 | -11.3 | -8.6 | — | -6.5 |
| GUARD | -5.0 | -26.2 | -27.8 | -24.5 | -6.5 | — |

**KL simetrizada entre histogramas de desempeño**:

| fila \ columna | ENS | VS | VA | CONS | CONS+P | GUARD |
|---|---:|---:|---:|---:|---:|---:|
| ENS | — | 0.305 | 0.303 | 0.228 | 0.086 | 0.024 |
| VS | 0.305 | — | 0.001 | 0.082 | 0.091 | 0.227 |
| VA | 0.303 | 0.001 | — | 0.084 | 0.092 | 0.225 |
| CONS | 0.228 | 0.082 | 0.084 | — | 0.095 | 0.128 |
| CONS+P | 0.086 | 0.091 | 0.092 | 0.095 | — | 0.043 |
| GUARD | 0.024 | 0.227 | 0.225 | 0.128 | 0.043 | — |

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
| `pes_ens_consensus_prior` | 0.915 | 0.895 | 0.888 | 0.930 | 0.932 | 0.921 | 0.928 | 0.932 |
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
| `pes_ens_consensus_prior` | 0.203 | 0.086 | 0.033 | 0.050 | 0.003 | 0.208 | 0.017 | 0.008 | 0.053 | 0.289 | 0.050 | 5.04 |
| `pes_ens_trf_guard` | 0.294 | 0.017 | 0.017 | 0.000 | 0.036 | 0.033 | 0.042 | 0.258 | 0.086 | 0.139 | 0.078 | 5.14 |

_Las distribuciones incluyen los pasos sin recursos (acción forzada 0)._

### 10.3 Registro de confianza del Transformer (no figura en la tesis)

- 258 decisiones con recursos disponibles en la referencia; confianza media 0.116. Q no factibles -> valor muy negativo; desplazamiento a valores no negativos; normalización; confianza = 1 - H_norm. NO comparable con la confianza de los ensambles (refleja sobre todo cuántas acciones quedan factibles).
- Fuente: general.scripts.agent_internals.load_confidences(h1/ml/pes_trf/inputs/2026-05-02_TRF_TRAIN/confsrl_2026-05-02.npy); figura no incluida en la tesis (h1/general/results/agent_internals/trf_agent_confidences.png).

## 10bis. Réplicas fuera de muestra de la referencia (Sección `sec:res-heldout`, Tabla `tab:heldout`)

_Fuente: `h1/general/results/heldout/heldout_gap.json` (generado con `writings/auxiliar/scripts/heldout_gap.py`) y `heldout_catalogue.json` (`python -m general.scripts.benchmark heldout`). Numeración 10bis para no alterar las referencias a §12/§13 de `instrucciones_sistema.md`._

- **Propósito**: los modelos optimizados y los ensambles eligieron su configuración sobre las mismas 64 secuencias de `sev_base`; las réplicas miden cuánto se sobreajustó cada configuración a esas secuencias (cuán optimista es la referencia). Ningún modelo se reentrenó ni se reoptimizó.
- **Procedimiento**: cada réplica sortea 8 × 8 = 64 secuencias nuevas: cada longitud y cada severidad inicial, de forma independiente (i.i.d.), con las frecuencias empíricas de `sequence_lengths.csv` e `initial_severity.csv` de la referencia, que son las mismas con las que se entrenan los modelos individuales. Semilla 42 + k (k = 1..5 → 43..47). En total 320 secuencias.
- **Exclusión**: no intervienen en el entrenamiento, en la optimización ni en ningún agregado de generalización (medias de 21 escenarios, peor escenario, degradación por familia, matrices de pares, conteos "N de 22"/"N de 21").
- **Integridad**: `benchmark heldout` guarda en `results/heldout/heldout_sK/` los dos CSV y `sampling_distribution.json` (frecuencias de sorteo, semilla y frecuencias obtenidas) y comprueba con sha256 que los 13 paquetes recibieron las mismas secuencias: 13 de 13 coinciden en las 5 réplicas.

**Frecuencias de sorteo** (de los CSV de la referencia):

| Severidad inicial | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Conteo (n = 360) | 40 | 64 | 64 | 80 | 56 | 16 | 40 |
| Probabilidad | 0.111 | 0.178 | 0.178 | 0.222 | 0.156 | 0.044 | 0.111 |

| Longitud | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Conteo (n = 64) | 12 | 9 | 10 | 13 | 8 | 7 | 2 | 3 |
| Probabilidad | 0.188 | 0.141 | 0.156 | 0.203 | 0.125 | 0.109 | 0.031 | 0.047 |

| Réplica | Semilla | Secuencias | Pasos |
|---|---:|---:|---:|
| `heldout_s1` | 43 | 64 | 340 |
| `heldout_s2` | 44 | 64 | 374 |
| `heldout_s3` | 45 | 64 | 369 |
| `heldout_s4` | 46 | 64 | 346 |
| `heldout_s5` | 47 | 64 | 389 |
| total | — | 320 | 1818 |

**Referencia frente a réplicas** (Tabla `tab:heldout`; ordenado por la referencia; rango entre los 13 modelos):

| Paquete | Nombre en el texto | Selección | Ref. | Fuera de muestra (320) | SD entre réplicas | Caída | $d$ | $p$ Welch | Rango ref. → f. m. |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `pes_ens` | Ensamble ponderado | manual (sin optimizar) | 0.937 | 0.939 | 0.003 | -0.002 | +0.04 | 0.75 | 1 → 1 |
| `pes_ens_trf_guard` | Compuerta del Transformer | Optuna | 0.928 | 0.928 | 0.004 | 0.000 | -0.01 | 0.96 | 2 → 3 |
| `pes_trf` | Transformer | Optuna | 0.927 | 0.929 | 0.003 | -0.001 | +0.03 | 0.81 | 3 → 2 |
| `pes_ens_consensus_prior` | Consenso con prior | Optuna | 0.918 | 0.915 | 0.007 | +0.003 | -0.07 | 0.61 | 4 → 6 |
| `pes_ens_accq` | Voto por acción | Optuna | 0.914 | 0.916 | 0.009 | -0.002 | +0.04 | 0.75 | 5 → 5 |
| `pes_ens_sprb` | Voto suave | Optuna | 0.914 | 0.917 | 0.008 | -0.003 | +0.06 | 0.66 | 6 → 4 |
| `pes_rdqn` | DQN recurrente | Optuna | 0.899 | 0.903 | 0.008 | -0.004 | +0.08 | 0.54 | 7 → 7 |
| `pes_dql` | Double Q-Learning | Optuna | 0.896 | 0.900 | 0.006 | -0.004 | +0.07 | 0.57 | 8 → 8 |
| `pes_dqn` | DQN | Optuna | 0.894 | 0.891 | 0.006 | +0.002 | -0.04 | 0.77 | 9 → 10 |
| `pes_ens_consensus` | Consenso | Optuna | 0.889 | 0.893 | 0.007 | -0.004 | +0.06 | 0.67 | 10 → 9 |
| `pes_a2c` | A2C | Optuna | 0.887 | 0.887 | 0.006 | 0.000 | -0.01 | 0.96 | 11 → 11 |
| `pes_ql` | Q-Learning | Optuna | 0.887 | 0.881 | 0.008 | +0.005 | -0.08 | 0.53 | 12 → 12 |
| `pes_base` | Q-Learning base | sin optimizar | 0.871 | 0.860 | 0.010 | +0.011 | -0.15 | 0.29 | 13 → 13 |

**Media por réplica**:

| Paquete | s1 | s2 | s3 | s4 | s5 |
|---|---:|---:|---:|---:|---:|
| `pes_ens` | 0.939 | 0.936 | 0.940 | 0.936 | 0.943 |
| `pes_ens_trf_guard` | 0.935 | 0.928 | 0.924 | 0.925 | 0.926 |
| `pes_trf` | 0.934 | 0.929 | 0.926 | 0.927 | 0.927 |
| `pes_ens_consensus_prior` | 0.916 | 0.924 | 0.916 | 0.904 | 0.914 |
| `pes_ens_accq` | 0.926 | 0.915 | 0.925 | 0.904 | 0.912 |
| `pes_ens_sprb` | 0.925 | 0.917 | 0.924 | 0.909 | 0.909 |
| `pes_rdqn` | 0.911 | 0.900 | 0.911 | 0.895 | 0.898 |
| `pes_dql` | 0.907 | 0.903 | 0.893 | 0.896 | 0.901 |
| `pes_dqn` | 0.891 | 0.894 | 0.894 | 0.881 | 0.897 |
| `pes_ens_consensus` | 0.903 | 0.897 | 0.890 | 0.885 | 0.891 |
| `pes_a2c` | 0.893 | 0.890 | 0.885 | 0.877 | 0.889 |
| `pes_ql` | 0.884 | 0.875 | 0.890 | 0.871 | 0.886 |
| `pes_base` | 0.873 | 0.858 | 0.847 | 0.864 | 0.858 |
| ENS − TRF | +0.004 | +0.008 | +0.014 | +0.010 | +0.016 |

- Caída = ref. − media de las 320 secuencias de las réplicas (positiva = pérdida, como la degradación). $d$ y $p$: réplicas agrupadas (n = 320) frente a la referencia (n = 64) del mismo modelo; $d > 0$ = mejor en las réplicas; Welch bilateral. SD entre réplicas = desviación estándar de las 5 medias (entre paréntesis en `tab:heldout`). Error estándar de la caída entre 0.005 y 0.010.
- Caída entre -0.004 (DQN recurrente) y +0.011 (Q-Learning base); |d| ≤ 0.15 (efecto despreciable) y p ≥ 0.29 en los 13. En los 11 optimizados con Optuna, entre -0.004 y +0.005 (máx.: Q-Learning); la mayor caída es la de Q-Learning base, que no se optimizó: diferencias de ese tamaño son compatibles con la variación entre muestras.
- Ordenamiento: Spearman ρ = 0.967 (p = 7.1e-08, 13 modelos; tesis: 0,97). Cambian de posición sólo modelos cuyas medias difieren en 0,004 o menos en ambas condiciones: Compuerta del Transformer 2 → 3, Transformer 3 → 2, Consenso con prior 4 → 6, Voto suave 6 → 4, DQN 9 → 10, Consenso 10 → 9. El ensamble ponderado es 1.º en ambas.
- **Ensamble ponderado vs Transformer** en las mismas 320 secuencias: diferencia media +0.010 (en la referencia +0.010), t pareada p = 7.5e-08 (tesis: p < 10⁻⁷), Wilcoxon p = 2.0e-08; mayor media en 5 de 5 réplicas.
- **Afirmación de la tesis** (`sec:res-heldout`): "Elegir la configuración sobre las 64 secuencias de la referencia no infló, por lo tanto, el desempeño medido en esa condición." `06Discussion` (último párrafo): la ventaja del ensamble ponderado tampoco se debe a que sus parámetros se eligieran sobre la referencia (se mantiene en 0,010). `07Conclusion` (limitaciones): la caída no superó 0,011 ni fue significativa.
- **Salvedad**: las réplicas se sortean con frecuencias estimadas a partir de las mismas 64 secuencias, no con datos nuevos del experimento original: miden el sobreajuste a esas secuencias concretas, no a la distribución.
- `reference_std` de `heldout_gap.json` usa ddof = 1 (p. ej. Transformer 0.0458), mientras que las σ de §4.2/§5.2 usan ddof = 0 (0.045); las medias coinciden.
- Figuras: las columnas `heldout_s1`..`s5` aparecen a la derecha de una línea vertical en los heatmaps 01/02/03/04/07 (en la tesis: `fig:heatmap-global`, `fig:heatmap-ens`, `fig:heatmap-welch`, `fig:ensemble-statistical-heatmaps`); `15_referencia_vs_heldout` (individual y ensemble) se genera pero la tesis no la usa. Valores por celda: `h1/general/results/<grupo>/cells/<pkg>__heldout_sK.json` (no se copian a `celdas` del JSON).

## 11. Advertencias metodológicas (deben respetarse al redactar)

- Una única semilla (42) por escenario de generalización: no hay intervalos de confianza del ordenamiento; sólo la referencia cuenta con 5 réplicas (fuera de muestra, §10bis). Los p de Welch no son independientes ni corregidos por comparaciones múltiples.
- Hiperparámetros de modelos y ensambles ajustados con las mismas 64 secuencias de la referencia. Medido en las 5 réplicas fuera de muestra (§10bis): caída entre -0.004 y +0.011, nunca significativa (|d| ≤ 0.15, p ≥ 0.29), ρ de Spearman 0.97: la referencia no resultó optimista. Salvedad: las réplicas se sortean con frecuencias estimadas a partir de esas mismas 64 secuencias, no con datos nuevos del experimento original.
- La media de generalización incluye los 3 escenarios estructurales (idénticos a la referencia) y escenarios más fáciles (p. ej. len_all_short, joint_low_short); leerla junto con el peor escenario.
- Entrenamiento sólo con severidades 2..8: filas S = 0, 1, 9 de las tablas Q quedan con valores iniciales aleatorios; parte de la caída tabular refleja falta de cobertura, no sólo capacidad de generalizar.
- Con severidades 10..12 una política que asigna mucho puede acercarse al óptimo: un rbar alto en sev_extrapolate_high no prueba por sí solo mejor generalización (A2C también supera allí su referencia).
- Arquitecturas de DQN recurrente y Transformer elegidas ad hoc (no optimizadas): las conclusiones se refieren a los modelos finales evaluados; los resultados no identifican la causa de la ventaja del Transformer.
- La confianza 1 - H_norm se calcula sobre softmax de valores Q (no probabilidades aprendidas, salvo A2C): es heurística, no calibrada; no se comprobó que sea mayor en decisiones acertadas.
- El registro de confianza del Transformer (media 0,116) usa otro cálculo que los ensambles: no comparar con tau_g ni con otros umbrales.
- La ventaja del ensamble ponderado proviene del prior de severidad y de tau = 15, no de la ponderación por confianza, y requiere ambos (sensibilidad en sec:res-posthoc). Sus parámetros se fijaron a mano sobre la referencia y sólo se justifican a posteriori.
- Una regla fija sin modelo, a = min(S + 2, R), calibrada con el óptimo, iguala al ensamble ponderado y supera al Transformer (sec:res-posthoc, sec:disc-rule): no presentar a los modelos como superiores a una heurística bien calibrada en este entorno.
- "Distancia de Lieber" en documentos previos = divergencia de Kullback-Leibler.
- h2/ es una línea suspendida: no citarla como trabajo realizado. La tesis no usa datos humanos (ds004477 sólo como contexto del entorno PES).

## 12. Puntos de atención detectados en el borrador LaTeX actual

Contrastados contra los `.tex` actuales el 2026-09-28 (los cinco puntos sobre 04Materials, 05Results, 06Discussion y el resumen de la versión del 2026-09-24 ya están corregidos; el de los números de ensayos de Optuna se resolvió porque ahora se leen de `inputs/*_BAYESIAN_OPT/optimization_results_*.txt`). Corregirlos cuando se edite el archivo afectado.

| # | Severidad | Archivo | Lugar | Hallazgo | Corrección sugerida |
|---:|---|---|---|---|---|
| 1 | observación | `audit.py` | criterio 6 (cobertura de paquetes) | Pasa sólo gracias a los nombres de archivo PES_<PKG>_results.png (\texttt{pes\_base} no contiene "pes_base"). Si se eliminan las figuras por modelo, el criterio falla. | Mantener esas figuras o mencionar los paquetes con \verb\|pes_base\| (el criterio de idioma ignora \verb). |

## 13. Mapa del documento LaTeX

- Estado del último `audit.py` (`writings/audit/AUDIT.md`): Compilación OK (60 páginas), 20 figuras y 15 tablas con label, 87 referencias internas, 21 imágenes; avisos (⚠/❌): ninguno; `.tex` no incluidos en Main.tex: `Acknowledgement.tex` (excluido de Main.tex a propósito).
- Recuento del generador sobre los `.tex` actuales: 20 figuras y 15 tablas con label, 90 referencias internas, 21 imágenes distintas (difiere de AUDIT.md: volver a ejecutar `audit.py`).
- Estructura en el repositorio (en el chat los archivos están planos):
  - `writings/00_Main/`: `.latexmkrc`, `IEEEtran.cls`, `Main.tex`, `References.bib`
  - `writings/01_Chapters/`: `000NHH-Frontpage.tex`, `00Abstract.tex`, `00Abstract_en.tex`, `01Introduction.tex`, `02Background.tex`, `03StateOfTheArt.tex`, `04Materials.tex`, `05Results.tex`, `06Discussion.tex`, `07Conclusion.tex`, `Appendix.tex`, `Acknowledgement.tex (excluido de Main.tex a propósito)`
  - `writings/02_Images/frontpage/`: `LOGO-ITBA.jpg`
  - `writings/02_Images/baseline/`: `random_player_normalised_performance.png`, `random_player_sequence_performance.png`
  - `writings/02_Images/per_model/`: `PES_A2C_results.png`, `PES_BASE_results.png`, `PES_DQL_results.png`, `PES_DQN_results.png`, `PES_ENS_results.png`, `PES_QL_results.png`, `PES_RDQN_results.png`, `PES_TRF_results.png`
  - `writings/02_Images/individual/`: `ind_01_desempeno_por_escenario.png`, `ind_03_welch_logp_por_escenario.png`, `ind_05_curvas_por_familia.png`, `ind_13_pares_cohen_d.png`
  - `writings/02_Images/ensemble/`: `ens_01_desempeno_por_escenario.png`, `ens_03_welch_logp_por_escenario.png`, `ens_06_curvas_extrapolacion.png`, `ens_07_cohen_d_por_escenario.png`, `ens_12_pares_welch_logp.png`, `ens_13_pares_cohen_d.png`
  - `writings/audit/`: `AUDIT.md`, `audit.py`
  - `writings/auxiliar/scripts/`: `build_results_context.py`, `ensemble_decisions.py`, `fixed_rule.py`, `heldout_gap.py`, `rebuild_thesis.py`, `sync_figures.py`, `trf_vs_ens.py`, `weighted_ens_oracle.py`, `weighted_ens_sensitivity.py`

| Archivo | Sección (definida en Main.tex) | Etiquetas |
|---|---|---|
| `000NHH-Frontpage.tex` | Portada | — |
| `00Abstract.tex` | Resumen (sin numerar) | — |
| `00Abstract_en.tex` | Abstract (sin numerar), dentro de otherlanguage{english} | — |
| `01Introduction.tex` | 1 Introducción | `sec:motivacion`, `sec:problema`, `sec:hypothesis` |
| `02Background.tex` | 2 Marco Teórico | `sec:background`, `sec:mdp-bg`, `eq:markov`, `eq:return`, `eq:bellman-opt`, `sec:value-bg`, `sec:memory-bg`, `sec:entropy-bg`, `eq:shannon`, `sec:ensembles-bg`, `sec:hpo-bg`, `sec:stats-bg` |
| `03StateOfTheArt.tex` | 3 Estado de la Cuestión | `sec:soa`, `sec:soa-seq`, `sec:soa-trf` |
| `04Materials.tex` | 4 Materiales y Métodos | `sec:methods`, `sec:env-dyn`, `eq:state-space`, `eq:state-transition`, `eq:transicion`, `eq:reward`, `sec:metric`, `eq:normalised-severity-materials`, `eq:dp-optimum`, `fig:baseline-random-raw`, `fig:baseline-random-normalised`, `fig:baseline-random`, `tab:packages`, `sec:tabular`, `tab:tabular-hparams`, `sec:deep`, `tab:deep-hparams`, `tab:a2c-hparams`, `sec:ens-methods`, `eq:ens-softmax`, `eq:shannon-norm`, `eq:ens-soft`, `eq:ens-prior`, `tab:ens-params`, `sec:scenario-catalogue`, `tab:scenarios` |
| `05Results.tex` | 5 Resultados | `sec:results`, `sec:res-individual`, `tab:global-mean`, `fig:heatmap-global`, `fig:extra-sev-skew`, `sec:res-ensembles`, `tab:ensemble-stress`, `tab:trf-vs-ens`, `sec:res-freq`, `tab:ens-freq`, `fig:heatmap-ens`, `sec:res-trf-vs-best`, `tab:trf-vs-best`, `fig:ensemble-extrapolation`, `fig:c-ens`, `sec:res-heldout`, `tab:heldout`, `sec:res-posthoc`, `tab:sensitivity`, `eq:rule-dynamics`, `tab:fixed-rule` |
| `06Discussion.tex` | 6 Discusión | `sec:discussion`, `sec:disc-individual`, `sec:ens-best`, `sec:disc-rule` |
| `07Conclusion.tex` | 7 Conclusiones | `sec:conclusion`, `sec:limitations`, `sec:future` |
| `Appendix.tex` | Apéndice | `ap:repro`, `tab:repro-commands`, `ap:orchestrator`, `ap:stat-maps`, `fig:heatmap-welch`, `fig:pairwise-cohen`, `fig:ensemble-statistical-heatmaps`, `fig:ensemble-cohen-scenario`, `fig:ensemble-pairwise`, `ap:per-model`, `fig:c-base`, `fig:c-ql`, `fig:c-dql`, `fig:c-dqn`, `fig:c-rdqn`, `fig:c-a2c`, `fig:c-trf` |

**Secciones y subsecciones** (numeración calculada; etiqueta entre paréntesis):

- `01Introduction.tex` — 1 Introducción: 1.1 Motivación (`sec:motivacion`); 1.2 Definición del problema (`sec:problema`); 1.3 Preguntas e hipótesis (`sec:hypothesis`); 1.4 Contribuciones
- `02Background.tex` — 2 Marco Teórico (`sec:background`): 2.1 Procesos de decisión de Markov (`sec:mdp-bg`); 2.2 Aprendizaje por refuerzo basado en valores (`sec:value-bg`); 2.3 Memoria, atención y actor–crítico (`sec:memory-bg`); 2.4 Entropía como medida de confianza (`sec:entropy-bg`); 2.5 Ensambles (`sec:ensembles-bg`); 2.6 Optimización bayesiana de hiperparámetros (`sec:hpo-bg`); 2.7 Comparación estadística (`sec:stats-bg`)
- `03StateOfTheArt.tex` — 3 Estado de la Cuestión (`sec:soa`): 3.1 Aprendizaje por refuerzo para decisiones secuenciales bajo incertidumbre (`sec:soa-seq`); 3.2 Transformers en aprendizaje por refuerzo (`sec:soa-trf`); 3.3 Ensambles en aprendizaje por refuerzo; 3.4 Generalización bajo variaciones controladas
- `04Materials.tex` — 4 Materiales y Métodos (`sec:methods`): 4.1 Implementación; 4.2 Entorno (`sec:env-dyn`); 4.3 Métrica de desempeño (`sec:metric`); 4.4 Modelos individuales; 4.4.1 Modelos tabulares (`sec:tabular`); 4.4.2 Modelos con redes neuronales (`sec:deep`); 4.5 Ensambles (`sec:ens-methods`); 4.6 Optimización de hiperparámetros; 4.7 Escenarios de generalización (`sec:scenario-catalogue`); 4.8 Protocolo de evaluación
- `05Results.tex` — 5 Resultados (`sec:results`): 5.1 Mejor modelo individual (`sec:res-individual`); 5.2 Ensambles (`sec:res-ensembles`); 5.2.1 Efecto de las reglas fijas sobre la decisión (`sec:res-freq`); 5.3 Transformer frente al mejor ensamble (`sec:res-trf-vs-best`); 5.4 Réplicas fuera de muestra (`sec:res-heldout`); 5.5 Justificación a posteriori del ensamble ponderado (`sec:res-posthoc`)
- `06Discussion.tex` — 6 Discusión (`sec:discussion`): 6.1 Interpretación de la ventaja del Transformer (`sec:disc-individual`); 6.2 Efecto de la regla de combinación (`sec:ens-best`); 6.3 Alcance de los resultados frente a una regla fija (`sec:disc-rule`)
- `07Conclusion.tex` — 7 Conclusiones (`sec:conclusion`): 7.1 Respuesta a las preguntas de investigación; 7.2 Limitaciones del estudio (`sec:limitations`); 7.3 Trabajo futuro (`sec:future`); 7.4 Conclusión general
- `Appendix.tex` — Apéndice: A Comandos de reproducción (`ap:repro`); B Ejecución de la evaluación (`ap:orchestrator`); C Mapas estadísticos complementarios (`ap:stat-maps`); D Resultados por secuencia de cada modelo (`ap:per-model`)

**Figuras de la tesis** (label → archivo en `02_Images/<carpeta>/`; título corto; capítulo):

- `fig:baseline-random` → baseline/random_player_sequence_performance.png + baseline/random_player_normalised_performance.png (Agente aleatorio; 04Materials.tex)
  - `fig:baseline-random-raw` → baseline/random_player_sequence_performance.png (Severidad cruda)
  - `fig:baseline-random-normalised` → baseline/random_player_normalised_performance.png (Desempeño normalizado)
- `fig:heatmap-global` → individual/ind_01_desempeno_por_escenario.png (Desempeño medio por escenario (modelos individuales); 05Results.tex; 5 columnas heldout_s1..s5 a la derecha de una línea vertical)
- `fig:extra-sev-skew` → individual/ind_05_curvas_por_familia.png (Curvas por secuencia en seis escenarios (modelos individuales); 05Results.tex; curvas ordenadas en sev_bimodal, sev_gauss_high, sev_beta_highskew, len_poisson, len_extrapolate_long, joint_high_long; trazo grueso = Transformer)
- `fig:heatmap-ens` → ensemble/ens_01_desempeno_por_escenario.png (Desempeño medio por escenario (ensambles); 05Results.tex; 5 columnas fuera de muestra tras una línea vertical)
- `fig:ensemble-extrapolation` → ensemble/ens_06_curvas_extrapolacion.png (Curvas de los ensambles en los escenarios fuera de rango; 05Results.tex; 3 escenarios fuera de rango; trazo grueso = ensamble ponderado)
- `fig:c-ens` → per_model/PES_ENS_results.png (Resultados por secuencia: ensamble ponderado; 05Results.tex)
- `fig:heatmap-welch` → individual/ind_03_welch_logp_por_escenario.png (Welch por escenario (modelos individuales); Appendix.tex; 5 columnas fuera de muestra tras una línea vertical)
- `fig:pairwise-cohen` → individual/ind_13_pares_cohen_d.png ($d$ de Cohen entre pares (modelos individuales); Appendix.tex)
- `fig:ensemble-statistical-heatmaps` → ensemble/ens_03_welch_logp_por_escenario.png (Welch por escenario (ensambles); Appendix.tex; Welch por escenario (ensambles, página apaisada); 5 columnas fuera de muestra tras una línea vertical)
- `fig:ensemble-cohen-scenario` → ensemble/ens_07_cohen_d_por_escenario.png ($d$ de Cohen por escenario (ensambles); Appendix.tex; d de Cohen por escenario (ensambles, página apaisada); 5 columnas fuera de muestra tras una línea vertical)
- `fig:ensemble-pairwise` → ensemble/ens_13_pares_cohen_d.png + ensemble/ens_12_pares_welch_logp.png (Comparación entre pares (ensambles); Appendix.tex)
- `fig:c-base` → per_model/PES_BASE_results.png (Resultados por secuencia: Q-Learning base; Appendix.tex)
- `fig:c-ql` → per_model/PES_QL_results.png (Resultados por secuencia: Q-Learning; Appendix.tex)
- `fig:c-dql` → per_model/PES_DQL_results.png (Resultados por secuencia: Double Q-Learning; Appendix.tex)
- `fig:c-dqn` → per_model/PES_DQN_results.png (Resultados por secuencia: DQN; Appendix.tex)
- `fig:c-rdqn` → per_model/PES_RDQN_results.png (Resultados por secuencia: DQN recurrente; Appendix.tex)
- `fig:c-a2c` → per_model/PES_A2C_results.png (Resultados por secuencia: A2C; Appendix.tex)
- `fig:c-trf` → per_model/PES_TRF_results.png (Resultados por secuencia: Transformer; Appendix.tex)

**Tablas de la tesis** (label → título corto; capítulo):

- `tab:packages` → Modelos individuales (04Materials.tex)
- `tab:tabular-hparams` → Hiperparámetros de los modelos tabulares (04Materials.tex)
- `tab:deep-hparams` → Hiperparámetros de los modelos de tipo DQN (04Materials.tex)
- `tab:a2c-hparams` → Hiperparámetros de A2C (04Materials.tex)
- `tab:ens-params` → Parámetros de los ensambles (04Materials.tex)
- `tab:scenarios` → Escenarios de evaluación (04Materials.tex)
- `tab:global-mean` → Desempeño de los modelos individuales (05Results.tex)
- `tab:ensemble-stress` → Desempeño de los ensambles (05Results.tex)
- `tab:trf-vs-ens` → Comparación de cada ensamble con el Transformer (05Results.tex)
- `tab:ens-freq` → Frecuencia con que las reglas cambian la decisión (05Results.tex)
- `tab:trf-vs-best` → Transformer frente al ensamble ponderado (05Results.tex)
- `tab:heldout` → Referencia frente a las réplicas fuera de muestra (05Results.tex)
- `tab:sensitivity` → Sensibilidad del ensamble ponderado (05Results.tex)
- `tab:fixed-rule` → Regla fija y modelos frente al óptimo (05Results.tex)
- `tab:repro-commands` → Comandos de reproducción (Appendix.tex)

**Figuras generadas en `h1/general/results/` que la tesis NO usa** (no están en `02_Images/`):

- individual: `02_degradacion_por_escenario`, `04_kl_acciones_por_escenario`, `06_curvas_estresores_universales`, `07_cohen_d_por_escenario`, `08_ranking_desempeno`, `09_degradacion_por_familia`, `10_desempeno_vs_estabilidad`, `11_perfiles_generalizacion`, `12_pares_welch_logp`, `14_pares_kl`, `15_referencia_vs_heldout`, `histogramas/ (subcarpeta)`, `modelos/ (subcarpeta)`, `recompensa/ (subcarpeta)`
- ensemble: `02_degradacion_por_escenario`, `04_kl_acciones_por_escenario`, `05_curvas_por_familia`, `08_ranking_desempeno`, `09_degradacion_por_familia`, `10_desempeno_vs_estabilidad`, `11_perfiles_generalizacion`, `14_pares_kl`, `15_referencia_vs_heldout`, `histogramas/ (subcarpeta)`, `modelos/ (subcarpeta)`, `recompensa/ (subcarpeta)`
- agent_internals: `trf_agent_confidences`, `trf_agent_cumulative_performance`, `trf_agent_normalised_performance`, `trf_agent_remapped_confidences`
- baseline: ninguna
- Cómo incluirlas: Agregar el stem a SUITE_FIGURES en writings/auxiliar/scripts/sync_figures.py y ejecutarlo (sync_figures borra de 02_Images todo lo que no esté en su lista); nombre destino ind_<stem>.png o ens_<stem>.png.

**Scripts auxiliares** (`writings/auxiliar/scripts/`, primera línea de su docstring):

- `build_results_context.py`: Genera ``mpes_resultados.md`` y ``mpes_resultados.json`` (contexto numérico de la tesis).
- `ensemble_decisions.py`: Frecuencia con que las reglas fijas de los ensambles cambian la decisión.
- `fixed_rule.py`: Regla fija ``a = min(S + k, R)`` frente a los modelos, en todos los escenarios.
- `heldout_gap.py`: Caída de cada modelo entre la referencia y sus réplicas fuera de muestra.
- `rebuild_thesis.py`: Rebuild the LaTeX manuscript and regenerate the PDF in writings/out.
- `sync_figures.py`: Sincroniza las figuras de la tesis con las que genera ``h1/``.
- `trf_vs_ens.py`: Comparación de cada ensamble con el Transformer individual en generalización.
- `weighted_ens_oracle.py`: Asignación óptima exacta frente a las reglas fijas del ensamble ponderado.
- `weighted_ens_sensitivity.py`: Sensibilidad uno-a-la-vez de los parámetros fijos del ensamble ponderado.

**Claves bibliográficas disponibles en `References.bib`** (54; únicas citables sin agregar entradas): `BCINE2022`, `Towers2024`, `SuttonBarto2018`, `Watkins1992`, `Hasselt2010`, `Ng1999`, `Mnih2015`, `Lin1992`, `Kingma2015`, `Hasselt2016`, `Huber1964`, `Hausknecht2015`, `Williams1992`, `Mnih2016`, `Vaswani2017`, `Parisotto2020`, `Chen2021`, `Lakshminarayanan2017`, `Bergstra2011`, `Akiba2019`, `Cohen1988`, `Welch1947`, `Kuhl2021`, `ds004477`, `Nijjar2025`, `Henderson2018`, `Cobbe2020`, `Kirk2023`, `mPES2026`, `Glorot2010`, `Shannon1948`, `Schulman2016`, `Wiering2008`, `Parisi2021`, `Kaelbling1998`, `Kochenderfer2015`, `Powell2022`, `DulacArnold2021`, `Libin2021`, `Ohi2020`, `Bastani2021`, `Bednarski2021`, `Komorowski2018`, `Yu2021`, `Mao2016`, `Gijsbrechts2022`, `Kong2019`, `Alshiekh2018`, `Jacobs1991`, `Cawley2010`, `Hochreiter1997`, `Student1908`, `Spearman1904`, `Pascanu2013`.
- Citadas en los capítulos incluidos: 54; sin citar: ninguna; citadas pero ausentes del `.bib`: ninguna.

## 14. Reproducción

- Entorno: win_mpes_env\Scripts\Activate.ps1 desde la raíz; comandos desde h1/; VIRTUAL_ENV, PYTHONIOENCODING=utf-8, TF_ENABLE_ONEDNN_OPTS=0.
- Benchmark (desde `h1/`): `python -m general.scripts.benchmark run --suite both`; `python -m general.scripts.benchmark heldout`; `python -m general.scripts.analysis`; `python -m general.scripts.figures`; `python -m general.scripts.random_baseline`; `python -m general.scripts.agent_internals`
- Tesis (desde la raíz): `cd writings`; `python audit\audit.py`; `python audit\audit.py --no-tex`; `python auxiliar\scripts\sync_figures.py`; `python auxiliar\scripts\ensemble_decisions.py --output ..\h1\general\results\ensemble\ens_decisions.json`; `python auxiliar\scripts\heldout_gap.py`; `python auxiliar\scripts\trf_vs_ens.py`; `python auxiliar\scripts\build_results_context.py`
- Réplicas fuera de muestra: run --suite both evalúa también heldout_s1..s5 (65 celdas: 13 modelos × 5); benchmark heldout escribe general/results/heldout/ (CSV + sampling_distribution.json por réplica, heldout_catalogue.json con sha256 y verificación de copias en los 13 paquetes); python writings/auxiliar/scripts/heldout_gap.py (desde la raíz, como en el Apéndice ap:orchestrator; independiente del directorio) escribe h1/general/results/heldout/heldout_gap.json (Tabla tab:heldout).
- Contexto para LLM: python writings/auxiliar/scripts/ensemble_decisions.py --output h1/general/results/ensemble/ens_decisions.json (desde la raíz; replay con TensorFlow) y luego python writings/auxiliar/scripts/build_results_context.py (regenera mpes_resultados.md y .json; --check compara con los archivos en disco).
