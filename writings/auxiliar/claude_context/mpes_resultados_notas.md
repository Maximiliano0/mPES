# Notas editables de `mpes_resultados.{md,json}`

Este archivo contiene el texto redactado a mano que
`writings/auxiliar/scripts/build_results_context.py` inserta en
`mpes_resultados.md` y `mpes_resultados.json`. Todo lo demás (tablas, métricas,
hiperparámetros, mapa del documento LaTeX, estado de la auditoría) se calcula
desde el repositorio en cada ejecución.

Reglas:

- Cada bloque empieza con una línea `## [nombre]` y termina en el bloque
  siguiente. No cambiar los nombres: el generador los busca por nombre.
- Las líneas `<!-- … -->` (también de varias líneas) son comentarios y se ignoran.
- Tipos de bloque: texto libre; lista (`- elemento`, una línea por elemento);
  diccionario (`- clave: valor`); tabla Markdown (la primera fila es el
  encabezado). Un valor `@nombre` inserta el bloque `nombre`.
- **Ningún número medido se escribe a mano.** Se usa un marcador
  `{{ruta|formato}}`: `ruta` es una clave del JSON generado separada por `/`
  (p. ej. `modelos_individuales/pes_trf/referencia_media`) o un valor derivado
  `_/...` (lista en la docstring del generador); `formato` es un formato de
  Python (`.3f`, `+.4f`, `d`…) o uno propio: `s2`/`s3`/`s4` (con signo y
  `0.00…` para el cero), `pct0`/`pct1` (×100), `miles` (miles separados por
  espacio), `sup` (potencia de diez `10⁻⁷` que acota un p), `e1` (`7.5e-08`).
  Una coma al final del formato (`.3f,`) usa coma decimal y `−` al final, el signo menos tipográfico.
  `{{md:texto|json:texto}}` escribe un texto distinto en cada archivo.
- Un marcador desconocido o una ruta inexistente detiene el generador.
- Tras editar: `python writings/auxiliar/scripts/build_results_context.py`
  (y `--check` para comprobar que los archivos en disco están al día).

## [encabezado]
Fuentes: `h1/general/results/` (`summary.json`, `comparison_metrics.json`, `matrices/`, `cells/`, `heldout/`), las decisiones del replay de `writings/auxiliar/scripts/ensemble_decisions.py`, los `inputs/best_params.json`, `inputs/*_BAYESIAN_OPT/` y `config/CONFIG.py` de cada paquete, y los `.tex` de `writings/`. Es la **fuente de verdad numérica** para redactar la tesis. Los números usan punto decimal; en LaTeX se escriben con coma (`$0{,}927$`). El archivo `mpes_resultados.json` contiene los mismos datos con más precisión y por celda. El texto redactado a mano vive en `writings/auxiliar/claude_context/mpes_resultados_notas.md`.

## [meta]
- titulo_tesis: mPES: Esquemas para Toma de Decisión Artificial en Escenarios Secuenciales
- subtitulo: Una evaluación comparativa de arquitecturas de Aprendizaje por Refuerzo para la asignación de recursos limitados bajo incertidumbre utilizando como escenario de prueba una crisis pandémica
- autor: Ing. Maximiliano Leonel Vega
- director: Dr. Ing. Rodrigo Ramele
- programa: Maestría en Ciencia de Datos, Instituto Tecnológico de Buenos Aires (ITBA), 2026

## [convenciones]
- metrica: rbar = (S_peor - S_agente) / (S_peor - S_mejor); 0 = no asignar, 1 = óptimo (S_mejor por programación dinámica exacta, mochila acotada 0..{{_/max_asignacion}} por ciudad, suma <= {{_/presupuesto_agente}}); rbar <= 1.
- referencia: escenario sev_base (distribución de entrenamiento), n = {{escenarios/sev_base/n_secuencias}} secuencias, {{escenarios/sev_base/n_pasos}} pasos.
- generalizacion: media de las medias de los {{_/n_gen}} escenarios distintos de sev_base, sin las {{_/n_heldout}} réplicas fuera de muestra (incluye los {{_/n_estructurales}} estructurales, que reproducen la referencia).
- degradacion: deg = mu(sev_base) - mu(escenario); positiva = pérdida; negativa = mejora.
- degradacion_media: promedio con signo de deg sobre los {{_/n_gen}} escenarios.
- cohen_d_celda: d del escenario frente a la referencia del mismo modelo; positivo = mejor en el escenario.
- log10p_celda: log10 del p bilateral de Welch escenario vs referencia del mismo modelo.
- kl_acc: KL entre la distribución de acciones ({{_/n_acciones}}) del escenario y la de la referencia del mismo modelo (desplazamiento de política, no de desempeño).
- pares: Comparaciones entre modelos de un mismo grupo juntando las secuencias de los {{_/n_gen}} escenarios de generalización (sin sev_base). d > 0: el primer modelo del par supera al segundo. kl_sym: KL simetrizada entre histogramas de desempeño ({{_/kl_bins}} bins en [0,1], eps = {{_/kl_eps}}).
- claves_celda: @convenciones_claves_celda
- derivado: Campos con prefijo derivado_ no aparecen en la tesis; se calcularon de las mismas matrices.
- estadistica: Sin corrección por comparaciones múltiples; secuencias comunes a todos los modelos (misma semilla), no independientes: los p son exploratorios; priorizar d.
- replicas_fuera_de_muestra: heldout_s1..heldout_s{{_/n_heldout}}: {{_/n_heldout}} réplicas i.i.d. de la referencia (ver replicas_fuera_de_muestra); excluidas de generalización ({{_/n_gen}}), peor escenario, familias, pares y conteos; no se copian a celdas ni a ganadores_por_escenario.

## [convenciones_claves_celda]
- mu: media rbar
- sd: desviación estándar entre secuencias
- min: mínimo
- max: máximo
- deg: degradación vs referencia
- d: Cohen d vs referencia
- log10p: log10 p de Welch vs referencia
- kl_acc: KL de acciones vs referencia
- f_opt: fracción de secuencias con rbar = 1
- f_lt08: fracción con rbar < 0,8
- a_media: asignación media por paso (incluye pasos sin recursos)
- n: nº de secuencias

## [resumen_ejecutivo]
- **Transformer (`pes_trf`) = mejor modelo individual.** Referencia {{modelos_individuales/pes_trf/referencia_media|.3f}} (σ {{modelos_individuales/pes_trf/referencia_sd|.3f}}, la menor entre individuales), generalización {{modelos_individuales/pes_trf/generalizacion_media_21|.3f}} (resto de optimizados entre {{_/gen_otros_opt_min|.3f}} y {{_/gen_otros_opt_max|.3f}}); mayor media en {{hechos_derivados/transformer_gana_individuales_exacto/n}} de {{_/n_ref_gen}} escenarios. Reduce la distancia al óptimo de Q-Learning en {{modelos_individuales/pes_trf/delta_rel_vs_ql_referencia|pct0}} % en la referencia. d = {{pares_citados_en_tesis/Transformer vs Q-Learning/d|.2f}} vs Q-Learning, {{pares_citados_en_tesis/Transformer vs DQN recurrente/d|.2f}} vs DQN recurrente, {{pares_citados_en_tesis/Transformer vs DQN/d|.2f}} vs DQN. → Respalda **H1**.
- **No es el que menos pierde en su peor caso**: peor escenario `{{modelos_individuales/pes_trf/peor_escenario}}` {{modelos_individuales/pes_trf/peor_escenario_media|.3f}} (degradación {{modelos_individuales/pes_trf/mayor_degradacion|.3f}}) frente a {{modelos_individuales/pes_dqn/mayor_degradacion|.3f}} de DQN y {{modelos_individuales/pes_a2c/mayor_degradacion|.3f}} de A2C.
- **Tabulares colapsan con severidad fuera de rango** (`sev_extrapolate_high`): Q-Learning base {{celdas/pes_base/sev_extrapolate_high/mu|.3f}} (degradación {{celdas/pes_base/sev_extrapolate_high/deg|.3f}}, peor celda del estudio), Q-Learning {{celdas/pes_ql/sev_extrapolate_high/mu|.3f}}, Double Q-Learning {{celdas/pes_dql/sev_extrapolate_high/mu|.3f}}: la tabla recorta S a {{_/max_severidad}}, fila nunca entrenada.
- **Las redes mantienen el desempeño en ese escenario**: Transformer {{celdas/pes_trf/sev_extrapolate_high/mu|.3f}} (y {{celdas/pes_trf/joint_extrap_both/mu|.3f}} en la conjunta), A2C {{celdas/pes_a2c/sev_extrapolate_high/mu|.3f}}, DQN {{celdas/pes_dqn/sev_extrapolate_high/mu|.3f}}; excepción: DQN recurrente {{celdas/pes_rdqn/sev_extrapolate_high/mu|.3f}} (su peor escenario). Advertencia: con S = 10..12 asignar mucho acerca al óptimo.
- **Secuencias largas (`len_extrapolate_long`) = peor escenario de {{_/ind_peor_len_long}} y de {{_/n_ens_peor_len_long}} ensambles.** Allí el mejor individual es el DQN recurrente ({{celdas/pes_rdqn/len_extrapolate_long/mu|.3f}} vs {{celdas/pes_trf/len_extrapolate_long/mu|.3f}} del Transformer): único escenario fuera de rango en que el Transformer no es primero.
- **Escenarios estructurales** reproducen exactamente la referencia en {{_/n_struct_identicos}} de los {{_/n_modelos}} modelos (control correcto).
- **Ensamble ponderado (`pes_ens`) = único sistema que supera al Transformer**: referencia {{ensambles/pes_ens/referencia_media|.3f}} (σ {{ensambles/pes_ens/referencia_sd|.3f}}, la menor de todo el estudio), generalización {{ensambles/pes_ens/generalizacion_media_21|.3f}}, peor escenario {{ensambles/pes_ens/peor_escenario_media|.3f}} ({{hechos_derivados/pes_ens_peor_menos_trf_peor|s3}} sobre el del Transformer), mayor degradación {{ensambles/pes_ens/mayor_degradacion|.3f}} (la menor). Δrel vs Q-Learning {{ensambles/pes_ens/delta_rel_vs_ql_referencia|pct0}} %. Óptimo en todas las secuencias de `sev_extrapolate_high` y `joint_extrap_both` ({{celdas/pes_ens/sev_extrapolate_high/mu|.3f,}}; σ {{celdas/pes_ens/sev_extrapolate_high/sd|g}}). d = {{pares_citados_en_tesis/Ensamble ponderado vs Consenso/d|.2f}} vs consenso y {{pares_citados_en_tesis/Ensamble ponderado vs Compuerta del Transformer/d|.2f}} vs compuerta; frente al Transformer en generalización d = {{hechos_derivados/ensambles_vs_transformer_generalizacion/pes_ens/cohen_d|.2f}} (despreciable, p = {{hechos_derivados/ensambles_vs_transformer_generalizacion/pes_ens/welch_p|e1}}).
- **Pero su acción final difiere de la del Transformer en el {{decisiones_ensambles/pes_ens/accion_final_distinta_del_transformer/referencia|.1f,}} % (referencia) / {{decisiones_ensambles/pes_ens/accion_final_distinta_del_transformer/generalizacion|.1f,}} % (generalización)** de las decisiones; el prior de severidad cambia la acción votada en el {{decisiones_ensambles/pes_ens/prior_cambia_accion_votada/referencia|.1f,}} % / {{decisiones_ensambles/pes_ens/prior_cambia_accion_votada/generalizacion|.1f,}} %; la cota casi nunca actúa. Con τ = {{ensambles/pes_ens/definicion/parametros/tau|g}} las distribuciones de los miembros son casi planas. La mejora es compatible con prior + temperatura, no con la ponderación por confianza.
- **Compuerta del Transformer ≈ Transformer**: generalización {{ensambles/pes_ens_trf_guard/generalizacion_media_21|.3f}} vs {{modelos_individuales/pes_trf/generalizacion_media_21|.3f}}; sigue al Transformer en el {{decisiones_ensambles/pes_ens_trf_guard/sigue_al_transformer_g>=0.5/referencia|.1f,}} % / {{decisiones_ensambles/pes_ens_trf_guard/sigue_al_transformer_g>=0.5/generalizacion|.1f,}} % de las decisiones.
- **Voto suave y voto por acción**: generalización más baja ({{ensambles/pes_ens_sprb/generalizacion_media_21|.3f}} / {{ensambles/pes_ens_accq/generalizacion_media_21|.3f}}), peor escenario `sev_extrapolate_high` ({{celdas/pes_ens_sprb/sev_extrapolate_high/mu|.3f}} / {{celdas/pes_ens_accq/sev_extrapolate_high/mu|.3f}}), con {{celdas/pes_ens_sprb/sev_extrapolate_high/f_lt08|pct0}} % / {{celdas/pes_ens_accq/sev_extrapolate_high/f_lt08|pct0}} % de secuencias bajo 0,8. Sus pesos dan más peso al DQN recurrente que al Transformer.
- **Consenso** tiene la referencia más baja de los ensambles ({{ensambles/pes_ens_consensus/referencia_media|.3f}}); la variante con prior mejora referencia ({{ensambles/pes_ens_consensus_prior/referencia_media|.3f}}) y peor escenario ({{ensambles/pes_ens_consensus/peor_escenario_media|.3f}} → {{ensambles/pes_ens_consensus_prior/peor_escenario_media|.3f}}), aunque su prior sólo cambia el {{decisiones_ensambles/pes_ens_consensus_prior/prior_cambia_accion_votada/referencia|.1f,}} % / {{decisiones_ensambles/pes_ens_consensus_prior/prior_cambia_accion_votada/generalizacion|.1f,}} % de las decisiones.
- **Agente aleatorio**: {{agente_aleatorio/rbar_media|.3f}} (cota inferior de referencia).
- **Réplicas fuera de muestra** (§10bis, Tabla `tab:heldout`): en {{replicas_fuera_de_muestra/total/n_secuencias}} secuencias nuevas sorteadas de la distribución de la referencia ningún modelo pierde desempeño apreciable: caída (ref. − réplicas) de {{replicas_fuera_de_muestra/hechos_derivados/caida_min/valor|s3}} a {{replicas_fuera_de_muestra/hechos_derivados/caida_max/valor|s3}} (|d| ≤ {{replicas_fuera_de_muestra/hechos_derivados/max_abs_d|.2f}}, p ≥ {{replicas_fuera_de_muestra/hechos_derivados/min_welch_p|.2f}}); en los {{_/ho_n_optuna}} optimizados con Optuna entre {{replicas_fuera_de_muestra/hechos_derivados/caida_rango_11_optuna/0|s3}} y {{replicas_fuera_de_muestra/hechos_derivados/caida_rango_11_optuna/1|s3}}; la mayor es la de {{_/ho_caida_max_nombre}}{{_/ho_caida_max_nota}}. Spearman ρ = {{replicas_fuera_de_muestra/spearman_referencia_vs_fuera_de_muestra/rho|.2f}} entre medias. El ensamble ponderado supera al Transformer por {{replicas_fuera_de_muestra/ensamble_ponderado_vs_transformer/diferencia_media|.3f}} (t pareada p < {{replicas_fuera_de_muestra/ensamble_ponderado_vs_transformer/t_pareada_p|sup}}) y en {{replicas_fuera_de_muestra/ensamble_ponderado_vs_transformer/replicas_ganadas}} de las {{replicas_fuera_de_muestra/ensamble_ponderado_vs_transformer/n_replicas}} réplicas. → La selección sobre las {{escenarios/sev_base/n_secuencias}} secuencias no infló la referencia.

## [definiciones]
- **Desempeño normalizado** por secuencia: $\bar r = (S_{peor} - S_{agente}) / (S_{peor} - S_{mejor})$. $0$ = no asignar recursos; $1$ = asignación óptima ($S_{mejor}$ exacto por programación dinámica, mochila acotada con asignaciones 0..{{_/max_asignacion}} por ciudad que suman ≤ {{_/presupuesto_agente}}). $\bar r \le 1$. No es una probabilidad.
- **Referencia** = escenario `sev_base` (distribución de entrenamiento), $n = {{escenarios/sev_base/n_secuencias}}$ secuencias ({{_/ref_bloques}} bloques × {{_/ref_secuencias_bloque}}), {{escenarios/sev_base/n_pasos}} pasos.
- **Generalización** = media de las medias de los **{{_/n_gen}}** escenarios distintos de la referencia, sin las {{_/n_heldout}} réplicas fuera de muestra (incluye {{_/n_estructurales}} estructurales que reproducen exactamente la referencia y escenarios más fáciles, p. ej. secuencias cortas).
- **Degradación** de una celda = $\bar r_{ref} - \bar r_{esc}$ (positiva = pérdida; negativa = mejora). **Degradación media** = promedio con signo sobre los {{_/n_gen}} escenarios. **Mayor degradación** = máximo de esas {{_/n_gen}}.
- **Por celda frente a la propia referencia**: $d$ de Cohen (positivo = mejor en el escenario), $\log_{10} p$ de Welch (bilateral) y KL de acciones (desplazamiento de la política sobre las {{_/n_acciones}} acciones).
- **Entre pares de modelos** (mismo grupo): se juntan las secuencias de los {{_/n_gen}} escenarios de generalización (sin `sev_base`); $d > 0$ ⇒ gana el modelo de la fila; KL simetrizada entre histogramas de desempeño ({{_/kl_bins}} bins en [0, 1], $\varepsilon = {{_/kl_eps_tex}}$).
- **Δrel vs Q-Learning** (fórmula en línea de `sec:res-ensembles`): $(\bar r_m - \bar r_{ql}) / (1 - \bar r_{ql})$, reducción relativa de la distancia al óptimo.
- Advertencia estadística: todas las celdas usan las mismas secuencias (semilla {{_/semilla_escenarios}}), no son independientes; no hay corrección por comparaciones múltiples. Los $p$ son exploratorios; la medida principal es $d$.

## [entorno]
- estado: (R, t, S): R en 0..{{_/presupuesto_agente}} recursos restantes, t en 0..{{_/max_pasos}} paso, S en 0..{{_/max_severidad}} severidad de la ciudad actual; {{_/dim_r}}x{{_/dim_t}}x{{_/dim_s}} = {{_/n_estados|miles}} estados (observación parcial).
- acciones: 0..{{_/max_asignacion}} unidades; una asignación mayor que R se recorta.
- presupuesto: {{_/presupuesto_total}} recursos por secuencia, {{_/preasignados}} preasignados por el entorno -> el agente administra {{_/presupuesto_agente}}.
- transicion: S_{i,t+1} = max(0, {{_/beta|.1f,}}·S_{i,t} - {{_/alpha|.1f,}}·a_i) para toda ciudad ya incorporada (alpha = {{_/alpha|.1f,}}, beta = {{_/beta|.1f,}}); la asignación de cada ciudad se fija al llegar.
- recompensa: r_t = -sum_{i<=t} S_{i,t+1}.
- estructura: {{_/ref_bloques}} bloques x {{_/ref_secuencias_bloque}} secuencias, longitudes {{_/min_pasos}}..{{_/max_pasos}}, {{escenarios/sev_base/n_pasos}} pasos; archivos initial_severity.csv y sequence_lengths.csv idénticos en todos los paquetes.
- entrenamiento: cada episodio sortea longitud y severidades de sus distribuciones empíricas en esos archivos: sólo severidades {{_/sev_ref_min}}..{{_/sev_ref_max}} (filas S = 0, 1, 9 de las tablas Q quedan aleatorias).
- entrada_redes: s = [R/{{_/presupuesto_agente}}, t/{{_/max_pasos}}, S/{{_/max_severidad}}]; A2C y pes_ens recortan S a {{_/max_severidad}}; DQN, DQN recurrente, Transformer y los otros cinco ensambles no recortan (S/{{_/max_severidad}} > 1 en sev_extrapolate_high).
- evaluacion: determinista (epsilon = 0, acción factible de mayor valor); sólo inferencia, sin reentrenar.
- software: Python {{_/ver/python}}, TensorFlow {{_/ver/tensorflow}}, Keras {{_/ver/keras}}, NumPy {{_/ver/numpy}}, Optuna {{_/ver/optuna}}, Gymnasium {{_/ver/gymnasium}}, SciPy {{_/ver/scipy}}.

## [protocolo_md]
- **Escenarios**: generados con semilla {{_/semilla_escenarios}}; dentro de cada escenario todos los modelos enfrentan las mismas secuencias. Los modelos individuales y los ensambles se evalúan como dos grupos (Q-Learning base se agrega como referencia al grupo individual).

## [agente_aleatorio]
- descripcion: Agente que en cada paso elige a ~ U{0..{{_/max_asignacion}}}, recortada al presupuesto restante ({{_/presupuesto_agente}}), sobre las {{agente_aleatorio/n}} secuencias de referencia; semilla {{_/semilla_aleatorio}}. Calculado con general.scripts.random_baseline.run_random_player (sin escribir figuras).
- figura: fig:baseline-random (random_player_sequence_performance.png, random_player_normalised_performance.png)

## [paquetes]
| Paquete | Nombre en el texto | Abreviatura |
|---|---|---|
| pes_base | Q-Learning base | QL-base |
| pes_ql | Q-Learning | QL |
| pes_dql | Double Q-Learning | DQL |
| pes_dqn | DQN | DQN |
| pes_rdqn | DQN recurrente | RDQN |
| pes_a2c | A2C | A2C |
| pes_trf | Transformer | TRF |
| pes_ens | Ensamble ponderado | ENS |
| pes_ens_sprb | Voto suave | VS |
| pes_ens_accq | Voto por acción | VA |
| pes_ens_consensus | Consenso | CONS |
| pes_ens_consensus_prior | Consenso con prior | CONS+P |
| pes_ens_trf_guard | Compuerta del Transformer | GUARD |

## [hiper.pes_base]
- tipo: Q-Learning tabular
- tabla_Q: {{_/tabla_q}}, init U(-1,1)
- decaimiento_epsilon: lineal
- optimizacion_bayesiana: no (valores originales de PES)

## [hiper.pes_ql]
- tipo: Q-Learning tabular optimizado
- decaimiento_epsilon: lineal

## [hiper.pes_dql]
- tipo: Double Q-Learning tabular + PBRS + calentamiento de epsilon
- decaimiento_epsilon: exponencial tras calentamiento
- potencial_pbrs: Phi(s) = -sum_i S_i

## [hiper.pes_dqn]
- tipo: Double DQN, red densa
- arquitectura: Input({{_/dim_entrada}}) -> {{_/arq/dqn_capas}} -> Dense({{_/n_acciones}})
- arquitectura_origen: optimización bayesiana

## [hiper.pes_rdqn]
- tipo: Double DQN con codificador LSTM (esquema DRQN)
- arquitectura: ventana W={{_/cfg/pes_rdqn/RDQN_HISTORY_LEN}} estados -> LSTM({{_/cfg/pes_rdqn/RDQN_LSTM_UNITS}}) -> {{_/arq/rdqn_capas}} -> Dense({{_/n_acciones}})
- arquitectura_origen: exploración ad hoc, fijada en config/CONFIG.py (NO optimizada)
- nota: Sólo los hiperparámetros de entrenamiento vienen de la optimización bayesiana. NO citar número de ensayos, score de Optuna ni los campos de arquitectura de su best_params.json (no corresponden al modelo desplegado).

## [hiper.pes_a2c]
- tipo: Advantage Actor-Critic (redes separadas)
- arquitectura: actor Input({{_/dim_entrada}})->{{_/arq/a2c_actor_capas}}->Dense({{_/n_acciones}},softmax); crítico Input({{_/dim_entrada}})->{{_/arq/a2c_critico_capas}}->Dense(1)
- decaimiento_lr: coseno hasta {{_/a2c_lr_min_pct|.2f,}} % del inicial
- penalizacion_gasto: r <- r - {{_/a2c_coef_gasto|.4g,}}·a
- nota: Recorta la severidad a {{_/max_severidad}} antes de escalar la entrada.

## [hiper.pes_trf]
- tipo: Double DQN con codificador Transformer causal (Pre-LN)
- arquitectura: ventana W={{_/cfg/pes_trf/TRF_HISTORY_LEN}} -> proyección d_model={{_/cfg/pes_trf/TRF_D_MODEL}} + vector de posición fijo (Glorot, no entrenado) -> {{_/cfg/pes_trf/TRF_NUM_LAYERS}} bloques [atención causal {{_/cfg/pes_trf/TRF_NUM_HEADS}} cabezas key_dim {{_/cfg/pes_trf/TRF_KEY_DIM}}; FFN {{_/cfg/pes_trf/TRF_FF_DIM}}; residual; LayerNorm previa] -> última posición -> {{_/arq/trf_capas}} -> Dense({{_/n_acciones}}); dropout {{_/cfg/pes_trf/TRF_DROPOUT|g}}
- arquitectura_origen: exploración ad hoc, fijada en config/CONFIG.py (NO optimizada)
- nota: Sólo los hiperparámetros de entrenamiento vienen de la optimización bayesiana. NO citar número de ensayos, score de Optuna ni los campos de arquitectura de su best_params.json (no corresponden al modelo desplegado).

## [origen_hiperparametros]
- pes_base: valores originales de PES (sin optimizar)
- pes_ql: Optuna/TPE, {{modelos_individuales/pes_ql/hiperparametros/ensayos_optuna}} ensayos, mejor #{{modelos_individuales/pes_ql/hiperparametros/mejor_ensayo}}
- pes_dql: Optuna/TPE, {{modelos_individuales/pes_dql/hiperparametros/ensayos_optuna}} ensayos ({{modelos_individuales/pes_dql/hiperparametros/ensayos_podados_median_pruner}} podados)
- pes_dqn: Optuna/TPE, {{modelos_individuales/pes_dqn/hiperparametros/ensayos_optuna}} ensayos, mejor #{{modelos_individuales/pes_dqn/hiperparametros/mejor_ensayo}} (arquitectura incluida)
- pes_rdqn: BO sólo de hiperparámetros de entrenamiento; arquitectura ad hoc
- pes_a2c: Optuna/TPE, {{modelos_individuales/pes_a2c/hiperparametros/ensayos_optuna}} ensayos, mejor #{{modelos_individuales/pes_a2c/hiperparametros/mejor_ensayo}}
- pes_trf: BO sólo de hiperparámetros de entrenamiento; arquitectura ad hoc

## [individuales_tabulares]
**Tabulares** (Tabla `tab:tabular-hparams`): Q-Learning base α {{modelos_individuales/pes_base/hiperparametros/alpha|g,}} · γ {{modelos_individuales/pes_base/hiperparametros/gamma|g,}} · ε {{modelos_individuales/pes_base/hiperparametros/epsilon_0|g,}}→{{modelos_individuales/pes_base/hiperparametros/epsilon_min|g,}} lineal · {{modelos_individuales/pes_base/hiperparametros/episodios|miles}} episodios. Q-Learning α {{modelos_individuales/pes_ql/hiperparametros/alpha|.4f,}} · γ {{modelos_individuales/pes_ql/hiperparametros/gamma|.4f,}} · ε {{modelos_individuales/pes_ql/hiperparametros/epsilon_0|.4f,}}→{{modelos_individuales/pes_ql/hiperparametros/epsilon_min|.4f,}} lineal · {{modelos_individuales/pes_ql/hiperparametros/episodios|miles}} episodios. Double Q-Learning α {{modelos_individuales/pes_dql/hiperparametros/alpha|.4f,}} · γ {{modelos_individuales/pes_dql/hiperparametros/gamma|.4f,}} · ε {{modelos_individuales/pes_dql/hiperparametros/epsilon_0|.4f,}}→{{modelos_individuales/pes_dql/hiperparametros/epsilon_min|.4f,}} exponencial con calentamiento w {{modelos_individuales/pes_dql/hiperparametros/w_calentamiento|.4f,}} y fracción q {{modelos_individuales/pes_dql/hiperparametros/q_fraccion_epsilon_min|.4f,}} · PBRS κ {{modelos_individuales/pes_dql/hiperparametros/pbrs_kappa|g,}} (Φ(s) = −Σ S_i) · {{modelos_individuales/pes_dql/hiperparametros/episodios|miles}} episodios. Semilla de entrenamiento = 42 + i + 1 (i = índice del mejor ensayo).

## [individuales_dqn_intro]
**Tipo DQN** (Tabla `tab:deep-hparams`; blanco Double DQN, enmascarado de acciones no factibles, pérdida de Huber, Adam, Glorot uniforme, recorte de gradiente, ε-greedy con calentamiento y decaimiento exponencial):

## [individuales_notas]
- DQN: `Input({{_/dim_entrada}}) → {{_/arq/dqn_capas_md}} → Dense({{_/n_acciones}})`, replay + red objetivo.
- DQN recurrente: ventana W = {{_/cfg/pes_rdqn/RDQN_HISTORY_LEN}} (relleno con ceros al inicio), LSTM({{_/cfg/pes_rdqn/RDQN_LSTM_UNITS}}), último estado oculto → {{_/arq/rdqn_capas_corto}} → {{_/n_acciones}} Q; el estado oculto no se conserva entre decisiones.
- Transformer: W = {{_/cfg/pes_trf/TRF_HISTORY_LEN}}, proyección a d_model = {{_/cfg/pes_trf/TRF_D_MODEL}} + vector de posición fijo (Glorot, no entrenado), {{_/cfg/pes_trf/TRF_NUM_LAYERS}} bloques Post-LN sin las compuertas de Parisotto (atención causal {{_/cfg/pes_trf/TRF_NUM_HEADS}} cabezas de dim. {{_/cfg/pes_trf/TRF_KEY_DIM}} + FFN {{_/cfg/pes_trf/TRF_FF_DIM}}, residual), sin dropout, última posición → {{_/arq/trf_capas_md}} → {{_/n_acciones}} Q.
- **RDQN y TRF**: arquitectura elegida por exploración *ad hoc* (optimizarla con BO era demasiado costoso); sólo sus hiperparámetros de entrenamiento vienen de la BO. No citar nº de ensayos, score de Optuna ni los campos de arquitectura de sus `best_params.json`.
- **A2C** (Tabla `tab:a2c-hparams`): actor `Input({{_/dim_entrada}})→{{_/arq/a2c_actor_corto}}→Dense({{_/n_acciones}}, softmax)`, crítico `Input({{_/dim_entrada}})→{{_/arq/a2c_critico_corto}}→Dense(1)`; lr actor {{modelos_individuales/pes_a2c/hiperparametros/lr_actor|g,}} / crítico {{modelos_individuales/pes_a2c/hiperparametros/lr_critico|g,}}, decaimiento coseno hasta {{_/a2c_lr_min_pct|.2f,}} %, γ {{modelos_individuales/pes_a2c/hiperparametros/gamma|.4f,}}, β_H {{modelos_individuales/pes_a2c/hiperparametros/coef_entropia|g,}}, GAE λ {{modelos_individuales/pes_a2c/hiperparametros/gae_lambda|.4f,}}, recorte {{modelos_individuales/pes_a2c/hiperparametros/clip_grad|.3f,}}, PBRS κ {{modelos_individuales/pes_a2c/hiperparametros/pbrs_kappa|.4f,}}, penalización de gasto r ← r − {{_/a2c_coef_gasto|.4g,}}·a, sesgo inicial del logit de a = {{_/max_asignacion}}: {{modelos_individuales/pes_a2c/hiperparametros/sesgo_logit_a10|.3f,−}}, {{modelos_individuales/pes_a2c/hiperparametros/episodios|miles}} episodios, semilla {{modelos_individuales/pes_a2c/hiperparametros/semilla}}.
- Q-Learning, Double Q-Learning, DQN y A2C {{_/reproducen_score}} el score de su mejor ensayo de Optuna ({{modelos_individuales/pes_ql/hiperparametros/score_optuna|.4f,}} / {{modelos_individuales/pes_dql/hiperparametros/score_optuna|.4f,}} / {{modelos_individuales/pes_dqn/hiperparametros/score_optuna|.4f,}} / {{modelos_individuales/pes_a2c/hiperparametros/score_optuna|.4f,}}).

## [ens.pes_ens]
- regla: Voto suave ponderado por (0,1 + c_k) sobre DQN, DQN recurrente y Transformer; factor 0,3 a a=0 si R>0; mezcla con prior de severidad gaussiano; argmax; cota de seguridad floor(S/2) si S>=6.
- a2c: configurado pero deshabilitado
- origen_parametros: fijados a mano sobre la referencia (commit d2f0c49, sin búsqueda sistemática); justificados a posteriori en sec:res-posthoc (sensibilidad, óptimo y regla fija)
- confianza: 1 - H/log2({{_/n_acciones}}) (normaliza por {{_/n_acciones}} acciones)
- nota: Recorta la severidad a {{_/max_severidad}} para miembros y prior. Sólo inferencia.

## [ens.pes_ens_sprb]
- regla: Voto suave: promedio de p_k con pesos efectivos w_k·c_k^rho; argmax.
- origen_parametros: mejor ensayo de Optuna/TPE sobre las {{escenarios/sev_base/n_secuencias}} secuencias de referencia (inputs/best_params.json)
- confianza: 1 - H_norm, normalizada por log(nº de acciones factibles)

## [ens.pes_ens_accq]
- regla: Voto por acción: cada miembro vota su argmax con peso w_k·c_k^rho; desempate por suma de Q estandarizados.
- origen_parametros: mejor ensayo de Optuna/TPE sobre las {{escenarios/sev_base/n_secuencias}} secuencias de referencia (inputs/best_params.json)
- confianza: 1 - H_norm, normalizada por log(nº de acciones factibles)

## [ens.pes_ens_consensus]
- regla: Voto ponderado + beta_a·(confianza de quienes coinciden) - beta_d·(confianza de quienes discrepan) + sum_k Qhat_k(a)·c_k; argmax.
- origen_parametros: mejor ensayo de Optuna/TPE sobre las {{escenarios/sev_base/n_secuencias}} secuencias de referencia (inputs/best_params.json)
- confianza: 1 - H_norm, normalizada por log(nº de acciones factibles)

## [ens.pes_ens_consensus_prior]
- regla: Como consenso pero con suma de distribuciones; factor 0,3 a a=0 si R>0; mezcla con prior de severidad; cota de seguridad floor(S/2) si S>=6. Confianza c_k = 1 − H_norm(p_k), igual que el resto (corregida el 2026-09-28: antes se aplicaba una segunda softmax a p_k).
- origen_parametros: mejor ensayo de Optuna/TPE sobre las {{escenarios/sev_base/n_secuencias}} secuencias de referencia (inputs/best_params.json)
- confianza: 1 - H_norm, normalizada por log(nº de acciones factibles)

## [ens.pes_ens_trf_guard]
- regla: Compuerta logística g = sigmoid(kappa_g·(c_trf - tau_g)); si g>=0,5 ejecuta la acción del Transformer, si no un voto suave de respaldo.
- origen_parametros: mejor ensayo de Optuna/TPE sobre las {{escenarios/sev_base/n_secuencias}} secuencias de referencia (inputs/best_params.json)
- confianza: 1 - H_norm, normalizada por log(nº de acciones factibles)

## [ensambles_notas]
- Todos: $p_k$ = softmax de temperatura τ de los Q (o salida del actor de A2C), renormalizada sobre acciones factibles; confianza $c_k = 1 - H_{norm}(p_k)$; peso efectivo $\tilde w_k = w_k c_k^{\rho}$ (salvo `pes_ens`, que usa $w_k(0{,}1 + c_k)$).
- `pes_ens`: pesos base {{ensambles/pes_ens/definicion/parametros/w_dqn|.2f,}} / {{ensambles/pes_ens/definicion/parametros/w_rdqn|.2f,}} / {{ensambles/pes_ens/definicion/parametros/w_trf|.1f,}} (DQN / DQN recurrente / Transformer) → el Transformer tiene el {{ensambles/pes_ens/definicion/parametros/peso_normalizado_trf|pct0}} % del peso normalizado; A2C deshabilitado; recorta S a {{_/max_severidad}}.
- En los {{_/n_ens_optuna}} ensambles optimizados, el score de Optuna **{{_/ens_score_coincide}}** con la media del benchmark en la referencia; los {{_/n_ens_optuna}} optimizan también `w_a2c` (rango 0–3). {{_/ens_tau_optimizado}}

<!-- Justificación a posteriori de pes_ens (sec:res-posthoc, tab:sensitivity, tab:fixed-rule).
     Números literales tomados de h1/general/results/ensemble/weighted_ens_sensitivity.json,
     weighted_ens_oracle.json y fixed_rule.json (scripts homónimos de writings/auxiliar/scripts/).
     Revisarlos a mano si se vuelven a correr esos scripts. -->
- **Justificación a posteriori de `pes_ens`** (sec:res-posthoc): τ, w_π, σ y el peso del Transformer tienen su máximo en el valor fijo en la referencia y en las réplicas (máximos angostos: un paso de la grilla cuesta 0,002–0,008); η, el término 0,1 y la cota, en meseta. En las réplicas, sin prior (w_π = 0) supera al Transformer por 0,004; con prior y τ = 1, por 0,001; con ambos, por 0,010 (± 0,002 EE pareado). Supera al Transformer por más de 2 EE con τ 2–30, w_π 0–0,25, σ 3–5, peso TRF 2,5–10.
- **Óptimo** (DP de S_mejor con reconstrucción, 384 secuencias ref. + réplicas; 6 % de ciudades con empates, todas S ≤ 5): con presupuesto nunca asigna 0 y asigna S + 0,7 a S + 1,9; gaussiana centrada en S: σ = 2,7 (≈ 3); centro libre: S + 2,2, σ = 1,4. La cota cambia 2 de 1 726 decisiones; η, el 1,7 % (0,0004 por secuencia); el prior, el 37 % (+0,012 por secuencia).
- **Regla fija sin modelo** a = min(S + k, R) (tab:fixed-rule), ref. / gen. / réplicas: k = 0 0,782 / 0,805 / 0,787; k = 1 0,900 / 0,907 / 0,906; k = 2 0,940 / 0,937 / 0,943; k = 3 0,939 / 0,940 / 0,941 (Transformer 0,927 / 0,930 / 0,929; pes_ens 0,937 / 0,939 / 0,939). k = 2 supera al Transformer en 18/21 escenarios de generalización y 5/5 réplicas; pierde con secuencias cortas (len_all_short 0,845 vs 0,936). k = 2 se leyó del óptimo con información completa (también es el mejor k en la referencia). **Lectura obligatoria al redactar** (sec:disc-rule): no invalida H1/H2 (comparan modelos entre sí) ni implica que los agentes no aprendieran (sin el óptimo, k = 0 rinde 0,78; los agentes 0,85–0,93 sólo con la recompensa); sí implica que la política óptima de mPES es simple y que en este entorno la ventaja práctica de los modelos frente a una heurística calibrada es nula.

## [decisiones]
- unidad: % de decisiones con recursos disponibles (R > 0)

## [decisiones_otros_hechos]
<!-- Los números de los dos primeros puntos (0,19; 0,02-0,03; 31 %) NO se pueden derivar de ens_decisions.json
     (el replay no registra las probabilidades de los miembros ni la acción votada frente a la del Transformer);
     son los valores que cita la tesis (sec:res-freq). Revisarlos a mano si cambia el replay. -->
- Con tau = {{ensambles/pes_ens/definicion/parametros/tau|g}} las distribuciones de los tres miembros quedan casi planas: en la referencia la acción más probable de cada uno tiene en promedio probabilidad 0,19 y supera a la segunda por 0,02-0,03.
- El voto (antes del prior) ya difiere de la acción del Transformer en el 31 % de las decisiones de la referencia.
- En sev_extrapolate_high y joint_extrap_both {{_/ens_extrap_coincide}} de pes_ens coinciden con las de su miembro Transformer (que recibe la severidad recortada a {{_/max_severidad}}).

## [decisiones_heldout]
Réplicas fuera de muestra (`heldout_s1`..`s{{_/n_heldout}}`, no incluidas en "Generalización"; replay = benchmark con diferencia máxima {{_/dec_max_diff_heldout|e1}}): compuerta sigue al Transformer {{decisiones_ensambles/pes_ens_trf_guard/replicas_fuera_de_muestra/sigue_al_transformer|.1f}} % y difiere {{decisiones_ensambles/pes_ens_trf_guard/replicas_fuera_de_muestra/accion_final_distinta_del_transformer|.1f}} %; consenso con prior: prior {{decisiones_ensambles/pes_ens_consensus_prior/replicas_fuera_de_muestra/prior_cambia_accion_votada|.1f}} %, cota {{decisiones_ensambles/pes_ens_consensus_prior/replicas_fuera_de_muestra/cota_cambia_accion|.1f}} %, acción distinta del Transformer {{decisiones_ensambles/pes_ens_consensus_prior/replicas_fuera_de_muestra/accion_final_distinta_del_transformer|.1f}} %; ensamble ponderado: prior {{decisiones_ensambles/pes_ens/replicas_fuera_de_muestra/prior_cambia_accion_votada|.1f}} %, cota {{decisiones_ensambles/pes_ens/replicas_fuera_de_muestra/cota_cambia_accion|.1f}} %, distinta {{decisiones_ensambles/pes_ens/replicas_fuera_de_muestra/accion_final_distinta_del_transformer|.1f}} %. {{_/dec_heldout_en_rango}}

## [escenarios]
| Escenario | Nombre en el texto | Descripción | Fuera de rango |
|---|---|---|---|
| sev_base | referencia | Distribución empírica de entrenamiento: initial_severity.csv y sequence_lengths.csv sin perturbar (severidades 2..8, longitudes 3..10). | no |
| sev_uniform | — | Severidad U{0..9}; longitudes empíricas. | no |
| sev_gauss_low | — | Severidad N(2; 1,5) truncada a [0, 9] (brotes leves); longitudes empíricas. | no |
| sev_gauss_mid | — | Severidad N(4,5; 2,0) truncada a [0, 9] (brotes medios); longitudes empíricas. | no |
| sev_gauss_high | — | Severidad N(7; 1,5) truncada a [0, 9] (brotes intensos); longitudes empíricas. | no |
| sev_weibull | — | Severidad Weibull(k = 1,5), escala para media ~4,5, recortada a [0, 9] (cola superior pesada); longitudes empíricas. | no |
| sev_beta_lowskew | — | Severidad 9·Beta(2, 5) (sesgo bajo); longitudes empíricas. | no |
| sev_beta_highskew | — | Severidad 9·Beta(5, 2) (sesgo alto); longitudes empíricas. | no |
| sev_bimodal | — | Severidad 0,5·N(2, 1) + 0,5·N(7, 1), recortada a [0, 9]; longitudes empíricas. | no |
| sev_extrapolate_high | extrapolación de severidad | Severidad U{10, 11, 12}, por ENCIMA de la cota S_max = 9 (fuera de rango); longitudes empíricas. | sí |
| len_all_short | — | Todas las secuencias de longitud 3; severidades empíricas. | no |
| len_all_long | — | Todas las secuencias de longitud 10; severidades empíricas. | no |
| len_geometric | — | Longitud 2 + Geom(p = 0,2) recortada a [3, 10]; severidades empíricas. | no |
| len_poisson | — | Longitud Poisson(λ = 5) recortada a [3, 10]; severidades empíricas. | no |
| len_extrapolate_long | secuencias largas | Longitud U{11..20}, por ENCIMA de la cota T_max = 10 (fuera de rango); severidades empíricas. | sí |
| joint_high_long | — | Severidad N(7; 1,5) truncada × todas las secuencias de longitud 10. | no |
| joint_low_short | — | Severidad N(2; 1,5) truncada × todas las secuencias de longitud 3. | no |
| joint_uniform_geom | — | Severidad U{0..9} × longitudes geométricas (p = 0,2). | no |
| joint_extrap_both | extrapolación conjunta | Severidad U{10..12} × longitud U{11..20}: ambas fuera de rango. | sí |
| struct_few_long_blocks | — | 4 bloques × 16 secuencias = 64; misma distribución que la referencia (control). | no |
| struct_many_short_blocks | — | 16 bloques × 4 secuencias = 64; misma distribución que la referencia (control). | no |
| struct_more_total | — | 8 bloques × 8 secuencias = 64: las mismas secuencias que la referencia (control; originalmente 8 × 16 = 128, que sólo repetía las 64 secuencias y que los evaluadores de Optuna de los ensambles recortaban a 64). | no |

## [escenario_heldout]
Réplica {{k}} de la referencia: {{bloques}} × {{secuencias_bloque}} secuencias nuevas; longitud y severidad inicial sorteadas de forma independiente (i.i.d.) con las frecuencias empíricas de la referencia; semilla {{semilla}}. Excluida de la generalización.

## [catalogo_notas]
- Familias: {{_/familias_conteo}} (+ {{_/n_heldout}} réplicas fuera de muestra, fuera de todos los agregados; ver §10bis). Se descartaron escenarios de severidad constante (S_peor = S_mejor ⇒ métrica indefinida).
- Los {{_/n_sev_en_rango}} escenarios de severidad dentro de rango incluyen S = 0, 1 o 9, que están en el espacio de estados pero no aparecen en el entrenamiento ({{_/sev_ref_min}}..{{_/sev_ref_max}}).

## [matrices_nota]
_Réplicas fuera de muestra: los CSV `h1/general/results/{individual,ensemble}/matrices/*.csv` (y `summary.json`) tienen {{_/n_heldout}} columnas extra al final (`heldout_s1`..`heldout_s{{_/n_heldout}}`); se omiten en estas tablas y se resumen en §10bis. Los heatmaps por escenario (01, 02, 03, 04, 07) las muestran a la derecha de una línea vertical._

## [pares_citados_donde]
texto de `sec:res-individual` y `sec:res-ensembles`; mapas en `fig:pairwise-cohen` y `fig:ensemble-pairwise`

## [pares_citados]
- Transformer vs Q-Learning: pes_trf vs pes_ql
- Transformer vs DQN recurrente: pes_trf vs pes_rdqn
- Transformer vs DQN: pes_trf vs pes_dqn
- Ensamble ponderado vs Consenso: pes_ens vs pes_ens_consensus
- Ensamble ponderado vs Compuerta del Transformer: pes_ens vs pes_ens_trf_guard

## [confianza_trf]
- calculo: Q no factibles -> valor muy negativo; desplazamiento a valores no negativos; normalización; confianza = 1 - H_norm. NO comparable con la confianza de los ensambles (refleja sobre todo cuántas acciones quedan factibles).

## [heldout_intro]
_Fuente: `h1/general/results/heldout/heldout_gap.json` (generado con `writings/auxiliar/scripts/heldout_gap.py`) y `heldout_catalogue.json` (`python -m general.scripts.benchmark heldout`). Numeración 10bis para no alterar las referencias a §12/§13 de `instrucciones_sistema.md`._

## [heldout_puntos]
- **Propósito**: los modelos optimizados y los ensambles eligieron su configuración sobre las mismas {{escenarios/sev_base/n_secuencias}} secuencias de `sev_base`; las réplicas miden cuánto se sobreajustó cada configuración a esas secuencias (cuán optimista es la referencia). Ningún modelo se reentrenó ni se reoptimizó.
- **Procedimiento**: cada réplica sortea {{_/ref_bloques}} × {{_/ref_secuencias_bloque}} = {{escenarios/sev_base/n_secuencias}} secuencias nuevas: cada longitud y cada severidad inicial, de forma independiente (i.i.d.), con las frecuencias empíricas de `sequence_lengths.csv` e `initial_severity.csv` de la referencia, que son las mismas con las que se entrenan los modelos individuales. Semilla {{_/semilla_escenarios}} + k (k = 1..{{_/n_heldout}} → {{_/ho_semillas}}). En total {{replicas_fuera_de_muestra/total/n_secuencias}} secuencias.
- **Exclusión**: no intervienen en el entrenamiento, en la optimización ni en ningún agregado de generalización (medias de {{_/n_gen}} escenarios, peor escenario, degradación por familia, matrices de pares, conteos "N de {{_/n_ref_gen}}"/"N de {{_/n_gen}}").
- **Integridad**: `benchmark heldout` guarda en `results/heldout/heldout_sK/` los dos CSV y `sampling_distribution.json` (frecuencias de sorteo, semilla y frecuencias obtenidas) y comprueba con sha256 que los {{_/n_modelos}} paquetes recibieron las mismas secuencias: {{_/ho_copias}}.

## [heldout_json]
- fuente: h1/general/results/heldout/heldout_gap.json (writings/auxiliar/scripts/heldout_gap.py) y heldout_catalogue.json (python -m general.scripts.benchmark heldout). Tesis: Sección sec:res-heldout y Tabla tab:heldout (05Results); también 04Materials (párrafo y fila de tab:scenarios), 06Discussion (último párrafo), 07Conclusion (sec:limitations) y Apéndice ap:orchestrator.
- proposito: Las optimizaciones eligieron su configuración sobre las mismas {{escenarios/sev_base/n_secuencias}} secuencias de sev_base; las réplicas miden cuánto se sobreajustó cada configuración a esas secuencias (cuán optimista es la referencia). Ningún modelo se reentrenó ni se reoptimizó.
- procedimiento: Cada réplica sortea {{_/ref_bloques}} x {{_/ref_secuencias_bloque}} = {{escenarios/sev_base/n_secuencias}} secuencias nuevas: cada longitud y cada severidad inicial, de forma independiente (i.i.d.), con las frecuencias empíricas de los CSV de la referencia (las mismas con las que se entrenan los modelos individuales). Semilla {{_/semilla_escenarios}} + k (k = 1..{{_/n_heldout}} -> {{_/ho_semillas}}). Total {{replicas_fuera_de_muestra/total/n_secuencias}} secuencias.
- exclusion: No intervienen en el entrenamiento, la optimización ni en ningún agregado de generalización (medias de {{_/n_gen}} escenarios, peor escenario, degradación por familia, matrices de pares, conteos "N de {{_/n_ref_gen}}"/"N de {{_/n_gen}}"). Valores por celda en h1/general/results/<grupo>/cells/<pkg>__heldout_sK.json y en las {{_/n_heldout}} últimas columnas de matrices/*.csv y summary.json.
- verificacion_copias: benchmark heldout guarda los CSV y sampling_distribution.json (frecuencias de sorteo, semilla y frecuencias obtenidas) en results/heldout/heldout_sK/ y comprueba con sha256 que los {{_/n_modelos}} paquetes recibieron las mismas secuencias: {{_/ho_copias}}.
- afirmacion_tesis: En las {{_/n_heldout}} réplicas ({{replicas_fuera_de_muestra/total/n_secuencias}} secuencias nuevas) ningún modelo pierde desempeño de forma apreciable: caída de {{replicas_fuera_de_muestra/hechos_derivados/caida_min/valor|.3f,}} a {{replicas_fuera_de_muestra/hechos_derivados/caida_max/valor|+.3f,}}, |d| <= {{replicas_fuera_de_muestra/hechos_derivados/max_abs_d|.2f,}}, p >= {{replicas_fuera_de_muestra/hechos_derivados/min_welch_p|.2f,}}; en los {{_/ho_n_optuna}} optimizados con Optuna entre {{replicas_fuera_de_muestra/hechos_derivados/caida_rango_11_optuna/0|.3f,}} y {{replicas_fuera_de_muestra/hechos_derivados/caida_rango_11_optuna/1|+.3f,}} (la mayor, {{_/ho_caida_max_nombre}}{{_/ho_caida_max_nota_corta}}). Spearman rho = {{replicas_fuera_de_muestra/spearman_referencia_vs_fuera_de_muestra/rho|.2f,}}. El ensamble ponderado supera al Transformer por {{replicas_fuera_de_muestra/ensamble_ponderado_vs_transformer/diferencia_media|.3f,}} (t pareada, p < {{replicas_fuera_de_muestra/ensamble_ponderado_vs_transformer/t_pareada_p|sup}}) y en {{replicas_fuera_de_muestra/ensamble_ponderado_vs_transformer/replicas_ganadas}} de las {{replicas_fuera_de_muestra/ensamble_ponderado_vs_transformer/n_replicas}} réplicas. "Elegir la configuración sobre las {{escenarios/sev_base/n_secuencias}} secuencias de la referencia no infló el desempeño medido en esa condición."
- salvedad: Las réplicas se sortean con frecuencias estimadas a partir de las mismas {{escenarios/sev_base/n_secuencias}} secuencias, no con datos nuevos del experimento original: miden el sobreajuste a esas secuencias concretas, no a la distribución.
- figuras: Columnas heldout_s1..s{{_/n_heldout}} a la derecha de una línea vertical en los heatmaps por escenario 01/02/03/04/07 (tesis: fig:heatmap-global, fig:heatmap-ens, fig:heatmap-welch, fig:ensemble-statistical-heatmaps). 15_referencia_vs_heldout (individual y ensemble) se genera pero no está en la tesis.

## [heldout_convenciones]
- caida: referencia - media de las {{replicas_fuera_de_muestra/total/n_secuencias}} secuencias de las réplicas; positiva = pérdida (como deg).
- cohen_d: réplicas agrupadas (n = {{replicas_fuera_de_muestra/total/n_secuencias}}) frente a la referencia (n = {{escenarios/sev_base/n_secuencias}}) del mismo modelo; positivo = mejor en las réplicas.
- welch_p: p bilateral de Welch, réplicas agrupadas vs referencia.
- sd_entre_replicas: desviación estándar de las {{_/n_heldout}} medias de réplica (entre paréntesis en tab:heldout).
- rango: posición entre los {{_/n_modelos}} modelos (ambos grupos) por media; 1 = mayor.
- referencia_sd_ddof1: ddof = 1; las referencia_sd de modelos_individuales/ensambles usan ddof = 0.

## [heldout_reproduccion]
- python -m general.scripts.benchmark run --suite both (desde h1/; incluye las {{_/ho_n_celdas}} celdas heldout)
- python -m general.scripts.benchmark heldout (desde h1/)
- python writings/auxiliar/scripts/heldout_gap.py (desde la raíz)

## [heldout_notas]
- Caída = ref. − media de las {{replicas_fuera_de_muestra/total/n_secuencias}} secuencias de las réplicas (positiva = pérdida, como la degradación). $d$ y $p$: réplicas agrupadas (n = {{replicas_fuera_de_muestra/total/n_secuencias}}) frente a la referencia (n = {{escenarios/sev_base/n_secuencias}}) del mismo modelo; $d > 0$ = mejor en las réplicas; Welch bilateral. SD entre réplicas = desviación estándar de las {{_/n_heldout}} medias (entre paréntesis en `tab:heldout`). Error estándar de la caída entre {{_/ho_se_min|.3f}} y {{_/ho_se_max|.3f}}.
- Caída entre {{replicas_fuera_de_muestra/hechos_derivados/caida_min/valor|s3}} ({{_/ho_caida_min_nombre}}) y {{replicas_fuera_de_muestra/hechos_derivados/caida_max/valor|s3}} ({{_/ho_caida_max_nombre}}); |d| ≤ {{replicas_fuera_de_muestra/hechos_derivados/max_abs_d|.2f}} (efecto despreciable) y p ≥ {{replicas_fuera_de_muestra/hechos_derivados/min_welch_p|.2f}} en los {{_/n_modelos}}. En los {{_/ho_n_optuna}} optimizados con Optuna, entre {{replicas_fuera_de_muestra/hechos_derivados/caida_rango_11_optuna/0|s3}} y {{replicas_fuera_de_muestra/hechos_derivados/caida_rango_11_optuna/1|s3}} (máx.: {{_/ho_caida_max_optuna_nombre}}); la mayor caída es la de {{_/ho_caida_max_nombre}}{{_/ho_caida_max_nota}}: diferencias de ese tamaño son compatibles con la variación entre muestras.
- Ordenamiento: Spearman ρ = {{replicas_fuera_de_muestra/spearman_referencia_vs_fuera_de_muestra/rho|.3f}} (p = {{replicas_fuera_de_muestra/spearman_referencia_vs_fuera_de_muestra/p|e1}}, {{replicas_fuera_de_muestra/spearman_referencia_vs_fuera_de_muestra/n_modelos}} modelos; tesis: {{replicas_fuera_de_muestra/spearman_referencia_vs_fuera_de_muestra/rho|.2f,}}). Cambian de posición sólo modelos cuyas medias difieren en {{_/ho_swap_max_diff|.3f,}} o menos en ambas condiciones: {{_/ho_cambios_rango}}. {{_/ho_primero}}
- **Ensamble ponderado vs Transformer** en las mismas {{replicas_fuera_de_muestra/total/n_secuencias}} secuencias: diferencia media {{replicas_fuera_de_muestra/ensamble_ponderado_vs_transformer/diferencia_media|+.3f}} (en la referencia {{ganadores_por_escenario/sev_base/ensambles_menos_transformer/pes_ens|+.3f}}), t pareada p = {{replicas_fuera_de_muestra/ensamble_ponderado_vs_transformer/t_pareada_p|e1}} (tesis: p < {{replicas_fuera_de_muestra/ensamble_ponderado_vs_transformer/t_pareada_p|sup}}), Wilcoxon p = {{replicas_fuera_de_muestra/ensamble_ponderado_vs_transformer/wilcoxon_p|e1}}; mayor media en {{replicas_fuera_de_muestra/ensamble_ponderado_vs_transformer/replicas_ganadas}} de {{replicas_fuera_de_muestra/ensamble_ponderado_vs_transformer/n_replicas}} réplicas.
- **Afirmación de la tesis** (`sec:res-heldout`): "Elegir la configuración sobre las {{escenarios/sev_base/n_secuencias}} secuencias de la referencia no infló, por lo tanto, el desempeño medido en esa condición." `06Discussion` (último párrafo): la ventaja del ensamble ponderado tampoco se debe a que sus parámetros se eligieran sobre la referencia (se mantiene en {{replicas_fuera_de_muestra/ensamble_ponderado_vs_transformer/diferencia_media|.3f,}}). `07Conclusion` (limitaciones): la caída no superó {{replicas_fuera_de_muestra/hechos_derivados/caida_max/valor|.3f,}} ni fue significativa.
- **Salvedad**: las réplicas se sortean con frecuencias estimadas a partir de las mismas {{escenarios/sev_base/n_secuencias}} secuencias, no con datos nuevos del experimento original: miden el sobreajuste a esas secuencias concretas, no a la distribución.
- `reference_std` de `heldout_gap.json` usa ddof = 1 (p. ej. Transformer {{replicas_fuera_de_muestra/modelos/pes_trf/referencia_sd_ddof1|.4f}}), mientras que las σ de §4.2/§5.2 usan ddof = 0 ({{modelos_individuales/pes_trf/referencia_sd|.3f}}); las medias coinciden.
- Figuras: las columnas `heldout_s1`..`s{{_/n_heldout}}` aparecen a la derecha de una línea vertical en los heatmaps 01/02/03/04/07 (en la tesis: `fig:heatmap-global`, `fig:heatmap-ens`, `fig:heatmap-welch`, `fig:ensemble-statistical-heatmaps`); `15_referencia_vs_heldout` (individual y ensemble) se genera pero la tesis no la usa. Valores por celda: `h1/general/results/<grupo>/cells/<pkg>__heldout_sK.json` (no se copian a `celdas` del JSON).

## [advertencias]
- Una única semilla ({{_/semilla_escenarios}}) por escenario de generalización: no hay intervalos de confianza del ordenamiento; sólo la referencia cuenta con {{_/n_heldout}} réplicas (fuera de muestra{{md:, §10bis|json:; ver replicas_fuera_de_muestra}}). Los p de Welch no son independientes ni corregidos por comparaciones múltiples.
- Hiperparámetros de modelos y ensambles ajustados con las mismas {{escenarios/sev_base/n_secuencias}} secuencias de la referencia. Medido en las {{_/n_heldout}} réplicas fuera de muestra ({{md:§10bis|json:replicas_fuera_de_muestra}}): caída entre {{replicas_fuera_de_muestra/hechos_derivados/caida_min/valor|s3}} y {{replicas_fuera_de_muestra/hechos_derivados/caida_max/valor|s3}}, {{_/ho_significativa}} (|d| ≤ {{replicas_fuera_de_muestra/hechos_derivados/max_abs_d|.2f}}, p ≥ {{replicas_fuera_de_muestra/hechos_derivados/min_welch_p|.2f}}), ρ de Spearman {{replicas_fuera_de_muestra/spearman_referencia_vs_fuera_de_muestra/rho|.2f}}: la referencia no resultó optimista. Salvedad: las réplicas se sortean con frecuencias estimadas a partir de esas mismas {{escenarios/sev_base/n_secuencias}} secuencias, no con datos nuevos del experimento original.
- La media de generalización incluye los {{_/n_estructurales}} escenarios estructurales (idénticos a la referencia) y escenarios más fáciles (p. ej. len_all_short, joint_low_short); leerla junto con el peor escenario.
- Entrenamiento sólo con severidades {{_/sev_ref_min}}..{{_/sev_ref_max}}: filas S = 0, 1, 9 de las tablas Q quedan con valores iniciales aleatorios; parte de la caída tabular refleja falta de cobertura, no sólo capacidad de generalizar.
- Con severidades 10..12 una política que asigna mucho puede acercarse al óptimo: un rbar alto en sev_extrapolate_high no prueba por sí solo mejor generalización (A2C también supera allí su referencia).
- Arquitecturas de DQN recurrente y Transformer elegidas ad hoc (no optimizadas): las conclusiones se refieren a los modelos finales evaluados; los resultados no identifican la causa de la ventaja del Transformer.
- La confianza 1 - H_norm se calcula sobre softmax de valores Q (no probabilidades aprendidas, salvo A2C): es heurística, no calibrada; no se comprobó que sea mayor en decisiones acertadas.
- El registro de confianza del Transformer (media {{confianza_transformer_registro/confianza_media|.3f,}}) usa otro cálculo que los ensambles: no comparar con tau_g ni con otros umbrales.
- La ventaja del ensamble ponderado proviene del prior de severidad y de tau = {{ensambles/pes_ens/definicion/parametros/tau|g}}, no de la ponderación por confianza, y requiere ambos (sensibilidad en sec:res-posthoc). Sus parámetros se fijaron a mano sobre la referencia y sólo se justifican a posteriori.
- Una regla fija sin modelo, a = min(S + 2, R), calibrada con el óptimo, iguala al ensamble ponderado y supera al Transformer (sec:res-posthoc, sec:disc-rule): no presentar a los modelos como superiores a una heurística bien calibrada en este entorno.
- "Distancia de Lieber" en documentos previos = divergencia de Kullback-Leibler.
- h2/ es una línea suspendida: no citarla como trabajo realizado. La tesis no usa datos humanos (ds004477 sólo como contexto del entorno PES).

## [puntos_de_atencion_intro]
Contrastados contra los `.tex` actuales el 2026-09-28 (los cinco puntos sobre 04Materials, 05Results, 06Discussion y el resumen de la versión del 2026-09-24 ya están corregidos; el de los números de ensayos de Optuna se resolvió porque ahora se leen de `inputs/*_BAYESIAN_OPT/optimization_results_*.txt`). Corregirlos cuando se edite el archivo afectado.

## [puntos_de_atencion]
| Severidad | Archivo | Lugar | Hallazgo | Corrección sugerida |
|---|---|---|---|---|
| observación | audit.py | criterio 6 (cobertura de paquetes) | Pasa sólo gracias a los nombres de archivo PES_<PKG>_results.png (\texttt{pes\_base} no contiene "pes_base"). Si se eliminan las figuras por modelo, el criterio falla. | Mantener esas figuras o mencionar los paquetes con \verb\|pes_base\| (el criterio de idioma ignora \verb). |

## [mapa]
- excluidos_a_proposito: Acknowledgement.tex
- nota_excluidos: excluido de Main.tex a propósito
- como_incluirlas: Agregar el stem a SUITE_FIGURES en writings/auxiliar/scripts/sync_figures.py y ejecutarlo (sync_figures borra de 02_Images todo lo que no esté en su lista); nombre destino ind_<stem>.png o ens_<stem>.png.

## [mapa_figuras_notas]
- fig:heatmap-global: {{_/n_heldout}} columnas heldout_s1..s{{_/n_heldout}} a la derecha de una línea vertical
- fig:heatmap-ens: {{_/n_heldout}} columnas fuera de muestra tras una línea vertical
- fig:heatmap-welch: {{_/n_heldout}} columnas fuera de muestra tras una línea vertical
- fig:ensemble-statistical-heatmaps: Welch por escenario (ensambles, página apaisada); {{_/n_heldout}} columnas fuera de muestra tras una línea vertical
- fig:ensemble-cohen-scenario: d de Cohen por escenario (ensambles, página apaisada); {{_/n_heldout}} columnas fuera de muestra tras una línea vertical
- fig:c-base … fig:c-trf: figuras por modelo en el Apéndice ap:per-model (redibujadas a tamaño de impresión por general.scripts.figures --only models)
- fig:extra-sev-skew: curvas ordenadas en sev_bimodal, sev_gauss_high, sev_beta_highskew, len_poisson, len_extrapolate_long, joint_high_long; trazo grueso = Transformer
- fig:ensemble-extrapolation: {{_/n_fuera_de_rango}} escenarios fuera de rango; trazo grueso = ensamble ponderado

## [reproduccion]
- entorno: win_mpes_env\Scripts\Activate.ps1 desde la raíz; comandos desde h1/; VIRTUAL_ENV, PYTHONIOENCODING=utf-8, TF_ENABLE_ONEDNN_OPTS=0.
- replicas_fuera_de_muestra: run --suite both evalúa también heldout_s1..s{{_/n_heldout}} ({{_/ho_n_celdas}} celdas: {{_/n_modelos}} modelos × {{_/n_heldout}}); benchmark heldout escribe general/results/heldout/ (CSV + sampling_distribution.json por réplica, heldout_catalogue.json con sha256 y verificación de copias en los {{_/n_modelos}} paquetes); python writings/auxiliar/scripts/heldout_gap.py (desde la raíz, como en el Apéndice ap:orchestrator; independiente del directorio) escribe h1/general/results/heldout/heldout_gap.json (Tabla tab:heldout).
- contexto: python writings/auxiliar/scripts/ensemble_decisions.py --output h1/general/results/ensemble/ens_decisions.json (desde la raíz; replay con TensorFlow) y luego python writings/auxiliar/scripts/build_results_context.py (regenera mpes_resultados.md y .json; --check compara con los archivos en disco).
- justificacion_pes_ens: desde la raíz, python writings/auxiliar/scripts/weighted_ens_sensitivity.py (≈ 16 min, reanudable), weighted_ens_oracle.py (≈ 2 min; --no-audit ≈ 1 min) y fixed_rule.py (segundos); escriben weighted_ens_sensitivity.json, weighted_ens_oracle.json y fixed_rule.json en h1/general/results/ensemble/ (sec:res-posthoc).

## [reproduccion_benchmark]
- python -m general.scripts.benchmark run --suite both
- python -m general.scripts.benchmark heldout
- python -m general.scripts.analysis
- python -m general.scripts.figures
- python -m general.scripts.random_baseline
- python -m general.scripts.agent_internals

## [reproduccion_tesis]
- cd writings
- python audit\audit.py
- python audit\audit.py --no-tex
- python auxiliar\scripts\sync_figures.py
- python auxiliar\scripts\ensemble_decisions.py --output ..\h1\general\results\ensemble\ens_decisions.json
- python auxiliar\scripts\heldout_gap.py
- python auxiliar\scripts\trf_vs_ens.py
- python auxiliar\scripts\build_results_context.py
