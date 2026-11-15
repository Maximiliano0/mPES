# Correcciones del jurado — seguimiento

Pre-dictámenes recibidos (ambos: «cumple con los objetivos y está en condiciones de ser defendido»; todos los
criterios en «Total»):

- **B** — Dr. Ing. Leopoldo De Bernardez: [Pre-DictamenTesis_MCD_Bernardez.pdf](Pre-DictamenTesis_MCD_Bernardez.pdf)
- **R** — Jurado sin nombre en el formulario: [Pre-DictamenTesis_MCD_Robino.pdf](Pre-DictamenTesis_MCD_Robino.pdf)

Las referencias a números de sección, tabla, figura y cita corresponden al PDF que leyó el jurado, que coincide con
`master` @ `9a1cdc4`: Tabla 4.1 = `tab:packages`, Tabla 5.7 = `tab:sensitivity`, Fig. 5.1 = `fig:heatmap-global`,
Fig. 5.3 = `fig:heatmap-ens`, Ec. 2.4 = `eq:attention`, Sec. 7.3 = Trabajo futuro, Sec. 7.4 = Conclusión general,
[32] = `Nijjar2025`, [48] = `Henderson2018`, [51] = `mPES2026`.

**Numeración actual de las citas (cambia con cada cita nueva):** después de B1, [32] → [38] `Nijjar2025`,
[48] → [53] `Henderson2018`, [51] → [56] `mPES2026`; `Bednarski2021` pasó a ser [5]. La Introducción también se renumeró: «Preguntas e hipótesis»
pasó a §1.4 y «Contribuciones» a §1.5.

**Presupuesto de páginas**: el cuerpo (Introducción → Conclusiones) ocupaba las págs. 1–37 al recibir los dictámenes;
después de B1 ocupaba las págs. 1–38 y después de B2 ocupaba las págs. 1–39 después de B3 ocupaba las págs. 1–40 y después de B4/B5 ocupaba las págs. 1–41. **Después de B8 ocupa las págs. 1–42. El 2026-11-13 el autor fijó el tope en 42 págs. (antes 40): el cuerpo queda así, sin recortes, y no puede crecer más.** R2 (2026-11-12) no recuperó páginas: los finales de capítulo cortos (Cap. 3 pág. 12, Cap. 6 pág. 39, Cap. 7 pág. 42) son los puntos donde recortar ~10–15 renglones libera una página entera. El autor lo aceptó de forma transitoria; hay que compensarlo con R2 (recuadros → párrafos) y/o R9 antes de agregar más texto. Ojo: cada capítulo empieza en página nueva, así que un desborde chico cuesta una página entera; la Introducción ocupa ahora media pág. 4. Las estimaciones de abajo suman unas +2,5 págs. sin compensar. Las compensaciones posibles son quitar los recuadros (R2) y
llevar los mapas completos al apéndice (R9). Medir después de cada bloque con `/thesis-audit`.

Estados: ⬜ pendiente · 🟨 en curso · ✅ hecho · ❓ requiere decisión del autor · 🚫 descartada por el autor

## Tabla resumen

| ID | Jurado | Tema | Dónde | Tipo | Págs. | Estado |
|----|--------|------|-------|------|-------|--------|
| R5 | R | «Estado de la Cuestión» → «Estado del Arte» | `Main.tex:198` | trivial | 0 (medido) | ✅ |
| B4 | B | Desplegar el acrónimo mPES | `01Introduction.tex:9` | trivial | 0 | ✅ |
| R4 | R | Ec. 2.4 es autoatención enmascarada, no atención genérica | `02Background.tex` §2.3 | menor | 0 (medido) | ✅ |
| R10 | R | Llevar al texto el criterio de «dos errores estándar pareados» de la Tabla 5.7 | `05Results.tex` §5.5 | menor | 0 (medido) | ✅ |
| R8 | R | Por qué la LSTM no conserva el estado oculto | `04Materials.tex` §4.4.2 | menor | 0 (medido) | ✅ |
| R7 | R | Qué es «Q-Learning tabular optimizado» | `04Materials.tex` §4.4, §4.4.1 | menor | 0 (medido) | ✅ |
| R6 | R | Cuántas ciudades están activas y cómo se elige cada una por *trial* | `04Materials.tex` §4.2 | mayor | 0 (medido; pág. 23 llena) | ✅ |
| R3 | R | El Cap. 2 remite al Cap. 4 antes de presentar el entorno | `02Background.tex` §2.1, §2.2 | mayor | 0 (medido; pág. 9 llena) | ✅ |
| B2 | B | Objetivo general y objetivos específicos en la Sec. 1.3 (hoy §1.4) | `01Introduction.tex` §1.4, §1.5; `07Conclusion.tex` §7.1 | mayor | +1,0 (medido; ~0,6 real) | ✅ |
| B1 | B | Decisiones reales que podrían usar el marco y bajo qué condiciones | `01Introduction.tex` §1.3 (nueva) | mayor | +1,0 (medido) | ✅ |
| B5 | B | Tabla análoga a la 4.1 para los ensambles | `04Materials.tex` §4.5 | menor | +1,0 (medido: cuerpo 1–41) | ✅ |
| B6 | B | Criterio de elección o descarte de los modelos | `04Materials.tex` §4.4 y §4.5 | mayor | 0 (medido; cabía en el Cap. 4) | ✅ |
| B3 | B | Justificar las semillas independientes en la Sec. 7.3 con [48] o Wang (2020) | `07Conclusion.tex` §7.3 | menor | +0,15 (medido: empujó el cuerpo a 1–40) | ✅ |
| B8 | B | H2 depende de la composición del ensamble | `06Discussion.tex` §6.2 | mayor | +1,0 (medido: cuerpo 1–42) | ✅ |
| B7 | B | Complejidad de variantes cuya política óptima no es simple | `06Discussion.tex` §6.3 | mayor | 0 (medido, después de recortar) | ✅ |
| B9 | B | Conexión con PES y las decisiones humano-agente de [32] | `07Conclusion.tex` §7.3 | menor | 0 (medido) | ✅ |
| R9 | R | Figs. 5.1 y 5.3: demasiados valores (7 × 27 = 189 celdas cada una) | `05Results.tex:54-63, 218-225` | mayor | 0 | 🚫 |
| R2 | R | Recuadros `destacado`: criterio poco claro e inusual en el ITBA | 12 recuadros en Caps. 3–7 | mayor | 0 (medido: cuerpo sigue en 1–42) | ✅ |
| R1 | R | Tono escueto y algo artificial: suavizar y explicar más | toda la tesis | mayor | 0 | 🚫 |

## Detalle por observación

### Bernardez

**B1 — Aplicaciones reales (Relevancia del tema).** Agregar en §1.1 qué decisiones reales comparten la estructura de
mPES: llegada secuencial de casos, presupuesto fijo, asignación irrevocable y horizonte desconocido. Ejemplos ya
citados en el Cap. 3: pruebas en fronteras (`Bastani2021`), equipamiento médico entre estados (`Bednarski2021`),
inventarios (`Gijsbrechts2022`) y mochila en línea (`Kong2019`). Indicar las condiciones de aplicación y remitir a la
§6.3 por las simplificaciones: unidades independientes, un solo recurso y dinámica determinista.

*Hecho (2026-11-11):* nueva §1.3 «Decisiones reales y condiciones de aplicación» en `01Introduction.tex`,
págs. 1–3 del PDF. Usa cinco referencias nuevas verificadas (`Mehrotra2020`, `Emanuel2020`, `HolguinVeras2013`,
`YuLina2021`, `Sinclair2023`) y `Bednarski2021`. Argumento: los casos reales comparten el dilema de mPES pero no sus
simplificaciones; por eso el aporte es un banco de pruebas y no una política trasladable, coherente con §6.3 y §7.4.
Se revisó con `thesis-judge` y se incorporaron sus hallazgos. Costo: +1,0 pág.
*Segunda revisión del juez (incorporada):* la ventaja se atribuye al Transformer y su atención, no a la ventana
(coherente con §6.1); la regla fija se acota a mPES (§6.3 no la generaliza); el contraste con Yu et al. es
«análogo»; se explicita que Yu et al. buscan equidad en varios períodos; el párrafo de condiciones se partió y el
cierre ya no parafrasea §7.4. Cuerpo: págs. 1–38.

**B2 — Objetivos.** Agregar en §1.3 (por ejemplo «Objetivos, preguntas e hipótesis») un objetivo general (comparar
arquitecturas de RL y reglas de ensamble en asignación secuencial bajo incertidumbre, con un óptimo exacto como
referencia) y objetivos específicos que se correspondan con los capítulos: construir el entorno y la métrica
normalizada, entrenar y optimizar 7 modelos, diseñar 6 reglas de ensamble, evaluar en 21 escenarios y 5 réplicas, y
analizar *post hoc* el mejor ensamble. Cuidar que no dupliquen las Contribuciones (§1.4).

*Hecho (2026-11-11):* §1.4 pasó a llamarse «Objetivos, preguntas e hipótesis» y suma un objetivo general (qué esquema de decisión de RL, individual o ensamble, rinde y generaliza mejor en un entorno simulado) y cuatro objetivos específicos. El autor excluyó a propósito el objetivo de formalizar el MDP y la métrica. Una oración vincula los objetivos 1–2 con P1 (y como insumo de P2) y los 3–4 con P2. Por los hallazgos de `thesis-judge`, las Contribuciones (§1.5) se reescribieron como productos y hallazgos (corrige «seis» → siete modelos) y §7.1 se renombró «Objetivos y preguntas de investigación». Cuerpo: págs. 1–39.
*Lección de páginas:* un título de §7.x que ocupa dos renglones empuja las Conclusiones a una página más (pasó con «Cumplimiento de los objetivos y respuesta a las preguntas»). La pág. 39 está llena.
*Verificación final del juez:* aprobada. C3 se reformuló para no contradecir §7.1 (el diseño no aísla la ponderación por confianza) y los rótulos de §7.1 indican qué objetivos cierra cada párrafo (1–2, 3–4, 3).

**B3 — Semillas independientes.** En el primer punto de §7.3, justificar con `Henderson2018` (ya citado en la §3.4)
que la variabilidad entre semillas de entrenamiento es una fuente de incertidumbre propia del RL profundo. Opcional:
Wang, H.-N. *et al.* (2020), *Deep reinforcement learning: a survey*, FITEE 21(12). Verificar el DOI antes de agregarlo
a `References.bib`.

*Hecho (2026-11-11):* se reescribió el primer punto de §7.3: hay que repetir el *entrenamiento* (no la evaluación, que es determinista); la dispersión informada proviene de las secuencias y no de la semilla; se citan `Henderson2018` (diferencias significativas sólo por la semilla) y `Agarwal2021`, nueva, [60] (intervalos de confianza con pocas ejecuciones). Se descartó Wang et al. (2020), porque no pudo verificarse que trate la variabilidad entre semillas.

**B4 — Acrónimo mPES.** Desplegarlo en la primera mención: *Multiple Pandemic Experiment Suite* (`README.md`,
`mPES2026`). ⚠️ Inconsistencia detectada al revisar: la Introducción y `BCINE2022` expanden PES como *Pandemic
Experiment Scenario*, pero `ds004477` lo titula *Pandemic Emergency Scenario*. Confirmar cuál es el nombre oficial y
unificarlo.

**B5 — Tabla de ensambles.** Agregar en §4.5 una tabla con las mismas columnas que la 4.1 (paquete, nombre en el
texto, descripción) y una columna de miembros: ponderado = DQN + DQN recurrente + Transformer; los otros cinco, además
el actor de A2C. Puede reemplazar la frase introductoria de la lista de reglas.

*Hecho B4 (2026-11-11):* `01Introduction.tex:9` dice ahora «mPES (*Multiple Pandemic Experiment Suite*)». El autor confirmó que el nombre oficial es *Pandemic Experiment Scenario*: el texto ya es correcto. `ds004477` conserva «Pandemic Emergency Scenario» porque es el título publicado del dataset.
*Hecho B5 (2026-11-11):* nueva Tabla `tab:ens-packages` al inicio de §4.5 (paquete, nombre en el texto, regla de combinación, uso de A2C), referida en la primera oración. Las tablas siguientes del Cap. 4 se renumeraron.

**B6 — Criterio de elección de modelos.** ❓ Lo tiene que redactar el autor, porque se trata de decisiones de diseño.
Preguntas a responder en §4.4/§4.5:

- ¿Por qué estas siete familias? (progresión tabular → profunda → con memoria → atención; A2C como representante de
  gradiente de política).
- ¿Por qué los ensambles excluyen a los tabulares? Hipótesis a confirmar: sus entradas se recortan a las cotas de la
  tabla y no producen una distribución comparable.
- ¿Por qué el ensamble ponderado excluye a A2C? `h1/ens/pes_ens/config/CONFIG.py:135` dice «A2C actor is currently
  very weak; disable by default», pero hoy A2C obtiene $0{,}887$ en la referencia, igual que Q-Learning. El comentario
  parece de una versión anterior de A2C y no sirve como justificación sin verificarlo. La Tabla `tab:ens-packages` (B5) muestra la exclusión («A2C: no»), así que el jurado puede preguntarlo.

También es uno de los tres temas que el jurado B pide preparar para la defensa.

*Hecho B6 (2026-11-11), con las respuestas del autor:* (1) DQN fue el punto de partida (sugerido por el director; en el texto figura como criterio, sin nombrarlo) y los demás modelos surgieron de la bibliografía; en §4.4 se explica qué capacidad agrega cada uno y que ningún modelo se excluyó después de evaluarlo (verificado en git). (2) Los tabulares quedan fuera de los ensambles por tener la menor capacidad de generalización. (3) A2C queda fuera del ponderado porque reducía su desempeño durante el ajuste ad hoc; como respaldo, Optuna le asignó el menor peso base en 4 de 5 ensambles. Sirve también para preparar la defensa. Pendiente de código (no de la tesis): el comentario «very weak» de `h1/ens/pes_ens/config/CONFIG.py:135` debería decir que su inclusión reducía el desempeño (agente `python-h1`).
*Ajustes por el juez (incorporados):* «una capacidad que otro del conjunto no tiene», en lugar de una cadena lineal; Double Q-Learning queda asociado sólo a la reducción de la sobreestimación (el PBRS lo comparten otros modelos); el argumento de los tabulares ya no usa el recorte de S (A2C también recorta); «ninguno de los siete modelos se excluyó; sólo se descartaron configuraciones».

**B7 — Variantes con política óptima no simple.** Ampliar la §6.3, que ya enumera las simplificaciones, con sus
consecuencias metodológicas: sin óptimo calculable, $\bar r$ necesita otra cota (la mejor política conocida o una
relajación); una regla fija ya no sirve de referencia calibrada; la memoria y la atención deberían pesar más, y el
*prior* de severidad pierde sustento. Plantearlo como expectativa, no como resultado.
*Arrastre de B1:* alinear las listas de simplificaciones, que hoy no coinciden. §1.3 tiene cinco (independencia, un
recurso, dinámica determinista, asignación fija y objetivo sin equidad); §6.3 tiene las tres primeras; §7.3 nombra
cuatro, y §7.2 ninguna. La equidad (`Emanuel2020`, `Sinclair2023`) sólo aparece en §1.3: conviene agregarla a §7.2 o
a §6.3.

*Hecho B7 (2026-11-11):* §6.3 suma dos párrafos. El primero explica por qué la política óptima es simple (descomposición por ciudad → mochila; equilibrio a = S) y qué rompe cada una de las cinco simplificaciones (contagio → estado exponencial; varios recursos → mochila multidimensional; dinámica estocástica o desconocida → óptimo sólo a posteriori; reparto en varios pasos; equidad → no hay un único óptimo). El segundo explica el efecto, como expectativa: $ar r$ sin cota, sin regla calibrada, menor ventaja del ponderado, más valor de los estados anteriores, tabulares impracticables y H2 más exigente. Arrastre de B1 cerrado: §1.3 remite a §6.3 y §7.3 agrega la equidad.
*Ajustes por el juez (incorporados):* el óptimo de mPES ya es a posteriori (con azar, $1-ar r$ deja de medir sólo el error del agente); α y β variables entre secuencias (los agentes ya los desconocen); con equidad, el óptimo depende de la ponderación (Sinclair2023); con varios recursos o reparto en varios pasos sigue siendo calculable pero sin regla simple; se reconoce §6.1 sobre la ventana; se separan los supuestos sobre la observación; la confianza ya no se presenta como el «principal mecanismo»; se corrigió la doble «y» de §7.3. Hubo que recortar dos veces para que la Discusión no pasara a la pág. 39. Cuerpo: 1–41.

**B8 — H2 y composición del ensamble.** Agregar en §6.2 que el rechazo de H2 depende de los miembros y de sus pesos.
Evidencia disponible: en el voto suave y el voto por acción los pesos favorecen al DQN recurrente, y en la
extrapolación de severidad su media queda por debajo de tres de sus cuatro miembros (§5.2). Con miembros más
parejos o sin uno débil podría haber ensambles que verifiquen H2. Matizar también el último punto de §7.2 o agregar
uno.

*Hecho B8 (2026-11-11):* en §6.2, el último recuadro («La ponderación por confianza, por sí sola, no basta…») se reemplazó por un párrafo. Dice que el resultado depende de la composición y los pesos: el voto suave y el voto por acción le dan más peso al DQN recurrente, el miembro más débil en la extrapolación de severidad, y ahí quedan por debajo de 3 de 4 miembros. Con el Transformer dominante y sin A2C, el voto con τ=15 supera al Transformer en 0,4% en held-out aun sin prior. Con otra composición (varias semillas del Transformer, Lakshminarayanan2017) podría verificarse la H2; no se respalda para lo evaluado, pero no se descarta en general. **Para R2: quedan 12 recuadros.**
*Ajustes por el juez (incorporados):* «promedio» en lugar de «voto»; «lo que indica que…» en lugar de una afirmación causal; la H2 se verificaría «siempre que una variante sin ponderación permita atribuir la ventaja a la confianza»; el resumen y el abstract ahora dicen «no bastó, en los ensambles evaluados» (siguen en una página cada uno).
*Hecho B9 (2026-11-11):* se amplió el segundo punto de §7.3 (calibración): usar los agentes de mPES como compañeros en los experimentos humano–agente de PES (`Nijjar2025`). Lo atribuido a Nijjar se limita a su resumen oficial (QUB Pure): los equipos superan a los individuos y confiar más en una IA «humanizada» no mejoró necesariamente el desempeño. El PDF completo no pudo descargarse (repositorio de Essex con ECONNRESET); si el autor lo consigue, conviene afinar la frase.
*Ajustes por el juez (incorporados):* «grupos» (como dice el resumen) en lugar de «equipos»; se quitó «del equipo»; los agentes de mPES quedan «adaptados al protocolo de PES»; \emph{humanizado} en lugar de comillas. **Bernardez: 9/9 cerradas.**

**B9 — Conexión con PES humano-agente.** ❓ Hace falta el texto de `Nijjar2025` (no está en
`writings/auxiliar/biblio/`) para describir con fidelidad qué encontraron sobre el desempeño de los agentes. Sin
leerlo, sólo cabe una frase genérica. Ubicación sugerida: trabajo futuro (§7.3), usar los agentes de mPES como
compañeros en el protocolo humano-agente de PES.

**Para la defensa (no se cambia el texto):**

- *Variabilidad entre semillas*: no hay historiales de Optuna de los modelos individuales (sólo
  `pes_ens_consensus_prior` guarda su `.db`). Una estimación parcial exige reentrenar, por ejemplo Transformer y DQN
  con 3–5 semillas y la mejor configuración, y evaluar en la referencia y las réplicas. Es tarea del agente
  `python-h1` y requiere confirmar el tiempo de cómputo.
- *Criterio de elección de los modelos*: ver B6.
- *Política aprendida frente a la regla $S + k$*: ya existen asignaciones medias por severidad (Transformer
  $S + 2{,}4$, ensamble $S + 1{,}6$, óptimo $S + 1{,}6$) en `weighted_ens_oracle.py`. Se puede preparar una lámina
  con la curva asignación–severidad de cada política.

### Robino

**R1 — Tono.** Pasada capítulo por capítulo al final, cuando el contenido ya esté estable: reemplazar enumeraciones
telegráficas por párrafos que expliquen el porqué y conectar cada sección con la siguiente. Hay que vigilar el
presupuesto de páginas. El agente `thesis-judge` sirve para la verificación.

*Descartada (2026-11-12):* decisión del autor; la considera irrelevante (el jurado calificó el criterio «Redacción»
como «Total»). No se hace la pasada de tono. Queda sin atender la repetición de las cifras $0{,}4\%$/$1{,}1\%$ en
§6.2 anotada en R2.

**R2 — Recuadros.** ❓ Opciones: (a) quitarlos todos y convertirlos en párrafos, con o sin una frase de cierre en
negrita (recomendado: es lo habitual en las tesis del ITBA, el jurado lo señala y libera espacio); (b) conservarlos y
explicar el criterio en la Introducción. Hay 13: Cap. 3 (1), Cap. 4 (2), Cap. 5 (6), Cap. 6 (3) y Cap. 7 (1). El
entorno `destacado` del preámbulo puede quedar definido aunque no se use.

*Hecho R2 (2026-11-12):* decisión del autor, opción (a) sin negrita. Se quitaron los 12 entornos `destacado` que
quedaban (Cap. 3: 1; Cap. 4: 2; Cap. 5: 6; Cap. 6: 2; Cap. 7: 1); el texto queda como párrafo normal, sin cambiar
ninguna palabra. En §5.1 y §5.5 se agregó una línea en blanco antes de la tabla siguiente. El preámbulo de `Main.tex`
no se tocó: el entorno queda definido y sin uso. Se actualizó `mpes_resultados_notas.md` («párrafo de síntesis»).
Auditoría en verde, 65 págs. **El cuerpo sigue en 1–42**: los capítulos se acortaron, pero ninguno liberó su última
página. Páginas finales cortas, candidatas para recuperar una página entera recortando ~10–15 renglones: Cap. 3
(pág. 12, ~10 renglones), Cap. 6 (pág. 39, ~12) y Cap. 7 (pág. 42, ~14). Para R1: sin el recuadro se nota la
repetición de las cifras $0{,}4\%$/$1{,}1\%$ entre el párrafo de síntesis de §6.2 y los dos párrafos siguientes.
*Ajustes por el juez (incorporados):* en §5.3, «muestra su desempeño» → «muestra el desempeño por secuencia del
ensamble ponderado» (el «su» quedaba ambiguo con el Transformer); en §3, «mPES comparte…» y «Se diferencia en que…»
se unieron en un solo párrafo. Auditoría en verde; cuerpo 1–42 (la pág. 12 del Cap. 3 baja a ~9 renglones).

**R3 — Referencias del Cap. 2 al Cap. 4.** Las del Cap. 2 son `sec:metric` (l. 49), `sec:env-dyn` (l. 93),
`eq:shannon-norm` (l. 160), `sec:ens-methods` (l. 163, 185), `eq:ens-softmax` (l. 170) y `sec:res-heldout` (l. 204);
las del Cap. 3, `sec:env-dyn` (l. 77) y `sec:scenario-catalogue` (l. 105). Propuesta: ampliar §1.2 con una
descripción mínima del entorno (una ciudad por paso, presupuesto, severidad que crece, longitud desconocida,
óptimo con información completa) y que el Cap. 2 remita a §1.2 o defina el concepto sin remitir hacia adelante.
Conservar sólo las remisiones que digan «se usa en…».

*Hecho R3 (2026-11-12):* directiva del autor: dar en el Cap. 2 un resumen del entorno sin desarrollarlo. Párrafo
nuevo al final de §2.1 («En mPES, que la Sección 4.2 formaliza…», pág. 6): ciudad por paso con severidad inicial $S$,
asignación de 0 a 10 de un presupuesto de 30, estado $(R,t,S)$, severidad ×1,4 por paso y −0,4 por unidad,
recompensa negativa, $L$ de 3 a 10 fuera del estado → parcialmente observable, óptimo por programación dinámica y
normalización 0–1. Se quitaron las remisiones al entorno de §2.1 (métrica, l. 49) y §2.2 (PBRS, l. 93); la frase de
§2.1 sobre resolver Bellman por programación dinámica pasó al párrafo nuevo (evita repetirla y compensa el espacio:
la primera versión desbordaba 2 renglones). Se conservan las remisiones de ensambles y evaluación (§4.5, §5.4).
Auditoría en verde, 65 págs.; cuerpo 1–42 (pág. 9 llena: no queda margen en el Cap. 2).
*Ajuste por el juez (incorporado):* el cierre decía que la Ec. de Bellman «se resuelve exactamente» y que la métrica
usa «ese óptimo», pero la Ec. 2.3 usa el retorno descontado y el óptimo de la métrica minimiza la suma de severidades
finales (Ec. 4.6). Ahora dice «la asignación que minimiza la suma de las severidades finales se calcula exactamente
por programación dinámica». La confusión venía de la frase original de §2.1. Cuerpo 1–42.

**R4 — Ec. 2.4.** Presentarla como «capa de autoatención con máscara causal». Opcional: dar antes la forma genérica
$\mathrm{Atención}(Q,K,V)=\mathrm{softmax}(QK^\top/\sqrt{d_h})V$ y luego la autoatención, con $Q = XW_q$ y lo demás
análogo, más la máscara $M$.

*Hecho R4 (2026-11-12):* en `02Background.tex` §2.3 (pág. 8) el Transformer «reemplaza la recurrencia por
autoatención de producto escalar»; «Una cabeza de autoatención con máscara causal pondera las posiciones según su
similitud»; el párrafo «Donde…» dice que $W_q$, $W_k$, $W_v$ «proyectan la misma $X$ en consultas, claves y valores»,
y «La autoatención de varias cabezas…». La ecuación no cambió. Se compensó el espacio acortando dos frases (la pág. 9
está llena). Auditoría en verde, 65 págs.; cuerpo 1–42.

**R5 — Título del Cap. 3.** Cambiar `\section{Estado de la Cuestión}` por `\section{Estado del Arte}` en `Main.tex:198`.

*Hecho R5 (2026-11-12):* `Main.tex:198` → `\section{Estado del Arte}` (mayúsculas como los demás capítulos; la
etiqueta `sec:soa` no cambia). Se regeneró el contexto curado con `build_results_context.py` (sólo cambió el mapa del
documento LaTeX, que además quedó al día con B1–B9 y R2–R3; ningún número de resultados). Auditoría en verde, 65
págs.; cuerpo 1–42.
No hay otras menciones en el texto.

**R6 — Ciudades activas y selección por *trial*.** Del código (`h1/ml/pes_trf/ext/pandemic.py`, `new_city`, `reset`):
en cada paso llega una ciudad nueva y todas las ya incorporadas siguen activas, así que en el paso $t$ hay $t+1$
ciudades activas, hasta $L \le 10$. El agente no elige la ciudad: el orden y la severidad inicial vienen dados por la
secuencia. En la evaluación, la fila correspondiente de `initial_severity.csv`/`sequence_lengths.csv`; en el
entrenamiento, un sorteo de las distribuciones empíricas de esos archivos (`number_cities_prob`, `severity_prob`).
Escribirlo explícitamente en §4.2, cerca de la Ec. 4.4.

*Hecho R6 (2026-11-12):* tres frases en `04Materials.tex` §4.2 (págs. 13–14): (1) «llega la ciudad siguiente de la
secuencia, que el agente no elige: sólo decide cuántas unidades de recurso le asigna»; (2) los 9 recursos
preasignados van a dos ciudades iniciales del protocolo heredado de PES (`INIT_NO_OF_CITIES = 2`, severidades 4 y 3,
recursos 3 y 6 con `numpy.random.seed(3)` en `__main__.py`), que no intervienen en las decisiones ni en la métrica;
(3) «Las ciudades incorporadas siguen activas hasta el final de la secuencia (en el paso $t$ hay $t+1$)». La primera
versión desbordaba 1 renglón; se compactaron (1) y (3). Auditoría en verde, 65 págs.; cuerpo 1–42 (pág. 23 llena).
*Ajuste por el juez (incorporado):* en la evaluación las dos ciudades iniciales siguen visibles y evolucionando
($t+3$ en pantalla), así que (3) dice ahora «en el paso $t$ hay $t+1$, sin contar las dos iniciales». El juez
verificó contra el código las cuatro afirmaciones (9 = 3 + 6 con `seed(3)`; fuera del estado y de la métrica; $t+1$;
el agente no elige la ciudad). Cuerpo 1–42.

**R7 — Q-Learning optimizado.** Explicar que «optimizado» significa que sus hiperparámetros ($\alpha_{lr}$, $\gamma$,
$\varepsilon_0$, $\varepsilon_{\min}$ y número de episodios; `h1/tabular/pes_ql/ext/optimize_rl.py:197-201`) se eligieron
con optimización bayesiana (TPE, §4.6), mientras que Q-Learning base usa los valores originales de PES. El algoritmo
es el mismo. Ajustar la descripción de la Tabla 4.1 y la §4.4.1 y adelantar una remisión a §4.6.

*Hecho R7 (2026-11-12):* directiva del autor: aclararlo una sola vez. En `04Materials.tex` §4.4 (pág. 16), la primera
mención antes de la Tabla 4.1 dice ahora que Q-Learning base usa los hiperparámetros originales de PES, sin
optimizar, y Q-Learning los elegidos por optimización bayesiana (remisión a §2.6), como los demás modelos. En §4.4.1
se quitó la frase que lo repetía («Q-Learning base y Q-Learning reducen $\varepsilon$ linealmente»). La tabla no
cambia. Auditoría en verde, 65 págs.; cuerpo 1–42.

**R8 — LSTM sin estado oculto entre decisiones.** Justificar con
`h1/ml/pes_rdqn/doc/pes_rdqn_theory.md` §4: la variante truncada reinicia la LSTM en cada pasada y procesa una ventana
fija de $W = 6$ estados. Así el *buffer* de repetición guarda ventanas de forma constante y muestras decorrelacionadas
(Hausknecht y Stone, 2015). Además, el Transformer recibe la misma ventana, por lo que la comparación de la §6.1
aísla la arquitectura y no la cantidad de historia. Mencionar la alternativa (propagar el estado dentro del episodio)
como trabajo futuro o limitación.

*Hecho R8 (2026-11-12):* en `04Materials.tex` §4.4.2, ítem «DQN recurrente» (pág. 17): la LSTM parte de cero en cada
ventana, como en las actualizaciones aleatorias de Hausknecht y Stone; así el *buffer* guarda ventanas de forma fija
muestreadas de manera independiente, y la LSTM recibe la misma información que el Transformer, de modo que su
comparación aísla la arquitectura (remisión a §6.1). Sin cita nueva. Para no desbordar el Cap. 4 se acortaron tres
frases del mismo capítulo sin cambiar su contenido (§4.3 programación dinámica; §4.4.2 introducción de las redes; A2C).
Auditoría en verde, 65 págs.; cuerpo 1–42.
*Ajustes por el juez (incorporados):* «análogo a las actualizaciones aleatorias» (en DRQN se desenrollan segmentos y
al jugar se conserva el estado; el PDF del repo es sólo un puntero LFS, la analogía se apoya en el artículo) y «su
diferencia no proviene de la entrada» en lugar de «su comparación aísla la arquitectura», que chocaba con §6.1 (el
efecto no se separa de las conexiones residuales ni de los hiperparámetros). El juez verificó en el código que el
estado oculto parte de cero al decidir y al entrenar y que el *buffer* guarda ventanas (6, 3) muestreadas
uniformemente. Cuerpo 1–42.

**R9 — Figuras 5.1 y 5.3.** ❓ Opciones: (a) en el cuerpo, una figura compacta por familia de escenarios (referencia,
media de severidad, de longitud, conjunta, estructural, fuera de rango y media *held-out*: 7 × 7 = 49 valores) y los
mapas completos en el Apéndice C (recomendado); (b) quitar los números de las celdas y dejar sólo el color. La opción
(a) requiere una figura nueva en `h1/general/scripts/figures.py` (agente `python-h1`), luego
`writings/auxiliar/scripts/sync_figures.py` y ajustar el texto que cita «15 de 21 escenarios».

*Descartada (2026-11-12):* decisión del autor; considera la observación irrelevante. Las Figuras 5.1 y 5.3 quedan
como están. Consecuencia para el presupuesto: era la principal compensación prevista; el cuerpo sigue en 1–42 y
quedan como opciones los finales de capítulo cortos (Cap. 3 pág. 12, Cap. 6 pág. 39, Cap. 7 pág. 42).

**R10 — Criterio de la Tabla 5.7.** Explicar en el texto: para cada configuración se calcula, secuencia por secuencia
en las 320 secuencias *held-out*, la diferencia de $\bar r$ con el Transformer; «supera» significa que la media de esas
diferencias es mayor que dos veces su error estándar (aprox. 95 % de confianza para una diferencia positiva). Verificar
el cálculo exacto en `writings/auxiliar/scripts/weighted_ens_sensitivity.py` antes de redactarlo.

*Hecho R10 (2026-11-12):* en `05Results.tex` §5.5 «Sensibilidad» (pág. 33 aprox.), el párrafo que sigue a la lista
explica ahora el criterio: comparación pareada en las 320 secuencias *held-out*; «supera» = media de las diferencias
de $\bar r$ (ensamble − Transformer) mayor que dos errores estándar (desviación/$\sqrt{320}$), ≈ prueba $t$ pareada con
$p < 0{,}05$; con los valores fijos, $0{,}010$ ± $0{,}002$. Verificado en `weighted_ens_sensitivity.py` (l. 368–375) y
en su JSON: el criterio reproduce la última columna de la tabla (p. ej. $\tau = 30$: 2,2 EE → sí; $\tau = 1$: 1,1 → no;
$w_\pi = 0{,}35$: 1,7 → no; peso base 1: 1,8 → no). El título de la tabla no cambió. Auditoría en verde, 65 págs.;
cuerpo 1–42.
*Ajuste por el juez (incorporado):* la frase siguiente decía que con el *prior* y $\tau = 1$ el ensamble «supera» al
Transformer en un $0{,}1\%$, lo que contradecía el criterio ($t = 1{,}15$; la tabla excluye $\tau = 1$). Ahora dice
«la diferencia es de un $0{,}1\%$, no significativa». El juez verificó el criterio en las 28 configuraciones.

## Orden de trabajo sugerido

1. Triviales y menores: R5, B4, R4, R10, R8, R7.
2. Entorno y estructura: R6, R3, B2, B1, B5.
3. Decisiones del autor: B6, R2, R9 y B9 (bloquean el resto del cuerpo).
4. Discusión y conclusiones: B8, B7, B3, B9.
5. Pasada de tono R1, capítulo por capítulo.
6. Revisión independiente con `thesis-judge` contra ambos dictámenes, compilación final y medición de páginas.
7. Opcional: nota de respuesta al jurado con el cambio aplicado en cada punto (archivo y página).

## Defensa (Bernardez, «Recomendaciones adicionales»)

No es una corrección del texto: son tres temas para llevar preparados. Material en `dict/defensa/`.

- **Criterio de elección de los modelos** ✅: Apéndice E.1 (`tab:model-choice`), basado en B6.
- **Política aprendida frente a la regla $S+k$** ✅: Apéndice E.2 (`fig:policy-rule`, `tab:policy-rule`), generado
  con `writings/auxiliar/scripts/policy_vs_rule.py` a partir de las decisiones registradas, sin volver a ejecutar
  modelos. Óptimo $S+1{,}6$, regla $S+2$ → $S+1{,}5$ efectivo, ponderado $S+1{,}6$, Transformer $S+2{,}2$ (en sus
  propias trayectorias; el $S+2{,}4$ de §6.2 se mide en los estados del ensamble) y DQN ≈ 7,7 constante.
- **Variabilidad entre semillas** ✅: Apéndice E.3 (`ap:seeds`, `tab:seeds`) y sección 3 de
  `dict/defensa/README.md`. `h1/general/scripts/seed_variability.py` reentrena con la configuración entregada,
  llamando directamente a las funciones de entrenamiento (nunca a `train_*`, que sobrescriben `inputs/`), y evalúa
  con el protocolo determinista, que reproduce a los entregados (0,887/0,881; 0,894/0,891; 0,927/0,929). Semillas
  nuevas: Q-Learning y DQN 5 (1001–1005), Transformer 4 (1001, 1002, 1004, 1005; la 1003 se cortó al ~15 % por
  decisión del autor y queda para otra PC: `run --models trf --seeds 1003`). Huellas de los entregados verificadas
  (`h1/general/work/seeds/hashes_entregados.txt`); modelos por semilla versionados con Git LFS (commit `9a45569`).
  - *Resultados* (`h1/general/results/individual/seed_comparison.json`): Q-Learning $0{,}859 \pm 0{,}022$ /
    $0{,}866 \pm 0{,}016$; DQN $0{,}874 \pm 0{,}005$ / $0{,}876 \pm 0{,}005$; Transformer $0{,}927 \pm 0{,}009$ /
    $0{,}927 \pm 0{,}010$ (ref. / *held-out*). La peor semilla del Transformer supera a la mejor de DQN; DQN y
    Q-Learning se superponen (la mejor de Q-Learning supera a las 5 de DQN); el DQN entregado supera a sus 5
    semillas (sesgo de Optuna) y el Transformer entregado queda dentro de una DE de su media; la ventaja del ponderado
    sobre la media del Transformer (0,010 y 0,012) es del orden de su desviación estándar. Alcance: referencia y
    réplicas, no los 21 escenarios de generalización.
  - *Hecho (2026-11-13):* Apéndice E.3 nuevo (p. 59 del PDF; no cuenta para el tope) y primer ítem de §7.3
    reescrito («Extender el reentrenamiento… a los demás modelos, a los ensambles y a los escenarios de
    generalización», remite al Apéndice E.3). Ajustes de `thesis-judge` aplicados: «se conserva la ventaja del
    Transformer» en lugar de «el ordenamiento»; alcance sin los 21 escenarios; la quinta semilla del Transformer se
    interrumpió por costo; §7.3 ya no dice «cada modelo se entrenó una sola vez»; $n$ = semillas nuevas en el
    epígrafe. Auditoría en verde, 66 páginas; cuerpo 1–42, sin cambios.
  - *Ampliado a 10 semillas (2026-11-15):* en otra PC (Python 3.12.10; la evaluación de los modelos existentes y el
    reentrenamiento de QL 1001 reproducen exactamente los valores anteriores) se entrenaron QL y DQN 1006–1010 y el
    Transformer 1003 y 1006–1010, sin cambios de código (`--models/--seeds`, 8 procesos en paralelo; commit
    `f0b532d`). Semillas nuevas, $n = 10$ (ref. / *held-out*): Q-Learning $0{,}865 \pm 0{,}017$ /
    $0{,}869 \pm 0{,}016$; DQN $0{,}869 \pm 0{,}009$ / $0{,}871 \pm 0{,}009$; Transformer $0{,}919 \pm 0{,}012$ /
    $0{,}920 \pm 0{,}011$. Se mantienen la ventaja del Transformer (peor semilla 0,895 > mejor DQN 0,880), la
    superposición DQN–Q-Learning y el sesgo del DQN entregado. Cambia la ventaja del ponderado sobre la media del
    Transformer: 0,019, unas 1,5–1,7 DE (antes «del orden de la DE»). Nuevo: DQN 1007, 1009 y 1010 convergen a
    $\min(R, 10)$ en todas las decisiones (0,858 / 0,860). Apéndice E.3 actualizado (primer párrafo, Tabla E.3 y
    párrafo de resultados) y `dict/defensa/README.md` §3; §7.3 sin cambios (no cita cantidades y sus afirmaciones
    siguen valiendo). Auditoría en verde (TinyTeX nuevo en esta PC): 66 páginas, sin Overfull/Underfull ni
    warnings; cuerpo 1–42, sin cambios (medido con las etiquetas de página del PDF; E.3 en las págs. 65–66 del PDF).
    `CONFIG.py` de `pes_ens` volvió a la versión del commit (tenía un cambio ajeno) y `pes_ens` reproduce su
    referencia (0,9373125, las 64 secuencias idénticas).
  - *Ampliado a los 6 modelos y a los ensambles (2026-11-15, aprobado por el autor):* Double Q-Learning, DQN
    recurrente y A2C con 10 semillas (commit `559d3e6`) y los 6 ensambles con 10 combinaciones de semillas de sus
    miembros (`h1/general/scripts/ensemble_seeds.py`). Apéndice E.3: primer párrafo (todos los modelos salvo
    Q-Learning base), Tabla E.3 con 6 filas, párrafo de resultados reescrito (nuevo: el Transformer y el DQN
    recurrente se superponen; A2C es el más sensible a la semilla) y Tabla E.4 + párrafo nuevos para los ensambles
    (el ponderado supera a su Transformer en 9/10 combinaciones y es el más estable). §7.3: el primer ítem ahora
    pide extender el estudio sólo a los escenarios de generalización y resume el resultado (más corto que antes).
    Números generados por `writings/auxiliar/scripts/seed_comparison.py` (extendido a 6 modelos y 6 ensambles).
    Auditoría en verde (Tabla E.3 en `\footnotesize`, Tabla E.4 con `adjustbox` para evitar Overfull): 67 páginas;
    cuerpo 1–42 sin cambios, §7.3 en la pág. 41; E.3 en las págs. 65–67 del PDF. `dict/defensa/README.md` §3 al día.
