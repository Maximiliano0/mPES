# Revisión de "AI slop": tesis v5.1 (54 páginas)

La revisión sigue las cuatro categorías SLOP:

- **P**: fuentes débiles o mal usadas.
- **O**: afirmaciones que exceden la evidencia.
- **S**: superficialidad.
- **L**: estilo de baja calidad.

Cada hallazgo indica la sección, qué dice el texto, cuál es el problema y qué propongo. No pude comprobar las referencias en la web porque la búsqueda está deshabilitada en esta sesión. Las contrasté con los docs `pes_*` del proyecto.

## Diagnóstico general

El estilo superficial ya está limpio tras las pasadas anteriores:

- No hay palabras de moda.
- Las muletillas aparecen pocas veces: "de modo que" 7, "sólo" 12, "lo que" 8 y "Por eso" 3 veces en todo el texto.

Lo que queda es de fondo. Son afirmaciones que el propio texto o los docs del proyecto contradicen, citas que no dicen lo que se les atribuye, y una Discusión que resume los resultados en lugar de interpretarlos.

## P: Fuentes

**P1 (alta). Fon y Parisi (2003), Sección 1.2.**
- **Qué dice la entrada:** es "Litigation and the evolution of legal remedies: A dynamic model" (*Public Choice*), un artículo de economía del derecho.
- **Cómo se usa:** respalda "la estructura de los modelos dinámicos de decisión" y responde al pedido de Rodrigo de citar "un modelo de pandemia como el de Parisi".
- **Problema:** quien abra la cita encuentra un trabajo sobre litigios, no sobre pandemias ni decisión secuencial.
- **Propuesta:** preguntarle a Rodrigo a qué Parisi se refería. Mientras tanto, quitar la cita o usar una referencia estándar de decisión secuencial (Bellman, 1957; Puterman, 1994).

**P2 (alta). $S_{\mathrm{mejor}}$ "obtenida por programación dinámica", Sección 4.3.** El resumen también dice que se calcula conociendo de antemano la secuencia completa.
- **Qué dicen los docs:** según `pes_ql_theory.md`, BestCase es una "asignación voraz informada", y `optimize_rl.py` recorta `mean_perf` a $[0,1]$ para evitar valores fuera de rango.
- **Consecuencias si es voraz:**
  - no es "la mínima severidad alcanzable";
  - $\bar r$ puede superar 1;
  - el 1,00 del Transformer en las extrapolaciones ya no significa "prácticamente el óptimo".
- **Verificar en h1:** `src/exp_utils.py::calculate_normalised_final_severity_performance_metric`.

**P3 (alta). Hiperparámetros de Double Q-Learning, Tabla 4.2 frente a `pes_dql_explained.md`.**
- **El doc:** `best_params.json` da $\alpha=0{,}2593$, $\gamma=0{,}9806$, $\varepsilon$ $0{,}8392/0{,}0799$, $w=0{,}0240$, $q=0{,}5174$, $\kappa=0{,}2177$, 860 000 episodios y semilla 143. Su `mean_perf` es 0,896344, la misma media que reporta la tesis.
- **La tabla:** da $\alpha=0{,}1132$, …, $\kappa=0{,}000392$ y 360 000 episodios.
- **Potencial:** el doc define $\Phi(s)=-\sum_i s_i$ y la tesis $\Phi(s)=-S$.
- **A favor de la tabla:** sus valores caen dentro del espacio de búsqueda documentado; 860 000 episodios no.
- **Conclusión:** una de las dos fuentes está desactualizada. Verificar `inputs/best_params.json` y `pandemic.py` en h1.

**P4 (media). Citas que no dicen lo que se les atribuye.**
- **Andrychowicz et al. (2021), Sección 2.5.** El texto dice que "estudian qué decisiones de diseño inciden más", pero no lo conecta con ninguna decisión de A2C de la tesis. Hay que quitarla o decir qué recomendación se siguió, si consta en h1.
- **Lakshminarayanan et al. (2017), Sección 3.3.** El texto dice "sin cambiar el entrenamiento de cada red". Ese trabajo entrena cada red con una regla de puntuación propia y promedia redes de la misma arquitectura. Los ensambles de la tesis combinan arquitecturas distintas, después de entrenarlas, con pesos por entropía. Hay que reformular la frase y marcar la diferencia.
- **Decision Transformer (Chen et al., 2021), Sección 3.2.** Es RL *offline* condicionado al retorno. El Transformer de la tesis es un DQN con codificador Transformer. Conviene aclarar en una frase que se toma la arquitectura, no el enfoque.

**P5 (media). Afirmaciones sin cita.**
- La entropía de Shannon se define sin citar a Shannon (1948).
- "procedimiento estadístico estándar en clasificación" (2.7) no tiene fuente.
- GAE (2.5) no cita a Schulman et al. (2016).

**P6 (baja). Entradas incompletas o con notas no verificadas.**
- `ds004477` no tiene el título ni los autores del conjunto de datos.
- `BCINE2022` tiene la nota "Financiado por Dstl/UK MoD…".
- `Towers2024` tiene la nota "Aceptado en NeurIPS D&B 2025".

## O: Afirmaciones que exceden la evidencia

**O1 (alta). "el que mejor conserva su desempeño", Sección 7.4.**
- **Qué muestra la Tabla 5.1:** la caída hasta el peor escenario es 0,067 para el Transformer (0,927→0,860), 0,053 para DQN y 0,061 para A2C. En media, DQN (+0,005) y A2C (+0,009) suben más que el Transformer (+0,003).
- **Problema:** el Transformer tiene el mayor nivel, pero no es el que menos pierde.
- **Propuesta:** decir "el de mayor desempeño en los escenarios de generalización". Revisar también cómo 7.1 responde a "cuál conserva mejor" (Pregunta 1).

**O2 (alta). Por qué debería ganar el Transformer, Secciones 3.2 y 6.1.**
- **Qué dice el texto:** "una asignación insuficiente se amplifica… un modelo que ve la historia tiene información que el estado actual no contiene".
- **Problema con la Ec. 4.2:** la asignación de cada ciudad se fija al llegar, y la evolución de las ciudades anteriores no depende de las decisiones futuras. Por eso la historia no cambia qué acción conviene ahora.
- **Qué omite realmente el estado:**
  - la longitud $T$ de la secuencia y las ciudades por venir;
  - la severidad de las ciudades activas, que sí entra en la recompensa. Esto hace que la recompensa no sea función de $(s,a,s')$, así que el "MDP finito" de 4.2.1 no se cumple estrictamente sobre $(R,t,S)$.
- **Propuesta:** reescribir la motivación a partir de esa información faltante.
- **Verificar en h1** si el entrenamiento usa las mismas 64 secuencias. Si es así, la ventana podría servir para reconocer la secuencia, y esa explicación alternativa habría que mencionarla.

**O3 (media). "no se explica por azar", Secciones 5.1.2 y 7.1.**
- **Problema:** la prueba de Welch se aplica sobre secuencias agrupadas que no son independientes: son las mismas para ambos modelos, con una sola semilla y 21 escenarios juntos. El protocolo (4.8) ya lo admite.
- **Propuesta:** decir "la diferencia es consistente en las secuencias evaluadas" y dejar la $d$ como cifra principal.

**O4 (media). Lectura de la extrapolación de severidad, Sección 5.1.1 y resumen.**
- **El dato:** el Transformer obtiene 1,00 fuera de rango y 0,93 dentro.
- **Problema:** se presenta como generalización sin comprobar si el escenario es fácil. Con severidades de 10 a 12, asignar mucho puede bastar. A2C (0,93) y DQN (0,89) también quedan por encima de su referencia. Además, la lectura depende de P2.
- **Problema relacionado:** la media de generalización incluye tres escenarios estructurales que reproducen la referencia y varios más fáciles, como las secuencias cortas. "Mantuvo ese nivel" es en parte una consecuencia de cómo se construyó la media, y conviene decirlo donde se la define.

**O5 (media). Discusión 6.2.**
- **Compuerta:** "los resultados indican que eso ocurre casi siempre". No se muestra la fracción de decisiones en que la compuerta sigue al Transformer; la única evidencia es indirecta (resultados casi idénticos). Además, la Figura 4.2 muestra una confianza media de 0,116, menor que $\tau_g=0{,}2248$. Aunque esa confianza se calcula de otra forma, el lector ve una contradicción. Conviene presentarlo como inferencia.
- **Heurísticas:** "sólo la modifican en parte de los casos" no está cuantificado.
- **Cierre:** "mejoró al Transformer sólo cuando la regla le dio el mayor peso…" generaliza desde un único caso. El consenso con prior también da al Transformer el mayor peso individual (2,840) y no lo supera. Con un solo ensamble ganador no se puede separar el efecto del peso del de las heurísticas.

**O6 (media). "35 % menos de severidad sin resolver", en el resumen, 5.1, 5.2 y 7.1.**
- **Problema:** $1-\bar r$ es la brecha respecto de $S_{\mathrm{mejor}}$, no la severidad que queda.
- **Propuesta:** "35 % menos de brecha respecto de la mejor asignación".

**O7 (baja). Resumen.** Presenta al ensamble ganador como uno que "combina DQN, la red recurrente y el Transformer", sin mencionar el *prior* de severidad ni la cota de seguridad. La Discusión considera que esas heurísticas son parte de la explicación.

**O8 (baja). Discusión 6.1.**
- **Redes con severidad fuera de rango:** "Las redes… siguen decidiendo bien" omite la excepción del DQN recurrente (0,83), que Resultados sí menciona.
- **Hiperparámetros:** que el Transformer haya tenido menos ensayos no descarta un efecto de los hiperparámetros.

## S: Superficialidad

**S1 (media). Discusión.** El título "Por qué gana el Transformer" promete una respuesta que no se da. Se puede profundizar sin inventar, con lo que la tesis ya tiene:
- **DQN sin ventana** queda a $d=0{,}51$ del Transformer, así que la ventana aporta, pero de forma moderada.
- **DQN recurrente:** con la misma ventana, supera al Transformer en secuencias largas (0,89 contra 0,86).
- **Hausknecht y Stone:** el DRQN mejora con observabilidad parcial. Acá el DQN recurrente no supera a DQN en generalización (0,889 contra 0,899).
- **Parisotto et al.:** el Transformer estándar es inestable en RL. Acá funcionó con $W=6$ y dos capas.
- **Packer et al.:** interpolación frente a extrapolación. Los tabulares fallan al extrapolar severidad y las redes al extrapolar longitud.

**S2 (media). Estado de la cuestión.**
- **Problema:** cada sección es un párrafo que enumera trabajos sin ubicar a mPES entre ellos.
- **Falta el antecedente más directo:** los ensambles de algoritmos de RL distintos, como Wiering y van Hasselt (2008), que combinan Q-learning, SARSA, actor–crítico y otros con voto y Boltzmann. Hay que verificarlo antes de citarlo.
- **Falta también** trabajo previo de RL aplicado a asignar recursos en epidemias.

**S3 (baja). Contenido de manual que no se usa en el Marco teórico** (alrededor de una página):
- la pérdida cuadrática del DQN, aunque se usa Huber;
- la atención no causal, además de la causal;
- el KDE de TPE;
- el retorno y la ecuación de Bellman, que no se referencian.

**S4 (baja). Figura 4.2.** El propio texto dice que esa confianza no es la que usan los ensambles y que refleja sobre todo cuántas acciones quedan. No aporta evidencia a ninguna de las dos preguntas; conviene moverla al Apéndice o quitarla.

## L: Estilo

- **Frase repetida:** "una asignación insuficiente se amplifica en los pasos siguientes" aparece en el resumen, la Introducción, el Estado de la cuestión, Materiales y la Discusión. Dejarla sólo en Materiales, junto a la Ec. 4.2.
- **"banco de pruebas"** aparece tres veces (resumen, Introducción y Conclusión).
- **Coloquialismos:** "valores $p$ muy chicos" y "un $p$ chico" deberían ser "pequeños". También "La columna que importa", "da lo mismo que el Transformer solo" y "se parten en dos grupos".
- **Títulos periodísticos:** "¿La ventaja del Transformer es real?", "Dónde falla cada familia", "Por qué gana el Transformer" y "Qué regla de combinación funciona". Cambiarlos por títulos descriptivos.
- **Apertura genérica:** "La toma de decisiones secuenciales bajo incertidumbre es un problema central para los sistemas de inteligencia artificial."

## Lo que verifiqué y está bien

- El peso del Transformer en el ensamble ponderado es del 82 % ($5/6{,}08$).
- Las reducciones del 35 % y del 45 % son correctas.
- Los ensayos de optimización son 10 para el Transformer y 47 para DQN.
- Los tabulares quedan por debajo de 0,8 en la extrapolación de severidad.
- El DQN recurrente es el mejor con secuencias largas.
- En el voto suave y el voto por acción, DQN pesa más que el Transformer.
- "16 de los 22 escenarios": en la figura, con dos decimales, el Transformer es estrictamente el más alto en 15 y empata en 2. Hay que confirmarlo con precisión completa en h1.

## Qué se puede corregir ya y qué depende de h1 o de Rodrigo

- **Sin h1:** O1, O3, O5, O6, O7, O8, S3, S4 y L.
- **Con h1 o con Rodrigo:** P1, P2, P3, O2 y O4. Para P4, P5, S1 y S2 sólo hay que agregar referencias verificadas.
