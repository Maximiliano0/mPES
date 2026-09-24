# v6: correcciones del informe de "AI slop" (23/09/2026)

**Resultado:** la tesis tiene 55 páginas (antes 54), la auditoría pasa los 29 controles y no hay páginas en blanco ni referencias sin resolver.

**Decisiones de Maxi:**
- El ensamble ponderado (`pes_ens`) se mantiene, con una nota que aclara que el paquete quedó archivado en h1.
- Donde los .md contradicen la tesis, se conserva la tesis si los conteos de parámetros entrenables solo cuadran con ella.

## Bibliografía
- **Se eliminaron:**
  - Fon y Parisi (2003). Es un artículo de economía del derecho sobre litigios y no corresponde a la temática.
  - Andrychowicz et al. (2021). Era una cita decorativa.
- **Se agregaron:**
  - Shannon (1948), para la entropía.
  - Schulman et al. (2016), para GAE.
  - Wiering y van Hasselt (2008), como antecedente de ensambles de algoritmos de RL distintos.
- **Se quitaron notas no verificables** de BCINE2022 (financiamiento) y de Towers2024 (aceptación).

## Cambios tomados de los .md de h1
- **Métrica:** según `pes_ql_theory.md`, $S_{\mathrm{mejor}}$ es una asignación voraz informada, no el resultado de programación dinámica.
  - $\bar r = 1$ ahora se describe como "igualar la asignación de referencia".
  - Se corrigió esto en el resumen, en Materiales, en Resultados y en Limitaciones.
- **Double Q-Learning (Tabla 4.2):** los hiperparámetros ahora son los de `best_params.json` según `pes_dql_explained.md`: α 0,2593; γ 0,9806; ε 0,8392/0,0799; w 0,0240; q 0,5174; κ 0,2177; 860 000 episodios.
- **PBRS:** el potencial es Φ(s) = −Σᵢ Sᵢ, la suma de las severidades de las ciudades (antes decía −S).
- **Transformer:** los bloques son Pre-LN, según `pes_trf_explained.md`.
- **DQN:** el entrenamiento final usa 175 000 episodios (`DQN_EPISODES`); antes decía 40 000. Se aclaró que en DQN, A2C y el Transformer cada ensayo usa un presupuesto reducido de episodios.
- **Entrenamiento:** se construye con los mismos archivos de severidades y longitudes que la referencia. Esto se agregó a Limitaciones.

## Contradicciones entre los .md y la tesis donde se conservó la tesis
Hay que corregir estos puntos en los .md de h1.
- **`pes_trf_explained.md`:**
  - Dice que el embedding posicional es aprendido y que después del último token hay directamente un `Dense(11)`.
  - Los 27 019 parámetros solo cuadran con un vector posicional fijo más una cabeza `Dense(32)` + `Dense(11)`.
  - La tabla de Optuna del .md (lr ≈ 5e-4, γ ≈ 0,97, arquitectura en la búsqueda) no coincide con la Tabla 4.3.
- **`pes_rdqn_explained.md`:** dice LSTM(64) → Dense(64), lr 0,001 y γ 0,96. Los 34 027 parámetros solo cuadran con dos capas de 96, como dice la tesis.
- **`pes_ens_explained.md`:** marca `pes_ens` como archivado, pero el benchmark vigente lo incluye.
- **Varios .md** traen tablas comparativas con medias viejas, por ejemplo `pes_base` ≈ 0,65 o `pes_rdqn` ≈ 0,91.

## Correcciones de contenido (informe SLOP)
- **Conclusión general (7.4) y respuesta 7.1:** el Transformer tiene el mayor desempeño, pero no es el que menos pierde. Cae 0,067, frente a 0,053 de DQN.
- **Estado del agente:** se agregó en 4.2.2 que (R, t, S) omite la severidad de las ciudades ya atendidas y la longitud de la secuencia. Esa severidad está en la recompensa, pero no cambia qué acción conviene.
- **Motivación del Transformer (3.2):** se reescribió en consecuencia.
- **Referencias de trabajos previos:**
  - Decision Transformer: se aclaró que la tesis toma la arquitectura, no el enfoque.
  - Parisotto et al.: se precisó qué encontraron.
  - Lakshminarayanan et al.: se reformuló. Ahora respalda el uso de la entropía en 2.7 en lugar de la frase sin fuente.
- **Resultados:**
  - Se quitó "no se explica por azar" y se agregó por qué los valores p deben leerse con cautela.
  - "35 % / 45 % menos de severidad sin resolver" pasó a "reduce la distancia a la referencia".
  - Se agregó una salvedad sobre la extrapolación de severidad y sobre qué incluye la media de generalización.
  - "prácticamente el óptimo" pasó a "iguala la asignación de referencia".
- **Discusión:**
  - Se reescribió 6.1 con lo que los datos descartan y se contrastó con Hausknecht y Stone, con Parisotto et al. y con Packer et al.
  - En 6.2, las afirmaciones no medidas (frecuencia de la compuerta y de las heurísticas) quedaron como inferencias.
  - Se corrigió el cierre, que generalizaba desde un único caso.
- **Resumen:** ahora menciona las dos reglas fijas del ensamble ganador.
- **Estilo:**
  - Los títulos pasaron a ser descriptivos.
  - Se quitaron coloquialismos ("chico", "la columna que importa", "da lo mismo").
  - Se eliminaron la apertura genérica de la Introducción y las repeticiones de "se amplifica" y de "banco de pruebas".
- **Marco teórico:** se quitaron la ecuación de atención no causal y el KDE, que no se usaban.
- **Figura de confianza del Transformer:** pasó de 4.5.8 al Apéndice D, con la aclaración de que no es comparable con los umbrales de los ensambles.

## v6.1: respuestas de Maxi sobre los pendientes
- Se agregó Parisi et al. (2021) en la Sección 1.2, como ejemplo de modelo compartimental con resolución espacial.
- `pes_ens` no está archivado, se analiza: se quitó la nota de Materiales 4.5.
- "Mayor media en 16 de 22" pasó a "15 de 22, y en otros 2 empata con el mejor", según la figura.
- La ventana y la arquitectura del Transformer y del DQN recurrente se eligieron por optimización bayesiana. Se corrigieron Materiales 4.4.2 y la Discusión 6.1.
- Se dividieron las oraciones de más de 40 palabras.
- Resultado: 55 páginas y auditoría 29/29.
