
## v5 (23/09/2026): consistencia, figuras y redundancia
- Referencias unificadas:
  - "Capítulo~\ref" para capítulos y "Sección~\ref" para secciones (se eliminó "Sec.").
  - "Ecuación~\eqref" en todas las ecuaciones.
  - Babel con `es-tabla`, para que los captions y el índice digan "Tabla" (antes decían "Cuadro").
  - `tocdepth` pasa de 4 a 3.
- Títulos en minúscula tipográfica: "Procesos de decisión de Markov", "Gradiente de política y actor–crítico", "Definición del problema", "Estructura del documento".
- Nombres: "DQN recurrente" en los dos resúmenes (antes decía "DQN con LSTM").
- Nuevas etiquetas y referencias:
  - Nueva etiqueta `sec:stats-bg`, citada desde la Tabla 5.2 y desde el protocolo de evaluación.
  - El Apéndice B, que antes no se citaba, ahora se cita desde el protocolo.
  - `sec:baseline-random` pasa a llamarse `sec:metric`.
- Figuras eliminadas (de 33 a 24 imágenes; las retiradas pasaron a `auxiliar/descartados/figuras/`):
  - Rankings `ind_08` y `ens_08`: duplican las Tablas 5.1 y 5.3.
  - Degradación `ind_02` y `ens_02`: se obtienen restando la primera columna del mapa de desempeño.
  - `ind_06`: las curvas de extrapolación se superponen con `ind_05`.
  - `ens_05`: se superpone con `ens_06`.
  - Tres trazas del Transformer: la confianza remapeada repite la confianza; el desempeño acumulado y el normalizado repiten `PES_TRF_results`.
  - Las dos figuras del agente aleatorio se unieron en una figura con subfiguras.
- Redundancia eliminada:
  - Párrafos de Parisotto/Chen y de Lakshminarayanan duplicados entre Marco teórico y Estado de la cuestión.
  - La fórmula de transición repetida en la Introducción.
  - La tabla de paquetes duplicaba los ensambles de la Tabla 4.5.
  - La ecuación de media redundante en Resultados.
  - La subsección 4.7 se integró en Ensambles (4.5.8).
- Resultado: 56 páginas (antes 59), auditoría 29/29, sin páginas vacías, sin "??".
- Fuentes en `claude/tesis_v5/`.
- Sigue pendiente fuera de `writings/`:
  - Las figuras `PES_*` tienen texto en inglés.
  - `ind_05` todavía dice "perturbación" en la leyenda.
  - Hay que verificar el embedding posicional del Transformer en h1.

## v5.1: recorte para cerrar capítulos en su página
- Introducción: se condensaron Motivación, Definición del problema, la frase sobre la segunda pregunta, las contribuciones 2 y 3, y la estructura del documento. El capítulo ahora ocupa 2 páginas (antes 3).
- Marco teórico: la ecuación $P_{test} = P_{train}$ pasó a estar en línea (no se citaba) y el capítulo ahora ocupa 8 páginas (antes 9).
- Se mantienen los saltos de página entre capítulos, que pidió el director.
- Resultado: 54 páginas, auditoría 29/29.
