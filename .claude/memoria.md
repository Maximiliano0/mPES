# Memoria de Claude Code — mPES

Copia versionada de la memoria local de Claude Code (`%USERPROFILE%\.claude\projects\<proyecto>\memory\`), que no
viaja con el repositorio. `.claude/CLAUDE.md` la importa, así que en otra PC se carga sola al abrir el proyecto. El
estado vigente de las correcciones está en [dict/correcciones.md](../dict/correcciones.md); si algo de este archivo lo
contradice, manda `correcciones.md`.

## Flujo de correcciones de la tesis (fijado por el autor)

1. El autor describe el error y la corrección que propone.
2. Claude evalúa y analiza la observación y la propuesta, sin editar.
3. Claude propone **una única** solución justificada (la del autor, ajustada u otra, con el porqué). Sin menús de
   variantes.
4. Claude explica cómo la aplicará y en qué documentos impacta.
5. Claude pide permiso. Recién entonces edita. La aprobación vale sólo para ese ítem.

- Claude no toma por su cuenta el próximo ítem de `dict/correcciones.md` ni propone correcciones que el autor no
  planteó. Con «Dame ideas», Claude redacta él mismo el contenido.
- Si al aplicar algo aprobado aparece un imprevisto (por ejemplo, una página que se desborda), se detiene y consulta.
- Después de aplicar: `/thesis-audit`, medir páginas del cuerpo y actualizar `dict/correcciones.md`.

## Tope de 42 páginas = sólo el cuerpo

Cuenta de la Introducción a las Conclusiones. Portada, resúmenes, índices, referencias y apéndice no cuentan
(decisión del autor, 2026-09-28). El 2026-11-13 el autor fijó el tope en 42 páginas (antes 40), igual al cuerpo actual: no se recorta para bajar a 40, y ninguna corrección nueva puede pasar de 42. Se mide con `pdftotext -layout` sobre el PDF compilado: el cuerpo termina en la
página anterior a «Referencias». Cada capítulo empieza en página nueva, así que una línea de más en el final de un
capítulo cuesta una página entera. Toda corrección que agregue texto informa su impacto en páginas.

## Enmascarado con −1e9: intencional

En los modelos individuales, las acciones infactibles se ponen en $-10^9$ antes de calcular la confianza por entropía
(p. ej. `h1/tabular/pes_base/ext/pandemic.py`, `options[o > resources_left] = -1e9`). Con menos de 10 recursos
restantes las acciones factibles parecen casi uniformes. Es la definición de confianza elegida por el autor («es a
drede», 2026-09-28): no reportarlo como error ni cambiarlo. El doble softmax de `pes_ens_consensus_prior` era otro
problema y ya se corrigió.

## `Acknowledgement.tex` fuera de `Main.tex`: intencional

`audit.py` lo lista como «.tex huérfano»; ese aviso es esperado. «0 advertencias» en la auditoría significa «sólo ese
aviso». No proponer incluirlo.

## Correcciones del jurado

Dos pre-dictámenes (Bernardez y Robino) en `dict/*.pdf`; ambos aprueban la defensa y las observaciones son
sugerencias. El texto que leyó el jurado es el de master @ `9a1cdc4` (`dict/mPES_pre_dictamenes.pdf`), así que sus
números de tabla, figura y cita corresponden a ese PDF. Seguimiento con IDs (B1–B9, R1–R10), estados y presupuesto de
páginas en `dict/correcciones.md`; se aplican con `/jury-corrections`.

## Variabilidad por semilla (estado al 2026-11-15)

- Q-Learning, DQN y el Transformer reentrenados con 10 semillas nuevas cada uno (1001–1010) con
  `h1/general/scripts/seed_variability.py`. Las últimas 16 (QL y DQN 1006–1010; Transformer 1003 y 1006–1010) se
  corrieron el 2026-11-14 en otra PC, sin cambios de código (`--models/--seeds`, 8 procesos en paralelo: DQN y el
  Transformer son de un hilo por diseño). Esa PC reproduce exactamente los valores anteriores. Resumen en
  `h1/general/results/individual/seed_variability.json` y comparación en `seed_comparison.json`
  (`writings/auxiliar/scripts/seed_comparison.py`).
- Resultados (r̄, semillas nuevas, n = 10, media ± DE, ref. / held-out): QL $0{,}865 \pm 0{,}017$ /
  $0{,}869 \pm 0{,}016$; DQN $0{,}869 \pm 0{,}009$ / $0{,}871 \pm 0{,}009$; Transformer $0{,}919 \pm 0{,}012$ /
  $0{,}920 \pm 0{,}011$. Transformer − DQN: $5{,}0$ y $4{,}9$ puntos, sin superposición de rangos (peor Transformer
  0,895 > mejor DQN 0,880). DQN y QL se superponen. El ensamble ponderado supera a las 10 semillas del Transformer;
  su ventaja sobre la media (0,019) equivale a 1,5–1,7 DE.
- DQN 1007, 1009 y 1010 convergen a la política trivial $\min(R, 10)$ en todas las decisiones (0,858 / 0,860):
  redes distintas, trayectorias idénticas. No es un error.
- Modelos por semilla y `result.json` en `h1/general/work/seeds/` (binarios en Git LFS: `git lfs pull`). Los
  `run_stdout*.log` de entrenamiento (cientos de MB) no se versionan.
- En la tesis: Apéndice E.3 (`ap:seeds`, actualizado a n = 10 el 2026-11-15) y primer ítem de §7.3 (sin cantidades;
  no cambió); material en `dict/defensa/README.md` §3.
- Páginas: recompilado el 2026-11-15, 66 páginas; el cuerpo sigue en 1–42 (tope) y E.3 ocupa las págs. 65–66 del
  PDF.
- Ampliado el 2026-11-15 a Double Q-Learning, DQN recurrente y A2C (10 semillas cada uno; el DQN recurrente usa la
  arquitectura de `CONFIG.py`, LSTM 64, no la de `best_params.json`). Semillas nuevas (ref. / held-out): DQL
  $0{,}874 \pm 0{,}009$ / $0{,}876 \pm 0{,}008$; RDQN $0{,}902 \pm 0{,}020$ / $0{,}898 \pm 0{,}023$; A2C
  $0{,}842 \pm 0{,}035$ / $0{,}842 \pm 0{,}037$. El Transformer y el DQN recurrente se superponen (medias a 0,016 /
  0,022). En la tesis desde el 2026-11-15 (Tabla E.3 con 6 filas).
- Ensambles (2026-11-15, `h1/general/scripts/ensemble_seeds.py`): reglas y parámetros fijos, miembros de 10
  combinaciones de semillas (mejores, peores, más cercanas a la media y 7 al azar). Ponderado $0{,}928 \pm 0{,}006$ /
  $0{,}932 \pm 0{,}004$ y supera a su Transformer en 9 de 10; compuerta ≈ su Transformer; los ensambles con A2C
  (votos, consensos) tienen desvíos de 0,015–0,022 y casi nunca superan a su Transformer. Los entregados quedan por
  encima de la media de sus combinaciones. Resumen en `h1/general/results/ensemble/ensemble_seeds.json`. En la tesis
  desde el 2026-11-15: Tabla E.4 (`tab:ens-seeds`) y su párrafo; §7.3 resume ambos estudios. Cuerpo sigue en 1–42.

## Restaurar la memoria local (opcional)

No hace falta: `.claude/CLAUDE.md` ya importa este archivo. Si además se quiere la memoria nativa de Claude Code en
la otra PC, basta con pedirle a Claude que guarde cada sección de este archivo como memoria.
