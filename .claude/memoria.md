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

## Tope de 40 páginas = sólo el cuerpo

Cuenta de la Introducción a las Conclusiones. Portada, resúmenes, índices, referencias y apéndice no cuentan
(decisión del autor, 2026-09-28). Se mide con `pdftotext -layout` sobre el PDF compilado: el cuerpo termina en la
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

## Variabilidad por semilla (estado al 2026-11-13)

- Q-Learning y DQN reentrenados con 5 semillas nuevas (1001–1005) y el Transformer con 4 (1001, 1002, 1004, 1005)
  con `h1/general/scripts/seed_variability.py`. La 1003 del Transformer se cortó al ~15 % (decisión del autor,
  2026-11-13) y queda para ampliar el estudio en otra PC (`run --models trf --seeds 1003` desde `h1/`, luego
  `summary` y `seed_comparison.py`). Resumen en `h1/general/results/individual/seed_variability.json` y comparación
  en `seed_comparison.json` (`writings/auxiliar/scripts/seed_comparison.py`).
- Resultados (r̄, semillas nuevas, media ± DE): QL $0{,}859 \pm 0{,}022$ / held-out $0{,}866 \pm 0{,}016$;
  DQN $0{,}874 \pm 0{,}005$ / $0{,}876 \pm 0{,}005$; Transformer (n = 4) $0{,}927 \pm 0{,}009$ / $0{,}927 \pm 0{,}010$.
  Transformer − DQN: $5{,}3$ puntos en referencia y $5{,}2$ en held-out, sin superposición de rangos. DQN y QL se
  superponen. El ensamble ponderado entregado supera a todas las semillas del Transformer (+$0{,}4$ y +$0{,}5$ puntos
  sobre la mejor).
- Modelos por semilla y `result.json` en `h1/general/work/seeds/` (binarios en Git LFS: `git lfs pull`). El
  `run_stdout.log` de entrenamiento (~550 MB) no se versiona.
- Pendiente con los resultados finales: proponer el Apéndice E.3 y una línea en §7.3, completar
  `dict/defensa/README.md` y la sección Defensa de `correcciones.md`, y bajar el cuerpo de 42 a 40 páginas.

## Restaurar la memoria local (opcional)

No hace falta: `.claude/CLAUDE.md` ya importa este archivo. Si además se quiere la memoria nativa de Claude Code en
la otra PC, basta con pedirle a Claude que guarde cada sección de este archivo como memoria.
