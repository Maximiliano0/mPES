---
name: jury-corrections
description: Trabaja una observación del jurado de tesis (pre-dictámenes en dict/) a partir de la descripción del error y de la corrección que propone el autor. Evalúa, propone una única solución justificada, explica cómo y dónde la aplicará, pide permiso y, una vez aprobada, la aplica, audita, mide páginas y actualiza dict/correcciones.md. Úsala cuando el usuario plantee una corrección del jurado o pida el estado ("/jury-corrections estado").
argument-hint: "<descripción del error y de la corrección propuesta> | estado"
---

# Correcciones del jurado

Pedido: `$ARGUMENTS`.

- `estado` → lee `dict/correcciones.md` y resume qué está hecho, qué falta y el presupuesto de páginas. No edites
  nada.
- Cualquier otro texto → es la descripción del autor de un error y de la corrección que propone. Sigue el flujo.

## Flujo (lo fija el autor; no alterarlo)

1. **El autor describe el error y su corrección propuesta.** Claude no elige por su cuenta el próximo ítem ni
   propone correcciones antes de que el autor plantee una. `dict/correcciones.md` es material de consulta, no una
   agenda que Claude ejecute.
2. **Claude evalúa y analiza**, sin editar:
   - La observación del jurado (texto literal en `dict/*.pdf`; ID y detalle en `dict/correcciones.md`, si existe).
   - La corrección que propone el autor: si responde lo que pidió el jurado, si es correcta según `h1/` y el contexto
     curado (`writings/auxiliar/claude_context/`), qué efectos secundarios tiene (coherencia con otras secciones,
     resumen y abstract, referencias cruzadas, páginas) y qué le falta o le sobra.
3. **Claude propone una única solución justificada**: puede ser la del autor tal cual, una versión ajustada u otra si
   la del autor tiene un problema, explicando por qué. Va con el texto actual (archivo:línea) y el texto nuevo en
   LaTeX listo (diff o antes/después). No presenta un menú de variantes.
4. **Claude explica cómo la aplicará y en qué documentos impacta**: cada archivo que se modificará (`.tex`, `.bib`,
   figuras, código de `h1/` vía `python-h1`, `dict/correcciones.md`, contexto curado) y cada documento cuyo
   resultado cambia (PDF: índice, numeración, páginas del cuerpo; resumen y abstract si cambia un número o una
   afirmación). También los pasos: edición, `thesis-audit`, medición de páginas y `thesis-judge` si es un cambio mayor.
5. **Claude pide permiso.** No edita nada hasta la aprobación explícita, que vale sólo para ese ítem.

Sólo se le pregunta al autor lo que no se puede deducir del proyecto (decisiones de diseño suyas, documentos que
Claude no tiene), y explicando qué falta.

## Después de la aprobación

1. Aplicar exactamente lo aprobado. Si al aplicarlo aparece algo no previsto, detenerse y consultar.
2. Ejecutar la skill `thesis-audit` en modo full: todo en verde, sin referencias indefinidas.
3. Informar las páginas del cuerpo (tope 40, de Introducción a Conclusiones).
4. En cambios mayores, pasarle al agente `thesis-judge` el texto del jurado y el fragmento nuevo, sin la valoración
   propia, y reportar su veredicto.
5. Actualizar `dict/correcciones.md`: estado ✅, una línea «Hecho:» con archivo, sección y página del PDF, y las
   estimaciones de páginas.
6. Reportar lo hecho. No hacer commit salvo pedido.

## Convenciones

Las de `writings/auxiliar/claude_context/instrucciones_sistema.md` §5–§7 y `CLAUDE.md`. No inventar datos ni citas:
toda cifra sale de `h1/` o de `mpes_resultados.json`, y toda cita nueva necesita un DOI o una URL verificados.
