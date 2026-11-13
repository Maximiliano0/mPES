---
name: create-pkg
description: Crea el esqueleto de un paquete pes_* nuevo de mPES (tabular/, ml/ o ens/) para probar un modelo o arquitectura, replicando el layout, las convenciones y el pipeline optimize → train → run de los paquetes existentes.
argument-hint: "<pes_nombre> <algoritmo o arquitectura>"
disable-model-invocation: true
---

# Crear paquete mPES

Argumentos: `$ARGUMENTS` (nombre del paquete y algoritmo; si falta alguno o el grupo es ambiguo, pregunta antes).

1. Lee `.github/prompts/create-pkg.prompt.md` completo y ejecuta sus pasos 0–7 (paquete de referencia, layout,
   adaptación, configuración, docs, actualizaciones globales y validación).
2. En el paso 6, además de `.github/copilot-instructions.md`, agrega el paquete a la tabla de `.claude/CLAUDE.md` si
   corresponde y a los ejemplos de uso de las skills `audit-project`, `update-pkg-docs` y `lint-and-typecheck` cuando
   enumeren paquetes.
3. Sólo andamiaje: no entrenes, no optimices, no hagas commit. Cierra con la skill `lint-and-typecheck`.
