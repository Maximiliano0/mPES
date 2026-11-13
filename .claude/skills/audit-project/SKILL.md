---
name: audit-project
description: Auditoría sistemática de corrección y consistencia código ↔ configuración ↔ documentación de un paquete de mPES (o de todo el proyecto con "all"): drift de docs, firmas, convenciones Windows, persistencia de Optuna y dependencias. Aplica las correcciones no ambiguas.
argument-hint: "<paquete>|utils|all"
disable-model-invocation: true
---

# Audit mPES — código, configuración y docs

Objetivo: `$ARGUMENTS`.

1. Lee `.github/prompts/audit-project.prompt.md` completo y ejecuta sus pasos (descubrimiento, chequeos estáticos,
   señales de auditoría, correcciones, verificación, reporte y falsos positivos conocidos). Traduce las herramientas
   de Copilot con la tabla de `.claude/CLAUDE.md`.
2. Respeta los permisos de directorio de los agentes: `h1/ml/` y `h1/tabular/` son de sólo lectura por defecto; si una
   corrección cae ahí, repórtala en «Surfaced for user review» en lugar de aplicarla, salvo que el usuario lo autorice.
3. El paso de pyright/pylint lo hace la skill `lint-and-typecheck`; los `.md`, con las reglas del agente
   `md-actualize`.
4. Para `all`, puedes repartir paquetes independientes entre agentes en paralelo (uno por familia: `tabular`, `ml`,
   `ens`, `general` + `utils`) y consolidar los informes; verifica tú cada hallazgo antes de editar.
5. Nada de commits, pushes ni borrados.
