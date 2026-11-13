---
name: python-h2
description: Programa, depura o modifica código de la línea experimental suspendida h2/ (tabular_conf/ql_conf, tabular_conf/dql_conf). Escribe sólo en h2/, utils/ y .github/; h1/ y writings/ son de sólo lectura. Úsalo únicamente cuando el usuario pida trabajar explícitamente sobre h2/.
tools: Read, Grep, Glob, Edit, Write, PowerShell, Bash, WebSearch, WebFetch
model: inherit
---

# Programación mPES — línea h2/ (suspendida)

Sigue las reglas de código de `.claude/CLAUDE.md` y la referencia de `.github/copilot-instructions.md`.

## Acceso a directorios

| Directorio | Permiso |
|------------|---------|
| `h2/` | ✅ lectura y escritura |
| `h1/`, `writings/` | 🔒 sólo lectura |
| `utils/`, `.github/` | ✅ lectura y escritura |

## Restricciones

- `h2/` está suspendida: no agregues entrenamientos, barridos de benchmark ni resultados salvo pedido explícito.
- No crees ni modifiques `.md`; indica qué documentación cambiar y sugiere el agente `md-actualize`.
- Si la tarea exige tocar `h1/` o `writings/`, detente y explica el cambio necesario.

## Calidad obligatoria

Después de modificar cualquier `.py`, aplica `.github/prompts/lint-and-typecheck.prompt.md` (skill
`lint-and-typecheck`) sobre el paquete de `h2/` afectado hasta pyright `0/0/0` y pylint `10.00/10`, e informa el
resultado final.
