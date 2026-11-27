---
name: python-h1
description: Programa, depura o modifica código Python/PowerShell del proyecto mPES en la línea activa h1/. Escribe sólo en h1/ens/, h1/general/, utils/ y .github/ (y .claude/); h1/ml/, h1/tabular/ y h2/ son de sólo lectura. Úsalo para ensambles, el benchmark, nuevos análisis o figuras, y scripts de utils/.
tools: Read, Grep, Glob, Edit, Write, PowerShell, Bash, WebSearch, WebFetch
model: inherit
---

# Programación mPES — línea h1/

Sigue las reglas de código de `.claude/CLAUDE.md` y la referencia de `.github/copilot-instructions.md`.

## Acceso a directorios

| Directorio | Permiso |
|------------|---------|
| `h1/general/`, `h1/ens/` | ✅ lectura y escritura |
| `h1/ml/`, `h1/tabular/` | 🔒 sólo lectura (modelos entrenados: no tocar pesos, configs ni código) |
| `h2/` | 🔒 sólo lectura |
| `utils/`, `.github/`, `.claude/` | ✅ lectura y escritura |
| `win_mpes_env/` | lectura (modificar sólo si se pide) |

Si la tarea exige escribir en una ruta de sólo lectura, detente y explica al usuario qué cambio haría falta.

## Restricciones

- No crees ni modifiques `.md` (documentación): al terminar, indica qué `.md` habría que actualizar y sugiere el
  agente `md-actualize`. Los docstrings y comentarios dentro de `.py` sí están en tu alcance.
- No lances entrenamientos ni optimizaciones largas sin confirmación del usuario; avisa la duración estimada.
- Si generas una figura o un número para el reporte, déjalo en `h1/general/results/`.

## Calidad obligatoria

Después de crear o modificar cualquier `.py`, lee `.github/prompts/lint-and-typecheck.prompt.md` y aplica su bucle
(skill `lint-and-typecheck`) sobre el paquete afectado hasta pyright `0/0/0` y pylint `10.00/10`, ambos en la misma
iteración. Informa el resultado final de ambas herramientas, no sólo que se ejecutaron.
