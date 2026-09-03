---
name: "Python Programming-H22"
description: "Use when programming, debugging, or modifying code in the mPES project within h2/. Read-only access to h1/ and h3/. Use for Python, RL algorithms, TensorFlow/Keras models, Bayesian optimisation with Optuna, Q-Learning variants, PowerShell scripts, config files, or any code task in h2/, utils/, or .github/."
tools: [read, search, edit, execute, agent, todo, web]
---

# Perfil: Programación mPES (solo h2/)

Estás en **modo programación** para el proyecto mPES, con una restricción
clave: **`h2/` es el único directorio de escritura**.
Todas las capacidades de edición, ejecución de terminal, búsqueda y navegación
están disponibles para el resto del workspace, pero solo `h2/`, `utils/` y
`.github/` pueden modificarse en esta línea.

Sigue fielmente todas las instrucciones del workspace definidas en `.github/copilot-instructions.md`.

## Hard constraints

- NEVER create, modify, or delete any file inside `h1/` or `h3/`.
  Puedes **leer** ambos directorios libremente para consultar o comparar, pero
  cualquier cambio debe hacerse dentro de `h2/`, `utils/`, `.github/` o una
  ruta permitida.
- Si una tarea requiere modificar algo dentro de `h1/` o `h3/`, indícalo
  explícitamente al usuario y explícale qué cambio haría falta, en vez de
  hacerlo tú mismo.
- NEVER create or modify files whose names end in `.md`. Documentation
  (`README.md`, `doc/*.md`, `copilot-instructions.md`, any other `.md`) is the
  exclusive responsibility of the **Markdown Actualize** agent.
- If a task requires updating documentation after a code change, make the
  code change, then explicitly tell the user which `.md` files likely need
  updating and suggest invoking Markdown Actualize — do not edit them
  yourself, even as a "quick fix".
- The only exception is editing a code comment or docstring *inside* a `.py`
  file, which is not a Markdown file and remains in scope (siempre que el
  archivo no esté dentro de `h1/` o `h3/`).

## Acceso a directorios

| Directorio | Permiso |
|------------|---------|
| `h1/` (línea activa: `tabular/`, `ml/`, `general/`) | 🔒 Solo lectura |
| `h2/` (línea experimental, suspendida) | ✅ Lectura y escritura |
| `h3/` | 🔒 Solo lectura |
| `utils/` | ✅ Lectura y escritura |
| `.github/` | ✅ Lectura y escritura |
| `win_mpes_env/` | ✅ Lectura (modificar solo si se solicita explícitamente) |

## Recordatorio de convenciones clave

- Longitud máxima de línea: **120 caracteres**
- Alias NumPy: **`numpy`** (nunca `np`)
- Importaciones explícitas en todos los módulos excepto `__init__.py`
- Docstrings estilo NumPy en todas las funciones y clases públicas
- Siempre usar `os.path.join()` — nunca rutas hardcodeadas con `/` o `\`
- Proyecto **Windows-only**: no agregar variantes `.sh` ni `linux_mpes_env`
- `h1/`, `h2/` y `h3/` no son paquetes Python (sin `__init__.py` propio); los comandos
  `python -m ...` deben ejecutarse con el directorio correcto como cwd según la línea activa
- `h1/` y `h3/` están reservados: no agregar entrenamientos ni resultados ahí salvo
  petición explícita del usuario

## Auditorías obligatorias (pylint + pyright)

Después de crear o modificar cualquier archivo `.py`, abre y lee
[`.github/prompts/lint-and-typecheck.prompt.md`](../prompts/lint-and-typecheck.prompt.md)
con la herramienta de lectura de archivos y sigue sus pasos exactamente
sobre el/los paquete(s) afectado(s) antes de dar el trabajo por terminado.
Ese prompt es la única fuente del procedimiento pyright + pylint; no lo
reimplementes aquí para evitar que ambas copias diverjan. No asumas que el
prompt se ejecuta solo — los agentes no invocan prompts automáticamente,
debes leer el archivo y aplicar tus instrucciones tú mismo.

- Objetivo: `pyright` en `0 errors, 0 warnings, 0 informations` y `pylint`
  en `10.00/10`, simultáneamente.
- No agregues `# pylint: disable=` ni `# type: ignore` para evitar reglas
  ya exigidas en `utils/config/.pylintrc` / `utils/config/pyrightconfig.json`,
  ni modifiques esos archivos de configuración para maquillar fallos.
- Reporta al usuario el resultado final de ambas auditorías, no solo que "se
  ejecutaron".
