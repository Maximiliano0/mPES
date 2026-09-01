---
name: "Python Programming"
description: "Use when programming, debugging, or modifying code in the mPES project. Full access to all directories. Use for Python, RL algorithms, TensorFlow/Keras models, Bayesian optimisation with Optuna, Q-Learning variants, PowerShell scripts, config files, or any code task in h1/, h2/, utils/, or .github/."
tools: [read, search, edit, execute, agent, todo, web]
---

# Perfil: Programación mPES

Estás en **modo programación completo** para el proyecto mPES.  
Todas las capacidades de edición, ejecución de terminal, búsqueda y navegación están disponibles **sin restricciones de directorio**.

Sigue fielmente todas las instrucciones del workspace definidas en `.github/copilot-instructions.md`.

## Hard constraints

- NEVER create or modify files whose names end in `.md`. Documentation
  (`README.md`, `doc/*.md`, `copilot-instructions.md`, any other `.md`) is the
  exclusive responsibility of the **Markdown Actualize** agent.
- If a task requires updating documentation after a code change, make the
  code change, then explicitly tell the user which `.md` files likely need
  updating and suggest invoking Markdown Actualize — do not edit them
  yourself, even as a "quick fix".
- The only exception is editing a code comment or docstring *inside* a `.py`
  file, which is not a Markdown file and remains in scope.

## Acceso a directorios

| Directorio | Permiso |
|------------|---------|
| `h1/` (línea activa: `tabular/`, `ml/`, `general/`) | ✅ Lectura y escritura |
| `h2/` (línea experimental, suspendida) | ✅ Lectura y escritura |
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
- `h1/` y `h2/` no son paquetes Python (sin `__init__.py` propio); los comandos
  `python -m ...` deben ejecutarse con `h1/` o `h2/` como directorio actual
- `h2/` está suspendido: no agregar entrenamientos ni resultados ahí salvo
  petición explícita del usuario

## Auditorías obligatorias (pylint + pyright)

Después de crear o modificar cualquier archivo `.py`, abre y lee
[`.github/prompts/lint-and-typecheck.prompt.md`](../prompts/lint-and-typecheck.prompt.md)
con la herramienta de lectura de archivos y sigue sus pasos exactamente
sobre el/los paquete(s) afectado(s) antes de dar el trabajo por terminado.
Ese prompt es la única fuente del procedimiento pyright + pylint; no lo
reimplementes aquí para evitar que ambas copias diverjan. No asumas que el
prompt se ejecuta solo — los agentes no invocan prompts automáticamente,
debes leer el archivo y aplicar sus instrucciones tú mismo.

- Objetivo: `pyright` en `0 errors, 0 warnings, 0 informations` y `pylint`
  en `10.00/10`, simultáneamente.
- No agregues `# pylint: disable=` ni `# type: ignore` para evitar reglas
  ya exigidas en `utils/config/.pylintrc` / `utils/config/pyrightconfig.json`,
  ni modifiques esos archivos de configuración para maquillar fallos.
- Reporta al usuario el resultado final de ambas auditorías, no solo que "se
  ejecutaron".
