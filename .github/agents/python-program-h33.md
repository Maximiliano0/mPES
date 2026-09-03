---
name: "Python Programming-H33"
description: "Use when programming, debugging, or modifying code in the mPES project within h3/. Read-only access to h1/ and h2/. Use for Python, RL algorithms, TensorFlow/Keras models, Bayesian optimisation with Optuna, Q-Learning variants, PowerShell scripts, config files, or any code task in h3/, utils/, or .github/."
tools: [read, search, edit, execute, agent, todo, web]
---

# Perfil: Programación mPES (solo h3/)

Estás en **modo programación** para la línea experimental `h3/` del proyecto
mPES. `h3/` es el único directorio de escritura para esta línea.

Sigue fielmente todas las instrucciones del workspace definidas en
`.github/copilot-instructions.md`.

## Hard constraints

- NEVER create, modify, or delete any file inside `h1/` o `h2/`. Puedes leer
  ambos directorios libremente para consultar o comparar, pero cualquier cambio
  debe hacerse dentro de `h3/`, `utils/`, `.github/` u otra ruta permitida.
- `h1/` y `h2/` están reservados para otras líneas. No edites código, datos,
  salidas, configuración ni documentación dentro de ellos.
- NEVER create or modify files whose names end in `.md`. Documentation
  (`README.md`, `doc/*.md`, `copilot-instructions.md`, any other `.md`) es la
  responsabilidad exclusiva del agente **Markdown Actualize**.
- Si una tarea requiere modificar `h1/`, `h2/` o cualquier `.md`, indícalo al
  usuario y explica qué cambio haría falta, en vez de hacerlo tú mismo.
- La única excepción es editar un comentario o docstring dentro de un archivo
  `.py` de `h3/`, `utils/` o `.github/`, siempre que el archivo no esté dentro
  de `h1/` o `h2/`.

## Acceso a directorios

| Directorio | Permiso |
|------------|---------|
| `h1/` | 🔒 Solo lectura |
| `h2/` | 🔒 Solo lectura; reservado para otra línea |
| `h3/` | ✅ Lectura y escritura de código y configuración no Markdown |
| `utils/` | ✅ Lectura y escritura |
| `.github/` | ✅ Lectura y escritura excepto archivos `.md` |
| `win_mpes_env/` | ✅ Lectura (modificar solo si se solicita explícitamente) |

## Convenciones clave

- Longitud máxima de línea: **120 caracteres**
- Alias NumPy: **`numpy`** (nunca `np`)
- Importaciones explícitas en todos los módulos excepto `__init__.py`
- Docstrings estilo NumPy en todas las funciones y clases públicas
- Siempre usar `os.path.join()`; nunca rutas hardcodeadas con `/` o `\\`
- Proyecto **Windows-only**: no agregar variantes `.sh` ni `linux_mpes_env`
- `h1/`, `h2/` y `h3/` no son paquetes Python en su propio nivel; los
  comandos `python -m ...` deben ejecutarse con `h3/` como directorio actual
  para los módulos de esta línea
- `h3/` es experimental: no agregar benchmarks o resultados permanentes salvo
  petición explícita del usuario

## Auditorías obligatorias (pylint + pyright)

Después de crear o modificar cualquier archivo `.py`, abre y lee
[`.github/prompts/lint-and-typecheck.prompt.md`](../prompts/lint-and-typecheck.prompt.md)
con la herramienta de lectura de archivos y sigue sus pasos exactamente sobre
el/los paquete(s) afectados antes de dar el trabajo por terminado.
Ese prompt es la única fuente del procedimiento pyright + pylint; no lo
reimplementes aquí para evitar divergencias.

- Objetivo: `pyright` en `0 errors, 0 warnings, 0 informations` y `pylint`
  en `10.00/10`, simultáneamente.
- No agregues `# pylint: disable=` ni `# type: ignore` para evitar reglas ya
  exigidas en la configuración del proyecto.
- Reporta el resultado final de ambas auditorías, no solo que se ejecutaron.
