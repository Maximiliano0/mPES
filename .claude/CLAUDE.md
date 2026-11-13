# CLAUDE.md — mPES (Multiple Pandemic Experiment Suite)

Tesis de Maestría en Ciencia de Datos (ITBA, 2026) de Ing. Maximiliano Vega, director Dr. Ing. Rodrigo Ramele:
«mPES: Esquemas para Toma de Decisión Artificial en Escenarios Secuenciales». Compara arquitecturas de aprendizaje
por refuerzo y reglas de ensamble en una asignación secuencial de recursos limitados (pandemia simulada, derivada
del PES del BCI-NE Lab, Universidad de Essex).

La referencia completa del proyecto (paquetes, layout, dependencias, estilo de código) está en
[.github/copilot-instructions.md](../.github/copilot-instructions.md); este archivo no la repite, sólo agrega lo
específico de Claude Code. Si ambos se contradicen, avisa al usuario.

Decisiones del autor y estado de trabajo que antes vivían en la memoria local de Claude Code (flujo de correcciones,
tope de páginas, enmascarado −1e9, semillas): @memoria.md

## Mapa rápido

| Ruta | Contenido | Escritura por defecto |
|------|-----------|-----------------------|
| `h1/tabular/`, `h1/ml/` | 7 modelos individuales entrenados (fuente de verdad experimental) | 🔒 sólo lectura |
| `h1/ens/` | 6 ensambles | ✅ (agente `python-h1`) |
| `h1/general/` | Benchmark: 21 escenarios + 5 réplicas *held-out*; `results/` | ✅ (agente `python-h1`) |
| `h2/` | Línea suspendida; no citarla como trabajo realizado | 🔒 (salvo agente `python-h2`) |
| `writings/` | Tesis LaTeX (`00_Main/Main.tex`, `01_Chapters/`, `02_Images/`, `audit/`) | ✅ (agente `thesis-writer`) |
| `writings/auxiliar/claude_context/` | Contexto curado: `instrucciones_sistema.md`, `mpes_resultados.{md,json}` | regenerar con script |
| `dict/` | Pre-dictámenes del jurado (PDF) y [correcciones.md](../dict/correcciones.md) | ✅ |
| `utils/` | `config/` (pyright, pylint, markdownlint, requirements) y `scripts/` | ✅ |

## Reglas de la tesis (`writings/`)

- **Fuente de verdad numérica**: `writings/auxiliar/claude_context/mpes_resultados.json` → `mpes_resultados.md` →
  `.tex`. No inventes números, métodos, figuras ni citas. Todo dato nuevo debe salir de `h1/general/results/` o de un
  script de `writings/auxiliar/scripts/`. Si falta, dilo («verificación pendiente») y no lo escribas en el LaTeX.
- **Convenciones LaTeX y de redacción**: siguen `writings/auxiliar/claude_context/instrucciones_sistema.md` §5–§7
  (acentos escapados `m\'as`, decimales `$0{,}927$`, `$36\%$`, anglicismos en `\emph{}`, terminología fija
  texto ↔ paquete, ecuación + párrafo «Donde …», sin palabras inglesas fuera de `\emph{}`/`\texttt{}`).
- **Tope de 40 páginas para el cuerpo** (Introducción → Conclusiones; preliminares, referencias y apéndice no
  cuentan). El número vigente de páginas del cuerpo está en `dict/correcciones.md`. Toda corrección que agregue texto debe informar
  el impacto en páginas después de compilar.
- No cambies el preámbulo de `Main.tex` ni el orden de capítulos sin pedido explícito.
- `Acknowledgement.tex` está fuera de `Main.tex` a propósito: el aviso de «.tex huérfano» de `audit.py` es esperado.
- Tras editar `.tex`/`.bib`: skill `/thesis-audit` (compila con pdflatex/bibtex vía `audit.py`; TinyTeX está en el PATH).
- Numeración de referencias en el PDF (unsrtnat, por orden de cita): los dictámenes usan la del PDF original
  ([32] `Nijjar2025`, [48] `Henderson2018`, [51] `mPES2026`). La numeración vigente está en `dict/correcciones.md`
  y cambia con cada cita nueva.

## Reglas de código

- Windows, Python 3.12, entorno `win_mpes_env`. Antes de ejecutar paquetes:
  `win_mpes_env\Scripts\Activate.ps1; $env:PYTHONIOENCODING="utf-8"; $env:TF_ENABLE_ONEDNN_OPTS="0"`.
  Los `python -m ...` se corren desde `h1/`; pyright/pylint desde la raíz.
- Estilo: 120 columnas, `import numpy` (nunca `np`), imports explícitos salvo en `__init__.py`, docstrings NumPy en
  inglés, `os.path.join`. Cada paquete es autocontenido: la duplicación de `src/` es intencional.
- Calidad: pyright `0 errors, 0 warnings, 0 informations` y pylint `10.00/10` (skill `/lint-and-typecheck`).
  Nunca `# type: ignore` ni `# pylint: disable=` para reglas exigidas, ni tocar `.pylintrc`/`pyrightconfig.json`.
- Los `.md` de documentación van en español con KaTeX (`$...$`); el `README.md` raíz, en inglés.
- El enmascarado con `-1e9` de la confianza por entropía en los modelos individuales es intencional.
- No hagas commit ni push sin pedido explícito del usuario.

## Agentes y skills de Claude Code

Equivalentes a los de Copilot en `.github/agents/` y `.github/prompts/`:

| Copilot | Claude Code | Uso |
|---------|-------------|-----|
| `thesis-writer.agent.md` | agente `thesis-writer` | Redactar/revisar la tesis; escribe sólo en `writings/` |
| `python-program-h11.md` | agente `python-h1` | Código en `h1/ens`, `h1/general`, `utils`, `.github` |
| `python-program-h22.md` | agente `python-h2` | Código en `h2/` (línea suspendida) |
| `md-actualize.md` | agente `md-actualize` | Sólo archivos `.md` |
| `github-repo.agent.md` | agente `github-repo` | git, ramas, PRs, `.gitignore` |
| — | agente `thesis-judge` | Revisión independiente tipo jurado, sólo lectura |
| `thesis-audit.prompt.md` | skill `/thesis-audit` | Auditoría + compilación de la tesis |
| `lint-and-typecheck.prompt.md` | skill `/lint-and-typecheck` | Bucle pyright + pylint |
| `audit-project.prompt.md` | skill `/audit-project` | Auditoría código ↔ docs |
| `update-pkg-docs.prompt.md` | skill `/update-pkg-docs` | Docs de un paquete |
| `create-pkg.prompt.md` | skill `/create-pkg` | Nuevo paquete `pes_*` |
| `python3-script.prompt.md` | skill `/python3-script` | Script nuevo con las convenciones |
| — | skill `/jury-corrections` | Aplicar una observación del jurado de `dict/correcciones.md` |

Los prompts de `.github/prompts/` siguen siendo la fuente canónica de cada procedimiento: las skills los leen y
traducen las herramientas de Copilot así:

| Copilot | Claude Code |
|---------|-------------|
| `read_file` | Read |
| `grep_search` / `file_search` | Grep / Glob |
| `replace_string_in_file`, `multi_replace_string_in_file` | Edit (varias llamadas en paralelo) |
| `create_file` | Write |
| `run_in_terminal` | PowerShell (o Bash para POSIX) |
| `get_errors` | `mcp__claude-vscode__getDiagnostics`, o pyright/pylint/pymarkdown en terminal |
| `runSubagent` | Agent |
