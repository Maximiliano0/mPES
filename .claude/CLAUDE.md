# CLAUDE.md — mPES (Multiple Pandemic Experiment Suite)

mPES compara arquitecturas de aprendizaje por refuerzo y reglas de ensamble en una asignación secuencial de recursos
limitados (pandemia simulada, derivada del PES del BCI-NE Lab, Universidad de Essex).

La referencia completa del proyecto (paquetes, layout, dependencias, estilo de código) está en
[.github/copilot-instructions.md](../.github/copilot-instructions.md); este archivo no la repite, sólo agrega lo
específico de Claude Code. Si ambos se contradicen, avisa al usuario.

## Mapa rápido

| Ruta | Contenido | Escritura por defecto |
|------|-----------|-----------------------|
| `h1/tabular/`, `h1/ml/` | 7 modelos individuales entrenados (fuente de verdad experimental) | 🔒 sólo lectura |
| `h1/ens/` | 6 ensambles | ✅ (agente `python-h1`) |
| `h1/general/` | Benchmark: 21 escenarios + 5 réplicas *held-out*, variabilidad entre semillas; `results/` | ✅ (agente `python-h1`) |
| `h2/` | Línea suspendida; no citarla como trabajo realizado | 🔒 (salvo agente `python-h2`) |
| `writings/out/` | Informe compilado (PDF) | 🔒 |
| `utils/` | `config/` (pyright, pylint, markdownlint, requirements) y `scripts/` | ✅ |

## Reglas de código

- Windows, Python 3.12, entorno `win_mpes_env`. Antes de ejecutar paquetes:
  `win_mpes_env\Scripts\Activate.ps1; $env:PYTHONIOENCODING="utf-8"; $env:TF_ENABLE_ONEDNN_OPTS="0"`.
  Los `python -m ...` se corren desde `h1/`; pyright/pylint desde la raíz.
- Estilo: 120 columnas, `import numpy` (nunca `np`), imports explícitos salvo en `__init__.py`, docstrings NumPy en
  inglés, `os.path.join`. Cada paquete es autocontenido: la duplicación de `src/` es intencional.
- Calidad: pyright `0 errors, 0 warnings, 0 informations` y pylint `10.00/10` (skill `/lint-and-typecheck`).
  Nunca `# type: ignore` ni `# pylint: disable=` para reglas exigidas, ni tocar `.pylintrc`/`pyrightconfig.json`.
- Los `.md` de documentación van en español con KaTeX (`$...$`); el `README.md` raíz, en inglés.
- No inventes números: todo resultado sale de `h1/general/results/` o de un script del repositorio.
- El enmascarado con `-1e9` de la confianza por entropía en los modelos individuales es intencional: con menos de
  10 recursos restantes las acciones factibles parecen casi uniformes, y esa es la definición de confianza elegida.
- Los modelos por semilla de `h1/general/work/seeds/` están en Git LFS (`git lfs pull`).
- No hagas commit ni push sin pedido explícito del usuario.

## Agentes y skills de Claude Code

Equivalentes a los de Copilot en `.github/agents/` y `.github/prompts/`:

| Copilot | Claude Code | Uso |
|---------|-------------|-----|
| `python-program-h11.md` | agente `python-h1` | Código en `h1/ens`, `h1/general`, `utils`, `.github` |
| `python-program-h22.md` | agente `python-h2` | Código en `h2/` (línea suspendida) |
| `md-actualize.md` | agente `md-actualize` | Sólo archivos `.md` |
| `github-repo.agent.md` | agente `github-repo` | git, ramas, PRs, `.gitattributes` |
| `lint-and-typecheck.prompt.md` | skill `/lint-and-typecheck` | Bucle pyright + pylint |
| `audit-project.prompt.md` | skill `/audit-project` | Auditoría código ↔ docs |
| `update-pkg-docs.prompt.md` | skill `/update-pkg-docs` | Docs de un paquete |
| `create-pkg.prompt.md` | skill `/create-pkg` | Nuevo paquete `pes_*` |
| `python3-script.prompt.md` | skill `/python3-script` | Script nuevo con las convenciones |

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
