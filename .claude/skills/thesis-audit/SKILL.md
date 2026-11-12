---
name: thesis-audit
description: Audita y compila la tesis LaTeX de writings/ con writings/audit/audit.py (pdflatex + bibtex), lee AUDIT.md y corrige iterativamente hasta que todo quede en verde. Úsala después de editar cualquier .tex o .bib, o cuando el usuario pida auditar o compilar la tesis.
argument-hint: "[full|quick]"
---

# Thesis Audit (bucle de corrección)

Modo: `$ARGUMENTS` (vacío o `full` = auditar y compilar; `quick` = `--no-tex`, sin compilar).

1. Lee `.github/prompts/thesis-audit.prompt.md` completo: es la fuente canónica del procedimiento (restricciones,
   revisión como juez evaluador, bucle y criterio de éxito). Aplícalo traduciendo las herramientas de Copilot con la
   tabla de `CLAUDE.md`.
2. Ejecución en este equipo (PowerShell, desde la raíz del repo):

   ```powershell
   win_mpes_env\Scripts\Activate.ps1; $env:PYTHONIOENCODING = "utf-8"; Set-Location writings
   python audit\audit.py            # full
   python audit\audit.py --no-tex   # quick
   ```

   TinyTeX (`pdflatex`, `bibtex`, `latexmk`) ya está en el PATH. Si el entorno virtual no existe, usa el Python del
   sistema: `audit.py` sólo usa la biblioteca estándar.
3. Hallazgos esperados que no se corrigen: el aviso de `Acknowledgement.tex` huérfano (excluido a propósito).
4. Tras compilar en modo full, mide el cuerpo de la tesis (tope: 40 páginas de Introducción a Conclusiones):

   ```bash
   pdftotext -layout writings/out/*.pdf - | grep -n "^7\. Conclusiones\|^Referencias Bibliogr" | head -2
   ```

   El índice da la página de inicio de Referencias; el cuerpo termina en la página anterior.
5. Si la revisión crítica merece una mirada independiente, delega en el agente `thesis-judge` pasándole los archivos,
   no tus conclusiones.

## Reporte

Estado de cada criterio de `AUDIT.md` (no sólo «se ejecutó»), archivos de `writings/` modificados y por qué, páginas
totales y del cuerpo, ruta del PDF y hallazgos que requieran otro agente (p. ej. regenerar una figura en `h1/`).
