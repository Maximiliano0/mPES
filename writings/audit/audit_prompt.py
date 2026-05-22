"""Genera un prompt autocontenido para auditar la tesis con un LLM.

Empaqueta los 6 criterios de auditoría junto con el contenido íntegro de
todos los capítulos `.tex` incluidos por `Main.tex` y la bibliografía
`References.bib`. La salida se escribe en `writings/audit/PROMPT.md` y
puede pegarse directamente en ChatGPT, Claude, Copilot, etc.

Uso
---
    python audit/audit_prompt.py
"""

##########################
##  Imports externos    ##
##########################
import re
import sys
from pathlib import Path


##########################
##  Configuración       ##
##########################
WRITINGS_DIR = Path(__file__).resolve().parent.parent
MAIN_TEX = WRITINGS_DIR / "00_Main" / "Main.tex"
CHAPTERS_DIR = WRITINGS_DIR / "01_Chapters"
BIB_FILE = WRITINGS_DIR / "00_Main" / "References.bib"
OUT_FILE = WRITINGS_DIR / "audit" / "PROMPT.md"

CRITERIA = """\
Eres un auditor académico experto en LaTeX, redacción técnica en español
y normas APA. Vas a auditar una tesis de maestría completa. Devuelve un
**informe en Markdown** con una sección por criterio. Para cada hallazgo
indica archivo, fragmento textual y propuesta concreta de corrección.

## Criterios obligatorios

1. **Sintaxis y compilación LaTeX.** Detecta entornos mal cerrados,
   macros desconocidas, argumentos faltantes, comillas tipográficas
   incorrectas, espacios irregulares antes de signos, uso indebido de
   `$...$` vs. `\\(...\\)`, y cualquier construcción que `pdflatex`
   probablemente rechace o advierta. Señala archivos `.tex` que no
   estén incluidos por `Main.tex` (huérfanos) y bloques duplicados.

2. **Figuras, tablas y numeración.** Verifica que cada entorno
   `figure` / `table` tenga `\\caption{}` y `\\label{...}` con prefijo
   coherente (`fig:`, `tab:`). Confirma que toda `\\ref`, `\\autoref` y
   `\\eqref` apunte a un `\\label` existente. Señala numeración
   inconsistente o tablas sin `booktabs`.

3. **Citas y bibliografía en formato APA.** Comprueba que cada `\\cite*`
   tenga una entrada correspondiente en `References.bib`; detecta
   duplicados de clave o títulos casi idénticos, entradas no citadas,
   campos faltantes (autor, año, título, editor/journal) y
   inconsistencias de mayúsculas. Sugiere la cita APA correcta cuando
   detectes errores.

4. **Coherencia y cohesión.** Evalúa el hilo argumental entre
   capítulos: progresión lógica, definiciones antes del uso,
   redundancias, párrafos sin verbo principal, conectores ausentes y
   referencias cruzadas rotas conceptualmente (p.ej. "ver Cap. 5" sin
   contenido relacionado).

5. **Idioma único: español.** El texto debe estar enteramente en
   español. Marca cualquier oración, frase o palabra en inglés que no
   sea nombre propio, identificador de software, término técnico
   consolidado (p. ej. *Q-Learning*, *Transformer*) o cita textual.
   Sugiere la traducción adecuada.

6. **Cobertura del proyecto.** El documento debe describir todos los
   paquetes del repositorio `mPES`. Verifica que cada uno aparezca y
   esté correctamente caracterizado:

   | Familia | Paquete | Algoritmo |
   |---------|---------|-----------|
   | tabular | pes_base | Q-Learning tabular (línea base) |
   | tabular | pes_ql   | Q-Learning + Optuna/TPE |
   | tabular | pes_dql  | Double Q-Learning + PBRS + warm-up |
   | ml      | pes_dqn  | Deep Q-Network |
   | ml      | pes_rdqn | Recurrent DQN (LSTM) — futuro |
   | ml      | pes_a2c  | Advantage Actor-Critic |
   | ml      | pes_trf  | Causal Transformer DQN |
   | ml      | pes_ens  | Ensemble (soft voting) — futuro |

   Señala paquetes ausentes o descripciones incorrectas.

## Formato de respuesta

- Una sección `## N. <Criterio>` por cada uno de los 6.
- Dentro de cada sección, una lista de hallazgos `- **archivo.tex** —
  descripción breve. _Sugerencia:_ ...`.
- Si un criterio no tiene hallazgos, escribe `Sin observaciones.`.
- Al final, sección `## Resumen ejecutivo` con un veredicto global
  (APROBADO / OBSERVADO / RECHAZADO) y los 3 problemas más críticos.
"""


##########################
##  Utilidades          ##
##########################
def read(path: Path) -> str:
    """Lee un archivo de texto en UTF-8 (fallback latin-1)."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def included_chapters() -> list[Path]:
    """Devuelve los `.tex` incluidos por `Main.tex`, en orden."""
    main_tex = read(MAIN_TEX)
    main_tex = re.sub(r"(?<!\\)%.*", "", main_tex)  # quita comentarios
    pat = re.compile(r"\\(?:subfile|include|input)\{([^}]+)\}")
    out: list[Path] = []
    for m in pat.finditer(main_tex):
        rel = m.group(1).strip()
        if not rel.endswith(".tex"):
            rel += ".tex"
        p = (WRITINGS_DIR / rel).resolve()
        if p.exists():
            out.append(p)
    return out


##########################
##  Generación prompt   ##
##########################
def build_prompt() -> str:
    """Construye el prompt completo en Markdown."""
    parts: list[str] = []
    parts.append("# Auditoría de tesis — Prompt para LLM\n")
    parts.append("> Generado automáticamente por "
                 "`writings/audit/audit_prompt.py`.\n")
    parts.append(CRITERIA)
    parts.append("\n---\n\n# Material a auditar\n")

    # Main.tex
    parts.append("\n## `00_Main/Main.tex`\n\n```latex\n")
    parts.append(read(MAIN_TEX))
    parts.append("\n```\n")

    # Capítulos
    for tex in included_chapters():
        rel = tex.relative_to(WRITINGS_DIR).as_posix()
        parts.append(f"\n## `{rel}`\n\n```latex\n")
        parts.append(read(tex))
        parts.append("\n```\n")

    # Bibliografía
    if BIB_FILE.exists():
        parts.append("\n## `00_Main/References.bib`\n\n```bibtex\n")
        parts.append(read(BIB_FILE))
        parts.append("\n```\n")

    return "".join(parts)


def main() -> int:
    """Punto de entrada CLI."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover  # pylint: disable=broad-except
        pass
    text = build_prompt()
    OUT_FILE.parent.mkdir(exist_ok=True)
    OUT_FILE.write_text(text, encoding="utf-8")
    size_kb = len(text.encode("utf-8")) // 1024
    n_chap = len(included_chapters())
    print(f"Prompt generado: {OUT_FILE}")
    print(f"  Capítulos incluidos: {n_chap}")
    print(f"  Tamaño: {size_kb} KB ({len(text):,} caracteres)")
    print("Pégalo en ChatGPT, Claude o Copilot para auditar la tesis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
