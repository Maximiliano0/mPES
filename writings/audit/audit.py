"""Auditoría programática de la tesis LaTeX.

Ejecuta 6 criterios programáticos y, salvo --no-tex, compila el
documento con pdflatex/bibtex. Los artefactos de compilación
(aux, log, bbl, pdf …) se guardan en ``writings/out/``. El PDF se
renombra con el título de la tesis. El informe se escribe en
``writings/audit/AUDIT.md`` y también se imprime en stdout.

Uso
---
    python audit/audit.py             # audita + compila
    python audit/audit.py --no-tex    # solo audita (sin pdflatex)
"""

##########################
##  Imports externos    ##
##########################
import argparse
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

##########################
##  Configuración       ##
##########################
WRITINGS_DIR  = Path(__file__).resolve().parent.parent
MAIN_DIR      = WRITINGS_DIR / "00_Main"
MAIN_TEX      = MAIN_DIR / "Main.tex"
CHAPTERS_DIR  = WRITINGS_DIR / "01_Chapters"
BIB_FILE      = MAIN_DIR / "References.bib"
OUT_DIR       = WRITINGS_DIR / "out"
AUDIT_FILE    = WRITINGS_DIR / "audit" / "AUDIT.md"
FRONTPAGE_TEX = CHAPTERS_DIR / "000NHH-Frontpage.tex"

# Slug de respaldo si no se puede extraer el título del .tex
_FALLBACK_SLUG = "mPES-Inteligencia-Artificial-para-la-Gestion-de-Crisis-Pandemicas"

##########################
##  Helpers             ##
##########################

def read(path: Path) -> str:
    """Lee un archivo de texto (UTF-8 con fallback latin-1)."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def strip_comments(tex: str) -> str:
    """Elimina comentarios LaTeX (%) de una cadena."""
    return re.sub(r"(?<!\\)%.*", "", tex)


def included_chapters() -> list[Path]:
    """Devuelve los .tex incluidos por Main.tex (\\subfile / \\include / \\input)."""
    main_tex = strip_comments(read(MAIN_TEX))
    pat = re.compile(r"\\(?:subfile|include|input)\{([^}]+)\}")
    out: list[Path] = []
    for m in pat.finditer(main_tex):
        rel = m.group(1).strip()
        if not rel.endswith(".tex"):
            rel += ".tex"
        # Los paths en Main.tex son relativos a MAIN_DIR (00_Main/),
        # no a WRITINGS_DIR. Ej: ../01_Chapters/Foo → writings/01_Chapters/Foo
        p = (MAIN_DIR / rel).resolve()
        if p.exists():
            out.append(p)
    return out


def all_tex_content() -> str:
    """Concatena Main.tex y todos los capítulos incluidos."""
    parts = [read(MAIN_TEX)]
    for tex in included_chapters():
        parts.append(read(tex))
    return "\n".join(parts)

##########################
##  Criterios           ##
##########################

def audit_orphans() -> list[str]:
    """Archivos .tex en 01_Chapters/ que Main.tex no incluye."""
    included = set(included_chapters())
    orphans  = sorted(set(CHAPTERS_DIR.glob("*.tex")) - included)
    return [f"- `{p.relative_to(WRITINGS_DIR).as_posix()}`" for p in orphans]


def audit_figures_tables() -> dict:
    """Cuenta etiquetas de figuras/tablas y detecta referencias rotas."""
    content    = all_tex_content()
    fig_labels = re.findall(r"\\label\{fig:[^}]+\}", content)
    tab_labels = re.findall(r"\\label\{tab:[^}]+\}", content)
    all_labels = set(re.findall(r"\\label\{([^}]+)\}", content))
    refs       = re.findall(r"\\(?:ref|autoref|eqref)\{([^}]+)\}", content)
    broken     = [r for r in refs if r not in all_labels]
    return {
        "fig_count":  len(fig_labels),
        "tab_count":  len(tab_labels),
        "ref_count":  len(refs),
        "broken_refs": broken,
    }


def audit_bib() -> dict:
    """Verifica coherencia entre \\cite y entradas de References.bib."""
    bib      = read(BIB_FILE) if BIB_FILE.exists() else ""
    bib_keys = set(re.findall(r"@\w+\{([^,\s]+)\s*,", bib))
    content  = all_tex_content()
    cited_raw = re.findall(r"\\cite[a-z*]*\{([^}]+)\}", content)
    cited: set[str] = set()
    for raw in cited_raw:
        for key in raw.split(","):
            cited.add(key.strip())
    return {
        "missing": sorted(cited - bib_keys),
        "uncited": sorted(bib_keys - cited),
    }


def audit_language() -> list[str]:
    """Detecta palabras en inglés que no sean términos técnicos consolidados."""
    ENGLISH_STOPWORDS = {
        "the", "this", "that", "with", "from", "which", "where", "when",
        "what", "how", "have", "has", "been", "will", "can", "may", "must",
        "should", "would", "could", "also", "thus", "however", "therefore",
        "because", "between", "within", "without", "through", "during",
        "before", "after", "above", "below", "more", "less", "most", "least",
        "each", "both", "either", "neither", "such", "these", "those",
        "their", "there", "here", "then", "than", "other", "another",
        "although", "while", "since", "until", "unless", "whereas",
    }
    ALLOWED = {
        "q-learning", "double", "deep", "transformer", "lstm", "softmax",
        "benchmark", "reward", "ensemble", "feedback", "pipeline",
        "reinforcement", "learning", "network", "replay", "target",
        "agent", "policy", "episode", "trial", "batch", "entropy",
        "q-value", "action", "state", "model", "loss", "output",
    }
    content = all_tex_content()
    plain   = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", content)
    plain   = re.sub(r"\\[a-zA-Z]+", " ", plain)
    plain   = re.sub(r"[^a-zA-Z\s]", " ", plain)
    words   = set(plain.lower().split())
    hits    = sorted((words & ENGLISH_STOPWORDS) - ALLOWED)
    return hits


def audit_packages() -> list[str]:
    """Verifica que cada paquete mPES esté mencionado."""
    PACKAGES: dict[str, tuple[str, str]] = {
        "pes_base":  ("tabular", "Q-Learning tabular"),
        "pes_ql":    ("tabular", "Q-Learning + Optuna/TPE"),
        "pes_dql":   ("tabular", "Double Q-Learning + PBRS"),
        "pes_dqn":   ("ml",      "Deep Q-Network"),
        "pes_rdqn":  ("ml",      "Recurrent DQN (LSTM)"),
        "pes_a2c":   ("ml",      "Advantage Actor-Critic"),
        "pes_trf":   ("ml",      "Causal Transformer DQN"),
        "pes_ens":   ("ml",      "Ensemble (soft voting)"),
    }
    content = all_tex_content().lower()
    rows: list[str] = []
    for pkg, (family, _algo) in PACKAGES.items():
        mentioned = pkg in content
        doc_dir   = WRITINGS_DIR.parent / family / pkg / "doc"
        doc_count = len(list(doc_dir.glob("*"))) if doc_dir.exists() else 0
        rows.append(f"| `{pkg}` | {'✅' if mentioned else '❌'} | {doc_count} archivos |")
    return rows


def audit_key_concepts() -> dict[str, bool]:
    """Verifica presencia de conceptos clave en el cuerpo del documento."""
    CONCEPTS = {
        "MDP":                             r"MDP",
        "Q-Learning":                      r"[Qq]-[Ll]earning",
        "Double Q-Learning":               r"[Dd]ouble\s+[Qq]-[Ll]earning",
        "PBRS / potential-based reward shaping": r"PBRS|potential.based",
        "Experience Replay":               r"[Ee]xperience\s+[Rr]eplay",
        "Target Network":                  r"[Tt]arget\s+[Nn]etwork",
        "Optuna / TPE":                    r"[Oo]ptuna|TPE",
        "Cohen d / Welch / KL":            r"[Cc]ohen|[Ww]elch|KL",
        "Shannon entropy / UQ":            r"[Ss]hannon|entrop",
        "Atención causal / Transformer":   r"[Tt]ransformer|atencion\s+causal",
    }
    content = all_tex_content()
    return {name: bool(re.search(pat, content)) for name, pat in CONCEPTS.items()}

##########################
##  Compilación LaTeX   ##
##########################

def _slugify(text: str) -> str:
    """Convierte texto en slug ASCII apto para nombre de archivo."""
    nfkd  = unicodedata.normalize("NFKD", text)
    ascii_ = nfkd.encode("ascii", "ignore").decode("ascii")
    slug  = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_)
    return slug.strip("-")


def _pdf_slug() -> str:
    """Extrae el título de la portada y devuelve su slug."""
    if not FRONTPAGE_TEX.exists():
        return _FALLBACK_SLUG
    raw = read(FRONTPAGE_TEX)
    # Busca el contenido de \textbf{...} dentro del bloque \Huge
    m = re.search(r"\\Huge\s*\\textbf\{(.*?)\}", raw, re.DOTALL)
    if not m:
        return _FALLBACK_SLUG
    title = m.group(1)
    title = re.sub(r"\\\\", " ", title)                     # \\ → espacio
    title = re.sub(r"\\'([aeiouAEIOUn])", r"\1", title)     # \'a → a
    title = re.sub(r'\\["\^`~]([aeiouAEIOU])', r"\1", title)
    title = re.sub(r"\{|\}", "", title)
    return _slugify(title.strip()) or _FALLBACK_SLUG


def compile_latex() -> dict:
    """Compila Main.tex con pdflatex/bibtex (4 pasos) y mueve los artefactos a writings/out/.

    La compilación se hace en MAIN_DIR para evitar restricciones de
    ``openout_any`` con rutas relativas (bibtex no puede escribir fuera del CWD
    en distribuciones modernas). Al finalizar, todo se mueve a OUT_DIR y el
    PDF se renombra con el título de la tesis.

    Returns
    -------
    dict
        returncode, pages, pdf_name, overfull, errors.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def run(cmd: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            cwd=str(MAIN_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    tex_cmd = [
        "pdflatex",
        "-cnf-line", "openout_any=a",
        "-interaction", "nonstopmode",
        "-file-line-error",
        "Main.tex",
    ]

    run(tex_cmd)             # paso 1 → genera Main.aux en MAIN_DIR
    run(["bibtex", "Main"])  # bibtex  → genera Main.bbl en MAIN_DIR

    # Repetir pdflatex hasta que las referencias se estabilicen
    # (mensaje "Rerun to get cross-references right" en el log).
    # Máximo 4 pasadas adicionales para evitar bucles patológicos.
    log_file = MAIN_DIR / "Main.log"
    final = run(tex_cmd)
    for _ in range(4):
        log_txt = log_file.read_text(encoding="utf-8", errors="replace") \
            if log_file.exists() else ""
        if "Rerun to get" not in log_txt and "Label(s) may have changed" not in log_txt:
            break
        final = run(tex_cmd)

    # Leer el log ANTES de moverlo a OUT_DIR
    log = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else ""

    overfull = re.findall(r"Overfull \\hbox[^\n]+", log)
    errors   = re.findall(r"^!.+", log, re.MULTILINE)
    pages    = None
    m_out    = re.search(r"Output written on[^\n]+", log)
    if m_out:
        m_pg = re.search(r"\((\d+) pages", m_out.group())
        pages = int(m_pg.group(1)) if m_pg else None

    # Mover artefactos de compilación a OUT_DIR
    _ARTIFACT_PATTERNS = [
        "*.aux", "*.log", "*.bbl", "*.blg",
        "*.toc", "*.lof", "*.lot", "*.out", "*.synctex.gz",
    ]
    for pattern in _ARTIFACT_PATTERNS:
        for f in MAIN_DIR.glob(pattern):
            dest = OUT_DIR / f.name
            if dest.exists():
                dest.unlink()
            shutil.move(str(f), str(dest))

    # Renombrar y mover PDF a OUT_DIR
    src_pdf  = MAIN_DIR / "Main.pdf"
    pdf_name = None
    if src_pdf.exists():
        slug     = _pdf_slug()
        dest_pdf = OUT_DIR / f"{slug}.pdf"
        if dest_pdf.exists():
            dest_pdf.unlink()
        shutil.move(str(src_pdf), str(dest_pdf))
        pdf_name = dest_pdf.name

    return {
        "returncode": final.returncode,
        "pages":      pages,
        "pdf_name":   pdf_name,
        "overfull":   overfull,
        "errors":     errors,
    }

##########################
##  Informe             ##
##########################

def build_report(tex_result: dict | None) -> str:
    """Construye el informe de auditoría en Markdown."""
    L: list[str] = []

    L += [f"# Auditoría LaTeX — `writings/`\n",
          f"Raíz: `{WRITINGS_DIR}`\n"]

    # 1 ── Sintaxis y compilación
    L.append("## 1. Sintaxis y compilación\n")
    orphans = audit_orphans()
    if orphans:
        L.append("**Archivos `.tex` huérfanos en `01_Chapters/` "
                 "(no incluidos por `Main.tex`):**\n")
        L += orphans
        L.append("")
    if tex_result is None:
        L.append("- Compilación pdflatex omitida (--no-tex).\n")
    else:
        if tex_result["errors"]:
            L.append("**Errores pdflatex:**\n")
            L += [f"- `{e}`" for e in tex_result["errors"][:10]]
            L.append("")
        else:
            pages = tex_result["pages"]
            pdf   = tex_result["pdf_name"]
            L.append(f"- ✅ Compilación exitosa — **{pages} páginas**.")
            if pdf:
                L.append(f"- 📄 PDF: `out/{pdf}`")
            L.append("")
        if tex_result["overfull"]:
            L.append("**Overfull \\hbox:**\n")
            L += [f"- `{o}`" for o in tex_result["overfull"][:10]]
            L.append("")
        else:
            L.append("- ✅ Sin Overfull \\hbox.\n")

    # 2 ── Figuras y tablas
    L.append("## 2. Figuras, tablas y numeración\n")
    ft = audit_figures_tables()
    L.append(f"- Figuras con label: **{ft['fig_count']}**.")
    L.append(f"- Tablas con label: **{ft['tab_count']}**.")
    L.append(f"- Referencias internas (`\\ref`/`\\autoref`/`\\eqref`): "
             f"**{ft['ref_count']}**.\n")
    if ft["broken_refs"]:
        L.append("**Referencias rotas:**\n")
        L += [f"- `\\ref{{{r}}}`" for r in ft["broken_refs"]]
        L.append("")
    else:
        L.append("- ✅ Sin referencias rotas.\n")

    # 3 ── Citas
    L.append("## 3. Citas y bibliografía (APA)\n")
    bib = audit_bib()
    if bib["missing"]:
        L.append("**Claves citadas sin entrada en `.bib`:**\n")
        L += [f"- `{k}`" for k in bib["missing"]]
        L.append("")
    else:
        L.append("- ✅ Todas las claves citadas existen en `References.bib`.\n")
    if bib["uncited"]:
        L.append("**Entradas `.bib` no citadas:**\n")
        L += [f"- `{k}`" for k in bib["uncited"]]
        L.append("")

    # 4 ── Coherencia
    L.append("## 4. Coherencia y cohesión\n")
    L.append("**Orden de capítulos en `Main.tex`:**\n")
    L += [f"- `{t.relative_to(WRITINGS_DIR).as_posix()}`"
          for t in included_chapters()]
    L.append("")

    # 5 ── Idioma
    L.append("## 5. Idioma único (español)\n")
    hits = audit_language()
    if hits:
        L.append("**Palabras en inglés detectadas:**\n")
        L.append(", ".join(f"`{w}`" for w in hits))
        L.append("")
    else:
        L.append("- ✅ No se detectaron palabras inglesas frecuentes.\n")

    # 6 ── Cobertura
    L.append("## 6. Cobertura de los `doc/` del proyecto\n")
    L.append("| Paquete | Mencionado | `doc/` presente |")
    L.append("|---------|------------|-----------------|")
    L += audit_packages()
    L.append("")
    L.append("**Conceptos clave en `writings/`:**\n")
    L += [f"- {'✅' if found else '❌'} {concept}"
          for concept, found in audit_key_concepts().items()]
    L.append("")

    return "\n".join(L)

##########################
##  Punto de entrada    ##
##########################

def main() -> int:
    """CLI: python audit/audit.py [--no-tex]."""
    parser = argparse.ArgumentParser(
        description="Auditoría programática de la tesis LaTeX.")
    parser.add_argument("--no-tex", action="store_true",
                        help="Omite la compilación pdflatex.")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    tex_result: dict | None = None
    if not args.no_tex:
        print("Compilando LaTeX …", flush=True)
        tex_result = compile_latex()
        if tex_result["errors"]:
            print(f"  ❌ pdflatex reportó errores.")
        else:
            print(f"  ✅ {tex_result['pages']} páginas  →  out/{tex_result['pdf_name']}")

    report = build_report(tex_result)
    AUDIT_FILE.parent.mkdir(exist_ok=True)
    AUDIT_FILE.write_text(report, encoding="utf-8")
    print(f"\nAUDIT.md → {AUDIT_FILE}\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
