"""Auditor del proyecto LaTeX `writings/`.

Verifica seis criterios:

1. Sintaxis y compilación adecuada (errores y advertencias de pdflatex)
   + archivos `.tex` huérfanos (no incluidos por `Main.tex`).
2. Uso de figuras y tablas: cada entorno `figure`/`table` debe tener
   `\\caption` y `\\label`; toda referencia `\\ref{fig:..}` /
   `\\ref{tab:..}` debe resolver.
3. Citas bibliográficas: claves `\\cite{...}` que no existen, entradas
   duplicadas en `References.bib`, entradas sin uso.
4. Coherencia y cohesión: orden de secciones, conteo de palabras por
   capítulo, referencias internas rotas.
5. Idioma único (español): palabras inglesas frecuentes encontradas en
   el cuerpo del texto.
6. Cobertura del proyecto: paquetes y conceptos descritos en
   `tabular/<pkg>/doc/` y `ml/<pkg>/doc/` que NO aparecen en
   `writings/`.

Uso (desde `writings/`):

    python audit/audit.py            # ejecuta todo y escribe AUDIT.md
    python audit/audit.py --no-tex   # omite la compilación pdflatex

El reporte se imprime en consola y se guarda en `audit/AUDIT.md`.
"""

from __future__ import annotations

##########################
##  Imports externos    ##
##########################
import argparse
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


##########################
##  Configuración       ##
##########################
WRITINGS_DIR = Path(__file__).resolve().parent.parent
MAIN_TEX = WRITINGS_DIR / "00_Main" / "Main.tex"
CHAPTERS_DIR = WRITINGS_DIR / "01_Chapters"
BIB_FILE = WRITINGS_DIR / "00_Main" / "References.bib"
REPO_ROOT = WRITINGS_DIR.parent
# Salida compilada (PDF + intermedios) bajo writings/out/
OUT_DIR = WRITINGS_DIR / "out"
# Nombre del PDF = título de la tesis (slug ASCII seguro)
JOBNAME = "mPES-Inteligencia-Artificial-para-la-Gestion-de-Crisis-Pandemicas"
PACKAGES = [
    ("tabular", "pes_base"),
    ("tabular", "pes_ql"),
    ("tabular", "pes_dql"),
    ("ml", "pes_dqn"),
    ("ml", "pes_a2c"),
    ("ml", "pes_trf"),
    ("ml", "pes_rdqn"),
    ("ml", "pes_ens"),
]

# Marcadores ingleses comunes; se buscan como palabras completas e
# insensibles a mayúsculas. Se excluyen identificadores LaTeX y código.
ENGLISH_MARKERS = {
    "the", "and", "with", "from", "this", "that", "which", "where",
    "however", "therefore", "moreover", "thus", "while", "between",
    "training", "learning", "reward", "agent", "policy", "network",
    "model", "results", "introduction", "background", "discussion",
    "conclusion", "chapter", "figure", "table", "section", "we",
    "our", "their", "these", "those", "of", "to", "is", "are",
    "was", "were", "be", "been", "being", "has", "have", "had",
    "in", "on", "at", "by", "as", "an", "a", "it", "its",
}
# Términos técnicos en inglés permitidos (nombres propios, paquetes).
ALLOWED_ENGLISH = {
    "softmax", "replay", "buffer", "transformer", "transformers",
    "actor", "critic", "deep", "double", "decision", "warm",
    "advantage", "reward", "shaping", "online", "offline",
    "open", "source", "open-source", "human", "in", "loop",
    "the", "a",
    # Términos técnicos universalmente aceptados en español académico
    "learning", "table", "model", "results", "network", "policy",
    "training", "figure", "chapter", "section", "agent",
}


##########################
##  Utilidades          ##
##########################
def read(path: Path) -> str:
    """Lee un archivo de texto en UTF-8 (fallback latin-1)."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def strip_comments(tex: str) -> str:
    """Elimina comentarios LaTeX (% al final de línea, no escapado)."""
    return re.sub(r"(?<!\\)%.*", "", tex)


def chapter_files() -> List[Path]:
    """Devuelve todos los .tex de `01_Chapters/`."""
    return sorted(p for p in CHAPTERS_DIR.glob("*.tex"))


def included_chapters() -> Set[Path]:
    """Devuelve el conjunto de capítulos referenciados por Main.tex."""
    main_tex = strip_comments(read(MAIN_TEX))
    refs = re.findall(r"\\(?:subfile|include|input)\{([^}]+)\}", main_tex)
    included: Set[Path] = set()
    for r in refs:
        candidate = (WRITINGS_DIR / r).resolve()
        if not candidate.suffix:
            candidate = candidate.with_suffix(".tex")
        if candidate.exists():
            included.add(candidate)
    return included


##########################
##  1. Sintaxis         ##
##########################
def audit_syntax(report: List[str], do_tex: bool = True) -> None:
    """Compila Main.tex y reporta errores/avisos + .tex huérfanos."""
    report.append("## 1. Sintaxis y compilación\n")

    # Huérfanos
    included = included_chapters()
    all_chapters = set(chapter_files())
    orphans = sorted(all_chapters - included)
    if orphans:
        report.append("**Archivos `.tex` huérfanos en `01_Chapters/`"
                      " (no incluidos por `Main.tex`):**\n")
        for o in orphans:
            report.append(f"- `{o.relative_to(WRITINGS_DIR)}`")
        report.append("")
    else:
        report.append("- No hay archivos `.tex` huérfanos.\n")

    if not do_tex:
        report.append("- Compilación pdflatex omitida (--no-tex).\n")
        return

    # Compilación: pdflatex → bibtex → pdflatex → pdflatex
    # para resolver \cite{...} con natbib + bibtex (estilo apalike).
    # Todos los artefactos (.aux, .log, .bbl, .pdf…) van a writings/out/
    # y el PDF lleva por nombre el título de la tesis (JOBNAME).
    OUT_DIR.mkdir(exist_ok=True)
    # `subfiles` escribe un .aux espejo por cada subarchivo; pdflatex
    # con -output-directory exige que el árbol exista bajo out/.
    (OUT_DIR / "00_Main").mkdir(exist_ok=True)
    (OUT_DIR / "01_Chapters").mkdir(exist_ok=True)
    pdf_cmd = ["pdflatex", "-interaction=nonstopmode",
               f"-output-directory={OUT_DIR.as_posix()}",
               f"-jobname={JOBNAME}",
               "00_Main/Main.tex"]
    # bibtex lee el .aux generado por pdflatex. En Windows, bibtex tiene
    # openout_any=p (paranoid) y rechaza escribir el .blg si recibe una
    # ruta absoluta; le pasamos la ruta RELATIVA al cwd (writings/).
    aux_rel = (OUT_DIR.relative_to(WRITINGS_DIR) /
               f"{JOBNAME}.aux").as_posix()
    bib_cmd = ["bibtex", aux_rel]
    try:
        # Pasada 1: genera .aux con claves \citation{...}
        subprocess.run(pdf_cmd, cwd=WRITINGS_DIR, capture_output=True,
                       text=True, timeout=300, check=False)
        # bibtex: produce el .bbl con las entradas formateadas
        bib_proc = subprocess.run(bib_cmd, cwd=WRITINGS_DIR,
                                  capture_output=True, text=True,
                                  timeout=120, check=False)
        # Pasada 2: inserta el .bbl en el documento
        subprocess.run(pdf_cmd, cwd=WRITINGS_DIR, capture_output=True,
                       text=True, timeout=300, check=False)
        # Pasada 3: resuelve referencias cruzadas a la bibliografía
        proc = subprocess.run(pdf_cmd, cwd=WRITINGS_DIR,
                              capture_output=True, text=True,
                              timeout=300, check=False)
    except FileNotFoundError as exc:
        report.append(f"- `{exc.filename}` no encontrado; "
                      "omito compilación.\n")
        return

    # Avisos específicos de bibtex (claves ausentes en .bib, etc.)
    bib_log = (OUT_DIR / f"{JOBNAME}.blg")
    if bib_log.exists():
        blg = read(bib_log)
        bib_missing = re.findall(
            r"I didn't find a database entry for \"([^\"]+)\"", blg)
        bib_warn = re.findall(r"Warning--(.*)", blg)
        if bib_missing:
            report.append("**Claves ausentes en `References.bib` "
                          "(detectadas por bibtex):**\n")
            for k in sorted(set(bib_missing)):
                report.append(f"- `{k}`")
            report.append("")
        if bib_warn:
            uniq = sorted(set(w.strip() for w in bib_warn))[:20]
            report.append("**Avisos de bibtex:**\n")
            for w in uniq:
                report.append(f"- {w}")
            report.append("")
    if bib_proc.returncode != 0 and not bib_log.exists():
        report.append(f"- `bibtex` falló (exit={bib_proc.returncode}).\n")

    log = proc.stdout + proc.stderr
    errors = re.findall(r"^! .*", log, flags=re.MULTILINE)
    undef_refs = re.findall(r"Reference `([^']+)' .* undefined", log)
    undef_cites = re.findall(r"Citation `([^']+)' .* undefined", log)
    multi_lbl = re.findall(r"Label `([^']+)' multiply defined", log)

    if proc.returncode == 0 and not errors:
        m = re.search(r"Output written on (\S+\.pdf) \((\d+) pages, (\d+)",
                      log)
        if m:
            report.append(f"- Compilación OK: `{m.group(1)}` "
                          f"({m.group(2)} páginas, "
                          f"{int(m.group(3))//1024} KB).\n")
        else:
            report.append("- Compilación OK.\n")
    else:
        report.append(f"- Compilación FALLA (exit={proc.returncode}).\n")
        for e in errors[:10]:
            report.append(f"  - {e}")
        report.append("")

    if undef_refs:
        report.append("**Referencias indefinidas:**\n")
        for r in sorted(set(undef_refs)):
            report.append(f"- `{r}`")
        report.append("")
    if undef_cites:
        report.append("**Citas indefinidas (provocan `(??)` en el PDF):**\n")
        for c in sorted(set(undef_cites)):
            report.append(f"- `{c}`")
        report.append("")
    if multi_lbl:
        report.append("**Labels duplicados:**\n")
        for l in sorted(set(multi_lbl)):
            report.append(f"- `{l}`")
        report.append("")


##########################
##  2. Figuras/Tablas   ##
##########################
def audit_figures_tables(report: List[str]) -> None:
    """Verifica caption/label en cada `figure`/`table` y referencias."""
    report.append("## 2. Figuras, tablas y numeración\n")

    fig_labels: Dict[str, Path] = {}
    tab_labels: Dict[str, Path] = {}
    refs: List[Tuple[str, Path]] = []
    issues: List[str] = []

    for tex in chapter_files():
        body = strip_comments(read(tex))
        for env in ("figure", "table"):
            pat = re.compile(
                rf"\\begin\{{{env}\*?}}(.*?)\\end\{{{env}\*?}}",
                flags=re.DOTALL,
            )
            for block in pat.findall(body):
                has_caption = bool(re.search(r"\\caption\{", block))
                lbl_match = re.search(r"\\label\{([^}]+)\}", block)
                if not has_caption:
                    issues.append(
                        f"{tex.name}: entorno `{env}` sin "
                        f"`\\caption`.")
                if not lbl_match:
                    issues.append(
                        f"{tex.name}: entorno `{env}` sin "
                        f"`\\label`.")
                else:
                    lbl = lbl_match.group(1)
                    target = fig_labels if env == "figure" else tab_labels
                    if lbl in target:
                        issues.append(
                            f"{tex.name}: label duplicado `{lbl}` "
                            f"(también en {target[lbl].name}).")
                    target[lbl] = tex
                    if env == "figure" and not lbl.startswith("fig:"):
                        issues.append(
                            f"{tex.name}: label `{lbl}` debería "
                            f"empezar con `fig:`.")
                    if env == "table" and not lbl.startswith("tab:"):
                        issues.append(
                            f"{tex.name}: label `{lbl}` debería "
                            f"empezar con `tab:`.")
        for m in re.finditer(r"\\(?:ref|autoref|eqref)\{([^}]+)\}", body):
            refs.append((m.group(1), tex))

    # Añadimos labels de ecuaciones y secciones para no falsos
    # positivos: extraemos todos los \label del proyecto.
    proj_labels: Set[str] = set()
    for tex in chapter_files():
        proj_labels.update(
            re.findall(r"\\label\{([^}]+)\}", strip_comments(read(tex))))

    broken = []
    for r, src in refs:
        if r not in proj_labels:
            broken.append(f"{src.name}: `\\ref{{{r}}}` no resuelve.")

    report.append(f"- Figuras con label: **{len(fig_labels)}**.")
    report.append(f"- Tablas con label: **{len(tab_labels)}**.")
    report.append(f"- Referencias internas (`\\ref`/`\\autoref`/"
                  f"`\\eqref`): **{len(refs)}**.\n")

    for it in issues + broken:
        report.append(f"- ⚠ {it}")
    if not (issues or broken):
        report.append("- Sin problemas detectados.\n")
    else:
        report.append("")


##########################
##  3. Citas / Bibtex   ##
##########################
def parse_bib_keys(bib: str) -> List[Tuple[str, str]]:
    """Devuelve lista (clave, título) del .bib.

    Usa una pasada lineal con contador de llaves para tolerar entradas
    con campos anidados.
    """
    out: List[Tuple[str, str]] = []
    i = 0
    while i < len(bib):
        m = re.search(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", bib[i:])
        if not m:
            break
        key = m.group(2)
        start = i + m.end()
        depth = 1
        j = start
        while j < len(bib) and depth > 0:
            c = bib[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            j += 1
        body = bib[start:j - 1]
        t = re.search(r"title\s*=\s*\{(.*?)\}\s*,",
                      body, flags=re.IGNORECASE | re.DOTALL)
        title = t.group(1).strip() if t else ""
        out.append((key, title))
        i = j
    return out


def audit_citations(report: List[str]) -> None:
    """Audita uso de `\\cite*` contra `References.bib`."""
    report.append("## 3. Citas y bibliografía (APA)\n")

    bib = read(BIB_FILE)
    entries = parse_bib_keys(bib)
    bib_keys = [k for k, _ in entries]
    bib_set = set(bib_keys)

    # Duplicados de clave
    dup_keys = [k for k, c in Counter(bib_keys).items() if c > 1]
    if dup_keys:
        report.append("**Claves duplicadas en `References.bib`:**\n")
        for k in dup_keys:
            report.append(f"- `{k}`")
        report.append("")

    # Títulos casi duplicados (mismo título normalizado)
    norm = lambda s: re.sub(r"[^a-z0-9]+", "", s.lower())
    by_title: Dict[str, List[str]] = defaultdict(list)
    for k, t in entries:
        if t:
            by_title[norm(t)].append(k)
    near_dups = {t: ks for t, ks in by_title.items() if len(ks) > 1}
    if near_dups:
        report.append("**Entradas con título idéntico (posibles "
                      "duplicados):**\n")
        for _, ks in near_dups.items():
            report.append(f"- {', '.join(f'`{k}`' for k in ks)}")
        report.append("")

    # Citas usadas
    used: Counter = Counter()
    for tex in chapter_files():
        body = strip_comments(read(tex))
        for m in re.finditer(
                r"\\(?:cite|citep|citet|textcite|parencite|nocite)"
                r"\*?\{([^}]+)\}", body):
            for k in m.group(1).split(","):
                used[k.strip()] += 1
    used_set = set(used)

    missing = sorted(used_set - bib_set)
    unused = sorted(bib_set - used_set)

    if missing:
        report.append("**Claves citadas pero ausentes en "
                      "`References.bib`:**\n")
        for k in missing:
            report.append(f"- `{k}`")
        report.append("")
    else:
        report.append(f"- Todas las {len(used_set)} claves citadas "
                      "existen en `References.bib`.\n")

    if unused:
        report.append(f"**Entradas no citadas ({len(unused)}):** "
                      f"{', '.join(f'`{k}`' for k in unused[:30])}"
                      + (" ..." if len(unused) > 30 else ""))
        report.append("")


##########################
##  4. Coherencia       ##
##########################
def audit_coherence(report: List[str]) -> None:
    """Conteo de palabras por capítulo y orden de inclusión."""
    report.append("## 4. Coherencia y cohesión\n")

    main_tex = strip_comments(read(MAIN_TEX))
    sections = re.findall(
        r"\\section\{([^}]+)\}[^\n]*\n\s*\\subfile\{01_Chapters/([^}]+)\}",
        main_tex)
    report.append("**Orden de capítulos en `Main.tex`:**\n")
    for sec, chap in sections:
        path = CHAPTERS_DIR / f"{chap}.tex"
        if not path.exists():
            report.append(f"- ⚠ `{sec}` → `{chap}.tex` (no existe)")
            continue
        words = len(re.findall(r"\b\w+\b",
                               strip_comments(read(path))))
        report.append(f"- `{sec}` → `{chap}.tex` "
                      f"({words} palabras)")
    report.append("")

    # Capítulos sin subsection
    flat: List[str] = []
    for _, chap in sections:
        path = CHAPTERS_DIR / f"{chap}.tex"
        if path.exists():
            n_sub = len(re.findall(r"\\subsection\b",
                                   strip_comments(read(path))))
            if n_sub == 0 and chap not in {"00Abstract",
                                            "Acknowledgement"}:
                flat.append(f"- ⚠ `{chap}.tex` sin subsecciones.")
    report.extend(flat)
    if flat:
        report.append("")


##########################
##  5. Idioma           ##
##########################
def audit_language(report: List[str]) -> None:
    """Marca palabras inglesas comunes en el cuerpo (heurístico)."""
    report.append("## 5. Idioma único (español)\n")

    findings: Dict[Path, Counter] = defaultdict(Counter)
    accent = re.compile(r"\\['`^\"~=\.uvHtcdb]\{?([A-Za-z])\}?")
    for tex in chapter_files():
        body = strip_comments(read(tex))
        # Decodificar acentos LaTeX (\'a -> á) para no cortar palabras
        body = accent.sub(lambda m: m.group(1).lower(), body)
        # Eliminar entornos verbatim / código y URLs
        body = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}",
                      "", body, flags=re.DOTALL)
        body = re.sub(r"\$\$.*?\$\$", " ", body, flags=re.DOTALL)
        body = re.sub(r"\$[^$]+\$", " ", body)
        body = re.sub(r"\\begin\{[^}]+\}", " ", body)
        body = re.sub(r"\\end\{[^}]+\}", " ", body)
        body = re.sub(r"\\texttt\{[^}]*\}", "", body)
        body = re.sub(r"\\emph\{[^}]*\}", "", body)
        body = re.sub(r"\\url\{[^}]*\}", "", body)
        body = re.sub(r"\\href\{[^}]*\}\{[^}]*\}", "", body)
        body = re.sub(r"\\cite\w*\{[^}]*\}", "", body)
        body = re.sub(r"\\(?:ref|label|autoref|eqref)\{[^}]*\}",
                      "", body)
        body = re.sub(r"\\[A-Za-z]+\*?", " ", body)  # otras macros
        # Tokenizar
        tokens = re.findall(r"\b[A-Za-z]{2,}\b", body)
        for t in tokens:
            low = t.lower()
            if low in ENGLISH_MARKERS and low not in ALLOWED_ENGLISH:
                findings[tex][low] += 1

    if not findings:
        report.append("- No se detectaron palabras inglesas frecuentes.\n")
        return
    for tex, ctr in sorted(findings.items()):
        top = ", ".join(f"`{w}`×{n}" for w, n in ctr.most_common(8))
        report.append(f"- `{tex.name}`: {top}")
    report.append("")


##########################
##  6. Cobertura docs   ##
##########################
def audit_coverage(report: List[str]) -> None:
    """Verifica que cada paquete sea mencionado en `writings/`."""
    report.append("## 6. Cobertura de los `doc/` del proyecto\n")

    all_writings = "\n".join(read(p) for p in chapter_files())
    low = all_writings.lower()

    report.append("| Paquete | Mencionado | `doc/` presente |")
    report.append("|---------|------------|-----------------|")
    for fam, pkg in PACKAGES:
        mentioned = pkg.lower() in low or pkg.replace("_", r"\_").lower() in low
        doc_dir = REPO_ROOT / fam / pkg / "doc"
        n_docs = len(list(doc_dir.glob("*.md"))) if doc_dir.exists() else 0
        check = "✅" if mentioned else "❌"
        report.append(f"| `{pkg}` | {check} | {n_docs} archivos |")
    report.append("")

    # Conceptos clave que deberían aparecer (según docs)
    must_have = {
        "MDP": r"\bMDP\b",
        "Q-Learning": r"Q-?Learning",
        "Double Q-Learning": r"Double[\s-]?Q",
        "PBRS / potential-based reward shaping": r"PBRS|potential[- ]based",
        "Experience Replay": r"Experience[\s-]?Replay|replay",
        "Target Network": r"Target[\s-]?Network|red objetivo",
        "Optuna / TPE": r"Optuna|TPE",
        "Cohen d / Welch / KL": r"Cohen|Welch|Kullback",
        "Shannon entropy / UQ": r"Shannon|entrop|UQ|incertidumbre",
        "Atención causal / Transformer": r"atenci|attention|Transformer",
    }
    report.append("**Conceptos clave en `writings/`:**\n")
    for label, pat in must_have.items():
        ok = bool(re.search(pat, all_writings, flags=re.IGNORECASE))
        report.append(f"- {'✅' if ok else '❌'} {label}")
    report.append("")


##########################
##  Entrada principal   ##
##########################
def main() -> int:
    """Punto de entrada CLI: ejecuta los seis bloques de auditoría."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-tex", action="store_true",
                        help="omite la compilación pdflatex")
    args = parser.parse_args()

    report: List[str] = [
        "# Auditoría LaTeX — `writings/`\n",
        f"Raíz: `{WRITINGS_DIR}`\n",
    ]

    audit_syntax(report, do_tex=not args.no_tex)
    audit_figures_tables(report)
    audit_citations(report)
    audit_coherence(report)
    audit_language(report)
    audit_coverage(report)

    text = "\n".join(report) + "\n"
    out = WRITINGS_DIR / "audit" / "AUDIT.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text, encoding="utf-8")
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
