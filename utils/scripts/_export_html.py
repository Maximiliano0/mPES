"""
utils/_export_html.py — Convert Markdown documentation to HTML.

Converts every ``.md`` file inside a package's ``doc/`` directory to a
matching ``.html`` in the same location, using the project's standard
template (KaTeX math + dark-mode CSS).

Usage
-----
    # Convert a single package (group prefix optional)
    python utils/scripts/_export_html.py pes_dqn
    python utils/scripts/_export_html.py ml/pes_dqn

    # Convert several packages at once
    python utils/scripts/_export_html.py pes_a2c pes_dqn pes_trf

    # Convert ALL packages (no arguments)
    python utils/scripts/_export_html.py

Notes
-----
Packages live under group directories ``tabular/`` (pes_base, pes_ql,
pes_dql) and ``ml/`` (pes_dqn, pes_rdqn, pes_a2c, pes_trf, pes_ens). The
workspace-level cross-package comparison document at
``doc/comparacion_modelos.md`` is also exported when no argument is
given (or when ``doc`` is passed explicitly).
"""

##########################
##  Imports externos    ##
##########################
import os
import sys
import glob
import markdown


##########################
##  Constantes          ##
##########################

# Mapping from package short name to its workspace-relative location.
# Packages are grouped by algorithm family: tabular vs deep-learning.
PACKAGE_GROUPS = {
    "pes_base": "tabular",
    "pes_ql":   "tabular",
    "pes_dql":  "tabular",
    "pes_dqn":  "ml",
    "pes_rdqn": "ml",
    "pes_a2c":  "ml",
    "pes_trf":  "ml",
    "pes_ens":  "ml",
}

ALL_PACKAGES = list(PACKAGE_GROUPS.keys())

# Workspace root: two levels up from this script (utils/scripts/_export_html.py).
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cross-package comparison document at the workspace root.
WORKSPACE_DOC_DIR = os.path.join(WORKSPACE_ROOT, "doc")

HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.css">
  <script defer
        src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.js"></script>
  <script defer
        src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body, {{delimiters:[
          {{left:'$$',right:'$$',display:true}},
          {{left:'$',right:'$',display:false}}
        ]}})"></script>
  <style>
    :root {{ --bg:#ffffff; --fg:#1a1a2e; --accent:#0f3460; --code-bg:#f4f4f8;
             --border:#ddd; --link:#1565c0; --table-stripe:#f9f9fc; }}
    @media (prefers-color-scheme:dark) {{
      :root {{ --bg:#1a1a2e; --fg:#e0e0e0; --accent:#64b5f6; --code-bg:#16213e;
               --border:#333; --link:#90caf9; --table-stripe:#1e2a45; }}
    }}
    *,*::before,*::after {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{ font-family:"Segoe UI",system-ui,-apple-system,sans-serif;
           line-height:1.7; color:var(--fg); background:var(--bg);
           max-width:52em; margin:2em auto; padding:0 1.5em; }}
    h1 {{ border-bottom:3px solid var(--accent); padding-bottom:.3em; }}
    h2 {{ border-bottom:1px solid var(--border); padding-bottom:.2em; margin-top:2em; }}
    h3,h4 {{ margin-top:1.6em; }}
    a {{ color:var(--link); text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
    code {{ font-family:"JetBrains Mono","Fira Code",Consolas,monospace;
           font-size:.92em; background:var(--code-bg); padding:.15em .35em;
           border-radius:4px; }}
    pre {{ background:var(--code-bg); padding:1em 1.2em; border-radius:8px;
          overflow-x:auto; border:1px solid var(--border); }}
    pre code {{ background:none; padding:0; font-size:.88em; }}
    table {{ border-collapse:collapse; width:100%; margin:1em 0; }}
    th,td {{ border:1px solid var(--border); padding:.55em .9em; text-align:left; }}
    th {{ background:var(--accent); color:#fff; }}
    tr:nth-child(even) {{ background:var(--table-stripe); }}
    blockquote {{ border-left:4px solid var(--accent); margin-left:0;
                 padding:.4em 1em; color:#666; background:var(--code-bg);
                 border-radius:0 6px 6px 0; }}
    .katex-display {{ overflow-x:auto; overflow-y:hidden; padding:.5em 0; }}
  </style>
</head>
<body>
  {body}
</body>
</html>'''

MARKDOWN_EXTENSIONS = ["tables", "fenced_code", "toc"]


##########################
##  Funciones           ##
##########################

def _extract_title(md_text):
    """Extract the first ``# heading`` from Markdown text as HTML title.

    Parameters
    ----------
    md_text : str
        Raw Markdown content.

    Returns
    -------
    str
        The heading text (without ``#``), or ``"Documentation"`` if none found.
    """
    for line in md_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("##"):
            return stripped.lstrip("# ").strip()
    return "Documentation"


def convert_md_to_html(md_path, html_path):
    """Convert a single Markdown file to HTML using the project template.

    Parameters
    ----------
    md_path : str
        Absolute path to the source ``.md`` file.
    html_path : str
        Absolute path for the output ``.html`` file.
    """
    with open(md_path, "r", encoding="utf-8") as fh:
        md_text = fh.read()

    title = _extract_title(md_text)
    body = markdown.markdown(md_text, extensions=MARKDOWN_EXTENSIONS)
    html = HTML_TEMPLATE.format(title=title, body=body)

    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)


def _resolve_pkg_dir(pkg_name):
    """Return absolute filesystem path to a package's directory.

    Accepts both the short name (``"pes_dqn"``) and the full
    group-qualified name (``"ml/pes_dqn"`` or ``"ml\\pes_dqn"``).

    Parameters
    ----------
    pkg_name : str
        Package short name or group/name relative path.

    Returns
    -------
    str or None
        Absolute path to the package directory, or ``None`` if the
        package is unknown and not directly findable on disk.
    """
    normalised = pkg_name.replace("\\", "/").strip("/")
    if "/" in normalised:
        candidate = os.path.join(WORKSPACE_ROOT, *normalised.split("/"))
        return candidate if os.path.isdir(candidate) else None

    group = PACKAGE_GROUPS.get(normalised)
    if group is not None:
        return os.path.join(WORKSPACE_ROOT, group, normalised)

    # Fallback: legacy top-level layout (no group prefix).
    legacy = os.path.join(WORKSPACE_ROOT, normalised)
    return legacy if os.path.isdir(legacy) else None


def export_package(pkg_name):
    """Convert every ``.md`` inside ``<pkg>/doc/`` to ``.html``.

    Parameters
    ----------
    pkg_name : str
        Package short name (e.g. ``"pes_dqn"``) or group-qualified
        path (``"ml/pes_dqn"``). The special name ``"doc"`` exports
        the workspace-level ``doc/`` directory (cross-package
        comparison document).

    Returns
    -------
    int
        Number of files converted.
    """
    if pkg_name == "doc":
        doc_dir = WORKSPACE_DOC_DIR
    else:
        pkg_dir = _resolve_pkg_dir(pkg_name)
        if pkg_dir is None:
            print(f"  SKIP {pkg_name} — package not found")
            return 0
        doc_dir = os.path.join(pkg_dir, "doc")
    if not os.path.isdir(doc_dir):
        print(f"  SKIP {pkg_name}/doc/ — directory not found")
        return 0

    md_files = sorted(glob.glob(os.path.join(doc_dir, "*.md")))
    if not md_files:
        print(f"  SKIP {pkg_name}/doc/ — no .md files")
        return 0

    count = 0
    for md_path in md_files:
        html_path = os.path.splitext(md_path)[0] + ".html"
        convert_md_to_html(md_path, html_path)
        rel = os.path.relpath(html_path, WORKSPACE_ROOT)
        print(f"  OK  {rel}")
        count += 1

    return count


##########################
##  Entry point         ##
##########################

if __name__ == "__main__":
    # When called with no arguments, export every package plus the
    # workspace-level doc/ directory.
    if len(sys.argv) > 1:
        packages = sys.argv[1:]
    else:
        packages = ALL_PACKAGES + ["doc"]

    total = 0
    for pkg in packages:
        total += export_package(pkg)

    print(f"\nDone — {total} file(s) converted.")
